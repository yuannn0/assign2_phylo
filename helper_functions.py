"""Helper functions for HW3"""
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib.axes import Axes


class Node:
    def __init__(
        self,
        name: str,
        left: "Node",
        left_distance: float,
        right: "Node",
        right_distance: float,
        confidence: float = None,
    ):
        """A node in a binary tree produced by neighbor joining algorithm.

        Parameters
        ----------
        name: str
            Name of the node.
        left: Node
            Left child.
        left_distance: float
            The distance to the left child.
        right: Node
            Right child.
        right_distance: float
            The distance to the right child.
        confidence: float
            The confidence level of the split determined by the bootstrap method.
            Only used if you implement Bonus Problem 1.

        Notes
        -----
        The current public API needs to remain as it is, i.e., don't change the
        names of the properties in the template, as the tests expect this kind
        of structure. However, feel free to add any methods/properties/attributes
        that you might need in your tree construction.

        """
        self.name = name
        self.left = left
        self.left_distance = left_distance
        self.right = right
        self.right_distance = right_distance
        self.confidence = confidence


def neighbor_joining(distances: np.ndarray, labels: list) -> Node:
    """The Neighbor-Joining algorithm.

    For the same results as in the later test dendrograms;
    add new nodes to the end of the list/matrix and
    in case of ties, use np.argmin to choose the joining pair.

    Parameters
    ----------
    distances: np.ndarray
        A 2d square, symmetric distance matrix containing distances between
        data points. The diagonal entries should always be zero; d(x, x) = 0.
    labels: list
        A list of labels corresponding to entries in the distances matrix.
        Use them to set names of nodes.

    Returns
    -------
    Node
        A root node of the neighbor joining tree.

    """
    # Copy the matrix ensures that the original one remains unchanged
    distance = distances.copy()
    node = []
    for i in labels:
        node.append(Node(name=i, left=None, left_distance=0.0, right=None, right_distance=0.0))

    node_count = 0

    while len(node) > 2:
        n = len(node)
        row_sum = np.sum(distance, axis=1)
        # Calculate Q matrix
        Q_matrix = ((n - 2)* distance - row_sum[:, None] - row_sum[None, :])
        np.fill_diagonal(Q_matrix, np.inf)

        # select the smallest Q value pair
        i, j = np.unravel_index(np.argmin(Q_matrix), Q_matrix.shape)
        # calculate brach lengths
        distance_i = distance[i, j] / 2 + (row_sum[i] - row_sum[j]) / (2 * (n - 2))
        distance_j = distance[i, j] - distance_i
        # create a new node that joins nodes i and j
        node_count += 1
        new_node = Node(name=f"internal_{node_count}", left=node[i], 
                    left_distance=float(distance_i), right=node[j], 
                    right_distance=float(distance_j))     
        # calculate distances from new node to remaining nodes
        remain_node = []
        for a in range(n):
            if a != i and a != j:
                remain_node.append(a)
        
        new_distance = (distance[i, remain_node] + distance[j, remain_node] - 
                        distance[i,j]) / 2

        # update the distance
        # remove i and j
        new_dis = np.delete(distance, [i, j], axis=0)
        new_dis = np.delete(new_dis, [i, j], axis=1)
        # add new nodes
        new_add = np.append(new_distance, 0.0)
        new_dis = np.vstack([new_dis, new_distance])
        new_dis = np.column_stack([new_dis, new_add])
        # update distance
        distance = new_dis
        node = [node[a] for a in remain_node] + [new_node]

        # connect the final two nodes to root
    final_distance = float(distance[0, 1]/ 2)

    root = Node(name="root", left=node[0], left_distance=final_distance, 
                right=node[1], right_distance=final_distance)

    return root
        

def plot_nj_tree(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.
        But make sure your function can work without them, for testing purposes.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree(tree=tree, ax=ax)
    >>> fig.savefig("example.png")

    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    leaf_count = 0
    label_color = kwargs.get("label_color", {})

    def draw_node(node, current_x):
        nonlocal leaf_count
        # assign y
        if node.left is None and node.right is None:
            current_y = leaf_count
            leaf_count += 1
            ax.annotate(node.name, xy=(current_x, current_y), xytext=(5, 0),
                        textcoords="offset points", va="center", 
                        color=label_color.get(node.name, "black"))
            return current_y

        # Draw subtree
        left_x = current_x + node.left_distance
        right_x = current_x + node.right_distance
        left_y = draw_node(node.left, left_x)
        right_y = draw_node(node.right, right_x)

        # verticla line
        ax.plot([current_x, current_x], [left_y, right_y], color="black")

        # horizontal line
        ax.plot([current_x, left_x], [left_y, left_y], color="black")
        ax.plot([current_x, right_x], [right_y, right_y], color="black")

        # Place the current node between subtree
        current_y = (left_y + right_y) / 2    
        return current_y
    root = draw_node(tree,0)
    ax.plot([-0.2, 0], [root, root], color="black")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
        
    return ax


def _find_a_parent_to_node(tree: Node, node: Node) -> tuple:
    """Utility function for reroot_tree"""
    stack = [tree]

    while len(stack) > 0:

        current_node = stack.pop()
        if node.name == current_node.left.name:
            return current_node, "left"
        elif node.name == current_node.right.name:
            return current_node, "right"

        stack += [
            n for n in [current_node.left, current_node.right] if n.left is not None
        ]

    return None



def _remove_child_from_parent(parent_node: Node, child_location: str) -> None:
    """Utility function for reroot_tree"""
    setattr(parent_node, child_location, None)
    setattr(parent_node, f"{child_location}_distance", 0.0)


def reroot_tree(original_tree: Node, outgroup_node: Node) -> Node:
    """A function to create a new root and invert a tree accordingly.

    This function reroots tree with nodes in original format. If you
    added any other relational parameters to your nodes, these parameters
    will not be inverted! You can modify this implementation or create
    additional functions to fix them.

    Parameters
    ----------
    original_tree: Node
        A root node of the original tree.
    outgroup_node: Node
        A Node to set as an outgroup (already included in a tree).
        Find it by it's name and then use it as parameter.

    Returns
    -------
    Node
        Inverted tree with a new root node.
    """
    tree = deepcopy(original_tree)

    parent, child_loc = _find_a_parent_to_node(tree, outgroup_node)
    distance = getattr(parent, f"{child_loc}_distance")
    _remove_child_from_parent(parent, child_loc)

    new_root = Node("new_root", parent, distance / 2, outgroup_node, distance / 2)
    child = parent

    while tree != child:
        parent, child_loc = _find_a_parent_to_node(tree, child)

        distance = getattr(parent, f"{child_loc}_distance")
        _remove_child_from_parent(parent, child_loc)

        empty_side = "left" if child.left is None else "right"
        setattr(child, f"{empty_side}_distance", distance)
        setattr(child, empty_side, parent)

        if tree.name == parent.name:
            break
        child = parent

    other_child_loc = "right" if child_loc == "left" else "left"
    other_child_distance = getattr(parent, f"{other_child_loc}_distance")

    setattr(child, f"{empty_side}_distance", other_child_distance + distance)
    setattr(child, empty_side, getattr(parent, other_child_loc))

    return new_root


def sort_children_by_leaves(tree: Node) -> None:
    """Sort the children of a tree by their corresponding number of leaves.

    The tree can be changed inplace.

    Parameters
    ----------
    tree: Node
        The root node of the tree.

    """
    if tree is None:
        return
    # sort children
    sort_children_by_leaves(tree.left)
    sort_children_by_leaves(tree.right)
    # count leaves
    def count_leaves(node: Node) -> int:
        if node is None:
            return 0
        if node.left is None and node.right is None:
            return 1
        return count_leaves(node.left) + count_leaves(node.right)
    
    if tree.left is not None and tree.right is not None:
        left_leaves = count_leaves(tree.left)
        right_leaves = count_leaves(tree.right)
        if left_leaves > right_leaves:
            tree.left, tree.right = tree.right, tree.left
            tree.left_distance, tree.right_distance = tree.right_distance, tree.left_distance



def plot_nj_tree_radial(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram
    with a radial layout.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree_radial(tree=tree, ax=ax)
    >>> fig.savefig("example_radial.png")

    """
    raise NotImplementedError()