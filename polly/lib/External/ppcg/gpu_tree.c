/*
 * Copyright 2013      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <string.h>

#include <isl/set.h>
#include <isl/union_set.h>

#include "gpu_tree.h"

/* The functions in this file are used to navigate part of a schedule tree
 * that is mapped to blocks.  Initially, this part consists of a linear
 * branch segment with a mark node with name "kernel" on the outer end
 * and a mark node with name "thread" on the inner end.
 * During the mapping to blocks, branching may be introduced, but only
 * one of the elements in each sequence contains the "thread" mark.
 * The filter of this element (and only this filter) contains
 * domain elements identified by the "core" argument of the functions
 * that move down this tree.
 *
 * Synchronization statements have a name that starts with "sync" and
 * a user pointer pointing to the kernel that contains the synchronization.
 * The functions inserting or detecting synchronizations take a ppcg_kernel
 * argument to be able to create or identify such statements.
 * They may also use two fields in this structure, the "core" field
 * to move around in the tree and the "n_sync" field to make sure that
 * each synchronization has a different name (within the kernel).
 */

/* Is "node" a mark node with an identifier called "name"?
 */
static int is_marked(__isl_keep isl_schedule_node *node, const char *name)
{
	isl_id *mark;
	int has_name;

	if (!node)
		return -1;

	if (isl_schedule_node_get_type(node) != isl_schedule_node_mark)
		return 0;

	mark = isl_schedule_node_mark_get_id(node);
	if (!mark)
		return -1;

	has_name = !strcmp(isl_id_get_name(mark), name);
	isl_id_free(mark);

	return has_name;
}

/* Is "node" a mark node with an identifier called "kernel"?
 */
int gpu_tree_node_is_kernel(__isl_keep isl_schedule_node *node)
{
	return is_marked(node, "kernel");
}

/* Is "node" a mark node with an identifier called "thread"?
 */
static int node_is_thread(__isl_keep isl_schedule_node *node)
{
	return is_marked(node, "thread");
}

/* Assuming "node" is a filter node, does it correspond to the branch
 * that contains the "thread" mark, i.e., does it contain any elements
 * in "core"?
 */
static int node_is_core(__isl_keep isl_schedule_node *node,
	__isl_keep isl_union_set *core)
{
	int disjoint;
	isl_union_set *filter;

	filter = isl_schedule_node_filter_get_filter(node);
	disjoint = isl_union_set_is_disjoint(filter, core);
	isl_union_set_free(filter);
	if (disjoint < 0)
		return -1;

	return !disjoint;
}

/* Move to the only child of "node" that has the "thread" mark as descendant,
 * where the branch containing this mark is identified by the domain elements
 * in "core".
 *
 * If "node" is not a sequence, then it only has one child and we move
 * to that single child.
 * Otherwise, we check each of the filters in the children, pick
 * the one that corresponds to "core" and return a pointer to the child
 * of the filter node.
 */
static __isl_give isl_schedule_node *core_child(
	__isl_take isl_schedule_node *node, __isl_keep isl_union_set *core)
{
	int i, n;

	if (isl_schedule_node_get_type(node) != isl_schedule_node_sequence)
		return isl_schedule_node_child(node, 0);

	n = isl_schedule_node_n_children(node);
	for (i = 0; i < n; ++i) {
		int is_core;

		node = isl_schedule_node_child(node, i);
		is_core = node_is_core(node, core);

		if (is_core < 0)
			return isl_schedule_node_free(node);
		if (is_core)
			return isl_schedule_node_child(node, 0);

		node = isl_schedule_node_parent(node);
	}

	isl_die(isl_schedule_node_get_ctx(node), isl_error_internal,
		"core child not found", return isl_schedule_node_free(node));
}

/* Move down the branch between "kernel" and "thread" until
 * the "thread" mark is reached, where the branch containing the "thread"
 * mark is identified by the domain elements in "core".
 */
__isl_give isl_schedule_node *gpu_tree_move_down_to_thread(
	__isl_take isl_schedule_node *node, __isl_keep isl_union_set *core)
{
	int is_thread;

	while ((is_thread = node_is_thread(node)) == 0)
		node = core_child(node, core);
	if (is_thread < 0)
		node = isl_schedule_node_free(node);

	return node;
}

/* Move up the tree underneath the "thread" mark until
 * the "thread" mark is reached.
 */
__isl_give isl_schedule_node *gpu_tree_move_up_to_thread(
	__isl_take isl_schedule_node *node)
{
	int is_thread;

	while ((is_thread = node_is_thread(node)) == 0)
		node = isl_schedule_node_parent(node);
	if (is_thread < 0)
		node = isl_schedule_node_free(node);

	return node;
}

/* Move up the tree underneath the "kernel" mark until
 * the "kernel" mark is reached.
 */
__isl_give isl_schedule_node *gpu_tree_move_up_to_kernel(
	__isl_take isl_schedule_node *node)
{
	int is_kernel;

	while ((is_kernel = gpu_tree_node_is_kernel(node)) == 0)
		node = isl_schedule_node_parent(node);
	if (is_kernel < 0)
		node = isl_schedule_node_free(node);

	return node;
}

/* Move down from the "kernel" mark (or at least a node with schedule
 * depth smaller than or equal to "depth") to a band node at schedule
 * depth "depth".  The "thread" mark is assumed to have a schedule
 * depth greater than or equal to "depth".  The branch containing the
 * "thread" mark is identified by the domain elements in "core".
 *
 * If the desired schedule depth is in the middle of band node,
 * then the band node is split into two pieces, the second piece
 * at the desired schedule depth.
 */
__isl_give isl_schedule_node *gpu_tree_move_down_to_depth(
	__isl_take isl_schedule_node *node, int depth,
	__isl_keep isl_union_set *core)
{
	int is_thread;

	while (node && isl_schedule_node_get_schedule_depth(node) < depth) {
		if (isl_schedule_node_get_type(node) ==
						    isl_schedule_node_band) {
			int node_depth, node_dim;
			node_depth = isl_schedule_node_get_schedule_depth(node);
			node_dim = isl_schedule_node_band_n_member(node);
			if (node_depth + node_dim > depth)
				node = isl_schedule_node_band_split(node,
							depth - node_depth);
		}
		node = core_child(node, core);
	}
	while ((is_thread = node_is_thread(node)) == 0 &&
	    isl_schedule_node_get_type(node) != isl_schedule_node_band)
		node = core_child(node, core);
	if (is_thread < 0)
		node = isl_schedule_node_free(node);

	return node;
}

/* Create a union set containing a single set with a tuple identifier
 * called "syncX" and user pointer equal to "kernel".
 */
static __isl_give isl_union_set *create_sync_domain(struct ppcg_kernel *kernel)
{
	isl_space *space;
	isl_id *id;
	char name[40];

	space = isl_space_set_alloc(kernel->ctx, 0, 0);
	snprintf(name, sizeof(name), "sync%d", kernel->n_sync++);
	id = isl_id_alloc(kernel->ctx, name, kernel);
	space = isl_space_set_tuple_id(space, isl_dim_set, id);
	return isl_union_set_from_set(isl_set_universe(space));
}

/* Is "id" the identifier of a synchronization statement inside "kernel"?
 * That is, does its name start with "sync" and does it point to "kernel"?
 */
int gpu_tree_id_is_sync(__isl_keep isl_id *id, struct ppcg_kernel *kernel)
{
	const char *name;

	name = isl_id_get_name(id);
	if (!name)
		return 0;
	else if (strncmp(name, "sync", 4))
		return 0;
	return isl_id_get_user(id) == kernel;
}

/* Does "domain" consist of a single set with a tuple identifier
 * corresponding to a synchronization for "kernel"?
 */
static int domain_is_sync(__isl_keep isl_union_set *domain,
	struct ppcg_kernel *kernel)
{
	int is_sync;
	isl_id *id;
	isl_set *set;

	if (isl_union_set_n_set(domain) != 1)
		return 0;
	set = isl_set_from_union_set(isl_union_set_copy(domain));
	id = isl_set_get_tuple_id(set);
	is_sync = gpu_tree_id_is_sync(id, kernel);
	isl_id_free(id);
	isl_set_free(set);

	return is_sync;
}

/* Does "node" point to a filter selecting a synchronization statement
 * for "kernel"?
 */
static int node_is_sync_filter(__isl_keep isl_schedule_node *node,
	struct ppcg_kernel *kernel)
{
	int is_sync;
	enum isl_schedule_node_type type;
	isl_union_set *domain;

	if (!node)
		return -1;
	type = isl_schedule_node_get_type(node);
	if (type != isl_schedule_node_filter)
		return 0;
	domain = isl_schedule_node_filter_get_filter(node);
	is_sync = domain_is_sync(domain, kernel);
	isl_union_set_free(domain);

	return is_sync;
}

/* Is "node" part of a sequence with a previous synchronization statement
 * for "kernel"?
 * That is, is the parent of "node" a filter such that there is
 * a previous filter that picks out exactly such a synchronization statement?
 */
static int has_preceding_sync(__isl_keep isl_schedule_node *node,
	struct ppcg_kernel *kernel)
{
	int found = 0;

	node = isl_schedule_node_copy(node);
	node = isl_schedule_node_parent(node);
	while (!found && isl_schedule_node_has_previous_sibling(node)) {
		node = isl_schedule_node_previous_sibling(node);
		if (!node)
			break;
		found = node_is_sync_filter(node, kernel);
	}
	if (!node)
		found = -1;
	isl_schedule_node_free(node);

	return found;
}

/* Is "node" part of a sequence with a subsequent synchronization statement
 * for "kernel"?
 * That is, is the parent of "node" a filter such that there is
 * a subsequent filter that picks out exactly such a synchronization statement?
 */
static int has_following_sync(__isl_keep isl_schedule_node *node,
	struct ppcg_kernel *kernel)
{
	int found = 0;

	node = isl_schedule_node_copy(node);
	node = isl_schedule_node_parent(node);
	while (!found && isl_schedule_node_has_next_sibling(node)) {
		node = isl_schedule_node_next_sibling(node);
		if (!node)
			break;
		found = node_is_sync_filter(node, kernel);
	}
	if (!node)
		found = -1;
	isl_schedule_node_free(node);

	return found;
}

/* Does the subtree rooted at "node" (which is a band node) contain
 * any synchronization statement for "kernel" that precedes
 * the core computation of "kernel" (identified by the elements
 * in kernel->core)?
 */
static int has_sync_before_core(__isl_keep isl_schedule_node *node,
	struct ppcg_kernel *kernel)
{
	int has_sync = 0;
	int is_thread;

	node = isl_schedule_node_copy(node);
	while ((is_thread = node_is_thread(node)) == 0) {
		node = core_child(node, kernel->core);
		has_sync = has_preceding_sync(node, kernel);
		if (has_sync < 0 || has_sync)
			break;
	}
	if (is_thread < 0 || !node)
		has_sync = -1;
	isl_schedule_node_free(node);

	return has_sync;
}

/* Does the subtree rooted at "node" (which is a band node) contain
 * any synchronization statement for "kernel" that follows
 * the core computation of "kernel" (identified by the elements
 * in kernel->core)?
 */
static int has_sync_after_core(__isl_keep isl_schedule_node *node,
	struct ppcg_kernel *kernel)
{
	int has_sync = 0;
	int is_thread;

	node = isl_schedule_node_copy(node);
	while ((is_thread = node_is_thread(node)) == 0) {
		node = core_child(node, kernel->core);
		has_sync = has_following_sync(node, kernel);
		if (has_sync < 0 || has_sync)
			break;
	}
	if (is_thread < 0 || !node)
		has_sync = -1;
	isl_schedule_node_free(node);

	return has_sync;
}

/* Insert (or extend) an extension on top of "node" that puts
 * a synchronization node for "kernel" before "node".
 * Return a pointer to the original node in the updated schedule tree.
 */
static __isl_give isl_schedule_node *insert_sync_before(
	__isl_take isl_schedule_node *node, struct ppcg_kernel *kernel)
{
	isl_union_set *domain;
	isl_schedule_node *graft;

	if (!node)
		return NULL;

	domain = create_sync_domain(kernel);
	graft = isl_schedule_node_from_domain(domain);
	node = isl_schedule_node_graft_before(node, graft);

	return node;
}

/* Insert (or extend) an extension on top of "node" that puts
 * a synchronization node for "kernel" afater "node".
 * Return a pointer to the original node in the updated schedule tree.
 */
static __isl_give isl_schedule_node *insert_sync_after(
	__isl_take isl_schedule_node *node, struct ppcg_kernel *kernel)
{
	isl_union_set *domain;
	isl_schedule_node *graft;

	if (!node)
		return NULL;

	domain = create_sync_domain(kernel);
	graft = isl_schedule_node_from_domain(domain);
	node = isl_schedule_node_graft_after(node, graft);

	return node;
}

/* Insert an extension on top of "node" that puts a synchronization node
 * for "kernel" before "node" unless there already is
 * such a synchronization node.
 */
__isl_give isl_schedule_node *gpu_tree_ensure_preceding_sync(
	__isl_take isl_schedule_node *node, struct ppcg_kernel *kernel)
{
	int has_sync;

	has_sync = has_preceding_sync(node, kernel);
	if (has_sync < 0)
		return isl_schedule_node_free(node);
	if (has_sync)
		return node;
	return insert_sync_before(node, kernel);
}

/* Insert an extension on top of "node" that puts a synchronization node
 * for "kernel" after "node" unless there already is
 * such a synchronization node.
 */
__isl_give isl_schedule_node *gpu_tree_ensure_following_sync(
	__isl_take isl_schedule_node *node, struct ppcg_kernel *kernel)
{
	int has_sync;

	has_sync = has_following_sync(node, kernel);
	if (has_sync < 0)
		return isl_schedule_node_free(node);
	if (has_sync)
		return node;
	return insert_sync_after(node, kernel);
}

/* Insert an extension on top of "node" that puts a synchronization node
 * for "kernel" after "node" unless there already is such a sync node or
 * "node" itself already * contains a synchronization node following
 * the core computation of "kernel".
 */
__isl_give isl_schedule_node *gpu_tree_ensure_sync_after_core(
	__isl_take isl_schedule_node *node, struct ppcg_kernel *kernel)
{
	int has_sync;

	has_sync = has_sync_after_core(node, kernel);
	if (has_sync < 0)
		return isl_schedule_node_free(node);
	if (has_sync)
		return node;
	has_sync = has_following_sync(node, kernel);
	if (has_sync < 0)
		return isl_schedule_node_free(node);
	if (has_sync)
		return node;
	return insert_sync_after(node, kernel);
}

/* Move left in the sequence on top of "node" to a synchronization node
 * for "kernel".
 * If "node" itself contains a synchronization node preceding
 * the core computation of "kernel", then return "node" itself.
 * Otherwise, if "node" does not have a preceding synchronization node,
 * then create one first.
 */
__isl_give isl_schedule_node *gpu_tree_move_left_to_sync(
	__isl_take isl_schedule_node *node, struct ppcg_kernel *kernel)
{
	int has_sync;
	int is_sync;

	has_sync = has_sync_before_core(node, kernel);
	if (has_sync < 0)
		return isl_schedule_node_free(node);
	if (has_sync)
		return node;
	node = gpu_tree_ensure_preceding_sync(node, kernel);
	node = isl_schedule_node_parent(node);
	while ((is_sync = node_is_sync_filter(node, kernel)) == 0)
		node = isl_schedule_node_previous_sibling(node);
	if (is_sync < 0)
		node = isl_schedule_node_free(node);
	node = isl_schedule_node_child(node, 0);

	return node;
}

/* Move right in the sequence on top of "node" to a synchronization node
 * for "kernel".
 * If "node" itself contains a synchronization node following
 * the core computation of "kernel", then return "node" itself.
 * Otherwise, if "node" does not have a following synchronization node,
 * then create one first.
 */
__isl_give isl_schedule_node *gpu_tree_move_right_to_sync(
	__isl_take isl_schedule_node *node, struct ppcg_kernel *kernel)
{
	int has_sync;
	int is_sync;

	has_sync = has_sync_after_core(node, kernel);
	if (has_sync < 0)
		return isl_schedule_node_free(node);
	if (has_sync)
		return node;
	node = gpu_tree_ensure_following_sync(node, kernel);
	node = isl_schedule_node_parent(node);
	while ((is_sync = node_is_sync_filter(node, kernel)) == 0)
		node = isl_schedule_node_next_sibling(node);
	if (is_sync < 0)
		node = isl_schedule_node_free(node);
	node = isl_schedule_node_child(node, 0);

	return node;
}
