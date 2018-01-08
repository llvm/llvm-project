/*
 * Copyright 2016      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege.
 */

#include <isl/ctx.h>
#include <isl/id.h>
#include <isl/val.h>
#include <isl/space.h>
#include <isl/aff.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/union_set.h>
#include <isl/union_map.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>

#include "ppcg.h"

/* Internal data structure for use during the detection of statements
 * that can be grouped.
 *
 * "sc" contains the original schedule constraints (not a copy).
 * "dep" contains the intersection of the validity and the proximity
 * constraints in "sc".  It may be NULL if it has not been computed yet.
 * "group_id" is the identifier for the next group that is extracted.
 *
 * "domain" is the set of statement instances that belong to any of the groups.
 * "contraction" maps the elements of "domain" to the corresponding group
 * instances.
 * "schedule" schedules the statements in each group relatively to each other.
 * These last three fields are NULL if no groups have been found so far.
 */
struct ppcg_grouping {
	isl_schedule_constraints *sc;

	isl_union_map *dep;
	int group_id;

	isl_union_set *domain;
	isl_union_pw_multi_aff *contraction;
	isl_schedule *schedule;
};

/* Clear all memory allocated by "grouping".
 */
static void ppcg_grouping_clear(struct ppcg_grouping *grouping)
{
	isl_union_map_free(grouping->dep);
	isl_union_set_free(grouping->domain);
	isl_union_pw_multi_aff_free(grouping->contraction);
	isl_schedule_free(grouping->schedule);
}

/* Compute the intersection of the proximity and validity dependences
 * in grouping->sc and store the result in grouping->dep, unless
 * this intersection has been computed before.
 */
static isl_stat ppcg_grouping_compute_dep(struct ppcg_grouping *grouping)
{
	isl_union_map *validity, *proximity;

	if (grouping->dep)
		return isl_stat_ok;

	validity = isl_schedule_constraints_get_validity(grouping->sc);
	proximity = isl_schedule_constraints_get_proximity(grouping->sc);
	grouping->dep = isl_union_map_intersect(validity, proximity);

	if (!grouping->dep)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Information extracted from one or more consecutive leaves
 * in the input schedule.
 *
 * "list" contains the sets of statement instances in the leaves,
 * one element in the list for each original leaf.
 * "domain" contains the union of the sets in "list".
 * "prefix" contains the prefix schedule of these elements.
 */
struct ppcg_grouping_leaf {
	isl_union_set *domain;
	isl_union_set_list *list;
	isl_multi_union_pw_aff *prefix;
};

/* Free all memory allocated for "leaves".
 */
static void ppcg_grouping_leaf_free(int n, struct ppcg_grouping_leaf leaves[])
{
	int i;

	if (!leaves)
		return;

	for (i = 0; i < n; ++i) {
		isl_union_set_free(leaves[i].domain);
		isl_union_set_list_free(leaves[i].list);
		isl_multi_union_pw_aff_free(leaves[i].prefix);
	}

	free(leaves);
}

/* Short-hand for retrieving the prefix schedule at "node"
 * in the form of an isl_multi_union_pw_aff.
 */
static __isl_give isl_multi_union_pw_aff *get_prefix(
	__isl_keep isl_schedule_node *node)
{
	return isl_schedule_node_get_prefix_schedule_multi_union_pw_aff(node);
}

/* Return an array of "n" elements with information extracted from
 * the "n" children of "node" starting at "first", all of which
 * are known to be filtered leaves.
 */
struct ppcg_grouping_leaf *extract_leaves(__isl_keep isl_schedule_node *node,
	int first, int n)
{
	int i;
	isl_ctx *ctx;
	struct ppcg_grouping_leaf *leaves;

	if (!node)
		return NULL;

	ctx = isl_schedule_node_get_ctx(node);
	leaves = isl_calloc_array(ctx, struct ppcg_grouping_leaf, n);
	if (!leaves)
		return NULL;

	for (i = 0; i < n; ++i) {
		isl_schedule_node *child;
		isl_union_set *domain;

		child = isl_schedule_node_get_child(node, first + i);
		child = isl_schedule_node_child(child, 0);
		domain = isl_schedule_node_get_domain(child);
		leaves[i].domain = isl_union_set_copy(domain);
		leaves[i].list = isl_union_set_list_from_union_set(domain);
		leaves[i].prefix = get_prefix(child);
		isl_schedule_node_free(child);
	}

	return leaves;
}

/* Internal data structure used by merge_leaves.
 *
 * "src" and "dst" point to the two consecutive leaves that are
 * under investigation for being merged.
 * "merge" is initially set to 0 and is set to 1 as soon as
 * it turns out that it is useful to merge the two leaves.
 */
struct ppcg_merge_leaves_data {
	int merge;
	struct ppcg_grouping_leaf *src;
	struct ppcg_grouping_leaf *dst;
};

/* Given a relation "map" between instances of two statements A and B,
 * does it relate every instance of A (according to the domain of "src")
 * to every instance of B (according to the domain of "dst")?
 */
static isl_bool covers_src_and_dst(__isl_keep isl_map *map,
	struct ppcg_grouping_leaf *src, struct ppcg_grouping_leaf *dst)
{
	isl_space *space;
	isl_set *set1, *set2;
	isl_bool is_subset;

	space = isl_space_domain(isl_map_get_space(map));
	set1 = isl_union_set_extract_set(src->domain, space);
	set2 = isl_map_domain(isl_map_copy(map));
	is_subset = isl_set_is_subset(set1, set2);
	isl_set_free(set1);
	isl_set_free(set2);
	if (is_subset < 0 || !is_subset)
		return is_subset;

	space = isl_space_range(isl_map_get_space(map));
	set1 = isl_union_set_extract_set(dst->domain, space);
	set2 = isl_map_range(isl_map_copy(map));
	is_subset = isl_set_is_subset(set1, set2);
	isl_set_free(set1);
	isl_set_free(set2);

	return is_subset;
}

/* Given a relation "map" between instances of two statements A and B,
 * are pairs of related instances executed together in the input schedule?
 * That is, is each pair of instances assigned the same value
 * by the corresponding prefix schedules?
 *
 * In particular, select the subset of "map" that has pairs of elements
 * with the same value for the prefix schedules and then check
 * if "map" is still a subset of the result.
 */
static isl_bool matches_prefix(__isl_keep isl_map *map,
	struct ppcg_grouping_leaf *src, struct ppcg_grouping_leaf *dst)
{
	isl_union_map *umap, *equal;
	isl_multi_union_pw_aff *src_prefix, *dst_prefix, *prefix;
	isl_bool is_subset;

	src_prefix = isl_multi_union_pw_aff_copy(src->prefix);
	dst_prefix = isl_multi_union_pw_aff_copy(dst->prefix);
	prefix = isl_multi_union_pw_aff_union_add(src_prefix, dst_prefix);

	umap = isl_union_map_from_map(isl_map_copy(map));
	equal = isl_union_map_copy(umap);
	equal = isl_union_map_eq_at_multi_union_pw_aff(equal, prefix);

	is_subset = isl_union_map_is_subset(umap, equal);

	isl_union_map_free(umap);
	isl_union_map_free(equal);

	return is_subset;
}

/* Given a set of validity and proximity schedule constraints "map"
 * between statements in consecutive leaves in a valid schedule,
 * should the two leaves be merged into one?
 *
 * In particular, the two are merged if the constraints form
 * a bijection between every instance of the first statement and
 * every instance of the second statement.  Moreover, each
 * pair of such dependent instances needs to be executed consecutively
 * in the input schedule.  That is, they need to be assigned
 * the same value by their prefix schedules.
 *
 * What this means is that for each instance of the first statement
 * there is exactly one instance of the second statement that
 * is executed immediately after the instance of the first statement and
 * that, moreover, both depends on this statement instance and
 * should be brought as close as possible to this statement instance.
 * In other words, it is both possible to execute the two instances
 * together (according to the input schedule) and desirable to do so
 * (according to the validity and proximity schedule constraints).
 */
static isl_stat check_merge(__isl_take isl_map *map, void *user)
{
	struct ppcg_merge_leaves_data *data = user;
	isl_bool ok;

	ok = covers_src_and_dst(map, data->src, data->dst);
	if (ok >= 0 && ok)
		ok = isl_map_is_bijective(map);
	if (ok >= 0 && ok)
		ok = matches_prefix(map, data->src, data->dst);

	isl_map_free(map);

	if (ok < 0)
		return isl_stat_error;
	if (!ok)
		return isl_stat_ok;

	data->merge = 1;
	return isl_stat_error;
}

/* Merge the leaves at position "pos" and "pos + 1" in "leaves".
 */
static isl_stat merge_pair(int n, struct ppcg_grouping_leaf leaves[], int pos)
{
	int i;

	leaves[pos].domain = isl_union_set_union(leaves[pos].domain,
						leaves[pos + 1].domain);
	leaves[pos].list = isl_union_set_list_concat(leaves[pos].list,
						leaves[pos + 1].list);
	leaves[pos].prefix = isl_multi_union_pw_aff_union_add(
				leaves[pos].prefix, leaves[pos + 1].prefix);
	for (i = pos + 1; i + 1 < n; ++i)
		leaves[i] = leaves[i + 1];
	leaves[n - 1].domain = NULL;
	leaves[n - 1].list = NULL;
	leaves[n - 1].prefix = NULL;

	if (!leaves[pos].domain || !leaves[pos].list || !leaves[pos].prefix)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Merge pairs of consecutive leaves in "leaves" taking into account
 * the intersection of validity and proximity schedule constraints "dep".
 *
 * If a leaf has been merged with the next leaf, then the combination
 * is checked again for merging with the next leaf.
 * That is, if the leaves are A, B and C, then B may not have been
 * merged with C, but after merging A and B, it could still be useful
 * to merge the combination AB with C.
 *
 * Two leaves A and B are merged if there are instances of at least
 * one pair of statements, one statement in A and one B, such that
 * the validity and proximity schedule constraints between them
 * make them suitable for merging according to check_merge.
 *
 * Return the final number of leaves in the sequence, or -1 on error.
 */
static int merge_leaves(int n, struct ppcg_grouping_leaf leaves[],
	__isl_keep isl_union_map *dep)
{
	int i;
	struct ppcg_merge_leaves_data data;

	for (i = n - 1; i >= 0; --i) {
		isl_union_map *dep_i;
		isl_stat ok;

		if (i + 1 >= n)
			continue;

		dep_i = isl_union_map_copy(dep);
		dep_i = isl_union_map_intersect_domain(dep_i,
				isl_union_set_copy(leaves[i].domain));
		dep_i = isl_union_map_intersect_range(dep_i,
				isl_union_set_copy(leaves[i + 1].domain));
		data.merge = 0;
		data.src = &leaves[i];
		data.dst = &leaves[i + 1];
		ok = isl_union_map_foreach_map(dep_i, &check_merge, &data);
		isl_union_map_free(dep_i);
		if (ok < 0 && !data.merge)
			return -1;
		if (!data.merge)
			continue;
		if (merge_pair(n, leaves, i) < 0)
			return -1;
		--n;
		++i;
	}

	return n;
}

/* Construct a schedule with "domain" as domain, that executes
 * the elements of "list" in order (as a sequence).
 */
static __isl_give isl_schedule *schedule_from_domain_and_list(
	__isl_keep isl_union_set *domain, __isl_keep isl_union_set_list *list)
{
	isl_schedule *schedule;
	isl_schedule_node *node;

	schedule = isl_schedule_from_domain(isl_union_set_copy(domain));
	node = isl_schedule_get_root(schedule);
	isl_schedule_free(schedule);
	node = isl_schedule_node_child(node, 0);
	list = isl_union_set_list_copy(list);
	node = isl_schedule_node_insert_sequence(node, list);
	schedule = isl_schedule_node_get_schedule(node);
	isl_schedule_node_free(node);

	return schedule;
}

/* Construct a unique identifier for a group in "grouping".
 *
 * The name is of the form G_n, with n the first value starting at
 * grouping->group_id that does not result in an identifier
 * that is already in use in the domain of the original schedule
 * constraints.
 */
static isl_id *construct_group_id(struct ppcg_grouping *grouping,
	__isl_take isl_space *space)
{
	isl_ctx *ctx;
	isl_id *id;
	isl_bool empty;
	isl_union_set *domain;

	if (!space)
		return NULL;

	ctx = isl_space_get_ctx(space);
	domain = isl_schedule_constraints_get_domain(grouping->sc);

	do {
		char buffer[20];
		isl_id *id;
		isl_set *set;

		snprintf(buffer, sizeof(buffer), "G_%d", grouping->group_id);
		grouping->group_id++;
		id = isl_id_alloc(ctx, buffer, NULL);
		space = isl_space_set_tuple_id(space, isl_dim_set, id);
		set = isl_union_set_extract_set(domain, isl_space_copy(space));
		empty = isl_set_plain_is_empty(set);
		isl_set_free(set);
	} while (empty >= 0 && !empty);

	if (empty < 0)
		space = isl_space_free(space);

	id = isl_space_get_tuple_id(space, isl_dim_set);

	isl_space_free(space);
	isl_union_set_free(domain);

	return id;
}

/* Construct a contraction from "prefix" and "domain" for a new group
 * in "grouping".
 *
 * The values of the prefix schedule "prefix" are used as instances
 * of the new group.  The identifier of the group is constructed
 * in such a way that it does not conflict with those of earlier
 * groups nor with statements in the domain of the original
 * schedule constraints.
 * The isl_multi_union_pw_aff "prefix" then simply needs to be
 * converted to an isl_union_pw_multi_aff.  However, this is not
 * possible if "prefix" is zero-dimensional, so in this case,
 * a contraction is constructed from "domain" instead.
 */
static isl_union_pw_multi_aff *group_contraction_from_prefix_and_domain(
	struct ppcg_grouping *grouping,
	__isl_keep isl_multi_union_pw_aff *prefix,
	__isl_keep isl_union_set *domain)
{
	isl_id *id;
	isl_space *space;
	int dim;

	space = isl_multi_union_pw_aff_get_space(prefix);
	if (!space)
		return NULL;
	dim = isl_space_dim(space, isl_dim_set);
	id = construct_group_id(grouping, space);
	if (dim == 0) {
		isl_multi_val *mv;

		space = isl_multi_union_pw_aff_get_space(prefix);
		space = isl_space_set_tuple_id(space, isl_dim_set, id);
		mv = isl_multi_val_zero(space);
		domain = isl_union_set_copy(domain);
		return isl_union_pw_multi_aff_multi_val_on_domain(domain, mv);
	}
	prefix = isl_multi_union_pw_aff_copy(prefix);
	prefix = isl_multi_union_pw_aff_set_tuple_id(prefix, isl_dim_out, id);
	return isl_union_pw_multi_aff_from_multi_union_pw_aff(prefix);
}

/* Extend "grouping" with groups corresponding to merged
 * leaves in the list of potentially merged leaves "leaves".
 *
 * The "list" field of each element in "leaves" contains a list
 * of the instances sets of the original leaves that have been
 * merged into this element.  If at least two of the original leaves
 * have been merged into a given element, then add the corresponding
 * group to "grouping".
 * In particular, the domain is extended with the statement instances
 * of the merged leaves, the contraction is extended with a mapping
 * of these statement instances to instances of a new group and
 * the schedule is extended with a schedule that executes
 * the statement instances according to the order of the leaves
 * in which they appear.
 * Since the instances of the groups should already be scheduled apart
 * in the schedule into which this schedule will be plugged in,
 * the schedules of the individual groups are combined independently
 * of each other (as a set).
 */
static isl_stat add_groups(struct ppcg_grouping *grouping,
	int n, struct ppcg_grouping_leaf leaves[])
{
	int i;

	for (i = 0; i < n; ++i) {
		int n_leaf;
		isl_schedule *schedule;
		isl_union_set *domain;
		isl_union_pw_multi_aff *upma;

		n_leaf = isl_union_set_list_n_union_set(leaves[i].list);
		if (n_leaf < 0)
			return isl_stat_error;
		if (n_leaf <= 1)
			continue;
		schedule = schedule_from_domain_and_list(leaves[i].domain,
							leaves[i].list);
		upma = group_contraction_from_prefix_and_domain(grouping,
					leaves[i].prefix, leaves[i].domain);

		domain = isl_union_set_copy(leaves[i].domain);
		if (grouping->domain) {
			domain = isl_union_set_union(domain, grouping->domain);
			upma = isl_union_pw_multi_aff_union_add(upma,
						grouping->contraction);
			schedule = isl_schedule_set(schedule,
						grouping->schedule);
		}
		grouping->domain = domain;
		grouping->contraction = upma;
		grouping->schedule = schedule;

		if (!grouping->domain || !grouping->contraction ||
		    !grouping->schedule)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Look for any pairs of consecutive leaves among the "n" children of "node"
 * starting at "first" that should be merged together.
 * Store the results in "grouping".
 *
 * First make sure the intersection of validity and proximity
 * schedule constraints is available and extract the required
 * information from the "n" leaves.
 * Then try and merge consecutive leaves based on the validity
 * and proximity constraints.
 * If any pairs were successfully merged, then add groups
 * corresponding to the merged leaves to "grouping".
 */
static isl_stat group_subsequence(__isl_keep isl_schedule_node *node,
	int first, int n, struct ppcg_grouping *grouping)
{
	int n_merge;
	struct ppcg_grouping_leaf *leaves;

	if (ppcg_grouping_compute_dep(grouping) < 0)
		return isl_stat_error;

	leaves = extract_leaves(node, first, n);
	if (!leaves)
		return isl_stat_error;

	n_merge = merge_leaves(n, leaves, grouping->dep);
	if (n_merge >= 0 && n_merge < n &&
	    add_groups(grouping, n_merge, leaves) < 0)
		return isl_stat_error;

	ppcg_grouping_leaf_free(n, leaves);

	return isl_stat_ok;
}

/* If "node" is a sequence, then check if it has any consecutive
 * leaves that should be merged together and store the results
 * in "grouping".
 *
 * In particular, call group_subsequence on each consecutive
 * sequence of (filtered) leaves among the children of "node".
 */
static isl_bool detect_groups(__isl_keep isl_schedule_node *node, void *user)
{
	int i, n, first;
	struct ppcg_grouping *grouping = user;

	if (isl_schedule_node_get_type(node) != isl_schedule_node_sequence)
		return isl_bool_true;

	n = isl_schedule_node_n_children(node);
	if (n < 0)
		return isl_bool_error;

	first = -1;
	for (i = 0; i < n; ++i) {
		isl_schedule_node *child;
		enum isl_schedule_node_type type;

		child = isl_schedule_node_get_child(node, i);
		child = isl_schedule_node_child(child, 0);
		type = isl_schedule_node_get_type(child);
		isl_schedule_node_free(child);

		if (first >= 0 && type != isl_schedule_node_leaf) {
			if (group_subsequence(node, first, i - first,
						grouping) < 0)
				return isl_bool_error;
			first = -1;
		}
		if (first < 0 && type == isl_schedule_node_leaf)
			first = i;
	}
	if (first >= 0) {
		if (group_subsequence(node, first, n - first, grouping) < 0)
			return isl_bool_error;
	}

	return isl_bool_true;
}

/* Complete "grouping" to cover all statement instances in the domain
 * of grouping->sc.
 *
 * In particular, grouping->domain is set to the full set of statement
 * instances; group->contraction is extended with an identity
 * contraction on the additional instances and group->schedule
 * is extended with an independent schedule on those additional instances.
 * In the extension of group->contraction, the additional instances
 * are split into those belong to different statements and those
 * that belong to some of the same statements.  The first group
 * is replaced by its universe in order to simplify the contraction extension.
 */
static void complete_grouping(struct ppcg_grouping *grouping)
{
	isl_union_set *domain, *left, *overlap;
	isl_union_pw_multi_aff *upma;
	isl_schedule *schedule;

	domain = isl_schedule_constraints_get_domain(grouping->sc);
	left = isl_union_set_subtract(isl_union_set_copy(domain),
				    isl_union_set_copy(grouping->domain));
	schedule = isl_schedule_from_domain(isl_union_set_copy(left));
	schedule = isl_schedule_set(schedule, grouping->schedule);
	grouping->schedule = schedule;

	overlap = isl_union_set_universe(grouping->domain);
	grouping->domain = domain;
	overlap = isl_union_set_intersect(isl_union_set_copy(left), overlap);
	left = isl_union_set_subtract(left, isl_union_set_copy(overlap));
	left = isl_union_set_universe(left);
	left = isl_union_set_union(left, overlap);
	upma = isl_union_set_identity_union_pw_multi_aff(left);
	upma = isl_union_pw_multi_aff_union_add(upma, grouping->contraction);
	grouping->contraction = upma;
}

/* Compute a schedule on the domain of "sc" that respects the schedule
 * constraints in "sc".
 *
 * "schedule" is a known correct schedule that is used to combine
 * groups of statements if options->group_chains is set.
 * In particular, statements that are executed consecutively in a sequence
 * in this schedule and where all instances of the second depend on
 * the instance of the first that is executed in the same iteration
 * of outer band nodes are grouped together into a single statement.
 * The schedule constraints are then mapped to these groups of statements
 * and the resulting schedule is expanded again to refer to the original
 * statements.
 */
__isl_give isl_schedule *ppcg_compute_schedule(
	__isl_take isl_schedule_constraints *sc,
	__isl_keep isl_schedule *schedule, struct ppcg_options *options)
{
	struct ppcg_grouping grouping = { sc };
	isl_union_pw_multi_aff *contraction;
	isl_union_map *umap;
	isl_schedule *res, *expansion;

	if (!options->group_chains)
		return isl_schedule_constraints_compute_schedule(sc);

	grouping.group_id = 0;
	if (isl_schedule_foreach_schedule_node_top_down(schedule,
			&detect_groups, &grouping) < 0)
		goto error;
	if (!grouping.contraction) {
		ppcg_grouping_clear(&grouping);
		return isl_schedule_constraints_compute_schedule(sc);
	}
	complete_grouping(&grouping);
	contraction = isl_union_pw_multi_aff_copy(grouping.contraction);
	umap = isl_union_map_from_union_pw_multi_aff(contraction);

	sc = isl_schedule_constraints_apply(sc, umap);

	res = isl_schedule_constraints_compute_schedule(sc);

	contraction = isl_union_pw_multi_aff_copy(grouping.contraction);
	expansion = isl_schedule_copy(grouping.schedule);
	res = isl_schedule_expand(res, contraction, expansion);

	ppcg_grouping_clear(&grouping);
	return res;
error:
	ppcg_grouping_clear(&grouping);
	isl_schedule_constraints_free(sc);
	return NULL;
}
