/*
 * Copyright 2015      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege
 */

#include "isl_map_private.h"

#include <isl/id.h>
#include <isl/schedule_node.h>
#include <isl/union_set.h>

#include "isl_mat_private.h"
#include "isl_scheduler_clustering.h"
#include "isl_scheduler_scc.h"
#include "isl_seq.h"
#include "isl_tarjan.h"

/* Initialize the clustering data structure "c" from "graph".
 *
 * In particular, allocate memory, extract the SCCs from "graph"
 * into c->scc, initialize scc_cluster and construct
 * a band of schedule rows for each SCC.
 * Within each SCC, there is only one SCC by definition.
 * Each SCC initially belongs to a cluster containing only that SCC.
 */
static isl_stat clustering_init(isl_ctx *ctx, struct isl_clustering *c,
	struct isl_sched_graph *graph)
{
	int i;

	c->n = graph->scc;
	c->scc = isl_calloc_array(ctx, struct isl_sched_graph, c->n);
	c->cluster = isl_calloc_array(ctx, struct isl_sched_graph, c->n);
	c->scc_cluster = isl_calloc_array(ctx, int, c->n);
	c->scc_node = isl_calloc_array(ctx, int, c->n);
	c->scc_in_merge = isl_calloc_array(ctx, int, c->n);
	if (!c->scc || !c->cluster ||
	    !c->scc_cluster || !c->scc_node || !c->scc_in_merge)
		return isl_stat_error;

	for (i = 0; i < c->n; ++i) {
		if (isl_sched_graph_extract_sub_graph(ctx, graph,
					&isl_sched_node_scc_exactly,
					&isl_sched_edge_scc_exactly,
					i, &c->scc[i]) < 0)
			return isl_stat_error;
		c->scc[i].scc = 1;
		if (isl_sched_graph_compute_maxvar(&c->scc[i]) < 0)
			return isl_stat_error;
		if (isl_schedule_node_compute_wcc_band(ctx, &c->scc[i]) < 0)
			return isl_stat_error;
		c->scc_cluster[i] = i;
	}

	return isl_stat_ok;
}

/* Free all memory allocated for "c".
 */
static void clustering_free(isl_ctx *ctx, struct isl_clustering *c)
{
	int i;

	if (c->scc)
		for (i = 0; i < c->n; ++i)
			isl_sched_graph_free(ctx, &c->scc[i]);
	free(c->scc);
	if (c->cluster)
		for (i = 0; i < c->n; ++i)
			isl_sched_graph_free(ctx, &c->cluster[i]);
	free(c->cluster);
	free(c->scc_cluster);
	free(c->scc_node);
	free(c->scc_in_merge);
}

/* Should we refrain from merging the cluster in "graph" with
 * any other cluster?
 * In particular, is its current schedule band empty and incomplete.
 */
static int bad_cluster(struct isl_sched_graph *graph)
{
	return graph->n_row < graph->maxvar &&
		graph->n_total_row == graph->band_start;
}

/* Is "edge" a proximity edge with a non-empty dependence relation?
 */
static isl_bool is_non_empty_proximity(struct isl_sched_edge *edge)
{
	if (!isl_sched_edge_is_proximity(edge))
		return isl_bool_false;
	return isl_bool_not(isl_map_plain_is_empty(edge->map));
}

/* Return the index of an edge in "graph" that can be used to merge
 * two clusters in "c".
 * Return graph->n_edge if no such edge can be found.
 * Return -1 on error.
 *
 * In particular, return a proximity edge between two clusters
 * that is not marked "no_merge" and such that neither of the
 * two clusters has an incomplete, empty band.
 *
 * If there are multiple such edges, then try and find the most
 * appropriate edge to use for merging.  In particular, pick the edge
 * with the greatest weight.  If there are multiple of those,
 * then pick one with the shortest distance between
 * the two cluster representatives.
 */
static int find_proximity(struct isl_sched_graph *graph,
	struct isl_clustering *c)
{
	int i, best = graph->n_edge, best_dist, best_weight;

	for (i = 0; i < graph->n_edge; ++i) {
		struct isl_sched_edge *edge = &graph->edge[i];
		int dist, weight;
		isl_bool prox;

		prox = is_non_empty_proximity(edge);
		if (prox < 0)
			return -1;
		if (!prox)
			continue;
		if (edge->no_merge)
			continue;
		if (bad_cluster(&c->scc[edge->src->scc]) ||
		    bad_cluster(&c->scc[edge->dst->scc]))
			continue;
		dist = c->scc_cluster[edge->dst->scc] -
			c->scc_cluster[edge->src->scc];
		if (dist == 0)
			continue;
		weight = edge->weight;
		if (best < graph->n_edge) {
			if (best_weight > weight)
				continue;
			if (best_weight == weight && best_dist <= dist)
				continue;
		}
		best = i;
		best_dist = dist;
		best_weight = weight;
	}

	return best;
}

/* Internal data structure used in mark_merge_sccs.
 *
 * "graph" is the dependence graph in which a strongly connected
 * component is constructed.
 * "scc_cluster" maps each SCC index to the cluster to which it belongs.
 * "src" and "dst" are the indices of the nodes that are being merged.
 */
struct isl_mark_merge_sccs_data {
	struct isl_sched_graph *graph;
	int *scc_cluster;
	int src;
	int dst;
};

/* Check whether the cluster containing node "i" depends on the cluster
 * containing node "j".  If "i" and "j" belong to the same cluster,
 * then they are taken to depend on each other to ensure that
 * the resulting strongly connected component consists of complete
 * clusters.  Furthermore, if "i" and "j" are the two nodes that
 * are being merged, then they are taken to depend on each other as well.
 * Otherwise, check if there is a (conditional) validity dependence
 * from node[j] to node[i], forcing node[i] to follow node[j].
 */
static isl_bool cluster_follows(int i, int j, void *user)
{
	struct isl_mark_merge_sccs_data *data = user;
	struct isl_sched_graph *graph = data->graph;
	int *scc_cluster = data->scc_cluster;

	if (data->src == i && data->dst == j)
		return isl_bool_true;
	if (data->src == j && data->dst == i)
		return isl_bool_true;
	if (scc_cluster[graph->node[i].scc] == scc_cluster[graph->node[j].scc])
		return isl_bool_true;

	return isl_sched_graph_has_validity_edge(graph, &graph->node[j],
							&graph->node[i]);
}

/* Mark all SCCs that belong to either of the two clusters in "c"
 * connected by the edge in "graph" with index "edge", or to any
 * of the intermediate clusters.
 * The marking is recorded in c->scc_in_merge.
 *
 * The given edge has been selected for merging two clusters,
 * meaning that there is at least a proximity edge between the two nodes.
 * However, there may also be (indirect) validity dependences
 * between the two nodes.  When merging the two clusters, all clusters
 * containing one or more of the intermediate nodes along the
 * indirect validity dependences need to be merged in as well.
 *
 * First collect all such nodes by computing the strongly connected
 * component (SCC) containing the two nodes connected by the edge, where
 * the two nodes are considered to depend on each other to make
 * sure they end up in the same SCC.  Similarly, each node is considered
 * to depend on every other node in the same cluster to ensure
 * that the SCC consists of complete clusters.
 *
 * Then the original SCCs that contain any of these nodes are marked
 * in c->scc_in_merge.
 */
static isl_stat mark_merge_sccs(isl_ctx *ctx, struct isl_sched_graph *graph,
	int edge, struct isl_clustering *c)
{
	struct isl_mark_merge_sccs_data data;
	struct isl_tarjan_graph *g;
	int i;

	for (i = 0; i < c->n; ++i)
		c->scc_in_merge[i] = 0;

	data.graph = graph;
	data.scc_cluster = c->scc_cluster;
	data.src = graph->edge[edge].src - graph->node;
	data.dst = graph->edge[edge].dst - graph->node;

	g = isl_tarjan_graph_component(ctx, graph->n, data.dst,
					&cluster_follows, &data);
	if (!g)
		goto error;

	i = g->op;
	if (i < 3)
		isl_die(ctx, isl_error_internal,
			"expecting at least two nodes in component",
			goto error);
	if (g->order[--i] != -1)
		isl_die(ctx, isl_error_internal,
			"expecting end of component marker", goto error);

	for (--i; i >= 0 && g->order[i] != -1; --i) {
		int scc = graph->node[g->order[i]].scc;
		c->scc_in_merge[scc] = 1;
	}

	isl_tarjan_graph_free(g);
	return isl_stat_ok;
error:
	isl_tarjan_graph_free(g);
	return isl_stat_error;
}

/* Construct the identifier "cluster_i".
 */
static __isl_give isl_id *cluster_id(isl_ctx *ctx, int i)
{
	char name[40];

	snprintf(name, sizeof(name), "cluster_%d", i);
	return isl_id_alloc(ctx, name, NULL);
}

/* Construct the space of the cluster with index "i" containing
 * the strongly connected component "scc".
 *
 * In particular, construct a space called cluster_i with dimension equal
 * to the number of schedule rows in the current band of "scc".
 */
static __isl_give isl_space *cluster_space(struct isl_sched_graph *scc, int i)
{
	int nvar;
	isl_space *space;
	isl_id *id;

	nvar = scc->n_total_row - scc->band_start;
	space = isl_space_copy(scc->node[0].space);
	space = isl_space_params(space);
	space = isl_space_set_from_params(space);
	space = isl_space_add_dims(space, isl_dim_set, nvar);
	id = cluster_id(isl_space_get_ctx(space), i);
	space = isl_space_set_tuple_id(space, isl_dim_set, id);

	return space;
}

/* Collect the domain of the graph for merging clusters.
 *
 * In particular, for each cluster with first SCC "i", construct
 * a set in the space called cluster_i with dimension equal
 * to the number of schedule rows in the current band of the cluster.
 */
static __isl_give isl_union_set *collect_domain(isl_ctx *ctx,
	struct isl_sched_graph *graph, struct isl_clustering *c)
{
	int i;
	isl_space *space;
	isl_union_set *domain;

	space = isl_space_params_alloc(ctx, 0);
	domain = isl_union_set_empty(space);

	for (i = 0; i < graph->scc; ++i) {
		isl_space *space;

		if (!c->scc_in_merge[i])
			continue;
		if (c->scc_cluster[i] != i)
			continue;
		space = cluster_space(&c->scc[i], i);
		domain = isl_union_set_add_set(domain, isl_set_universe(space));
	}

	return domain;
}

/* Construct a map from the original instances to the corresponding
 * cluster instance in the current bands of the clusters in "c".
 */
static __isl_give isl_union_map *collect_cluster_map(isl_ctx *ctx,
	struct isl_sched_graph *graph, struct isl_clustering *c)
{
	int i, j;
	isl_space *space;
	isl_union_map *cluster_map;

	space = isl_space_params_alloc(ctx, 0);
	cluster_map = isl_union_map_empty(space);
	for (i = 0; i < graph->scc; ++i) {
		int start, n;
		isl_id *id;

		if (!c->scc_in_merge[i])
			continue;

		id = cluster_id(ctx, c->scc_cluster[i]);
		start = c->scc[i].band_start;
		n = c->scc[i].n_total_row - start;
		for (j = 0; j < c->scc[i].n; ++j) {
			isl_multi_aff *ma;
			isl_map *map;
			struct isl_sched_node *node = &c->scc[i].node[j];

			ma = isl_sched_node_extract_partial_schedule_multi_aff(
								node, start, n);
			ma = isl_multi_aff_set_tuple_id(ma, isl_dim_out,
							    isl_id_copy(id));
			map = isl_map_from_multi_aff(ma);
			cluster_map = isl_union_map_add_map(cluster_map, map);
		}
		isl_id_free(id);
	}

	return cluster_map;
}

/* Add "umap" to the schedule constraints "sc" of all types of "edge"
 * that are not isl_edge_condition or isl_edge_conditional_validity.
 */
static __isl_give isl_schedule_constraints *add_non_conditional_constraints(
	struct isl_sched_edge *edge, __isl_keep isl_union_map *umap,
	__isl_take isl_schedule_constraints *sc)
{
	enum isl_edge_type t;

	if (!sc)
		return NULL;

	for (t = isl_edge_first; t <= isl_edge_last; ++t) {
		if (t == isl_edge_condition ||
		    t == isl_edge_conditional_validity)
			continue;
		if (!isl_sched_edge_has_type(edge, t))
			continue;
		sc = isl_schedule_constraints_add(sc, t,
						    isl_union_map_copy(umap));
	}

	return sc;
}

/* Add schedule constraints of types isl_edge_condition and
 * isl_edge_conditional_validity to "sc" by applying "umap" to
 * the domains of the wrapped relations in domain and range
 * of the corresponding tagged constraints of "edge".
 */
static __isl_give isl_schedule_constraints *add_conditional_constraints(
	struct isl_sched_edge *edge, __isl_keep isl_union_map *umap,
	__isl_take isl_schedule_constraints *sc)
{
	enum isl_edge_type t;
	isl_union_map *tagged;

	for (t = isl_edge_condition; t <= isl_edge_conditional_validity; ++t) {
		if (!isl_sched_edge_has_type(edge, t))
			continue;
		if (t == isl_edge_condition)
			tagged = isl_union_map_copy(edge->tagged_condition);
		else
			tagged = isl_union_map_copy(edge->tagged_validity);
		tagged = isl_union_map_zip(tagged);
		tagged = isl_union_map_apply_domain(tagged,
					isl_union_map_copy(umap));
		tagged = isl_union_map_zip(tagged);
		sc = isl_schedule_constraints_add(sc, t, tagged);
		if (!sc)
			return NULL;
	}

	return sc;
}

/* Given a mapping "cluster_map" from the original instances to
 * the cluster instances, add schedule constraints on the clusters
 * to "sc" corresponding to the original constraints represented by "edge".
 *
 * For non-tagged dependence constraints, the cluster constraints
 * are obtained by applying "cluster_map" to the edge->map.
 *
 * For tagged dependence constraints, "cluster_map" needs to be applied
 * to the domains of the wrapped relations in domain and range
 * of the tagged dependence constraints.  Pick out the mappings
 * from these domains from "cluster_map" and construct their product.
 * This mapping can then be applied to the pair of domains.
 */
static __isl_give isl_schedule_constraints *collect_edge_constraints(
	struct isl_sched_edge *edge, __isl_keep isl_union_map *cluster_map,
	__isl_take isl_schedule_constraints *sc)
{
	isl_union_map *umap;
	isl_space *space;
	isl_union_set *uset;
	isl_union_map *umap1, *umap2;

	if (!sc)
		return NULL;

	umap = isl_union_map_from_map(isl_map_copy(edge->map));
	umap = isl_union_map_apply_domain(umap,
				isl_union_map_copy(cluster_map));
	umap = isl_union_map_apply_range(umap,
				isl_union_map_copy(cluster_map));
	sc = add_non_conditional_constraints(edge, umap, sc);
	isl_union_map_free(umap);

	if (!sc ||
	    (!isl_sched_edge_is_condition(edge) &&
	     !isl_sched_edge_is_conditional_validity(edge)))
		return sc;

	space = isl_space_domain(isl_map_get_space(edge->map));
	uset = isl_union_set_from_set(isl_set_universe(space));
	umap1 = isl_union_map_copy(cluster_map);
	umap1 = isl_union_map_intersect_domain(umap1, uset);
	space = isl_space_range(isl_map_get_space(edge->map));
	uset = isl_union_set_from_set(isl_set_universe(space));
	umap2 = isl_union_map_copy(cluster_map);
	umap2 = isl_union_map_intersect_domain(umap2, uset);
	umap = isl_union_map_product(umap1, umap2);

	sc = add_conditional_constraints(edge, umap, sc);

	isl_union_map_free(umap);
	return sc;
}

/* Given a mapping "cluster_map" from the original instances to
 * the cluster instances, add schedule constraints on the clusters
 * to "sc" corresponding to all edges in "graph" between nodes that
 * belong to SCCs that are marked for merging in "scc_in_merge".
 */
static __isl_give isl_schedule_constraints *collect_constraints(
	struct isl_sched_graph *graph, int *scc_in_merge,
	__isl_keep isl_union_map *cluster_map,
	__isl_take isl_schedule_constraints *sc)
{
	int i;

	for (i = 0; i < graph->n_edge; ++i) {
		struct isl_sched_edge *edge = &graph->edge[i];

		if (!scc_in_merge[edge->src->scc])
			continue;
		if (!scc_in_merge[edge->dst->scc])
			continue;
		sc = collect_edge_constraints(edge, cluster_map, sc);
	}

	return sc;
}

/* Construct a dependence graph for scheduling clusters with respect
 * to each other and store the result in "merge_graph".
 * In particular, the nodes of the graph correspond to the schedule
 * dimensions of the current bands of those clusters that have been
 * marked for merging in "c".
 *
 * First construct an isl_schedule_constraints object for this domain
 * by transforming the edges in "graph" to the domain.
 * Then initialize a dependence graph for scheduling from these
 * constraints.
 */
static isl_stat init_merge_graph(isl_ctx *ctx, struct isl_sched_graph *graph,
	struct isl_clustering *c, struct isl_sched_graph *merge_graph)
{
	isl_union_set *domain;
	isl_union_map *cluster_map;
	isl_schedule_constraints *sc;
	isl_stat r;

	domain = collect_domain(ctx, graph, c);
	sc = isl_schedule_constraints_on_domain(domain);
	if (!sc)
		return isl_stat_error;
	cluster_map = collect_cluster_map(ctx, graph, c);
	sc = collect_constraints(graph, c->scc_in_merge, cluster_map, sc);
	isl_union_map_free(cluster_map);

	r = isl_sched_graph_init(merge_graph, sc);

	isl_schedule_constraints_free(sc);

	return r;
}

/* Compute the maximal number of remaining schedule rows that still need
 * to be computed for the nodes that belong to clusters with the maximal
 * dimension for the current band (i.e., the band that is to be merged).
 * Only clusters that are about to be merged are considered.
 * "maxvar" is the maximal dimension for the current band.
 * "c" contains information about the clusters.
 *
 * Return the maximal number of remaining schedule rows or
 * isl_size_error on error.
 */
static isl_size compute_maxvar_max_slack(int maxvar, struct isl_clustering *c)
{
	int i, j;
	int max_slack;

	max_slack = 0;
	for (i = 0; i < c->n; ++i) {
		int nvar;
		struct isl_sched_graph *scc;

		if (!c->scc_in_merge[i])
			continue;
		scc = &c->scc[i];
		nvar = scc->n_total_row - scc->band_start;
		if (nvar != maxvar)
			continue;
		for (j = 0; j < scc->n; ++j) {
			struct isl_sched_node *node = &scc->node[j];
			int slack;

			if (isl_sched_node_update_vmap(node) < 0)
				return isl_size_error;
			slack = node->nvar - node->rank;
			if (slack > max_slack)
				max_slack = slack;
		}
	}

	return max_slack;
}

/* If there are any clusters where the dimension of the current band
 * (i.e., the band that is to be merged) is smaller than "maxvar" and
 * if there are any nodes in such a cluster where the number
 * of remaining schedule rows that still need to be computed
 * is greater than "max_slack", then return the smallest current band
 * dimension of all these clusters.  Otherwise return the original value
 * of "maxvar".  Return isl_size_error in case of any error.
 * Only clusters that are about to be merged are considered.
 * "c" contains information about the clusters.
 */
static isl_size limit_maxvar_to_slack(int maxvar, int max_slack,
	struct isl_clustering *c)
{
	int i, j;

	for (i = 0; i < c->n; ++i) {
		int nvar;
		struct isl_sched_graph *scc;

		if (!c->scc_in_merge[i])
			continue;
		scc = &c->scc[i];
		nvar = scc->n_total_row - scc->band_start;
		if (nvar >= maxvar)
			continue;
		for (j = 0; j < scc->n; ++j) {
			struct isl_sched_node *node = &scc->node[j];
			int slack;

			if (isl_sched_node_update_vmap(node) < 0)
				return isl_size_error;
			slack = node->nvar - node->rank;
			if (slack > max_slack) {
				maxvar = nvar;
				break;
			}
		}
	}

	return maxvar;
}

/* Adjust merge_graph->maxvar based on the number of remaining schedule rows
 * that still need to be computed.  In particular, if there is a node
 * in a cluster where the dimension of the current band is smaller
 * than merge_graph->maxvar, but the number of remaining schedule rows
 * is greater than that of any node in a cluster with the maximal
 * dimension for the current band (i.e., merge_graph->maxvar),
 * then adjust merge_graph->maxvar to the (smallest) current band dimension
 * of those clusters.  Without this adjustment, the total number of
 * schedule dimensions would be increased, resulting in a skewed view
 * of the number of coincident dimensions.
 * "c" contains information about the clusters.
 *
 * If the maximize_band_depth option is set and merge_graph->maxvar is reduced,
 * then there is no point in attempting any merge since it will be rejected
 * anyway.  Set merge_graph->maxvar to zero in such cases.
 */
static isl_stat adjust_maxvar_to_slack(isl_ctx *ctx,
	struct isl_sched_graph *merge_graph, struct isl_clustering *c)
{
	isl_size max_slack, maxvar;

	max_slack = compute_maxvar_max_slack(merge_graph->maxvar, c);
	if (max_slack < 0)
		return isl_stat_error;
	maxvar = limit_maxvar_to_slack(merge_graph->maxvar, max_slack, c);
	if (maxvar < 0)
		return isl_stat_error;

	if (maxvar < merge_graph->maxvar) {
		if (isl_options_get_schedule_maximize_band_depth(ctx))
			merge_graph->maxvar = 0;
		else
			merge_graph->maxvar = maxvar;
	}

	return isl_stat_ok;
}

/* Return the number of coincident dimensions in the current band of "graph",
 * where the nodes of "graph" are assumed to be scheduled by a single band.
 */
static int get_n_coincident(struct isl_sched_graph *graph)
{
	int i;

	for (i = graph->band_start; i < graph->n_total_row; ++i)
		if (!graph->node[0].coincident[i])
			break;

	return i - graph->band_start;
}

/* Should the clusters be merged based on the cluster schedule
 * in the current (and only) band of "merge_graph", given that
 * coincidence should be maximized?
 *
 * If the number of coincident schedule dimensions in the merged band
 * would be less than the maximal number of coincident schedule dimensions
 * in any of the merged clusters, then the clusters should not be merged.
 */
static isl_bool ok_to_merge_coincident(struct isl_clustering *c,
	struct isl_sched_graph *merge_graph)
{
	int i;
	int n_coincident;
	int max_coincident;

	max_coincident = 0;
	for (i = 0; i < c->n; ++i) {
		if (!c->scc_in_merge[i])
			continue;
		n_coincident = get_n_coincident(&c->scc[i]);
		if (n_coincident > max_coincident)
			max_coincident = n_coincident;
	}

	n_coincident = get_n_coincident(merge_graph);

	return isl_bool_ok(n_coincident >= max_coincident);
}

/* Return the transformation on "node" expressed by the current (and only)
 * band of "merge_graph" applied to the clusters in "c".
 *
 * First find the representation of "node" in its SCC in "c" and
 * extract the transformation expressed by the current band.
 * Then extract the transformation applied by "merge_graph"
 * to the cluster to which this SCC belongs.
 * Combine the two to obtain the complete transformation on the node.
 *
 * Note that the range of the first transformation is an anonymous space,
 * while the domain of the second is named "cluster_X".  The range
 * of the former therefore needs to be adjusted before the two
 * can be combined.
 */
static __isl_give isl_map *extract_node_transformation(isl_ctx *ctx,
	struct isl_sched_node *node, struct isl_clustering *c,
	struct isl_sched_graph *merge_graph)
{
	struct isl_sched_node *scc_node, *cluster_node;
	int start, n;
	isl_id *id;
	isl_space *space;
	isl_multi_aff *ma, *ma2;

	scc_node = isl_sched_graph_find_node(ctx, &c->scc[node->scc],
						node->space);
	if (scc_node && !isl_sched_graph_is_node(&c->scc[node->scc], scc_node))
		isl_die(ctx, isl_error_internal, "unable to find node",
			return NULL);
	start = c->scc[node->scc].band_start;
	n = c->scc[node->scc].n_total_row - start;
	ma = isl_sched_node_extract_partial_schedule_multi_aff(scc_node,
								start, n);
	space = cluster_space(&c->scc[node->scc], c->scc_cluster[node->scc]);
	cluster_node = isl_sched_graph_find_node(ctx, merge_graph, space);
	if (cluster_node && !isl_sched_graph_is_node(merge_graph, cluster_node))
		isl_die(ctx, isl_error_internal, "unable to find cluster",
			space = isl_space_free(space));
	id = isl_space_get_tuple_id(space, isl_dim_set);
	ma = isl_multi_aff_set_tuple_id(ma, isl_dim_out, id);
	isl_space_free(space);
	n = merge_graph->n_total_row;
	ma2 = isl_sched_node_extract_partial_schedule_multi_aff(cluster_node,
								0, n);
	ma = isl_multi_aff_pullback_multi_aff(ma2, ma);

	return isl_map_from_multi_aff(ma);
}

/* Give a set of distances "set", are they bounded by a small constant
 * in direction "pos"?
 * In practice, check if they are bounded by 2 by checking that there
 * are no elements with a value greater than or equal to 3 or
 * smaller than or equal to -3.
 */
static isl_bool distance_is_bounded(__isl_keep isl_set *set, int pos)
{
	isl_bool bounded;
	isl_set *test;

	if (!set)
		return isl_bool_error;

	test = isl_set_copy(set);
	test = isl_set_lower_bound_si(test, isl_dim_set, pos, 3);
	bounded = isl_set_is_empty(test);
	isl_set_free(test);

	if (bounded < 0 || !bounded)
		return bounded;

	test = isl_set_copy(set);
	test = isl_set_upper_bound_si(test, isl_dim_set, pos, -3);
	bounded = isl_set_is_empty(test);
	isl_set_free(test);

	return bounded;
}

/* Does the set "set" have a fixed (but possible parametric) value
 * at dimension "pos"?
 */
static isl_bool has_single_value(__isl_keep isl_set *set, int pos)
{
	isl_size n;
	isl_bool single;

	n = isl_set_dim(set, isl_dim_set);
	if (n < 0)
		return isl_bool_error;
	set = isl_set_copy(set);
	set = isl_set_project_out(set, isl_dim_set, pos + 1, n - (pos + 1));
	set = isl_set_project_out(set, isl_dim_set, 0, pos);
	single = isl_set_is_singleton(set);
	isl_set_free(set);

	return single;
}

/* Does "map" have a fixed (but possible parametric) value
 * at dimension "pos" of either its domain or its range?
 */
static isl_bool has_singular_src_or_dst(__isl_keep isl_map *map, int pos)
{
	isl_set *set;
	isl_bool single;

	set = isl_map_domain(isl_map_copy(map));
	single = has_single_value(set, pos);
	isl_set_free(set);

	if (single < 0 || single)
		return single;

	set = isl_map_range(isl_map_copy(map));
	single = has_single_value(set, pos);
	isl_set_free(set);

	return single;
}

/* Does the edge "edge" from "graph" have bounded dependence distances
 * in the merged graph "merge_graph" of a selection of clusters in "c"?
 *
 * Extract the complete transformations of the source and destination
 * nodes of the edge, apply them to the edge constraints and
 * compute the differences.  Finally, check if these differences are bounded
 * in each direction.
 *
 * If the dimension of the band is greater than the number of
 * dimensions that can be expected to be optimized by the edge
 * (based on its weight), then also allow the differences to be unbounded
 * in the remaining dimensions, but only if either the source or
 * the destination has a fixed value in that direction.
 * This allows a statement that produces values that are used by
 * several instances of another statement to be merged with that
 * other statement.
 * However, merging such clusters will introduce an inherently
 * large proximity distance inside the merged cluster, meaning
 * that proximity distances will no longer be optimized in
 * subsequent merges.  These merges are therefore only allowed
 * after all other possible merges have been tried.
 * The first time such a merge is encountered, the weight of the edge
 * is replaced by a negative weight.  The second time (i.e., after
 * all merges over edges with a non-negative weight have been tried),
 * the merge is allowed.
 */
static isl_bool has_bounded_distances(isl_ctx *ctx, struct isl_sched_edge *edge,
	struct isl_sched_graph *graph, struct isl_clustering *c,
	struct isl_sched_graph *merge_graph)
{
	int i, n_slack;
	isl_size n;
	isl_bool bounded;
	isl_map *map, *t;
	isl_set *dist;

	map = isl_map_copy(edge->map);
	t = extract_node_transformation(ctx, edge->src, c, merge_graph);
	map = isl_map_apply_domain(map, t);
	t = extract_node_transformation(ctx, edge->dst, c, merge_graph);
	map = isl_map_apply_range(map, t);
	dist = isl_map_deltas(isl_map_copy(map));

	bounded = isl_bool_true;
	n = isl_set_dim(dist, isl_dim_set);
	if (n < 0)
		goto error;
	n_slack = n - edge->weight;
	if (edge->weight < 0)
		n_slack -= graph->max_weight + 1;
	for (i = 0; i < n; ++i) {
		isl_bool bounded_i, singular_i;

		bounded_i = distance_is_bounded(dist, i);
		if (bounded_i < 0)
			goto error;
		if (bounded_i)
			continue;
		if (edge->weight >= 0)
			bounded = isl_bool_false;
		n_slack--;
		if (n_slack < 0)
			break;
		singular_i = has_singular_src_or_dst(map, i);
		if (singular_i < 0)
			goto error;
		if (singular_i)
			continue;
		bounded = isl_bool_false;
		break;
	}
	if (!bounded && i >= n && edge->weight >= 0)
		edge->weight -= graph->max_weight + 1;
	isl_map_free(map);
	isl_set_free(dist);

	return bounded;
error:
	isl_map_free(map);
	isl_set_free(dist);
	return isl_bool_error;
}

/* Should the clusters be merged based on the cluster schedule
 * in the current (and only) band of "merge_graph"?
 * "graph" is the original dependence graph, while "c" records
 * which SCCs are involved in the latest merge.
 *
 * In particular, is there at least one proximity constraint
 * that is optimized by the merge?
 *
 * A proximity constraint is considered to be optimized
 * if the dependence distances are small.
 */
static isl_bool ok_to_merge_proximity(isl_ctx *ctx,
	struct isl_sched_graph *graph, struct isl_clustering *c,
	struct isl_sched_graph *merge_graph)
{
	int i;

	for (i = 0; i < graph->n_edge; ++i) {
		struct isl_sched_edge *edge = &graph->edge[i];
		isl_bool bounded;

		if (!isl_sched_edge_is_proximity(edge))
			continue;
		if (!c->scc_in_merge[edge->src->scc])
			continue;
		if (!c->scc_in_merge[edge->dst->scc])
			continue;
		if (c->scc_cluster[edge->dst->scc] ==
		    c->scc_cluster[edge->src->scc])
			continue;
		bounded = has_bounded_distances(ctx, edge, graph, c,
						merge_graph);
		if (bounded < 0 || bounded)
			return bounded;
	}

	return isl_bool_false;
}

/* Should the clusters be merged based on the cluster schedule
 * in the current (and only) band of "merge_graph"?
 * "graph" is the original dependence graph, while "c" records
 * which SCCs are involved in the latest merge.
 *
 * If the current band is empty, then the clusters should not be merged.
 *
 * If the band depth should be maximized and the merge schedule
 * is incomplete (meaning that the dimension of some of the schedule
 * bands in the original schedule will be reduced), then the clusters
 * should not be merged.
 *
 * If the schedule_maximize_coincidence option is set, then check that
 * the number of coincident schedule dimensions is not reduced.
 *
 * Finally, only allow the merge if at least one proximity
 * constraint is optimized.
 */
static isl_bool ok_to_merge(isl_ctx *ctx, struct isl_sched_graph *graph,
	struct isl_clustering *c, struct isl_sched_graph *merge_graph)
{
	if (merge_graph->n_total_row == merge_graph->band_start)
		return isl_bool_false;

	if (isl_options_get_schedule_maximize_band_depth(ctx) &&
	    merge_graph->n_total_row < merge_graph->maxvar)
		return isl_bool_false;

	if (isl_options_get_schedule_maximize_coincidence(ctx)) {
		isl_bool ok;

		ok = ok_to_merge_coincident(c, merge_graph);
		if (ok < 0 || !ok)
			return ok;
	}

	return ok_to_merge_proximity(ctx, graph, c, merge_graph);
}

/* Apply the schedule in "t_node" to the "n" rows starting at "first"
 * of the schedule in "node" and return the result.
 *
 * That is, essentially compute
 *
 *	T * N(first:first+n-1)
 *
 * taking into account the constant term and the parameter coefficients
 * in "t_node".
 */
static __isl_give isl_mat *node_transformation(isl_ctx *ctx,
	struct isl_sched_node *t_node, struct isl_sched_node *node,
	int first, int n)
{
	int i, j;
	isl_mat *t;
	isl_size n_row, n_col;
	int n_param, n_var;

	n_param = node->nparam;
	n_var = node->nvar;
	n_row = isl_mat_rows(t_node->sched);
	n_col = isl_mat_cols(node->sched);
	if (n_row < 0 || n_col < 0)
		return NULL;
	t = isl_mat_alloc(ctx, n_row, n_col);
	if (!t)
		return NULL;
	for (i = 0; i < n_row; ++i) {
		isl_seq_cpy(t->row[i], t_node->sched->row[i], 1 + n_param);
		isl_seq_clr(t->row[i] + 1 + n_param, n_var);
		for (j = 0; j < n; ++j)
			isl_seq_addmul(t->row[i],
					t_node->sched->row[i][1 + n_param + j],
					node->sched->row[first + j],
					1 + n_param + n_var);
	}
	return t;
}

/* Apply the cluster schedule in "t_node" to the current band
 * schedule of the nodes in "graph".
 *
 * In particular, replace the rows starting at band_start
 * by the result of applying the cluster schedule in "t_node"
 * to the original rows.
 *
 * The coincidence of the schedule is determined by the coincidence
 * of the cluster schedule.
 */
static isl_stat transform(isl_ctx *ctx, struct isl_sched_graph *graph,
	struct isl_sched_node *t_node)
{
	int i, j;
	isl_size n_new;
	int start, n;

	start = graph->band_start;
	n = graph->n_total_row - start;

	n_new = isl_mat_rows(t_node->sched);
	if (n_new < 0)
		return isl_stat_error;
	for (i = 0; i < graph->n; ++i) {
		struct isl_sched_node *node = &graph->node[i];
		isl_mat *t;

		t = node_transformation(ctx, t_node, node, start, n);
		node->sched = isl_mat_drop_rows(node->sched, start, n);
		node->sched = isl_mat_concat(node->sched, t);
		node->sched_map = isl_map_free(node->sched_map);
		if (!node->sched)
			return isl_stat_error;
		for (j = 0; j < n_new; ++j)
			node->coincident[start + j] = t_node->coincident[j];
	}
	graph->n_total_row -= n;
	graph->n_row -= n;
	graph->n_total_row += n_new;
	graph->n_row += n_new;

	return isl_stat_ok;
}

/* Merge the clusters marked for merging in "c" into a single
 * cluster using the cluster schedule in the current band of "merge_graph".
 * The representative SCC for the new cluster is the SCC with
 * the smallest index.
 *
 * The current band schedule of each SCC in the new cluster is obtained
 * by applying the schedule of the corresponding original cluster
 * to the original band schedule.
 * All SCCs in the new cluster have the same number of schedule rows.
 */
static isl_stat merge(isl_ctx *ctx, struct isl_clustering *c,
	struct isl_sched_graph *merge_graph)
{
	int i;
	int cluster = -1;
	isl_space *space;

	for (i = 0; i < c->n; ++i) {
		struct isl_sched_node *node;

		if (!c->scc_in_merge[i])
			continue;
		if (cluster < 0)
			cluster = i;
		space = cluster_space(&c->scc[i], c->scc_cluster[i]);
		node = isl_sched_graph_find_node(ctx, merge_graph, space);
		isl_space_free(space);
		if (!node)
			return isl_stat_error;
		if (!isl_sched_graph_is_node(merge_graph, node))
			isl_die(ctx, isl_error_internal,
				"unable to find cluster",
				return isl_stat_error);
		if (transform(ctx, &c->scc[i], node) < 0)
			return isl_stat_error;
		c->scc_cluster[i] = cluster;
	}

	return isl_stat_ok;
}

/* Try and merge the clusters of SCCs marked in c->scc_in_merge
 * by scheduling the current cluster bands with respect to each other.
 *
 * Construct a dependence graph with a space for each cluster and
 * with the coordinates of each space corresponding to the schedule
 * dimensions of the current band of that cluster.
 * Construct a cluster schedule in this cluster dependence graph and
 * apply it to the current cluster bands if it is applicable
 * according to ok_to_merge.
 *
 * If the number of remaining schedule dimensions in a cluster
 * with a non-maximal current schedule dimension is greater than
 * the number of remaining schedule dimensions in clusters
 * with a maximal current schedule dimension, then restrict
 * the number of rows to be computed in the cluster schedule
 * to the minimal such non-maximal current schedule dimension.
 * Do this by adjusting merge_graph.maxvar.
 *
 * Return isl_bool_true if the clusters have effectively been merged
 * into a single cluster.
 *
 * Note that since the standard scheduling algorithm minimizes the maximal
 * distance over proximity constraints, the proximity constraints between
 * the merged clusters may not be optimized any further than what is
 * sufficient to bring the distances within the limits of the internal
 * proximity constraints inside the individual clusters.
 * It may therefore make sense to perform an additional translation step
 * to bring the clusters closer to each other, while maintaining
 * the linear part of the merging schedule found using the standard
 * scheduling algorithm.
 */
static isl_bool try_merge(isl_ctx *ctx, struct isl_sched_graph *graph,
	struct isl_clustering *c)
{
	struct isl_sched_graph merge_graph = { 0 };
	isl_bool merged;

	if (init_merge_graph(ctx, graph, c, &merge_graph) < 0)
		goto error;

	if (isl_sched_graph_compute_maxvar(&merge_graph) < 0)
		goto error;
	if (adjust_maxvar_to_slack(ctx, &merge_graph,c) < 0)
		goto error;
	if (isl_schedule_node_compute_wcc_band(ctx, &merge_graph) < 0)
		goto error;
	merged = ok_to_merge(ctx, graph, c, &merge_graph);
	if (merged && merge(ctx, c, &merge_graph) < 0)
		goto error;

	isl_sched_graph_free(ctx, &merge_graph);
	return merged;
error:
	isl_sched_graph_free(ctx, &merge_graph);
	return isl_bool_error;
}

/* Is there any edge marked "no_merge" between two SCCs that are
 * about to be merged (i.e., that are set in "scc_in_merge")?
 * "merge_edge" is the proximity edge along which the clusters of SCCs
 * are going to be merged.
 *
 * If there is any edge between two SCCs with a negative weight,
 * while the weight of "merge_edge" is non-negative, then this
 * means that the edge was postponed.  "merge_edge" should then
 * also be postponed since merging along the edge with negative weight should
 * be postponed until all edges with non-negative weight have been tried.
 * Replace the weight of "merge_edge" by a negative weight as well and
 * tell the caller not to attempt a merge.
 */
static int any_no_merge(struct isl_sched_graph *graph, int *scc_in_merge,
	struct isl_sched_edge *merge_edge)
{
	int i;

	for (i = 0; i < graph->n_edge; ++i) {
		struct isl_sched_edge *edge = &graph->edge[i];

		if (!scc_in_merge[edge->src->scc])
			continue;
		if (!scc_in_merge[edge->dst->scc])
			continue;
		if (edge->no_merge)
			return 1;
		if (merge_edge->weight >= 0 && edge->weight < 0) {
			merge_edge->weight -= graph->max_weight + 1;
			return 1;
		}
	}

	return 0;
}

/* Merge the two clusters in "c" connected by the edge in "graph"
 * with index "edge" into a single cluster.
 * If it turns out to be impossible to merge these two clusters,
 * then mark the edge as "no_merge" such that it will not be
 * considered again.
 *
 * First mark all SCCs that need to be merged.  This includes the SCCs
 * in the two clusters, but it may also include the SCCs
 * of intermediate clusters.
 * If there is already a no_merge edge between any pair of such SCCs,
 * then simply mark the current edge as no_merge as well.
 * Likewise, if any of those edges was postponed by has_bounded_distances,
 * then postpone the current edge as well.
 * Otherwise, try and merge the clusters and mark "edge" as "no_merge"
 * if the clusters did not end up getting merged, unless the non-merge
 * is due to the fact that the edge was postponed.  This postponement
 * can be recognized by a change in weight (from non-negative to negative).
 */
static isl_stat merge_clusters_along_edge(isl_ctx *ctx,
	struct isl_sched_graph *graph, int edge, struct isl_clustering *c)
{
	isl_bool merged;
	int edge_weight = graph->edge[edge].weight;

	if (mark_merge_sccs(ctx, graph, edge, c) < 0)
		return isl_stat_error;

	if (any_no_merge(graph, c->scc_in_merge, &graph->edge[edge]))
		merged = isl_bool_false;
	else
		merged = try_merge(ctx, graph, c);
	if (merged < 0)
		return isl_stat_error;
	if (!merged && edge_weight == graph->edge[edge].weight)
		graph->edge[edge].no_merge = 1;

	return isl_stat_ok;
}

/* Does "node" belong to the cluster identified by "cluster"?
 */
static int node_cluster_exactly(struct isl_sched_node *node, int cluster)
{
	return node->cluster == cluster;
}

/* Does "edge" connect two nodes belonging to the cluster
 * identified by "cluster"?
 */
static int edge_cluster_exactly(struct isl_sched_edge *edge, int cluster)
{
	return edge->src->cluster == cluster && edge->dst->cluster == cluster;
}

/* Swap the schedule of "node1" and "node2".
 * Both nodes have been derived from the same node in a common parent graph.
 * Since the "coincident" field is shared with that node
 * in the parent graph, there is no need to also swap this field.
 */
static void swap_sched(struct isl_sched_node *node1,
	struct isl_sched_node *node2)
{
	isl_mat *sched;
	isl_map *sched_map;

	sched = node1->sched;
	node1->sched = node2->sched;
	node2->sched = sched;

	sched_map = node1->sched_map;
	node1->sched_map = node2->sched_map;
	node2->sched_map = sched_map;
}

/* Copy the current band schedule from the SCCs that form the cluster
 * with index "pos" to the actual cluster at position "pos".
 * By construction, the index of the first SCC that belongs to the cluster
 * is also "pos".
 *
 * The order of the nodes inside both the SCCs and the cluster
 * is assumed to be same as the order in the original "graph".
 *
 * Since the SCC graphs will no longer be used after this function,
 * the schedules are actually swapped rather than copied.
 */
static isl_stat copy_partial(struct isl_sched_graph *graph,
	struct isl_clustering *c, int pos)
{
	int i, j;

	c->cluster[pos].n_total_row = c->scc[pos].n_total_row;
	c->cluster[pos].n_row = c->scc[pos].n_row;
	c->cluster[pos].maxvar = c->scc[pos].maxvar;
	j = 0;
	for (i = 0; i < graph->n; ++i) {
		int k;
		int s;

		if (graph->node[i].cluster != pos)
			continue;
		s = graph->node[i].scc;
		k = c->scc_node[s]++;
		swap_sched(&c->cluster[pos].node[j], &c->scc[s].node[k]);
		if (c->scc[s].maxvar > c->cluster[pos].maxvar)
			c->cluster[pos].maxvar = c->scc[s].maxvar;
		++j;
	}

	return isl_stat_ok;
}

/* Is there a (conditional) validity dependence from node[j] to node[i],
 * forcing node[i] to follow node[j] or do the nodes belong to the same
 * cluster?
 */
static isl_bool node_follows_strong_or_same_cluster(int i, int j, void *user)
{
	struct isl_sched_graph *graph = user;

	if (graph->node[i].cluster == graph->node[j].cluster)
		return isl_bool_true;
	return isl_sched_graph_has_validity_edge(graph, &graph->node[j],
							&graph->node[i]);
}

/* Extract the merged clusters of SCCs in "graph", sort them, and
 * store them in c->clusters.  Update c->scc_cluster accordingly.
 *
 * First keep track of the cluster containing the SCC to which a node
 * belongs in the node itself.
 * Then extract the clusters into c->clusters, copying the current
 * band schedule from the SCCs that belong to the cluster.
 * Do this only once per cluster.
 *
 * Finally, topologically sort the clusters and update c->scc_cluster
 * to match the new scc numbering.  While the SCCs were originally
 * sorted already, some SCCs that depend on some other SCCs may
 * have been merged with SCCs that appear before these other SCCs.
 * A reordering may therefore be required.
 */
static isl_stat extract_clusters(isl_ctx *ctx, struct isl_sched_graph *graph,
	struct isl_clustering *c)
{
	int i;

	for (i = 0; i < graph->n; ++i)
		graph->node[i].cluster = c->scc_cluster[graph->node[i].scc];

	for (i = 0; i < graph->scc; ++i) {
		if (c->scc_cluster[i] != i)
			continue;
		if (isl_sched_graph_extract_sub_graph(ctx, graph,
				&node_cluster_exactly,
				&edge_cluster_exactly, i, &c->cluster[i]) < 0)
			return isl_stat_error;
		c->cluster[i].src_scc = -1;
		c->cluster[i].dst_scc = -1;
		if (copy_partial(graph, c, i) < 0)
			return isl_stat_error;
	}

	if (isl_sched_graph_detect_ccs(ctx, graph,
				&node_follows_strong_or_same_cluster) < 0)
		return isl_stat_error;
	for (i = 0; i < graph->n; ++i)
		c->scc_cluster[graph->node[i].scc] = graph->node[i].cluster;

	return isl_stat_ok;
}

/* Compute weights on the proximity edges of "graph" that can
 * be used by find_proximity to find the most appropriate
 * proximity edge to use to merge two clusters in "c".
 * The weights are also used by has_bounded_distances to determine
 * whether the merge should be allowed.
 * Store the maximum of the computed weights in graph->max_weight.
 *
 * The computed weight is a measure for the number of remaining schedule
 * dimensions that can still be completely aligned.
 * In particular, compute the number of equalities between
 * input dimensions and output dimensions in the proximity constraints.
 * The directions that are already handled by outer schedule bands
 * are projected out prior to determining this number.
 *
 * Edges that will never be considered by find_proximity are ignored.
 */
static isl_stat compute_weights(struct isl_sched_graph *graph,
	struct isl_clustering *c)
{
	int i;

	graph->max_weight = 0;

	for (i = 0; i < graph->n_edge; ++i) {
		struct isl_sched_edge *edge = &graph->edge[i];
		struct isl_sched_node *src = edge->src;
		struct isl_sched_node *dst = edge->dst;
		isl_basic_map *hull;
		isl_bool prox;
		isl_size n_in, n_out, n;

		prox = is_non_empty_proximity(edge);
		if (prox < 0)
			return isl_stat_error;
		if (!prox)
			continue;
		if (bad_cluster(&c->scc[edge->src->scc]) ||
		    bad_cluster(&c->scc[edge->dst->scc]))
			continue;
		if (c->scc_cluster[edge->dst->scc] ==
		    c->scc_cluster[edge->src->scc])
			continue;

		hull = isl_map_affine_hull(isl_map_copy(edge->map));
		hull = isl_basic_map_transform_dims(hull, isl_dim_in, 0,
						    isl_mat_copy(src->vmap));
		hull = isl_basic_map_transform_dims(hull, isl_dim_out, 0,
						    isl_mat_copy(dst->vmap));
		hull = isl_basic_map_project_out(hull,
						isl_dim_in, 0, src->rank);
		hull = isl_basic_map_project_out(hull,
						isl_dim_out, 0, dst->rank);
		hull = isl_basic_map_remove_divs(hull);
		n_in = isl_basic_map_dim(hull, isl_dim_in);
		n_out = isl_basic_map_dim(hull, isl_dim_out);
		if (n_in < 0 || n_out < 0)
			hull = isl_basic_map_free(hull);
		hull = isl_basic_map_drop_constraints_not_involving_dims(hull,
							isl_dim_in, 0, n_in);
		hull = isl_basic_map_drop_constraints_not_involving_dims(hull,
							isl_dim_out, 0, n_out);
		n = isl_basic_map_n_equality(hull);
		isl_basic_map_free(hull);
		if (n < 0)
			return isl_stat_error;
		edge->weight = n;

		if (edge->weight > graph->max_weight)
			graph->max_weight = edge->weight;
	}

	return isl_stat_ok;
}

/* Call isl_schedule_node_compute_finish_band on each of the clusters in "c" and
 * update "node" to arrange for them to be executed in an order
 * possibly involving set nodes that generalizes the topological order
 * determined by the scc fields of the nodes in "graph".
 *
 * Note that at this stage, there are graph->scc clusters and
 * their positions in c->cluster are determined by the values
 * of c->scc_cluster.
 *
 * Construct an isl_scc_graph and perform the decomposition
 * using this graph.
 */
static __isl_give isl_schedule_node *finish_bands_decompose(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph,
	struct isl_clustering *c)
{
	isl_ctx *ctx;
	struct isl_scc_graph *scc_graph;

	ctx = isl_schedule_node_get_ctx(node);

	scc_graph = isl_scc_graph_from_sched_graph(ctx, graph, c);
	node = isl_scc_graph_decompose(scc_graph, node);
	isl_scc_graph_free(scc_graph);

	return node;
}

/* Call isl_schedule_node_compute_finish_band on each of the clusters in "c"
 * in their topological order.  This order is determined by the scc
 * fields of the nodes in "graph".
 * Combine the results in a sequence expressing the topological order.
 *
 * If there is only one cluster left, then there is no need to introduce
 * a sequence node.  Also, in this case, the cluster necessarily contains
 * the SCC at position 0 in the original graph and is therefore also
 * stored in the first cluster of "c".
 *
 * If there are more than two clusters left, then some subsets of the clusters
 * may still be independent of each other.  These could then still
 * be reordered with respect to each other.  Call finish_bands_decompose
 * to try and construct an ordering involving set and sequence nodes
 * that generalizes the topological order.
 * Note that at the outermost level there can be no independent components
 * because isl_schedule_node_compute_wcc_clustering is called
 * on a (weakly) connected component.
 */
static __isl_give isl_schedule_node *finish_bands_clustering(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph,
	struct isl_clustering *c)
{
	int i;
	isl_ctx *ctx;
	isl_union_set_list *filters;

	if (graph->scc == 1)
		return isl_schedule_node_compute_finish_band(node,
							&c->cluster[0], 0);
	if (graph->scc > 2)
		return finish_bands_decompose(node, graph, c);

	ctx = isl_schedule_node_get_ctx(node);

	filters = isl_sched_graph_extract_sccs(ctx, graph);
	node = isl_schedule_node_insert_sequence(node, filters);

	for (i = 0; i < graph->scc; ++i) {
		int j = c->scc_cluster[i];
		node = isl_schedule_node_grandchild(node, i, 0);
		node = isl_schedule_node_compute_finish_band(node,
							&c->cluster[j], 0);
		node = isl_schedule_node_grandparent(node);
	}

	return node;
}

/* Compute a schedule for a connected dependence graph by first considering
 * each strongly connected component (SCC) in the graph separately and then
 * incrementally combining them into clusters.
 * Return the updated schedule node.
 *
 * Initially, each cluster consists of a single SCC, each with its
 * own band schedule.  The algorithm then tries to merge pairs
 * of clusters along a proximity edge until no more suitable
 * proximity edges can be found.  During this merging, the schedule
 * is maintained in the individual SCCs.
 * After the merging is completed, the full resulting clusters
 * are extracted and in finish_bands_clustering,
 * isl_schedule_node_compute_finish_band is called on each of them to integrate
 * the band into "node" and to continue the computation.
 *
 * compute_weights initializes the weights that are used by find_proximity.
 */
__isl_give isl_schedule_node *isl_schedule_node_compute_wcc_clustering(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph)
{
	isl_ctx *ctx;
	struct isl_clustering c;
	int i;

	ctx = isl_schedule_node_get_ctx(node);

	if (clustering_init(ctx, &c, graph) < 0)
		goto error;

	if (compute_weights(graph, &c) < 0)
		goto error;

	for (;;) {
		i = find_proximity(graph, &c);
		if (i < 0)
			goto error;
		if (i >= graph->n_edge)
			break;
		if (merge_clusters_along_edge(ctx, graph, i, &c) < 0)
			goto error;
	}

	if (extract_clusters(ctx, graph, &c) < 0)
		goto error;

	node = finish_bands_clustering(node, graph, &c);

	clustering_free(ctx, &c);
	return node;
error:
	clustering_free(ctx, &c);
	return isl_schedule_node_free(node);
}
