/*
 * Copyright 2011      INRIA Saclay
 * Copyright 2012-2014 Ecole Normale Superieure
 * Copyright 2015-2016 Sven Verdoolaege
 * Copyright 2016      INRIA Paris
 * Copyright 2017      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 * and Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 * and Centre de Recherche Inria de Paris, 2 rue Simone Iff - Voie DQ12,
 * CS 42112, 75589 Paris Cedex 12, France
 */

#include <isl_ctx_private.h>
#include <isl_map_private.h>
#include <isl_space_private.h>
#include <isl_aff_private.h>
#include <isl/hash.h>
#include <isl/id.h>
#include <isl/constraint.h>
#include <isl/schedule.h>
#include <isl_schedule_constraints.h>
#include <isl/schedule_node.h>
#include <isl_mat_private.h>
#include <isl_vec_private.h>
#include <isl/set.h>
#include <isl_union_set_private.h>
#include <isl_seq.h>
#include <isl_tab.h>
#include <isl_dim_map.h>
#include <isl/map_to_basic_set.h>
#include <isl_sort.h>
#include <isl_options_private.h>
#include <isl_tarjan.h>
#include <isl_morph.h>
#include <isl/ilp.h>
#include <isl_val_private.h>

#include "isl_scheduler.h"
#include "isl_scheduler_clustering.h"

/*
 * The scheduling algorithm implemented in this file was inspired by
 * Bondhugula et al., "Automatic Transformations for Communication-Minimized
 * Parallelization and Locality Optimization in the Polyhedral Model".
 *
 * For a detailed description of the variant implemented in isl,
 * see Verdoolaege and Janssens, "Scheduling for PPCG" (2017).
 */


static isl_bool node_has_tuples(const void *entry, const void *val)
{
	struct isl_sched_node *node = (struct isl_sched_node *)entry;
	isl_space *space = (isl_space *) val;

	return isl_space_has_equal_tuples(node->space, space);
}

int isl_sched_node_scc_exactly(struct isl_sched_node *node, int scc)
{
	return node->scc == scc;
}

static int node_scc_at_most(struct isl_sched_node *node, int scc)
{
	return node->scc <= scc;
}

static int node_scc_at_least(struct isl_sched_node *node, int scc)
{
	return node->scc >= scc;
}

/* Is "edge" marked as being of type "type"?
 */
int isl_sched_edge_has_type(struct isl_sched_edge *edge,
	enum isl_edge_type type)
{
	return ISL_FL_ISSET(edge->types, 1 << type);
}

/* Mark "edge" as being of type "type".
 */
static void set_type(struct isl_sched_edge *edge, enum isl_edge_type type)
{
	ISL_FL_SET(edge->types, 1 << type);
}

/* No longer mark "edge" as being of type "type"?
 */
static void clear_type(struct isl_sched_edge *edge, enum isl_edge_type type)
{
	ISL_FL_CLR(edge->types, 1 << type);
}

/* Is "edge" marked as a validity edge?
 */
static int is_validity(struct isl_sched_edge *edge)
{
	return isl_sched_edge_has_type(edge, isl_edge_validity);
}

/* Mark "edge" as a validity edge.
 */
static void set_validity(struct isl_sched_edge *edge)
{
	set_type(edge, isl_edge_validity);
}

/* Is "edge" marked as a proximity edge?
 */
int isl_sched_edge_is_proximity(struct isl_sched_edge *edge)
{
	return isl_sched_edge_has_type(edge, isl_edge_proximity);
}

/* Is "edge" marked as a local edge?
 */
static int is_local(struct isl_sched_edge *edge)
{
	return isl_sched_edge_has_type(edge, isl_edge_local);
}

/* Mark "edge" as a local edge.
 */
static void set_local(struct isl_sched_edge *edge)
{
	set_type(edge, isl_edge_local);
}

/* No longer mark "edge" as a local edge.
 */
static void clear_local(struct isl_sched_edge *edge)
{
	clear_type(edge, isl_edge_local);
}

/* Is "edge" marked as a coincidence edge?
 */
static int is_coincidence(struct isl_sched_edge *edge)
{
	return isl_sched_edge_has_type(edge, isl_edge_coincidence);
}

/* Is "edge" marked as a condition edge?
 */
int isl_sched_edge_is_condition(struct isl_sched_edge *edge)
{
	return isl_sched_edge_has_type(edge, isl_edge_condition);
}

/* Is "edge" marked as a conditional validity edge?
 */
int isl_sched_edge_is_conditional_validity(struct isl_sched_edge *edge)
{
	return isl_sched_edge_has_type(edge, isl_edge_conditional_validity);
}

/* Is "edge" of a type that can appear multiple times between
 * the same pair of nodes?
 *
 * Condition edges and conditional validity edges may have tagged
 * dependence relations, in which case an edge is added for each
 * pair of tags.
 */
static int is_multi_edge_type(struct isl_sched_edge *edge)
{
	return isl_sched_edge_is_condition(edge) ||
		isl_sched_edge_is_conditional_validity(edge);
}

/* Initialize node_table based on the list of nodes.
 */
static int graph_init_table(isl_ctx *ctx, struct isl_sched_graph *graph)
{
	int i;

	graph->node_table = isl_hash_table_alloc(ctx, graph->n);
	if (!graph->node_table)
		return -1;

	for (i = 0; i < graph->n; ++i) {
		struct isl_hash_table_entry *entry;
		uint32_t hash;

		hash = isl_space_get_tuple_hash(graph->node[i].space);
		entry = isl_hash_table_find(ctx, graph->node_table, hash,
					    &node_has_tuples,
					    graph->node[i].space, 1);
		if (!entry)
			return -1;
		entry->data = &graph->node[i];
	}

	return 0;
}

/* Return a pointer to the node that lives within the given space,
 * an invalid node if there is no such node, or NULL in case of error.
 */
struct isl_sched_node *isl_sched_graph_find_node(isl_ctx *ctx,
	struct isl_sched_graph *graph, __isl_keep isl_space *space)
{
	struct isl_hash_table_entry *entry;
	uint32_t hash;

	if (!space)
		return NULL;

	hash = isl_space_get_tuple_hash(space);
	entry = isl_hash_table_find(ctx, graph->node_table, hash,
				    &node_has_tuples, space, 0);
	if (!entry)
		return NULL;
	if (entry == isl_hash_table_entry_none)
		return graph->node + graph->n;

	return entry->data;
}

/* Is "node" a node in "graph"?
 */
int isl_sched_graph_is_node(struct isl_sched_graph *graph,
	struct isl_sched_node *node)
{
	return node && node >= &graph->node[0] && node < &graph->node[graph->n];
}

static isl_bool edge_has_src_and_dst(const void *entry, const void *val)
{
	const struct isl_sched_edge *edge = entry;
	const struct isl_sched_edge *temp = val;

	return isl_bool_ok(edge->src == temp->src && edge->dst == temp->dst);
}

/* Add the given edge to graph->edge_table[type].
 */
static isl_stat graph_edge_table_add(isl_ctx *ctx,
	struct isl_sched_graph *graph, enum isl_edge_type type,
	struct isl_sched_edge *edge)
{
	struct isl_hash_table_entry *entry;
	uint32_t hash;

	hash = isl_hash_init();
	hash = isl_hash_builtin(hash, edge->src);
	hash = isl_hash_builtin(hash, edge->dst);
	entry = isl_hash_table_find(ctx, graph->edge_table[type], hash,
				    &edge_has_src_and_dst, edge, 1);
	if (!entry)
		return isl_stat_error;
	entry->data = edge;

	return isl_stat_ok;
}

/* Add "edge" to all relevant edge tables.
 * That is, for every type of the edge, add it to the corresponding table.
 */
static isl_stat graph_edge_tables_add(isl_ctx *ctx,
	struct isl_sched_graph *graph, struct isl_sched_edge *edge)
{
	enum isl_edge_type t;

	for (t = isl_edge_first; t <= isl_edge_last; ++t) {
		if (!isl_sched_edge_has_type(edge, t))
			continue;
		if (graph_edge_table_add(ctx, graph, t, edge) < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Allocate the edge_tables based on the maximal number of edges of
 * each type.
 */
static int graph_init_edge_tables(isl_ctx *ctx, struct isl_sched_graph *graph)
{
	int i;

	for (i = 0; i <= isl_edge_last; ++i) {
		graph->edge_table[i] = isl_hash_table_alloc(ctx,
							    graph->max_edge[i]);
		if (!graph->edge_table[i])
			return -1;
	}

	return 0;
}

/* If graph->edge_table[type] contains an edge from the given source
 * to the given destination, then return the hash table entry of this edge.
 * Otherwise, return NULL.
 */
static struct isl_hash_table_entry *graph_find_edge_entry(
	struct isl_sched_graph *graph,
	enum isl_edge_type type,
	struct isl_sched_node *src, struct isl_sched_node *dst)
{
	isl_ctx *ctx = isl_space_get_ctx(src->space);
	uint32_t hash;
	struct isl_sched_edge temp = { .src = src, .dst = dst };

	hash = isl_hash_init();
	hash = isl_hash_builtin(hash, temp.src);
	hash = isl_hash_builtin(hash, temp.dst);
	return isl_hash_table_find(ctx, graph->edge_table[type], hash,
				    &edge_has_src_and_dst, &temp, 0);
}


/* If graph->edge_table[type] contains an edge from the given source
 * to the given destination, then return this edge.
 * Return "none" if no such edge can be found.
 * Return NULL on error.
 */
static struct isl_sched_edge *graph_find_edge(struct isl_sched_graph *graph,
	enum isl_edge_type type,
	struct isl_sched_node *src, struct isl_sched_node *dst,
	struct isl_sched_edge *none)
{
	struct isl_hash_table_entry *entry;

	entry = graph_find_edge_entry(graph, type, src, dst);
	if (!entry)
		return NULL;
	if (entry == isl_hash_table_entry_none)
		return none;

	return entry->data;
}

/* Check whether the dependence graph has an edge of the given type
 * between the given two nodes.
 */
static isl_bool graph_has_edge(struct isl_sched_graph *graph,
	enum isl_edge_type type,
	struct isl_sched_node *src, struct isl_sched_node *dst)
{
	struct isl_sched_edge dummy;
	struct isl_sched_edge *edge;
	isl_bool empty;

	edge = graph_find_edge(graph, type, src, dst, &dummy);
	if (!edge)
		return isl_bool_error;
	if (edge == &dummy)
		return isl_bool_false;

	empty = isl_map_plain_is_empty(edge->map);

	return isl_bool_not(empty);
}

/* Look for any edge with the same src, dst and map fields as "model".
 *
 * Return the matching edge if one can be found.
 * Return "model" if no matching edge is found.
 * Return NULL on error.
 */
static struct isl_sched_edge *graph_find_matching_edge(
	struct isl_sched_graph *graph, struct isl_sched_edge *model)
{
	enum isl_edge_type i;
	struct isl_sched_edge *edge;

	for (i = isl_edge_first; i <= isl_edge_last; ++i) {
		int is_equal;

		edge = graph_find_edge(graph, i, model->src, model->dst, model);
		if (!edge)
			return NULL;
		if (edge == model)
			continue;
		is_equal = isl_map_plain_is_equal(model->map, edge->map);
		if (is_equal < 0)
			return NULL;
		if (is_equal)
			return edge;
	}

	return model;
}

/* Remove the given edge from all the edge_tables that refer to it.
 */
static isl_stat graph_remove_edge(struct isl_sched_graph *graph,
	struct isl_sched_edge *edge)
{
	isl_ctx *ctx = isl_map_get_ctx(edge->map);
	enum isl_edge_type i;

	for (i = isl_edge_first; i <= isl_edge_last; ++i) {
		struct isl_hash_table_entry *entry;

		entry = graph_find_edge_entry(graph, i, edge->src, edge->dst);
		if (!entry)
			return isl_stat_error;
		if (entry == isl_hash_table_entry_none)
			continue;
		if (entry->data != edge)
			continue;
		isl_hash_table_remove(ctx, graph->edge_table[i], entry);
	}

	return isl_stat_ok;
}

/* Check whether the dependence graph has any edge
 * between the given two nodes.
 */
static isl_bool graph_has_any_edge(struct isl_sched_graph *graph,
	struct isl_sched_node *src, struct isl_sched_node *dst)
{
	enum isl_edge_type i;
	isl_bool r;

	for (i = isl_edge_first; i <= isl_edge_last; ++i) {
		r = graph_has_edge(graph, i, src, dst);
		if (r < 0 || r)
			return r;
	}

	return r;
}

/* Check whether the dependence graph has a validity edge
 * between the given two nodes.
 *
 * Conditional validity edges are essentially validity edges that
 * can be ignored if the corresponding condition edges are iteration private.
 * Here, we are only checking for the presence of validity
 * edges, so we need to consider the conditional validity edges too.
 * In particular, this function is used during the detection
 * of strongly connected components and we cannot ignore
 * conditional validity edges during this detection.
 */
isl_bool isl_sched_graph_has_validity_edge(struct isl_sched_graph *graph,
	struct isl_sched_node *src, struct isl_sched_node *dst)
{
	isl_bool r;

	r = graph_has_edge(graph, isl_edge_validity, src, dst);
	if (r < 0 || r)
		return r;

	return graph_has_edge(graph, isl_edge_conditional_validity, src, dst);
}

/* Perform all the required memory allocations for a schedule graph "graph"
 * with "n_node" nodes and "n_edge" edge and initialize the corresponding
 * fields.
 */
static isl_stat graph_alloc(isl_ctx *ctx, struct isl_sched_graph *graph,
	int n_node, int n_edge)
{
	int i;

	graph->n = n_node;
	graph->n_edge = n_edge;
	graph->node = isl_calloc_array(ctx, struct isl_sched_node, graph->n);
	graph->sorted = isl_calloc_array(ctx, int, graph->n);
	graph->region = isl_alloc_array(ctx,
					struct isl_trivial_region, graph->n);
	graph->edge = isl_calloc_array(ctx,
					struct isl_sched_edge, graph->n_edge);

	graph->intra_hmap = isl_map_to_basic_set_alloc(ctx, 2 * n_edge);
	graph->intra_hmap_param = isl_map_to_basic_set_alloc(ctx, 2 * n_edge);
	graph->inter_hmap = isl_map_to_basic_set_alloc(ctx, 2 * n_edge);

	if (!graph->node || !graph->region || (graph->n_edge && !graph->edge) ||
	    !graph->sorted)
		return isl_stat_error;

	for(i = 0; i < graph->n; ++i)
		graph->sorted[i] = i;

	return isl_stat_ok;
}

/* Free the memory associated to node "node" in "graph".
 * The "coincident" field is shared by nodes in a graph and its subgraph.
 * It therefore only needs to be freed for the original dependence graph,
 * i.e., one that is not the result of splitting.
 */
static void clear_node(struct isl_sched_graph *graph,
	struct isl_sched_node *node)
{
	isl_space_free(node->space);
	isl_set_free(node->hull);
	isl_multi_aff_free(node->compress);
	isl_pw_multi_aff_free(node->decompress);
	isl_mat_free(node->sched);
	isl_map_free(node->sched_map);
	isl_mat_free(node->indep);
	isl_mat_free(node->vmap);
	if (graph->root == graph)
		free(node->coincident);
	isl_multi_val_free(node->sizes);
	isl_basic_set_free(node->bounds);
	isl_vec_free(node->max);
}

void isl_sched_graph_free(isl_ctx *ctx, struct isl_sched_graph *graph)
{
	int i;

	isl_map_to_basic_set_free(graph->intra_hmap);
	isl_map_to_basic_set_free(graph->intra_hmap_param);
	isl_map_to_basic_set_free(graph->inter_hmap);

	if (graph->node)
		for (i = 0; i < graph->n; ++i)
			clear_node(graph, &graph->node[i]);
	free(graph->node);
	free(graph->sorted);
	if (graph->edge)
		for (i = 0; i < graph->n_edge; ++i) {
			isl_map_free(graph->edge[i].map);
			isl_union_map_free(graph->edge[i].tagged_condition);
			isl_union_map_free(graph->edge[i].tagged_validity);
		}
	free(graph->edge);
	free(graph->region);
	for (i = 0; i <= isl_edge_last; ++i)
		isl_hash_table_free(ctx, graph->edge_table[i]);
	isl_hash_table_free(ctx, graph->node_table);
	isl_basic_set_free(graph->lp);
}

/* For each "set" on which this function is called, increment
 * graph->n by one and update graph->maxvar.
 */
static isl_stat init_n_maxvar(__isl_take isl_set *set, void *user)
{
	struct isl_sched_graph *graph = user;
	isl_size nvar = isl_set_dim(set, isl_dim_set);

	graph->n++;
	if (nvar > graph->maxvar)
		graph->maxvar = nvar;

	isl_set_free(set);

	if (nvar < 0)
		return isl_stat_error;
	return isl_stat_ok;
}

/* Compute the number of rows that should be allocated for the schedule.
 * In particular, we need one row for each variable or one row
 * for each basic map in the dependences.
 * Note that it is practically impossible to exhaust both
 * the number of dependences and the number of variables.
 */
static isl_stat compute_max_row(struct isl_sched_graph *graph,
	__isl_keep isl_schedule_constraints *sc)
{
	int n_edge;
	isl_stat r;
	isl_union_set *domain;

	graph->n = 0;
	graph->maxvar = 0;
	domain = isl_schedule_constraints_get_domain(sc);
	r = isl_union_set_foreach_set(domain, &init_n_maxvar, graph);
	isl_union_set_free(domain);
	if (r < 0)
		return isl_stat_error;
	n_edge = isl_schedule_constraints_n_basic_map(sc);
	if (n_edge < 0)
		return isl_stat_error;
	graph->max_row = n_edge + graph->maxvar;

	return isl_stat_ok;
}

/* Does "bset" have any defining equalities for its set variables?
 */
static isl_bool has_any_defining_equality(__isl_keep isl_basic_set *bset)
{
	int i;
	isl_size n;

	n = isl_basic_set_dim(bset, isl_dim_set);
	if (n < 0)
		return isl_bool_error;

	for (i = 0; i < n; ++i) {
		isl_bool has;

		has = isl_basic_set_has_defining_equality(bset, isl_dim_set, i,
							NULL);
		if (has < 0 || has)
			return has;
	}

	return isl_bool_false;
}

/* Set the entries of node->max to the value of the schedule_max_coefficient
 * option, if set.
 */
static isl_stat set_max_coefficient(isl_ctx *ctx, struct isl_sched_node *node)
{
	int max;

	max = isl_options_get_schedule_max_coefficient(ctx);
	if (max == -1)
		return isl_stat_ok;

	node->max = isl_vec_alloc(ctx, node->nvar);
	node->max = isl_vec_set_si(node->max, max);
	if (!node->max)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Set the entries of node->max to the minimum of the schedule_max_coefficient
 * option (if set) and half of the minimum of the sizes in the other
 * dimensions.  Round up when computing the half such that
 * if the minimum of the sizes is one, half of the size is taken to be one
 * rather than zero.
 * If the global minimum is unbounded (i.e., if both
 * the schedule_max_coefficient is not set and the sizes in the other
 * dimensions are unbounded), then store a negative value.
 * If the schedule coefficient is close to the size of the instance set
 * in another dimension, then the schedule may represent a loop
 * coalescing transformation (especially if the coefficient
 * in that other dimension is one).  Forcing the coefficient to be
 * smaller than or equal to half the minimal size should avoid this
 * situation.
 */
static isl_stat compute_max_coefficient(isl_ctx *ctx,
	struct isl_sched_node *node)
{
	int max;
	int i, j;
	isl_vec *v;

	max = isl_options_get_schedule_max_coefficient(ctx);
	v = isl_vec_alloc(ctx, node->nvar);
	if (!v)
		return isl_stat_error;

	for (i = 0; i < node->nvar; ++i) {
		isl_int_set_si(v->el[i], max);
		isl_int_mul_si(v->el[i], v->el[i], 2);
	}

	for (i = 0; i < node->nvar; ++i) {
		isl_val *size;

		size = isl_multi_val_get_val(node->sizes, i);
		if (!size)
			goto error;
		if (!isl_val_is_int(size)) {
			isl_val_free(size);
			continue;
		}
		for (j = 0; j < node->nvar; ++j) {
			if (j == i)
				continue;
			if (isl_int_is_neg(v->el[j]) ||
			    isl_int_gt(v->el[j], size->n))
				isl_int_set(v->el[j], size->n);
		}
		isl_val_free(size);
	}

	for (i = 0; i < node->nvar; ++i)
		isl_int_cdiv_q_ui(v->el[i], v->el[i], 2);

	node->max = v;
	return isl_stat_ok;
error:
	isl_vec_free(v);
	return isl_stat_error;
}

/* Construct an identifier for node "node", which will represent "set".
 * The name of the identifier is either "compressed" or
 * "compressed_<name>", with <name> the name of the space of "set".
 * The user pointer of the identifier points to "node".
 */
static __isl_give isl_id *construct_compressed_id(__isl_keep isl_set *set,
	struct isl_sched_node *node)
{
	isl_bool has_name;
	isl_ctx *ctx;
	isl_id *id;
	isl_printer *p;
	const char *name;
	char *id_name;

	has_name = isl_set_has_tuple_name(set);
	if (has_name < 0)
		return NULL;

	ctx = isl_set_get_ctx(set);
	if (!has_name)
		return isl_id_alloc(ctx, "compressed", node);

	p = isl_printer_to_str(ctx);
	name = isl_set_get_tuple_name(set);
	p = isl_printer_print_str(p, "compressed_");
	p = isl_printer_print_str(p, name);
	id_name = isl_printer_get_str(p);
	isl_printer_free(p);

	id = isl_id_alloc(ctx, id_name, node);
	free(id_name);

	return id;
}

/* Construct a map that isolates the variable in position "pos" in "set".
 *
 * That is, construct
 *
 *	[i_0, ..., i_pos-1, i_pos+1, ...] -> [i_pos]
 */
static __isl_give isl_map *isolate(__isl_take isl_set *set, int pos)
{
	isl_map *map;

	map = isl_set_project_onto_map(set, isl_dim_set, pos, 1);
	map = isl_map_project_out(map, isl_dim_in, pos, 1);
	return map;
}

/* Compute and return the size of "set" in dimension "dim".
 * The size is taken to be the difference in values for that variable
 * for fixed values of the other variables.
 * This assumes that "set" is convex.
 * In particular, the variable is first isolated from the other variables
 * in the range of a map
 *
 *	[i_0, ..., i_dim-1, i_dim+1, ...] -> [i_dim]
 *
 * and then duplicated
 *
 *	[i_0, ..., i_dim-1, i_dim+1, ...] -> [[i_dim] -> [i_dim']]
 *
 * The shared variables are then projected out and the maximal value
 * of i_dim' - i_dim is computed.
 */
static __isl_give isl_val *compute_size(__isl_take isl_set *set, int dim)
{
	isl_map *map;
	isl_local_space *ls;
	isl_aff *obj;
	isl_val *v;

	map = isolate(set, dim);
	map = isl_map_range_product(map, isl_map_copy(map));
	map = isl_set_unwrap(isl_map_range(map));
	set = isl_map_deltas(map);
	ls = isl_local_space_from_space(isl_set_get_space(set));
	obj = isl_aff_var_on_domain(ls, isl_dim_set, 0);
	v = isl_set_max_val(set, obj);
	isl_aff_free(obj);
	isl_set_free(set);

	return v;
}

/* Perform a compression on "node" where "hull" represents the constraints
 * that were used to derive the compression, while "compress" and
 * "decompress" map the original space to the compressed space and
 * vice versa.
 *
 * If "node" was not compressed already, then simply store
 * the compression information.
 * Otherwise the "original" space is actually the result
 * of a previous compression, which is then combined
 * with the present compression.
 *
 * The dimensionality of the compressed domain is also adjusted.
 * Other information, such as the sizes and the maximal coefficient values,
 * has not been computed yet and therefore does not need to be adjusted.
 */
static isl_stat compress_node(struct isl_sched_node *node,
	__isl_take isl_set *hull, __isl_take isl_multi_aff *compress,
	__isl_take isl_pw_multi_aff *decompress)
{
	node->nvar = isl_multi_aff_dim(compress, isl_dim_out);
	if (!node->compressed) {
		node->compressed = 1;
		node->hull = hull;
		node->compress = compress;
		node->decompress = decompress;
	} else {
		hull = isl_set_preimage_multi_aff(hull,
					    isl_multi_aff_copy(node->compress));
		node->hull = isl_set_intersect(node->hull, hull);
		node->compress = isl_multi_aff_pullback_multi_aff(
						compress, node->compress);
		node->decompress = isl_pw_multi_aff_pullback_pw_multi_aff(
						node->decompress, decompress);
	}

	if (!node->hull || !node->compress || !node->decompress)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Given that dimension "pos" in "set" has a fixed value
 * in terms of the other dimensions, (further) compress "node"
 * by projecting out this dimension.
 * "set" may be the result of a previous compression.
 * "uncompressed" is the original domain (without compression).
 *
 * The compression function simply projects out the dimension.
 * The decompression function adds back the dimension
 * in the right position as an expression of the other dimensions
 * derived from "set".
 * As in extract_node, the compressed space has an identifier
 * that references "node" such that each compressed space is unique and
 * such that the node can be recovered from the compressed space.
 *
 * The constraint removed through the compression is added to the "hull"
 * such that only edges that relate to the original domains
 * are taken into account.
 * In particular, it is obtained by composing compression and decompression and
 * taking the relation among the variables in the range.
 */
static isl_stat project_out_fixed(struct isl_sched_node *node,
	__isl_keep isl_set *uncompressed, __isl_take isl_set *set, int pos)
{
	isl_id *id;
	isl_space *space;
	isl_set *domain;
	isl_map *map;
	isl_multi_aff *compress;
	isl_pw_multi_aff *decompress, *pma;
	isl_multi_pw_aff *mpa;
	isl_set *hull;

	map = isolate(isl_set_copy(set), pos);
	pma = isl_pw_multi_aff_from_map(map);
	domain = isl_pw_multi_aff_domain(isl_pw_multi_aff_copy(pma));
	pma = isl_pw_multi_aff_gist(pma, domain);
	space = isl_pw_multi_aff_get_domain_space(pma);
	mpa = isl_multi_pw_aff_identity(isl_space_map_from_set(space));
	mpa = isl_multi_pw_aff_range_splice(mpa, pos,
				    isl_multi_pw_aff_from_pw_multi_aff(pma));
	decompress = isl_pw_multi_aff_from_multi_pw_aff(mpa);
	space = isl_set_get_space(set);
	compress = isl_multi_aff_project_out_map(space, isl_dim_set, pos, 1);
	id = construct_compressed_id(uncompressed, node);
	compress = isl_multi_aff_set_tuple_id(compress, isl_dim_out, id);
	space = isl_space_reverse(isl_multi_aff_get_space(compress));
	decompress = isl_pw_multi_aff_reset_space(decompress, space);
	pma = isl_pw_multi_aff_pullback_multi_aff(
	    isl_pw_multi_aff_copy(decompress), isl_multi_aff_copy(compress));
	hull = isl_map_range(isl_map_from_pw_multi_aff(pma));

	isl_set_free(set);

	return compress_node(node, hull, compress, decompress);
}

/* Compute the size of the compressed domain in each dimension and
 * store the results in node->sizes.
 * "uncompressed" is the original domain (without compression).
 *
 * First compress the domain if needed and then compute the size
 * in each direction.
 * If the domain is not convex, then the sizes are computed
 * on a convex superset in order to avoid picking up sizes
 * that are valid for the individual disjuncts, but not for
 * the domain as a whole.
 *
 * If any of the sizes turns out to be zero, then this means
 * that this dimension has a fixed value in terms of
 * the other dimensions.  Perform an (extra) compression
 * to remove this dimension.
 */
static isl_stat compute_sizes(struct isl_sched_node *node,
	__isl_keep isl_set *uncompressed)
{
	int j;
	isl_size n;
	isl_multi_val *mv;
	isl_set *set = isl_set_copy(uncompressed);

	if (node->compressed)
		set = isl_set_preimage_pw_multi_aff(set,
				    isl_pw_multi_aff_copy(node->decompress));
	set = isl_set_from_basic_set(isl_set_simple_hull(set));
	mv = isl_multi_val_zero(isl_set_get_space(set));
	n = isl_set_dim(set, isl_dim_set);
	if (n < 0)
		mv = isl_multi_val_free(mv);
	for (j = 0; j < n; ++j) {
		isl_bool is_zero;
		isl_val *v;

		v = compute_size(isl_set_copy(set), j);
		is_zero = isl_val_is_zero(v);
		mv = isl_multi_val_set_val(mv, j, v);
		if (is_zero >= 0 && is_zero) {
			isl_multi_val_free(mv);
			if (project_out_fixed(node, uncompressed, set, j) < 0)
				return isl_stat_error;
			return compute_sizes(node, uncompressed);
		}
	}
	node->sizes = mv;
	isl_set_free(set);
	if (!node->sizes)
		return isl_stat_error;
	return isl_stat_ok;
}

/* Compute the size of the instance set "set" of "node", after compression,
 * as well as bounds on the corresponding coefficients, if needed.
 *
 * The sizes are needed when the schedule_treat_coalescing option is set.
 * The bounds are needed when the schedule_treat_coalescing option or
 * the schedule_max_coefficient option is set.
 *
 * If the schedule_treat_coalescing option is not set, then at most
 * the bounds need to be set and this is done in set_max_coefficient.
 * Otherwise, compute the size of the compressed domain
 * in each direction and store the results in node->size.
 * Finally, set the bounds on the coefficients based on the sizes
 * and the schedule_max_coefficient option in compute_max_coefficient.
 */
static isl_stat compute_sizes_and_max(isl_ctx *ctx, struct isl_sched_node *node,
	__isl_take isl_set *set)
{
	isl_stat r;

	if (!isl_options_get_schedule_treat_coalescing(ctx)) {
		isl_set_free(set);
		return set_max_coefficient(ctx, node);
	}

	r = compute_sizes(node, set);
	isl_set_free(set);
	if (r < 0)
		return isl_stat_error;
	return compute_max_coefficient(ctx, node);
}

/* Add a new node to the graph representing the given instance set.
 * "nvar" is the (possibly compressed) number of variables and
 * may be smaller than then number of set variables in "set"
 * if "compressed" is set.
 * If "compressed" is set, then "hull" represents the constraints
 * that were used to derive the compression, while "compress" and
 * "decompress" map the original space to the compressed space and
 * vice versa.
 * If "compressed" is not set, then "hull", "compress" and "decompress"
 * should be NULL.
 *
 * Compute the size of the instance set and bounds on the coefficients,
 * if needed.
 */
static isl_stat add_node(struct isl_sched_graph *graph,
	__isl_take isl_set *set, int nvar, int compressed,
	__isl_take isl_set *hull, __isl_take isl_multi_aff *compress,
	__isl_take isl_pw_multi_aff *decompress)
{
	isl_size nparam;
	isl_ctx *ctx;
	isl_mat *sched;
	isl_space *space;
	int *coincident;
	struct isl_sched_node *node;

	nparam = isl_set_dim(set, isl_dim_param);
	if (nparam < 0)
		goto error;

	ctx = isl_set_get_ctx(set);
	if (!ctx->opt->schedule_parametric)
		nparam = 0;
	sched = isl_mat_alloc(ctx, 0, 1 + nparam + nvar);
	node = &graph->node[graph->n];
	graph->n++;
	space = isl_set_get_space(set);
	node->space = space;
	node->nvar = nvar;
	node->nparam = nparam;
	node->sched = sched;
	node->sched_map = NULL;
	coincident = isl_calloc_array(ctx, int, graph->max_row);
	node->coincident = coincident;
	node->compressed = compressed;
	node->hull = hull;
	node->compress = compress;
	node->decompress = decompress;
	if (compute_sizes_and_max(ctx, node, set) < 0)
		return isl_stat_error;

	if (!space || !sched || (graph->max_row && !coincident))
		return isl_stat_error;
	if (compressed && (!hull || !compress || !decompress))
		return isl_stat_error;

	return isl_stat_ok;
error:
	isl_set_free(set);
	isl_set_free(hull);
	isl_multi_aff_free(compress);
	isl_pw_multi_aff_free(decompress);
	return isl_stat_error;
}

/* Add a new node to the graph representing the given set.
 *
 * If any of the set variables is defined by an equality, then
 * we perform variable compression such that we can perform
 * the scheduling on the compressed domain.
 * In this case, an identifier is used that references the new node
 * such that each compressed space is unique and
 * such that the node can be recovered from the compressed space.
 */
static isl_stat extract_node(__isl_take isl_set *set, void *user)
{
	isl_size nvar;
	isl_bool has_equality;
	isl_id *id;
	isl_basic_set *hull;
	isl_set *hull_set;
	isl_morph *morph;
	isl_multi_aff *compress, *decompress_ma;
	isl_pw_multi_aff *decompress;
	struct isl_sched_graph *graph = user;

	hull = isl_set_affine_hull(isl_set_copy(set));
	hull = isl_basic_set_remove_divs(hull);
	nvar = isl_set_dim(set, isl_dim_set);
	has_equality = has_any_defining_equality(hull);

	if (nvar < 0 || has_equality < 0)
		goto error;
	if (!has_equality) {
		isl_basic_set_free(hull);
		return add_node(graph, set, nvar, 0, NULL, NULL, NULL);
	}

	id = construct_compressed_id(set, &graph->node[graph->n]);
	morph = isl_basic_set_variable_compression_with_id(hull, id);
	isl_id_free(id);
	nvar = isl_morph_ran_dim(morph, isl_dim_set);
	if (nvar < 0)
		set = isl_set_free(set);
	compress = isl_morph_get_var_multi_aff(morph);
	morph = isl_morph_inverse(morph);
	decompress_ma = isl_morph_get_var_multi_aff(morph);
	decompress = isl_pw_multi_aff_from_multi_aff(decompress_ma);
	isl_morph_free(morph);

	hull_set = isl_set_from_basic_set(hull);
	return add_node(graph, set, nvar, 1, hull_set, compress, decompress);
error:
	isl_basic_set_free(hull);
	isl_set_free(set);
	return isl_stat_error;
}

struct isl_extract_edge_data {
	enum isl_edge_type type;
	struct isl_sched_graph *graph;
};

/* Merge edge2 into edge1, freeing the contents of edge2.
 * Return 0 on success and -1 on failure.
 *
 * edge1 and edge2 are assumed to have the same value for the map field.
 */
static int merge_edge(struct isl_sched_edge *edge1,
	struct isl_sched_edge *edge2)
{
	edge1->types |= edge2->types;
	isl_map_free(edge2->map);

	if (isl_sched_edge_is_condition(edge2)) {
		if (!edge1->tagged_condition)
			edge1->tagged_condition = edge2->tagged_condition;
		else
			edge1->tagged_condition =
				isl_union_map_union(edge1->tagged_condition,
						    edge2->tagged_condition);
	}

	if (isl_sched_edge_is_conditional_validity(edge2)) {
		if (!edge1->tagged_validity)
			edge1->tagged_validity = edge2->tagged_validity;
		else
			edge1->tagged_validity =
				isl_union_map_union(edge1->tagged_validity,
						    edge2->tagged_validity);
	}

	if (isl_sched_edge_is_condition(edge2) && !edge1->tagged_condition)
		return -1;
	if (isl_sched_edge_is_conditional_validity(edge2) &&
	    !edge1->tagged_validity)
		return -1;

	return 0;
}

/* Insert dummy tags in domain and range of "map".
 *
 * In particular, if "map" is of the form
 *
 *	A -> B
 *
 * then return
 *
 *	[A -> dummy_tag] -> [B -> dummy_tag]
 *
 * where the dummy_tags are identical and equal to any dummy tags
 * introduced by any other call to this function.
 */
static __isl_give isl_map *insert_dummy_tags(__isl_take isl_map *map)
{
	static char dummy;
	isl_ctx *ctx;
	isl_id *id;
	isl_space *space;
	isl_set *domain, *range;

	ctx = isl_map_get_ctx(map);

	id = isl_id_alloc(ctx, NULL, &dummy);
	space = isl_space_params(isl_map_get_space(map));
	space = isl_space_set_from_params(space);
	space = isl_space_set_tuple_id(space, isl_dim_set, id);
	space = isl_space_map_from_set(space);

	domain = isl_map_wrap(map);
	range = isl_map_wrap(isl_map_universe(space));
	map = isl_map_from_domain_and_range(domain, range);
	map = isl_map_zip(map);

	return map;
}

/* Given that at least one of "src" or "dst" is compressed, return
 * a map between the spaces of these nodes restricted to the affine
 * hull that was used in the compression.
 */
static __isl_give isl_map *extract_hull(struct isl_sched_node *src,
	struct isl_sched_node *dst)
{
	isl_set *dom, *ran;

	if (src->compressed)
		dom = isl_set_copy(src->hull);
	else
		dom = isl_set_universe(isl_space_copy(src->space));
	if (dst->compressed)
		ran = isl_set_copy(dst->hull);
	else
		ran = isl_set_universe(isl_space_copy(dst->space));

	return isl_map_from_domain_and_range(dom, ran);
}

/* Intersect the domains of the nested relations in domain and range
 * of "tagged" with "map".
 */
static __isl_give isl_map *map_intersect_domains(__isl_take isl_map *tagged,
	__isl_keep isl_map *map)
{
	isl_set *set;

	tagged = isl_map_zip(tagged);
	set = isl_map_wrap(isl_map_copy(map));
	tagged = isl_map_intersect_domain(tagged, set);
	tagged = isl_map_zip(tagged);
	return tagged;
}

/* Return a pointer to the node that lives in the domain space of "map",
 * an invalid node if there is no such node, or NULL in case of error.
 */
static struct isl_sched_node *find_domain_node(isl_ctx *ctx,
	struct isl_sched_graph *graph, __isl_keep isl_map *map)
{
	struct isl_sched_node *node;
	isl_space *space;

	space = isl_space_domain(isl_map_get_space(map));
	node = isl_sched_graph_find_node(ctx, graph, space);
	isl_space_free(space);

	return node;
}

/* Return a pointer to the node that lives in the range space of "map",
 * an invalid node if there is no such node, or NULL in case of error.
 */
static struct isl_sched_node *find_range_node(isl_ctx *ctx,
	struct isl_sched_graph *graph, __isl_keep isl_map *map)
{
	struct isl_sched_node *node;
	isl_space *space;

	space = isl_space_range(isl_map_get_space(map));
	node = isl_sched_graph_find_node(ctx, graph, space);
	isl_space_free(space);

	return node;
}

/* Refrain from adding a new edge based on "map".
 * Instead, just free the map.
 * "tagged" is either a copy of "map" with additional tags or NULL.
 */
static isl_stat skip_edge(__isl_take isl_map *map, __isl_take isl_map *tagged)
{
	isl_map_free(map);
	isl_map_free(tagged);

	return isl_stat_ok;
}

/* Add a new edge to the graph based on the given map
 * and add it to data->graph->edge_table[data->type].
 * If a dependence relation of a given type happens to be identical
 * to one of the dependence relations of a type that was added before,
 * then we don't create a new edge, but instead mark the original edge
 * as also representing a dependence of the current type.
 *
 * Edges of type isl_edge_condition or isl_edge_conditional_validity
 * may be specified as "tagged" dependence relations.  That is, "map"
 * may contain elements (i -> a) -> (j -> b), where i -> j denotes
 * the dependence on iterations and a and b are tags.
 * edge->map is set to the relation containing the elements i -> j,
 * while edge->tagged_condition and edge->tagged_validity contain
 * the union of all the "map" relations
 * for which extract_edge is called that result in the same edge->map.
 *
 * If the source or the destination node is compressed, then
 * intersect both "map" and "tagged" with the constraints that
 * were used to construct the compression.
 * This ensures that there are no schedule constraints defined
 * outside of these domains, while the scheduler no longer has
 * any control over those outside parts.
 */
static isl_stat extract_edge(__isl_take isl_map *map, void *user)
{
	isl_bool empty;
	isl_ctx *ctx = isl_map_get_ctx(map);
	struct isl_extract_edge_data *data = user;
	struct isl_sched_graph *graph = data->graph;
	struct isl_sched_node *src, *dst;
	struct isl_sched_edge *edge;
	isl_map *tagged = NULL;

	if (data->type == isl_edge_condition ||
	    data->type == isl_edge_conditional_validity) {
		if (isl_map_can_zip(map)) {
			tagged = isl_map_copy(map);
			map = isl_set_unwrap(isl_map_domain(isl_map_zip(map)));
		} else {
			tagged = insert_dummy_tags(isl_map_copy(map));
		}
	}

	src = find_domain_node(ctx, graph, map);
	dst = find_range_node(ctx, graph, map);

	if (!src || !dst)
		goto error;
	if (!isl_sched_graph_is_node(graph, src) ||
	    !isl_sched_graph_is_node(graph, dst))
		return skip_edge(map, tagged);

	if (src->compressed || dst->compressed) {
		isl_map *hull;
		hull = extract_hull(src, dst);
		if (tagged)
			tagged = map_intersect_domains(tagged, hull);
		map = isl_map_intersect(map, hull);
	}

	empty = isl_map_plain_is_empty(map);
	if (empty < 0)
		goto error;
	if (empty)
		return skip_edge(map, tagged);

	graph->edge[graph->n_edge].src = src;
	graph->edge[graph->n_edge].dst = dst;
	graph->edge[graph->n_edge].map = map;
	graph->edge[graph->n_edge].types = 0;
	graph->edge[graph->n_edge].tagged_condition = NULL;
	graph->edge[graph->n_edge].tagged_validity = NULL;
	set_type(&graph->edge[graph->n_edge], data->type);
	if (data->type == isl_edge_condition)
		graph->edge[graph->n_edge].tagged_condition =
					isl_union_map_from_map(tagged);
	if (data->type == isl_edge_conditional_validity)
		graph->edge[graph->n_edge].tagged_validity =
					isl_union_map_from_map(tagged);

	edge = graph_find_matching_edge(graph, &graph->edge[graph->n_edge]);
	if (!edge) {
		graph->n_edge++;
		return isl_stat_error;
	}
	if (edge == &graph->edge[graph->n_edge])
		return graph_edge_table_add(ctx, graph, data->type,
				    &graph->edge[graph->n_edge++]);

	if (merge_edge(edge, &graph->edge[graph->n_edge]) < 0)
		return isl_stat_error;

	return graph_edge_table_add(ctx, graph, data->type, edge);
error:
	isl_map_free(map);
	isl_map_free(tagged);
	return isl_stat_error;
}

/* Initialize the schedule graph "graph" from the schedule constraints "sc".
 *
 * The context is included in the domain before the nodes of
 * the graphs are extracted in order to be able to exploit
 * any possible additional equalities.
 * Note that this intersection is only performed locally here.
 */
isl_stat isl_sched_graph_init(struct isl_sched_graph *graph,
	__isl_keep isl_schedule_constraints *sc)
{
	isl_ctx *ctx;
	isl_union_set *domain;
	isl_union_map *c;
	struct isl_extract_edge_data data;
	enum isl_edge_type i;
	isl_stat r;
	isl_size n;

	if (!sc)
		return isl_stat_error;

	ctx = isl_schedule_constraints_get_ctx(sc);

	domain = isl_schedule_constraints_get_domain(sc);
	n = isl_union_set_n_set(domain);
	graph->n = n;
	isl_union_set_free(domain);
	if (n < 0)
		return isl_stat_error;

	n = isl_schedule_constraints_n_map(sc);
	if (n < 0 || graph_alloc(ctx, graph, graph->n, n) < 0)
		return isl_stat_error;

	if (compute_max_row(graph, sc) < 0)
		return isl_stat_error;
	graph->root = graph;
	graph->n = 0;
	domain = isl_schedule_constraints_get_domain(sc);
	domain = isl_union_set_intersect_params(domain,
				    isl_schedule_constraints_get_context(sc));
	r = isl_union_set_foreach_set(domain, &extract_node, graph);
	isl_union_set_free(domain);
	if (r < 0)
		return isl_stat_error;
	if (graph_init_table(ctx, graph) < 0)
		return isl_stat_error;
	for (i = isl_edge_first; i <= isl_edge_last; ++i) {
		isl_size n;

		c = isl_schedule_constraints_get(sc, i);
		n = isl_union_map_n_map(c);
		graph->max_edge[i] = n;
		isl_union_map_free(c);
		if (n < 0)
			return isl_stat_error;
	}
	if (graph_init_edge_tables(ctx, graph) < 0)
		return isl_stat_error;
	graph->n_edge = 0;
	data.graph = graph;
	for (i = isl_edge_first; i <= isl_edge_last; ++i) {
		isl_stat r;

		data.type = i;
		c = isl_schedule_constraints_get(sc, i);
		r = isl_union_map_foreach_map(c, &extract_edge, &data);
		isl_union_map_free(c);
		if (r < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Check whether there is any dependence from node[j] to node[i]
 * or from node[i] to node[j].
 */
static isl_bool node_follows_weak(int i, int j, void *user)
{
	isl_bool f;
	struct isl_sched_graph *graph = user;

	f = graph_has_any_edge(graph, &graph->node[j], &graph->node[i]);
	if (f < 0 || f)
		return f;
	return graph_has_any_edge(graph, &graph->node[i], &graph->node[j]);
}

/* Check whether there is a (conditional) validity dependence from node[j]
 * to node[i], forcing node[i] to follow node[j].
 */
static isl_bool node_follows_strong(int i, int j, void *user)
{
	struct isl_sched_graph *graph = user;

	return isl_sched_graph_has_validity_edge(graph, &graph->node[j],
							&graph->node[i]);
}

/* Use Tarjan's algorithm for computing the strongly connected components
 * in the dependence graph only considering those edges defined by "follows".
 */
isl_stat isl_sched_graph_detect_ccs(isl_ctx *ctx,
	struct isl_sched_graph *graph,
	isl_bool (*follows)(int i, int j, void *user))
{
	int i, n;
	struct isl_tarjan_graph *g = NULL;

	g = isl_tarjan_graph_init(ctx, graph->n, follows, graph);
	if (!g)
		return isl_stat_error;

	graph->scc = 0;
	i = 0;
	n = graph->n;
	while (n) {
		while (g->order[i] != -1) {
			graph->node[g->order[i]].scc = graph->scc;
			--n;
			++i;
		}
		++i;
		graph->scc++;
	}

	isl_tarjan_graph_free(g);

	return isl_stat_ok;
}

/* Apply Tarjan's algorithm to detect the strongly connected components
 * in the dependence graph.
 * Only consider the (conditional) validity dependences and clear "weak".
 */
static isl_stat detect_sccs(isl_ctx *ctx, struct isl_sched_graph *graph)
{
	graph->weak = 0;
	return isl_sched_graph_detect_ccs(ctx, graph, &node_follows_strong);
}

/* Apply Tarjan's algorithm to detect the (weakly) connected components
 * in the dependence graph.
 * Consider all dependences and set "weak".
 */
static isl_stat detect_wccs(isl_ctx *ctx, struct isl_sched_graph *graph)
{
	graph->weak = 1;
	return isl_sched_graph_detect_ccs(ctx, graph, &node_follows_weak);
}

static int cmp_scc(const void *a, const void *b, void *data)
{
	struct isl_sched_graph *graph = data;
	const int *i1 = a;
	const int *i2 = b;

	return graph->node[*i1].scc - graph->node[*i2].scc;
}

/* Sort the elements of graph->sorted according to the corresponding SCCs.
 */
static int sort_sccs(struct isl_sched_graph *graph)
{
	return isl_sort(graph->sorted, graph->n, sizeof(int), &cmp_scc, graph);
}

/* Return a non-parametric set in the compressed space of "node" that is
 * bounded by the size in each direction
 *
 *	{ [x] : -S_i <= x_i <= S_i }
 *
 * If S_i is infinity in direction i, then there are no constraints
 * in that direction.
 *
 * Cache the result in node->bounds.
 */
static __isl_give isl_basic_set *get_size_bounds(struct isl_sched_node *node)
{
	isl_space *space;
	isl_basic_set *bounds;
	int i;

	if (node->bounds)
		return isl_basic_set_copy(node->bounds);

	if (node->compressed)
		space = isl_pw_multi_aff_get_domain_space(node->decompress);
	else
		space = isl_space_copy(node->space);
	space = isl_space_drop_all_params(space);
	bounds = isl_basic_set_universe(space);

	for (i = 0; i < node->nvar; ++i) {
		isl_val *size;

		size = isl_multi_val_get_val(node->sizes, i);
		if (!size)
			return isl_basic_set_free(bounds);
		if (!isl_val_is_int(size)) {
			isl_val_free(size);
			continue;
		}
		bounds = isl_basic_set_upper_bound_val(bounds, isl_dim_set, i,
							isl_val_copy(size));
		bounds = isl_basic_set_lower_bound_val(bounds, isl_dim_set, i,
							isl_val_neg(size));
	}

	node->bounds = isl_basic_set_copy(bounds);
	return bounds;
}

/* Compress the dependence relation "map", if needed, i.e.,
 * when the source node "src" and/or the destination node "dst"
 * has been compressed.
 */
static __isl_give isl_map *compress(__isl_take isl_map *map,
	struct isl_sched_node *src, struct isl_sched_node *dst)
{
	if (src->compressed)
		map = isl_map_preimage_domain_pw_multi_aff(map,
					isl_pw_multi_aff_copy(src->decompress));
	if (dst->compressed)
		map = isl_map_preimage_range_pw_multi_aff(map,
					isl_pw_multi_aff_copy(dst->decompress));
	return map;
}

/* Drop some constraints from "delta" that could be exploited
 * to construct loop coalescing schedules.
 * In particular, drop those constraint that bound the difference
 * to the size of the domain.
 * First project out the parameters to improve the effectiveness.
 */
static __isl_give isl_set *drop_coalescing_constraints(
	__isl_take isl_set *delta, struct isl_sched_node *node)
{
	isl_size nparam;
	isl_basic_set *bounds;

	nparam = isl_set_dim(delta, isl_dim_param);
	if (nparam < 0)
		return isl_set_free(delta);

	bounds = get_size_bounds(node);

	delta = isl_set_project_out(delta, isl_dim_param, 0, nparam);
	delta = isl_set_remove_divs(delta);
	delta = isl_set_plain_gist_basic_set(delta, bounds);
	return delta;
}

/* Given a dependence relation R from "node" to itself,
 * construct the set of coefficients of valid constraints for elements
 * in that dependence relation.
 * In particular, the result contains tuples of coefficients
 * c_0, c_n, c_x such that
 *
 *	c_0 + c_n n + c_x y - c_x x >= 0 for each (x,y) in R
 *
 * or, equivalently,
 *
 *	c_0 + c_n n + c_x d >= 0 for each d in delta R = { y - x | (x,y) in R }
 *
 * We choose here to compute the dual of delta R.
 * Alternatively, we could have computed the dual of R, resulting
 * in a set of tuples c_0, c_n, c_x, c_y, and then
 * plugged in (c_0, c_n, c_x, -c_x).
 *
 * If "need_param" is set, then the resulting coefficients effectively
 * include coefficients for the parameters c_n.  Otherwise, they may
 * have been projected out already.
 * Since the constraints may be different for these two cases,
 * they are stored in separate caches.
 * In particular, if no parameter coefficients are required and
 * the schedule_treat_coalescing option is set, then the parameters
 * are projected out and some constraints that could be exploited
 * to construct coalescing schedules are removed before the dual
 * is computed.
 *
 * If "node" has been compressed, then the dependence relation
 * is also compressed before the set of coefficients is computed.
 */
static __isl_give isl_basic_set *intra_coefficients(
	struct isl_sched_graph *graph, struct isl_sched_node *node,
	__isl_take isl_map *map, int need_param)
{
	isl_ctx *ctx;
	isl_set *delta;
	isl_map *key;
	isl_basic_set *coef;
	isl_maybe_isl_basic_set m;
	isl_map_to_basic_set **hmap = &graph->intra_hmap;
	int treat;

	if (!map)
		return NULL;

	ctx = isl_map_get_ctx(map);
	treat = !need_param && isl_options_get_schedule_treat_coalescing(ctx);
	if (!treat)
		hmap = &graph->intra_hmap_param;
	m = isl_map_to_basic_set_try_get(*hmap, map);
	if (m.valid < 0 || m.valid) {
		isl_map_free(map);
		return m.value;
	}

	key = isl_map_copy(map);
	map = compress(map, node, node);
	delta = isl_map_deltas(map);
	if (treat)
		delta = drop_coalescing_constraints(delta, node);
	delta = isl_set_remove_divs(delta);
	coef = isl_set_coefficients(delta);
	*hmap = isl_map_to_basic_set_set(*hmap, key, isl_basic_set_copy(coef));

	return coef;
}

/* Given a dependence relation R, construct the set of coefficients
 * of valid constraints for elements in that dependence relation.
 * In particular, the result contains tuples of coefficients
 * c_0, c_n, c_x, c_y such that
 *
 *	c_0 + c_n n + c_x x + c_y y >= 0 for each (x,y) in R
 *
 * If the source or destination nodes of "edge" have been compressed,
 * then the dependence relation is also compressed before
 * the set of coefficients is computed.
 */
static __isl_give isl_basic_set *inter_coefficients(
	struct isl_sched_graph *graph, struct isl_sched_edge *edge,
	__isl_take isl_map *map)
{
	isl_set *set;
	isl_map *key;
	isl_basic_set *coef;
	isl_maybe_isl_basic_set m;

	m = isl_map_to_basic_set_try_get(graph->inter_hmap, map);
	if (m.valid < 0 || m.valid) {
		isl_map_free(map);
		return m.value;
	}

	key = isl_map_copy(map);
	map = compress(map, edge->src, edge->dst);
	set = isl_map_wrap(isl_map_remove_divs(map));
	coef = isl_set_coefficients(set);
	graph->inter_hmap = isl_map_to_basic_set_set(graph->inter_hmap, key,
					isl_basic_set_copy(coef));

	return coef;
}

/* Return the position of the coefficients of the variables in
 * the coefficients constraints "coef".
 *
 * The space of "coef" is of the form
 *
 *	{ coefficients[[cst, params] -> S] }
 *
 * Return the position of S.
 */
static isl_size coef_var_offset(__isl_keep isl_basic_set *coef)
{
	isl_size offset;
	isl_space *space;

	space = isl_space_unwrap(isl_basic_set_get_space(coef));
	offset = isl_space_dim(space, isl_dim_in);
	isl_space_free(space);

	return offset;
}

/* Return the offset of the coefficient of the constant term of "node"
 * within the (I)LP.
 *
 * Within each node, the coefficients have the following order:
 *	- positive and negative parts of c_i_x
 *	- c_i_n (if parametric)
 *	- c_i_0
 */
static int node_cst_coef_offset(struct isl_sched_node *node)
{
	return node->start + 2 * node->nvar + node->nparam;
}

/* Return the offset of the coefficients of the parameters of "node"
 * within the (I)LP.
 *
 * Within each node, the coefficients have the following order:
 *	- positive and negative parts of c_i_x
 *	- c_i_n (if parametric)
 *	- c_i_0
 */
static int node_par_coef_offset(struct isl_sched_node *node)
{
	return node->start + 2 * node->nvar;
}

/* Return the offset of the coefficients of the variables of "node"
 * within the (I)LP.
 *
 * Within each node, the coefficients have the following order:
 *	- positive and negative parts of c_i_x
 *	- c_i_n (if parametric)
 *	- c_i_0
 */
static int node_var_coef_offset(struct isl_sched_node *node)
{
	return node->start;
}

/* Return the position of the pair of variables encoding
 * coefficient "i" of "node".
 *
 * The order of these variable pairs is the opposite of
 * that of the coefficients, with 2 variables per coefficient.
 */
static int node_var_coef_pos(struct isl_sched_node *node, int i)
{
	return node_var_coef_offset(node) + 2 * (node->nvar - 1 - i);
}

/* Construct an isl_dim_map for mapping constraints on coefficients
 * for "node" to the corresponding positions in graph->lp.
 * "offset" is the offset of the coefficients for the variables
 * in the input constraints.
 * "s" is the sign of the mapping.
 *
 * The input constraints are given in terms of the coefficients
 * (c_0, c_x) or (c_0, c_n, c_x).
 * The mapping produced by this function essentially plugs in
 * (0, c_i_x^+ - c_i_x^-) if s = 1 and
 * (0, -c_i_x^+ + c_i_x^-) if s = -1 or
 * (0, 0, c_i_x^+ - c_i_x^-) if s = 1 and
 * (0, 0, -c_i_x^+ + c_i_x^-) if s = -1.
 * In graph->lp, the c_i_x^- appear before their c_i_x^+ counterpart.
 * Furthermore, the order of these pairs is the opposite of that
 * of the corresponding coefficients.
 *
 * The caller can extend the mapping to also map the other coefficients
 * (and therefore not plug in 0).
 */
static __isl_give isl_dim_map *intra_dim_map(isl_ctx *ctx,
	struct isl_sched_graph *graph, struct isl_sched_node *node,
	int offset, int s)
{
	int pos;
	isl_size total;
	isl_dim_map *dim_map;

	total = isl_basic_set_dim(graph->lp, isl_dim_all);
	if (!node || total < 0)
		return NULL;

	pos = node_var_coef_pos(node, 0);
	dim_map = isl_dim_map_alloc(ctx, total);
	isl_dim_map_range(dim_map, pos, -2, offset, 1, node->nvar, -s);
	isl_dim_map_range(dim_map, pos + 1, -2, offset, 1, node->nvar, s);

	return dim_map;
}

/* Construct an isl_dim_map for mapping constraints on coefficients
 * for "src" (node i) and "dst" (node j) to the corresponding positions
 * in graph->lp.
 * "offset" is the offset of the coefficients for the variables of "src"
 * in the input constraints.
 * "s" is the sign of the mapping.
 *
 * The input constraints are given in terms of the coefficients
 * (c_0, c_n, c_x, c_y).
 * The mapping produced by this function essentially plugs in
 * (c_j_0 - c_i_0, c_j_n - c_i_n,
 *  -(c_i_x^+ - c_i_x^-), c_j_x^+ - c_j_x^-) if s = 1 and
 * (-c_j_0 + c_i_0, -c_j_n + c_i_n,
 *  c_i_x^+ - c_i_x^-, -(c_j_x^+ - c_j_x^-)) if s = -1.
 * In graph->lp, the c_*^- appear before their c_*^+ counterpart.
 * Furthermore, the order of these pairs is the opposite of that
 * of the corresponding coefficients.
 *
 * The caller can further extend the mapping.
 */
static __isl_give isl_dim_map *inter_dim_map(isl_ctx *ctx,
	struct isl_sched_graph *graph, struct isl_sched_node *src,
	struct isl_sched_node *dst, int offset, int s)
{
	int pos;
	isl_size total;
	isl_dim_map *dim_map;

	total = isl_basic_set_dim(graph->lp, isl_dim_all);
	if (!src || !dst || total < 0)
		return NULL;

	dim_map = isl_dim_map_alloc(ctx, total);

	pos = node_cst_coef_offset(dst);
	isl_dim_map_range(dim_map, pos, 0, 0, 0, 1, s);
	pos = node_par_coef_offset(dst);
	isl_dim_map_range(dim_map, pos, 1, 1, 1, dst->nparam, s);
	pos = node_var_coef_pos(dst, 0);
	isl_dim_map_range(dim_map, pos, -2, offset + src->nvar, 1,
			  dst->nvar, -s);
	isl_dim_map_range(dim_map, pos + 1, -2, offset + src->nvar, 1,
			  dst->nvar, s);

	pos = node_cst_coef_offset(src);
	isl_dim_map_range(dim_map, pos, 0, 0, 0, 1, -s);
	pos = node_par_coef_offset(src);
	isl_dim_map_range(dim_map, pos, 1, 1, 1, src->nparam, -s);
	pos = node_var_coef_pos(src, 0);
	isl_dim_map_range(dim_map, pos, -2, offset, 1, src->nvar, s);
	isl_dim_map_range(dim_map, pos + 1, -2, offset, 1, src->nvar, -s);

	return dim_map;
}

/* Add the constraints from "src" to "dst" using "dim_map",
 * after making sure there is enough room in "dst" for the extra constraints.
 */
static __isl_give isl_basic_set *add_constraints_dim_map(
	__isl_take isl_basic_set *dst, __isl_take isl_basic_set *src,
	__isl_take isl_dim_map *dim_map)
{
	isl_size n_eq, n_ineq;

	n_eq = isl_basic_set_n_equality(src);
	n_ineq = isl_basic_set_n_inequality(src);
	if (n_eq < 0 || n_ineq < 0)
		dst = isl_basic_set_free(dst);
	dst = isl_basic_set_extend_constraints(dst, n_eq, n_ineq);
	dst = isl_basic_set_add_constraints_dim_map(dst, src, dim_map);
	return dst;
}

/* Add constraints to graph->lp that force validity for the given
 * dependence from a node i to itself.
 * That is, add constraints that enforce
 *
 *	(c_i_0 + c_i_n n + c_i_x y) - (c_i_0 + c_i_n n + c_i_x x)
 *	= c_i_x (y - x) >= 0
 *
 * for each (x,y) in R.
 * We obtain general constraints on coefficients (c_0, c_x)
 * of valid constraints for (y - x) and then plug in (0, c_i_x^+ - c_i_x^-),
 * where c_i_x = c_i_x^+ - c_i_x^-, with c_i_x^+ and c_i_x^- non-negative.
 * In graph->lp, the c_i_x^- appear before their c_i_x^+ counterpart.
 * Note that the result of intra_coefficients may also contain
 * parameter coefficients c_n, in which case 0 is plugged in for them as well.
 */
static isl_stat add_intra_validity_constraints(struct isl_sched_graph *graph,
	struct isl_sched_edge *edge)
{
	isl_size offset;
	isl_map *map = isl_map_copy(edge->map);
	isl_ctx *ctx = isl_map_get_ctx(map);
	isl_dim_map *dim_map;
	isl_basic_set *coef;
	struct isl_sched_node *node = edge->src;

	coef = intra_coefficients(graph, node, map, 0);

	offset = coef_var_offset(coef);
	if (offset < 0)
		coef = isl_basic_set_free(coef);
	if (!coef)
		return isl_stat_error;

	dim_map = intra_dim_map(ctx, graph, node, offset, 1);
	graph->lp = add_constraints_dim_map(graph->lp, coef, dim_map);

	return isl_stat_ok;
}

/* Add constraints to graph->lp that force validity for the given
 * dependence from node i to node j.
 * That is, add constraints that enforce
 *
 *	(c_j_0 + c_j_n n + c_j_x y) - (c_i_0 + c_i_n n + c_i_x x) >= 0
 *
 * for each (x,y) in R.
 * We obtain general constraints on coefficients (c_0, c_n, c_x, c_y)
 * of valid constraints for R and then plug in
 * (c_j_0 - c_i_0, c_j_n - c_i_n, -(c_i_x^+ - c_i_x^-), c_j_x^+ - c_j_x^-),
 * where c_* = c_*^+ - c_*^-, with c_*^+ and c_*^- non-negative.
 * In graph->lp, the c_*^- appear before their c_*^+ counterpart.
 */
static isl_stat add_inter_validity_constraints(struct isl_sched_graph *graph,
	struct isl_sched_edge *edge)
{
	isl_size offset;
	isl_map *map;
	isl_ctx *ctx;
	isl_dim_map *dim_map;
	isl_basic_set *coef;
	struct isl_sched_node *src = edge->src;
	struct isl_sched_node *dst = edge->dst;

	if (!graph->lp)
		return isl_stat_error;

	map = isl_map_copy(edge->map);
	ctx = isl_map_get_ctx(map);
	coef = inter_coefficients(graph, edge, map);

	offset = coef_var_offset(coef);
	if (offset < 0)
		coef = isl_basic_set_free(coef);
	if (!coef)
		return isl_stat_error;

	dim_map = inter_dim_map(ctx, graph, src, dst, offset, 1);

	edge->start = graph->lp->n_ineq;
	graph->lp = add_constraints_dim_map(graph->lp, coef, dim_map);
	if (!graph->lp)
		return isl_stat_error;
	edge->end = graph->lp->n_ineq;

	return isl_stat_ok;
}

/* Add constraints to graph->lp that bound the dependence distance for the given
 * dependence from a node i to itself.
 * If s = 1, we add the constraint
 *
 *	c_i_x (y - x) <= m_0 + m_n n
 *
 * or
 *
 *	-c_i_x (y - x) + m_0 + m_n n >= 0
 *
 * for each (x,y) in R.
 * If s = -1, we add the constraint
 *
 *	-c_i_x (y - x) <= m_0 + m_n n
 *
 * or
 *
 *	c_i_x (y - x) + m_0 + m_n n >= 0
 *
 * for each (x,y) in R.
 * We obtain general constraints on coefficients (c_0, c_n, c_x)
 * of valid constraints for (y - x) and then plug in (m_0, m_n, -s * c_i_x),
 * with each coefficient (except m_0) represented as a pair of non-negative
 * coefficients.
 *
 *
 * If "local" is set, then we add constraints
 *
 *	c_i_x (y - x) <= 0
 *
 * or
 *
 *	-c_i_x (y - x) <= 0
 *
 * instead, forcing the dependence distance to be (less than or) equal to 0.
 * That is, we plug in (0, 0, -s * c_i_x),
 * intra_coefficients is not required to have c_n in its result when
 * "local" is set.  If they are missing, then (0, -s * c_i_x) is plugged in.
 * Note that dependences marked local are treated as validity constraints
 * by add_all_validity_constraints and therefore also have
 * their distances bounded by 0 from below.
 */
static isl_stat add_intra_proximity_constraints(struct isl_sched_graph *graph,
	struct isl_sched_edge *edge, int s, int local)
{
	isl_size offset;
	isl_size nparam;
	isl_map *map = isl_map_copy(edge->map);
	isl_ctx *ctx = isl_map_get_ctx(map);
	isl_dim_map *dim_map;
	isl_basic_set *coef;
	struct isl_sched_node *node = edge->src;

	coef = intra_coefficients(graph, node, map, !local);
	nparam = isl_space_dim(node->space, isl_dim_param);

	offset = coef_var_offset(coef);
	if (nparam < 0 || offset < 0)
		coef = isl_basic_set_free(coef);
	if (!coef)
		return isl_stat_error;

	dim_map = intra_dim_map(ctx, graph, node, offset, -s);

	if (!local) {
		isl_dim_map_range(dim_map, 1, 0, 0, 0, 1, 1);
		isl_dim_map_range(dim_map, 4, 2, 1, 1, nparam, -1);
		isl_dim_map_range(dim_map, 5, 2, 1, 1, nparam, 1);
	}
	graph->lp = add_constraints_dim_map(graph->lp, coef, dim_map);

	return isl_stat_ok;
}

/* Add constraints to graph->lp that bound the dependence distance for the given
 * dependence from node i to node j.
 * If s = 1, we add the constraint
 *
 *	(c_j_0 + c_j_n n + c_j_x y) - (c_i_0 + c_i_n n + c_i_x x)
 *		<= m_0 + m_n n
 *
 * or
 *
 *	-(c_j_0 + c_j_n n + c_j_x y) + (c_i_0 + c_i_n n + c_i_x x) +
 *		m_0 + m_n n >= 0
 *
 * for each (x,y) in R.
 * If s = -1, we add the constraint
 *
 *	-((c_j_0 + c_j_n n + c_j_x y) - (c_i_0 + c_i_n n + c_i_x x))
 *		<= m_0 + m_n n
 *
 * or
 *
 *	(c_j_0 + c_j_n n + c_j_x y) - (c_i_0 + c_i_n n + c_i_x x) +
 *		m_0 + m_n n >= 0
 *
 * for each (x,y) in R.
 * We obtain general constraints on coefficients (c_0, c_n, c_x, c_y)
 * of valid constraints for R and then plug in
 * (m_0 - s*c_j_0 + s*c_i_0, m_n - s*c_j_n + s*c_i_n,
 *  s*c_i_x, -s*c_j_x)
 * with each coefficient (except m_0, c_*_0 and c_*_n)
 * represented as a pair of non-negative coefficients.
 *
 *
 * If "local" is set (and s = 1), then we add constraints
 *
 *	(c_j_0 + c_j_n n + c_j_x y) - (c_i_0 + c_i_n n + c_i_x x) <= 0
 *
 * or
 *
 *	-((c_j_0 + c_j_n n + c_j_x y) + (c_i_0 + c_i_n n + c_i_x x)) >= 0
 *
 * instead, forcing the dependence distance to be (less than or) equal to 0.
 * That is, we plug in
 * (-s*c_j_0 + s*c_i_0, -s*c_j_n + s*c_i_n, s*c_i_x, -s*c_j_x).
 * Note that dependences marked local are treated as validity constraints
 * by add_all_validity_constraints and therefore also have
 * their distances bounded by 0 from below.
 */
static isl_stat add_inter_proximity_constraints(struct isl_sched_graph *graph,
	struct isl_sched_edge *edge, int s, int local)
{
	isl_size offset;
	isl_size nparam;
	isl_map *map = isl_map_copy(edge->map);
	isl_ctx *ctx = isl_map_get_ctx(map);
	isl_dim_map *dim_map;
	isl_basic_set *coef;
	struct isl_sched_node *src = edge->src;
	struct isl_sched_node *dst = edge->dst;

	coef = inter_coefficients(graph, edge, map);
	nparam = isl_space_dim(src->space, isl_dim_param);

	offset = coef_var_offset(coef);
	if (nparam < 0 || offset < 0)
		coef = isl_basic_set_free(coef);
	if (!coef)
		return isl_stat_error;

	dim_map = inter_dim_map(ctx, graph, src, dst, offset, -s);

	if (!local) {
		isl_dim_map_range(dim_map, 1, 0, 0, 0, 1, 1);
		isl_dim_map_range(dim_map, 4, 2, 1, 1, nparam, -1);
		isl_dim_map_range(dim_map, 5, 2, 1, 1, nparam, 1);
	}

	graph->lp = add_constraints_dim_map(graph->lp, coef, dim_map);

	return isl_stat_ok;
}

/* Should the distance over "edge" be forced to zero?
 * That is, is it marked as a local edge?
 * If "use_coincidence" is set, then coincidence edges are treated
 * as local edges.
 */
static int force_zero(struct isl_sched_edge *edge, int use_coincidence)
{
	return is_local(edge) || (use_coincidence && is_coincidence(edge));
}

/* Add all validity constraints to graph->lp.
 *
 * An edge that is forced to be local needs to have its dependence
 * distances equal to zero.  We take care of bounding them by 0 from below
 * here.  add_all_proximity_constraints takes care of bounding them by 0
 * from above.
 *
 * If "use_coincidence" is set, then we treat coincidence edges as local edges.
 * Otherwise, we ignore them.
 */
static int add_all_validity_constraints(struct isl_sched_graph *graph,
	int use_coincidence)
{
	int i;

	for (i = 0; i < graph->n_edge; ++i) {
		struct isl_sched_edge *edge = &graph->edge[i];
		int zero;

		zero = force_zero(edge, use_coincidence);
		if (!is_validity(edge) && !zero)
			continue;
		if (edge->src != edge->dst)
			continue;
		if (add_intra_validity_constraints(graph, edge) < 0)
			return -1;
	}

	for (i = 0; i < graph->n_edge; ++i) {
		struct isl_sched_edge *edge = &graph->edge[i];
		int zero;

		zero = force_zero(edge, use_coincidence);
		if (!is_validity(edge) && !zero)
			continue;
		if (edge->src == edge->dst)
			continue;
		if (add_inter_validity_constraints(graph, edge) < 0)
			return -1;
	}

	return 0;
}

/* Add constraints to graph->lp that bound the dependence distance
 * for all dependence relations.
 * If a given proximity dependence is identical to a validity
 * dependence, then the dependence distance is already bounded
 * from below (by zero), so we only need to bound the distance
 * from above.  (This includes the case of "local" dependences
 * which are treated as validity dependence by add_all_validity_constraints.)
 * Otherwise, we need to bound the distance both from above and from below.
 *
 * If "use_coincidence" is set, then we treat coincidence edges as local edges.
 * Otherwise, we ignore them.
 */
static int add_all_proximity_constraints(struct isl_sched_graph *graph,
	int use_coincidence)
{
	int i;

	for (i = 0; i < graph->n_edge; ++i) {
		struct isl_sched_edge *edge = &graph->edge[i];
		int zero;

		zero = force_zero(edge, use_coincidence);
		if (!isl_sched_edge_is_proximity(edge) && !zero)
			continue;
		if (edge->src == edge->dst &&
		    add_intra_proximity_constraints(graph, edge, 1, zero) < 0)
			return -1;
		if (edge->src != edge->dst &&
		    add_inter_proximity_constraints(graph, edge, 1, zero) < 0)
			return -1;
		if (is_validity(edge) || zero)
			continue;
		if (edge->src == edge->dst &&
		    add_intra_proximity_constraints(graph, edge, -1, 0) < 0)
			return -1;
		if (edge->src != edge->dst &&
		    add_inter_proximity_constraints(graph, edge, -1, 0) < 0)
			return -1;
	}

	return 0;
}

/* Normalize the rows of "indep" such that all rows are lexicographically
 * positive and such that each row contains as many final zeros as possible,
 * given the choice for the previous rows.
 * Do this by performing elementary row operations.
 */
static __isl_give isl_mat *normalize_independent(__isl_take isl_mat *indep)
{
	indep = isl_mat_reverse_gauss(indep);
	indep = isl_mat_lexnonneg_rows(indep);
	return indep;
}

/* Extract the linear part of the current schedule for node "node".
 */
static __isl_give isl_mat *extract_linear_schedule(struct isl_sched_node *node)
{
	isl_size n_row = isl_mat_rows(node->sched);

	if (n_row < 0)
		return NULL;
	return isl_mat_sub_alloc(node->sched, 0, n_row,
			      1 + node->nparam, node->nvar);
}

/* Compute a basis for the rows in the linear part of the schedule
 * and extend this basis to a full basis.  The remaining rows
 * can then be used to force linear independence from the rows
 * in the schedule.
 *
 * In particular, given the schedule rows S, we compute
 *
 *	S   = H Q
 *	S U = H
 *
 * with H the Hermite normal form of S.  That is, all but the
 * first rank columns of H are zero and so each row in S is
 * a linear combination of the first rank rows of Q.
 * The matrix Q can be used as a variable transformation
 * that isolates the directions of S in the first rank rows.
 * Transposing S U = H yields
 *
 *	U^T S^T = H^T
 *
 * with all but the first rank rows of H^T zero.
 * The last rows of U^T are therefore linear combinations
 * of schedule coefficients that are all zero on schedule
 * coefficients that are linearly dependent on the rows of S.
 * At least one of these combinations is non-zero on
 * linearly independent schedule coefficients.
 * The rows are normalized to involve as few of the last
 * coefficients as possible and to have a positive initial value.
 */
isl_stat isl_sched_node_update_vmap(struct isl_sched_node *node)
{
	isl_mat *H, *U, *Q;

	H = extract_linear_schedule(node);

	H = isl_mat_left_hermite(H, 0, &U, &Q);
	isl_mat_free(node->indep);
	isl_mat_free(node->vmap);
	node->vmap = Q;
	node->indep = isl_mat_transpose(U);
	node->rank = isl_mat_initial_non_zero_cols(H);
	node->indep = isl_mat_drop_rows(node->indep, 0, node->rank);
	node->indep = normalize_independent(node->indep);
	isl_mat_free(H);

	if (!node->indep || !node->vmap || node->rank < 0)
		return isl_stat_error;
	return isl_stat_ok;
}

/* Is "edge" marked as a validity or a conditional validity edge?
 */
static int is_any_validity(struct isl_sched_edge *edge)
{
	return is_validity(edge) ||
		isl_sched_edge_is_conditional_validity(edge);
}

/* How many times should we count the constraints in "edge"?
 *
 * We count as follows
 * validity		-> 1 (>= 0)
 * validity+proximity	-> 2 (>= 0 and upper bound)
 * proximity		-> 2 (lower and upper bound)
 * local(+any)		-> 2 (>= 0 and <= 0)
 *
 * If an edge is only marked conditional_validity then it counts
 * as zero since it is only checked afterwards.
 *
 * If "use_coincidence" is set, then we treat coincidence edges as local edges.
 * Otherwise, we ignore them.
 */
static int edge_multiplicity(struct isl_sched_edge *edge, int use_coincidence)
{
	if (isl_sched_edge_is_proximity(edge) ||
	    force_zero(edge, use_coincidence))
		return 2;
	if (is_validity(edge))
		return 1;
	return 0;
}

/* How many times should the constraints in "edge" be counted
 * as a parametric intra-node constraint?
 *
 * Only proximity edges that are not forced zero need
 * coefficient constraints that include coefficients for parameters.
 * If the edge is also a validity edge, then only
 * an upper bound is introduced.  Otherwise, both lower and upper bounds
 * are introduced.
 */
static int parametric_intra_edge_multiplicity(struct isl_sched_edge *edge,
	int use_coincidence)
{
	if (edge->src != edge->dst)
		return 0;
	if (!isl_sched_edge_is_proximity(edge))
		return 0;
	if (force_zero(edge, use_coincidence))
		return 0;
	if (is_validity(edge))
		return 1;
	else
		return 2;
}

/* Add "f" times the number of equality and inequality constraints of "bset"
 * to "n_eq" and "n_ineq" and free "bset".
 */
static isl_stat update_count(__isl_take isl_basic_set *bset,
	int f, int *n_eq, int *n_ineq)
{
	isl_size eq, ineq;

	eq = isl_basic_set_n_equality(bset);
	ineq = isl_basic_set_n_inequality(bset);
	isl_basic_set_free(bset);

	if (eq < 0 || ineq < 0)
		return isl_stat_error;

	*n_eq += eq;
	*n_ineq += ineq;

	return isl_stat_ok;
}

/* Count the number of equality and inequality constraints
 * that will be added for the given map.
 *
 * The edges that require parameter coefficients are counted separately.
 *
 * "use_coincidence" is set if we should take into account coincidence edges.
 */
static isl_stat count_map_constraints(struct isl_sched_graph *graph,
	struct isl_sched_edge *edge, __isl_take isl_map *map,
	int *n_eq, int *n_ineq, int use_coincidence)
{
	isl_map *copy;
	isl_basic_set *coef;
	int f = edge_multiplicity(edge, use_coincidence);
	int fp = parametric_intra_edge_multiplicity(edge, use_coincidence);

	if (f == 0) {
		isl_map_free(map);
		return isl_stat_ok;
	}

	if (edge->src != edge->dst) {
		coef = inter_coefficients(graph, edge, map);
		return update_count(coef, f, n_eq, n_ineq);
	}

	if (fp > 0) {
		copy = isl_map_copy(map);
		coef = intra_coefficients(graph, edge->src, copy, 1);
		if (update_count(coef, fp, n_eq, n_ineq) < 0)
			goto error;
	}

	if (f > fp) {
		copy = isl_map_copy(map);
		coef = intra_coefficients(graph, edge->src, copy, 0);
		if (update_count(coef, f - fp, n_eq, n_ineq) < 0)
			goto error;
	}

	isl_map_free(map);
	return isl_stat_ok;
error:
	isl_map_free(map);
	return isl_stat_error;
}

/* Count the number of equality and inequality constraints
 * that will be added to the main lp problem.
 * We count as follows
 * validity		-> 1 (>= 0)
 * validity+proximity	-> 2 (>= 0 and upper bound)
 * proximity		-> 2 (lower and upper bound)
 * local(+any)		-> 2 (>= 0 and <= 0)
 *
 * If "use_coincidence" is set, then we treat coincidence edges as local edges.
 * Otherwise, we ignore them.
 */
static int count_constraints(struct isl_sched_graph *graph,
	int *n_eq, int *n_ineq, int use_coincidence)
{
	int i;

	*n_eq = *n_ineq = 0;
	for (i = 0; i < graph->n_edge; ++i) {
		struct isl_sched_edge *edge = &graph->edge[i];
		isl_map *map = isl_map_copy(edge->map);

		if (count_map_constraints(graph, edge, map, n_eq, n_ineq,
					    use_coincidence) < 0)
			return -1;
	}

	return 0;
}

/* Count the number of constraints that will be added by
 * add_bound_constant_constraints to bound the values of the constant terms
 * and increment *n_eq and *n_ineq accordingly.
 *
 * In practice, add_bound_constant_constraints only adds inequalities.
 */
static isl_stat count_bound_constant_constraints(isl_ctx *ctx,
	struct isl_sched_graph *graph, int *n_eq, int *n_ineq)
{
	if (isl_options_get_schedule_max_constant_term(ctx) == -1)
		return isl_stat_ok;

	*n_ineq += graph->n;

	return isl_stat_ok;
}

/* Add constraints to bound the values of the constant terms in the schedule,
 * if requested by the user.
 *
 * The maximal value of the constant terms is defined by the option
 * "schedule_max_constant_term".
 */
static isl_stat add_bound_constant_constraints(isl_ctx *ctx,
	struct isl_sched_graph *graph)
{
	int i, k;
	int max;
	isl_size total;

	max = isl_options_get_schedule_max_constant_term(ctx);
	if (max == -1)
		return isl_stat_ok;

	total = isl_basic_set_dim(graph->lp, isl_dim_set);
	if (total < 0)
		return isl_stat_error;

	for (i = 0; i < graph->n; ++i) {
		struct isl_sched_node *node = &graph->node[i];
		int pos;

		k = isl_basic_set_alloc_inequality(graph->lp);
		if (k < 0)
			return isl_stat_error;
		isl_seq_clr(graph->lp->ineq[k], 1 + total);
		pos = node_cst_coef_offset(node);
		isl_int_set_si(graph->lp->ineq[k][1 + pos], -1);
		isl_int_set_si(graph->lp->ineq[k][0], max);
	}

	return isl_stat_ok;
}

/* Count the number of constraints that will be added by
 * add_bound_coefficient_constraints and increment *n_eq and *n_ineq
 * accordingly.
 *
 * In practice, add_bound_coefficient_constraints only adds inequalities.
 */
static int count_bound_coefficient_constraints(isl_ctx *ctx,
	struct isl_sched_graph *graph, int *n_eq, int *n_ineq)
{
	int i;

	if (isl_options_get_schedule_max_coefficient(ctx) == -1 &&
	    !isl_options_get_schedule_treat_coalescing(ctx))
		return 0;

	for (i = 0; i < graph->n; ++i)
		*n_ineq += graph->node[i].nparam + 2 * graph->node[i].nvar;

	return 0;
}

/* Add constraints to graph->lp that bound the values of
 * the parameter schedule coefficients of "node" to "max" and
 * the variable schedule coefficients to the corresponding entry
 * in node->max.
 * In either case, a negative value means that no bound needs to be imposed.
 *
 * For parameter coefficients, this amounts to adding a constraint
 *
 *	c_n <= max
 *
 * i.e.,
 *
 *	-c_n + max >= 0
 *
 * The variables coefficients are, however, not represented directly.
 * Instead, the variable coefficients c_x are written as differences
 * c_x = c_x^+ - c_x^-.
 * That is,
 *
 *	-max_i <= c_x_i <= max_i
 *
 * is encoded as
 *
 *	-max_i <= c_x_i^+ - c_x_i^- <= max_i
 *
 * or
 *
 *	-(c_x_i^+ - c_x_i^-) + max_i >= 0
 *	c_x_i^+ - c_x_i^- + max_i >= 0
 */
static isl_stat node_add_coefficient_constraints(isl_ctx *ctx,
	struct isl_sched_graph *graph, struct isl_sched_node *node, int max)
{
	int i, j, k;
	isl_size total;
	isl_vec *ineq;

	total = isl_basic_set_dim(graph->lp, isl_dim_set);
	if (total < 0)
		return isl_stat_error;

	for (j = 0; j < node->nparam; ++j) {
		int dim;

		if (max < 0)
			continue;

		k = isl_basic_set_alloc_inequality(graph->lp);
		if (k < 0)
			return isl_stat_error;
		dim = 1 + node_par_coef_offset(node) + j;
		isl_seq_clr(graph->lp->ineq[k], 1 + total);
		isl_int_set_si(graph->lp->ineq[k][dim], -1);
		isl_int_set_si(graph->lp->ineq[k][0], max);
	}

	ineq = isl_vec_alloc(ctx, 1 + total);
	ineq = isl_vec_clr(ineq);
	if (!ineq)
		return isl_stat_error;
	for (i = 0; i < node->nvar; ++i) {
		int pos = 1 + node_var_coef_pos(node, i);

		if (isl_int_is_neg(node->max->el[i]))
			continue;

		isl_int_set_si(ineq->el[pos], 1);
		isl_int_set_si(ineq->el[pos + 1], -1);
		isl_int_set(ineq->el[0], node->max->el[i]);

		k = isl_basic_set_alloc_inequality(graph->lp);
		if (k < 0)
			goto error;
		isl_seq_cpy(graph->lp->ineq[k], ineq->el, 1 + total);

		isl_seq_neg(ineq->el + pos, ineq->el + pos, 2);
		k = isl_basic_set_alloc_inequality(graph->lp);
		if (k < 0)
			goto error;
		isl_seq_cpy(graph->lp->ineq[k], ineq->el, 1 + total);

		isl_seq_clr(ineq->el + pos, 2);
	}
	isl_vec_free(ineq);

	return isl_stat_ok;
error:
	isl_vec_free(ineq);
	return isl_stat_error;
}

/* Add constraints that bound the values of the variable and parameter
 * coefficients of the schedule.
 *
 * The maximal value of the coefficients is defined by the option
 * 'schedule_max_coefficient' and the entries in node->max.
 * These latter entries are only set if either the schedule_max_coefficient
 * option or the schedule_treat_coalescing option is set.
 */
static isl_stat add_bound_coefficient_constraints(isl_ctx *ctx,
	struct isl_sched_graph *graph)
{
	int i;
	int max;

	max = isl_options_get_schedule_max_coefficient(ctx);

	if (max == -1 && !isl_options_get_schedule_treat_coalescing(ctx))
		return isl_stat_ok;

	for (i = 0; i < graph->n; ++i) {
		struct isl_sched_node *node = &graph->node[i];

		if (node_add_coefficient_constraints(ctx, graph, node, max) < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Add a constraint to graph->lp that equates the value at position
 * "sum_pos" to the sum of the "n" values starting at "first".
 */
static isl_stat add_sum_constraint(struct isl_sched_graph *graph,
	int sum_pos, int first, int n)
{
	int i, k;
	isl_size total;

	total = isl_basic_set_dim(graph->lp, isl_dim_set);
	if (total < 0)
		return isl_stat_error;

	k = isl_basic_set_alloc_equality(graph->lp);
	if (k < 0)
		return isl_stat_error;
	isl_seq_clr(graph->lp->eq[k], 1 + total);
	isl_int_set_si(graph->lp->eq[k][1 + sum_pos], -1);
	for (i = 0; i < n; ++i)
		isl_int_set_si(graph->lp->eq[k][1 + first + i], 1);

	return isl_stat_ok;
}

/* Add a constraint to graph->lp that equates the value at position
 * "sum_pos" to the sum of the parameter coefficients of all nodes.
 */
static isl_stat add_param_sum_constraint(struct isl_sched_graph *graph,
	int sum_pos)
{
	int i, j, k;
	isl_size total;

	total = isl_basic_set_dim(graph->lp, isl_dim_set);
	if (total < 0)
		return isl_stat_error;

	k = isl_basic_set_alloc_equality(graph->lp);
	if (k < 0)
		return isl_stat_error;
	isl_seq_clr(graph->lp->eq[k], 1 + total);
	isl_int_set_si(graph->lp->eq[k][1 + sum_pos], -1);
	for (i = 0; i < graph->n; ++i) {
		int pos = 1 + node_par_coef_offset(&graph->node[i]);

		for (j = 0; j < graph->node[i].nparam; ++j)
			isl_int_set_si(graph->lp->eq[k][pos + j], 1);
	}

	return isl_stat_ok;
}

/* Add a constraint to graph->lp that equates the value at position
 * "sum_pos" to the sum of the variable coefficients of all nodes.
 */
static isl_stat add_var_sum_constraint(struct isl_sched_graph *graph,
	int sum_pos)
{
	int i, j, k;
	isl_size total;

	total = isl_basic_set_dim(graph->lp, isl_dim_set);
	if (total < 0)
		return isl_stat_error;

	k = isl_basic_set_alloc_equality(graph->lp);
	if (k < 0)
		return isl_stat_error;
	isl_seq_clr(graph->lp->eq[k], 1 + total);
	isl_int_set_si(graph->lp->eq[k][1 + sum_pos], -1);
	for (i = 0; i < graph->n; ++i) {
		struct isl_sched_node *node = &graph->node[i];
		int pos = 1 + node_var_coef_offset(node);

		for (j = 0; j < 2 * node->nvar; ++j)
			isl_int_set_si(graph->lp->eq[k][pos + j], 1);
	}

	return isl_stat_ok;
}

/* Construct an ILP problem for finding schedule coefficients
 * that result in non-negative, but small dependence distances
 * over all dependences.
 * In particular, the dependence distances over proximity edges
 * are bounded by m_0 + m_n n and we compute schedule coefficients
 * with small values (preferably zero) of m_n and m_0.
 *
 * All variables of the ILP are non-negative.  The actual coefficients
 * may be negative, so each coefficient is represented as the difference
 * of two non-negative variables.  The negative part always appears
 * immediately before the positive part.
 * Other than that, the variables have the following order
 *
 *	- sum of positive and negative parts of m_n coefficients
 *	- m_0
 *	- sum of all c_n coefficients
 *		(unconstrained when computing non-parametric schedules)
 *	- sum of positive and negative parts of all c_x coefficients
 *	- positive and negative parts of m_n coefficients
 *	- for each node
 *		- positive and negative parts of c_i_x, in opposite order
 *		- c_i_n (if parametric)
 *		- c_i_0
 *
 * The constraints are those from the edges plus two or three equalities
 * to express the sums.
 *
 * If "use_coincidence" is set, then we treat coincidence edges as local edges.
 * Otherwise, we ignore them.
 */
static isl_stat setup_lp(isl_ctx *ctx, struct isl_sched_graph *graph,
	int use_coincidence)
{
	int i;
	isl_size nparam;
	unsigned total;
	isl_space *space;
	int parametric;
	int param_pos;
	int n_eq, n_ineq;

	parametric = ctx->opt->schedule_parametric;
	nparam = isl_space_dim(graph->node[0].space, isl_dim_param);
	if (nparam < 0)
		return isl_stat_error;
	param_pos = 4;
	total = param_pos + 2 * nparam;
	for (i = 0; i < graph->n; ++i) {
		struct isl_sched_node *node = &graph->node[graph->sorted[i]];
		if (isl_sched_node_update_vmap(node) < 0)
			return isl_stat_error;
		node->start = total;
		total += 1 + node->nparam + 2 * node->nvar;
	}

	if (count_constraints(graph, &n_eq, &n_ineq, use_coincidence) < 0)
		return isl_stat_error;
	if (count_bound_constant_constraints(ctx, graph, &n_eq, &n_ineq) < 0)
		return isl_stat_error;
	if (count_bound_coefficient_constraints(ctx, graph, &n_eq, &n_ineq) < 0)
		return isl_stat_error;

	space = isl_space_set_alloc(ctx, 0, total);
	isl_basic_set_free(graph->lp);
	n_eq += 2 + parametric;

	graph->lp = isl_basic_set_alloc_space(space, 0, n_eq, n_ineq);

	if (add_sum_constraint(graph, 0, param_pos, 2 * nparam) < 0)
		return isl_stat_error;
	if (parametric && add_param_sum_constraint(graph, 2) < 0)
		return isl_stat_error;
	if (add_var_sum_constraint(graph, 3) < 0)
		return isl_stat_error;
	if (add_bound_constant_constraints(ctx, graph) < 0)
		return isl_stat_error;
	if (add_bound_coefficient_constraints(ctx, graph) < 0)
		return isl_stat_error;
	if (add_all_validity_constraints(graph, use_coincidence) < 0)
		return isl_stat_error;
	if (add_all_proximity_constraints(graph, use_coincidence) < 0)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Analyze the conflicting constraint found by
 * isl_tab_basic_set_non_trivial_lexmin.  If it corresponds to the validity
 * constraint of one of the edges between distinct nodes, living, moreover
 * in distinct SCCs, then record the source and sink SCC as this may
 * be a good place to cut between SCCs.
 */
static int check_conflict(int con, void *user)
{
	int i;
	struct isl_sched_graph *graph = user;

	if (graph->src_scc >= 0)
		return 0;

	con -= graph->lp->n_eq;

	if (con >= graph->lp->n_ineq)
		return 0;

	for (i = 0; i < graph->n_edge; ++i) {
		if (!is_validity(&graph->edge[i]))
			continue;
		if (graph->edge[i].src == graph->edge[i].dst)
			continue;
		if (graph->edge[i].src->scc == graph->edge[i].dst->scc)
			continue;
		if (graph->edge[i].start > con)
			continue;
		if (graph->edge[i].end <= con)
			continue;
		graph->src_scc = graph->edge[i].src->scc;
		graph->dst_scc = graph->edge[i].dst->scc;
	}

	return 0;
}

/* Check whether the next schedule row of the given node needs to be
 * non-trivial.  Lower-dimensional domains may have some trivial rows,
 * but as soon as the number of remaining required non-trivial rows
 * is as large as the number or remaining rows to be computed,
 * all remaining rows need to be non-trivial.
 */
static int needs_row(struct isl_sched_graph *graph, struct isl_sched_node *node)
{
	return node->nvar - node->rank >= graph->maxvar - graph->n_row;
}

/* Construct a non-triviality region with triviality directions
 * corresponding to the rows of "indep".
 * The rows of "indep" are expressed in terms of the schedule coefficients c_i,
 * while the triviality directions are expressed in terms of
 * pairs of non-negative variables c^+_i - c^-_i, with c^-_i appearing
 * before c^+_i.  Furthermore,
 * the pairs of non-negative variables representing the coefficients
 * are stored in the opposite order.
 */
static __isl_give isl_mat *construct_trivial(__isl_keep isl_mat *indep)
{
	isl_ctx *ctx;
	isl_mat *mat;
	int i, j;
	isl_size n, n_var;

	n = isl_mat_rows(indep);
	n_var = isl_mat_cols(indep);
	if (n < 0 || n_var < 0)
		return NULL;

	ctx = isl_mat_get_ctx(indep);
	mat = isl_mat_alloc(ctx, n, 2 * n_var);
	if (!mat)
		return NULL;
	for (i = 0; i < n; ++i) {
		for (j = 0; j < n_var; ++j) {
			int nj = n_var - 1 - j;
			isl_int_neg(mat->row[i][2 * nj], indep->row[i][j]);
			isl_int_set(mat->row[i][2 * nj + 1], indep->row[i][j]);
		}
	}

	return mat;
}

/* Solve the ILP problem constructed in setup_lp.
 * For each node such that all the remaining rows of its schedule
 * need to be non-trivial, we construct a non-triviality region.
 * This region imposes that the next row is independent of previous rows.
 * In particular, the non-triviality region enforces that at least
 * one of the linear combinations in the rows of node->indep is non-zero.
 */
static __isl_give isl_vec *solve_lp(isl_ctx *ctx, struct isl_sched_graph *graph)
{
	int i;
	isl_vec *sol;
	isl_basic_set *lp;

	for (i = 0; i < graph->n; ++i) {
		struct isl_sched_node *node = &graph->node[i];
		isl_mat *trivial;

		graph->region[i].pos = node_var_coef_offset(node);
		if (needs_row(graph, node))
			trivial = construct_trivial(node->indep);
		else
			trivial = isl_mat_zero(ctx, 0, 0);
		graph->region[i].trivial = trivial;
	}
	lp = isl_basic_set_copy(graph->lp);
	sol = isl_tab_basic_set_non_trivial_lexmin(lp, 2, graph->n,
				       graph->region, &check_conflict, graph);
	for (i = 0; i < graph->n; ++i)
		isl_mat_free(graph->region[i].trivial);
	return sol;
}

/* Extract the coefficients for the variables of "node" from "sol".
 *
 * Each schedule coefficient c_i_x is represented as the difference
 * between two non-negative variables c_i_x^+ - c_i_x^-.
 * The c_i_x^- appear before their c_i_x^+ counterpart.
 * Furthermore, the order of these pairs is the opposite of that
 * of the corresponding coefficients.
 *
 * Return c_i_x = c_i_x^+ - c_i_x^-
 */
static __isl_give isl_vec *extract_var_coef(struct isl_sched_node *node,
	__isl_keep isl_vec *sol)
{
	int i;
	int pos;
	isl_vec *csol;

	if (!sol)
		return NULL;
	csol = isl_vec_alloc(isl_vec_get_ctx(sol), node->nvar);
	if (!csol)
		return NULL;

	pos = 1 + node_var_coef_offset(node);
	for (i = 0; i < node->nvar; ++i)
		isl_int_sub(csol->el[node->nvar - 1 - i],
			    sol->el[pos + 2 * i + 1], sol->el[pos + 2 * i]);

	return csol;
}

/* Update the schedules of all nodes based on the given solution
 * of the LP problem.
 * The new row is added to the current band.
 * All possibly negative coefficients are encoded as a difference
 * of two non-negative variables, so we need to perform the subtraction
 * here.
 *
 * If coincident is set, then the caller guarantees that the new
 * row satisfies the coincidence constraints.
 */
static int update_schedule(struct isl_sched_graph *graph,
	__isl_take isl_vec *sol, int coincident)
{
	int i, j;
	isl_vec *csol = NULL;

	if (!sol)
		goto error;
	if (sol->size == 0)
		isl_die(sol->ctx, isl_error_internal,
			"no solution found", goto error);
	if (graph->n_total_row >= graph->max_row)
		isl_die(sol->ctx, isl_error_internal,
			"too many schedule rows", goto error);

	for (i = 0; i < graph->n; ++i) {
		struct isl_sched_node *node = &graph->node[i];
		int pos;
		isl_size row = isl_mat_rows(node->sched);

		isl_vec_free(csol);
		csol = extract_var_coef(node, sol);
		if (row < 0 || !csol)
			goto error;

		isl_map_free(node->sched_map);
		node->sched_map = NULL;
		node->sched = isl_mat_add_rows(node->sched, 1);
		if (!node->sched)
			goto error;
		pos = node_cst_coef_offset(node);
		node->sched = isl_mat_set_element(node->sched,
					row, 0, sol->el[1 + pos]);
		pos = node_par_coef_offset(node);
		for (j = 0; j < node->nparam; ++j)
			node->sched = isl_mat_set_element(node->sched,
					row, 1 + j, sol->el[1 + pos + j]);
		for (j = 0; j < node->nvar; ++j)
			node->sched = isl_mat_set_element(node->sched,
					row, 1 + node->nparam + j, csol->el[j]);
		node->coincident[graph->n_total_row] = coincident;
	}
	isl_vec_free(sol);
	isl_vec_free(csol);

	graph->n_row++;
	graph->n_total_row++;

	return 0;
error:
	isl_vec_free(sol);
	isl_vec_free(csol);
	return -1;
}

/* Convert row "row" of node->sched into an isl_aff living in "ls"
 * and return this isl_aff.
 */
static __isl_give isl_aff *extract_schedule_row(__isl_take isl_local_space *ls,
	struct isl_sched_node *node, int row)
{
	int j;
	isl_int v;
	isl_aff *aff;

	isl_int_init(v);

	aff = isl_aff_zero_on_domain(ls);
	if (isl_mat_get_element(node->sched, row, 0, &v) < 0)
		goto error;
	aff = isl_aff_set_constant(aff, v);
	for (j = 0; j < node->nparam; ++j) {
		if (isl_mat_get_element(node->sched, row, 1 + j, &v) < 0)
			goto error;
		aff = isl_aff_set_coefficient(aff, isl_dim_param, j, v);
	}
	for (j = 0; j < node->nvar; ++j) {
		if (isl_mat_get_element(node->sched, row,
					1 + node->nparam + j, &v) < 0)
			goto error;
		aff = isl_aff_set_coefficient(aff, isl_dim_in, j, v);
	}

	isl_int_clear(v);

	return aff;
error:
	isl_int_clear(v);
	isl_aff_free(aff);
	return NULL;
}

/* Convert the "n" rows starting at "first" of node->sched into a multi_aff
 * and return this multi_aff.
 *
 * The result is defined over the uncompressed node domain.
 */
__isl_give isl_multi_aff *isl_sched_node_extract_partial_schedule_multi_aff(
	struct isl_sched_node *node, int first, int n)
{
	int i;
	isl_space *space;
	isl_local_space *ls;
	isl_aff *aff;
	isl_multi_aff *ma;
	isl_size nrow;

	if (!node)
		return NULL;
	nrow = isl_mat_rows(node->sched);
	if (nrow < 0)
		return NULL;
	if (node->compressed)
		space = isl_pw_multi_aff_get_domain_space(node->decompress);
	else
		space = isl_space_copy(node->space);
	ls = isl_local_space_from_space(isl_space_copy(space));
	space = isl_space_from_domain(space);
	space = isl_space_add_dims(space, isl_dim_out, n);
	ma = isl_multi_aff_zero(space);

	for (i = first; i < first + n; ++i) {
		aff = extract_schedule_row(isl_local_space_copy(ls), node, i);
		ma = isl_multi_aff_set_aff(ma, i - first, aff);
	}

	isl_local_space_free(ls);

	if (node->compressed)
		ma = isl_multi_aff_pullback_multi_aff(ma,
					isl_multi_aff_copy(node->compress));

	return ma;
}

/* Convert node->sched into a multi_aff and return this multi_aff.
 *
 * The result is defined over the uncompressed node domain.
 */
static __isl_give isl_multi_aff *node_extract_schedule_multi_aff(
	struct isl_sched_node *node)
{
	isl_size nrow;

	nrow = isl_mat_rows(node->sched);
	if (nrow < 0)
		return NULL;
	return isl_sched_node_extract_partial_schedule_multi_aff(node, 0, nrow);
}

/* Convert node->sched into a map and return this map.
 *
 * The result is cached in node->sched_map, which needs to be released
 * whenever node->sched is updated.
 * It is defined over the uncompressed node domain.
 */
static __isl_give isl_map *node_extract_schedule(struct isl_sched_node *node)
{
	if (!node->sched_map) {
		isl_multi_aff *ma;

		ma = node_extract_schedule_multi_aff(node);
		node->sched_map = isl_map_from_multi_aff(ma);
	}

	return isl_map_copy(node->sched_map);
}

/* Construct a map that can be used to update a dependence relation
 * based on the current schedule.
 * That is, construct a map expressing that source and sink
 * are executed within the same iteration of the current schedule.
 * This map can then be intersected with the dependence relation.
 * This is not the most efficient way, but this shouldn't be a critical
 * operation.
 */
static __isl_give isl_map *specializer(struct isl_sched_node *src,
	struct isl_sched_node *dst)
{
	isl_map *src_sched, *dst_sched;

	src_sched = node_extract_schedule(src);
	dst_sched = node_extract_schedule(dst);
	return isl_map_apply_range(src_sched, isl_map_reverse(dst_sched));
}

/* Intersect the domains of the nested relations in domain and range
 * of "umap" with "map".
 */
static __isl_give isl_union_map *intersect_domains(
	__isl_take isl_union_map *umap, __isl_keep isl_map *map)
{
	isl_union_set *uset;

	umap = isl_union_map_zip(umap);
	uset = isl_union_set_from_set(isl_map_wrap(isl_map_copy(map)));
	umap = isl_union_map_intersect_domain(umap, uset);
	umap = isl_union_map_zip(umap);
	return umap;
}

/* Update the dependence relation of the given edge based
 * on the current schedule.
 * If the dependence is carried completely by the current schedule, then
 * it is removed from the edge_tables.  It is kept in the list of edges
 * as otherwise all edge_tables would have to be recomputed.
 *
 * If the edge is of a type that can appear multiple times
 * between the same pair of nodes, then it is added to
 * the edge table (again).  This prevents the situation
 * where none of these edges is referenced from the edge table
 * because the one that was referenced turned out to be empty and
 * was therefore removed from the table.
 */
static isl_stat update_edge(isl_ctx *ctx, struct isl_sched_graph *graph,
	struct isl_sched_edge *edge)
{
	int empty;
	isl_map *id;

	id = specializer(edge->src, edge->dst);
	edge->map = isl_map_intersect(edge->map, isl_map_copy(id));
	if (!edge->map)
		goto error;

	if (edge->tagged_condition) {
		edge->tagged_condition =
			intersect_domains(edge->tagged_condition, id);
		if (!edge->tagged_condition)
			goto error;
	}
	if (edge->tagged_validity) {
		edge->tagged_validity =
			intersect_domains(edge->tagged_validity, id);
		if (!edge->tagged_validity)
			goto error;
	}

	empty = isl_map_plain_is_empty(edge->map);
	if (empty < 0)
		goto error;
	if (empty) {
		if (graph_remove_edge(graph, edge) < 0)
			goto error;
	} else if (is_multi_edge_type(edge)) {
		if (graph_edge_tables_add(ctx, graph, edge) < 0)
			goto error;
	}

	isl_map_free(id);
	return isl_stat_ok;
error:
	isl_map_free(id);
	return isl_stat_error;
}

/* Does the domain of "umap" intersect "uset"?
 */
static int domain_intersects(__isl_keep isl_union_map *umap,
	__isl_keep isl_union_set *uset)
{
	int empty;

	umap = isl_union_map_copy(umap);
	umap = isl_union_map_intersect_domain(umap, isl_union_set_copy(uset));
	empty = isl_union_map_is_empty(umap);
	isl_union_map_free(umap);

	return empty < 0 ? -1 : !empty;
}

/* Does the range of "umap" intersect "uset"?
 */
static int range_intersects(__isl_keep isl_union_map *umap,
	__isl_keep isl_union_set *uset)
{
	int empty;

	umap = isl_union_map_copy(umap);
	umap = isl_union_map_intersect_range(umap, isl_union_set_copy(uset));
	empty = isl_union_map_is_empty(umap);
	isl_union_map_free(umap);

	return empty < 0 ? -1 : !empty;
}

/* Are the condition dependences of "edge" local with respect to
 * the current schedule?
 *
 * That is, are domain and range of the condition dependences mapped
 * to the same point?
 *
 * In other words, is the condition false?
 */
static int is_condition_false(struct isl_sched_edge *edge)
{
	isl_union_map *umap;
	isl_map *map, *sched, *test;
	int empty, local;

	empty = isl_union_map_is_empty(edge->tagged_condition);
	if (empty < 0 || empty)
		return empty;

	umap = isl_union_map_copy(edge->tagged_condition);
	umap = isl_union_map_zip(umap);
	umap = isl_union_set_unwrap(isl_union_map_domain(umap));
	map = isl_map_from_union_map(umap);

	sched = node_extract_schedule(edge->src);
	map = isl_map_apply_domain(map, sched);
	sched = node_extract_schedule(edge->dst);
	map = isl_map_apply_range(map, sched);

	test = isl_map_identity(isl_map_get_space(map));
	local = isl_map_is_subset(map, test);
	isl_map_free(map);
	isl_map_free(test);

	return local;
}

/* For each conditional validity constraint that is adjacent
 * to a condition with domain in condition_source or range in condition_sink,
 * turn it into an unconditional validity constraint.
 */
static int unconditionalize_adjacent_validity(struct isl_sched_graph *graph,
	__isl_take isl_union_set *condition_source,
	__isl_take isl_union_set *condition_sink)
{
	int i;

	condition_source = isl_union_set_coalesce(condition_source);
	condition_sink = isl_union_set_coalesce(condition_sink);

	for (i = 0; i < graph->n_edge; ++i) {
		int adjacent;
		isl_union_map *validity;

		if (!isl_sched_edge_is_conditional_validity(&graph->edge[i]))
			continue;
		if (is_validity(&graph->edge[i]))
			continue;

		validity = graph->edge[i].tagged_validity;
		adjacent = domain_intersects(validity, condition_sink);
		if (adjacent >= 0 && !adjacent)
			adjacent = range_intersects(validity, condition_source);
		if (adjacent < 0)
			goto error;
		if (!adjacent)
			continue;

		set_validity(&graph->edge[i]);
	}

	isl_union_set_free(condition_source);
	isl_union_set_free(condition_sink);
	return 0;
error:
	isl_union_set_free(condition_source);
	isl_union_set_free(condition_sink);
	return -1;
}

/* Update the dependence relations of all edges based on the current schedule
 * and enforce conditional validity constraints that are adjacent
 * to satisfied condition constraints.
 *
 * First check if any of the condition constraints are satisfied
 * (i.e., not local to the outer schedule) and keep track of
 * their domain and range.
 * Then update all dependence relations (which removes the non-local
 * constraints).
 * Finally, if any condition constraints turned out to be satisfied,
 * then turn all adjacent conditional validity constraints into
 * unconditional validity constraints.
 */
static int update_edges(isl_ctx *ctx, struct isl_sched_graph *graph)
{
	int i;
	int any = 0;
	isl_union_set *source, *sink;

	source = isl_union_set_empty(isl_space_params_alloc(ctx, 0));
	sink = isl_union_set_empty(isl_space_params_alloc(ctx, 0));
	for (i = 0; i < graph->n_edge; ++i) {
		int local;
		isl_union_set *uset;
		isl_union_map *umap;

		if (!isl_sched_edge_is_condition(&graph->edge[i]))
			continue;
		if (is_local(&graph->edge[i]))
			continue;
		local = is_condition_false(&graph->edge[i]);
		if (local < 0)
			goto error;
		if (local)
			continue;

		any = 1;

		umap = isl_union_map_copy(graph->edge[i].tagged_condition);
		uset = isl_union_map_domain(umap);
		source = isl_union_set_union(source, uset);

		umap = isl_union_map_copy(graph->edge[i].tagged_condition);
		uset = isl_union_map_range(umap);
		sink = isl_union_set_union(sink, uset);
	}

	for (i = 0; i < graph->n_edge; ++i) {
		if (update_edge(ctx, graph, &graph->edge[i]) < 0)
			goto error;
	}

	if (any)
		return unconditionalize_adjacent_validity(graph, source, sink);

	isl_union_set_free(source);
	isl_union_set_free(sink);
	return 0;
error:
	isl_union_set_free(source);
	isl_union_set_free(sink);
	return -1;
}

static void next_band(struct isl_sched_graph *graph)
{
	graph->band_start = graph->n_total_row;
}

/* Return the union of the universe domains of the nodes in "graph"
 * that satisfy "pred".
 */
static __isl_give isl_union_set *isl_sched_graph_domain(isl_ctx *ctx,
	struct isl_sched_graph *graph,
	int (*pred)(struct isl_sched_node *node, int data), int data)
{
	int i;
	isl_set *set;
	isl_union_set *dom;

	for (i = 0; i < graph->n; ++i)
		if (pred(&graph->node[i], data))
			break;

	if (i >= graph->n)
		isl_die(ctx, isl_error_internal,
			"empty component", return NULL);

	set = isl_set_universe(isl_space_copy(graph->node[i].space));
	dom = isl_union_set_from_set(set);

	for (i = i + 1; i < graph->n; ++i) {
		if (!pred(&graph->node[i], data))
			continue;
		set = isl_set_universe(isl_space_copy(graph->node[i].space));
		dom = isl_union_set_union(dom, isl_union_set_from_set(set));
	}

	return dom;
}

/* Return a union of universe domains corresponding to the nodes
 * in the SCC with index "scc".
 */
__isl_give isl_union_set *isl_sched_graph_extract_scc(isl_ctx *ctx,
	struct isl_sched_graph *graph, int scc)
{
	return isl_sched_graph_domain(ctx, graph,
					&isl_sched_node_scc_exactly, scc);
}

/* Return a list of unions of universe domains, where each element
 * in the list corresponds to an SCC (or WCC) indexed by node->scc.
 */
__isl_give isl_union_set_list *isl_sched_graph_extract_sccs(isl_ctx *ctx,
	struct isl_sched_graph *graph)
{
	int i;
	isl_union_set_list *filters;

	filters = isl_union_set_list_alloc(ctx, graph->scc);
	for (i = 0; i < graph->scc; ++i) {
		isl_union_set *dom;

		dom = isl_sched_graph_extract_scc(ctx, graph, i);
		filters = isl_union_set_list_add(filters, dom);
	}

	return filters;
}

/* Return a list of two unions of universe domains, one for the SCCs up
 * to and including graph->src_scc and another for the other SCCs.
 */
static __isl_give isl_union_set_list *extract_split(isl_ctx *ctx,
	struct isl_sched_graph *graph)
{
	isl_union_set *dom;
	isl_union_set_list *filters;

	filters = isl_union_set_list_alloc(ctx, 2);
	dom = isl_sched_graph_domain(ctx, graph,
					&node_scc_at_most, graph->src_scc);
	filters = isl_union_set_list_add(filters, dom);
	dom = isl_sched_graph_domain(ctx, graph,
					&node_scc_at_least, graph->src_scc + 1);
	filters = isl_union_set_list_add(filters, dom);

	return filters;
}

/* Copy nodes that satisfy node_pred from the src dependence graph
 * to the dst dependence graph.
 */
static isl_stat copy_nodes(struct isl_sched_graph *dst,
	struct isl_sched_graph *src,
	int (*node_pred)(struct isl_sched_node *node, int data), int data)
{
	int i;

	dst->n = 0;
	for (i = 0; i < src->n; ++i) {
		int j;

		if (!node_pred(&src->node[i], data))
			continue;

		j = dst->n;
		dst->node[j].space = isl_space_copy(src->node[i].space);
		dst->node[j].compressed = src->node[i].compressed;
		dst->node[j].hull = isl_set_copy(src->node[i].hull);
		dst->node[j].compress =
			isl_multi_aff_copy(src->node[i].compress);
		dst->node[j].decompress =
			isl_pw_multi_aff_copy(src->node[i].decompress);
		dst->node[j].nvar = src->node[i].nvar;
		dst->node[j].nparam = src->node[i].nparam;
		dst->node[j].sched = isl_mat_copy(src->node[i].sched);
		dst->node[j].sched_map = isl_map_copy(src->node[i].sched_map);
		dst->node[j].coincident = src->node[i].coincident;
		dst->node[j].sizes = isl_multi_val_copy(src->node[i].sizes);
		dst->node[j].bounds = isl_basic_set_copy(src->node[i].bounds);
		dst->node[j].max = isl_vec_copy(src->node[i].max);
		dst->n++;

		if (!dst->node[j].space || !dst->node[j].sched)
			return isl_stat_error;
		if (dst->node[j].compressed &&
		    (!dst->node[j].hull || !dst->node[j].compress ||
		     !dst->node[j].decompress))
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Copy non-empty edges that satisfy edge_pred from the src dependence graph
 * to the dst dependence graph.
 * If the source or destination node of the edge is not in the destination
 * graph, then it must be a backward proximity edge and it should simply
 * be ignored.
 */
static isl_stat copy_edges(isl_ctx *ctx, struct isl_sched_graph *dst,
	struct isl_sched_graph *src,
	int (*edge_pred)(struct isl_sched_edge *edge, int data), int data)
{
	int i;

	dst->n_edge = 0;
	for (i = 0; i < src->n_edge; ++i) {
		struct isl_sched_edge *edge = &src->edge[i];
		isl_map *map;
		isl_union_map *tagged_condition;
		isl_union_map *tagged_validity;
		struct isl_sched_node *dst_src, *dst_dst;

		if (!edge_pred(edge, data))
			continue;

		if (isl_map_plain_is_empty(edge->map))
			continue;

		dst_src = isl_sched_graph_find_node(ctx, dst, edge->src->space);
		dst_dst = isl_sched_graph_find_node(ctx, dst, edge->dst->space);
		if (!dst_src || !dst_dst)
			return isl_stat_error;
		if (!isl_sched_graph_is_node(dst, dst_src) ||
		    !isl_sched_graph_is_node(dst, dst_dst)) {
			if (is_validity(edge) ||
			    isl_sched_edge_is_conditional_validity(edge))
				isl_die(ctx, isl_error_internal,
					"backward (conditional) validity edge",
					return isl_stat_error);
			continue;
		}

		map = isl_map_copy(edge->map);
		tagged_condition = isl_union_map_copy(edge->tagged_condition);
		tagged_validity = isl_union_map_copy(edge->tagged_validity);

		dst->edge[dst->n_edge].src = dst_src;
		dst->edge[dst->n_edge].dst = dst_dst;
		dst->edge[dst->n_edge].map = map;
		dst->edge[dst->n_edge].tagged_condition = tagged_condition;
		dst->edge[dst->n_edge].tagged_validity = tagged_validity;
		dst->edge[dst->n_edge].types = edge->types;
		dst->n_edge++;

		if (edge->tagged_condition && !tagged_condition)
			return isl_stat_error;
		if (edge->tagged_validity && !tagged_validity)
			return isl_stat_error;

		if (graph_edge_tables_add(ctx, dst,
					    &dst->edge[dst->n_edge - 1]) < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Compute the maximal number of variables over all nodes.
 * This is the maximal number of linearly independent schedule
 * rows that we need to compute.
 * Just in case we end up in a part of the dependence graph
 * with only lower-dimensional domains, we make sure we will
 * compute the required amount of extra linearly independent rows.
 */
isl_stat isl_sched_graph_compute_maxvar(struct isl_sched_graph *graph)
{
	int i;

	graph->maxvar = 0;
	for (i = 0; i < graph->n; ++i) {
		struct isl_sched_node *node = &graph->node[i];
		int nvar;

		if (isl_sched_node_update_vmap(node) < 0)
			return isl_stat_error;
		nvar = node->nvar + graph->n_row - node->rank;
		if (nvar > graph->maxvar)
			graph->maxvar = nvar;
	}

	return isl_stat_ok;
}

/* Extract the subgraph of "graph" that consists of the nodes satisfying
 * "node_pred" and the edges satisfying "edge_pred" and store
 * the result in "sub".
 */
isl_stat isl_sched_graph_extract_sub_graph(isl_ctx *ctx,
	struct isl_sched_graph *graph,
	int (*node_pred)(struct isl_sched_node *node, int data),
	int (*edge_pred)(struct isl_sched_edge *edge, int data),
	int data, struct isl_sched_graph *sub)
{
	int i, n = 0, n_edge = 0;
	int t;

	for (i = 0; i < graph->n; ++i)
		if (node_pred(&graph->node[i], data))
			++n;
	for (i = 0; i < graph->n_edge; ++i)
		if (edge_pred(&graph->edge[i], data))
			++n_edge;
	if (graph_alloc(ctx, sub, n, n_edge) < 0)
		return isl_stat_error;
	sub->root = graph->root;
	if (copy_nodes(sub, graph, node_pred, data) < 0)
		return isl_stat_error;
	if (graph_init_table(ctx, sub) < 0)
		return isl_stat_error;
	for (t = 0; t <= isl_edge_last; ++t)
		sub->max_edge[t] = graph->max_edge[t];
	if (graph_init_edge_tables(ctx, sub) < 0)
		return isl_stat_error;
	if (copy_edges(ctx, sub, graph, edge_pred, data) < 0)
		return isl_stat_error;
	sub->n_row = graph->n_row;
	sub->max_row = graph->max_row;
	sub->n_total_row = graph->n_total_row;
	sub->band_start = graph->band_start;

	return isl_stat_ok;
}

static __isl_give isl_schedule_node *compute_schedule(isl_schedule_node *node,
	struct isl_sched_graph *graph);
static __isl_give isl_schedule_node *compute_schedule_wcc(
	isl_schedule_node *node, struct isl_sched_graph *graph);

/* Compute a schedule for a subgraph of "graph".  In particular, for
 * the graph composed of nodes that satisfy node_pred and edges that
 * that satisfy edge_pred.
 * If the subgraph is known to consist of a single component, then wcc should
 * be set and then we call compute_schedule_wcc on the constructed subgraph.
 * Otherwise, we call compute_schedule, which will check whether the subgraph
 * is connected.
 *
 * The schedule is inserted at "node" and the updated schedule node
 * is returned.
 */
static __isl_give isl_schedule_node *compute_sub_schedule(
	__isl_take isl_schedule_node *node, isl_ctx *ctx,
	struct isl_sched_graph *graph,
	int (*node_pred)(struct isl_sched_node *node, int data),
	int (*edge_pred)(struct isl_sched_edge *edge, int data),
	int data, int wcc)
{
	struct isl_sched_graph split = { 0 };

	if (isl_sched_graph_extract_sub_graph(ctx, graph, node_pred, edge_pred,
						data, &split) < 0)
		goto error;

	if (wcc)
		node = compute_schedule_wcc(node, &split);
	else
		node = compute_schedule(node, &split);

	isl_sched_graph_free(ctx, &split);
	return node;
error:
	isl_sched_graph_free(ctx, &split);
	return isl_schedule_node_free(node);
}

int isl_sched_edge_scc_exactly(struct isl_sched_edge *edge, int scc)
{
	return edge->src->scc == scc && edge->dst->scc == scc;
}

static int edge_dst_scc_at_most(struct isl_sched_edge *edge, int scc)
{
	return edge->dst->scc <= scc;
}

static int edge_src_scc_at_least(struct isl_sched_edge *edge, int scc)
{
	return edge->src->scc >= scc;
}

/* Reset the current band by dropping all its schedule rows.
 */
static isl_stat reset_band(struct isl_sched_graph *graph)
{
	int i;
	int drop;

	drop = graph->n_total_row - graph->band_start;
	graph->n_total_row -= drop;
	graph->n_row -= drop;

	for (i = 0; i < graph->n; ++i) {
		struct isl_sched_node *node = &graph->node[i];

		isl_map_free(node->sched_map);
		node->sched_map = NULL;

		node->sched = isl_mat_drop_rows(node->sched,
						graph->band_start, drop);

		if (!node->sched)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Split the current graph into two parts and compute a schedule for each
 * part individually.  In particular, one part consists of all SCCs up
 * to and including graph->src_scc, while the other part contains the other
 * SCCs.  The split is enforced by a sequence node inserted at position "node"
 * in the schedule tree.  Return the updated schedule node.
 * If either of these two parts consists of a sequence, then it is spliced
 * into the sequence containing the two parts.
 *
 * The current band is reset. It would be possible to reuse
 * the previously computed rows as the first rows in the next
 * band, but recomputing them may result in better rows as we are looking
 * at a smaller part of the dependence graph.
 */
static __isl_give isl_schedule_node *compute_split_schedule(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph)
{
	isl_ctx *ctx;
	isl_union_set_list *filters;

	if (!node)
		return NULL;

	if (reset_band(graph) < 0)
		return isl_schedule_node_free(node);

	next_band(graph);

	ctx = isl_schedule_node_get_ctx(node);
	filters = extract_split(ctx, graph);
	node = isl_schedule_node_insert_sequence(node, filters);
	node = isl_schedule_node_grandchild(node, 1, 0);

	node = compute_sub_schedule(node, ctx, graph,
				&node_scc_at_least, &edge_src_scc_at_least,
				graph->src_scc + 1, 0);
	node = isl_schedule_node_grandparent(node);
	node = isl_schedule_node_grandchild(node, 0, 0);
	node = compute_sub_schedule(node, ctx, graph,
				&node_scc_at_most, &edge_dst_scc_at_most,
				graph->src_scc, 0);
	node = isl_schedule_node_grandparent(node);

	node = isl_schedule_node_sequence_splice_children(node);

	return node;
}

/* Insert a band node at position "node" in the schedule tree corresponding
 * to the current band in "graph".  Mark the band node permutable
 * if "permutable" is set.
 * The partial schedules and the coincidence property are extracted
 * from the graph nodes.
 * Return the updated schedule node.
 */
static __isl_give isl_schedule_node *insert_current_band(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph,
	int permutable)
{
	int i;
	int start, end, n;
	isl_multi_aff *ma;
	isl_multi_pw_aff *mpa;
	isl_multi_union_pw_aff *mupa;

	if (!node)
		return NULL;

	if (graph->n < 1)
		isl_die(isl_schedule_node_get_ctx(node), isl_error_internal,
			"graph should have at least one node",
			return isl_schedule_node_free(node));

	start = graph->band_start;
	end = graph->n_total_row;
	n = end - start;

	ma = isl_sched_node_extract_partial_schedule_multi_aff(&graph->node[0],
								start, n);
	mpa = isl_multi_pw_aff_from_multi_aff(ma);
	mupa = isl_multi_union_pw_aff_from_multi_pw_aff(mpa);

	for (i = 1; i < graph->n; ++i) {
		isl_multi_union_pw_aff *mupa_i;

		ma = isl_sched_node_extract_partial_schedule_multi_aff(
						&graph->node[i], start, n);
		mpa = isl_multi_pw_aff_from_multi_aff(ma);
		mupa_i = isl_multi_union_pw_aff_from_multi_pw_aff(mpa);
		mupa = isl_multi_union_pw_aff_union_add(mupa, mupa_i);
	}
	node = isl_schedule_node_insert_partial_schedule(node, mupa);

	for (i = 0; i < n; ++i)
		node = isl_schedule_node_band_member_set_coincident(node, i,
					graph->node[0].coincident[start + i]);
	node = isl_schedule_node_band_set_permutable(node, permutable);

	return node;
}

/* Update the dependence relations based on the current schedule,
 * add the current band to "node" and then continue with the computation
 * of the next band.
 * Return the updated schedule node.
 */
static __isl_give isl_schedule_node *compute_next_band(
	__isl_take isl_schedule_node *node,
	struct isl_sched_graph *graph, int permutable)
{
	isl_ctx *ctx;

	if (!node)
		return NULL;

	ctx = isl_schedule_node_get_ctx(node);
	if (update_edges(ctx, graph) < 0)
		return isl_schedule_node_free(node);
	node = insert_current_band(node, graph, permutable);
	next_band(graph);

	node = isl_schedule_node_child(node, 0);
	node = compute_schedule(node, graph);
	node = isl_schedule_node_parent(node);

	return node;
}

/* Add the constraints "coef" derived from an edge from "node" to itself
 * to graph->lp in order to respect the dependences and to try and carry them.
 * "pos" is the sequence number of the edge that needs to be carried.
 * "coef" represents general constraints on coefficients (c_0, c_x)
 * of valid constraints for (y - x) with x and y instances of the node.
 *
 * The constraints added to graph->lp need to enforce
 *
 *	(c_j_0 + c_j_x y) - (c_j_0 + c_j_x x)
 *	= c_j_x (y - x) >= e_i
 *
 * for each (x,y) in the dependence relation of the edge.
 * That is, (-e_i, c_j_x) needs to be plugged in for (c_0, c_x),
 * taking into account that each coefficient in c_j_x is represented
 * as a pair of non-negative coefficients.
 */
static isl_stat add_intra_constraints(struct isl_sched_graph *graph,
	struct isl_sched_node *node, __isl_take isl_basic_set *coef, int pos)
{
	isl_size offset;
	isl_ctx *ctx;
	isl_dim_map *dim_map;

	offset = coef_var_offset(coef);
	if (offset < 0)
		coef = isl_basic_set_free(coef);
	if (!coef)
		return isl_stat_error;

	ctx = isl_basic_set_get_ctx(coef);
	dim_map = intra_dim_map(ctx, graph, node, offset, 1);
	isl_dim_map_range(dim_map, 3 + pos, 0, 0, 0, 1, -1);
	graph->lp = add_constraints_dim_map(graph->lp, coef, dim_map);

	return isl_stat_ok;
}

/* Add the constraints "coef" derived from an edge from "src" to "dst"
 * to graph->lp in order to respect the dependences and to try and carry them.
 * "pos" is the sequence number of the edge that needs to be carried or
 * -1 if no attempt should be made to carry the dependences.
 * "coef" represents general constraints on coefficients (c_0, c_n, c_x, c_y)
 * of valid constraints for (x, y) with x and y instances of "src" and "dst".
 *
 * The constraints added to graph->lp need to enforce
 *
 *	(c_k_0 + c_k_n n + c_k_x y) - (c_j_0 + c_j_n n + c_j_x x) >= e_i
 *
 * for each (x,y) in the dependence relation of the edge or
 *
 *	(c_k_0 + c_k_n n + c_k_x y) - (c_j_0 + c_j_n n + c_j_x x) >= 0
 *
 * if pos is -1.
 * That is,
 * (-e_i + c_k_0 - c_j_0, c_k_n - c_j_n, -c_j_x, c_k_x)
 * or
 * (c_k_0 - c_j_0, c_k_n - c_j_n, -c_j_x, c_k_x)
 * needs to be plugged in for (c_0, c_n, c_x, c_y),
 * taking into account that each coefficient in c_j_x and c_k_x is represented
 * as a pair of non-negative coefficients.
 */
static isl_stat add_inter_constraints(struct isl_sched_graph *graph,
	struct isl_sched_node *src, struct isl_sched_node *dst,
	__isl_take isl_basic_set *coef, int pos)
{
	isl_size offset;
	isl_ctx *ctx;
	isl_dim_map *dim_map;

	offset = coef_var_offset(coef);
	if (offset < 0)
		coef = isl_basic_set_free(coef);
	if (!coef)
		return isl_stat_error;

	ctx = isl_basic_set_get_ctx(coef);
	dim_map = inter_dim_map(ctx, graph, src, dst, offset, 1);
	if (pos >= 0)
		isl_dim_map_range(dim_map, 3 + pos, 0, 0, 0, 1, -1);
	graph->lp = add_constraints_dim_map(graph->lp, coef, dim_map);

	return isl_stat_ok;
}

/* Data structure for keeping track of the data needed
 * to exploit non-trivial lineality spaces.
 *
 * "any_non_trivial" is true if there are any non-trivial lineality spaces.
 * If "any_non_trivial" is not true, then "equivalent" and "mask" may be NULL.
 * "equivalent" connects instances to other instances on the same line(s).
 * "mask" contains the domain spaces of "equivalent".
 * Any instance set not in "mask" does not have a non-trivial lineality space.
 */
struct isl_exploit_lineality_data {
	isl_bool any_non_trivial;
	isl_union_map *equivalent;
	isl_union_set *mask;
};

/* Data structure collecting information used during the construction
 * of an LP for carrying dependences.
 *
 * "intra" is a sequence of coefficient constraints for intra-node edges.
 * "inter" is a sequence of coefficient constraints for inter-node edges.
 * "lineality" contains data used to exploit non-trivial lineality spaces.
 */
struct isl_carry {
	isl_basic_set_list *intra;
	isl_basic_set_list *inter;
	struct isl_exploit_lineality_data lineality;
};

/* Free all the data stored in "carry".
 */
static void isl_carry_clear(struct isl_carry *carry)
{
	isl_basic_set_list_free(carry->intra);
	isl_basic_set_list_free(carry->inter);
	isl_union_map_free(carry->lineality.equivalent);
	isl_union_set_free(carry->lineality.mask);
}

/* Return a pointer to the node in "graph" that lives in "space".
 * If the requested node has been compressed, then "space"
 * corresponds to the compressed space.
 * The graph is assumed to have such a node.
 * Return NULL in case of error.
 *
 * First try and see if "space" is the space of an uncompressed node.
 * If so, return that node.
 * Otherwise, "space" was constructed by construct_compressed_id and
 * contains a user pointer pointing to the node in the tuple id.
 * However, this node belongs to the original dependence graph.
 * If "graph" is a subgraph of this original dependence graph,
 * then the node with the same space still needs to be looked up
 * in the current graph.
 */
static struct isl_sched_node *graph_find_compressed_node(isl_ctx *ctx,
	struct isl_sched_graph *graph, __isl_keep isl_space *space)
{
	isl_id *id;
	struct isl_sched_node *node;

	if (!space)
		return NULL;

	node = isl_sched_graph_find_node(ctx, graph, space);
	if (!node)
		return NULL;
	if (isl_sched_graph_is_node(graph, node))
		return node;

	id = isl_space_get_tuple_id(space, isl_dim_set);
	node = isl_id_get_user(id);
	isl_id_free(id);

	if (!node)
		return NULL;

	if (!isl_sched_graph_is_node(graph->root, node))
		isl_die(ctx, isl_error_internal,
			"space points to invalid node", return NULL);
	if (graph != graph->root)
		node = isl_sched_graph_find_node(ctx, graph, node->space);
	if (!isl_sched_graph_is_node(graph, node))
		isl_die(ctx, isl_error_internal,
			"unable to find node", return NULL);

	return node;
}

/* Internal data structure for add_all_constraints.
 *
 * "graph" is the schedule constraint graph for which an LP problem
 * is being constructed.
 * "carry_inter" indicates whether inter-node edges should be carried.
 * "pos" is the position of the next edge that needs to be carried.
 */
struct isl_add_all_constraints_data {
	isl_ctx *ctx;
	struct isl_sched_graph *graph;
	int carry_inter;
	int pos;
};

/* Add the constraints "coef" derived from an edge from a node to itself
 * to data->graph->lp in order to respect the dependences and
 * to try and carry them.
 *
 * The space of "coef" is of the form
 *
 *	coefficients[[c_cst] -> S[c_x]]
 *
 * with S[c_x] the (compressed) space of the node.
 * Extract the node from the space and call add_intra_constraints.
 */
static isl_stat lp_add_intra(__isl_take isl_basic_set *coef, void *user)
{
	struct isl_add_all_constraints_data *data = user;
	isl_space *space;
	struct isl_sched_node *node;

	space = isl_basic_set_get_space(coef);
	space = isl_space_range(isl_space_unwrap(space));
	node = graph_find_compressed_node(data->ctx, data->graph, space);
	isl_space_free(space);
	return add_intra_constraints(data->graph, node, coef, data->pos++);
}

/* Add the constraints "coef" derived from an edge from a node j
 * to a node k to data->graph->lp in order to respect the dependences and
 * to try and carry them (provided data->carry_inter is set).
 *
 * The space of "coef" is of the form
 *
 *	coefficients[[c_cst, c_n] -> [S_j[c_x] -> S_k[c_y]]]
 *
 * with S_j[c_x] and S_k[c_y] the (compressed) spaces of the nodes.
 * Extract the nodes from the space and call add_inter_constraints.
 */
static isl_stat lp_add_inter(__isl_take isl_basic_set *coef, void *user)
{
	struct isl_add_all_constraints_data *data = user;
	isl_space *space, *dom;
	struct isl_sched_node *src, *dst;
	int pos;

	space = isl_basic_set_get_space(coef);
	space = isl_space_unwrap(isl_space_range(isl_space_unwrap(space)));
	dom = isl_space_domain(isl_space_copy(space));
	src = graph_find_compressed_node(data->ctx, data->graph, dom);
	isl_space_free(dom);
	space = isl_space_range(space);
	dst = graph_find_compressed_node(data->ctx, data->graph, space);
	isl_space_free(space);

	pos = data->carry_inter ? data->pos++ : -1;
	return add_inter_constraints(data->graph, src, dst, coef, pos);
}

/* Add constraints to graph->lp that force all (conditional) validity
 * dependences to be respected and attempt to carry them.
 * "intra" is the sequence of coefficient constraints for intra-node edges.
 * "inter" is the sequence of coefficient constraints for inter-node edges.
 * "carry_inter" indicates whether inter-node edges should be carried or
 * only respected.
 */
static isl_stat add_all_constraints(isl_ctx *ctx, struct isl_sched_graph *graph,
	__isl_keep isl_basic_set_list *intra,
	__isl_keep isl_basic_set_list *inter, int carry_inter)
{
	struct isl_add_all_constraints_data data = { ctx, graph, carry_inter };

	data.pos = 0;
	if (isl_basic_set_list_foreach(intra, &lp_add_intra, &data) < 0)
		return isl_stat_error;
	if (isl_basic_set_list_foreach(inter, &lp_add_inter, &data) < 0)
		return isl_stat_error;
	return isl_stat_ok;
}

/* Internal data structure for count_all_constraints
 * for keeping track of the number of equality and inequality constraints.
 */
struct isl_sched_count {
	int n_eq;
	int n_ineq;
};

/* Add the number of equality and inequality constraints of "bset"
 * to data->n_eq and data->n_ineq.
 */
static isl_stat bset_update_count(__isl_take isl_basic_set *bset, void *user)
{
	struct isl_sched_count *data = user;

	return update_count(bset, 1, &data->n_eq, &data->n_ineq);
}

/* Count the number of equality and inequality constraints
 * that will be added to the carry_lp problem.
 * We count each edge exactly once.
 * "intra" is the sequence of coefficient constraints for intra-node edges.
 * "inter" is the sequence of coefficient constraints for inter-node edges.
 */
static isl_stat count_all_constraints(__isl_keep isl_basic_set_list *intra,
	__isl_keep isl_basic_set_list *inter, int *n_eq, int *n_ineq)
{
	struct isl_sched_count data;

	data.n_eq = data.n_ineq = 0;
	if (isl_basic_set_list_foreach(inter, &bset_update_count, &data) < 0)
		return isl_stat_error;
	if (isl_basic_set_list_foreach(intra, &bset_update_count, &data) < 0)
		return isl_stat_error;

	*n_eq = data.n_eq;
	*n_ineq = data.n_ineq;

	return isl_stat_ok;
}

/* Construct an LP problem for finding schedule coefficients
 * such that the schedule carries as many validity dependences as possible.
 * In particular, for each dependence i, we bound the dependence distance
 * from below by e_i, with 0 <= e_i <= 1 and then maximize the sum
 * of all e_i's.  Dependences with e_i = 0 in the solution are simply
 * respected, while those with e_i > 0 (in practice e_i = 1) are carried.
 * "intra" is the sequence of coefficient constraints for intra-node edges.
 * "inter" is the sequence of coefficient constraints for inter-node edges.
 * "n_edge" is the total number of edges.
 * "carry_inter" indicates whether inter-node edges should be carried or
 * only respected.  That is, if "carry_inter" is not set, then
 * no e_i variables are introduced for the inter-node edges.
 *
 * All variables of the LP are non-negative.  The actual coefficients
 * may be negative, so each coefficient is represented as the difference
 * of two non-negative variables.  The negative part always appears
 * immediately before the positive part.
 * Other than that, the variables have the following order
 *
 *	- sum of (1 - e_i) over all edges
 *	- sum of all c_n coefficients
 *		(unconstrained when computing non-parametric schedules)
 *	- sum of positive and negative parts of all c_x coefficients
 *	- for each edge
 *		- e_i
 *	- for each node
 *		- positive and negative parts of c_i_x, in opposite order
 *		- c_i_n (if parametric)
 *		- c_i_0
 *
 * The constraints are those from the (validity) edges plus three equalities
 * to express the sums and n_edge inequalities to express e_i <= 1.
 */
static isl_stat setup_carry_lp(isl_ctx *ctx, struct isl_sched_graph *graph,
	int n_edge, __isl_keep isl_basic_set_list *intra,
	__isl_keep isl_basic_set_list *inter, int carry_inter)
{
	int i;
	int k;
	isl_space *space;
	unsigned total;
	int n_eq, n_ineq;

	total = 3 + n_edge;
	for (i = 0; i < graph->n; ++i) {
		struct isl_sched_node *node = &graph->node[graph->sorted[i]];
		node->start = total;
		total += 1 + node->nparam + 2 * node->nvar;
	}

	if (count_all_constraints(intra, inter, &n_eq, &n_ineq) < 0)
		return isl_stat_error;

	space = isl_space_set_alloc(ctx, 0, total);
	isl_basic_set_free(graph->lp);
	n_eq += 3;
	n_ineq += n_edge;
	graph->lp = isl_basic_set_alloc_space(space, 0, n_eq, n_ineq);
	graph->lp = isl_basic_set_set_rational(graph->lp);

	k = isl_basic_set_alloc_equality(graph->lp);
	if (k < 0)
		return isl_stat_error;
	isl_seq_clr(graph->lp->eq[k], 1 + total);
	isl_int_set_si(graph->lp->eq[k][0], -n_edge);
	isl_int_set_si(graph->lp->eq[k][1], 1);
	for (i = 0; i < n_edge; ++i)
		isl_int_set_si(graph->lp->eq[k][4 + i], 1);

	if (add_param_sum_constraint(graph, 1) < 0)
		return isl_stat_error;
	if (add_var_sum_constraint(graph, 2) < 0)
		return isl_stat_error;

	for (i = 0; i < n_edge; ++i) {
		k = isl_basic_set_alloc_inequality(graph->lp);
		if (k < 0)
			return isl_stat_error;
		isl_seq_clr(graph->lp->ineq[k], 1 + total);
		isl_int_set_si(graph->lp->ineq[k][4 + i], -1);
		isl_int_set_si(graph->lp->ineq[k][0], 1);
	}

	if (add_all_constraints(ctx, graph, intra, inter, carry_inter) < 0)
		return isl_stat_error;

	return isl_stat_ok;
}

static __isl_give isl_schedule_node *compute_component_schedule(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph,
	int wcc);

/* If the schedule_split_scaled option is set and if the linear
 * parts of the scheduling rows for all nodes in the graphs have
 * a non-trivial common divisor, then remove this
 * common divisor from the linear part.
 * Otherwise, insert a band node directly and continue with
 * the construction of the schedule.
 *
 * If a non-trivial common divisor is found, then
 * the linear part is reduced and the remainder is ignored.
 * The pieces of the graph that are assigned different remainders
 * form (groups of) strongly connected components within
 * the scaled down band.  If needed, they can therefore
 * be ordered along this remainder in a sequence node.
 * However, this ordering is not enforced here in order to allow
 * the scheduler to combine some of the strongly connected components.
 */
static __isl_give isl_schedule_node *split_scaled(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph)
{
	int i;
	int row;
	isl_ctx *ctx;
	isl_int gcd, gcd_i;
	isl_size n_row;

	if (!node)
		return NULL;

	ctx = isl_schedule_node_get_ctx(node);
	if (!ctx->opt->schedule_split_scaled)
		return compute_next_band(node, graph, 0);
	if (graph->n <= 1)
		return compute_next_band(node, graph, 0);
	n_row = isl_mat_rows(graph->node[0].sched);
	if (n_row < 0)
		return isl_schedule_node_free(node);

	isl_int_init(gcd);
	isl_int_init(gcd_i);

	isl_int_set_si(gcd, 0);

	row = n_row - 1;

	for (i = 0; i < graph->n; ++i) {
		struct isl_sched_node *node = &graph->node[i];
		isl_size cols = isl_mat_cols(node->sched);

		if (cols < 0)
			break;
		isl_seq_gcd(node->sched->row[row] + 1, cols - 1, &gcd_i);
		isl_int_gcd(gcd, gcd, gcd_i);
	}

	isl_int_clear(gcd_i);
	if (i < graph->n)
		goto error;

	if (isl_int_cmp_si(gcd, 1) <= 0) {
		isl_int_clear(gcd);
		return compute_next_band(node, graph, 0);
	}

	for (i = 0; i < graph->n; ++i) {
		struct isl_sched_node *node = &graph->node[i];

		isl_int_fdiv_q(node->sched->row[row][0],
			       node->sched->row[row][0], gcd);
		isl_int_mul(node->sched->row[row][0],
			    node->sched->row[row][0], gcd);
		node->sched = isl_mat_scale_down_row(node->sched, row, gcd);
		if (!node->sched)
			goto error;
	}

	isl_int_clear(gcd);

	return compute_next_band(node, graph, 0);
error:
	isl_int_clear(gcd);
	return isl_schedule_node_free(node);
}

/* Is the schedule row "sol" trivial on node "node"?
 * That is, is the solution zero on the dimensions linearly independent of
 * the previously found solutions?
 * Return 1 if the solution is trivial, 0 if it is not and -1 on error.
 *
 * Each coefficient is represented as the difference between
 * two non-negative values in "sol".
 * We construct the schedule row s and check if it is linearly
 * independent of previously computed schedule rows
 * by computing T s, with T the linear combinations that are zero
 * on linearly dependent schedule rows.
 * If the result consists of all zeros, then the solution is trivial.
 */
static int is_trivial(struct isl_sched_node *node, __isl_keep isl_vec *sol)
{
	int trivial;
	isl_vec *node_sol;

	if (!sol)
		return -1;
	if (node->nvar == node->rank)
		return 0;

	node_sol = extract_var_coef(node, sol);
	node_sol = isl_mat_vec_product(isl_mat_copy(node->indep), node_sol);
	if (!node_sol)
		return -1;

	trivial = isl_seq_first_non_zero(node_sol->el,
					node->nvar - node->rank) == -1;

	isl_vec_free(node_sol);

	return trivial;
}

/* Is the schedule row "sol" trivial on any node where it should
 * not be trivial?
 * Return 1 if any solution is trivial, 0 if they are not and -1 on error.
 */
static int is_any_trivial(struct isl_sched_graph *graph,
	__isl_keep isl_vec *sol)
{
	int i;

	for (i = 0; i < graph->n; ++i) {
		struct isl_sched_node *node = &graph->node[i];
		int trivial;

		if (!needs_row(graph, node))
			continue;
		trivial = is_trivial(node, sol);
		if (trivial < 0 || trivial)
			return trivial;
	}

	return 0;
}

/* Does the schedule represented by "sol" perform loop coalescing on "node"?
 * If so, return the position of the coalesced dimension.
 * Otherwise, return node->nvar or -1 on error.
 *
 * In particular, look for pairs of coefficients c_i and c_j such that
 * |c_j/c_i| > ceil(size_i/2), i.e., |c_j| > |c_i * ceil(size_i/2)|.
 * If any such pair is found, then return i.
 * If size_i is infinity, then no check on c_i needs to be performed.
 */
static int find_node_coalescing(struct isl_sched_node *node,
	__isl_keep isl_vec *sol)
{
	int i, j;
	isl_int max;
	isl_vec *csol;

	if (node->nvar <= 1)
		return node->nvar;

	csol = extract_var_coef(node, sol);
	if (!csol)
		return -1;
	isl_int_init(max);
	for (i = 0; i < node->nvar; ++i) {
		isl_val *v;

		if (isl_int_is_zero(csol->el[i]))
			continue;
		v = isl_multi_val_get_val(node->sizes, i);
		if (!v)
			goto error;
		if (!isl_val_is_int(v)) {
			isl_val_free(v);
			continue;
		}
		v = isl_val_div_ui(v, 2);
		v = isl_val_ceil(v);
		if (!v)
			goto error;
		isl_int_mul(max, v->n, csol->el[i]);
		isl_val_free(v);

		for (j = 0; j < node->nvar; ++j) {
			if (j == i)
				continue;
			if (isl_int_abs_gt(csol->el[j], max))
				break;
		}
		if (j < node->nvar)
			break;
	}

	isl_int_clear(max);
	isl_vec_free(csol);
	return i;
error:
	isl_int_clear(max);
	isl_vec_free(csol);
	return -1;
}

/* Force the schedule coefficient at position "pos" of "node" to be zero
 * in "tl".
 * The coefficient is encoded as the difference between two non-negative
 * variables.  Force these two variables to have the same value.
 */
static __isl_give isl_tab_lexmin *zero_out_node_coef(
	__isl_take isl_tab_lexmin *tl, struct isl_sched_node *node, int pos)
{
	int dim;
	isl_ctx *ctx;
	isl_vec *eq;

	ctx = isl_space_get_ctx(node->space);
	dim = isl_tab_lexmin_dim(tl);
	if (dim < 0)
		return isl_tab_lexmin_free(tl);
	eq = isl_vec_alloc(ctx, 1 + dim);
	eq = isl_vec_clr(eq);
	if (!eq)
		return isl_tab_lexmin_free(tl);

	pos = 1 + node_var_coef_pos(node, pos);
	isl_int_set_si(eq->el[pos], 1);
	isl_int_set_si(eq->el[pos + 1], -1);
	tl = isl_tab_lexmin_add_eq(tl, eq->el);
	isl_vec_free(eq);

	return tl;
}

/* Return the lexicographically smallest rational point in the basic set
 * from which "tl" was constructed, double checking that this input set
 * was not empty.
 */
static __isl_give isl_vec *non_empty_solution(__isl_keep isl_tab_lexmin *tl)
{
	isl_vec *sol;

	sol = isl_tab_lexmin_get_solution(tl);
	if (!sol)
		return NULL;
	if (sol->size == 0)
		isl_die(isl_vec_get_ctx(sol), isl_error_internal,
			"error in schedule construction",
			return isl_vec_free(sol));
	return sol;
}

/* Does the solution "sol" of the LP problem constructed by setup_carry_lp
 * carry any of the "n_edge" groups of dependences?
 * The value in the first position is the sum of (1 - e_i) over all "n_edge"
 * edges, with 0 <= e_i <= 1 equal to 1 when the dependences represented
 * by the edge are carried by the solution.
 * If the sum of the (1 - e_i) is smaller than "n_edge" then at least
 * one of those is carried.
 *
 * Note that despite the fact that the problem is solved using a rational
 * solver, the solution is guaranteed to be integral.
 * Specifically, the dependence distance lower bounds e_i (and therefore
 * also their sum) are integers.  See Lemma 5 of [1].
 *
 * Any potential denominator of the sum is cleared by this function.
 * The denominator is not relevant for any of the other elements
 * in the solution.
 *
 * [1] P. Feautrier, Some Efficient Solutions to the Affine Scheduling
 *     Problem, Part II: Multi-Dimensional Time.
 *     In Intl. Journal of Parallel Programming, 1992.
 */
static int carries_dependences(__isl_keep isl_vec *sol, int n_edge)
{
	isl_int_divexact(sol->el[1], sol->el[1], sol->el[0]);
	isl_int_set_si(sol->el[0], 1);
	return isl_int_cmp_si(sol->el[1], n_edge) < 0;
}

/* Return the lexicographically smallest rational point in "lp",
 * assuming that all variables are non-negative and performing some
 * additional sanity checks.
 * If "want_integral" is set, then compute the lexicographically smallest
 * integer point instead.
 * In particular, "lp" should not be empty by construction.
 * Double check that this is the case.
 * If dependences are not carried for any of the "n_edge" edges,
 * then return an empty vector.
 *
 * If the schedule_treat_coalescing option is set and
 * if the computed schedule performs loop coalescing on a given node,
 * i.e., if it is of the form
 *
 *	c_i i + c_j j + ...
 *
 * with |c_j/c_i| >= size_i, then force the coefficient c_i to be zero
 * to cut out this solution.  Repeat this process until no more loop
 * coalescing occurs or until no more dependences can be carried.
 * In the latter case, revert to the previously computed solution.
 *
 * If the caller requests an integral solution and if coalescing should
 * be treated, then perform the coalescing treatment first as
 * an integral solution computed before coalescing treatment
 * would carry the same number of edges and would therefore probably
 * also be coalescing.
 *
 * To allow the coalescing treatment to be performed first,
 * the initial solution is allowed to be rational and it is only
 * cut out (if needed) in the next iteration, if no coalescing measures
 * were taken.
 */
static __isl_give isl_vec *non_neg_lexmin(struct isl_sched_graph *graph,
	__isl_take isl_basic_set *lp, int n_edge, int want_integral)
{
	int i, pos, cut;
	isl_ctx *ctx;
	isl_tab_lexmin *tl;
	isl_vec *sol = NULL, *prev;
	int treat_coalescing;
	int try_again;

	if (!lp)
		return NULL;
	ctx = isl_basic_set_get_ctx(lp);
	treat_coalescing = isl_options_get_schedule_treat_coalescing(ctx);
	tl = isl_tab_lexmin_from_basic_set(lp);

	cut = 0;
	do {
		int integral;

		try_again = 0;
		if (cut)
			tl = isl_tab_lexmin_cut_to_integer(tl);
		prev = sol;
		sol = non_empty_solution(tl);
		if (!sol)
			goto error;

		integral = isl_int_is_one(sol->el[0]);
		if (!carries_dependences(sol, n_edge)) {
			if (!prev)
				prev = isl_vec_alloc(ctx, 0);
			isl_vec_free(sol);
			sol = prev;
			break;
		}
		prev = isl_vec_free(prev);
		cut = want_integral && !integral;
		if (cut)
			try_again = 1;
		if (!treat_coalescing)
			continue;
		for (i = 0; i < graph->n; ++i) {
			struct isl_sched_node *node = &graph->node[i];

			pos = find_node_coalescing(node, sol);
			if (pos < 0)
				goto error;
			if (pos < node->nvar)
				break;
		}
		if (i < graph->n) {
			try_again = 1;
			tl = zero_out_node_coef(tl, &graph->node[i], pos);
			cut = 0;
		}
	} while (try_again);

	isl_tab_lexmin_free(tl);

	return sol;
error:
	isl_tab_lexmin_free(tl);
	isl_vec_free(prev);
	isl_vec_free(sol);
	return NULL;
}

/* If "edge" is an edge from a node to itself, then add the corresponding
 * dependence relation to "umap".
 * If "node" has been compressed, then the dependence relation
 * is also compressed first.
 */
static __isl_give isl_union_map *add_intra(__isl_take isl_union_map *umap,
	struct isl_sched_edge *edge)
{
	isl_map *map;
	struct isl_sched_node *node = edge->src;

	if (edge->src != edge->dst)
		return umap;

	map = isl_map_copy(edge->map);
	map = compress(map, node, node);
	umap = isl_union_map_add_map(umap, map);
	return umap;
}

/* If "edge" is an edge from a node to another node, then add the corresponding
 * dependence relation to "umap".
 * If the source or destination nodes of "edge" have been compressed,
 * then the dependence relation is also compressed first.
 */
static __isl_give isl_union_map *add_inter(__isl_take isl_union_map *umap,
	struct isl_sched_edge *edge)
{
	isl_map *map;

	if (edge->src == edge->dst)
		return umap;

	map = isl_map_copy(edge->map);
	map = compress(map, edge->src, edge->dst);
	umap = isl_union_map_add_map(umap, map);
	return umap;
}

/* Internal data structure used by union_drop_coalescing_constraints
 * to collect bounds on all relevant statements.
 *
 * "graph" is the schedule constraint graph for which an LP problem
 * is being constructed.
 * "bounds" collects the bounds.
 */
struct isl_collect_bounds_data {
	isl_ctx *ctx;
	struct isl_sched_graph *graph;
	isl_union_set *bounds;
};

/* Add the size bounds for the node with instance deltas in "set"
 * to data->bounds.
 */
static isl_stat collect_bounds(__isl_take isl_set *set, void *user)
{
	struct isl_collect_bounds_data *data = user;
	struct isl_sched_node *node;
	isl_space *space;
	isl_set *bounds;

	space = isl_set_get_space(set);
	isl_set_free(set);

	node = graph_find_compressed_node(data->ctx, data->graph, space);
	isl_space_free(space);

	bounds = isl_set_from_basic_set(get_size_bounds(node));
	data->bounds = isl_union_set_add_set(data->bounds, bounds);

	return isl_stat_ok;
}

/* Drop some constraints from "delta" that could be exploited
 * to construct loop coalescing schedules.
 * In particular, drop those constraint that bound the difference
 * to the size of the domain.
 * Do this for each set/node in "delta" separately.
 * The parameters are assumed to have been projected out by the caller.
 */
static __isl_give isl_union_set *union_drop_coalescing_constraints(isl_ctx *ctx,
	struct isl_sched_graph *graph, __isl_take isl_union_set *delta)
{
	struct isl_collect_bounds_data data = { ctx, graph };

	data.bounds = isl_union_set_empty(isl_space_params_alloc(ctx, 0));
	if (isl_union_set_foreach_set(delta, &collect_bounds, &data) < 0)
		data.bounds = isl_union_set_free(data.bounds);
	delta = isl_union_set_plain_gist(delta, data.bounds);

	return delta;
}

/* Given a non-trivial lineality space "lineality", add the corresponding
 * universe set to data->mask and add a map from elements to
 * other elements along the lines in "lineality" to data->equivalent.
 * If this is the first time this function gets called
 * (data->any_non_trivial is still false), then set data->any_non_trivial and
 * initialize data->mask and data->equivalent.
 *
 * In particular, if the lineality space is defined by equality constraints
 *
 *	E x = 0
 *
 * then construct an affine mapping
 *
 *	f : x -> E x
 *
 * and compute the equivalence relation of having the same image under f:
 *
 *	{ x -> x' : E x = E x' }
 */
static isl_stat add_non_trivial_lineality(__isl_take isl_basic_set *lineality,
	struct isl_exploit_lineality_data *data)
{
	isl_mat *eq;
	isl_space *space;
	isl_set *univ;
	isl_multi_aff *ma;
	isl_multi_pw_aff *mpa;
	isl_map *map;
	isl_size n;

	if (isl_basic_set_check_no_locals(lineality) < 0)
		goto error;

	space = isl_basic_set_get_space(lineality);
	if (!data->any_non_trivial) {
		data->equivalent = isl_union_map_empty(isl_space_copy(space));
		data->mask = isl_union_set_empty(isl_space_copy(space));
	}
	data->any_non_trivial = isl_bool_true;

	univ = isl_set_universe(isl_space_copy(space));
	data->mask = isl_union_set_add_set(data->mask, univ);

	eq = isl_basic_set_extract_equalities(lineality);
	n = isl_mat_rows(eq);
	if (n < 0)
		space = isl_space_free(space);
	eq = isl_mat_insert_zero_rows(eq, 0, 1);
	eq = isl_mat_set_element_si(eq, 0, 0, 1);
	space = isl_space_from_domain(space);
	space = isl_space_add_dims(space, isl_dim_out, n);
	ma = isl_multi_aff_from_aff_mat(space, eq);
	mpa = isl_multi_pw_aff_from_multi_aff(ma);
	map = isl_multi_pw_aff_eq_map(mpa, isl_multi_pw_aff_copy(mpa));
	data->equivalent = isl_union_map_add_map(data->equivalent, map);

	isl_basic_set_free(lineality);
	return isl_stat_ok;
error:
	isl_basic_set_free(lineality);
	return isl_stat_error;
}

/* Check if the lineality space "set" is non-trivial (i.e., is not just
 * the origin or, in other words, satisfies a number of equality constraints
 * that is smaller than the dimension of the set).
 * If so, extend data->mask and data->equivalent accordingly.
 *
 * The input should not have any local variables already, but
 * isl_set_remove_divs is called to make sure it does not.
 */
static isl_stat add_lineality(__isl_take isl_set *set, void *user)
{
	struct isl_exploit_lineality_data *data = user;
	isl_basic_set *hull;
	isl_size dim;
	isl_size n_eq;

	set = isl_set_remove_divs(set);
	hull = isl_set_unshifted_simple_hull(set);
	dim = isl_basic_set_dim(hull, isl_dim_set);
	n_eq = isl_basic_set_n_equality(hull);
	if (dim < 0 || n_eq < 0)
		goto error;
	if (dim != n_eq)
		return add_non_trivial_lineality(hull, data);
	isl_basic_set_free(hull);
	return isl_stat_ok;
error:
	isl_basic_set_free(hull);
	return isl_stat_error;
}

/* Check if the difference set on intra-node schedule constraints "intra"
 * has any non-trivial lineality space.
 * If so, then extend the difference set to a difference set
 * on equivalent elements.  That is, if "intra" is
 *
 *	{ y - x : (x,y) \in V }
 *
 * and elements are equivalent if they have the same image under f,
 * then return
 *
 *	{ y' - x' : (x,y) \in V and f(x) = f(x') and f(y) = f(y') }
 *
 * or, since f is linear,
 *
 *	{ y' - x' : (x,y) \in V and f(y - x) = f(y' - x') }
 *
 * The results of the search for non-trivial lineality spaces is stored
 * in "data".
 */
static __isl_give isl_union_set *exploit_intra_lineality(
	__isl_take isl_union_set *intra,
	struct isl_exploit_lineality_data *data)
{
	isl_union_set *lineality;
	isl_union_set *uset;

	data->any_non_trivial = isl_bool_false;
	lineality = isl_union_set_copy(intra);
	lineality = isl_union_set_combined_lineality_space(lineality);
	if (isl_union_set_foreach_set(lineality, &add_lineality, data) < 0)
		data->any_non_trivial = isl_bool_error;
	isl_union_set_free(lineality);

	if (data->any_non_trivial < 0)
		return isl_union_set_free(intra);
	if (!data->any_non_trivial)
		return intra;

	uset = isl_union_set_copy(intra);
	intra = isl_union_set_subtract(intra, isl_union_set_copy(data->mask));
	uset = isl_union_set_apply(uset, isl_union_map_copy(data->equivalent));
	intra = isl_union_set_union(intra, uset);

	intra = isl_union_set_remove_divs(intra);

	return intra;
}

/* If the difference set on intra-node schedule constraints was found to have
 * any non-trivial lineality space by exploit_intra_lineality,
 * as recorded in "data", then extend the inter-node
 * schedule constraints "inter" to schedule constraints on equivalent elements.
 * That is, if "inter" is V and
 * elements are equivalent if they have the same image under f, then return
 *
 *	{ (x', y') : (x,y) \in V and f(x) = f(x') and f(y) = f(y') }
 */
static __isl_give isl_union_map *exploit_inter_lineality(
	__isl_take isl_union_map *inter,
	struct isl_exploit_lineality_data *data)
{
	isl_union_map *umap;

	if (data->any_non_trivial < 0)
		return isl_union_map_free(inter);
	if (!data->any_non_trivial)
		return inter;

	umap = isl_union_map_copy(inter);
	inter = isl_union_map_subtract_range(inter,
				isl_union_set_copy(data->mask));
	umap = isl_union_map_apply_range(umap,
				isl_union_map_copy(data->equivalent));
	inter = isl_union_map_union(inter, umap);
	umap = isl_union_map_copy(inter);
	inter = isl_union_map_subtract_domain(inter,
				isl_union_set_copy(data->mask));
	umap = isl_union_map_apply_range(isl_union_map_copy(data->equivalent),
				umap);
	inter = isl_union_map_union(inter, umap);

	inter = isl_union_map_remove_divs(inter);

	return inter;
}

/* For each (conditional) validity edge in "graph",
 * add the corresponding dependence relation using "add"
 * to a collection of dependence relations and return the result.
 * If "coincidence" is set, then coincidence edges are considered as well.
 */
static __isl_give isl_union_map *collect_validity(struct isl_sched_graph *graph,
	__isl_give isl_union_map *(*add)(__isl_take isl_union_map *umap,
		struct isl_sched_edge *edge), int coincidence)
{
	int i;
	isl_space *space;
	isl_union_map *umap;

	space = isl_space_copy(graph->node[0].space);
	umap = isl_union_map_empty(space);

	for (i = 0; i < graph->n_edge; ++i) {
		struct isl_sched_edge *edge = &graph->edge[i];

		if (!is_any_validity(edge) &&
		    (!coincidence || !is_coincidence(edge)))
			continue;

		umap = add(umap, edge);
	}

	return umap;
}

/* For each dependence relation on a (conditional) validity edge
 * from a node to itself,
 * construct the set of coefficients of valid constraints for elements
 * in that dependence relation and collect the results.
 * If "coincidence" is set, then coincidence edges are considered as well.
 *
 * In particular, for each dependence relation R, constraints
 * on coefficients (c_0, c_x) are constructed such that
 *
 *	c_0 + c_x d >= 0 for each d in delta R = { y - x | (x,y) in R }
 *
 * If the schedule_treat_coalescing option is set, then some constraints
 * that could be exploited to construct coalescing schedules
 * are removed before the dual is computed, but after the parameters
 * have been projected out.
 * The entire computation is essentially the same as that performed
 * by intra_coefficients, except that it operates on multiple
 * edges together and that the parameters are always projected out.
 *
 * Additionally, exploit any non-trivial lineality space
 * in the difference set after removing coalescing constraints and
 * store the results of the non-trivial lineality space detection in "data".
 * The procedure is currently run unconditionally, but it is unlikely
 * to find any non-trivial lineality spaces if no coalescing constraints
 * have been removed.
 *
 * Note that if a dependence relation is a union of basic maps,
 * then each basic map needs to be treated individually as it may only
 * be possible to carry the dependences expressed by some of those
 * basic maps and not all of them.
 * The collected validity constraints are therefore not coalesced and
 * it is assumed that they are not coalesced automatically.
 * Duplicate basic maps can be removed, however.
 * In particular, if the same basic map appears as a disjunct
 * in multiple edges, then it only needs to be carried once.
 */
static __isl_give isl_basic_set_list *collect_intra_validity(isl_ctx *ctx,
	struct isl_sched_graph *graph, int coincidence,
	struct isl_exploit_lineality_data *data)
{
	isl_union_map *intra;
	isl_union_set *delta;
	isl_basic_set_list *list;

	intra = collect_validity(graph, &add_intra, coincidence);
	delta = isl_union_map_deltas(intra);
	delta = isl_union_set_project_out_all_params(delta);
	delta = isl_union_set_remove_divs(delta);
	if (isl_options_get_schedule_treat_coalescing(ctx))
		delta = union_drop_coalescing_constraints(ctx, graph, delta);
	delta = exploit_intra_lineality(delta, data);
	list = isl_union_set_get_basic_set_list(delta);
	isl_union_set_free(delta);

	return isl_basic_set_list_coefficients(list);
}

/* For each dependence relation on a (conditional) validity edge
 * from a node to some other node,
 * construct the set of coefficients of valid constraints for elements
 * in that dependence relation and collect the results.
 * If "coincidence" is set, then coincidence edges are considered as well.
 *
 * In particular, for each dependence relation R, constraints
 * on coefficients (c_0, c_n, c_x, c_y) are constructed such that
 *
 *	c_0 + c_n n + c_x x + c_y y >= 0 for each (x,y) in R
 *
 * This computation is essentially the same as that performed
 * by inter_coefficients, except that it operates on multiple
 * edges together.
 *
 * Additionally, exploit any non-trivial lineality space
 * that may have been discovered by collect_intra_validity
 * (as stored in "data").
 *
 * Note that if a dependence relation is a union of basic maps,
 * then each basic map needs to be treated individually as it may only
 * be possible to carry the dependences expressed by some of those
 * basic maps and not all of them.
 * The collected validity constraints are therefore not coalesced and
 * it is assumed that they are not coalesced automatically.
 * Duplicate basic maps can be removed, however.
 * In particular, if the same basic map appears as a disjunct
 * in multiple edges, then it only needs to be carried once.
 */
static __isl_give isl_basic_set_list *collect_inter_validity(
	struct isl_sched_graph *graph, int coincidence,
	struct isl_exploit_lineality_data *data)
{
	isl_union_map *inter;
	isl_union_set *wrap;
	isl_basic_set_list *list;

	inter = collect_validity(graph, &add_inter, coincidence);
	inter = exploit_inter_lineality(inter, data);
	inter = isl_union_map_remove_divs(inter);
	wrap = isl_union_map_wrap(inter);
	list = isl_union_set_get_basic_set_list(wrap);
	isl_union_set_free(wrap);
	return isl_basic_set_list_coefficients(list);
}

/* Construct an LP problem for finding schedule coefficients
 * such that the schedule carries as many of the "n_edge" groups of
 * dependences as possible based on the corresponding coefficient
 * constraints and return the lexicographically smallest non-trivial solution.
 * "intra" is the sequence of coefficient constraints for intra-node edges.
 * "inter" is the sequence of coefficient constraints for inter-node edges.
 * If "want_integral" is set, then compute an integral solution
 * for the coefficients rather than using the numerators
 * of a rational solution.
 * "carry_inter" indicates whether inter-node edges should be carried or
 * only respected.
 *
 * If none of the "n_edge" groups can be carried
 * then return an empty vector.
 */
static __isl_give isl_vec *compute_carrying_sol_coef(isl_ctx *ctx,
	struct isl_sched_graph *graph, int n_edge,
	__isl_keep isl_basic_set_list *intra,
	__isl_keep isl_basic_set_list *inter, int want_integral,
	int carry_inter)
{
	isl_basic_set *lp;

	if (setup_carry_lp(ctx, graph, n_edge, intra, inter, carry_inter) < 0)
		return NULL;

	lp = isl_basic_set_copy(graph->lp);
	return non_neg_lexmin(graph, lp, n_edge, want_integral);
}

/* Construct an LP problem for finding schedule coefficients
 * such that the schedule carries as many of the validity dependences
 * as possible and
 * return the lexicographically smallest non-trivial solution.
 * If "fallback" is set, then the carrying is performed as a fallback
 * for the Pluto-like scheduler.
 * If "coincidence" is set, then try and carry coincidence edges as well.
 *
 * The variable "n_edge" stores the number of groups that should be carried.
 * If none of the "n_edge" groups can be carried
 * then return an empty vector.
 * If, moreover, "n_edge" is zero, then the LP problem does not even
 * need to be constructed.
 *
 * If a fallback solution is being computed, then compute an integral solution
 * for the coefficients rather than using the numerators
 * of a rational solution.
 *
 * If a fallback solution is being computed, if there are any intra-node
 * dependences, and if requested by the user, then first try
 * to only carry those intra-node dependences.
 * If this fails to carry any dependences, then try again
 * with the inter-node dependences included.
 */
static __isl_give isl_vec *compute_carrying_sol(isl_ctx *ctx,
	struct isl_sched_graph *graph, int fallback, int coincidence)
{
	isl_size n_intra, n_inter;
	int n_edge;
	struct isl_carry carry = { 0 };
	isl_vec *sol;

	carry.intra = collect_intra_validity(ctx, graph, coincidence,
						&carry.lineality);
	carry.inter = collect_inter_validity(graph, coincidence,
						&carry.lineality);
	n_intra = isl_basic_set_list_n_basic_set(carry.intra);
	n_inter = isl_basic_set_list_n_basic_set(carry.inter);
	if (n_intra < 0 || n_inter < 0)
		goto error;

	if (fallback && n_intra > 0 &&
	    isl_options_get_schedule_carry_self_first(ctx)) {
		sol = compute_carrying_sol_coef(ctx, graph, n_intra,
				carry.intra, carry.inter, fallback, 0);
		if (!sol || sol->size != 0 || n_inter == 0) {
			isl_carry_clear(&carry);
			return sol;
		}
		isl_vec_free(sol);
	}

	n_edge = n_intra + n_inter;
	if (n_edge == 0) {
		isl_carry_clear(&carry);
		return isl_vec_alloc(ctx, 0);
	}

	sol = compute_carrying_sol_coef(ctx, graph, n_edge,
				carry.intra, carry.inter, fallback, 1);
	isl_carry_clear(&carry);
	return sol;
error:
	isl_carry_clear(&carry);
	return NULL;
}

/* Construct a schedule row for each node such that as many validity dependences
 * as possible are carried and then continue with the next band.
 * If "fallback" is set, then the carrying is performed as a fallback
 * for the Pluto-like scheduler.
 * If "coincidence" is set, then try and carry coincidence edges as well.
 *
 * If there are no validity dependences, then no dependence can be carried and
 * the procedure is guaranteed to fail.  If there is more than one component,
 * then try computing a schedule on each component separately
 * to prevent or at least postpone this failure.
 *
 * If a schedule row is computed, then check that dependences are carried
 * for at least one of the edges.
 *
 * If the computed schedule row turns out to be trivial on one or
 * more nodes where it should not be trivial, then we throw it away
 * and try again on each component separately.
 *
 * If there is only one component, then we accept the schedule row anyway,
 * but we do not consider it as a complete row and therefore do not
 * increment graph->n_row.  Note that the ranks of the nodes that
 * do get a non-trivial schedule part will get updated regardless and
 * graph->maxvar is computed based on these ranks.  The test for
 * whether more schedule rows are required in compute_schedule_wcc
 * is therefore not affected.
 *
 * Insert a band corresponding to the schedule row at position "node"
 * of the schedule tree and continue with the construction of the schedule.
 * This insertion and the continued construction is performed by split_scaled
 * after optionally checking for non-trivial common divisors.
 */
static __isl_give isl_schedule_node *carry(__isl_take isl_schedule_node *node,
	struct isl_sched_graph *graph, int fallback, int coincidence)
{
	int trivial;
	isl_ctx *ctx;
	isl_vec *sol;

	if (!node)
		return NULL;

	ctx = isl_schedule_node_get_ctx(node);
	sol = compute_carrying_sol(ctx, graph, fallback, coincidence);
	if (!sol)
		return isl_schedule_node_free(node);
	if (sol->size == 0) {
		isl_vec_free(sol);
		if (graph->scc > 1)
			return compute_component_schedule(node, graph, 1);
		isl_die(ctx, isl_error_unknown, "unable to carry dependences",
			return isl_schedule_node_free(node));
	}

	trivial = is_any_trivial(graph, sol);
	if (trivial < 0) {
		sol = isl_vec_free(sol);
	} else if (trivial && graph->scc > 1) {
		isl_vec_free(sol);
		return compute_component_schedule(node, graph, 1);
	}

	if (update_schedule(graph, sol, 0) < 0)
		return isl_schedule_node_free(node);
	if (trivial)
		graph->n_row--;

	return split_scaled(node, graph);
}

/* Construct a schedule row for each node such that as many validity dependences
 * as possible are carried and then continue with the next band.
 * Do so as a fallback for the Pluto-like scheduler.
 * If "coincidence" is set, then try and carry coincidence edges as well.
 */
static __isl_give isl_schedule_node *carry_fallback(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph,
	int coincidence)
{
	return carry(node, graph, 1, coincidence);
}

/* Construct a schedule row for each node such that as many validity dependences
 * as possible are carried and then continue with the next band.
 * Do so for the case where the Feautrier scheduler was selected
 * by the user.
 */
static __isl_give isl_schedule_node *carry_feautrier(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph)
{
	return carry(node, graph, 0, 0);
}

/* Construct a schedule row for each node such that as many validity dependences
 * as possible are carried and then continue with the next band.
 * Do so as a fallback for the Pluto-like scheduler.
 */
static __isl_give isl_schedule_node *carry_dependences(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph)
{
	return carry_fallback(node, graph, 0);
}

/* Construct a schedule row for each node such that as many validity or
 * coincidence dependences as possible are carried and
 * then continue with the next band.
 * Do so as a fallback for the Pluto-like scheduler.
 */
static __isl_give isl_schedule_node *carry_coincidence(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph)
{
	return carry_fallback(node, graph, 1);
}

/* Topologically sort statements mapped to the same schedule iteration
 * and add insert a sequence node in front of "node"
 * corresponding to this order.
 * If "initialized" is set, then it may be assumed that
 * isl_sched_graph_compute_maxvar
 * has been called on the current band.  Otherwise, call
 * isl_sched_graph_compute_maxvar if and before carry_dependences gets called.
 *
 * If it turns out to be impossible to sort the statements apart,
 * because different dependences impose different orderings
 * on the statements, then we extend the schedule such that
 * it carries at least one more dependence.
 */
static __isl_give isl_schedule_node *sort_statements(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph,
	int initialized)
{
	isl_ctx *ctx;
	isl_union_set_list *filters;

	if (!node)
		return NULL;

	ctx = isl_schedule_node_get_ctx(node);
	if (graph->n < 1)
		isl_die(ctx, isl_error_internal,
			"graph should have at least one node",
			return isl_schedule_node_free(node));

	if (graph->n == 1)
		return node;

	if (update_edges(ctx, graph) < 0)
		return isl_schedule_node_free(node);

	if (graph->n_edge == 0)
		return node;

	if (detect_sccs(ctx, graph) < 0)
		return isl_schedule_node_free(node);

	next_band(graph);
	if (graph->scc < graph->n) {
		if (!initialized && isl_sched_graph_compute_maxvar(graph) < 0)
			return isl_schedule_node_free(node);
		return carry_dependences(node, graph);
	}

	filters = isl_sched_graph_extract_sccs(ctx, graph);
	node = isl_schedule_node_insert_sequence(node, filters);

	return node;
}

/* Are there any (non-empty) (conditional) validity edges in the graph?
 */
static int has_validity_edges(struct isl_sched_graph *graph)
{
	int i;

	for (i = 0; i < graph->n_edge; ++i) {
		int empty;

		empty = isl_map_plain_is_empty(graph->edge[i].map);
		if (empty < 0)
			return -1;
		if (empty)
			continue;
		if (is_any_validity(&graph->edge[i]))
			return 1;
	}

	return 0;
}

/* Should we apply a Feautrier step?
 * That is, did the user request the Feautrier algorithm and are
 * there any validity dependences (left)?
 */
static int need_feautrier_step(isl_ctx *ctx, struct isl_sched_graph *graph)
{
	if (ctx->opt->schedule_algorithm != ISL_SCHEDULE_ALGORITHM_FEAUTRIER)
		return 0;

	return has_validity_edges(graph);
}

/* Compute a schedule for a connected dependence graph using Feautrier's
 * multi-dimensional scheduling algorithm and return the updated schedule node.
 *
 * The original algorithm is described in [1].
 * The main idea is to minimize the number of scheduling dimensions, by
 * trying to satisfy as many dependences as possible per scheduling dimension.
 *
 * [1] P. Feautrier, Some Efficient Solutions to the Affine Scheduling
 *     Problem, Part II: Multi-Dimensional Time.
 *     In Intl. Journal of Parallel Programming, 1992.
 */
static __isl_give isl_schedule_node *compute_schedule_wcc_feautrier(
	isl_schedule_node *node, struct isl_sched_graph *graph)
{
	return carry_feautrier(node, graph);
}

/* Turn off the "local" bit on all (condition) edges.
 */
static void clear_local_edges(struct isl_sched_graph *graph)
{
	int i;

	for (i = 0; i < graph->n_edge; ++i)
		if (isl_sched_edge_is_condition(&graph->edge[i]))
			clear_local(&graph->edge[i]);
}

/* Does "graph" have both condition and conditional validity edges?
 */
static int need_condition_check(struct isl_sched_graph *graph)
{
	int i;
	int any_condition = 0;
	int any_conditional_validity = 0;

	for (i = 0; i < graph->n_edge; ++i) {
		if (isl_sched_edge_is_condition(&graph->edge[i]))
			any_condition = 1;
		if (isl_sched_edge_is_conditional_validity(&graph->edge[i]))
			any_conditional_validity = 1;
	}

	return any_condition && any_conditional_validity;
}

/* Does "graph" contain any coincidence edge?
 */
static int has_any_coincidence(struct isl_sched_graph *graph)
{
	int i;

	for (i = 0; i < graph->n_edge; ++i)
		if (is_coincidence(&graph->edge[i]))
			return 1;

	return 0;
}

/* Extract the final schedule row as a map with the iteration domain
 * of "node" as domain.
 */
static __isl_give isl_map *final_row(struct isl_sched_node *node)
{
	isl_multi_aff *ma;
	isl_size n_row;

	n_row = isl_mat_rows(node->sched);
	if (n_row < 0)
		return NULL;
	ma = isl_sched_node_extract_partial_schedule_multi_aff(node,
								n_row - 1, 1);
	return isl_map_from_multi_aff(ma);
}

/* Is the conditional validity dependence in the edge with index "edge_index"
 * violated by the latest (i.e., final) row of the schedule?
 * That is, is i scheduled after j
 * for any conditional validity dependence i -> j?
 */
static int is_violated(struct isl_sched_graph *graph, int edge_index)
{
	isl_map *src_sched, *dst_sched, *map;
	struct isl_sched_edge *edge = &graph->edge[edge_index];
	int empty;

	src_sched = final_row(edge->src);
	dst_sched = final_row(edge->dst);
	map = isl_map_copy(edge->map);
	map = isl_map_apply_domain(map, src_sched);
	map = isl_map_apply_range(map, dst_sched);
	map = isl_map_order_gt(map, isl_dim_in, 0, isl_dim_out, 0);
	empty = isl_map_is_empty(map);
	isl_map_free(map);

	if (empty < 0)
		return -1;

	return !empty;
}

/* Does "graph" have any satisfied condition edges that
 * are adjacent to the conditional validity constraint with
 * domain "conditional_source" and range "conditional_sink"?
 *
 * A satisfied condition is one that is not local.
 * If a condition was forced to be local already (i.e., marked as local)
 * then there is no need to check if it is in fact local.
 *
 * Additionally, mark all adjacent condition edges found as local.
 */
static int has_adjacent_true_conditions(struct isl_sched_graph *graph,
	__isl_keep isl_union_set *conditional_source,
	__isl_keep isl_union_set *conditional_sink)
{
	int i;
	int any = 0;

	for (i = 0; i < graph->n_edge; ++i) {
		int adjacent, local;
		isl_union_map *condition;

		if (!isl_sched_edge_is_condition(&graph->edge[i]))
			continue;
		if (is_local(&graph->edge[i]))
			continue;

		condition = graph->edge[i].tagged_condition;
		adjacent = domain_intersects(condition, conditional_sink);
		if (adjacent >= 0 && !adjacent)
			adjacent = range_intersects(condition,
							conditional_source);
		if (adjacent < 0)
			return -1;
		if (!adjacent)
			continue;

		set_local(&graph->edge[i]);

		local = is_condition_false(&graph->edge[i]);
		if (local < 0)
			return -1;
		if (!local)
			any = 1;
	}

	return any;
}

/* Are there any violated conditional validity dependences with
 * adjacent condition dependences that are not local with respect
 * to the current schedule?
 * That is, is the conditional validity constraint violated?
 *
 * Additionally, mark all those adjacent condition dependences as local.
 * We also mark those adjacent condition dependences that were not marked
 * as local before, but just happened to be local already.  This ensures
 * that they remain local if the schedule is recomputed.
 *
 * We first collect domain and range of all violated conditional validity
 * dependences and then check if there are any adjacent non-local
 * condition dependences.
 */
static int has_violated_conditional_constraint(isl_ctx *ctx,
	struct isl_sched_graph *graph)
{
	int i;
	int any = 0;
	isl_union_set *source, *sink;

	source = isl_union_set_empty(isl_space_params_alloc(ctx, 0));
	sink = isl_union_set_empty(isl_space_params_alloc(ctx, 0));
	for (i = 0; i < graph->n_edge; ++i) {
		isl_union_set *uset;
		isl_union_map *umap;
		int violated;

		if (!isl_sched_edge_is_conditional_validity(&graph->edge[i]))
			continue;

		violated = is_violated(graph, i);
		if (violated < 0)
			goto error;
		if (!violated)
			continue;

		any = 1;

		umap = isl_union_map_copy(graph->edge[i].tagged_validity);
		uset = isl_union_map_domain(umap);
		source = isl_union_set_union(source, uset);
		source = isl_union_set_coalesce(source);

		umap = isl_union_map_copy(graph->edge[i].tagged_validity);
		uset = isl_union_map_range(umap);
		sink = isl_union_set_union(sink, uset);
		sink = isl_union_set_coalesce(sink);
	}

	if (any)
		any = has_adjacent_true_conditions(graph, source, sink);

	isl_union_set_free(source);
	isl_union_set_free(sink);
	return any;
error:
	isl_union_set_free(source);
	isl_union_set_free(sink);
	return -1;
}

/* Examine the current band (the rows between graph->band_start and
 * graph->n_total_row), deciding whether to drop it or add it to "node"
 * and then continue with the computation of the next band, if any.
 * If "initialized" is set, then it may be assumed that
 * isl_sched_graph_compute_maxvar
 * has been called on the current band.  Otherwise, call
 * isl_sched_graph_compute_maxvar if and before carry_dependences gets called.
 *
 * The caller keeps looking for a new row as long as
 * graph->n_row < graph->maxvar.  If the latest attempt to find
 * such a row failed (i.e., we still have graph->n_row < graph->maxvar),
 * then we either
 * - split between SCCs and start over (assuming we found an interesting
 *	pair of SCCs between which to split)
 * - continue with the next band (assuming the current band has at least
 *	one row)
 * - if there is more than one SCC left, then split along all SCCs
 * - if outer coincidence needs to be enforced, then try to carry as many
 *	validity or coincidence dependences as possible and
 *	continue with the next band
 * - try to carry as many validity dependences as possible and
 *	continue with the next band
 * In each case, we first insert a band node in the schedule tree
 * if any rows have been computed.
 *
 * If the caller managed to complete the schedule and the current band
 * is empty, then finish off by topologically
 * sorting the statements based on the remaining dependences.
 * If, on the other hand, the current band has at least one row,
 * then continue with the next band.  Note that this next band
 * will necessarily be empty, but the graph may still be split up
 * into weakly connected components before arriving back here.
 */
__isl_give isl_schedule_node *isl_schedule_node_compute_finish_band(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph,
	int initialized)
{
	int empty;

	if (!node)
		return NULL;

	empty = graph->n_total_row == graph->band_start;
	if (graph->n_row < graph->maxvar) {
		isl_ctx *ctx;

		ctx = isl_schedule_node_get_ctx(node);
		if (!ctx->opt->schedule_maximize_band_depth && !empty)
			return compute_next_band(node, graph, 1);
		if (graph->src_scc >= 0)
			return compute_split_schedule(node, graph);
		if (!empty)
			return compute_next_band(node, graph, 1);
		if (graph->scc > 1)
			return compute_component_schedule(node, graph, 1);
		if (!initialized && isl_sched_graph_compute_maxvar(graph) < 0)
			return isl_schedule_node_free(node);
		if (isl_options_get_schedule_outer_coincidence(ctx))
			return carry_coincidence(node, graph);
		return carry_dependences(node, graph);
	}

	if (!empty)
		return compute_next_band(node, graph, 1);
	return sort_statements(node, graph, initialized);
}

/* Construct a band of schedule rows for a connected dependence graph.
 * The caller is responsible for determining the strongly connected
 * components and calling isl_sched_graph_compute_maxvar first.
 *
 * We try to find a sequence of as many schedule rows as possible that result
 * in non-negative dependence distances (independent of the previous rows
 * in the sequence, i.e., such that the sequence is tilable), with as
 * many of the initial rows as possible satisfying the coincidence constraints.
 * The computation stops if we can't find any more rows or if we have found
 * all the rows we wanted to find.
 *
 * If ctx->opt->schedule_outer_coincidence is set, then we force the
 * outermost dimension to satisfy the coincidence constraints.  If this
 * turns out to be impossible, we fall back on the general scheme above
 * and try to carry as many dependences as possible.
 *
 * If "graph" contains both condition and conditional validity dependences,
 * then we need to check that that the conditional schedule constraint
 * is satisfied, i.e., there are no violated conditional validity dependences
 * that are adjacent to any non-local condition dependences.
 * If there are, then we mark all those adjacent condition dependences
 * as local and recompute the current band.  Those dependences that
 * are marked local will then be forced to be local.
 * The initial computation is performed with no dependences marked as local.
 * If we are lucky, then there will be no violated conditional validity
 * dependences adjacent to any non-local condition dependences.
 * Otherwise, we mark some additional condition dependences as local and
 * recompute.  We continue this process until there are no violations left or
 * until we are no longer able to compute a schedule.
 * Since there are only a finite number of dependences,
 * there will only be a finite number of iterations.
 */
isl_stat isl_schedule_node_compute_wcc_band(isl_ctx *ctx,
	struct isl_sched_graph *graph)
{
	int has_coincidence;
	int use_coincidence;
	int force_coincidence = 0;
	int check_conditional;

	if (sort_sccs(graph) < 0)
		return isl_stat_error;

	clear_local_edges(graph);
	check_conditional = need_condition_check(graph);
	has_coincidence = has_any_coincidence(graph);

	if (ctx->opt->schedule_outer_coincidence)
		force_coincidence = 1;

	use_coincidence = has_coincidence;
	while (graph->n_row < graph->maxvar) {
		isl_vec *sol;
		int violated;
		int coincident;

		graph->src_scc = -1;
		graph->dst_scc = -1;

		if (setup_lp(ctx, graph, use_coincidence) < 0)
			return isl_stat_error;
		sol = solve_lp(ctx, graph);
		if (!sol)
			return isl_stat_error;
		if (sol->size == 0) {
			int empty = graph->n_total_row == graph->band_start;

			isl_vec_free(sol);
			if (use_coincidence && (!force_coincidence || !empty)) {
				use_coincidence = 0;
				continue;
			}
			return isl_stat_ok;
		}
		coincident = !has_coincidence || use_coincidence;
		if (update_schedule(graph, sol, coincident) < 0)
			return isl_stat_error;

		if (!check_conditional)
			continue;
		violated = has_violated_conditional_constraint(ctx, graph);
		if (violated < 0)
			return isl_stat_error;
		if (!violated)
			continue;
		if (reset_band(graph) < 0)
			return isl_stat_error;
		use_coincidence = has_coincidence;
	}

	return isl_stat_ok;
}

/* Compute a schedule for a connected dependence graph by considering
 * the graph as a whole and return the updated schedule node.
 *
 * The actual schedule rows of the current band are computed by
 * isl_schedule_node_compute_wcc_band.  isl_schedule_node_compute_finish_band
 * takes care of integrating the band into "node" and continuing
 * the computation.
 */
static __isl_give isl_schedule_node *compute_schedule_wcc_whole(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph)
{
	isl_ctx *ctx;

	if (!node)
		return NULL;

	ctx = isl_schedule_node_get_ctx(node);
	if (isl_schedule_node_compute_wcc_band(ctx, graph) < 0)
		return isl_schedule_node_free(node);

	return isl_schedule_node_compute_finish_band(node, graph, 1);
}

/* Compute a schedule for a connected dependence graph and return
 * the updated schedule node.
 *
 * If Feautrier's algorithm is selected, we first recursively try to satisfy
 * as many validity dependences as possible. When all validity dependences
 * are satisfied we extend the schedule to a full-dimensional schedule.
 *
 * Call compute_schedule_wcc_whole or isl_schedule_node_compute_wcc_clustering
 * depending on whether the user has selected the option to try and
 * compute a schedule for the entire (weakly connected) component first.
 * If there is only a single strongly connected component (SCC), then
 * there is no point in trying to combine SCCs
 * in isl_schedule_node_compute_wcc_clustering, so compute_schedule_wcc_whole
 * is called instead.
 */
static __isl_give isl_schedule_node *compute_schedule_wcc(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph)
{
	isl_ctx *ctx;

	if (!node)
		return NULL;

	ctx = isl_schedule_node_get_ctx(node);
	if (detect_sccs(ctx, graph) < 0)
		return isl_schedule_node_free(node);

	if (isl_sched_graph_compute_maxvar(graph) < 0)
		return isl_schedule_node_free(node);

	if (need_feautrier_step(ctx, graph))
		return compute_schedule_wcc_feautrier(node, graph);

	if (graph->scc <= 1 || isl_options_get_schedule_whole_component(ctx))
		return compute_schedule_wcc_whole(node, graph);
	else
		return isl_schedule_node_compute_wcc_clustering(node, graph);
}

/* Compute a schedule for each group of nodes identified by node->scc
 * separately and then combine them in a sequence node (or as set node
 * if graph->weak is set) inserted at position "node" of the schedule tree.
 * Return the updated schedule node.
 *
 * If "wcc" is set then each of the groups belongs to a single
 * weakly connected component in the dependence graph so that
 * there is no need for compute_sub_schedule to look for weakly
 * connected components.
 *
 * If a set node would be introduced and if the number of components
 * is equal to the number of nodes, then check if the schedule
 * is already complete.  If so, a redundant set node would be introduced
 * (without any further descendants) stating that the statements
 * can be executed in arbitrary order, which is also expressed
 * by the absence of any node.  Refrain from inserting any nodes
 * in this case and simply return.
 */
static __isl_give isl_schedule_node *compute_component_schedule(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph,
	int wcc)
{
	int component;
	isl_ctx *ctx;
	isl_union_set_list *filters;

	if (!node)
		return NULL;

	if (graph->weak && graph->scc == graph->n) {
		if (isl_sched_graph_compute_maxvar(graph) < 0)
			return isl_schedule_node_free(node);
		if (graph->n_row >= graph->maxvar)
			return node;
	}

	ctx = isl_schedule_node_get_ctx(node);
	filters = isl_sched_graph_extract_sccs(ctx, graph);
	if (graph->weak)
		node = isl_schedule_node_insert_set(node, filters);
	else
		node = isl_schedule_node_insert_sequence(node, filters);

	for (component = 0; component < graph->scc; ++component) {
		node = isl_schedule_node_grandchild(node, component, 0);
		node = compute_sub_schedule(node, ctx, graph,
				    &isl_sched_node_scc_exactly,
				    &isl_sched_edge_scc_exactly,
				    component, wcc);
		node = isl_schedule_node_grandparent(node);
	}

	return node;
}

/* Compute a schedule for the given dependence graph and insert it at "node".
 * Return the updated schedule node.
 *
 * We first check if the graph is connected (through validity and conditional
 * validity dependences) and, if not, compute a schedule
 * for each component separately.
 * If the schedule_serialize_sccs option is set, then we check for strongly
 * connected components instead and compute a separate schedule for
 * each such strongly connected component.
 */
static __isl_give isl_schedule_node *compute_schedule(isl_schedule_node *node,
	struct isl_sched_graph *graph)
{
	isl_ctx *ctx;

	if (!node)
		return NULL;

	ctx = isl_schedule_node_get_ctx(node);
	if (isl_options_get_schedule_serialize_sccs(ctx)) {
		if (detect_sccs(ctx, graph) < 0)
			return isl_schedule_node_free(node);
	} else {
		if (detect_wccs(ctx, graph) < 0)
			return isl_schedule_node_free(node);
	}

	if (graph->scc > 1)
		return compute_component_schedule(node, graph, 1);

	return compute_schedule_wcc(node, graph);
}

/* Compute a schedule on sc->domain that respects the given schedule
 * constraints.
 *
 * In particular, the schedule respects all the validity dependences.
 * If the default isl scheduling algorithm is used, it tries to minimize
 * the dependence distances over the proximity dependences.
 * If Feautrier's scheduling algorithm is used, the proximity dependence
 * distances are only minimized during the extension to a full-dimensional
 * schedule.
 *
 * If there are any condition and conditional validity dependences,
 * then the conditional validity dependences may be violated inside
 * a tilable band, provided they have no adjacent non-local
 * condition dependences.
 */
__isl_give isl_schedule *isl_schedule_constraints_compute_schedule(
	__isl_take isl_schedule_constraints *sc)
{
	isl_ctx *ctx = isl_schedule_constraints_get_ctx(sc);
	struct isl_sched_graph graph = { 0 };
	isl_schedule *sched;
	isl_schedule_node *node;
	isl_union_set *domain;
	isl_size n;

	sc = isl_schedule_constraints_align_params(sc);

	domain = isl_schedule_constraints_get_domain(sc);
	n = isl_union_set_n_set(domain);
	if (n == 0) {
		isl_schedule_constraints_free(sc);
		return isl_schedule_from_domain(domain);
	}

	if (n < 0 || isl_sched_graph_init(&graph, sc) < 0)
		domain = isl_union_set_free(domain);

	node = isl_schedule_node_from_domain(domain);
	node = isl_schedule_node_child(node, 0);
	if (graph.n > 0)
		node = compute_schedule(node, &graph);
	sched = isl_schedule_node_get_schedule(node);
	isl_schedule_node_free(node);

	isl_sched_graph_free(ctx, &graph);
	isl_schedule_constraints_free(sc);

	return sched;
}

/* Compute a schedule for the given union of domains that respects
 * all the validity dependences and minimizes
 * the dependence distances over the proximity dependences.
 *
 * This function is kept for backward compatibility.
 */
__isl_give isl_schedule *isl_union_set_compute_schedule(
	__isl_take isl_union_set *domain,
	__isl_take isl_union_map *validity,
	__isl_take isl_union_map *proximity)
{
	isl_schedule_constraints *sc;

	sc = isl_schedule_constraints_on_domain(domain);
	sc = isl_schedule_constraints_set_validity(sc, validity);
	sc = isl_schedule_constraints_set_proximity(sc, proximity);

	return isl_schedule_constraints_compute_schedule(sc);
}
