/*
 * Copyright 2021      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege
 */

#include <stdio.h>

#include <isl/ctx.h>
#include <isl/schedule_node.h>
#include <isl/union_set.h>

#include "isl_hash_private.h"
#include "isl_scheduler_scc.h"
#include "isl_sort.h"

/* Internal data structure for ordering the SCCs of "graph",
 * where each SCC i consists of the single cluster determined
 * by c->scc_cluster[i].  The nodes in this cluster all have
 * their "scc" field set to i.
 *
 * "graph" is the original schedule graph.
 * "c" contains the clustering information.
 *
 * "n" is the number of SCCs in the isl_scc_graph, which may be
 * a subset of those in "graph".
 * "graph_scc" maps the local index of an SCC in this isl_scc_graph
 * to the corresponding index in "graph", i.e, the index of c->scc_cluster.
 * The entries of "graph_scc" are kept in topological order.
 *
 * "component" contains the component to which an SCC belongs,
 * where the component is represented by the index of the first SCC
 * in the component.
 * The index of this first SCC is always smaller than or equal
 * to the index of the SCC itself.
 * This field is initialized by isl_scc_graph_init_component and
 * used by detect_components.
 * During construction, "component" may also contain the index
 * of some other SCC in the component, but then it is necessarily
 * smaller than the index of the current SCC and the first SCC
 * can be reached by recursively looking up "component".
 * "size" contains the number of elements in the components
 * indexed by a component sequence number.
 *
 * "pos" is used locally inside isl_scc_graph_sort_components
 * to store the position of the next SCC within a component.
 * It is also used inside isl_scc_graph_sub to map
 * the position in the original graph to the position in the subgraph.
 *
 * "sorted" contains the (possibly) reordered local indices,
 * sorted per component.  Within each component, the original
 * topological order is preserved.
 *
 * "edge_table" contains "n" edge tables, one for each SCC
 * in this isl_scc_graph.  Each table contains the local indices
 * of the SCCs that depend on this SCC.  These local indices
 * are encoded as pointers to the corresponding entry in "graph_scc".
 * The value stored at that location is the global SCC index.
 * "reverse_edge_table" contains the inverse edges.
 */
struct isl_scc_graph {
	isl_ctx *ctx;
	struct isl_sched_graph *graph;
	struct isl_clustering *c;

	int n;
	int *graph_scc;
	int *component;
	int *size;
	int *pos;
	int *sorted;
	struct isl_hash_table **edge_table;
	struct isl_hash_table **reverse_edge_table;
};

/* The source SCC of a collection of edges.
 *
 * "scc_graph" is the SCC graph containing the edges.
 * "src" is the local index of the source SCC.
 */
struct isl_edge_src {
	struct isl_scc_graph *scc_graph;
	int src;
};

/* isl_hash_table_foreach callback for printing an edge
 * between "src" and the node identified by "entry".
 * The edge is printed in terms of the global SCC indices.
 */
static isl_stat print_edge(void **entry, void *user)
{
	int *dst = *entry;
	int *src = user;

	fprintf(stderr, "%d -> %d; ", *src, *dst);

	return isl_stat_ok;
}

/* Print some debugging information about "scc_graph".
 *
 * In particular, print the nodes and the edges (both forward and backward).
 */
void isl_scc_graph_dump(struct isl_scc_graph *scc_graph)
{
	int i;
	isl_ctx *ctx;

	if (!scc_graph)
		return;

	ctx = scc_graph->ctx;
	for (i = 0; i < scc_graph->n; ++i) {
		if (i)
			fprintf(stderr, ", ");
		fprintf(stderr, "%d", scc_graph->graph_scc[i]);
	}
	fprintf(stderr, "\n");
	for (i = 0; i < scc_graph->n; ++i) {
		isl_hash_table_foreach(ctx, scc_graph->edge_table[i],
			&print_edge, &scc_graph->graph_scc[i]);
	}
	fprintf(stderr, "\n");
	for (i = 0; i < scc_graph->n; ++i) {
		isl_hash_table_foreach(ctx, scc_graph->reverse_edge_table[i],
			&print_edge, &scc_graph->graph_scc[i]);
	}
	fprintf(stderr, "\n");
}

/* Free all memory allocated for "scc_graph" and return NULL.
 */
struct isl_scc_graph *isl_scc_graph_free(struct isl_scc_graph *scc_graph)
{
	int i;
	isl_ctx *ctx;

	if (!scc_graph)
		return NULL;

	ctx = scc_graph->ctx;
	if (scc_graph->edge_table) {
		for (i = 0; i < scc_graph->n; ++i)
			isl_hash_table_free(ctx, scc_graph->edge_table[i]);
	}
	if (scc_graph->reverse_edge_table) {
		for (i = 0; i < scc_graph->n; ++i)
			isl_hash_table_free(ctx,
					    scc_graph->reverse_edge_table[i]);
	}

	free(scc_graph->graph_scc);
	free(scc_graph->component);
	free(scc_graph->size);
	free(scc_graph->pos);
	free(scc_graph->sorted);
	free(scc_graph->edge_table);
	free(scc_graph->reverse_edge_table);
	isl_ctx_deref(scc_graph->ctx);
	free(scc_graph);
	return NULL;
}

/* Return an encoding of the local SCC index "pos" in "scc_graph"
 * as a pointer.
 * In particular, return a pointer to the corresponding entry
 * in scc_graph->graph_scc.
 */
static void *isl_scc_graph_encode_local_index(struct isl_scc_graph *scc_graph,
	int pos)
{
	return &scc_graph->graph_scc[pos];
}

/* Return the local SCC index in "scc_graph" corresponding
 * to the "data" encoding in the edge table.
 */
static int isl_scc_graph_local_index(struct isl_scc_graph *scc_graph, int *data)
{
	return data - &scc_graph->graph_scc[0];
}

/* isl_hash_table_find callback to check whether the given entry
 * refers to an SCC encoded as "val".
 */
static isl_bool is_scc_node(const void *entry, const void *val)
{
	return entry == val;
}

/* Return the edge from local SCC index "src" to local SCC index "dst"
 * in "edge_table" of "scc_graph", creating one if "reserve" is set.
 * If "reserve" is not set, then return isl_hash_table_entry_none
 * if there is no such edge.
 *
 * The destination of the edge is encoded as a pointer
 * to the corresponding entry in scc_graph->graph_scc.
 */
struct isl_hash_table_entry *isl_scc_graph_find_edge(
	struct isl_scc_graph *scc_graph, struct isl_hash_table **edge_table,
	int src, int dst, int reserve)
{
	isl_ctx *ctx;
	uint32_t hash;
	void *val;

	ctx = scc_graph->ctx;
	hash = isl_hash_builtin(isl_hash_init(), dst);
	val = isl_scc_graph_encode_local_index(scc_graph, dst);
	return isl_hash_table_find(ctx, edge_table[src], hash,
					&is_scc_node, val, reserve);
}

/* Remove the edge between the SCCs with local indices "src" and
 * "dst" in "scc_graph", if it exits.
 * Return isl_bool_true if this is the case.
 *
 * The edge is only removed from scc_graph->edge_table.
 * scc_graph->reverse_edge_table is assumed to be empty
 * when this function is called.
 */
static isl_bool isl_scc_graph_remove_edge(struct isl_scc_graph *scc_graph,
	int src, int dst)
{
	isl_ctx *ctx;
	struct isl_hash_table_entry *edge_entry;

	edge_entry = isl_scc_graph_find_edge(scc_graph, scc_graph->edge_table,
						src, dst, 0);
	if (edge_entry == isl_hash_table_entry_none)
		return isl_bool_false;
	if (!edge_entry)
		return isl_bool_error;

	ctx = scc_graph->ctx;
	isl_hash_table_remove(ctx, scc_graph->edge_table[src], edge_entry);

	return isl_bool_true;
}

/* Internal data structure used by next_nodes.
 *
 * "scc_graph" is the SCC graph.
 * "next" collects the next nodes.
 * "n" is the number of next nodes already collected.
 */
struct isl_extract_dst_data {
	struct isl_scc_graph *scc_graph;
	int *next;
	int n;
};

/* Given an entry in the edge table, add the corresponding
 * target local SCC index to data->next.
 */
static isl_stat extract_dst(void **entry, void *user)
{
	int *dst = *entry;
	struct isl_extract_dst_data *data = user;

	data->next[data->n++] = isl_scc_graph_local_index(data->scc_graph, dst);

	return isl_stat_ok;
}

/* isl_sort callback for sorting integers in increasing order.
 */
static int cmp_int(const void *a, const void *b, void *data)
{
	const int *i1 = a;
	const int *i2 = b;

	return *i1 - *i2;
}

/* Return the local indices of the SCCs in "scc_graph"
 * for which there is an edge from the SCC with local index "i".
 * The indices are returned in increasing order,
 * i.e., in the original topological order.
 */
static int *next_nodes(struct isl_scc_graph *scc_graph, int i)
{
	struct isl_extract_dst_data data;
	int n_next;
	int *next;

	n_next = scc_graph->edge_table[i]->n;
	next = isl_alloc_array(scc_graph->ctx, int, n_next);
	if (!next)
		return NULL;
	data.scc_graph = scc_graph;
	data.next = next;
	data.n = 0;
	if (isl_hash_table_foreach(scc_graph->ctx, scc_graph->edge_table[i],
			&extract_dst, &data) < 0)
		goto error;
	if (isl_sort(next, n_next, sizeof(int), &cmp_int, NULL) < 0)
		goto error;
	return next;
error:
	free(next);
	return NULL;
}

/* Internal data structure for foreach_reachable.
 *
 * "scc_graph" is the SCC graph being visited.
 * "fn" is the function that needs to be called on each reachable node.
 * "user" is the user argument to "fn".
 */
struct isl_foreach_reachable_data {
	struct isl_scc_graph *scc_graph;
	isl_bool (*fn)(int pos, void *user);
	void *user;
};

static isl_stat foreach_reachable(struct isl_foreach_reachable_data *data,
	int pos);

/* isl_hash_table_foreach callback for calling data->fn on each SCC
 * reachable from the SCC encoded in "entry",
 * continuing from an SCC as long as data->fn returns isl_bool_true.
 */
static isl_stat recurse_foreach_reachable(void **entry, void *user)
{
	struct isl_foreach_reachable_data *data = user;
	int pos;
	isl_bool more;

	pos = isl_scc_graph_local_index(data->scc_graph, *entry);
	more = data->fn(pos, data->user);
	if (more < 0)
		return isl_stat_error;
	if (!more)
		return isl_stat_ok;

	return foreach_reachable(data, pos);
}

/* Call data->fn on each SCC reachable from the SCC with local index "pos",
 * continuing from an SCC as long as data->fn returns isl_bool_true.
 *
 * Handle chains directly and recurse when an SCC has more than one
 * outgoing edge.
 */
static isl_stat foreach_reachable(struct isl_foreach_reachable_data *data,
	int pos)
{
	isl_ctx *ctx;
	struct isl_hash_table **edge_table = data->scc_graph->edge_table;

	while (edge_table[pos]->n == 1) {
		struct isl_hash_table_entry *entry;
		isl_bool more;

		entry = isl_hash_table_first(edge_table[pos]);
		pos = isl_scc_graph_local_index(data->scc_graph, entry->data);
		more = data->fn(pos, data->user);
		if (more < 0)
			return isl_stat_error;
		if (!more)
			return isl_stat_ok;
	}

	if (edge_table[pos]->n == 0)
		return isl_stat_ok;

	ctx = data->scc_graph->ctx;
	return isl_hash_table_foreach(ctx, edge_table[pos],
					&recurse_foreach_reachable, data);
}

/* If there is an edge from data->src to "pos", then remove it.
 * Return isl_bool_true if descendants of "pos" still need to be considered.
 *
 * Descendants only need to be considered if no edge is removed.
 */
static isl_bool elim_or_next(int pos, void *user)
{
	struct isl_edge_src *data = user;
	struct isl_scc_graph *scc_graph = data->scc_graph;
	isl_bool removed;

	removed = isl_scc_graph_remove_edge(scc_graph, data->src, pos);
	return isl_bool_not(removed);
}

/* Remove transitive edges from "scc_graph".
 *
 * Consider the SCC nodes "i" in reverse topological order.
 * If there is more than one edge emanating from a node,
 * then eliminate the edges to those nodes that can also be reached
 * through an edge to a node with a smaller index.
 * In particular, consider all but the last next nodes "next[j]"
 * in reverse topological order.  If any node "k" can be reached
 * from such a node for which there is also an edge from "i"
 * then this edge can be removed because this node can also
 * be reached from "i" through the edge to "next[j]".
 * If such an edge is removed, then any further descendant of "k"
 * does not need to be considered since these were already considered
 * for a previous "next[j]" equal to "k", or "k" is the last next node,
 * in which case there is no further node with an edge from "i".
 */
static struct isl_scc_graph *isl_scc_graph_reduce(
	struct isl_scc_graph *scc_graph)
{
	struct isl_edge_src elim_data;
	struct isl_foreach_reachable_data data = {
		.scc_graph = scc_graph,
		.fn = &elim_or_next,
		.user = &elim_data,
	};
	int i, j;

	elim_data.scc_graph = scc_graph;
	for (i = scc_graph->n - 3; i >= 0; --i) {
		int *next;
		int n_next;

		n_next = scc_graph->edge_table[i]->n;
		if (n_next <= 1)
			continue;
		next = next_nodes(scc_graph, i);
		if (!next)
			return isl_scc_graph_free(scc_graph);

		elim_data.src = i;
		for (j = n_next - 2; j >= 0; --j)
			if (foreach_reachable(&data, next[j]) < 0)
				break;
		free(next);
		if (j >= 0)
			return isl_scc_graph_free(scc_graph);
	}

	return scc_graph;
}

/* Add an edge to "edge_table" between the SCCs with local indices "src" and
 * "dst" in "scc_graph".
 *
 * If the edge already appeared in the table, then it is simply overwritten
 * with the same information.
 */
static isl_stat isl_scc_graph_add_edge(struct isl_scc_graph *scc_graph,
	struct isl_hash_table **edge_table, int src, int dst)
{
	struct isl_hash_table_entry *edge_entry;

	edge_entry =
		isl_scc_graph_find_edge(scc_graph, edge_table, src, dst, 1);
	if (!edge_entry)
		return isl_stat_error;
	edge_entry->data = &scc_graph->graph_scc[dst];

	return isl_stat_ok;
}

/* Add an edge from "dst" to data->src
 * to data->scc_graph->reverse_edge_table.
 */
static isl_stat add_reverse(void **entry, void *user)
{
	struct isl_edge_src *data = user;
	int dst;

	dst = isl_scc_graph_local_index(data->scc_graph, *entry);
	return isl_scc_graph_add_edge(data->scc_graph,
			data->scc_graph->reverse_edge_table, dst, data->src);
}

/* Add an (inverse) edge to scc_graph->reverse_edge_table
 * for each edge in scc_graph->edge_table.
 */
static struct isl_scc_graph *isl_scc_graph_add_reverse_edges(
	struct isl_scc_graph *scc_graph)
{
	struct isl_edge_src data;
	isl_ctx *ctx;

	if (!scc_graph)
		return NULL;

	ctx = scc_graph->ctx;
	data.scc_graph = scc_graph;
	for (data.src = 0; data.src < scc_graph->n; ++data.src) {
		if (isl_hash_table_foreach(ctx, scc_graph->edge_table[data.src],
				&add_reverse, &data) < 0)
			return isl_scc_graph_free(scc_graph);
	}
	return scc_graph;
}

/* Given an edge in the schedule graph, add an edge between
 * the corresponding SCCs in "scc_graph", if they are distinct.
 *
 * This function is used to create edges in the original isl_scc_graph.
 * where the local SCC indices are equal to the corresponding global
 * indices.
 */
static isl_stat add_scc_edge(void **entry, void *user)
{
	struct isl_sched_edge *edge = *entry;
	struct isl_scc_graph *scc_graph = user;
	int src = edge->src->scc;
	int dst = edge->dst->scc;

	if (src == dst)
		return isl_stat_ok;

	return isl_scc_graph_add_edge(scc_graph, scc_graph->edge_table,
					src, dst);
}

/* Allocate an isl_scc_graph for ordering "n" SCCs of "graph"
 * with clustering information in "c".
 *
 * The caller still needs to fill in the edges.
 */
static struct isl_scc_graph *isl_scc_graph_alloc(isl_ctx *ctx, int n,
	struct isl_sched_graph *graph, struct isl_clustering *c)
{
	int i;
	struct isl_scc_graph *scc_graph;

	scc_graph = isl_alloc_type(ctx, struct isl_scc_graph);
	if (!scc_graph)
		return NULL;

	scc_graph->ctx = ctx;
	isl_ctx_ref(ctx);
	scc_graph->graph = graph;
	scc_graph->c = c;

	scc_graph->n = n;
	scc_graph->graph_scc = isl_alloc_array(ctx, int, n);
	scc_graph->component = isl_alloc_array(ctx, int, n);
	scc_graph->size = isl_alloc_array(ctx, int, n);
	scc_graph->pos = isl_alloc_array(ctx, int, n);
	scc_graph->sorted = isl_alloc_array(ctx, int, n);
	scc_graph->edge_table =
		isl_calloc_array(ctx, struct isl_hash_table *, n);
	scc_graph->reverse_edge_table =
		isl_calloc_array(ctx, struct isl_hash_table *, n);
	if (!scc_graph->graph_scc || !scc_graph->component ||
	    !scc_graph->size || !scc_graph->pos || !scc_graph->sorted ||
	    !scc_graph->edge_table || !scc_graph->reverse_edge_table)
		return isl_scc_graph_free(scc_graph);

	for (i = 0; i < n; ++i) {
		scc_graph->edge_table[i] = isl_hash_table_alloc(ctx, 2);
		scc_graph->reverse_edge_table[i] = isl_hash_table_alloc(ctx, 2);
		if (!scc_graph->edge_table[i] ||
		    !scc_graph->reverse_edge_table[i])
			return isl_scc_graph_free(scc_graph);
	}

	return scc_graph;
}

/* Construct an isl_scc_graph for ordering the SCCs of "graph",
 * where each SCC i consists of the single cluster determined
 * by c->scc_cluster[i].  The nodes in this cluster all have
 * their "scc" field set to i.
 *
 * The initial isl_scc_graph has as many SCCs as "graph" and
 * their local indices are the same as their indices in "graph".
 *
 * Add edges between different SCCs for each (conditional) validity edge
 * between nodes in those SCCs, remove transitive edges and
 * construct the inverse edges from the remaining forward edges.
 */
struct isl_scc_graph *isl_scc_graph_from_sched_graph(isl_ctx *ctx,
	struct isl_sched_graph *graph, struct isl_clustering *c)
{
	int i;
	struct isl_scc_graph *scc_graph;

	scc_graph = isl_scc_graph_alloc(ctx, graph->scc, graph, c);
	if (!scc_graph)
		return NULL;

	for (i = 0; i < graph->scc; ++i)
		scc_graph->graph_scc[i] = i;

	if (isl_hash_table_foreach(ctx, graph->edge_table[isl_edge_validity],
					&add_scc_edge, scc_graph) < 0)
		return isl_scc_graph_free(scc_graph);
	if (isl_hash_table_foreach(ctx,
			    graph->edge_table[isl_edge_conditional_validity],
			    &add_scc_edge, scc_graph) < 0)
		return isl_scc_graph_free(scc_graph);

	scc_graph = isl_scc_graph_reduce(scc_graph);
	scc_graph = isl_scc_graph_add_reverse_edges(scc_graph);

	return scc_graph;
}

/* Internal data structure for copy_edge.
 *
 * "scc_graph" is the original graph.
 * "sub" is the subgraph to which edges are being copied.
 * "src" is the local index in "scc_graph" of the source of the edges
 * currently being copied.
 */
struct isl_copy_edge_data {
	struct isl_scc_graph *scc_graph;
	struct isl_scc_graph *sub;
	int src;
};

/* isl_hash_table_foreach callback for copying the edge
 * from data->src to the node identified by "entry"
 * to data->sub, provided the two nodes belong to the same component.
 * Note that by construction, there are no edges between different components
 * in the region handled by detect_components, but there may
 * be edges to nodes outside this region.
 * The components therefore need to be initialized for all nodes
 * in isl_scc_graph_init_component.
 */
static isl_stat copy_edge(void **entry, void *user)
{
	struct isl_copy_edge_data *data = user;
	struct isl_scc_graph *scc_graph = data->scc_graph;
	struct isl_scc_graph *sub = data->sub;
	int dst, sub_dst, sub_src;

	dst = isl_scc_graph_local_index(data->scc_graph, *entry);
	if (scc_graph->component[dst] != scc_graph->component[data->src])
		return isl_stat_ok;

	sub_src = scc_graph->pos[data->src];
	sub_dst = scc_graph->pos[dst];

	return isl_scc_graph_add_edge(sub, sub->edge_table, sub_src, sub_dst);
}

/* Construct a subgraph of "scc_graph" for the components
 * consisting of the "n" SCCs with local indices in "pos".
 * These SCCs have the same value in scc_graph->component and
 * this value is different from that of any other SCC.
 *
 * The forward edges with source and destination in the component
 * are copied from "scc_graph".
 * The local index in the subgraph corresponding to a local index
 * in "scc_graph" is stored in scc_graph->pos for use by copy_edge().
 * The inverse edges are constructed directly from the forward edges.
 */
static struct isl_scc_graph *isl_scc_graph_sub(struct isl_scc_graph *scc_graph,
	int *pos, int n)
{
	int i;
	isl_ctx *ctx;
	struct isl_scc_graph *sub;
	struct isl_copy_edge_data data;

	if (!scc_graph)
		return NULL;

	ctx = scc_graph->ctx;
	sub = isl_scc_graph_alloc(ctx, n, scc_graph->graph, scc_graph->c);
	if (!sub)
		return sub;

	for (i = 0; i < n; ++i)
		sub->graph_scc[i] = scc_graph->graph_scc[pos[i]];

	for (i = 0; i < n; ++i)
		scc_graph->pos[pos[i]] = i;

	data.scc_graph = scc_graph;
	data.sub = sub;
	for (i = 0; i < n; ++i) {
		data.src = pos[i];
		if (isl_hash_table_foreach(ctx, scc_graph->edge_table[pos[i]],
				&copy_edge, &data) < 0)
			return isl_scc_graph_free(sub);
	}

	sub = isl_scc_graph_add_reverse_edges(sub);

	return sub;
}

/* Return a union of universe domains corresponding to the nodes
 * in the SCC with local index "pos".
 */
static __isl_give isl_union_set *isl_scc_graph_extract_local_scc(
	struct isl_scc_graph *scc_graph, int pos)
{
	return isl_sched_graph_extract_scc(scc_graph->ctx, scc_graph->graph,
					scc_graph->graph_scc[pos]);
}

/* Construct a filter corresponding to a sequence of "n" local SCC indices
 * determined by successive calls to "el",
 * add this filter to "list" and
 * return the result.
 */
static __isl_give isl_union_set_list *add_scc_seq(
	struct isl_scc_graph *scc_graph,
	int (*el)(int i, void *user), void *user, int n,
	__isl_take isl_union_set_list *list)
{
	int i;
	isl_union_set *dom;

	dom = isl_union_set_empty_ctx(scc_graph->ctx);
	for (i = 0; i < n; ++i)
		dom = isl_union_set_union(dom,
		    isl_scc_graph_extract_local_scc(scc_graph, el(i, user)));

	return isl_union_set_list_add(list, dom);
}

/* add_scc_seq callback that, on successive calls, returns a sequence
 * of local SCC indices starting at "first".
 */
static int offset(int i, void *user)
{
	int *first = user;

	return *first + i;
}

/* Construct a filter corresponding to a sequence of "n" local SCC indices
 * starting at "first", add this filter to "list" and return the result.
 */
static __isl_give isl_union_set_list *isl_scc_graph_add_scc_seq(
	struct isl_scc_graph *scc_graph, int first, int n,
	__isl_take isl_union_set_list *list)
{
	return add_scc_seq(scc_graph, &offset, &first, n, list);
}

/* add_scc_seq callback that, on successive calls, returns the sequence
 * of local SCC indices in "seq".
 */
static int at(int i, void *user)
{
	int *seq = user;

	return seq[i];
}

/* Construct a filter corresponding to the sequence of "n" local SCC indices
 * stored in "seq", add this filter to "list" and return the result.
 */
static __isl_give isl_union_set_list *isl_scc_graph_add_scc_indirect_seq(
	struct isl_scc_graph *scc_graph, int *seq, int n,
	__isl_take isl_union_set_list *list)
{
	return add_scc_seq(scc_graph, &at, seq, n, list);
}

/* Extract out a list of filters for a sequence node that splits
 * the graph along the SCC with local index "pos".
 *
 * The list contains (at most) three elements,
 * the SCCs before "pos" (in the topological order),
 * "pos" itself, and
 * the SCCs after "pos".
 */
static __isl_give isl_union_set_list *extract_split_scc(
	struct isl_scc_graph *scc_graph, int pos)
{
	isl_union_set *dom;
	isl_union_set_list *filters;

	filters = isl_union_set_list_alloc(scc_graph->ctx, 3);
	if (pos > 0)
		filters = isl_scc_graph_add_scc_seq(scc_graph, 0, pos, filters);
	dom = isl_scc_graph_extract_local_scc(scc_graph, pos);
	filters = isl_union_set_list_add(filters, dom);
	if (pos + 1 < scc_graph->n)
		filters = isl_scc_graph_add_scc_seq(scc_graph,
				pos + 1, scc_graph->n - (pos + 1), filters);
	return filters;
}

/* Call isl_schedule_node_compute_finish_band on the cluster
 * corresponding to the SCC with local index "pos".
 *
 * First obtain the corresponding SCC index in scc_graph->graph and
 * then obtain the corresponding cluster.
 */
static __isl_give isl_schedule_node *isl_scc_graph_finish_band(
	struct isl_scc_graph *scc_graph, __isl_take isl_schedule_node *node,
	int pos)
{
	struct isl_clustering *c = scc_graph->c;
	int cluster;

	cluster = c->scc_cluster[scc_graph->graph_scc[pos]];
	return isl_schedule_node_compute_finish_band(node,
						&c->cluster[cluster], 0);
}

/* Given that the SCCs in "scc_graph" form a chain,
 * call isl_schedule_node_compute_finish_band on each of the clusters
 * in scc_graph->c and update "node" to arrange for them to be executed
 * in topological order.
 */
static __isl_give isl_schedule_node *isl_scc_graph_chain(
	struct isl_scc_graph *scc_graph, __isl_take isl_schedule_node *node)
{
	int i;
	isl_union_set *dom;
	isl_union_set_list *filters;

	filters = isl_union_set_list_alloc(scc_graph->ctx, scc_graph->n);
	for (i = 0; i < scc_graph->n; ++i) {
		dom = isl_scc_graph_extract_local_scc(scc_graph, i);
		filters = isl_union_set_list_add(filters, dom);
	}

	node = isl_schedule_node_insert_sequence(node, filters);

	for (i = 0; i < scc_graph->n; ++i) {
		node = isl_schedule_node_grandchild(node, i, 0);
		node = isl_scc_graph_finish_band(scc_graph, node, i);
		node = isl_schedule_node_grandparent(node);
	}

	return node;
}

/* Recursively call isl_scc_graph_decompose on a subgraph
 * consisting of the "n" SCCs with local indices in "pos".
 *
 * If this component contains only a single SCC,
 * then there is no need for a further recursion and
 * isl_schedule_node_compute_finish_band can be called directly.
 */
static __isl_give isl_schedule_node *recurse(struct isl_scc_graph *scc_graph,
	int *pos, int n, __isl_take isl_schedule_node *node)
{
	struct isl_scc_graph *sub;

	if (n == 1)
		return isl_scc_graph_finish_band(scc_graph, node, pos[0]);

	sub = isl_scc_graph_sub(scc_graph, pos, n);
	if (!sub)
		return isl_schedule_node_free(node);
	node = isl_scc_graph_decompose(sub, node);
	isl_scc_graph_free(sub);

	return node;
}

/* Initialize the component field of "scc_graph".
 * Initially, each SCC belongs to its own single-element component.
 *
 * Note that the SCC on which isl_scc_graph_decompose performs a split
 * also needs to be assigned a component because the components
 * are also used in copy_edge to extract a subgraph.
 */
static void isl_scc_graph_init_component(struct isl_scc_graph *scc_graph)
{
	int i;

	for (i = 0; i < scc_graph->n; ++i)
		scc_graph->component[i] = i;
}

/* Set the component of "a" to be the same as that of "b" and
 * return the original component of "a".
 */
static int assign(int *component, int a, int b)
{
	int t;

	t = component[a];
	component[a] = component[b];
	return t;
}

/* Merge the components containing the SCCs with indices "a" and "b".
 *
 * If "a" and "b" already belong to the same component, then nothing
 * needs to be done.
 * Otherwise, make sure both point to the same component.
 * In particular, use the SCC in the component entries with the smallest index.
 * If the other SCC was the first of its component then the entire
 * component now (eventually) points to the other component.
 * Otherwise, the earlier parts of the component still need
 * to be merged with the other component.
 *
 * At each stage, either a or b is replaced by either a or b itself,
 * in which case the merging terminates because a and b already
 * point to the same component, or an SCC index with a smaller value.
 * This ensures the merging terminates at some point.
 */
static void isl_scc_graph_merge_src_dst(struct isl_scc_graph *scc_graph,
	int a, int b)
{
	int *component = scc_graph->component;

	while (component[a] != component[b]) {
		if (component[a] < component[b])
			b = assign(component, b, a);
		else
			a = assign(component, a, b);
	}
}

/* Internal data structure for isl_scc_graph_merge_components.
 *
 * "scc_graph" is the SCC graph containing the edges.
 * "src" is the local index of the source SCC.
 * "end" is the local index beyond the sequence being considered.
 */
struct isl_merge_src_dst_data {
	struct isl_scc_graph *scc_graph;
	int src;
	int end;
};

/* isl_hash_table_foreach callback for merging the components
 * of data->src and the node represented by "entry", provided
 * it is within the sequence being considered.
 */
static isl_stat merge_src_dst(void **entry, void *user)
{
	struct isl_merge_src_dst_data *data = user;
	int dst;

	dst = isl_scc_graph_local_index(data->scc_graph, *entry);
	if (dst >= data->end)
		return isl_stat_ok;

	isl_scc_graph_merge_src_dst(data->scc_graph, data->src, dst);

	return isl_stat_ok;
}

/* Merge components of the "n" SCCs starting at "first" that are connected
 * by an edge.
 */
static isl_stat isl_scc_graph_merge_components(struct isl_scc_graph *scc_graph,
	int first, int n)
{
	int i;
	struct isl_merge_src_dst_data data;
	isl_ctx *ctx = scc_graph->ctx;

	data.scc_graph = scc_graph;
	data.end = first + n;
	for (i = 0; i < n; ++i) {
		data.src = first + i;
		if (isl_hash_table_foreach(ctx, scc_graph->edge_table[data.src],
				&merge_src_dst, &data) < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}

/* Sort the "n" local SCC indices starting at "first" according
 * to component, store them in scc_graph->sorted and
 * return the number of components.
 * The sizes of the components are stored in scc_graph->size.
 * Only positions starting at "first" are used within
 * scc_graph->sorted and scc_graph->size.
 *
 * The representation of the components is first normalized.
 * The normalization ensures that each SCC in a component
 * points to the first SCC in the component, whereas
 * before this function is called, some SCCs may only point
 * to some other SCC in the component with a smaller index.
 *
 * Internally, the sizes of the components are first stored
 * at indices corresponding to the first SCC in the component.
 * They are subsequently moved into consecutive positions
 * while reordering the local indices.
 * This reordering is performed by first determining the position
 * of the first SCC in each component and
 * then putting the "n" local indices in the right position
 * according to the component, preserving the topological order
 * within each component.
 */
static int isl_scc_graph_sort_components(struct isl_scc_graph *scc_graph,
	int first, int n)
{
	int i, j;
	int sum;
	int *component = scc_graph->component;
	int *size = scc_graph->size;
	int *pos = scc_graph->pos;
	int *sorted = scc_graph->sorted;
	int n_component;

	n_component = 0;
	for (i = 0; i < n; ++i) {
		size[first + i] = 0;
		if (component[first + i] == first + i)
			n_component++;
		else
			component[first + i] = component[component[first + i]];
		size[component[first + i]]++;
	}

	sum = first;
	i = 0;
	for (j = 0; j < n_component; ++j) {
		while (size[first + i] == 0)
			++i;
		pos[first + i] = sum;
		sum += size[first + i];
		size[first + j] = size[first + i++];
	}
	for (i = 0; i < n; ++i)
		sorted[pos[component[first + i]]++] = first + i;

	return n_component;
}

/* Extract out a list of filters for a set node that splits up
 * the graph into "n_component" components.
 * "first" is the initial position in "scc_graph" where information
 * about the components is stored.
 * In particular, the first "n_component" entries of scc_graph->size
 * at this position contain the number of SCCs in each component.
 * The entries of scc_graph->sorted starting at "first"
 * contain the local indices of the SCC in those components.
 */
static __isl_give isl_union_set_list *extract_components(
	struct isl_scc_graph *scc_graph, int first, int n_component)
{
	int i;
	int sum;
	int *size = scc_graph->size;
	int *sorted = scc_graph->sorted;
	isl_ctx *ctx = scc_graph->ctx;
	isl_union_set_list *filters;

	filters = isl_union_set_list_alloc(ctx, n_component);
	sum = first;
	for (i = 0; i < n_component; ++i) {
		int n;

		n = size[first + i];
		filters = isl_scc_graph_add_scc_indirect_seq(scc_graph,
			&sorted[sum], n, filters);
		sum += n;
	}

	return filters;
}

/* Detect components in the subgraph consisting of the "n" SCCs
 * with local index starting at "first" and further decompose them,
 * calling isl_schedule_node_compute_finish_band on each
 * of the corresponding clusters.
 *
 * If there is only one SCC, then isl_schedule_node_compute_finish_band
 * can be called directly.
 * Otherwise, determine the components and rearrange the local indices
 * according to component, but preserving the topological order within
 * each component, in scc_graph->sorted.  The sizes of the components
 * are stored in scc_graph->size.
 * If there is only one component, it can be further decomposed
 * directly by a call to recurse().
 * Otherwise, introduce a set node separating the components and
 * call recurse() on each component separately.
 */
static __isl_give isl_schedule_node *detect_components(
	struct isl_scc_graph *scc_graph, int first, int n,
	__isl_take isl_schedule_node *node)
{
	int i;
	int *size = scc_graph->size;
	int *sorted = scc_graph->sorted;
	int n_component;
	int sum;
	isl_union_set_list *filters;

	if (n == 1)
		return isl_scc_graph_finish_band(scc_graph, node, first);

	if (isl_scc_graph_merge_components(scc_graph, first, n) < 0)
		return isl_schedule_node_free(node);

	n_component = isl_scc_graph_sort_components(scc_graph, first, n);
	if (n_component == 1)
		return recurse(scc_graph, &sorted[first], n, node);

	filters = extract_components(scc_graph, first, n_component);
	node = isl_schedule_node_insert_set(node, filters);

	sum = first;
	for (i = 0; i < n_component; ++i) {
		int n;

		n = size[first + i];
		node = isl_schedule_node_grandchild(node, i, 0);
		node = recurse(scc_graph, &sorted[sum], n, node);
		node = isl_schedule_node_grandparent(node);
		sum += n;
	}

	return node;
}

/* Given a sequence node "node", where the filter at position "child"
 * represents the "n" SCCs with local index starting at "first",
 * detect components in this subgraph and further decompose them,
 * calling isl_schedule_node_compute_finish_band on each
 * of the corresponding clusters.
 */
static __isl_give isl_schedule_node *detect_components_at(
	struct isl_scc_graph *scc_graph, int first, int n,
	__isl_take isl_schedule_node *node, int child)
{
	node = isl_schedule_node_grandchild(node, child, 0);
	node = detect_components(scc_graph, first, n, node);
	node = isl_schedule_node_grandparent(node);

	return node;
}

/* Return the local index of an SCC on which to split "scc_graph".
 * Return scc_graph->n if no suitable split SCC can be found.
 *
 * In particular, look for an SCC that is involved in the largest number
 * of edges.  Splitting the graph on such an SCC has the highest chance
 * of exposing independent SCCs in the remaining part(s).
 * There is no point in splitting a chain of nodes,
 * so return scc_graph->n if the entire graph forms a chain.
 */
static int best_split(struct isl_scc_graph *scc_graph)
{
	int i;
	int split = scc_graph->n;
	int split_score = -1;

	for (i = 0; i < scc_graph->n; ++i) {
		int n_fwd, n_bwd;

		n_fwd = scc_graph->edge_table[i]->n;
		n_bwd = scc_graph->reverse_edge_table[i]->n;
		if (n_fwd <= 1 && n_bwd <= 1)
			continue;
		if (split_score >= n_fwd + n_bwd)
			continue;
		split = i;
		split_score = n_fwd + n_bwd;
	}

	return split;
}

/* Call isl_schedule_node_compute_finish_band on each of the clusters
 * in scc_graph->c and update "node" to arrange for them to be executed
 * in an order possibly involving set nodes that generalizes
 * the topological order determined by the scc fields of the nodes
 * in scc_graph->graph.
 *
 * First try and find a suitable SCC on which to split the graph.
 * If no such SCC can be found then the graph forms a chain and
 * it is handled as such.
 * Otherwise, break up the graph into (at most) three parts,
 * the SCCs before the selected SCC (in the topological order),
 * the selected SCC itself, and
 * the SCCs after the selected SCC.
 * The first and last part (if they exist) are decomposed recursively and
 * the three parts are combined in a sequence.
 *
 * Since the outermost node of the recursive pieces may also be a sequence,
 * these potential sequence nodes are spliced into the top-level sequence node.
 */
__isl_give isl_schedule_node *isl_scc_graph_decompose(
	struct isl_scc_graph *scc_graph, __isl_take isl_schedule_node *node)
{
	int i;
	int split;
	isl_union_set_list *filters;

	if (!scc_graph)
		return isl_schedule_node_free(node);

	split = best_split(scc_graph);

	if (split == scc_graph->n)
		return isl_scc_graph_chain(scc_graph, node);

	filters = extract_split_scc(scc_graph, split);
	node = isl_schedule_node_insert_sequence(node, filters);

	isl_scc_graph_init_component(scc_graph);

	i = 0;
	if (split > 0)
		node = detect_components_at(scc_graph, 0, split, node, i++);
	node = isl_schedule_node_grandchild(node, i++, 0);
	node = isl_scc_graph_finish_band(scc_graph, node, split);
	node = isl_schedule_node_grandparent(node);
	if (split + 1 < scc_graph->n)
		node = detect_components_at(scc_graph,
			    split + 1, scc_graph->n - (split + 1), node, i++);

	node = isl_schedule_node_sequence_splice_children(node);

	return node;
}
