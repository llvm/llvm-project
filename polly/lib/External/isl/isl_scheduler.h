#ifndef ISL_SCHEDULER_H
#define ISL_SCHEDULER_H

#include <isl/aff_type.h>
#include <isl/hash.h>
#include <isl/id_type.h>
#include <isl/map_type.h>
#include <isl/map_to_basic_set.h>
#include <isl/mat.h>
#include <isl/space_type.h>
#include <isl/set_type.h>
#include <isl/val_type.h>
#include <isl/vec.h>
#include <isl/union_map_type.h>

#include "isl_schedule_constraints.h"
#include "isl_tab.h"

/* Internal information about a node that is used during the construction
 * of a schedule.
 * space represents the original space in which the domain lives;
 *	that is, the space is not affected by compression
 * sched is a matrix representation of the schedule being constructed
 *	for this node; if compressed is set, then this schedule is
 *	defined over the compressed domain space
 * sched_map is an isl_map representation of the same (partial) schedule
 *	sched_map may be NULL; if compressed is set, then this map
 *	is defined over the uncompressed domain space
 * rank is the number of linearly independent rows in the linear part
 *	of sched
 * the rows of "vmap" represent a change of basis for the node
 *	variables; the first rank rows span the linear part of
 *	the schedule rows; the remaining rows are linearly independent
 * the rows of "indep" represent linear combinations of the schedule
 * coefficients that are non-zero when the schedule coefficients are
 * linearly independent of previously computed schedule rows.
 * start is the first variable in the LP problem in the sequences that
 *	represents the schedule coefficients of this node
 * nvar is the dimension of the (compressed) domain
 * nparam is the number of parameters or 0 if we are not constructing
 *	a parametric schedule
 *
 * If compressed is set, then hull represents the constraints
 * that were used to derive the compression, while compress and
 * decompress map the original space to the compressed space and
 * vice versa.
 *
 * scc is the index of SCC (or WCC) this node belongs to
 *
 * "cluster" is only used inside extract_clusters and identifies
 * the cluster of SCCs that the node belongs to.
 *
 * coincident contains a boolean for each of the rows of the schedule,
 * indicating whether the corresponding scheduling dimension satisfies
 * the coincidence constraints in the sense that the corresponding
 * dependence distances are zero.
 *
 * If the schedule_treat_coalescing option is set, then
 * "sizes" contains the sizes of the (compressed) instance set
 * in each direction.  If there is no fixed size in a given direction,
 * then the corresponding size value is set to infinity.
 * If the schedule_treat_coalescing option or the schedule_max_coefficient
 * option is set, then "max" contains the maximal values for
 * schedule coefficients of the (compressed) variables.  If no bound
 * needs to be imposed on a particular variable, then the corresponding
 * value is negative.
 * If not NULL, then "bounds" contains a non-parametric set
 * in the compressed space that is bounded by the size in each direction.
 */
struct isl_sched_node {
	isl_space *space;
	int	compressed;
	isl_set	*hull;
	isl_multi_aff *compress;
	isl_pw_multi_aff *decompress;
	isl_mat *sched;
	isl_map *sched_map;
	int	 rank;
	isl_mat *indep;
	isl_mat *vmap;
	int	 start;
	int	 nvar;
	int	 nparam;

	int	 scc;
	int	 cluster;

	int	*coincident;

	isl_multi_val *sizes;
	isl_basic_set *bounds;
	isl_vec *max;
};

int isl_sched_node_scc_exactly(struct isl_sched_node *node, int scc);

isl_stat isl_sched_node_update_vmap(struct isl_sched_node *node);
__isl_give isl_multi_aff *isl_sched_node_extract_partial_schedule_multi_aff(
	struct isl_sched_node *node, int first, int n);

/* An edge in the dependence graph.  An edge may be used to
 * ensure validity of the generated schedule, to minimize the dependence
 * distance or both
 *
 * map is the dependence relation, with i -> j in the map if j depends on i
 * tagged_condition and tagged_validity contain the union of all tagged
 *	condition or conditional validity dependence relations that
 *	specialize the dependence relation "map"; that is,
 *	if (i -> a) -> (j -> b) is an element of "tagged_condition"
 *	or "tagged_validity", then i -> j is an element of "map".
 *	If these fields are NULL, then they represent the empty relation.
 * src is the source node
 * dst is the sink node
 *
 * types is a bit vector containing the types of this edge.
 * validity is set if the edge is used to ensure correctness
 * coincidence is used to enforce zero dependence distances
 * proximity is set if the edge is used to minimize dependence distances
 * condition is set if the edge represents a condition
 *	for a conditional validity schedule constraint
 * local can only be set for condition edges and indicates that
 *	the dependence distance over the edge should be zero
 * conditional_validity is set if the edge is used to conditionally
 *	ensure correctness
 *
 * For validity edges, start and end mark the sequence of inequality
 * constraints in the LP problem that encode the validity constraint
 * corresponding to this edge.
 *
 * During clustering, an edge may be marked "no_merge" if it should
 * not be used to merge clusters.
 * The weight is also only used during clustering and it is
 * an indication of how many schedule dimensions on either side
 * of the schedule constraints can be aligned.
 * If the weight is negative, then this means that this edge was postponed
 * by has_bounded_distances or any_no_merge.  The original weight can
 * be retrieved by adding 1 + graph->max_weight, with "graph"
 * the graph containing this edge.
 */
struct isl_sched_edge {
	isl_map *map;
	isl_union_map *tagged_condition;
	isl_union_map *tagged_validity;

	struct isl_sched_node *src;
	struct isl_sched_node *dst;

	unsigned types;

	int start;
	int end;

	int no_merge;
	int weight;
};

int isl_sched_edge_has_type(struct isl_sched_edge *edge,
	enum isl_edge_type type);
int isl_sched_edge_is_condition(struct isl_sched_edge *edge);
int isl_sched_edge_is_conditional_validity(struct isl_sched_edge *edge);
int isl_sched_edge_scc_exactly(struct isl_sched_edge *edge, int scc);
int isl_sched_edge_is_proximity(struct isl_sched_edge *edge);

/* Internal information about the dependence graph used during
 * the construction of the schedule.
 *
 * intra_hmap is a cache, mapping dependence relations to their dual,
 *	for dependences from a node to itself, possibly without
 *	coefficients for the parameters
 * intra_hmap_param is a cache, mapping dependence relations to their dual,
 *	for dependences from a node to itself, including coefficients
 *	for the parameters
 * inter_hmap is a cache, mapping dependence relations to their dual,
 *	for dependences between distinct nodes
 * if compression is involved then the key for these maps
 * is the original, uncompressed dependence relation, while
 * the value is the dual of the compressed dependence relation.
 *
 * n is the number of nodes
 * node is the list of nodes
 * maxvar is the maximal number of variables over all nodes
 * max_row is the allocated number of rows in the schedule
 * n_row is the current (maximal) number of linearly independent
 *	rows in the node schedules
 * n_total_row is the current number of rows in the node schedules
 * band_start is the starting row in the node schedules of the current band
 * root is set to the original dependence graph from which this graph
 *	is derived through splitting.  If this graph is not the result of
 *	splitting, then the root field points to the graph itself.
 *
 * sorted contains a list of node indices sorted according to the
 *	SCC to which a node belongs
 *
 * n_edge is the number of edges
 * edge is the list of edges
 * max_edge contains the maximal number of edges of each type;
 *	in particular, it contains the number of edges in the inital graph.
 * edge_table contains pointers into the edge array, hashed on the source
 *	and sink spaces; there is one such table for each type;
 *	a given edge may be referenced from more than one table
 *	if the corresponding relation appears in more than one of the
 *	sets of dependences; however, for each type there is only
 *	a single edge between a given pair of source and sink space
 *	in the entire graph
 *
 * node_table contains pointers into the node array, hashed on the space tuples
 *
 * region contains a list of variable sequences that should be non-trivial
 *
 * lp contains the (I)LP problem used to obtain new schedule rows
 *
 * src_scc and dst_scc are the source and sink SCCs of an edge with
 *	conflicting constraints
 *
 * scc represents the number of components
 * weak is set if the components are weakly connected
 *
 * max_weight is used during clustering and represents the maximal
 * weight of the relevant proximity edges.
 */
struct isl_sched_graph {
	isl_map_to_basic_set *intra_hmap;
	isl_map_to_basic_set *intra_hmap_param;
	isl_map_to_basic_set *inter_hmap;

	struct isl_sched_node *node;
	int n;
	int maxvar;
	int max_row;
	int n_row;

	int *sorted;

	int n_total_row;
	int band_start;

	struct isl_sched_graph *root;

	struct isl_sched_edge *edge;
	int n_edge;
	int max_edge[isl_edge_last + 1];
	struct isl_hash_table *edge_table[isl_edge_last + 1];

	struct isl_hash_table *node_table;
	struct isl_trivial_region *region;

	isl_basic_set *lp;

	int src_scc;
	int dst_scc;

	int scc;
	int weak;

	int max_weight;
};

isl_stat isl_sched_graph_init(struct isl_sched_graph *graph,
	__isl_keep isl_schedule_constraints *sc);
void isl_sched_graph_free(isl_ctx *ctx, struct isl_sched_graph *graph);

int isl_sched_graph_is_node(struct isl_sched_graph *graph,
	struct isl_sched_node *node);
isl_bool isl_sched_graph_has_validity_edge(struct isl_sched_graph *graph,
	struct isl_sched_node *src, struct isl_sched_node *dst);

struct isl_sched_node *isl_sched_graph_find_node(isl_ctx *ctx,
	struct isl_sched_graph *graph, __isl_keep isl_space *space);

isl_stat isl_sched_graph_detect_ccs(isl_ctx *ctx, struct isl_sched_graph *graph,
	isl_bool (*follows)(int i, int j, void *user));

__isl_give isl_union_set *isl_sched_graph_extract_scc(isl_ctx *ctx,
	struct isl_sched_graph *graph, int scc);
__isl_give isl_union_set_list *isl_sched_graph_extract_sccs(isl_ctx *ctx,
	struct isl_sched_graph *graph);
isl_stat isl_sched_graph_extract_sub_graph(isl_ctx *ctx,
	struct isl_sched_graph *graph,
	int (*node_pred)(struct isl_sched_node *node, int data),
	int (*edge_pred)(struct isl_sched_edge *edge, int data),
	int data, struct isl_sched_graph *sub);
isl_stat isl_sched_graph_compute_maxvar(struct isl_sched_graph *graph);
isl_stat isl_schedule_node_compute_wcc_band(isl_ctx *ctx,
	struct isl_sched_graph *graph);
__isl_give isl_schedule_node *isl_schedule_node_compute_finish_band(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph,
	int initialized);

#endif
