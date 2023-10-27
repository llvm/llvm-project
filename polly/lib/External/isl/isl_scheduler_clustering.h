#ifndef ISL_SCHEDULER_CLUSTERING_H
#define ISL_SCHEDULER_CLUSTERING_H

#include "isl_scheduler.h"

/* Clustering information used by isl_schedule_node_compute_wcc_clustering.
 *
 * "n" is the number of SCCs in the original dependence graph
 * "scc" is an array of "n" elements, each representing an SCC
 * of the original dependence graph.  All entries in the same cluster
 * have the same number of schedule rows.
 * "scc_cluster" maps each SCC index to the cluster to which it belongs,
 * where each cluster is represented by the index of the first SCC
 * in the cluster.  Initially, each SCC belongs to a cluster containing
 * only that SCC.
 *
 * "scc_in_merge" is used by merge_clusters_along_edge to keep
 * track of which SCCs need to be merged.
 *
 * "cluster" contains the merged clusters of SCCs after the clustering
 * has completed.
 *
 * "scc_node" is a temporary data structure used inside copy_partial.
 * For each SCC, it keeps track of the number of nodes in the SCC
 * that have already been copied.
 */
struct isl_clustering {
	int n;
	struct isl_sched_graph *scc;
	struct isl_sched_graph *cluster;
	int *scc_cluster;
	int *scc_node;
	int *scc_in_merge;
};

__isl_give isl_schedule_node *isl_schedule_node_compute_wcc_clustering(
	__isl_take isl_schedule_node *node, struct isl_sched_graph *graph);

#endif
