#ifndef ISL_SCHEDULER_SCC_H
#define ISL_SCHEDULER_SCC_H

#include <isl/ctx.h>

#include "isl_scheduler.h"
#include "isl_scheduler_clustering.h"

struct isl_scc_graph;

struct isl_scc_graph *isl_scc_graph_from_sched_graph(isl_ctx *ctx,
	struct isl_sched_graph *graph, struct isl_clustering *c);
__isl_give isl_schedule_node *isl_scc_graph_decompose(
	struct isl_scc_graph *scc_graph, __isl_take isl_schedule_node *node);
struct isl_scc_graph *isl_scc_graph_free(struct isl_scc_graph *scc_graph);

void isl_scc_graph_dump(struct isl_scc_graph *scc_graph);

#endif
