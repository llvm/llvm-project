//===-- OpenMP/TaskGraphPlan.h - A taskgraph's mapped data ------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Deciding where a recorded taskgraph's mapped data lives, and keeping that
// decision current across replays.  See TaskGraphDataPlanTy in
// Shared/TaskGraph.h for the representation.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_OPENMP_TASKGRAPHPLAN_H
#define OMPTARGET_OPENMP_TASKGRAPHPLAN_H

#include "Shared/TaskGraph.h"

#include "llvm/ADT/SmallVector.h"

struct DeviceTy;

namespace llvm {
namespace omp {
namespace target {

/// Build \p Graph's plan: gather every leaf's mapped host ranges into disjoint
/// ranges, place each argument in its range, and decide which maps perform the
/// implicit transfers.  Device addresses are left for the first replay to
/// resolve.  Call once, before the graph is offered to a plugin.
///
/// Returns false if the graph maps something the plan cannot express.  The
/// graph has not been claimed yet, so libomp then replays it on the host.
bool buildTaskGraphPlan(DeviceTy &Device, TaskGraphTy &Graph);

/// Begin a replay of the taskgraph \p Graph belongs to: take the initial
/// values of its firstprivates from their variables, as they stand when the
/// replay starts (OpenMP 6.0 [14.3]), however much later the graph runs.
void beginTaskGraphReplay(TaskGraphTy &Graph);

/// Resolve every one of \p Graph's ranges against the device's mapping table
/// for a replay about to start, and report the argument entries that changed
/// in \p Updates.  \p Resync is set when the backend has to re-read the whole
/// plan instead, because a replay it never saw may have changed it.
///
/// A range the graph brings into the device data environment and finds absent
/// is mapped, so that a `target enter data` in a graph behaves as it does
/// outside one, including leaving the mapping behind when the graph's maps do
/// not balance.  A self-contained range gets storage of its own instead.  Every
/// range is then held with a hold reference until the replay ends.
///
/// Only addresses, ownership and transfer enablement are re-derived; the
/// grouping and extents are fixed, so a backend never has to change the shape
/// of a lowered graph.
///
/// Must be paired with releaseTaskGraphMappings even when it returns false,
/// since a partial acquisition still holds what it took; TaskGraphMappingsGuard
/// does that.  Returns false for a mapping state the plan cannot express, which
/// declines this replay to the host.
bool acquireTaskGraphMappings(DeviceTy &Device, TaskGraphTy &Graph,
                              SmallVectorImpl<TaskGraphArgUpdateTy> &Updates,
                              bool &Resync);

/// Tell the plan that the updates the last acquisition produced have reached
/// the backend, so the next one need only report what changes from here.
void commitTaskGraphUpdates(TaskGraphTy &Graph);

/// Drop the hold references acquireTaskGraphMappings took, and remove the
/// mappings the graph created that its maps did not ask to outlive it.
/// \p Replayed is false for a declined replay, which moved no data and so has
/// no attached pointers to restore.
void releaseTaskGraphMappings(DeviceTy &Device, TaskGraphTy &Graph,
                              bool Replayed = true);

/// Releases a graph's mappings unless the acquisition reaches the point of
/// handing the graph to the backend, which then releases them once the graph
/// has run.
class TaskGraphMappingsGuard {
public:
  TaskGraphMappingsGuard(DeviceTy &Device, TaskGraphTy &Graph)
      : Device(Device), Graph(Graph) {}
  ~TaskGraphMappingsGuard() {
    if (Armed)
      releaseTaskGraphMappings(Device, Graph, /*Replayed=*/false);
  }

  /// Hand the hold to the replay.
  void release() { Armed = false; }

  TaskGraphMappingsGuard(const TaskGraphMappingsGuard &) = delete;
  TaskGraphMappingsGuard &operator=(const TaskGraphMappingsGuard &) = delete;

private:
  DeviceTy &Device;
  TaskGraphTy &Graph;
  bool Armed = true;
};

/// Release everything the plan holds for \p Graph, for teardown.
void releaseTaskGraphPlan(DeviceTy &Device, TaskGraphTy &Graph);

} // namespace target
} // namespace omp
} // namespace llvm

#endif // OMPTARGET_OPENMP_TASKGRAPHPLAN_H
