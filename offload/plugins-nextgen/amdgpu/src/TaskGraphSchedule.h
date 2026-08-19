//===-- TaskGraphSchedule.h - Lower a taskgraph onto in-order lanes -*- C++
//-*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowering of the shared processed-taskgraph region tree (Shared/TaskGraph.h)
// onto a fixed set of in-order lanes, each of which the plugin backs with a
// stream over one HSA queue.
//
// This is a plugin-local policy, not part of the shared representation: it is
// the right lowering for a backend whose only ordering primitives are in-order
// queues plus completion signals, and the wrong one for a backend with a native
// graph API, which would translate the same tree into a native DAG and never
// assign lanes at all.
//
// The lowering decides where each leaf goes and what it must wait for, and
// hands each leaf to a sink as soon as that is decided.  What a placed leaf
// becomes -- packets on a stream, a record a test checks -- is the sink's
// business, so the algorithm is shared without a schedule in between.
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOAD_PLUGINS_NEXTGEN_AMDGPU_TASKGRAPHSCHEDULE_H
#define OFFLOAD_PLUGINS_NEXTGEN_AMDGPU_TASKGRAPHSCHEDULE_H

#include "Shared/TaskGraph.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

/// What a lowering onto lanes delivers its output to.
///
/// Every leaf of the graph is placed exactly once, as an *item*.  Items are
/// numbered from 0 in the order they are placed, and that is the number by
/// which a later item's waits name them.  Placement order is a topological
/// order: an item only ever waits for items placed before it.
class TaskGraphLaneSinkTy {
public:
  virtual ~TaskGraphLaneSinkTy() = default;

  /// Place \p Leaf as item \p Item, at the end of \p Lane.  It must not start
  /// before every item in \p Waits has completed.
  ///
  /// \p Waits is minimal for in-order lanes.  It never names an item on
  /// \p Lane, since lane order covers those; it names at most one item per
  /// other lane, the latest one needed; and it names no item that another of
  /// its waits, or the lane's own earlier items, already imply has completed.
  virtual Error place(uint32_t Item, const TaskGraphNode &Leaf, uint32_t Lane,
                      ArrayRef<uint32_t> Waits) = 0;
};

/// Lower \p Graph's region tree onto \p NumLanes in-order lanes, placing its
/// leaves into \p Sink.
///
/// Running every lane concurrently, in order within each lane, with each item
/// started only once its waits have completed, respects every dependence the
/// region tree implies.
///
/// Fails on a malformed graph (no root, a cyclic irreducible region, zero
/// lanes), and with whatever error the sink returns, at which point lowering
/// stops.  Any well-formed tree can be lowered onto any positive number of
/// lanes: with fewer lanes than available concurrency the excess is serialized
/// by sharing lanes, which costs parallelism but stays correct.
Error lowerTaskGraphOntoLanes(const TaskGraphTy &Graph, uint32_t NumLanes,
                              TaskGraphLaneSinkTy &Sink);

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm

#endif // OFFLOAD_PLUGINS_NEXTGEN_AMDGPU_TASKGRAPHSCHEDULE_H
