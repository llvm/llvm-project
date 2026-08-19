//===-- Shared/TaskGraph.h - Processed-taskgraph object model ----- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The representation of a target subgraph of a processed OpenMP taskgraph --
// a subtree of its regions that holds target constructs and nothing else --
// shared between libomptarget and the device plugins.  libomp runs the rest of
// the taskgraph itself, and has libomptarget run each subgraph a plugin claims
// when the replay reaches it.
//
// The representation here is somewhat decoupled from the one used within
// libomp, partly because of the different code styles adopted by each library,
// and partly because different aspects of the representation are important when
// recording/host-replaying the graph vs. executing it on an offload device.
// Particular care has been taken over avoiding the need for strict binary
// compatibility (e.g. structure layout) between libomp and libomptarget -- the
// interface between the two is via function calls with POD argument types.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_SHARED_TASKGRAPH_H
#define OMPTARGET_SHARED_TASKGRAPH_H

#include "omptarget.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <vector>

namespace llvm {
namespace omp {
namespace target {

//===----------------------------------------------------------------------===//
// Object model (LLVM-style RTTI)
//
// libomp streams each subgraph in preorder, with each region announcing
// its child count up front (prefix notation).  We materialize it as a tree of
// the types below, laid out in a single contiguous block: see BuildCursor and
// the two-pass allocation it drives.
//
//===----------------------------------------------------------------------===//

/// Taskgraph node/region type discriminator.  Order is significant here.
enum class TGKind : uint8_t {
  // Interior regions (own an ordered child list).
  ParRegion,
  SeqRegion,
  ExclRegion,
  IrreducibleRegion,

  // Leaves.
  TargetNode,
  // Data leaves form their own sub-range at the tail of the leaf range.
  TargetEnterDataNode,
  TargetExitDataNode,
  TargetUpdateDataNode,
};

struct TaskGraphRegion;

/// Common base for every taskgraph element.
struct TaskGraphElement {
  const TGKind Kind;
  TaskGraphRegion *Parent = nullptr; // null iff this is the graph root
  uint32_t ChildIndex = 0;           // ordinal within Parent's child list

  TGKind getKind() const { return Kind; }

protected:
  explicit TaskGraphElement(TGKind K) : Kind(K) {}
};

//===----------------------------------------------------------------------===//
// Regions (interior nodes)
//===----------------------------------------------------------------------===//

/// An interior region owning an ordered list of children.  The child pointer
/// array is bump-allocated from the same block and filled in child order by
/// the build cursor; its length is known up front from the stream's prefix
/// count, so the region is fully wired before its children are streamed in.
struct TaskGraphRegion : TaskGraphElement {
  uint32_t NumChildren = 0;
  TaskGraphElement **Children = nullptr;

  llvm::MutableArrayRef<TaskGraphElement *> children() {
    return {Children, NumChildren};
  }
  llvm::ArrayRef<const TaskGraphElement *> children() const {
    return llvm::ArrayRef<TaskGraphElement *>(Children, NumChildren);
  }

  static bool classof(const TaskGraphElement *E) {
    return E->getKind() >= TGKind::ParRegion &&
           E->getKind() <= TGKind::IrreducibleRegion;
  }

protected:
  TaskGraphRegion(TGKind K, uint32_t N) : TaskGraphElement(K), NumChildren(N) {}
};

/// Parallel region (children may run concurrently).
struct TaskGraphParRegion : TaskGraphRegion {
  explicit TaskGraphParRegion(uint32_t N)
      : TaskGraphRegion(TGKind::ParRegion, N) {}
  static bool classof(const TaskGraphElement *E) {
    return E->getKind() == TGKind::ParRegion;
  }
};

/// Sequential region (children run in order).
struct TaskGraphSeqRegion : TaskGraphRegion {
  explicit TaskGraphSeqRegion(uint32_t N)
      : TaskGraphRegion(TGKind::SeqRegion, N) {}
  static bool classof(const TaskGraphElement *E) {
    return E->getKind() == TGKind::SeqRegion;
  }
};

/// Children are mutually exclusive (mutexinoutset), unordered.
struct TaskGraphExclRegion : TaskGraphRegion {
  explicit TaskGraphExclRegion(uint32_t N)
      : TaskGraphRegion(TGKind::ExclRegion, N) {}
  static bool classof(const TaskGraphElement *E) {
    return E->getKind() == TGKind::ExclRegion;
  }
};

/// An intra-region dependence edge of an irreducible region, naming children by
/// ordinal.  Both endpoints name a child: the region's sources and sinks are
/// whichever children have no incoming or no outgoing edge, same as for the
/// corresponding edges in libomp.
struct TaskGraphEdge {
  int32_t Src;
  int32_t Dst;
};

/// An irreducible region.  Dependencies between children are represented by
/// an explicit edge array (a pointer into the same allocation block as the
/// rest of the transmitted graph).
struct TaskGraphIrreducibleRegion : TaskGraphRegion {
  uint32_t NumEdges = 0;
  uint32_t EdgeFill = 0;          // transient build cursor into Edges
  TaskGraphEdge *Edges = nullptr; // len == NumEdges, into same block

  llvm::ArrayRef<TaskGraphEdge> edges() const { return {Edges, NumEdges}; }

  explicit TaskGraphIrreducibleRegion(uint32_t N)
      : TaskGraphRegion(TGKind::IrreducibleRegion, N) {}
  static bool classof(const TaskGraphElement *E) {
    return E->getKind() == TGKind::IrreducibleRegion;
  }
};

//===----------------------------------------------------------------------===//
// Leaves (nodes)
//===----------------------------------------------------------------------===//

/// Common base for the leaf nodes.
struct TaskGraphNode : TaskGraphElement {
  int64_t DeviceId = OFFLOAD_DEVICE_DEFAULT;

  static bool classof(const TaskGraphElement *E) {
    return E->getKind() >= TGKind::TargetNode &&
           E->getKind() <= TGKind::TargetUpdateDataNode;
  }

protected:
  explicit TaskGraphNode(TGKind K) : TaskGraphElement(K) {}
};

/// A target kernel launch.  KernelArgs is an opaque KernelArgsTy*, deep-copied
/// by libomp at record time.
struct TaskGraphTargetNode : TaskGraphNode {
  int32_t NumTeams = 0;
  int32_t ThreadLimit = 0;
  void *HostPtr = nullptr;
  void *KernelArgs = nullptr;

  TaskGraphTargetNode() : TaskGraphNode(TGKind::TargetNode) {}
  static bool classof(const TaskGraphElement *E) {
    return E->getKind() == TGKind::TargetNode;
  }
};

/// Base for target enter/exit/update data ops.  The specific op is encoded by
/// the Kind.  The arg arrays are libomp-owned (no copy).
struct TaskGraphTargetDataNode : TaskGraphNode {
  int32_t ArgNum = 0;
  void **ArgsBase = nullptr;
  void **Args = nullptr;
  int64_t *ArgSizes = nullptr;
  int64_t *ArgTypes = nullptr;
  void **ArgNames = nullptr;
  void **ArgMappers = nullptr;

  static bool classof(const TaskGraphElement *E) {
    return E->getKind() >= TGKind::TargetEnterDataNode &&
           E->getKind() <= TGKind::TargetUpdateDataNode;
  }

protected:
  explicit TaskGraphTargetDataNode(TGKind K) : TaskGraphNode(K) {}
};

struct TaskGraphTargetEnterDataNode : TaskGraphTargetDataNode {
  TaskGraphTargetEnterDataNode()
      : TaskGraphTargetDataNode(TGKind::TargetEnterDataNode) {}
  static bool classof(const TaskGraphElement *E) {
    return E->getKind() == TGKind::TargetEnterDataNode;
  }
};

struct TaskGraphTargetExitDataNode : TaskGraphTargetDataNode {
  TaskGraphTargetExitDataNode()
      : TaskGraphTargetDataNode(TGKind::TargetExitDataNode) {}
  static bool classof(const TaskGraphElement *E) {
    return E->getKind() == TGKind::TargetExitDataNode;
  }
};

struct TaskGraphTargetUpdateDataNode : TaskGraphTargetDataNode {
  TaskGraphTargetUpdateDataNode()
      : TaskGraphTargetDataNode(TGKind::TargetUpdateDataNode) {}
  static bool classof(const TaskGraphElement *E) {
    return E->getKind() == TGKind::TargetUpdateDataNode;
  }
};

//===----------------------------------------------------------------------===//
// libomptarget services available to a plugin while lowering a graph
//===----------------------------------------------------------------------===//

/// What a plugin needs from libomptarget to lower a recorded graph, passed as a
/// function pointer rather than called directly.
///
/// A plugin cannot name a libomptarget symbol: plugins are static archives
/// linked into both libomptarget.so and LLVMOffload, so a direct call would
/// leave LLVMOffload with an undefined reference.
///
/// Every DeviceId passed here is a concrete device: libomptarget normalizes
/// OFFLOAD_DEVICE_DEFAULT away across the whole tree before a plugin sees it.
struct TaskGraphServicesTy {
  /// Resolve a recorded host function pointer to the plugin-side kernel object
  /// (a GenericKernelTy *), or null if the device has no matching entry.
  void *(*resolveKernel)(int64_t DeviceId, void *HostPtr) = nullptr;
};

//===----------------------------------------------------------------------===//
// The graph
//===----------------------------------------------------------------------===//

class TaskGraphBuilder;

/// Where the build pass links the next streamed element.
struct BuildCursor {
  /// The region being filled, or null at the root.
  TaskGraphRegion *Cur = nullptr;
  /// The next slot in Cur->children().
  uint32_t Index = 0;
};

/// A processed taskgraph, streamed in by libomp and built by TaskGraphBuilder
/// (Shared/TaskGraphBuilder.h).  A plugin is handed a const graph, which it
/// only reads.
class TaskGraphTy {
public:
  TaskGraphTy(int64_t DeviceId, uintptr_t GraphId)
      : DeviceId(DeviceId), GraphId(GraphId) {}
  ~TaskGraphTy() { std::free(Block); }
  TaskGraphTy(const TaskGraphTy &) = delete;
  TaskGraphTy &operator=(const TaskGraphTy &) = delete;

  /// The device the graph is replayed on, concrete once the graph is offered
  /// to a plugin.
  int64_t DeviceId;
  uintptr_t GraphId;
  /// The claiming plugin's own form of the graph: whatever its finalize
  /// returned, opaque to everything but that plugin.  Null for a graph no
  /// plugin claimed.
  void *PluginGraph = nullptr;

  /// Filled in by libomptarget before the graph is offered to a plugin.
  TaskGraphServicesTy Services;

  /// The top of the tree: the first element streamed outside any region.  A
  /// region in general, but a graph holding a single leaf arrives as that leaf.
  TaskGraphElement *getRoot() { return Root; }
  const TaskGraphElement *getRoot() const { return Root; }

  /// The leaves in preorder.  Empty for a graph that was not well formed.
  ArrayRef<TaskGraphNode *> leaves() { return Leaves; }
  ArrayRef<const TaskGraphNode *> leaves() const {
    return ArrayRef<TaskGraphNode *>(Leaves);
  }

private:
  friend class TaskGraphBuilder;

  TaskGraphElement *Root = nullptr;
  std::vector<TaskGraphNode *> Leaves;

  /// The single block holding the whole tree, null on the measuring pass.
  char *Block = nullptr;
  /// The block's size: accumulated on the measuring pass, given on the build
  /// pass.
  size_t MeasuredBytes = 0;
  /// How much of the block the build pass has used.
  size_t BlockUsed = 0;
  BuildCursor Cursor;
  /// How the stream broke the transmission protocol, if it did.
  const char *Malformed = nullptr;
};

} // namespace target
} // namespace omp
} // namespace llvm

#endif // OMPTARGET_SHARED_TASKGRAPH_H
