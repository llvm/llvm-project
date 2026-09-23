//===-- Shared/TaskGraph.h - Processed-taskgraph object model ----- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The representation of a processed OpenMP taskgraph, shared between
// libomptarget and the device plugins.
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

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <new>

namespace llvm {
namespace omp {
namespace target {

//===----------------------------------------------------------------------===//
// Object model (LLVM-style RTTI)
//
// libomp streams the processed graph in preorder, with each region announcing
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
  TaskNode,
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
  llvm::ArrayRef<TaskGraphElement *> children() const {
    return {Children, NumChildren};
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
  __tgt_taskgraph_relocate_ty Relocate = nullptr;

  // mutexinoutset membership: two leaves whose sets intersect must not run
  // concurrently.  The bits are libomp-owned, rather than being deep-copied. A
  // backend that cannot express this constraint must decline the graph rather
  // than ignore it.
  const uint64_t *MutexBits = nullptr; // ceil(MutexNumBits / 64) words
  int32_t MutexNumBits = 0;

  static bool classof(const TaskGraphElement *E) {
    return E->getKind() >= TGKind::TaskNode &&
           E->getKind() <= TGKind::TargetUpdateDataNode;
  }

protected:
  explicit TaskGraphNode(TGKind K) : TaskGraphElement(K) {}
};

/// A host subtree, replayed by calling back into libomp.
struct TaskGraphTaskNode : TaskGraphNode {
  void *Region = nullptr;

  TaskGraphTaskNode() : TaskGraphNode(TGKind::TaskNode) {}
  static bool classof(const TaskGraphElement *E) {
    return E->getKind() == TGKind::TaskNode;
  }
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
// Graph construction
//===----------------------------------------------------------------------===//

struct BuildCursor {
  // Region being filled (null at root)
  TaskGraphRegion *Cur = nullptr;
  // next slot in Cur->children()
  uint32_t Index = 0;
};

struct TaskGraphTy {
  int64_t DeviceId;
  uintptr_t GraphId;
  /// A pointer to the libomp callback to execute a host-only graph region.
  __tgt_taskgraph_host_exec_ty HostCb;
  /// Graph-wide count of distinct runtime mutexes referenced by the leaves'
  /// mutex sets.
  int32_t NumMutexes = 0;
  /// Accumulator for the number of required bytes during the first pass.
  size_t MeasuredBytes = 0;
  /// The single block used for storage of the whole tree and related metadata.
  char *Block = nullptr;
  /// The amount of the block used during the second pass.
  size_t BlockUsed = 0;
  /// The root region, set once the graph has been fully received.
  TaskGraphRegion *Root = nullptr;
  BuildCursor Cursor;

  static size_t alignUp(size_t X, size_t A) { return (X + A - 1) & ~(A - 1); }
  bool building() const { return Block != nullptr; }

  /// Accumulate size (first pass), or bump-allocate from block (second pass).
  void *alloc(size_t Sz, size_t Al) {
    if (!building()) {
      MeasuredBytes = alignUp(MeasuredBytes, Al) + Sz;
      return nullptr;
    }
    BlockUsed = alignUp(BlockUsed, Al);
    assert(BlockUsed + Sz <= MeasuredBytes &&
           "taskgraph single-block overflow");
    void *P = Block + BlockUsed;
    BlockUsed += Sz;
    return P;
  }

  /// Allocate (+ placement-new on the build pass) a region and its trailing
  /// child-pointer array.  Returns null on the measuring pass.
  template <class RegionT> RegionT *makeRegion(int32_t NumChildren) {
    void *P = alloc(sizeof(RegionT), alignof(RegionT));
    void *CA = alloc(sizeof(TaskGraphElement *) * NumChildren,
                     alignof(TaskGraphElement *));
    if (!building())
      return nullptr;
    auto *R = new (P) RegionT(static_cast<uint32_t>(NumChildren));
    R->Children = static_cast<TaskGraphElement **>(CA);
    return R;
  }

  // Build-cursor operations (build pass only).
  void linkChild(TaskGraphElement *E) {
    E->Parent = Cursor.Cur;
    E->ChildIndex = Cursor.Index;
    if (Cursor.Cur)
      Cursor.Cur->Children[Cursor.Index++] = E;
  }
  void pushRegion(TaskGraphRegion *R) {
    linkChild(R);
    Cursor.Cur = R;
    Cursor.Index = 0;
  }
  void popRegion() {
    // Resume the parent at the slot after the region we just finished.
    uint32_t Slot = Cursor.Cur->ChildIndex;
    Cursor.Cur = Cursor.Cur->Parent;
    Cursor.Index = Slot + 1;
  }

  TaskGraphTy(int64_t DeviceId, uintptr_t GraphId,
              __tgt_taskgraph_host_exec_ty HostCb, int32_t NumMutexes,
              size_t ByteSize)
      : DeviceId(DeviceId), GraphId(GraphId), HostCb(HostCb),
        NumMutexes(NumMutexes), MeasuredBytes(ByteSize) {
    if (MeasuredBytes)
      Block = static_cast<char *>(std::malloc(MeasuredBytes));
    if (building()) {
      Cursor.Cur = nullptr;
      Cursor.Index = 0;
    }
  }

  ~TaskGraphTy() { std::free(Block); }
};

} // namespace target
} // namespace omp
} // namespace llvm

#endif // OMPTARGET_SHARED_TASKGRAPH_H
