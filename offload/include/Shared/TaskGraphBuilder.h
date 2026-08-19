//===-- Shared/TaskGraphBuilder.h - Build a streamed taskgraph ---- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Construction of a TaskGraphTy from the preorder stream libomp sends through
// the taskgraph transmission ABI (omptarget.h).  Used by libomptarget, and by
// the unit tests that stream graphs the same way; a plugin only ever reads a
// finished graph.
//
// The stream is sent twice.  On the measuring pass the builder only adds up
// the bytes the tree will occupy; on the build pass it lays the tree out in a
// single block of that size, which never moves while it is filled.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_SHARED_TASKGRAPHBUILDER_H
#define OMPTARGET_SHARED_TASKGRAPHBUILDER_H

#include "Shared/TaskGraph.h"

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <vector>

namespace llvm {
namespace omp {
namespace target {

class TaskGraphBuilder {
public:
  explicit TaskGraphBuilder(TaskGraphTy &G) : G(G) {}

  /// Begin a pass over the stream: the measuring pass for a \p ByteSize of
  /// zero, and otherwise the build pass into a block of \p ByteSize bytes.
  static TaskGraphTy *start(int64_t DeviceId, uintptr_t GraphId,
                            size_t ByteSize) {
    auto *G = new TaskGraphTy(DeviceId, GraphId);
    G->MeasuredBytes = ByteSize;
    if (ByteSize)
      G->Block = static_cast<char *>(std::malloc(ByteSize));
    return G;
  }

  /// End the pass, returning the size of block the graph needs.  On the build
  /// pass, a well-formed tree also gets its leaves collected.
  size_t finish() {
    if (!building())
      return alignUp(G.MeasuredBytes, alignof(std::max_align_t));
    assert(G.BlockUsed <= G.MeasuredBytes && "taskgraph block overrun");
    if (!whyMalformed(G))
      collectLeaves(G.Root, G.Leaves);
    return G.MeasuredBytes;
  }

  /// Whether this is the build pass rather than the measuring pass.
  bool building() const { return G.Block != nullptr; }

  /// Account for (measuring pass) or bump-allocate (build pass) \p Size bytes.
  void *alloc(size_t Size, size_t Align) {
    if (!building()) {
      G.MeasuredBytes = alignUp(G.MeasuredBytes, Align) + Size;
      return nullptr;
    }
    G.BlockUsed = alignUp(G.BlockUsed, Align);
    assert(G.BlockUsed + Size <= G.MeasuredBytes &&
           "taskgraph single-block overflow");
    void *P = G.Block + G.BlockUsed;
    G.BlockUsed += Size;
    return P;
  }

  /// Allocate a leaf, constructed on the build pass.  Null on the measuring
  /// pass.
  template <class NodeT> NodeT *makeLeaf() {
    void *P = alloc(sizeof(NodeT), alignof(NodeT));
    return building() ? new (P) NodeT() : nullptr;
  }

  /// Allocate a region and its child-pointer array, constructed on the build
  /// pass.  Null on the measuring pass.
  template <class RegionT> RegionT *makeRegion(int32_t NumChildren) {
    void *P = alloc(sizeof(RegionT), alignof(RegionT));
    void *Children = alloc(sizeof(TaskGraphElement *) * NumChildren,
                           alignof(TaskGraphElement *));
    if (!building())
      return nullptr;
    auto *R = new (P) RegionT(static_cast<uint32_t>(NumChildren));
    R->Children = static_cast<TaskGraphElement **>(Children);
    return R;
  }

  /// Allocate an irreducible region with room for \p NumEdges edges.
  TaskGraphIrreducibleRegion *makeIrreducibleRegion(int32_t NumChildren,
                                                    int32_t NumEdges) {
    auto *R = makeRegion<TaskGraphIrreducibleRegion>(NumChildren);
    void *Edges =
        alloc(sizeof(TaskGraphEdge) * NumEdges, alignof(TaskGraphEdge));
    if (!building())
      return nullptr;
    R->NumEdges = static_cast<uint32_t>(NumEdges);
    R->Edges = static_cast<TaskGraphEdge *>(Edges);
    return R;
  }

  // Build-cursor operations, for the build pass only.

  /// Link \p E as the next child of the region being filled, or as the root.
  void linkChild(TaskGraphElement *E) {
    BuildCursor &Cursor = G.Cursor;
    E->Parent = Cursor.Cur;
    E->ChildIndex = Cursor.Index;
    if (Cursor.Cur) {
      if (Cursor.Index == Cursor.Cur->NumChildren)
        return setMalformed("more children than their region announced");
      Cursor.Cur->Children[Cursor.Index++] = E;
      return;
    }
    if (G.Root)
      return setMalformed("more than one root");
    G.Root = E;
  }
  /// Link \p R and fill it with the children that follow.
  void pushRegion(TaskGraphRegion *R) {
    linkChild(R);
    G.Cursor.Cur = R;
    G.Cursor.Index = 0;
  }
  /// Finish the region being filled.
  void popRegion() {
    BuildCursor &Cursor = G.Cursor;
    if (!Cursor.Cur)
      return setMalformed("a region end with no region open");
    // Resume the parent at the slot after the region just finished.
    uint32_t Slot = Cursor.Cur->ChildIndex;
    Cursor.Cur = Cursor.Cur->Parent;
    Cursor.Index = Slot + 1;
  }
  /// Add an edge to the irreducible region being filled.
  void addEdge(int32_t Src, int32_t Dst) {
    auto *Irr =
        llvm::dyn_cast_if_present<TaskGraphIrreducibleRegion>(G.Cursor.Cur);
    if (!Irr)
      return setMalformed("an edge outside an irreducible region");
    if (Irr->EdgeFill == Irr->NumEdges)
      return setMalformed("more edges than their irreducible region announced");
    if (Src < 0 || Dst < 0 || static_cast<uint32_t>(Src) >= Irr->NumChildren ||
        static_cast<uint32_t>(Dst) >= Irr->NumChildren)
      return setMalformed("an edge naming no child of its region");
    Irr->Edges[Irr->EdgeFill++] = {Src, Dst};
  }

  /// An explanation for why \p G is malformed if is it, else null.
  static const char *whyMalformed(const TaskGraphTy &G) {
    if (G.Malformed)
      return G.Malformed;
    if (!G.Root)
      return "no root";
    return whyMalformed(G.Root);
  }

private:
  /// Note the first way the stream broke the transmission protocol.
  void setMalformed(const char *Why) {
    if (!G.Malformed)
      G.Malformed = Why;
  }

  static size_t alignUp(size_t X, size_t A) { return (X + A - 1) & ~(A - 1); }

  static const char *whyMalformed(const TaskGraphElement *E) {
    const auto *R = llvm::dyn_cast<TaskGraphRegion>(E);
    if (!R)
      return nullptr;
    for (const TaskGraphElement *C : R->children()) {
      if (!C)
        return "fewer children than their region announced";
      if (const char *Why = whyMalformed(C))
        return Why;
    }
    const auto *Irr = llvm::dyn_cast<TaskGraphIrreducibleRegion>(R);
    if (!Irr)
      return nullptr;
    if (Irr->EdgeFill != Irr->NumEdges)
      return "fewer edges than their irreducible region announced";

    // Kahn's algorithm: the region is acyclic iff every child gets ordered.
    std::vector<uint32_t> InDegree(Irr->NumChildren, 0);
    for (const TaskGraphEdge &Edge : Irr->edges())
      ++InDegree[Edge.Dst];
    std::vector<uint32_t> Ready;
    for (uint32_t I = 0; I < Irr->NumChildren; ++I)
      if (!InDegree[I])
        Ready.push_back(I);
    for (size_t Head = 0; Head < Ready.size(); ++Head)
      for (const TaskGraphEdge &Edge : Irr->edges())
        if (static_cast<uint32_t>(Edge.Src) == Ready[Head] &&
            !--InDegree[Edge.Dst])
          Ready.push_back(Edge.Dst);
    return Ready.size() == Irr->NumChildren
               ? nullptr
               : "a cycle among an irreducible region's edges";
  }

  static void collectLeaves(TaskGraphElement *E,
                            std::vector<TaskGraphNode *> &Out) {
    if (auto *R = llvm::dyn_cast<TaskGraphRegion>(E)) {
      for (TaskGraphElement *C : R->children())
        collectLeaves(C, Out);
      return;
    }
    Out.push_back(llvm::cast<TaskGraphNode>(E));
  }

  TaskGraphTy &G;
};

} // namespace target
} // namespace omp
} // namespace llvm

#endif // OMPTARGET_SHARED_TASKGRAPHBUILDER_H
