//===- LoopFusion.cpp - Code to perform loop fusion -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop fusion.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopFusionUtils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <iomanip>
#include <sstream>
#define DEBUG_TYPE "affine-loop-fusion"

using namespace mlir;

namespace {
/// Loop fusion pass. This pass currently supports a greedy fusion policy,
/// which fuses loop nests with single-writer/single-reader memref dependences
/// with the goal of improving locality.

// TODO: Support fusion of source loop nests which write to multiple
// memrefs, where each memref can have multiple users (if profitable).
// TODO: Extend this pass to check for fusion preventing dependences,
// and add support for more general loop fusion algorithms.

struct LoopFusion : public AffineLoopFusionBase<LoopFusion> {
  LoopFusion() = default;
  LoopFusion(unsigned fastMemorySpace, uint64_t localBufSizeThresholdBytes,
             bool maximalFusion, enum FusionMode affineFusionMode) {
    this->fastMemorySpace = fastMemorySpace;
    this->localBufSizeThreshold = localBufSizeThresholdBytes / 1024;
    this->maximalFusion = maximalFusion;
    this->affineFusionMode = affineFusionMode;
  }

  void runOnOperation() override;
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::createLoopFusionPass(unsigned fastMemorySpace,
                           uint64_t localBufSizeThreshold, bool maximalFusion,
                           enum FusionMode affineFusionMode) {
  return std::make_unique<LoopFusion>(fastMemorySpace, localBufSizeThreshold,
                                      maximalFusion, affineFusionMode);
}

namespace {

// LoopNestStateCollector walks loop nests and collects load and store
// operations, and whether or not a region holding op other than ForOp and IfOp
// was encountered in the loop nest.
struct LoopNestStateCollector {
  SmallVector<AffineForOp, 4> forOps;
  SmallVector<Operation *, 4> loadOpInsts;
  SmallVector<Operation *, 4> storeOpInsts;
  bool hasNonAffineRegionOp = false;

  void collect(Operation *opToWalk) {
    opToWalk->walk([&](Operation *op) {
      if (isa<AffineForOp>(op))
        forOps.push_back(cast<AffineForOp>(op));
      else if (op->getNumRegions() != 0 && !isa<AffineIfOp>(op))
        hasNonAffineRegionOp = true;
      else if (isa<AffineReadOpInterface>(op))
        loadOpInsts.push_back(op);
      else if (isa<AffineWriteOpInterface>(op))
        storeOpInsts.push_back(op);
    });
  }
};

// MemRefDependenceGraph is a graph data structure where graph nodes are
// top-level operations in a FuncOp which contain load/store ops, and edges
// are memref dependences between the nodes.
// TODO: Add a more flexible dependence graph representation.
// TODO: Add a depth parameter to dependence graph construction.
struct MemRefDependenceGraph {
public:
  // Node represents a node in the graph. A Node is either an entire loop nest
  // rooted at the top level which contains loads/stores, or a top level
  // load/store.
  struct Node {
    // The unique identifier of this node in the graph.
    unsigned id;
    // The top-level statement which is (or contains) a load/store.
    Operation *op;
    // List of load operations.
    SmallVector<Operation *, 4> loads;
    // List of store op insts.
    SmallVector<Operation *, 4> stores;
    Node(unsigned id, Operation *op) : id(id), op(op) {}

    // Returns the load op count for 'memref'.
    unsigned getLoadOpCount(Value memref) {
      unsigned loadOpCount = 0;
      for (auto *loadOpInst : loads) {
        if (memref == cast<AffineReadOpInterface>(loadOpInst).getMemRef())
          ++loadOpCount;
      }
      return loadOpCount;
    }

    // Returns the store op count for 'memref'.
    unsigned getStoreOpCount(Value memref) {
      unsigned storeOpCount = 0;
      for (auto *storeOpInst : stores) {
        if (memref == cast<AffineWriteOpInterface>(storeOpInst).getMemRef())
          ++storeOpCount;
      }
      return storeOpCount;
    }

    // Returns all store ops in 'storeOps' which access 'memref'.
    void getStoreOpsForMemref(Value memref,
                              SmallVectorImpl<Operation *> *storeOps) {
      for (auto *storeOpInst : stores) {
        if (memref == cast<AffineWriteOpInterface>(storeOpInst).getMemRef())
          storeOps->push_back(storeOpInst);
      }
    }

    // Returns all load ops in 'loadOps' which access 'memref'.
    void getLoadOpsForMemref(Value memref,
                             SmallVectorImpl<Operation *> *loadOps) {
      for (auto *loadOpInst : loads) {
        if (memref == cast<AffineReadOpInterface>(loadOpInst).getMemRef())
          loadOps->push_back(loadOpInst);
      }
    }

    // Returns all memrefs in 'loadAndStoreMemrefSet' for which this node
    // has at least one load and store operation.
    void getLoadAndStoreMemrefSet(DenseSet<Value> *loadAndStoreMemrefSet) {
      llvm::SmallDenseSet<Value, 2> loadMemrefs;
      for (auto *loadOpInst : loads) {
        loadMemrefs.insert(cast<AffineReadOpInterface>(loadOpInst).getMemRef());
      }
      for (auto *storeOpInst : stores) {
        auto memref = cast<AffineWriteOpInterface>(storeOpInst).getMemRef();
        if (loadMemrefs.count(memref) > 0)
          loadAndStoreMemrefSet->insert(memref);
      }
    }
  };

  // Edge represents a data dependence between nodes in the graph.
  struct Edge {
    // The id of the node at the other end of the edge.
    // If this edge is stored in Edge = Node.inEdges[i], then
    // 'Node.inEdges[i].id' is the identifier of the source node of the edge.
    // If this edge is stored in Edge = Node.outEdges[i], then
    // 'Node.outEdges[i].id' is the identifier of the dest node of the edge.
    unsigned id;
    // The SSA value on which this edge represents a dependence.
    // If the value is a memref, then the dependence is between graph nodes
    // which contain accesses to the same memref 'value'. If the value is a
    // non-memref value, then the dependence is between a graph node which
    // defines an SSA value and another graph node which uses the SSA value
    // (e.g. a constant or load operation defining a value which is used inside
    // a loop nest).
    Value value;
  };

  // Map from node id to Node.
  DenseMap<unsigned, Node> nodes;
  // Map from node id to list of input edges.
  DenseMap<unsigned, SmallVector<Edge, 2>> inEdges;
  // Map from node id to list of output edges.
  DenseMap<unsigned, SmallVector<Edge, 2>> outEdges;
  // Map from memref to a count on the dependence edges associated with that
  // memref.
  DenseMap<Value, unsigned> memrefEdgeCount;
  // The next unique identifier to use for newly created graph nodes.
  unsigned nextNodeId = 0;

  MemRefDependenceGraph() = default;

  // Initializes the dependence graph based on operations in 'f'.
  // Returns true on success, false otherwise.
  bool init(func::FuncOp f);

  // Returns the graph node for 'id'.
  Node *getNode(unsigned id) {
    auto it = nodes.find(id);
    assert(it != nodes.end());
    return &it->second;
  }

  // Returns the graph node for 'forOp'.
  Node *getForOpNode(AffineForOp forOp) {
    for (auto &idAndNode : nodes)
      if (idAndNode.second.op == forOp.getOperation())
        return &idAndNode.second;
    return nullptr;
  }

  // Adds a node with 'op' to the graph and returns its unique identifier.
  unsigned addNode(Operation *op) {
    Node node(nextNodeId++, op);
    nodes.insert({node.id, node});
    return node.id;
  }

  // Remove node 'id' (and its associated edges) from graph.
  void removeNode(unsigned id) {
    // Remove each edge in 'inEdges[id]'.
    if (inEdges.count(id) > 0) {
      SmallVector<Edge, 2> oldInEdges = inEdges[id];
      for (auto &inEdge : oldInEdges) {
        removeEdge(inEdge.id, id, inEdge.value);
      }
    }
    // Remove each edge in 'outEdges[id]'.
    if (outEdges.count(id) > 0) {
      SmallVector<Edge, 2> oldOutEdges = outEdges[id];
      for (auto &outEdge : oldOutEdges) {
        removeEdge(id, outEdge.id, outEdge.value);
      }
    }
    // Erase remaining node state.
    inEdges.erase(id);
    outEdges.erase(id);
    nodes.erase(id);
  }

  // Returns true if node 'id' writes to any memref which escapes (or is an
  // argument to) the function/block. Returns false otherwise.
  bool writesToLiveInOrEscapingMemrefs(unsigned id) {
    Node *node = getNode(id);
    for (auto *storeOpInst : node->stores) {
      auto memref = cast<AffineWriteOpInterface>(storeOpInst).getMemRef();
      auto *op = memref.getDefiningOp();
      // Return true if 'memref' is a block argument.
      if (!op)
        return true;
      // Return true if any use of 'memref' escapes the function.
      for (auto *user : memref.getUsers())
        if (!isa<AffineMapAccessInterface>(*user))
          return true;
    }
    return false;
  }

  // Returns true iff there is an edge from node 'srcId' to node 'dstId' which
  // is for 'value' if non-null, or for any value otherwise. Returns false
  // otherwise.
  bool hasEdge(unsigned srcId, unsigned dstId, Value value = nullptr) {
    if (outEdges.count(srcId) == 0 || inEdges.count(dstId) == 0) {
      return false;
    }
    bool hasOutEdge = llvm::any_of(outEdges[srcId], [=](Edge &edge) {
      return edge.id == dstId && (!value || edge.value == value);
    });
    bool hasInEdge = llvm::any_of(inEdges[dstId], [=](Edge &edge) {
      return edge.id == srcId && (!value || edge.value == value);
    });
    return hasOutEdge && hasInEdge;
  }

  // Adds an edge from node 'srcId' to node 'dstId' for 'value'.
  void addEdge(unsigned srcId, unsigned dstId, Value value) {
    if (!hasEdge(srcId, dstId, value)) {
      outEdges[srcId].push_back({dstId, value});
      inEdges[dstId].push_back({srcId, value});
      if (value.getType().isa<MemRefType>())
        memrefEdgeCount[value]++;
    }
  }

  // Removes an edge from node 'srcId' to node 'dstId' for 'value'.
  void removeEdge(unsigned srcId, unsigned dstId, Value value) {
    assert(inEdges.count(dstId) > 0);
    assert(outEdges.count(srcId) > 0);
    if (value.getType().isa<MemRefType>()) {
      assert(memrefEdgeCount.count(value) > 0);
      memrefEdgeCount[value]--;
    }
    // Remove 'srcId' from 'inEdges[dstId]'.
    for (auto *it = inEdges[dstId].begin(); it != inEdges[dstId].end(); ++it) {
      if ((*it).id == srcId && (*it).value == value) {
        inEdges[dstId].erase(it);
        break;
      }
    }
    // Remove 'dstId' from 'outEdges[srcId]'.
    for (auto *it = outEdges[srcId].begin(); it != outEdges[srcId].end();
         ++it) {
      if ((*it).id == dstId && (*it).value == value) {
        outEdges[srcId].erase(it);
        break;
      }
    }
  }

  // Returns true if there is a path in the dependence graph from node 'srcId'
  // to node 'dstId'. Returns false otherwise.
  bool hasDependencePath(unsigned srcId, unsigned dstId) {
    // Worklist state is: <node-id, next-output-edge-index-to-visit>
    SmallVector<std::pair<unsigned, unsigned>, 4> worklist;
    worklist.push_back({srcId, 0});
    // Run DFS traversal to see if 'dstId' is reachable from 'srcId'.
    while (!worklist.empty()) {
      auto &idAndIndex = worklist.back();
      // Return true if we have reached 'dstId'.
      if (idAndIndex.first == dstId)
        return true;
      // Pop and continue if node has no out edges, or if all out edges have
      // already been visited.
      if (outEdges.count(idAndIndex.first) == 0 ||
          idAndIndex.second == outEdges[idAndIndex.first].size()) {
        worklist.pop_back();
        continue;
      }
      // Get graph edge to traverse.
      Edge edge = outEdges[idAndIndex.first][idAndIndex.second];
      // Increment next output edge index for 'idAndIndex'.
      ++idAndIndex.second;
      // Add node at 'edge.id' to worklist.
      worklist.push_back({edge.id, 0});
    }
    return false;
  }

  // Returns the input edge count for node 'id' and 'memref' from src nodes
  // which access 'memref' with a store operation.
  unsigned getIncomingMemRefAccesses(unsigned id, Value memref) {
    unsigned inEdgeCount = 0;
    if (inEdges.count(id) > 0)
      for (auto &inEdge : inEdges[id])
        if (inEdge.value == memref) {
          Node *srcNode = getNode(inEdge.id);
          // Only count in edges from 'srcNode' if 'srcNode' accesses 'memref'
          if (srcNode->getStoreOpCount(memref) > 0)
            ++inEdgeCount;
        }
    return inEdgeCount;
  }

  // Returns the output edge count for node 'id' and 'memref' (if non-null),
  // otherwise returns the total output edge count from node 'id'.
  unsigned getOutEdgeCount(unsigned id, Value memref = nullptr) {
    unsigned outEdgeCount = 0;
    if (outEdges.count(id) > 0)
      for (auto &outEdge : outEdges[id])
        if (!memref || outEdge.value == memref)
          ++outEdgeCount;
    return outEdgeCount;
  }

  /// Return all nodes which define SSA values used in node 'id'.
  void gatherDefiningNodes(unsigned id, DenseSet<unsigned> &definingNodes) {
    for (MemRefDependenceGraph::Edge edge : inEdges[id])
      // By definition of edge, if the edge value is a non-memref value,
      // then the dependence is between a graph node which defines an SSA value
      // and another graph node which uses the SSA value.
      if (!edge.value.getType().isa<MemRefType>())
        definingNodes.insert(edge.id);
  }

  // Computes and returns an insertion point operation, before which the
  // the fused <srcId, dstId> loop nest can be inserted while preserving
  // dependences. Returns nullptr if no such insertion point is found.
  Operation *getFusedLoopNestInsertionPoint(unsigned srcId, unsigned dstId) {
    if (outEdges.count(srcId) == 0)
      return getNode(dstId)->op;

    // Skip if there is any defining node of 'dstId' that depends on 'srcId'.
    DenseSet<unsigned> definingNodes;
    gatherDefiningNodes(dstId, definingNodes);
    if (llvm::any_of(definingNodes, [&](unsigned id) {
          return hasDependencePath(srcId, id);
        })) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Can't fuse: a defining op with a user in the dst "
                    "loop has dependence from the src loop\n");
      return nullptr;
    }

    // Build set of insts in range (srcId, dstId) which depend on 'srcId'.
    SmallPtrSet<Operation *, 2> srcDepInsts;
    for (auto &outEdge : outEdges[srcId])
      if (outEdge.id != dstId)
        srcDepInsts.insert(getNode(outEdge.id)->op);

    // Build set of insts in range (srcId, dstId) on which 'dstId' depends.
    SmallPtrSet<Operation *, 2> dstDepInsts;
    for (auto &inEdge : inEdges[dstId])
      if (inEdge.id != srcId)
        dstDepInsts.insert(getNode(inEdge.id)->op);

    Operation *srcNodeInst = getNode(srcId)->op;
    Operation *dstNodeInst = getNode(dstId)->op;

    // Computing insertion point:
    // *) Walk all operation positions in Block operation list in the
    //    range (src, dst). For each operation 'op' visited in this search:
    //   *) Store in 'firstSrcDepPos' the first position where 'op' has a
    //      dependence edge from 'srcNode'.
    //   *) Store in 'lastDstDepPost' the last position where 'op' has a
    //      dependence edge to 'dstNode'.
    // *) Compare 'firstSrcDepPos' and 'lastDstDepPost' to determine the
    //    operation insertion point (or return null pointer if no such
    //    insertion point exists: 'firstSrcDepPos' <= 'lastDstDepPos').
    SmallVector<Operation *, 2> depInsts;
    Optional<unsigned> firstSrcDepPos;
    Optional<unsigned> lastDstDepPos;
    unsigned pos = 0;
    for (Block::iterator it = std::next(Block::iterator(srcNodeInst));
         it != Block::iterator(dstNodeInst); ++it) {
      Operation *op = &(*it);
      if (srcDepInsts.count(op) > 0 && firstSrcDepPos == None)
        firstSrcDepPos = pos;
      if (dstDepInsts.count(op) > 0)
        lastDstDepPos = pos;
      depInsts.push_back(op);
      ++pos;
    }

    if (firstSrcDepPos.has_value()) {
      if (lastDstDepPos.has_value()) {
        if (firstSrcDepPos.value() <= lastDstDepPos.value()) {
          // No valid insertion point exists which preserves dependences.
          return nullptr;
        }
      }
      // Return the insertion point at 'firstSrcDepPos'.
      return depInsts[firstSrcDepPos.value()];
    }
    // No dependence targets in range (or only dst deps in range), return
    // 'dstNodInst' insertion point.
    return dstNodeInst;
  }

  // Updates edge mappings from node 'srcId' to node 'dstId' after fusing them,
  // taking into account that:
  //   *) if 'removeSrcId' is true, 'srcId' will be removed after fusion,
  //   *) memrefs in 'privateMemRefs' has been replaced in node at 'dstId' by a
  //      private memref.
  void updateEdges(unsigned srcId, unsigned dstId,
                   const DenseSet<Value> &privateMemRefs, bool removeSrcId) {
    // For each edge in 'inEdges[srcId]': add new edge remapping to 'dstId'.
    if (inEdges.count(srcId) > 0) {
      SmallVector<Edge, 2> oldInEdges = inEdges[srcId];
      for (auto &inEdge : oldInEdges) {
        // Add edge from 'inEdge.id' to 'dstId' if it's not a private memref.
        if (privateMemRefs.count(inEdge.value) == 0)
          addEdge(inEdge.id, dstId, inEdge.value);
      }
    }
    // For each edge in 'outEdges[srcId]': remove edge from 'srcId' to 'dstId'.
    // If 'srcId' is going to be removed, remap all the out edges to 'dstId'.
    if (outEdges.count(srcId) > 0) {
      SmallVector<Edge, 2> oldOutEdges = outEdges[srcId];
      for (auto &outEdge : oldOutEdges) {
        // Remove any out edges from 'srcId' to 'dstId' across memrefs.
        if (outEdge.id == dstId)
          removeEdge(srcId, outEdge.id, outEdge.value);
        else if (removeSrcId) {
          addEdge(dstId, outEdge.id, outEdge.value);
          removeEdge(srcId, outEdge.id, outEdge.value);
        }
      }
    }
    // Remove any edges in 'inEdges[dstId]' on 'oldMemRef' (which is being
    // replaced by a private memref). These edges could come from nodes
    // other than 'srcId' which were removed in the previous step.
    if (inEdges.count(dstId) > 0 && !privateMemRefs.empty()) {
      SmallVector<Edge, 2> oldInEdges = inEdges[dstId];
      for (auto &inEdge : oldInEdges)
        if (privateMemRefs.count(inEdge.value) > 0)
          removeEdge(inEdge.id, dstId, inEdge.value);
    }
  }

  // Update edge mappings for nodes 'sibId' and 'dstId' to reflect fusion
  // of sibling node 'sibId' into node 'dstId'.
  void updateEdges(unsigned sibId, unsigned dstId) {
    // For each edge in 'inEdges[sibId]':
    // *) Add new edge from source node 'inEdge.id' to 'dstNode'.
    // *) Remove edge from source node 'inEdge.id' to 'sibNode'.
    if (inEdges.count(sibId) > 0) {
      SmallVector<Edge, 2> oldInEdges = inEdges[sibId];
      for (auto &inEdge : oldInEdges) {
        addEdge(inEdge.id, dstId, inEdge.value);
        removeEdge(inEdge.id, sibId, inEdge.value);
      }
    }

    // For each edge in 'outEdges[sibId]' to node 'id'
    // *) Add new edge from 'dstId' to 'outEdge.id'.
    // *) Remove edge from 'sibId' to 'outEdge.id'.
    if (outEdges.count(sibId) > 0) {
      SmallVector<Edge, 2> oldOutEdges = outEdges[sibId];
      for (auto &outEdge : oldOutEdges) {
        addEdge(dstId, outEdge.id, outEdge.value);
        removeEdge(sibId, outEdge.id, outEdge.value);
      }
    }
  }

  // Adds ops in 'loads' and 'stores' to node at 'id'.
  void addToNode(unsigned id, const SmallVectorImpl<Operation *> &loads,
                 const SmallVectorImpl<Operation *> &stores) {
    Node *node = getNode(id);
    llvm::append_range(node->loads, loads);
    llvm::append_range(node->stores, stores);
  }

  void clearNodeLoadAndStores(unsigned id) {
    Node *node = getNode(id);
    node->loads.clear();
    node->stores.clear();
  }

  // Calls 'callback' for each input edge incident to node 'id' which carries a
  // memref dependence.
  void forEachMemRefInputEdge(unsigned id,
                              const std::function<void(Edge)> &callback) {
    if (inEdges.count(id) > 0)
      forEachMemRefEdge(inEdges[id], callback);
  }

  // Calls 'callback' for each output edge from node 'id' which carries a
  // memref dependence.
  void forEachMemRefOutputEdge(unsigned id,
                               const std::function<void(Edge)> &callback) {
    if (outEdges.count(id) > 0)
      forEachMemRefEdge(outEdges[id], callback);
  }

  // Calls 'callback' for each edge in 'edges' which carries a memref
  // dependence.
  void forEachMemRefEdge(ArrayRef<Edge> edges,
                         const std::function<void(Edge)> &callback) {
    for (const auto &edge : edges) {
      // Skip if 'edge' is not a memref dependence edge.
      if (!edge.value.getType().isa<MemRefType>())
        continue;
      assert(nodes.count(edge.id) > 0);
      // Skip if 'edge.id' is not a loop nest.
      if (!isa<AffineForOp>(getNode(edge.id)->op))
        continue;
      // Visit current input edge 'edge'.
      callback(edge);
    }
  }

  void print(raw_ostream &os) const {
    os << "\nMemRefDependenceGraph\n";
    os << "\nNodes:\n";
    for (const auto &idAndNode : nodes) {
      os << "Node: " << idAndNode.first << "\n";
      auto it = inEdges.find(idAndNode.first);
      if (it != inEdges.end()) {
        for (const auto &e : it->second)
          os << "  InEdge: " << e.id << " " << e.value << "\n";
      }
      it = outEdges.find(idAndNode.first);
      if (it != outEdges.end()) {
        for (const auto &e : it->second)
          os << "  OutEdge: " << e.id << " " << e.value << "\n";
      }
    }
  }
  void dump() const { print(llvm::errs()); }
};

/// Returns true if node 'srcId' can be removed after fusing it with node
/// 'dstId'. The node can be removed if any of the following conditions are met:
///   1. 'srcId' has no output dependences after fusion and no escaping memrefs.
///   2. 'srcId' has no output dependences after fusion, has escaping memrefs
///       and the fusion slice is maximal.
///   3. 'srcId' has output dependences after fusion, the fusion slice is
///      maximal and the fusion insertion point dominates all the dependences.
static bool canRemoveSrcNodeAfterFusion(
    unsigned srcId, unsigned dstId, const ComputationSliceState &fusionSlice,
    Operation *fusedLoopInsPoint, const DenseSet<Value> &escapingMemRefs,
    MemRefDependenceGraph *mdg) {

  Operation *dstNodeOp = mdg->getNode(dstId)->op;
  bool hasOutDepsAfterFusion = false;

  for (auto &outEdge : mdg->outEdges[srcId]) {
    Operation *depNodeOp = mdg->getNode(outEdge.id)->op;
    // Skip dependence with dstOp since it will be removed after fusion.
    if (depNodeOp == dstNodeOp)
      continue;

    // Only fusion within the same block is supported. Use domination analysis
    // when needed.
    if (depNodeOp->getBlock() != dstNodeOp->getBlock())
      return false;

    // Check if the insertion point of the fused loop dominates the dependence.
    // Otherwise, the src loop can't be removed.
    if (fusedLoopInsPoint != depNodeOp &&
        !fusedLoopInsPoint->isBeforeInBlock(depNodeOp)) {
      LLVM_DEBUG(llvm::dbgs() << "Src loop can't be removed: dst loop doesn't "
                                 "dominate dependence\n");
      return false;
    }

    hasOutDepsAfterFusion = true;
  }

  // If src loop has dependences after fusion or it writes to an live-out or
  // escaping memref, we can only remove it if the fusion slice is maximal so
  // that all the dependences are preserved.
  if (hasOutDepsAfterFusion || !escapingMemRefs.empty()) {
    Optional<bool> isMaximal = fusionSlice.isMaximal();
    if (!isMaximal) {
      LLVM_DEBUG(llvm::dbgs() << "Src loop can't be removed: can't determine "
                                 "if fusion is maximal\n");
      return false;
    }

    if (!*isMaximal) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Src loop can't be removed: fusion is not maximal\n");
      return false;
    }
  }

  return true;
}

/// Returns in 'srcIdCandidates' the producer fusion candidates for consumer
/// 'dstId'. Candidates are sorted by node id order. This order corresponds to
/// the program order when the 'mdg' is created. However, program order is not
/// guaranteed and must not be required by the client. Program order won't be
/// held if the 'mdg' is reused from a previous fusion step or if the node
/// creation order changes in the future to support more advance cases.
// TODO: Move this to a loop fusion utility once 'mdg' is also moved.
static void getProducerCandidates(unsigned dstId, MemRefDependenceGraph *mdg,
                                  SmallVectorImpl<unsigned> &srcIdCandidates) {
  // Skip if no input edges along which to fuse.
  if (mdg->inEdges.count(dstId) == 0)
    return;

  // Gather memrefs from loads in 'dstId'.
  auto *dstNode = mdg->getNode(dstId);
  DenseSet<Value> consumedMemrefs;
  for (Operation *load : dstNode->loads)
    consumedMemrefs.insert(cast<AffineReadOpInterface>(load).getMemRef());

  // Traverse 'dstId' incoming edges and gather the nodes that contain a store
  // to one of the consumed memrefs.
  for (auto &srcEdge : mdg->inEdges[dstId]) {
    auto *srcNode = mdg->getNode(srcEdge.id);
    // Skip if 'srcNode' is not a loop nest.
    if (!isa<AffineForOp>(srcNode->op))
      continue;

    if (any_of(srcNode->stores, [&](Operation *op) {
          auto storeOp = cast<AffineWriteOpInterface>(op);
          return consumedMemrefs.count(storeOp.getMemRef()) > 0;
        }))
      srcIdCandidates.push_back(srcNode->id);
  }

  llvm::sort(srcIdCandidates);
  srcIdCandidates.erase(
      std::unique(srcIdCandidates.begin(), srcIdCandidates.end()),
      srcIdCandidates.end());
}

/// Returns in 'producerConsumerMemrefs' the memrefs involved in a
/// producer-consumer dependence between 'srcId' and 'dstId'.
static void
gatherProducerConsumerMemrefs(unsigned srcId, unsigned dstId,
                              MemRefDependenceGraph *mdg,
                              DenseSet<Value> &producerConsumerMemrefs) {
  auto *dstNode = mdg->getNode(dstId);
  auto *srcNode = mdg->getNode(srcId);
  gatherProducerConsumerMemrefs(srcNode->stores, dstNode->loads,
                                producerConsumerMemrefs);
}

/// Returns in 'escapingMemRefs' the memrefs from affine store ops in node 'id'
/// that escape the function. A memref escapes the function if either:
///   1. It's a function argument, or
///   2. It's used by a non-affine op (e.g., std load/store, std call, etc.)
void gatherEscapingMemrefs(unsigned id, MemRefDependenceGraph *mdg,
                           DenseSet<Value> &escapingMemRefs) {
  auto *node = mdg->getNode(id);
  for (auto *storeOpInst : node->stores) {
    auto memref = cast<AffineWriteOpInterface>(storeOpInst).getMemRef();
    if (escapingMemRefs.count(memref))
      continue;
    // Check if 'memref' escapes because it's a block argument.
    if (memref.isa<BlockArgument>()) {
      escapingMemRefs.insert(memref);
      continue;
    }
    // Check if 'memref' escapes through a non-affine op (e.g., std load/store,
    // call op, etc.).
    for (Operation *user : memref.getUsers())
      if (!isa<AffineMapAccessInterface>(*user))
        escapingMemRefs.insert(memref);
  }
}

} // namespace

// Initializes the data dependence graph by walking operations in 'f'.
// Assigns each node in the graph a node id based on program order in 'f'.
// TODO: Add support for taking a Block arg to construct the
// dependence graph at a different depth.
bool MemRefDependenceGraph::init(func::FuncOp f) {
  LLVM_DEBUG(llvm::dbgs() << "--- Initializing MDG ---\n");
  DenseMap<Value, SetVector<unsigned>> memrefAccesses;

  // TODO: support multi-block functions.
  if (!llvm::hasSingleElement(f))
    return false;

  DenseMap<Operation *, unsigned> forToNodeMap;
  for (auto &op : f.front()) {
    if (auto forOp = dyn_cast<AffineForOp>(op)) {
      // Create graph node 'id' to represent top-level 'forOp' and record
      // all loads and store accesses it contains.
      LoopNestStateCollector collector;
      collector.collect(&op);
      // Return false if a region holding op other than 'affine.for' and
      // 'affine.if' was found (not currently supported).
      if (collector.hasNonAffineRegionOp)
        return false;
      Node node(nextNodeId++, &op);
      for (auto *opInst : collector.loadOpInsts) {
        node.loads.push_back(opInst);
        auto memref = cast<AffineReadOpInterface>(opInst).getMemRef();
        memrefAccesses[memref].insert(node.id);
      }
      for (auto *opInst : collector.storeOpInsts) {
        node.stores.push_back(opInst);
        auto memref = cast<AffineWriteOpInterface>(opInst).getMemRef();
        memrefAccesses[memref].insert(node.id);
      }
      forToNodeMap[&op] = node.id;
      nodes.insert({node.id, node});
    } else if (auto loadOp = dyn_cast<AffineReadOpInterface>(op)) {
      // Create graph node for top-level load op.
      Node node(nextNodeId++, &op);
      node.loads.push_back(&op);
      auto memref = cast<AffineReadOpInterface>(op).getMemRef();
      memrefAccesses[memref].insert(node.id);
      nodes.insert({node.id, node});
    } else if (auto storeOp = dyn_cast<AffineWriteOpInterface>(op)) {
      // Create graph node for top-level store op.
      Node node(nextNodeId++, &op);
      node.stores.push_back(&op);
      auto memref = cast<AffineWriteOpInterface>(op).getMemRef();
      memrefAccesses[memref].insert(node.id);
      nodes.insert({node.id, node});
    } else if (op.getNumRegions() != 0) {
      // Return false if another region is found (not currently supported).
      return false;
    } else if (op.getNumResults() > 0 && !op.use_empty()) {
      // Create graph node for top-level producer of SSA values, which
      // could be used by loop nest nodes.
      Node node(nextNodeId++, &op);
      nodes.insert({node.id, node});
    } else if (isa<CallOpInterface>(op)) {
      // Create graph node for top-level Call Op that takes any argument of
      // memref type. Call Op that returns one or more memref type results
      // is already taken care of, by the previous conditions.
      if (llvm::any_of(op.getOperandTypes(),
                       [&](Type t) { return t.isa<MemRefType>(); })) {
        Node node(nextNodeId++, &op);
        nodes.insert({node.id, node});
      }
    } else if (auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      // Create graph node for top-level op, which could have a memory write
      // side effect.
      SmallVector<MemoryEffects::EffectInstance, 1> effects;
      effectInterface.getEffects(effects);
      if (llvm::any_of(effects, [](const MemoryEffects::EffectInstance &it) {
            return isa<MemoryEffects::Write, MemoryEffects::Free>(
                it.getEffect());
          })) {
        Node node(nextNodeId++, &op);
        nodes.insert({node.id, node});
      }
    }
  }

  for (auto &idAndNode : nodes) {
    LLVM_DEBUG(llvm::dbgs() << "Create node " << idAndNode.first << " for:\n"
                            << *(idAndNode.second.op) << "\n");
    (void)idAndNode;
  }

  // Add dependence edges between nodes which produce SSA values and their
  // users. Load ops can be considered as the ones producing SSA values.
  for (auto &idAndNode : nodes) {
    const Node &node = idAndNode.second;
    // Stores don't define SSA values, skip them.
    if (!node.stores.empty())
      continue;
    auto *opInst = node.op;
    for (auto value : opInst->getResults()) {
      for (auto *user : value.getUsers()) {
        SmallVector<AffineForOp, 4> loops;
        getLoopIVs(*user, &loops);
        if (loops.empty())
          continue;
        assert(forToNodeMap.count(loops[0].getOperation()) > 0);
        unsigned userLoopNestId = forToNodeMap[loops[0].getOperation()];
        addEdge(node.id, userLoopNestId, value);
      }
    }
  }

  // Walk memref access lists and add graph edges between dependent nodes.
  for (auto &memrefAndList : memrefAccesses) {
    unsigned n = memrefAndList.second.size();
    for (unsigned i = 0; i < n; ++i) {
      unsigned srcId = memrefAndList.second[i];
      bool srcHasStore =
          getNode(srcId)->getStoreOpCount(memrefAndList.first) > 0;
      for (unsigned j = i + 1; j < n; ++j) {
        unsigned dstId = memrefAndList.second[j];
        bool dstHasStore =
            getNode(dstId)->getStoreOpCount(memrefAndList.first) > 0;
        if (srcHasStore || dstHasStore)
          addEdge(srcId, dstId, memrefAndList.first);
      }
    }
  }
  return true;
}

// Sinks all sequential loops to the innermost levels (while preserving
// relative order among them) and moves all parallel loops to the
// outermost (while again preserving relative order among them).
// This can increase the loop depth at which we can fuse a slice, since we are
// pushing loop carried dependence to a greater depth in the loop nest.
static void sinkSequentialLoops(MemRefDependenceGraph::Node *node) {
  assert(isa<AffineForOp>(node->op));
  AffineForOp newRootForOp = sinkSequentialLoops(cast<AffineForOp>(node->op));
  node->op = newRootForOp.getOperation();
}

//  TODO: improve/complete this when we have target data.
static unsigned getMemRefEltSizeInBytes(MemRefType memRefType) {
  auto elementType = memRefType.getElementType();

  unsigned sizeInBits;
  if (elementType.isIntOrFloat()) {
    sizeInBits = elementType.getIntOrFloatBitWidth();
  } else {
    auto vectorType = elementType.cast<VectorType>();
    sizeInBits =
        vectorType.getElementTypeBitWidth() * vectorType.getNumElements();
  }
  return llvm::divideCeil(sizeInBits, 8);
}

// Creates and returns a private (single-user) memref for fused loop rooted
// at 'forOp', with (potentially reduced) memref size based on the
// MemRefRegion written to by 'srcStoreOpInst' at depth 'dstLoopDepth'.
// TODO: consider refactoring the common code from generateDma and
// this one.
static Value createPrivateMemRef(AffineForOp forOp, Operation *srcStoreOpInst,
                                 unsigned dstLoopDepth,
                                 Optional<unsigned> fastMemorySpace,
                                 uint64_t localBufSizeThreshold) {
  auto *forInst = forOp.getOperation();

  // Create builder to insert alloc op just before 'forOp'.
  OpBuilder b(forInst);
  // Builder to create constants at the top level.
  OpBuilder top(forInst->getParentOfType<func::FuncOp>().getBody());
  // Create new memref type based on slice bounds.
  auto oldMemRef = cast<AffineWriteOpInterface>(srcStoreOpInst).getMemRef();
  auto oldMemRefType = oldMemRef.getType().cast<MemRefType>();
  unsigned rank = oldMemRefType.getRank();

  // Compute MemRefRegion for 'srcStoreOpInst' at depth 'dstLoopDepth'.
  MemRefRegion region(srcStoreOpInst->getLoc());
  bool validRegion = succeeded(region.compute(srcStoreOpInst, dstLoopDepth));
  (void)validRegion;
  assert(validRegion && "unexpected memref region failure");
  SmallVector<int64_t, 4> newShape;
  std::vector<SmallVector<int64_t, 4>> lbs;
  SmallVector<int64_t, 8> lbDivisors;
  lbs.reserve(rank);
  // Query 'region' for 'newShape' and lower bounds of MemRefRegion accessed
  // by 'srcStoreOpInst' at depth 'dstLoopDepth'.
  Optional<int64_t> numElements =
      region.getConstantBoundingSizeAndShape(&newShape, &lbs, &lbDivisors);
  assert(numElements && "non-constant number of elts in local buffer");

  const FlatAffineValueConstraints *cst = region.getConstraints();
  // 'outerIVs' holds the values that this memory region is symbolic/parametric
  // on; this would correspond to loop IVs surrounding the level at which the
  // slice is being materialized.
  SmallVector<Value, 8> outerIVs;
  cst->getValues(rank, cst->getNumVars(), &outerIVs);

  // Build 'rank' AffineExprs from MemRefRegion 'lbs'
  SmallVector<AffineExpr, 4> offsets;
  offsets.reserve(rank);
  for (unsigned d = 0; d < rank; ++d) {
    assert(lbs[d].size() == cst->getNumCols() - rank && "incorrect bound size");

    AffineExpr offset = top.getAffineConstantExpr(0);
    for (unsigned j = 0, e = cst->getNumCols() - rank - 1; j < e; j++) {
      offset = offset + lbs[d][j] * top.getAffineDimExpr(j);
    }
    assert(lbDivisors[d] > 0);
    offset =
        (offset + lbs[d][cst->getNumCols() - 1 - rank]).floorDiv(lbDivisors[d]);
    offsets.push_back(offset);
  }

  // Create 'newMemRefType' using 'newShape' from MemRefRegion accessed
  // by 'srcStoreOpInst'.
  uint64_t bufSize =
      getMemRefEltSizeInBytes(oldMemRefType) * numElements.value();
  unsigned newMemSpace;
  if (bufSize <= localBufSizeThreshold && fastMemorySpace.has_value()) {
    newMemSpace = fastMemorySpace.value();
  } else {
    newMemSpace = oldMemRefType.getMemorySpaceAsInt();
  }
  auto newMemRefType = MemRefType::get(newShape, oldMemRefType.getElementType(),
                                       {}, newMemSpace);

  // Create new private memref for fused loop 'forOp'. 'newShape' is always
  // a constant shape.
  // TODO: Create/move alloc ops for private memrefs closer to their
  // consumer loop nests to reduce their live range. Currently they are added
  // at the beginning of the function, because loop nests can be reordered
  // during the fusion pass.
  Value newMemRef = top.create<memref::AllocOp>(forOp.getLoc(), newMemRefType);

  // Build an AffineMap to remap access functions based on lower bound offsets.
  SmallVector<AffineExpr, 4> remapExprs;
  remapExprs.reserve(rank);
  for (unsigned i = 0; i < rank; i++) {
    auto dimExpr = b.getAffineDimExpr(outerIVs.size() + i);

    auto remapExpr =
        simplifyAffineExpr(dimExpr - offsets[i], outerIVs.size() + rank, 0);
    remapExprs.push_back(remapExpr);
  }

  auto indexRemap =
      AffineMap::get(outerIVs.size() + rank, 0, remapExprs, forOp.getContext());

  // Replace all users of 'oldMemRef' with 'newMemRef'.
  LogicalResult res =
      replaceAllMemRefUsesWith(oldMemRef, newMemRef, {}, indexRemap,
                               /*extraOperands=*/outerIVs,
                               /*symbolOperands=*/{},
                               /*domOpFilter=*/&*forOp.getBody()->begin());
  assert(succeeded(res) &&
         "replaceAllMemrefUsesWith should always succeed here");
  (void)res;
  return newMemRef;
}

/// Walking from node 'srcId' to node 'dstId' (exclusive of 'srcId' and
/// 'dstId'), if there is any non-affine operation accessing 'memref', return
/// true. Otherwise, return false.
static bool hasNonAffineUsersOnThePath(unsigned srcId, unsigned dstId,
                                       Value memref,
                                       MemRefDependenceGraph *mdg) {
  auto *srcNode = mdg->getNode(srcId);
  auto *dstNode = mdg->getNode(dstId);
  Value::user_range users = memref.getUsers();
  // For each MemRefDependenceGraph's node that is between 'srcNode' and
  // 'dstNode' (exclusive of 'srcNodes' and 'dstNode'), check whether any
  // non-affine operation in the node accesses the 'memref'.
  for (auto &idAndNode : mdg->nodes) {
    Operation *op = idAndNode.second.op;
    // Take care of operations between 'srcNode' and 'dstNode'.
    if (srcNode->op->isBeforeInBlock(op) && op->isBeforeInBlock(dstNode->op)) {
      // Walk inside the operation to find any use of the memref.
      // Interrupt the walk if found.
      auto walkResult = op->walk([&](Operation *user) {
        // Skip affine ops.
        if (isa<AffineMapAccessInterface>(*user))
          return WalkResult::advance();
        // Find a non-affine op that uses the memref.
        if (llvm::is_contained(users, user))
          return WalkResult::interrupt();
        return WalkResult::advance();
      });
      if (walkResult.wasInterrupted())
        return true;
    }
  }
  return false;
}

/// Check whether a memref value in node 'srcId' has a non-affine that
/// is between node 'srcId' and node 'dstId' (exclusive of 'srcNode' and
/// 'dstNode').
static bool hasNonAffineUsersOnThePath(unsigned srcId, unsigned dstId,
                                       MemRefDependenceGraph *mdg) {
  // Collect memref values in node 'srcId'.
  auto *srcNode = mdg->getNode(srcId);
  llvm::SmallDenseSet<Value, 2> memRefValues;
  srcNode->op->walk([&](Operation *op) {
    // Skip affine ops.
    if (isa<AffineForOp>(op))
      return WalkResult::advance();
    for (Value v : op->getOperands())
      // Collect memref values only.
      if (v.getType().isa<MemRefType>())
        memRefValues.insert(v);
    return WalkResult::advance();
  });
  // Looking for users between node 'srcId' and node 'dstId'.
  for (Value memref : memRefValues)
    if (hasNonAffineUsersOnThePath(srcId, dstId, memref, mdg))
      return true;
  return false;
}

// Checks the profitability of fusing a backwards slice of the loop nest
// surrounding 'srcOpInst' into the loop nest surrounding 'dstLoadOpInsts'.
// The argument 'srcStoreOpInst' is used to calculate the storage reduction on
// the memref being produced and consumed, which is an input to the cost model.
// For producer-consumer fusion, 'srcStoreOpInst' will be the same as
// 'srcOpInst', as we are slicing w.r.t to that producer. For input-reuse
// fusion, 'srcOpInst' will be the src loop nest LoadOp which reads from the
// same memref as dst loop nest load ops, and 'srcStoreOpInst' will be the
// unique store op in the src node, which will be used to check that the write
// region is the same after input-reuse fusion. Computation slices are provided
// in 'depthSliceUnions' for each legal fusion depth. The maximal depth at which
// fusion is legal is provided in 'maxLegalFusionDepth'. Returns true if it is
// profitable to fuse the candidate loop nests. Returns false otherwise.
// `dstLoopDepth` is set to the most profitable depth at which to materialize
// the source loop nest slice.
// The profitability model executes the following steps:
// *) Computes the backward computation slice at 'srcOpInst'. This
//    computation slice of the loop nest surrounding 'srcOpInst' is
//    represented by modified src loop bounds in 'sliceState', which are
//    functions of loop IVs in the loop nest surrounding 'srcOpInst'.
// *) Computes the cost of unfused src/dst loop nests (currently the cost of a
//    loop nest is the total number of dynamic operation instances in the loop
//    nest).
// *) Computes the cost of fusing a slice of the src loop nest into the dst
//    loop nest at various values of dst loop depth, attempting to fuse
//    the largest computation slice at the maximal dst loop depth (closest to
//    the load) to minimize reuse distance and potentially enable subsequent
//    load/store forwarding.
//    NOTE: 'dstLoopDepth' refers to the loop depth within the destination loop
//    nest, at which the src computation slice is inserted/fused.
//    NOTE: We attempt to maximize the dst loop depth, but there are cases
//    where a particular setting for 'dstLoopNest' might fuse an unsliced
//    loop (within the src computation slice) at a depth which results in
//    excessive recomputation (see unit tests for examples).
// *) Compares the total cost of the unfused loop nests to the min cost fused
//    loop nest computed in the previous step, and returns true if the latter
//    is lower.
// TODO: Extend profitability analysis to support scenarios with multiple
// stores.
static bool isFusionProfitable(Operation *srcOpInst, Operation *srcStoreOpInst,
                               AffineForOp dstForOp,
                               ArrayRef<ComputationSliceState> depthSliceUnions,
                               unsigned maxLegalFusionDepth,
                               unsigned *dstLoopDepth,
                               double computeToleranceThreshold) {
  LLVM_DEBUG({
    llvm::dbgs() << "Checking whether fusion is profitable between src op:\n";
    llvm::dbgs() << ' ' << *srcOpInst << " and destination loop:\n";
    llvm::dbgs() << dstForOp << "\n";
  });

  if (maxLegalFusionDepth == 0) {
    LLVM_DEBUG(llvm::dbgs() << "Can't fuse: maxLegalFusionDepth == 0 .\n");
    return false;
  }

  // Compute cost of sliced and unsliced src loop nest.
  SmallVector<AffineForOp, 4> srcLoopIVs;
  getLoopIVs(*srcOpInst, &srcLoopIVs);

  // Walk src loop nest and collect stats.
  LoopNestStats srcLoopNestStats;
  if (!getLoopNestStats(srcLoopIVs[0], &srcLoopNestStats))
    return false;

  // Compute cost of dst loop nest.
  LoopNestStats dstLoopNestStats;
  if (!getLoopNestStats(dstForOp, &dstLoopNestStats))
    return false;

  // Search for min cost value for 'dstLoopDepth'. At each value of
  // 'dstLoopDepth' from 'maxLegalLoopDepth' to '1', compute computation slice
  // bounds between 'srcOpInst' and each op in 'dstOpinsts' (taking the union
  // of these bounds). Next the union slice bounds are used to calculate
  // the cost of the slice and the cost of the slice inserted into the dst
  // loop nest at 'dstLoopDepth'.
  uint64_t minFusedLoopNestComputeCost = std::numeric_limits<uint64_t>::max();
  double maxStorageReduction = 0.0;
  Optional<uint64_t> sliceMemEstimate;

  // The best loop depth at which to materialize the slice.
  Optional<unsigned> bestDstLoopDepth;

  // Compute op instance count for the src loop nest without iteration slicing.
  uint64_t srcLoopNestCost = getComputeCost(srcLoopIVs[0], srcLoopNestStats);

  // Compute src loop nest write region size.
  MemRefRegion srcWriteRegion(srcStoreOpInst->getLoc());
  if (failed(srcWriteRegion.compute(srcStoreOpInst, /*loopDepth=*/0))) {
    LLVM_DEBUG(llvm::dbgs()
               << "Unable to compute MemRefRegion for source operation\n.");
    return false;
  }

  Optional<int64_t> maybeSrcWriteRegionSizeBytes =
      srcWriteRegion.getRegionSize();
  if (!maybeSrcWriteRegionSizeBytes.has_value())
    return false;
  int64_t srcWriteRegionSizeBytes = maybeSrcWriteRegionSizeBytes.value();

  // Compute op instance count for the src loop nest.
  uint64_t dstLoopNestCost = getComputeCost(dstForOp, dstLoopNestStats);

  // Evaluate all depth choices for materializing the slice in the destination
  // loop nest.
  for (unsigned i = maxLegalFusionDepth; i >= 1; --i) {
    const ComputationSliceState &slice = depthSliceUnions[i - 1];
    // Skip slice union if it wasn't computed for this depth.
    if (slice.isEmpty())
      continue;

    int64_t fusedLoopNestComputeCost;
    if (!getFusionComputeCost(srcLoopIVs[0], srcLoopNestStats, dstForOp,
                              dstLoopNestStats, slice,
                              &fusedLoopNestComputeCost)) {
      LLVM_DEBUG(llvm::dbgs() << "Unable to compute fusion compute cost.\n.");
      continue;
    }

    double additionalComputeFraction =
        fusedLoopNestComputeCost /
            (static_cast<double>(srcLoopNestCost) + dstLoopNestCost) -
        1;

    // Determine what the slice write MemRefRegion would be, if the src loop
    // nest slice 'slice' were to be inserted into the dst loop nest at loop
    // depth 'i'.
    MemRefRegion sliceWriteRegion(srcStoreOpInst->getLoc());
    if (failed(sliceWriteRegion.compute(srcStoreOpInst, /*loopDepth=*/0,
                                        &slice))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to compute slice write region at loopDepth: " << i
                 << "\n");
      continue;
    }

    Optional<int64_t> maybeSliceWriteRegionSizeBytes =
        sliceWriteRegion.getRegionSize();
    if (!maybeSliceWriteRegionSizeBytes.has_value() ||
        maybeSliceWriteRegionSizeBytes.value() == 0) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to get slice write region size at loopDepth: " << i
                 << "\n");
      continue;
    }
    int64_t sliceWriteRegionSizeBytes = maybeSliceWriteRegionSizeBytes.value();

    // If we are fusing for reuse, check that write regions remain the same.
    // TODO: Write region check should check sizes and offsets in
    // each dimension, so that we are sure they are covering the same memref
    // region. Also, move this out to a isMemRefRegionSuperSet helper function.
    if (srcOpInst != srcStoreOpInst &&
        sliceWriteRegionSizeBytes != srcWriteRegionSizeBytes)
      continue;

    double storageReduction = static_cast<double>(srcWriteRegionSizeBytes) /
                              static_cast<double>(sliceWriteRegionSizeBytes);

    LLVM_DEBUG({
      std::stringstream msg;
      msg << "  evaluating fusion profitability at depth : " << i << "\n"
          << std::fixed << std::setprecision(2)
          << "   additional compute fraction: "
          << 100.0 * additionalComputeFraction << "%\n"
          << "   storage reduction factor: " << storageReduction << "x\n"
          << "   fused nest cost: " << fusedLoopNestComputeCost << "\n"
          << "   src write region size: " << srcWriteRegionSizeBytes << "\n"
          << "   slice write region size: " << sliceWriteRegionSizeBytes
          << "\n";
      llvm::dbgs() << msg.str();
    });

    // TODO: This is a placeholder cost model.
    // Among all choices that add an acceptable amount of redundant computation
    // (as per computeToleranceThreshold), we will simply pick the one that
    // reduces the intermediary size the most.
    if ((storageReduction > maxStorageReduction) &&
        (additionalComputeFraction < computeToleranceThreshold)) {
      maxStorageReduction = storageReduction;
      bestDstLoopDepth = i;
      minFusedLoopNestComputeCost = fusedLoopNestComputeCost;
      sliceMemEstimate = sliceWriteRegionSizeBytes;
    }
  }

  // A simple cost model: fuse if it reduces the memory footprint.

  if (!bestDstLoopDepth) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "All fusion choices involve more than the threshold amount of "
           "redundant computation; NOT fusing.\n");
    return false;
  }

  if (!bestDstLoopDepth) {
    LLVM_DEBUG(llvm::dbgs() << "no fusion depth could be evaluated.\n");
    return false;
  }

  // Set dstLoopDepth based on best values from search.
  *dstLoopDepth = *bestDstLoopDepth;

  LLVM_DEBUG(
      llvm::dbgs() << " LoopFusion fusion stats:"
                   << "\n  best loop depth: " << bestDstLoopDepth
                   << "\n  src loop nest compute cost: " << srcLoopNestCost
                   << "\n  dst loop nest compute cost: " << dstLoopNestCost
                   << "\n  fused loop nest compute cost: "
                   << minFusedLoopNestComputeCost << "\n");

  auto dstMemSize = getMemoryFootprintBytes(dstForOp);
  auto srcMemSize = getMemoryFootprintBytes(srcLoopIVs[0]);

  Optional<double> storageReduction;

  if (!dstMemSize || !srcMemSize) {
    LLVM_DEBUG(llvm::dbgs()
               << "  fusion memory benefit cannot be evaluated; NOT fusing.\n");
    return false;
  }

  auto srcMemSizeVal = srcMemSize.value();
  auto dstMemSizeVal = dstMemSize.value();

  assert(sliceMemEstimate && "expected value");
  auto fusedMem = dstMemSizeVal + sliceMemEstimate.value();

  LLVM_DEBUG(llvm::dbgs() << "   src mem: " << srcMemSizeVal << "\n"
                          << "   dst mem: " << dstMemSizeVal << "\n"
                          << "   fused mem: " << fusedMem << "\n"
                          << "   slice mem: " << sliceMemEstimate << "\n");

  if (static_cast<long>(fusedMem) > srcMemSizeVal + dstMemSizeVal) {
    LLVM_DEBUG(llvm::dbgs() << "Fusion is not profitable; NOT fusing.\n");
    return false;
  }
  storageReduction =
      100.0 *
      (1.0 - fusedMem / (static_cast<double>(srcMemSizeVal) + dstMemSizeVal));

  double additionalComputeFraction =
      100.0 * (minFusedLoopNestComputeCost /
                   (static_cast<double>(srcLoopNestCost) + dstLoopNestCost) -
               1);
  (void)additionalComputeFraction;
  LLVM_DEBUG({
    std::stringstream msg;
    msg << " fusion is most profitable at depth " << *dstLoopDepth << " with "
        << std::setprecision(2) << additionalComputeFraction
        << "% redundant computation and a ";
    msg << (storageReduction ? std::to_string(*storageReduction) : "<unknown>");
    msg << "% storage reduction.\n";
    llvm::dbgs() << msg.str();
  });

  return true;
}

namespace {

// GreedyFusion greedily fuses loop nests which have a producer/consumer or
// input-reuse relationship on a memref, with the goal of improving locality.
//
// The steps of the producer-consumer fusion algorithm are as follows:
//
// *) A worklist is initialized with node ids from the dependence graph.
// *) For each node id in the worklist:
//   *) Pop an AffineForOp of the worklist. This 'dstAffineForOp' will be a
//      candidate destination AffineForOp into which fusion will be attempted.
//   *) Add each LoadOp currently in 'dstAffineForOp' into list 'dstLoadOps'.
//   *) For each LoadOp in 'dstLoadOps' do:
//      *) Look up dependent loop nests which have a single store op to the same
//         memref.
//      *) Check if dependences would be violated by the fusion.
//      *) Get a computation slice of 'srcLoopNest', which adjusts its loop
//         bounds to be functions of 'dstLoopNest' IVs and symbols.
//      *) Fuse the 'srcLoopNest' computation slice into the 'dstLoopNest',
//         at a loop depth determined by the cost model in 'isFusionProfitable'.
//      *) Add the newly fused load/store operations to the state,
//         and also add newly fused load ops to 'dstLoopOps' to be considered
//         as fusion dst load ops in another iteration.
//      *) Remove old src loop nest and its associated state.
//
// The steps of the input-reuse fusion algorithm are as follows:
//
// *) Initialize 'worklist' with node ids from the dependence graph.
// *) For each 'dstNode' in the worklist:
//   *) Find a candidate sibling node 'sibNode' to fuse with 'dstNode' which
//      loads from the same memref, but which has no dependence paths to/from.
//   *) Get a computation slice of 'sibLoopNest', which adjusts its loop
//      bounds to be functions of 'dstLoopNest' IVs and symbols.
//   *) Fuse the 'sibLoopNest' computation slice into the 'dstLoopNest',
//      at a loop depth determined by the cost model in 'isFusionProfitable'.
//      This function also checks that the memref write region of 'sibLoopNest',
//      is preserved in the fused loop nest.
//   *) Update graph state to reflect the fusion of 'sibNode' into 'dstNode'.
//
// Given a graph where top-level operations are vertices in the set 'V' and
// edges in the set 'E' are dependences between vertices, this algorithm
// takes O(V) time for initialization, and has runtime O(V + E).
//
// This greedy algorithm is not 'maximal' due to the current restriction of
// fusing along single producer consumer edges, but there is a TODO: to fix
// this.
//
// TODO: Experiment with other fusion policies.
struct GreedyFusion {
public:
  // The data dependence graph to traverse during fusion.
  MemRefDependenceGraph *mdg;
  // Worklist of graph nodes visited during the fusion pass.
  SmallVector<unsigned, 8> worklist;
  // Parameter for local buffer size threshold.
  unsigned localBufSizeThreshold;
  // Parameter for fast memory space.
  Optional<unsigned> fastMemorySpace;
  // If true, ignore any additional (redundant) computation tolerance threshold
  // that would have prevented fusion.
  bool maximalFusion;
  // The amount of additional computation that is tolerated while fusing
  // pair-wise as a fraction of the total computation.
  double computeToleranceThreshold;

  using Node = MemRefDependenceGraph::Node;

  GreedyFusion(MemRefDependenceGraph *mdg, unsigned localBufSizeThreshold,
               Optional<unsigned> fastMemorySpace, bool maximalFusion,
               double computeToleranceThreshold)
      : mdg(mdg), localBufSizeThreshold(localBufSizeThreshold),
        fastMemorySpace(fastMemorySpace), maximalFusion(maximalFusion),
        computeToleranceThreshold(computeToleranceThreshold) {}

  /// Initializes 'worklist' with nodes from 'mdg'.
  void init() {
    // TODO: Add a priority queue for prioritizing nodes by different
    // metrics (e.g. arithmetic intensity/flops-to-bytes ratio).
    worklist.clear();
    for (auto &idAndNode : mdg->nodes) {
      const Node &node = idAndNode.second;
      worklist.push_back(node.id);
    }
  }
  /// Run only sibling fusion on the `mdg`.
  void runSiblingFusionOnly() {
    fuseSiblingNodes();
    eraseUnusedMemRefAllocations();
  }

  /// Run only producer/consumer fusion on the `mdg`.
  void runProducerConsumerFusionOnly() {
    fuseProducerConsumerNodes(
        /*maxSrcUserCount=*/std::numeric_limits<unsigned>::max());
    eraseUnusedMemRefAllocations();
  }

  // Run the GreedyFusion pass.
  // *) First pass through the nodes fuses single-use producer nodes into their
  //    unique consumer.
  // *) Second pass fuses sibling nodes which share no dependence edges.
  // *) Third pass fuses any remaining producer nodes into their users.
  void runGreedyFusion() {
    // TODO: Run this repeatedly until a fixed-point is reached.
    fuseProducerConsumerNodes(/*maxSrcUserCount=*/1);
    fuseSiblingNodes();
    fuseProducerConsumerNodes(
        /*maxSrcUserCount=*/std::numeric_limits<unsigned>::max());
    eraseUnusedMemRefAllocations();
  }

  void fuseProducerConsumerNodes(unsigned maxSrcUserCount) {
    LLVM_DEBUG(llvm::dbgs() << "--- Producer/Consumer Fusion ---\n");
    init();
    while (!worklist.empty()) {
      unsigned dstId = worklist.back();
      worklist.pop_back();

      // Skip if this node was removed (fused into another node).
      if (mdg->nodes.count(dstId) == 0)
        continue;
      // Get 'dstNode' into which to attempt fusion.
      auto *dstNode = mdg->getNode(dstId);
      // Skip if 'dstNode' is not a loop nest.
      if (!isa<AffineForOp>(dstNode->op))
        continue;
      // Skip if 'dstNode' is a loop nest returning values.
      // TODO: support loop nests that return values.
      if (dstNode->op->getNumResults() > 0)
        continue;

      LLVM_DEBUG(llvm::dbgs() << "Evaluating dst loop " << dstId << "\n");

      // Sink sequential loops in 'dstNode' (and thus raise parallel loops)
      // while preserving relative order. This can increase the maximum loop
      // depth at which we can fuse a slice of a producer loop nest into a
      // consumer loop nest.
      sinkSequentialLoops(dstNode);
      auto dstAffineForOp = cast<AffineForOp>(dstNode->op);

      // Try to fuse 'dstNode' with candidate producer loops until a fixed point
      // is reached. Fusing two loops may expose new fusion opportunities.
      bool dstNodeChanged;
      do {
        // Gather src loop candidates for 'dstNode' and visit them in "quasi"
        // reverse program order to minimize the number of iterations needed to
        // reach the fixed point. Note that this is a best effort approach since
        // 'getProducerCandidates' does not always guarantee that program order
        // in 'srcIdCandidates'.
        dstNodeChanged = false;
        SmallVector<unsigned, 16> srcIdCandidates;
        getProducerCandidates(dstId, mdg, srcIdCandidates);

        for (unsigned srcId : llvm::reverse(srcIdCandidates)) {
          // Get 'srcNode' from which to attempt fusion into 'dstNode'.
          auto *srcNode = mdg->getNode(srcId);
          auto srcAffineForOp = cast<AffineForOp>(srcNode->op);
          LLVM_DEBUG(llvm::dbgs() << "Evaluating src loop " << srcId
                                  << " for dst loop " << dstId << "\n");

          // Skip if 'srcNode' is a loop nest returning values.
          // TODO: support loop nests that return values.
          if (isa<AffineForOp>(srcNode->op) && srcNode->op->getNumResults() > 0)
            continue;

          DenseSet<Value> producerConsumerMemrefs;
          gatherProducerConsumerMemrefs(srcId, dstId, mdg,
                                        producerConsumerMemrefs);

          // Skip if 'srcNode' out edge count on any memref is greater than
          // 'maxSrcUserCount'.
          if (any_of(producerConsumerMemrefs, [&](Value memref) {
                return mdg->getOutEdgeCount(srcNode->id, memref) >
                       maxSrcUserCount;
              }))
            continue;

          // Gather memrefs in 'srcNode' that are written and escape to the
          // function (e.g., memref function arguments, returned memrefs,
          // memrefs passed to function calls, etc.).
          DenseSet<Value> srcEscapingMemRefs;
          gatherEscapingMemrefs(srcNode->id, mdg, srcEscapingMemRefs);

          // Skip if there are non-affine operations in between the 'srcNode'
          // and 'dstNode' using their memrefs. If so, we wouldn't be able to
          // compute a legal insertion point for now. 'srcNode' and 'dstNode'
          // memrefs with non-affine operation users would be considered
          // escaping memrefs so we can limit this check to only scenarios with
          // escaping memrefs.
          if (!srcEscapingMemRefs.empty() &&
              hasNonAffineUsersOnThePath(srcId, dstId, mdg)) {
            LLVM_DEBUG(
                llvm::dbgs()
                << "Can't fuse: non-affine users in between the loops\n.");
            continue;
          }

          // Compute an operation list insertion point for the fused loop
          // nest which preserves dependences.
          Operation *fusedLoopInsPoint =
              mdg->getFusedLoopNestInsertionPoint(srcNode->id, dstNode->id);
          if (fusedLoopInsPoint == nullptr)
            continue;

          // Compute the innermost common loop depth for dstNode
          // producer-consumer loads/stores.
          SmallVector<Operation *, 2> dstMemrefOps;
          for (Operation *op : dstNode->loads)
            if (producerConsumerMemrefs.count(
                    cast<AffineReadOpInterface>(op).getMemRef()) > 0)
              dstMemrefOps.push_back(op);
          for (Operation *op : dstNode->stores)
            if (producerConsumerMemrefs.count(
                    cast<AffineWriteOpInterface>(op).getMemRef()))
              dstMemrefOps.push_back(op);
          unsigned dstLoopDepthTest = getInnermostCommonLoopDepth(dstMemrefOps);

          // Check the feasibility of fusing src loop nest into dst loop nest
          // at loop depths in range [1, dstLoopDepthTest].
          unsigned maxLegalFusionDepth = 0;
          SmallVector<ComputationSliceState, 8> depthSliceUnions;
          depthSliceUnions.resize(dstLoopDepthTest);
          FusionStrategy strategy(FusionStrategy::ProducerConsumer);
          for (unsigned i = 1; i <= dstLoopDepthTest; ++i) {
            FusionResult result = mlir::canFuseLoops(
                srcAffineForOp, dstAffineForOp,
                /*dstLoopDepth=*/i, &depthSliceUnions[i - 1], strategy);

            if (result.value == FusionResult::Success)
              maxLegalFusionDepth = i;
          }

          if (maxLegalFusionDepth == 0) {
            LLVM_DEBUG(llvm::dbgs()
                       << "Can't fuse: fusion is not legal at any depth\n");
            continue;
          }

          // Check if fusion would be profitable. We skip profitability analysis
          // for maximal fusion since we already know the maximal legal depth to
          // fuse.
          unsigned bestDstLoopDepth = maxLegalFusionDepth;
          if (!maximalFusion) {
            // Retrieve producer stores from the src loop.
            SmallVector<Operation *, 2> producerStores;
            for (Operation *op : srcNode->stores)
              if (producerConsumerMemrefs.count(
                      cast<AffineWriteOpInterface>(op).getMemRef()))
                producerStores.push_back(op);

            // TODO: Suppport multiple producer stores in profitability
            // analysis. We limit profitability analysis to only scenarios with
            // a single producer store for now. Note that some multi-store
            // producer scenarios will still go through profitability analysis
            // if only one of the stores is involved the producer-consumer
            // relationship of the candidate loops.
            assert(!producerStores.empty() && "Expected producer store");
            if (producerStores.size() > 1)
              LLVM_DEBUG(llvm::dbgs() << "Skipping profitability analysis. Not "
                                         "supported for this case\n");
            else if (!isFusionProfitable(producerStores[0], producerStores[0],
                                         dstAffineForOp, depthSliceUnions,
                                         maxLegalFusionDepth, &bestDstLoopDepth,
                                         computeToleranceThreshold))
              continue;
          }

          assert(bestDstLoopDepth > 0 && "Unexpected loop fusion depth");
          ComputationSliceState &bestSlice =
              depthSliceUnions[bestDstLoopDepth - 1];
          assert(!bestSlice.isEmpty() && "Missing slice union for depth");

          // Determine if 'srcId' can be removed after fusion, taking into
          // account remaining dependences, escaping memrefs and the fusion
          // insertion point.
          bool removeSrcNode = canRemoveSrcNodeAfterFusion(
              srcId, dstId, bestSlice, fusedLoopInsPoint, srcEscapingMemRefs,
              mdg);

          DenseSet<Value> privateMemrefs;
          for (Value memref : producerConsumerMemrefs) {
            // If `memref` is an escaping one, do not create a private memref
            // for the below scenarios, since doing so will leave the escaping
            // memref unmodified as all the writes originally meant for the
            // escaping memref would be performed on the private memref:
            // 1. The source is to be removed after fusion,
            // OR
            // 2. The destination writes to `memref`.
            if (srcEscapingMemRefs.count(memref) > 0 &&
                (removeSrcNode || dstNode->getStoreOpCount(memref) > 0))
              continue;

            // Don't create a private memref if 'srcNode' has in edges on
            // 'memref' or 'dstNode' has out edges on 'memref'.
            if (mdg->getIncomingMemRefAccesses(srcId, memref) > 0 ||
                mdg->getOutEdgeCount(dstId, memref) > 0)
              continue;

            // If 'srcNode' will be removed but it has out edges on 'memref' to
            // nodes other than 'dstNode', we have to preserve dependences and
            // cannot create a private memref.
            if (removeSrcNode &&
                any_of(mdg->outEdges[srcId], [&](const auto &edge) {
                  return edge.value == memref && edge.id != dstId;
                }))
              continue;

            // Create a private version of this memref.
            privateMemrefs.insert(memref);
          }

          // Fuse computation slice of 'srcLoopNest' into 'dstLoopNest'.
          fuseLoops(srcAffineForOp, dstAffineForOp, bestSlice);
          dstNodeChanged = true;

          LLVM_DEBUG(llvm::dbgs()
                     << "Fused src loop " << srcId << " into dst loop " << dstId
                     << " at depth " << bestDstLoopDepth << ":\n"
                     << dstAffineForOp << "\n");

          // Move 'dstAffineForOp' before 'insertPointInst' if needed.
          if (fusedLoopInsPoint != dstAffineForOp.getOperation())
            dstAffineForOp.getOperation()->moveBefore(fusedLoopInsPoint);

          // Update edges between 'srcNode' and 'dstNode'.
          mdg->updateEdges(srcNode->id, dstNode->id, privateMemrefs,
                           removeSrcNode);

          // Create private memrefs.
          if (!privateMemrefs.empty()) {
            // Gather stores for all the private-to-be memrefs.
            DenseMap<Value, SmallVector<Operation *, 4>> privateMemRefToStores;
            dstAffineForOp.walk([&](AffineWriteOpInterface storeOp) {
              Value storeMemRef = storeOp.getMemRef();
              if (privateMemrefs.count(storeMemRef) > 0)
                privateMemRefToStores[storeMemRef].push_back(
                    storeOp.getOperation());
            });

            // Replace original memrefs with private memrefs. Note that all the
            // loads and stores on these memrefs will be replaced with a new
            // loads and stores. Any reference to the original ones becomes
            // invalid after this point.
            for (auto &memrefToStoresPair : privateMemRefToStores) {
              // TODO: Use union of memref write regions to compute
              // private memref footprint.
              SmallVector<Operation *, 4> &storesForMemref =
                  memrefToStoresPair.second;
              Value newMemRef = createPrivateMemRef(
                  dstAffineForOp, storesForMemref[0], bestDstLoopDepth,
                  fastMemorySpace, localBufSizeThreshold);
              // Create new node in dependence graph for 'newMemRef' alloc op.
              unsigned newMemRefNodeId =
                  mdg->addNode(newMemRef.getDefiningOp());
              // Add edge from 'newMemRef' node to dstNode.
              mdg->addEdge(newMemRefNodeId, dstId, newMemRef);
            }
            // One or more entries for 'newMemRef' alloc op are inserted into
            // the DenseMap mdg->nodes. Since an insertion may cause DenseMap to
            // reallocate, update dstNode.
            dstNode = mdg->getNode(dstId);
          }

          // Collect dst loop stats after memref privatization transformation.
          LoopNestStateCollector dstLoopCollector;
          dstLoopCollector.collect(dstAffineForOp.getOperation());

          // Clear and add back loads and stores.
          mdg->clearNodeLoadAndStores(dstNode->id);
          mdg->addToNode(dstId, dstLoopCollector.loadOpInsts,
                         dstLoopCollector.storeOpInsts);

          if (removeSrcNode) {
            LLVM_DEBUG(llvm::dbgs()
                       << "Removing src loop " << srcId << " after fusion\n");
            // srcNode is no longer valid after it is removed from mdg.
            srcAffineForOp.erase();
            mdg->removeNode(srcId);
            srcNode = nullptr;
          }
        }
      } while (dstNodeChanged);
    }
  }

  // Visits each node in the graph, and for each node, attempts to fuse it with
  // its sibling nodes (nodes which share a parent, but no dependence edges).
  void fuseSiblingNodes() {
    LLVM_DEBUG(llvm::dbgs() << "--- Sibling Fusion ---\n");
    init();
    while (!worklist.empty()) {
      unsigned dstId = worklist.back();
      worklist.pop_back();

      // Skip if this node was removed (fused into another node).
      if (mdg->nodes.count(dstId) == 0)
        continue;
      // Get 'dstNode' into which to attempt fusion.
      auto *dstNode = mdg->getNode(dstId);
      // Skip if 'dstNode' is not a loop nest.
      if (!isa<AffineForOp>(dstNode->op))
        continue;
      // Attempt to fuse 'dstNode' with its sibling nodes in the graph.
      fuseWithSiblingNodes(dstNode);
    }
  }

  // Attempt to fuse 'dstNode' with sibling nodes in the graph.
  void fuseWithSiblingNodes(Node *dstNode) {
    DenseSet<unsigned> visitedSibNodeIds;
    std::pair<unsigned, Value> idAndMemref;
    auto dstAffineForOp = cast<AffineForOp>(dstNode->op);

    while (findSiblingNodeToFuse(dstNode, &visitedSibNodeIds, &idAndMemref)) {
      unsigned sibId = idAndMemref.first;
      Value memref = idAndMemref.second;
      // TODO: Check that 'sibStoreOpInst' post-dominates all other
      // stores to the same memref in 'sibNode' loop nest.
      auto *sibNode = mdg->getNode(sibId);
      // Compute an operation list insertion point for the fused loop
      // nest which preserves dependences.
      assert(sibNode->op->getBlock() == dstNode->op->getBlock());
      Operation *insertPointInst =
          sibNode->op->isBeforeInBlock(dstNode->op)
              ? mdg->getFusedLoopNestInsertionPoint(sibNode->id, dstNode->id)
              : mdg->getFusedLoopNestInsertionPoint(dstNode->id, sibNode->id);
      if (insertPointInst == nullptr)
        continue;

      // Check if fusion would be profitable and at what depth.

      // Get unique 'sibNode' load op to 'memref'.
      SmallVector<Operation *, 2> sibLoadOpInsts;
      sibNode->getLoadOpsForMemref(memref, &sibLoadOpInsts);
      // Currently findSiblingNodeToFuse searches for siblings with one load.
      assert(sibLoadOpInsts.size() == 1);
      Operation *sibLoadOpInst = sibLoadOpInsts[0];
      assert(!sibNode->stores.empty());
      // TODO: Choose the store which postdominates all other stores.
      auto *sibStoreOpInst = sibNode->stores.back();

      // Gather 'dstNode' load ops to 'memref'.
      SmallVector<Operation *, 2> dstLoadOpInsts;
      dstNode->getLoadOpsForMemref(memref, &dstLoadOpInsts);

      SmallVector<AffineForOp, 4> dstLoopIVs;
      getLoopIVs(*dstLoadOpInsts[0], &dstLoopIVs);
      unsigned dstLoopDepthTest = dstLoopIVs.size();
      auto sibAffineForOp = cast<AffineForOp>(sibNode->op);

      // Compute loop depth and slice union for fusion.
      SmallVector<ComputationSliceState, 8> depthSliceUnions;
      depthSliceUnions.resize(dstLoopDepthTest);
      unsigned maxLegalFusionDepth = 0;
      FusionStrategy strategy(memref);
      for (unsigned i = 1; i <= dstLoopDepthTest; ++i) {
        FusionResult result = mlir::canFuseLoops(
            sibAffineForOp, dstAffineForOp,
            /*dstLoopDepth=*/i, &depthSliceUnions[i - 1], strategy);

        if (result.value == FusionResult::Success)
          maxLegalFusionDepth = i;
      }

      // Skip if fusion is not feasible at any loop depths.
      if (maxLegalFusionDepth == 0)
        continue;

      unsigned bestDstLoopDepth = maxLegalFusionDepth;
      if (!maximalFusion) {
        // Check if fusion would be profitable.
        if (!isFusionProfitable(sibLoadOpInst, sibStoreOpInst, dstAffineForOp,
                                depthSliceUnions, maxLegalFusionDepth,
                                &bestDstLoopDepth, computeToleranceThreshold))
          continue;
      }

      assert(bestDstLoopDepth > 0 && "Unexpected loop fusion depth");
      assert(!depthSliceUnions[bestDstLoopDepth - 1].isEmpty() &&
             "Fusion depth has no computed slice union");
      // Check if source loop is being inserted in the innermost
      // destination loop. Based on this, the fused loop may be optimized
      // further inside `fuseLoops`.
      bool isInnermostInsertion = (bestDstLoopDepth == dstLoopDepthTest);
      // Fuse computation slice of 'sibLoopNest' into 'dstLoopNest'.
      mlir::fuseLoops(sibAffineForOp, dstAffineForOp,
                      depthSliceUnions[bestDstLoopDepth - 1],
                      isInnermostInsertion);

      auto dstForInst = cast<AffineForOp>(dstNode->op);
      // Update operation position of fused loop nest (if needed).
      if (insertPointInst != dstForInst.getOperation()) {
        dstForInst->moveBefore(insertPointInst);
      }
      // Update data dependence graph state post fusion.
      updateStateAfterSiblingFusion(sibNode, dstNode);
    }
  }

  // Searches function argument uses and the graph from 'dstNode' looking for a
  // fusion candidate sibling node which shares no dependences with 'dstNode'
  // but which loads from the same memref. Returns true and sets
  // 'idAndMemrefToFuse' on success. Returns false otherwise.
  bool findSiblingNodeToFuse(Node *dstNode,
                             DenseSet<unsigned> *visitedSibNodeIds,
                             std::pair<unsigned, Value> *idAndMemrefToFuse) {
    // Returns true if 'sibNode' can be fused with 'dstNode' for input reuse
    // on 'memref'.
    auto canFuseWithSibNode = [&](Node *sibNode, Value memref) {
      // Skip if 'outEdge' is not a read-after-write dependence.
      // TODO: Remove restrict to single load op restriction.
      if (sibNode->getLoadOpCount(memref) != 1)
        return false;
      // Skip if there exists a path of dependent edges between
      // 'sibNode' and 'dstNode'.
      if (mdg->hasDependencePath(sibNode->id, dstNode->id) ||
          mdg->hasDependencePath(dstNode->id, sibNode->id))
        return false;
      // Skip sib node if it loads to (and stores from) the same memref on
      // which it also has an input dependence edge.
      DenseSet<Value> loadAndStoreMemrefSet;
      sibNode->getLoadAndStoreMemrefSet(&loadAndStoreMemrefSet);
      if (llvm::any_of(loadAndStoreMemrefSet, [=](Value memref) {
            return mdg->getIncomingMemRefAccesses(sibNode->id, memref) > 0;
          }))
        return false;

      // Check that all stores are to the same memref.
      DenseSet<Value> storeMemrefs;
      for (auto *storeOpInst : sibNode->stores) {
        storeMemrefs.insert(
            cast<AffineWriteOpInterface>(storeOpInst).getMemRef());
      }
      if (storeMemrefs.size() != 1)
        return false;

      // Skip if a memref value in one node is used by a non-affine memref
      // access that lies between 'dstNode' and 'sibNode'.
      if (hasNonAffineUsersOnThePath(dstNode->id, sibNode->id, mdg) ||
          hasNonAffineUsersOnThePath(sibNode->id, dstNode->id, mdg))
        return false;
      return true;
    };

    // Search for siblings which load the same memref function argument.
    auto fn = dstNode->op->getParentOfType<func::FuncOp>();
    for (unsigned i = 0, e = fn.getNumArguments(); i != e; ++i) {
      for (auto *user : fn.getArgument(i).getUsers()) {
        if (auto loadOp = dyn_cast<AffineReadOpInterface>(user)) {
          // Gather loops surrounding 'use'.
          SmallVector<AffineForOp, 4> loops;
          getLoopIVs(*user, &loops);
          // Skip 'use' if it is not within a loop nest.
          if (loops.empty())
            continue;
          Node *sibNode = mdg->getForOpNode(loops[0]);
          assert(sibNode != nullptr);
          // Skip 'use' if it not a sibling to 'dstNode'.
          if (sibNode->id == dstNode->id)
            continue;
          // Skip 'use' if it has been visited.
          if (visitedSibNodeIds->count(sibNode->id) > 0)
            continue;
          // Skip 'use' if it does not load from the same memref as 'dstNode'.
          auto memref = loadOp.getMemRef();
          if (dstNode->getLoadOpCount(memref) == 0)
            continue;
          // Check if 'sibNode/dstNode' can be input-reuse fused on 'memref'.
          if (canFuseWithSibNode(sibNode, memref)) {
            visitedSibNodeIds->insert(sibNode->id);
            idAndMemrefToFuse->first = sibNode->id;
            idAndMemrefToFuse->second = memref;
            return true;
          }
        }
      }
    }

    // Search for siblings by following edges through an intermediate src node.
    // Collect candidate 'dstNode' input edges in 'inEdges'.
    SmallVector<MemRefDependenceGraph::Edge, 2> inEdges;
    mdg->forEachMemRefInputEdge(
        dstNode->id, [&](MemRefDependenceGraph::Edge inEdge) {
          // Add 'inEdge' if it is a read-after-write dependence.
          if (dstNode->getLoadOpCount(inEdge.value) > 0 &&
              mdg->getNode(inEdge.id)->getStoreOpCount(inEdge.value) > 0)
            inEdges.push_back(inEdge);
        });

    // Search for sibling nodes to fuse by visiting output edges from each input
    // edge in 'inEdges'.
    for (auto &inEdge : inEdges) {
      // Collect candidate output edges from each node 'inEdge.id' in 'inEdges'.
      SmallVector<MemRefDependenceGraph::Edge, 2> outEdges;
      mdg->forEachMemRefOutputEdge(
          inEdge.id, [&](MemRefDependenceGraph::Edge outEdge) {
            unsigned sibNodeId = outEdge.id;
            if (visitedSibNodeIds->count(sibNodeId) > 0)
              return;
            // Skip output edge if not a sibling using the same memref.
            if (outEdge.id == dstNode->id || outEdge.value != inEdge.value)
              return;
            auto *sibNode = mdg->getNode(sibNodeId);
            if (!isa<AffineForOp>(sibNode->op))
              return;
            // Check if 'sibNode/dstNode' can be input-reuse fused on 'memref'.
            if (canFuseWithSibNode(sibNode, outEdge.value)) {
              // Add candidate 'outEdge' to sibling node.
              outEdges.push_back(outEdge);
            }
          });

      // Add first candidate if any were returned.
      if (!outEdges.empty()) {
        visitedSibNodeIds->insert(outEdges[0].id);
        idAndMemrefToFuse->first = outEdges[0].id;
        idAndMemrefToFuse->second = outEdges[0].value;
        return true;
      }
    }
    return false;
  }

  /// Update data dependence graph state to reflect sibling fusion of 'sibNode'
  /// into 'dstNode'.
  void updateStateAfterSiblingFusion(Node *sibNode, Node *dstNode) {
    // Update 'sibNode' and 'dstNode' input/output edges to reflect fusion.
    mdg->updateEdges(sibNode->id, dstNode->id);

    // Collect dst loop stats after memref privatization transformation.
    auto dstForInst = cast<AffineForOp>(dstNode->op);
    LoopNestStateCollector dstLoopCollector;
    dstLoopCollector.collect(dstForInst.getOperation());
    // Clear and add back loads and stores
    mdg->clearNodeLoadAndStores(dstNode->id);
    mdg->addToNode(dstNode->id, dstLoopCollector.loadOpInsts,
                   dstLoopCollector.storeOpInsts);
    // Remove old sibling loop nest if it no longer has outgoing dependence
    // edges, and it does not write to a memref which escapes the
    // function.
    if (mdg->getOutEdgeCount(sibNode->id) == 0) {
      Operation *op = sibNode->op;
      mdg->removeNode(sibNode->id);
      op->erase();
    }
  }

  // Clean up any allocs with no users.
  void eraseUnusedMemRefAllocations() {
    for (auto &pair : mdg->memrefEdgeCount) {
      if (pair.second > 0)
        continue;
      auto memref = pair.first;
      // Skip if there exist other uses (return operation or function calls).
      if (!memref.use_empty())
        continue;
      // Use list expected to match the dep graph info.
      auto *op = memref.getDefiningOp();
      if (isa_and_nonnull<memref::AllocOp>(op))
        op->erase();
    }
  }
};

} // namespace

void LoopFusion::runOnOperation() {
  MemRefDependenceGraph g;
  if (!g.init(getOperation()))
    return;

  Optional<unsigned> fastMemorySpaceOpt;
  if (fastMemorySpace.hasValue())
    fastMemorySpaceOpt = fastMemorySpace;
  unsigned localBufSizeThresholdBytes = localBufSizeThreshold * 1024;
  GreedyFusion fusion(&g, localBufSizeThresholdBytes, fastMemorySpaceOpt,
                      maximalFusion, computeToleranceThreshold);

  if (affineFusionMode == FusionMode::ProducerConsumer)
    fusion.runProducerConsumerFusionOnly();
  else if (affineFusionMode == FusionMode::Sibling)
    fusion.runSiblingFusionOnly();
  else
    fusion.runGreedyFusion();
}
