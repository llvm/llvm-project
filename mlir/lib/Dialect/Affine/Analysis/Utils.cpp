//===- Utils.cpp ---- Misc utilities for analysis -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements miscellaneous analysis routines for non-loop IR
// structures.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IntegerSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

#define DEBUG_TYPE "analysis-utils"

using namespace mlir;
using namespace presburger;

using llvm::SmallDenseMap;

using Node = MemRefDependenceGraph::Node;

// LoopNestStateCollector walks loop nests and collects load and store
// operations, and whether or not a region holding op other than ForOp and IfOp
// was encountered in the loop nest.
void LoopNestStateCollector::collect(Operation *opToWalk) {
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

// Returns the load op count for 'memref'.
unsigned Node::getLoadOpCount(Value memref) const {
  unsigned loadOpCount = 0;
  for (Operation *loadOp : loads) {
    if (memref == cast<AffineReadOpInterface>(loadOp).getMemRef())
      ++loadOpCount;
  }
  return loadOpCount;
}

// Returns the store op count for 'memref'.
unsigned Node::getStoreOpCount(Value memref) const {
  unsigned storeOpCount = 0;
  for (Operation *storeOp : stores) {
    if (memref == cast<AffineWriteOpInterface>(storeOp).getMemRef())
      ++storeOpCount;
  }
  return storeOpCount;
}

// Returns all store ops in 'storeOps' which access 'memref'.
void Node::getStoreOpsForMemref(Value memref,
                                SmallVectorImpl<Operation *> *storeOps) const {
  for (Operation *storeOp : stores) {
    if (memref == cast<AffineWriteOpInterface>(storeOp).getMemRef())
      storeOps->push_back(storeOp);
  }
}

// Returns all load ops in 'loadOps' which access 'memref'.
void Node::getLoadOpsForMemref(Value memref,
                               SmallVectorImpl<Operation *> *loadOps) const {
  for (Operation *loadOp : loads) {
    if (memref == cast<AffineReadOpInterface>(loadOp).getMemRef())
      loadOps->push_back(loadOp);
  }
}

// Returns all memrefs in 'loadAndStoreMemrefSet' for which this node
// has at least one load and store operation.
void Node::getLoadAndStoreMemrefSet(
    DenseSet<Value> *loadAndStoreMemrefSet) const {
  llvm::SmallDenseSet<Value, 2> loadMemrefs;
  for (Operation *loadOp : loads) {
    loadMemrefs.insert(cast<AffineReadOpInterface>(loadOp).getMemRef());
  }
  for (Operation *storeOp : stores) {
    auto memref = cast<AffineWriteOpInterface>(storeOp).getMemRef();
    if (loadMemrefs.count(memref) > 0)
      loadAndStoreMemrefSet->insert(memref);
  }
}

// Returns the graph node for 'id'.
Node *MemRefDependenceGraph::getNode(unsigned id) {
  auto it = nodes.find(id);
  assert(it != nodes.end());
  return &it->second;
}

// Returns the graph node for 'forOp'.
Node *MemRefDependenceGraph::getForOpNode(AffineForOp forOp) {
  for (auto &idAndNode : nodes)
    if (idAndNode.second.op == forOp)
      return &idAndNode.second;
  return nullptr;
}

// Adds a node with 'op' to the graph and returns its unique identifier.
unsigned MemRefDependenceGraph::addNode(Operation *op) {
  Node node(nextNodeId++, op);
  nodes.insert({node.id, node});
  return node.id;
}

// Remove node 'id' (and its associated edges) from graph.
void MemRefDependenceGraph::removeNode(unsigned id) {
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
// argument to) the block. Returns false otherwise.
bool MemRefDependenceGraph::writesToLiveInOrEscapingMemrefs(unsigned id) {
  Node *node = getNode(id);
  for (auto *storeOpInst : node->stores) {
    auto memref = cast<AffineWriteOpInterface>(storeOpInst).getMemRef();
    auto *op = memref.getDefiningOp();
    // Return true if 'memref' is a block argument.
    if (!op)
      return true;
    // Return true if any use of 'memref' does not deference it in an affine
    // way.
    for (auto *user : memref.getUsers())
      if (!isa<AffineMapAccessInterface>(*user))
        return true;
  }
  return false;
}

// Returns true iff there is an edge from node 'srcId' to node 'dstId' which
// is for 'value' if non-null, or for any value otherwise. Returns false
// otherwise.
bool MemRefDependenceGraph::hasEdge(unsigned srcId, unsigned dstId,
                                    Value value) {
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
void MemRefDependenceGraph::addEdge(unsigned srcId, unsigned dstId,
                                    Value value) {
  if (!hasEdge(srcId, dstId, value)) {
    outEdges[srcId].push_back({dstId, value});
    inEdges[dstId].push_back({srcId, value});
    if (value.getType().isa<MemRefType>())
      memrefEdgeCount[value]++;
  }
}

// Removes an edge from node 'srcId' to node 'dstId' for 'value'.
void MemRefDependenceGraph::removeEdge(unsigned srcId, unsigned dstId,
                                       Value value) {
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
  for (auto *it = outEdges[srcId].begin(); it != outEdges[srcId].end(); ++it) {
    if ((*it).id == dstId && (*it).value == value) {
      outEdges[srcId].erase(it);
      break;
    }
  }
}

// Returns true if there is a path in the dependence graph from node 'srcId'
// to node 'dstId'. Returns false otherwise. `srcId`, `dstId`, and the
// operations that the edges connected are expected to be from the same block.
bool MemRefDependenceGraph::hasDependencePath(unsigned srcId, unsigned dstId) {
  // Worklist state is: <node-id, next-output-edge-index-to-visit>
  SmallVector<std::pair<unsigned, unsigned>, 4> worklist;
  worklist.push_back({srcId, 0});
  Operation *dstOp = getNode(dstId)->op;
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
    // Add node at 'edge.id' to the worklist. We don't need to consider
    // nodes that are "after" dstId in the containing block; one can't have a
    // path to `dstId` from any of those nodes.
    bool afterDst = dstOp->isBeforeInBlock(getNode(edge.id)->op);
    if (!afterDst && edge.id != idAndIndex.first)
      worklist.push_back({edge.id, 0});
  }
  return false;
}

// Returns the input edge count for node 'id' and 'memref' from src nodes
// which access 'memref' with a store operation.
unsigned MemRefDependenceGraph::getIncomingMemRefAccesses(unsigned id,
                                                          Value memref) {
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
unsigned MemRefDependenceGraph::getOutEdgeCount(unsigned id, Value memref) {
  unsigned outEdgeCount = 0;
  if (outEdges.count(id) > 0)
    for (auto &outEdge : outEdges[id])
      if (!memref || outEdge.value == memref)
        ++outEdgeCount;
  return outEdgeCount;
}

/// Return all nodes which define SSA values used in node 'id'.
void MemRefDependenceGraph::gatherDefiningNodes(
    unsigned id, DenseSet<unsigned> &definingNodes) {
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
Operation *
MemRefDependenceGraph::getFusedLoopNestInsertionPoint(unsigned srcId,
                                                      unsigned dstId) {
  if (outEdges.count(srcId) == 0)
    return getNode(dstId)->op;

  // Skip if there is any defining node of 'dstId' that depends on 'srcId'.
  DenseSet<unsigned> definingNodes;
  gatherDefiningNodes(dstId, definingNodes);
  if (llvm::any_of(definingNodes,
                   [&](unsigned id) { return hasDependencePath(srcId, id); })) {
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
  std::optional<unsigned> firstSrcDepPos;
  std::optional<unsigned> lastDstDepPos;
  unsigned pos = 0;
  for (Block::iterator it = std::next(Block::iterator(srcNodeInst));
       it != Block::iterator(dstNodeInst); ++it) {
    Operation *op = &(*it);
    if (srcDepInsts.count(op) > 0 && firstSrcDepPos == std::nullopt)
      firstSrcDepPos = pos;
    if (dstDepInsts.count(op) > 0)
      lastDstDepPos = pos;
    depInsts.push_back(op);
    ++pos;
  }

  if (firstSrcDepPos.has_value()) {
    if (lastDstDepPos.has_value()) {
      if (*firstSrcDepPos <= *lastDstDepPos) {
        // No valid insertion point exists which preserves dependences.
        return nullptr;
      }
    }
    // Return the insertion point at 'firstSrcDepPos'.
    return depInsts[*firstSrcDepPos];
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
void MemRefDependenceGraph::updateEdges(unsigned srcId, unsigned dstId,
                                        const DenseSet<Value> &privateMemRefs,
                                        bool removeSrcId) {
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
void MemRefDependenceGraph::updateEdges(unsigned sibId, unsigned dstId) {
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
void MemRefDependenceGraph::addToNode(
    unsigned id, const SmallVectorImpl<Operation *> &loads,
    const SmallVectorImpl<Operation *> &stores) {
  Node *node = getNode(id);
  llvm::append_range(node->loads, loads);
  llvm::append_range(node->stores, stores);
}

void MemRefDependenceGraph::clearNodeLoadAndStores(unsigned id) {
  Node *node = getNode(id);
  node->loads.clear();
  node->stores.clear();
}

// Calls 'callback' for each input edge incident to node 'id' which carries a
// memref dependence.
void MemRefDependenceGraph::forEachMemRefInputEdge(
    unsigned id, const std::function<void(Edge)> &callback) {
  if (inEdges.count(id) > 0)
    forEachMemRefEdge(inEdges[id], callback);
}

// Calls 'callback' for each output edge from node 'id' which carries a
// memref dependence.
void MemRefDependenceGraph::forEachMemRefOutputEdge(
    unsigned id, const std::function<void(Edge)> &callback) {
  if (outEdges.count(id) > 0)
    forEachMemRefEdge(outEdges[id], callback);
}

// Calls 'callback' for each edge in 'edges' which carries a memref
// dependence.
void MemRefDependenceGraph::forEachMemRefEdge(
    ArrayRef<Edge> edges, const std::function<void(Edge)> &callback) {
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

void MemRefDependenceGraph::print(raw_ostream &os) const {
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

void mlir::getAffineForIVs(Operation &op, SmallVectorImpl<AffineForOp> *loops) {
  auto *currOp = op.getParentOp();
  AffineForOp currAffineForOp;
  // Traverse up the hierarchy collecting all 'affine.for' operation while
  // skipping over 'affine.if' operations.
  while (currOp) {
    if (AffineForOp currAffineForOp = dyn_cast<AffineForOp>(currOp))
      loops->push_back(currAffineForOp);
    currOp = currOp->getParentOp();
  }
  std::reverse(loops->begin(), loops->end());
}

void mlir::getEnclosingAffineOps(Operation &op,
                                 SmallVectorImpl<Operation *> *ops) {
  ops->clear();
  Operation *currOp = op.getParentOp();

  // Traverse up the hierarchy collecting all `affine.for`, `affine.if`, and
  // affine.parallel operations.
  while (currOp) {
    if (isa<AffineIfOp, AffineForOp, AffineParallelOp>(currOp))
      ops->push_back(currOp);
    currOp = currOp->getParentOp();
  }
  std::reverse(ops->begin(), ops->end());
}

// Populates 'cst' with FlatAffineValueConstraints which represent original
// domain of the loop bounds that define 'ivs'.
LogicalResult
ComputationSliceState::getSourceAsConstraints(FlatAffineValueConstraints &cst) {
  assert(!ivs.empty() && "Cannot have a slice without its IVs");
  cst = FlatAffineValueConstraints(/*numDims=*/ivs.size(), /*numSymbols=*/0,
                                   /*numLocals=*/0, ivs);
  for (Value iv : ivs) {
    AffineForOp loop = getForInductionVarOwner(iv);
    assert(loop && "Expected affine for");
    if (failed(cst.addAffineForOpDomain(loop)))
      return failure();
  }
  return success();
}

// Populates 'cst' with FlatAffineValueConstraints which represent slice bounds.
LogicalResult
ComputationSliceState::getAsConstraints(FlatAffineValueConstraints *cst) {
  assert(!lbOperands.empty());
  // Adds src 'ivs' as dimension variables in 'cst'.
  unsigned numDims = ivs.size();
  // Adds operands (dst ivs and symbols) as symbols in 'cst'.
  unsigned numSymbols = lbOperands[0].size();

  SmallVector<Value, 4> values(ivs);
  // Append 'ivs' then 'operands' to 'values'.
  values.append(lbOperands[0].begin(), lbOperands[0].end());
  *cst = FlatAffineValueConstraints(numDims, numSymbols, 0, values);

  // Add loop bound constraints for values which are loop IVs of the destination
  // of fusion and equality constraints for symbols which are constants.
  for (unsigned i = numDims, end = values.size(); i < end; ++i) {
    Value value = values[i];
    assert(cst->containsVar(value) && "value expected to be present");
    if (isValidSymbol(value)) {
      // Check if the symbol is a constant.
      if (auto cOp = value.getDefiningOp<arith::ConstantIndexOp>())
        cst->addBound(BoundType::EQ, value, cOp.value());
    } else if (auto loop = getForInductionVarOwner(value)) {
      if (failed(cst->addAffineForOpDomain(loop)))
        return failure();
    }
  }

  // Add slices bounds on 'ivs' using maps 'lbs'/'ubs' with 'lbOperands[0]'
  LogicalResult ret = cst->addSliceBounds(ivs, lbs, ubs, lbOperands[0]);
  assert(succeeded(ret) &&
         "should not fail as we never have semi-affine slice maps");
  (void)ret;
  return success();
}

// Clears state bounds and operand state.
void ComputationSliceState::clearBounds() {
  lbs.clear();
  ubs.clear();
  lbOperands.clear();
  ubOperands.clear();
}

void ComputationSliceState::dump() const {
  llvm::errs() << "\tIVs:\n";
  for (Value iv : ivs)
    llvm::errs() << "\t\t" << iv << "\n";

  llvm::errs() << "\tLBs:\n";
  for (auto en : llvm::enumerate(lbs)) {
    llvm::errs() << "\t\t" << en.value() << "\n";
    llvm::errs() << "\t\tOperands:\n";
    for (Value lbOp : lbOperands[en.index()])
      llvm::errs() << "\t\t\t" << lbOp << "\n";
  }

  llvm::errs() << "\tUBs:\n";
  for (auto en : llvm::enumerate(ubs)) {
    llvm::errs() << "\t\t" << en.value() << "\n";
    llvm::errs() << "\t\tOperands:\n";
    for (Value ubOp : ubOperands[en.index()])
      llvm::errs() << "\t\t\t" << ubOp << "\n";
  }
}

/// Fast check to determine if the computation slice is maximal. Returns true if
/// each slice dimension maps to an existing dst dimension and both the src
/// and the dst loops for those dimensions have the same bounds. Returns false
/// if both the src and the dst loops don't have the same bounds. Returns
/// std::nullopt if none of the above can be proven.
std::optional<bool> ComputationSliceState::isSliceMaximalFastCheck() const {
  assert(lbs.size() == ubs.size() && !lbs.empty() && !ivs.empty() &&
         "Unexpected number of lbs, ubs and ivs in slice");

  for (unsigned i = 0, end = lbs.size(); i < end; ++i) {
    AffineMap lbMap = lbs[i];
    AffineMap ubMap = ubs[i];

    // Check if this slice is just an equality along this dimension.
    if (!lbMap || !ubMap || lbMap.getNumResults() != 1 ||
        ubMap.getNumResults() != 1 ||
        lbMap.getResult(0) + 1 != ubMap.getResult(0) ||
        // The condition above will be true for maps describing a single
        // iteration (e.g., lbMap.getResult(0) = 0, ubMap.getResult(0) = 1).
        // Make sure we skip those cases by checking that the lb result is not
        // just a constant.
        lbMap.getResult(0).isa<AffineConstantExpr>())
      return std::nullopt;

    // Limited support: we expect the lb result to be just a loop dimension for
    // now.
    AffineDimExpr result = lbMap.getResult(0).dyn_cast<AffineDimExpr>();
    if (!result)
      return std::nullopt;

    // Retrieve dst loop bounds.
    AffineForOp dstLoop =
        getForInductionVarOwner(lbOperands[i][result.getPosition()]);
    if (!dstLoop)
      return std::nullopt;
    AffineMap dstLbMap = dstLoop.getLowerBoundMap();
    AffineMap dstUbMap = dstLoop.getUpperBoundMap();

    // Retrieve src loop bounds.
    AffineForOp srcLoop = getForInductionVarOwner(ivs[i]);
    assert(srcLoop && "Expected affine for");
    AffineMap srcLbMap = srcLoop.getLowerBoundMap();
    AffineMap srcUbMap = srcLoop.getUpperBoundMap();

    // Limited support: we expect simple src and dst loops with a single
    // constant component per bound for now.
    if (srcLbMap.getNumResults() != 1 || srcUbMap.getNumResults() != 1 ||
        dstLbMap.getNumResults() != 1 || dstUbMap.getNumResults() != 1)
      return std::nullopt;

    AffineExpr srcLbResult = srcLbMap.getResult(0);
    AffineExpr dstLbResult = dstLbMap.getResult(0);
    AffineExpr srcUbResult = srcUbMap.getResult(0);
    AffineExpr dstUbResult = dstUbMap.getResult(0);
    if (!srcLbResult.isa<AffineConstantExpr>() ||
        !srcUbResult.isa<AffineConstantExpr>() ||
        !dstLbResult.isa<AffineConstantExpr>() ||
        !dstUbResult.isa<AffineConstantExpr>())
      return std::nullopt;

    // Check if src and dst loop bounds are the same. If not, we can guarantee
    // that the slice is not maximal.
    if (srcLbResult != dstLbResult || srcUbResult != dstUbResult ||
        srcLoop.getStep() != dstLoop.getStep())
      return false;
  }

  return true;
}

/// Returns true if it is deterministically verified that the original iteration
/// space of the slice is contained within the new iteration space that is
/// created after fusing 'this' slice into its destination.
std::optional<bool> ComputationSliceState::isSliceValid() {
  // Fast check to determine if the slice is valid. If the following conditions
  // are verified to be true, slice is declared valid by the fast check:
  // 1. Each slice loop is a single iteration loop bound in terms of a single
  //    destination loop IV.
  // 2. Loop bounds of the destination loop IV (from above) and those of the
  //    source loop IV are exactly the same.
  // If the fast check is inconclusive or false, we proceed with a more
  // expensive analysis.
  // TODO: Store the result of the fast check, as it might be used again in
  // `canRemoveSrcNodeAfterFusion`.
  std::optional<bool> isValidFastCheck = isSliceMaximalFastCheck();
  if (isValidFastCheck && *isValidFastCheck)
    return true;

  // Create constraints for the source loop nest using which slice is computed.
  FlatAffineValueConstraints srcConstraints;
  // TODO: Store the source's domain to avoid computation at each depth.
  if (failed(getSourceAsConstraints(srcConstraints))) {
    LLVM_DEBUG(llvm::dbgs() << "Unable to compute source's domain\n");
    return std::nullopt;
  }
  // As the set difference utility currently cannot handle symbols in its
  // operands, validity of the slice cannot be determined.
  if (srcConstraints.getNumSymbolVars() > 0) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot handle symbols in source domain\n");
    return std::nullopt;
  }
  // TODO: Handle local vars in the source domains while using the 'projectOut'
  // utility below. Currently, aligning is not done assuming that there will be
  // no local vars in the source domain.
  if (srcConstraints.getNumLocalVars() != 0) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot handle locals in source domain\n");
    return std::nullopt;
  }

  // Create constraints for the slice loop nest that would be created if the
  // fusion succeeds.
  FlatAffineValueConstraints sliceConstraints;
  if (failed(getAsConstraints(&sliceConstraints))) {
    LLVM_DEBUG(llvm::dbgs() << "Unable to compute slice's domain\n");
    return std::nullopt;
  }

  // Projecting out every dimension other than the 'ivs' to express slice's
  // domain completely in terms of source's IVs.
  sliceConstraints.projectOut(ivs.size(),
                              sliceConstraints.getNumVars() - ivs.size());

  LLVM_DEBUG(llvm::dbgs() << "Domain of the source of the slice:\n");
  LLVM_DEBUG(srcConstraints.dump());
  LLVM_DEBUG(llvm::dbgs() << "Domain of the slice if this fusion succeeds "
                             "(expressed in terms of its source's IVs):\n");
  LLVM_DEBUG(sliceConstraints.dump());

  // TODO: Store 'srcSet' to avoid recalculating for each depth.
  PresburgerSet srcSet(srcConstraints);
  PresburgerSet sliceSet(sliceConstraints);
  PresburgerSet diffSet = sliceSet.subtract(srcSet);

  if (!diffSet.isIntegerEmpty()) {
    LLVM_DEBUG(llvm::dbgs() << "Incorrect slice\n");
    return false;
  }
  return true;
}

/// Returns true if the computation slice encloses all the iterations of the
/// sliced loop nest. Returns false if it does not. Returns std::nullopt if it
/// cannot determine if the slice is maximal or not.
std::optional<bool> ComputationSliceState::isMaximal() const {
  // Fast check to determine if the computation slice is maximal. If the result
  // is inconclusive, we proceed with a more expensive analysis.
  std::optional<bool> isMaximalFastCheck = isSliceMaximalFastCheck();
  if (isMaximalFastCheck)
    return isMaximalFastCheck;

  // Create constraints for the src loop nest being sliced.
  FlatAffineValueConstraints srcConstraints(/*numDims=*/ivs.size(),
                                            /*numSymbols=*/0,
                                            /*numLocals=*/0, ivs);
  for (Value iv : ivs) {
    AffineForOp loop = getForInductionVarOwner(iv);
    assert(loop && "Expected affine for");
    if (failed(srcConstraints.addAffineForOpDomain(loop)))
      return std::nullopt;
  }

  // Create constraints for the slice using the dst loop nest information. We
  // retrieve existing dst loops from the lbOperands.
  SmallVector<Value> consumerIVs;
  for (Value lbOp : lbOperands[0])
    if (getForInductionVarOwner(lbOp))
      consumerIVs.push_back(lbOp);

  // Add empty IV Values for those new loops that are not equalities and,
  // therefore, are not yet materialized in the IR.
  for (int i = consumerIVs.size(), end = ivs.size(); i < end; ++i)
    consumerIVs.push_back(Value());

  FlatAffineValueConstraints sliceConstraints(/*numDims=*/consumerIVs.size(),
                                              /*numSymbols=*/0,
                                              /*numLocals=*/0, consumerIVs);

  if (failed(sliceConstraints.addDomainFromSliceMaps(lbs, ubs, lbOperands[0])))
    return std::nullopt;

  if (srcConstraints.getNumDimVars() != sliceConstraints.getNumDimVars())
    // Constraint dims are different. The integer set difference can't be
    // computed so we don't know if the slice is maximal.
    return std::nullopt;

  // Compute the difference between the src loop nest and the slice integer
  // sets.
  PresburgerSet srcSet(srcConstraints);
  PresburgerSet sliceSet(sliceConstraints);
  PresburgerSet diffSet = srcSet.subtract(sliceSet);
  return diffSet.isIntegerEmpty();
}

unsigned MemRefRegion::getRank() const {
  return memref.getType().cast<MemRefType>().getRank();
}

std::optional<int64_t> MemRefRegion::getConstantBoundingSizeAndShape(
    SmallVectorImpl<int64_t> *shape, std::vector<SmallVector<int64_t, 4>> *lbs,
    SmallVectorImpl<int64_t> *lbDivisors) const {
  auto memRefType = memref.getType().cast<MemRefType>();
  unsigned rank = memRefType.getRank();
  if (shape)
    shape->reserve(rank);

  assert(rank == cst.getNumDimVars() && "inconsistent memref region");

  // Use a copy of the region constraints that has upper/lower bounds for each
  // memref dimension with static size added to guard against potential
  // over-approximation from projection or union bounding box. We may not add
  // this on the region itself since they might just be redundant constraints
  // that will need non-trivials means to eliminate.
  FlatAffineValueConstraints cstWithShapeBounds(cst);
  for (unsigned r = 0; r < rank; r++) {
    cstWithShapeBounds.addBound(BoundType::LB, r, 0);
    int64_t dimSize = memRefType.getDimSize(r);
    if (ShapedType::isDynamic(dimSize))
      continue;
    cstWithShapeBounds.addBound(BoundType::UB, r, dimSize - 1);
  }

  // Find a constant upper bound on the extent of this memref region along each
  // dimension.
  int64_t numElements = 1;
  int64_t diffConstant;
  int64_t lbDivisor;
  for (unsigned d = 0; d < rank; d++) {
    SmallVector<int64_t, 4> lb;
    std::optional<int64_t> diff =
        cstWithShapeBounds.getConstantBoundOnDimSize64(d, &lb, &lbDivisor);
    if (diff.has_value()) {
      diffConstant = *diff;
      assert(diffConstant >= 0 && "Dim size bound can't be negative");
      assert(lbDivisor > 0);
    } else {
      // If no constant bound is found, then it can always be bound by the
      // memref's dim size if the latter has a constant size along this dim.
      auto dimSize = memRefType.getDimSize(d);
      if (dimSize == ShapedType::kDynamic)
        return std::nullopt;
      diffConstant = dimSize;
      // Lower bound becomes 0.
      lb.resize(cstWithShapeBounds.getNumSymbolVars() + 1, 0);
      lbDivisor = 1;
    }
    numElements *= diffConstant;
    if (lbs) {
      lbs->push_back(lb);
      assert(lbDivisors && "both lbs and lbDivisor or none");
      lbDivisors->push_back(lbDivisor);
    }
    if (shape) {
      shape->push_back(diffConstant);
    }
  }
  return numElements;
}

void MemRefRegion::getLowerAndUpperBound(unsigned pos, AffineMap &lbMap,
                                         AffineMap &ubMap) const {
  assert(pos < cst.getNumDimVars() && "invalid position");
  auto memRefType = memref.getType().cast<MemRefType>();
  unsigned rank = memRefType.getRank();

  assert(rank == cst.getNumDimVars() && "inconsistent memref region");

  auto boundPairs = cst.getLowerAndUpperBound(
      pos, /*offset=*/0, /*num=*/rank, cst.getNumDimAndSymbolVars(),
      /*localExprs=*/{}, memRefType.getContext());
  lbMap = boundPairs.first;
  ubMap = boundPairs.second;
  assert(lbMap && "lower bound for a region must exist");
  assert(ubMap && "upper bound for a region must exist");
  assert(lbMap.getNumInputs() == cst.getNumDimAndSymbolVars() - rank);
  assert(ubMap.getNumInputs() == cst.getNumDimAndSymbolVars() - rank);
}

LogicalResult MemRefRegion::unionBoundingBox(const MemRefRegion &other) {
  assert(memref == other.memref);
  return cst.unionBoundingBox(*other.getConstraints());
}

/// Computes the memory region accessed by this memref with the region
/// represented as constraints symbolic/parametric in 'loopDepth' loops
/// surrounding opInst and any additional Function symbols.
//  For example, the memref region for this load operation at loopDepth = 1 will
//  be as below:
//
//    affine.for %i = 0 to 32 {
//      affine.for %ii = %i to (d0) -> (d0 + 8) (%i) {
//        load %A[%ii]
//      }
//    }
//
// region:  {memref = %A, write = false, {%i <= m0 <= %i + 7} }
// The last field is a 2-d FlatAffineValueConstraints symbolic in %i.
//
// TODO: extend this to any other memref dereferencing ops
// (dma_start, dma_wait).
LogicalResult MemRefRegion::compute(Operation *op, unsigned loopDepth,
                                    const ComputationSliceState *sliceState,
                                    bool addMemRefDimBounds) {
  assert((isa<AffineReadOpInterface, AffineWriteOpInterface>(op)) &&
         "affine read/write op expected");

  MemRefAccess access(op);
  memref = access.memref;
  write = access.isStore();

  unsigned rank = access.getRank();

  LLVM_DEBUG(llvm::dbgs() << "MemRefRegion::compute: " << *op
                          << "\ndepth: " << loopDepth << "\n";);

  // 0-d memrefs.
  if (rank == 0) {
    SmallVector<Value, 4> ivs;
    getAffineIVs(*op, ivs);
    assert(loopDepth <= ivs.size() && "invalid 'loopDepth'");
    // The first 'loopDepth' IVs are symbols for this region.
    ivs.resize(loopDepth);
    // A 0-d memref has a 0-d region.
    cst = FlatAffineValueConstraints(rank, loopDepth, /*numLocals=*/0, ivs);
    return success();
  }

  // Build the constraints for this region.
  AffineValueMap accessValueMap;
  access.getAccessMap(&accessValueMap);
  AffineMap accessMap = accessValueMap.getAffineMap();

  unsigned numDims = accessMap.getNumDims();
  unsigned numSymbols = accessMap.getNumSymbols();
  unsigned numOperands = accessValueMap.getNumOperands();
  // Merge operands with slice operands.
  SmallVector<Value, 4> operands;
  operands.resize(numOperands);
  for (unsigned i = 0; i < numOperands; ++i)
    operands[i] = accessValueMap.getOperand(i);

  if (sliceState != nullptr) {
    operands.reserve(operands.size() + sliceState->lbOperands[0].size());
    // Append slice operands to 'operands' as symbols.
    for (auto extraOperand : sliceState->lbOperands[0]) {
      if (!llvm::is_contained(operands, extraOperand)) {
        operands.push_back(extraOperand);
        numSymbols++;
      }
    }
  }
  // We'll first associate the dims and symbols of the access map to the dims
  // and symbols resp. of cst. This will change below once cst is
  // fully constructed out.
  cst = FlatAffineValueConstraints(numDims, numSymbols, 0, operands);

  // Add equality constraints.
  // Add inequalities for loop lower/upper bounds.
  for (unsigned i = 0; i < numDims + numSymbols; ++i) {
    auto operand = operands[i];
    if (auto affineFor = getForInductionVarOwner(operand)) {
      // Note that cst can now have more dimensions than accessMap if the
      // bounds expressions involve outer loops or other symbols.
      // TODO: rewrite this to use getInstIndexSet; this way
      // conditionals will be handled when the latter supports it.
      if (failed(cst.addAffineForOpDomain(affineFor)))
        return failure();
    } else if (auto parallelOp = getAffineParallelInductionVarOwner(operand)) {
      if (failed(cst.addAffineParallelOpDomain(parallelOp)))
        return failure();
    } else if (isValidSymbol(operand)) {
      // Check if the symbol is a constant.
      Value symbol = operand;
      if (auto constVal = getConstantIntValue(symbol))
        cst.addBound(BoundType::EQ, symbol, constVal.value());
    } else {
      LLVM_DEBUG(llvm::dbgs() << "unknown affine dimensional value");
      return failure();
    }
  }

  // Add lower/upper bounds on loop IVs using bounds from 'sliceState'.
  if (sliceState != nullptr) {
    // Add dim and symbol slice operands.
    for (auto operand : sliceState->lbOperands[0]) {
      cst.addInductionVarOrTerminalSymbol(operand);
    }
    // Add upper/lower bounds from 'sliceState' to 'cst'.
    LogicalResult ret =
        cst.addSliceBounds(sliceState->ivs, sliceState->lbs, sliceState->ubs,
                           sliceState->lbOperands[0]);
    assert(succeeded(ret) &&
           "should not fail as we never have semi-affine slice maps");
    (void)ret;
  }

  // Add access function equalities to connect loop IVs to data dimensions.
  if (failed(cst.composeMap(&accessValueMap))) {
    op->emitError("getMemRefRegion: compose affine map failed");
    LLVM_DEBUG(accessValueMap.getAffineMap().dump());
    return failure();
  }

  // Set all variables appearing after the first 'rank' variables as
  // symbolic variables - so that the ones corresponding to the memref
  // dimensions are the dimensional variables for the memref region.
  cst.setDimSymbolSeparation(cst.getNumDimAndSymbolVars() - rank);

  // Eliminate any loop IVs other than the outermost 'loopDepth' IVs, on which
  // this memref region is symbolic.
  SmallVector<Value, 4> enclosingIVs;
  getAffineIVs(*op, enclosingIVs);
  assert(loopDepth <= enclosingIVs.size() && "invalid loop depth");
  enclosingIVs.resize(loopDepth);
  SmallVector<Value, 4> vars;
  cst.getValues(cst.getNumDimVars(), cst.getNumDimAndSymbolVars(), &vars);
  for (Value var : vars) {
    if ((isAffineInductionVar(var)) && !llvm::is_contained(enclosingIVs, var)) {
      cst.projectOut(var);
    }
  }

  // Project out any local variables (these would have been added for any
  // mod/divs).
  cst.projectOut(cst.getNumDimAndSymbolVars(), cst.getNumLocalVars());

  // Constant fold any symbolic variables.
  cst.constantFoldVarRange(/*pos=*/cst.getNumDimVars(),
                           /*num=*/cst.getNumSymbolVars());

  assert(cst.getNumDimVars() == rank && "unexpected MemRefRegion format");

  // Add upper/lower bounds for each memref dimension with static size
  // to guard against potential over-approximation from projection.
  // TODO: Support dynamic memref dimensions.
  if (addMemRefDimBounds) {
    auto memRefType = memref.getType().cast<MemRefType>();
    for (unsigned r = 0; r < rank; r++) {
      cst.addBound(BoundType::LB, /*pos=*/r, /*value=*/0);
      if (memRefType.isDynamicDim(r))
        continue;
      cst.addBound(BoundType::UB, /*pos=*/r, memRefType.getDimSize(r) - 1);
    }
  }
  cst.removeTrivialRedundancy();

  LLVM_DEBUG(llvm::dbgs() << "Memory region:\n");
  LLVM_DEBUG(cst.dump());
  return success();
}

std::optional<int64_t>
mlir::getMemRefIntOrFloatEltSizeInBytes(MemRefType memRefType) {
  auto elementType = memRefType.getElementType();

  unsigned sizeInBits;
  if (elementType.isIntOrFloat()) {
    sizeInBits = elementType.getIntOrFloatBitWidth();
  } else if (auto vectorType = elementType.dyn_cast<VectorType>()) {
    if (vectorType.getElementType().isIntOrFloat())
      sizeInBits =
          vectorType.getElementTypeBitWidth() * vectorType.getNumElements();
    else
      return std::nullopt;
  } else {
    return std::nullopt;
  }
  return llvm::divideCeil(sizeInBits, 8);
}

// Returns the size of the region.
std::optional<int64_t> MemRefRegion::getRegionSize() {
  auto memRefType = memref.getType().cast<MemRefType>();

  if (!memRefType.getLayout().isIdentity()) {
    LLVM_DEBUG(llvm::dbgs() << "Non-identity layout map not yet supported\n");
    return false;
  }

  // Indices to use for the DmaStart op.
  // Indices for the original memref being DMAed from/to.
  SmallVector<Value, 4> memIndices;
  // Indices for the faster buffer being DMAed into/from.
  SmallVector<Value, 4> bufIndices;

  // Compute the extents of the buffer.
  std::optional<int64_t> numElements = getConstantBoundingSizeAndShape();
  if (!numElements) {
    LLVM_DEBUG(llvm::dbgs() << "Dynamic shapes not yet supported\n");
    return std::nullopt;
  }
  auto eltSize = getMemRefIntOrFloatEltSizeInBytes(memRefType);
  if (!eltSize)
    return std::nullopt;
  return *eltSize * *numElements;
}

/// Returns the size of memref data in bytes if it's statically shaped,
/// std::nullopt otherwise.  If the element of the memref has vector type, takes
/// into account size of the vector as well.
//  TODO: improve/complete this when we have target data.
std::optional<uint64_t>
mlir::getIntOrFloatMemRefSizeInBytes(MemRefType memRefType) {
  if (!memRefType.hasStaticShape())
    return std::nullopt;
  auto elementType = memRefType.getElementType();
  if (!elementType.isIntOrFloat() && !elementType.isa<VectorType>())
    return std::nullopt;

  auto sizeInBytes = getMemRefIntOrFloatEltSizeInBytes(memRefType);
  if (!sizeInBytes)
    return std::nullopt;
  for (unsigned i = 0, e = memRefType.getRank(); i < e; i++) {
    sizeInBytes = *sizeInBytes * memRefType.getDimSize(i);
  }
  return sizeInBytes;
}

template <typename LoadOrStoreOp>
LogicalResult mlir::boundCheckLoadOrStoreOp(LoadOrStoreOp loadOrStoreOp,
                                            bool emitError) {
  static_assert(llvm::is_one_of<LoadOrStoreOp, AffineReadOpInterface,
                                AffineWriteOpInterface>::value,
                "argument should be either a AffineReadOpInterface or a "
                "AffineWriteOpInterface");

  Operation *op = loadOrStoreOp.getOperation();
  MemRefRegion region(op->getLoc());
  if (failed(region.compute(op, /*loopDepth=*/0, /*sliceState=*/nullptr,
                            /*addMemRefDimBounds=*/false)))
    return success();

  LLVM_DEBUG(llvm::dbgs() << "Memory region");
  LLVM_DEBUG(region.getConstraints()->dump());

  bool outOfBounds = false;
  unsigned rank = loadOrStoreOp.getMemRefType().getRank();

  // For each dimension, check for out of bounds.
  for (unsigned r = 0; r < rank; r++) {
    FlatAffineValueConstraints ucst(*region.getConstraints());

    // Intersect memory region with constraint capturing out of bounds (both out
    // of upper and out of lower), and check if the constraint system is
    // feasible. If it is, there is at least one point out of bounds.
    SmallVector<int64_t, 4> ineq(rank + 1, 0);
    int64_t dimSize = loadOrStoreOp.getMemRefType().getDimSize(r);
    // TODO: handle dynamic dim sizes.
    if (dimSize == -1)
      continue;

    // Check for overflow: d_i >= memref dim size.
    ucst.addBound(BoundType::LB, r, dimSize);
    outOfBounds = !ucst.isEmpty();
    if (outOfBounds && emitError) {
      loadOrStoreOp.emitOpError()
          << "memref out of upper bound access along dimension #" << (r + 1);
    }

    // Check for a negative index.
    FlatAffineValueConstraints lcst(*region.getConstraints());
    std::fill(ineq.begin(), ineq.end(), 0);
    // d_i <= -1;
    lcst.addBound(BoundType::UB, r, -1);
    outOfBounds = !lcst.isEmpty();
    if (outOfBounds && emitError) {
      loadOrStoreOp.emitOpError()
          << "memref out of lower bound access along dimension #" << (r + 1);
    }
  }
  return failure(outOfBounds);
}

// Explicitly instantiate the template so that the compiler knows we need them!
template LogicalResult
mlir::boundCheckLoadOrStoreOp(AffineReadOpInterface loadOp, bool emitError);
template LogicalResult
mlir::boundCheckLoadOrStoreOp(AffineWriteOpInterface storeOp, bool emitError);

// Returns in 'positions' the Block positions of 'op' in each ancestor
// Block from the Block containing operation, stopping at 'limitBlock'.
static void findInstPosition(Operation *op, Block *limitBlock,
                             SmallVectorImpl<unsigned> *positions) {
  Block *block = op->getBlock();
  while (block != limitBlock) {
    // FIXME: This algorithm is unnecessarily O(n) and should be improved to not
    // rely on linear scans.
    int instPosInBlock = std::distance(block->begin(), op->getIterator());
    positions->push_back(instPosInBlock);
    op = block->getParentOp();
    block = op->getBlock();
  }
  std::reverse(positions->begin(), positions->end());
}

// Returns the Operation in a possibly nested set of Blocks, where the
// position of the operation is represented by 'positions', which has a
// Block position for each level of nesting.
static Operation *getInstAtPosition(ArrayRef<unsigned> positions,
                                    unsigned level, Block *block) {
  unsigned i = 0;
  for (auto &op : *block) {
    if (i != positions[level]) {
      ++i;
      continue;
    }
    if (level == positions.size() - 1)
      return &op;
    if (auto childAffineForOp = dyn_cast<AffineForOp>(op))
      return getInstAtPosition(positions, level + 1,
                               childAffineForOp.getBody());

    for (auto &region : op.getRegions()) {
      for (auto &b : region)
        if (auto *ret = getInstAtPosition(positions, level + 1, &b))
          return ret;
    }
    return nullptr;
  }
  return nullptr;
}

// Adds loop IV bounds to 'cst' for loop IVs not found in 'ivs'.
static LogicalResult addMissingLoopIVBounds(SmallPtrSet<Value, 8> &ivs,
                                            FlatAffineValueConstraints *cst) {
  for (unsigned i = 0, e = cst->getNumDimVars(); i < e; ++i) {
    auto value = cst->getValue(i);
    if (ivs.count(value) == 0) {
      assert(isAffineForInductionVar(value));
      auto loop = getForInductionVarOwner(value);
      if (failed(cst->addAffineForOpDomain(loop)))
        return failure();
    }
  }
  return success();
}

/// Returns the innermost common loop depth for the set of operations in 'ops'.
// TODO: Move this to LoopUtils.
unsigned mlir::getInnermostCommonLoopDepth(
    ArrayRef<Operation *> ops, SmallVectorImpl<AffineForOp> *surroundingLoops) {
  unsigned numOps = ops.size();
  assert(numOps > 0 && "Expected at least one operation");

  std::vector<SmallVector<AffineForOp, 4>> loops(numOps);
  unsigned loopDepthLimit = std::numeric_limits<unsigned>::max();
  for (unsigned i = 0; i < numOps; ++i) {
    getAffineForIVs(*ops[i], &loops[i]);
    loopDepthLimit =
        std::min(loopDepthLimit, static_cast<unsigned>(loops[i].size()));
  }

  unsigned loopDepth = 0;
  for (unsigned d = 0; d < loopDepthLimit; ++d) {
    unsigned i;
    for (i = 1; i < numOps; ++i) {
      if (loops[i - 1][d] != loops[i][d])
        return loopDepth;
    }
    if (surroundingLoops)
      surroundingLoops->push_back(loops[i - 1][d]);
    ++loopDepth;
  }
  return loopDepth;
}

/// Computes in 'sliceUnion' the union of all slice bounds computed at
/// 'loopDepth' between all dependent pairs of ops in 'opsA' and 'opsB', and
/// then verifies if it is valid. Returns 'SliceComputationResult::Success' if
/// union was computed correctly, an appropriate failure otherwise.
SliceComputationResult
mlir::computeSliceUnion(ArrayRef<Operation *> opsA, ArrayRef<Operation *> opsB,
                        unsigned loopDepth, unsigned numCommonLoops,
                        bool isBackwardSlice,
                        ComputationSliceState *sliceUnion) {
  // Compute the union of slice bounds between all pairs in 'opsA' and
  // 'opsB' in 'sliceUnionCst'.
  FlatAffineValueConstraints sliceUnionCst;
  assert(sliceUnionCst.getNumDimAndSymbolVars() == 0);
  std::vector<std::pair<Operation *, Operation *>> dependentOpPairs;
  for (auto *i : opsA) {
    MemRefAccess srcAccess(i);
    for (auto *j : opsB) {
      MemRefAccess dstAccess(j);
      if (srcAccess.memref != dstAccess.memref)
        continue;
      // Check if 'loopDepth' exceeds nesting depth of src/dst ops.
      if ((!isBackwardSlice && loopDepth > getNestingDepth(i)) ||
          (isBackwardSlice && loopDepth > getNestingDepth(j))) {
        LLVM_DEBUG(llvm::dbgs() << "Invalid loop depth\n");
        return SliceComputationResult::GenericFailure;
      }

      bool readReadAccesses = isa<AffineReadOpInterface>(srcAccess.opInst) &&
                              isa<AffineReadOpInterface>(dstAccess.opInst);
      FlatAffineValueConstraints dependenceConstraints;
      // Check dependence between 'srcAccess' and 'dstAccess'.
      DependenceResult result = checkMemrefAccessDependence(
          srcAccess, dstAccess, /*loopDepth=*/numCommonLoops + 1,
          &dependenceConstraints, /*dependenceComponents=*/nullptr,
          /*allowRAR=*/readReadAccesses);
      if (result.value == DependenceResult::Failure) {
        LLVM_DEBUG(llvm::dbgs() << "Dependence check failed\n");
        return SliceComputationResult::GenericFailure;
      }
      if (result.value == DependenceResult::NoDependence)
        continue;
      dependentOpPairs.emplace_back(i, j);

      // Compute slice bounds for 'srcAccess' and 'dstAccess'.
      ComputationSliceState tmpSliceState;
      mlir::getComputationSliceState(i, j, &dependenceConstraints, loopDepth,
                                     isBackwardSlice, &tmpSliceState);

      if (sliceUnionCst.getNumDimAndSymbolVars() == 0) {
        // Initialize 'sliceUnionCst' with the bounds computed in previous step.
        if (failed(tmpSliceState.getAsConstraints(&sliceUnionCst))) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Unable to compute slice bound constraints\n");
          return SliceComputationResult::GenericFailure;
        }
        assert(sliceUnionCst.getNumDimAndSymbolVars() > 0);
        continue;
      }

      // Compute constraints for 'tmpSliceState' in 'tmpSliceCst'.
      FlatAffineValueConstraints tmpSliceCst;
      if (failed(tmpSliceState.getAsConstraints(&tmpSliceCst))) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Unable to compute slice bound constraints\n");
        return SliceComputationResult::GenericFailure;
      }

      // Align coordinate spaces of 'sliceUnionCst' and 'tmpSliceCst' if needed.
      if (!sliceUnionCst.areVarsAlignedWithOther(tmpSliceCst)) {

        // Pre-constraint var alignment: record loop IVs used in each constraint
        // system.
        SmallPtrSet<Value, 8> sliceUnionIVs;
        for (unsigned k = 0, l = sliceUnionCst.getNumDimVars(); k < l; ++k)
          sliceUnionIVs.insert(sliceUnionCst.getValue(k));
        SmallPtrSet<Value, 8> tmpSliceIVs;
        for (unsigned k = 0, l = tmpSliceCst.getNumDimVars(); k < l; ++k)
          tmpSliceIVs.insert(tmpSliceCst.getValue(k));

        sliceUnionCst.mergeAndAlignVarsWithOther(/*offset=*/0, &tmpSliceCst);

        // Post-constraint var alignment: add loop IV bounds missing after
        // var alignment to constraint systems. This can occur if one constraint
        // system uses an loop IV that is not used by the other. The call
        // to unionBoundingBox below expects constraints for each Loop IV, even
        // if they are the unsliced full loop bounds added here.
        if (failed(addMissingLoopIVBounds(sliceUnionIVs, &sliceUnionCst)))
          return SliceComputationResult::GenericFailure;
        if (failed(addMissingLoopIVBounds(tmpSliceIVs, &tmpSliceCst)))
          return SliceComputationResult::GenericFailure;
      }
      // Compute union bounding box of 'sliceUnionCst' and 'tmpSliceCst'.
      if (sliceUnionCst.getNumLocalVars() > 0 ||
          tmpSliceCst.getNumLocalVars() > 0 ||
          failed(sliceUnionCst.unionBoundingBox(tmpSliceCst))) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Unable to compute union bounding box of slice bounds\n");
        return SliceComputationResult::GenericFailure;
      }
    }
  }

  // Empty union.
  if (sliceUnionCst.getNumDimAndSymbolVars() == 0)
    return SliceComputationResult::GenericFailure;

  // Gather loops surrounding ops from loop nest where slice will be inserted.
  SmallVector<Operation *, 4> ops;
  for (auto &dep : dependentOpPairs) {
    ops.push_back(isBackwardSlice ? dep.second : dep.first);
  }
  SmallVector<AffineForOp, 4> surroundingLoops;
  unsigned innermostCommonLoopDepth =
      getInnermostCommonLoopDepth(ops, &surroundingLoops);
  if (loopDepth > innermostCommonLoopDepth) {
    LLVM_DEBUG(llvm::dbgs() << "Exceeds max loop depth\n");
    return SliceComputationResult::GenericFailure;
  }

  // Store 'numSliceLoopIVs' before converting dst loop IVs to dims.
  unsigned numSliceLoopIVs = sliceUnionCst.getNumDimVars();

  // Convert any dst loop IVs which are symbol variables to dim variables.
  sliceUnionCst.convertLoopIVSymbolsToDims();
  sliceUnion->clearBounds();
  sliceUnion->lbs.resize(numSliceLoopIVs, AffineMap());
  sliceUnion->ubs.resize(numSliceLoopIVs, AffineMap());

  // Get slice bounds from slice union constraints 'sliceUnionCst'.
  sliceUnionCst.getSliceBounds(/*offset=*/0, numSliceLoopIVs,
                               opsA[0]->getContext(), &sliceUnion->lbs,
                               &sliceUnion->ubs);

  // Add slice bound operands of union.
  SmallVector<Value, 4> sliceBoundOperands;
  sliceUnionCst.getValues(numSliceLoopIVs,
                          sliceUnionCst.getNumDimAndSymbolVars(),
                          &sliceBoundOperands);

  // Copy src loop IVs from 'sliceUnionCst' to 'sliceUnion'.
  sliceUnion->ivs.clear();
  sliceUnionCst.getValues(0, numSliceLoopIVs, &sliceUnion->ivs);

  // Set loop nest insertion point to block start at 'loopDepth'.
  sliceUnion->insertPoint =
      isBackwardSlice
          ? surroundingLoops[loopDepth - 1].getBody()->begin()
          : std::prev(surroundingLoops[loopDepth - 1].getBody()->end());

  // Give each bound its own copy of 'sliceBoundOperands' for subsequent
  // canonicalization.
  sliceUnion->lbOperands.resize(numSliceLoopIVs, sliceBoundOperands);
  sliceUnion->ubOperands.resize(numSliceLoopIVs, sliceBoundOperands);

  // Check if the slice computed is valid. Return success only if it is verified
  // that the slice is valid, otherwise return appropriate failure status.
  std::optional<bool> isSliceValid = sliceUnion->isSliceValid();
  if (!isSliceValid) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot determine if the slice is valid\n");
    return SliceComputationResult::GenericFailure;
  }
  if (!*isSliceValid)
    return SliceComputationResult::IncorrectSliceFailure;

  return SliceComputationResult::Success;
}

// TODO: extend this to handle multiple result maps.
static std::optional<uint64_t> getConstDifference(AffineMap lbMap,
                                                  AffineMap ubMap) {
  assert(lbMap.getNumResults() == 1 && "expected single result bound map");
  assert(ubMap.getNumResults() == 1 && "expected single result bound map");
  assert(lbMap.getNumDims() == ubMap.getNumDims());
  assert(lbMap.getNumSymbols() == ubMap.getNumSymbols());
  AffineExpr lbExpr(lbMap.getResult(0));
  AffineExpr ubExpr(ubMap.getResult(0));
  auto loopSpanExpr = simplifyAffineExpr(ubExpr - lbExpr, lbMap.getNumDims(),
                                         lbMap.getNumSymbols());
  auto cExpr = loopSpanExpr.dyn_cast<AffineConstantExpr>();
  if (!cExpr)
    return std::nullopt;
  return cExpr.getValue();
}

// Builds a map 'tripCountMap' from AffineForOp to constant trip count for loop
// nest surrounding represented by slice loop bounds in 'slice'. Returns true
// on success, false otherwise (if a non-constant trip count was encountered).
// TODO: Make this work with non-unit step loops.
bool mlir::buildSliceTripCountMap(
    const ComputationSliceState &slice,
    llvm::SmallDenseMap<Operation *, uint64_t, 8> *tripCountMap) {
  unsigned numSrcLoopIVs = slice.ivs.size();
  // Populate map from AffineForOp -> trip count
  for (unsigned i = 0; i < numSrcLoopIVs; ++i) {
    AffineForOp forOp = getForInductionVarOwner(slice.ivs[i]);
    auto *op = forOp.getOperation();
    AffineMap lbMap = slice.lbs[i];
    AffineMap ubMap = slice.ubs[i];
    // If lower or upper bound maps are null or provide no results, it implies
    // that source loop was not at all sliced, and the entire loop will be a
    // part of the slice.
    if (!lbMap || lbMap.getNumResults() == 0 || !ubMap ||
        ubMap.getNumResults() == 0) {
      // The iteration of src loop IV 'i' was not sliced. Use full loop bounds.
      if (forOp.hasConstantLowerBound() && forOp.hasConstantUpperBound()) {
        (*tripCountMap)[op] =
            forOp.getConstantUpperBound() - forOp.getConstantLowerBound();
        continue;
      }
      std::optional<uint64_t> maybeConstTripCount = getConstantTripCount(forOp);
      if (maybeConstTripCount.has_value()) {
        (*tripCountMap)[op] = *maybeConstTripCount;
        continue;
      }
      return false;
    }
    std::optional<uint64_t> tripCount = getConstDifference(lbMap, ubMap);
    // Slice bounds are created with a constant ub - lb difference.
    if (!tripCount.has_value())
      return false;
    (*tripCountMap)[op] = *tripCount;
  }
  return true;
}

// Return the number of iterations in the given slice.
uint64_t mlir::getSliceIterationCount(
    const llvm::SmallDenseMap<Operation *, uint64_t, 8> &sliceTripCountMap) {
  uint64_t iterCount = 1;
  for (const auto &count : sliceTripCountMap) {
    iterCount *= count.second;
  }
  return iterCount;
}

const char *const kSliceFusionBarrierAttrName = "slice_fusion_barrier";
// Computes slice bounds by projecting out any loop IVs from
// 'dependenceConstraints' at depth greater than 'loopDepth', and computes slice
// bounds in 'sliceState' which represent the one loop nest's IVs in terms of
// the other loop nest's IVs, symbols and constants (using 'isBackwardsSlice').
void mlir::getComputationSliceState(
    Operation *depSourceOp, Operation *depSinkOp,
    FlatAffineValueConstraints *dependenceConstraints, unsigned loopDepth,
    bool isBackwardSlice, ComputationSliceState *sliceState) {
  // Get loop nest surrounding src operation.
  SmallVector<AffineForOp, 4> srcLoopIVs;
  getAffineForIVs(*depSourceOp, &srcLoopIVs);
  unsigned numSrcLoopIVs = srcLoopIVs.size();

  // Get loop nest surrounding dst operation.
  SmallVector<AffineForOp, 4> dstLoopIVs;
  getAffineForIVs(*depSinkOp, &dstLoopIVs);
  unsigned numDstLoopIVs = dstLoopIVs.size();

  assert((!isBackwardSlice && loopDepth <= numSrcLoopIVs) ||
         (isBackwardSlice && loopDepth <= numDstLoopIVs));

  // Project out dimensions other than those up to 'loopDepth'.
  unsigned pos = isBackwardSlice ? numSrcLoopIVs + loopDepth : loopDepth;
  unsigned num =
      isBackwardSlice ? numDstLoopIVs - loopDepth : numSrcLoopIVs - loopDepth;
  dependenceConstraints->projectOut(pos, num);

  // Add slice loop IV values to 'sliceState'.
  unsigned offset = isBackwardSlice ? 0 : loopDepth;
  unsigned numSliceLoopIVs = isBackwardSlice ? numSrcLoopIVs : numDstLoopIVs;
  dependenceConstraints->getValues(offset, offset + numSliceLoopIVs,
                                   &sliceState->ivs);

  // Set up lower/upper bound affine maps for the slice.
  sliceState->lbs.resize(numSliceLoopIVs, AffineMap());
  sliceState->ubs.resize(numSliceLoopIVs, AffineMap());

  // Get bounds for slice IVs in terms of other IVs, symbols, and constants.
  dependenceConstraints->getSliceBounds(offset, numSliceLoopIVs,
                                        depSourceOp->getContext(),
                                        &sliceState->lbs, &sliceState->ubs);

  // Set up bound operands for the slice's lower and upper bounds.
  SmallVector<Value, 4> sliceBoundOperands;
  unsigned numDimsAndSymbols = dependenceConstraints->getNumDimAndSymbolVars();
  for (unsigned i = 0; i < numDimsAndSymbols; ++i) {
    if (i < offset || i >= offset + numSliceLoopIVs) {
      sliceBoundOperands.push_back(dependenceConstraints->getValue(i));
    }
  }

  // Give each bound its own copy of 'sliceBoundOperands' for subsequent
  // canonicalization.
  sliceState->lbOperands.resize(numSliceLoopIVs, sliceBoundOperands);
  sliceState->ubOperands.resize(numSliceLoopIVs, sliceBoundOperands);

  // Set destination loop nest insertion point to block start at 'dstLoopDepth'.
  sliceState->insertPoint =
      isBackwardSlice ? dstLoopIVs[loopDepth - 1].getBody()->begin()
                      : std::prev(srcLoopIVs[loopDepth - 1].getBody()->end());

  llvm::SmallDenseSet<Value, 8> sequentialLoops;
  if (isa<AffineReadOpInterface>(depSourceOp) &&
      isa<AffineReadOpInterface>(depSinkOp)) {
    // For read-read access pairs, clear any slice bounds on sequential loops.
    // Get sequential loops in loop nest rooted at 'srcLoopIVs[0]'.
    getSequentialLoops(isBackwardSlice ? srcLoopIVs[0] : dstLoopIVs[0],
                       &sequentialLoops);
  }
  auto getSliceLoop = [&](unsigned i) {
    return isBackwardSlice ? srcLoopIVs[i] : dstLoopIVs[i];
  };
  auto isInnermostInsertion = [&]() {
    return (isBackwardSlice ? loopDepth >= srcLoopIVs.size()
                            : loopDepth >= dstLoopIVs.size());
  };
  llvm::SmallDenseMap<Operation *, uint64_t, 8> sliceTripCountMap;
  auto srcIsUnitSlice = [&]() {
    return (buildSliceTripCountMap(*sliceState, &sliceTripCountMap) &&
            (getSliceIterationCount(sliceTripCountMap) == 1));
  };
  // Clear all sliced loop bounds beginning at the first sequential loop, or
  // first loop with a slice fusion barrier attribute..

  for (unsigned i = 0; i < numSliceLoopIVs; ++i) {
    Value iv = getSliceLoop(i).getInductionVar();
    if (sequentialLoops.count(iv) == 0 &&
        getSliceLoop(i)->getAttr(kSliceFusionBarrierAttrName) == nullptr)
      continue;
    // Skip reset of bounds of reduction loop inserted in the destination loop
    // that meets the following conditions:
    //    1. Slice is  single trip count.
    //    2. Loop bounds of the source and destination match.
    //    3. Is being inserted at the innermost insertion point.
    std::optional<bool> isMaximal = sliceState->isMaximal();
    if (isLoopParallelAndContainsReduction(getSliceLoop(i)) &&
        isInnermostInsertion() && srcIsUnitSlice() && isMaximal && *isMaximal)
      continue;
    for (unsigned j = i; j < numSliceLoopIVs; ++j) {
      sliceState->lbs[j] = AffineMap();
      sliceState->ubs[j] = AffineMap();
    }
    break;
  }
}

/// Creates a computation slice of the loop nest surrounding 'srcOpInst',
/// updates the slice loop bounds with any non-null bound maps specified in
/// 'sliceState', and inserts this slice into the loop nest surrounding
/// 'dstOpInst' at loop depth 'dstLoopDepth'.
// TODO: extend the slicing utility to compute slices that
// aren't necessarily a one-to-one relation b/w the source and destination. The
// relation between the source and destination could be many-to-many in general.
// TODO: the slice computation is incorrect in the cases
// where the dependence from the source to the destination does not cover the
// entire destination index set. Subtract out the dependent destination
// iterations from destination index set and check for emptiness --- this is one
// solution.
AffineForOp
mlir::insertBackwardComputationSlice(Operation *srcOpInst, Operation *dstOpInst,
                                     unsigned dstLoopDepth,
                                     ComputationSliceState *sliceState) {
  // Get loop nest surrounding src operation.
  SmallVector<AffineForOp, 4> srcLoopIVs;
  getAffineForIVs(*srcOpInst, &srcLoopIVs);
  unsigned numSrcLoopIVs = srcLoopIVs.size();

  // Get loop nest surrounding dst operation.
  SmallVector<AffineForOp, 4> dstLoopIVs;
  getAffineForIVs(*dstOpInst, &dstLoopIVs);
  unsigned dstLoopIVsSize = dstLoopIVs.size();
  if (dstLoopDepth > dstLoopIVsSize) {
    dstOpInst->emitError("invalid destination loop depth");
    return AffineForOp();
  }

  // Find the op block positions of 'srcOpInst' within 'srcLoopIVs'.
  SmallVector<unsigned, 4> positions;
  // TODO: This code is incorrect since srcLoopIVs can be 0-d.
  findInstPosition(srcOpInst, srcLoopIVs[0]->getBlock(), &positions);

  // Clone src loop nest and insert it a the beginning of the operation block
  // of the loop at 'dstLoopDepth' in 'dstLoopIVs'.
  auto dstAffineForOp = dstLoopIVs[dstLoopDepth - 1];
  OpBuilder b(dstAffineForOp.getBody(), dstAffineForOp.getBody()->begin());
  auto sliceLoopNest =
      cast<AffineForOp>(b.clone(*srcLoopIVs[0].getOperation()));

  Operation *sliceInst =
      getInstAtPosition(positions, /*level=*/0, sliceLoopNest.getBody());
  // Get loop nest surrounding 'sliceInst'.
  SmallVector<AffineForOp, 4> sliceSurroundingLoops;
  getAffineForIVs(*sliceInst, &sliceSurroundingLoops);

  // Sanity check.
  unsigned sliceSurroundingLoopsSize = sliceSurroundingLoops.size();
  (void)sliceSurroundingLoopsSize;
  assert(dstLoopDepth + numSrcLoopIVs >= sliceSurroundingLoopsSize);
  unsigned sliceLoopLimit = dstLoopDepth + numSrcLoopIVs;
  (void)sliceLoopLimit;
  assert(sliceLoopLimit >= sliceSurroundingLoopsSize);

  // Update loop bounds for loops in 'sliceLoopNest'.
  for (unsigned i = 0; i < numSrcLoopIVs; ++i) {
    auto forOp = sliceSurroundingLoops[dstLoopDepth + i];
    if (AffineMap lbMap = sliceState->lbs[i])
      forOp.setLowerBound(sliceState->lbOperands[i], lbMap);
    if (AffineMap ubMap = sliceState->ubs[i])
      forOp.setUpperBound(sliceState->ubOperands[i], ubMap);
  }
  return sliceLoopNest;
}

// Constructs  MemRefAccess populating it with the memref, its indices and
// opinst from 'loadOrStoreOpInst'.
MemRefAccess::MemRefAccess(Operation *loadOrStoreOpInst) {
  if (auto loadOp = dyn_cast<AffineReadOpInterface>(loadOrStoreOpInst)) {
    memref = loadOp.getMemRef();
    opInst = loadOrStoreOpInst;
    llvm::append_range(indices, loadOp.getMapOperands());
  } else {
    assert(isa<AffineWriteOpInterface>(loadOrStoreOpInst) &&
           "Affine read/write op expected");
    auto storeOp = cast<AffineWriteOpInterface>(loadOrStoreOpInst);
    opInst = loadOrStoreOpInst;
    memref = storeOp.getMemRef();
    llvm::append_range(indices, storeOp.getMapOperands());
  }
}

unsigned MemRefAccess::getRank() const {
  return memref.getType().cast<MemRefType>().getRank();
}

bool MemRefAccess::isStore() const {
  return isa<AffineWriteOpInterface>(opInst);
}

/// Returns the nesting depth of this statement, i.e., the number of loops
/// surrounding this statement.
unsigned mlir::getNestingDepth(Operation *op) {
  Operation *currOp = op;
  unsigned depth = 0;
  while ((currOp = currOp->getParentOp())) {
    if (isa<AffineForOp>(currOp))
      depth++;
  }
  return depth;
}

/// Equal if both affine accesses are provably equivalent (at compile
/// time) when considering the memref, the affine maps and their respective
/// operands. The equality of access functions + operands is checked by
/// subtracting fully composed value maps, and then simplifying the difference
/// using the expression flattener.
/// TODO: this does not account for aliasing of memrefs.
bool MemRefAccess::operator==(const MemRefAccess &rhs) const {
  if (memref != rhs.memref)
    return false;

  AffineValueMap diff, thisMap, rhsMap;
  getAccessMap(&thisMap);
  rhs.getAccessMap(&rhsMap);
  AffineValueMap::difference(thisMap, rhsMap, &diff);
  return llvm::all_of(diff.getAffineMap().getResults(),
                      [](AffineExpr e) { return e == 0; });
}

void mlir::getAffineIVs(Operation &op, SmallVectorImpl<Value> &ivs) {
  auto *currOp = op.getParentOp();
  AffineForOp currAffineForOp;
  // Traverse up the hierarchy collecting all 'affine.for' and affine.parallel
  // operation while skipping over 'affine.if' operations.
  while (currOp) {
    if (AffineForOp currAffineForOp = dyn_cast<AffineForOp>(currOp))
      ivs.push_back(currAffineForOp.getInductionVar());
    else if (auto parOp = dyn_cast<AffineParallelOp>(currOp))
      llvm::append_range(ivs, parOp.getIVs());
    currOp = currOp->getParentOp();
  }
  std::reverse(ivs.begin(), ivs.end());
}

/// Returns the number of surrounding loops common to 'loopsA' and 'loopsB',
/// where each lists loops from outer-most to inner-most in loop nest.
unsigned mlir::getNumCommonSurroundingLoops(Operation &a, Operation &b) {
  SmallVector<Value, 4> loopsA, loopsB;
  getAffineIVs(a, loopsA);
  getAffineIVs(b, loopsB);

  unsigned minNumLoops = std::min(loopsA.size(), loopsB.size());
  unsigned numCommonLoops = 0;
  for (unsigned i = 0; i < minNumLoops; ++i) {
    if (loopsA[i] != loopsB[i])
      break;
    ++numCommonLoops;
  }
  return numCommonLoops;
}

static std::optional<int64_t> getMemoryFootprintBytes(Block &block,
                                                      Block::iterator start,
                                                      Block::iterator end,
                                                      int memorySpace) {
  SmallDenseMap<Value, std::unique_ptr<MemRefRegion>, 4> regions;

  // Walk this 'affine.for' operation to gather all memory regions.
  auto result = block.walk(start, end, [&](Operation *opInst) -> WalkResult {
    if (!isa<AffineReadOpInterface, AffineWriteOpInterface>(opInst)) {
      // Neither load nor a store op.
      return WalkResult::advance();
    }

    // Compute the memref region symbolic in any IVs enclosing this block.
    auto region = std::make_unique<MemRefRegion>(opInst->getLoc());
    if (failed(
            region->compute(opInst,
                            /*loopDepth=*/getNestingDepth(&*block.begin())))) {
      return opInst->emitError("error obtaining memory region\n");
    }

    auto it = regions.find(region->memref);
    if (it == regions.end()) {
      regions[region->memref] = std::move(region);
    } else if (failed(it->second->unionBoundingBox(*region))) {
      return opInst->emitWarning(
          "getMemoryFootprintBytes: unable to perform a union on a memory "
          "region");
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return std::nullopt;

  int64_t totalSizeInBytes = 0;
  for (const auto &region : regions) {
    std::optional<int64_t> size = region.second->getRegionSize();
    if (!size.has_value())
      return std::nullopt;
    totalSizeInBytes += *size;
  }
  return totalSizeInBytes;
}

std::optional<int64_t> mlir::getMemoryFootprintBytes(AffineForOp forOp,
                                                     int memorySpace) {
  auto *forInst = forOp.getOperation();
  return ::getMemoryFootprintBytes(
      *forInst->getBlock(), Block::iterator(forInst),
      std::next(Block::iterator(forInst)), memorySpace);
}

/// Returns whether a loop is parallel and contains a reduction loop.
bool mlir::isLoopParallelAndContainsReduction(AffineForOp forOp) {
  SmallVector<LoopReduction> reductions;
  if (!isLoopParallel(forOp, &reductions))
    return false;
  return !reductions.empty();
}

/// Returns in 'sequentialLoops' all sequential loops in loop nest rooted
/// at 'forOp'.
void mlir::getSequentialLoops(AffineForOp forOp,
                              llvm::SmallDenseSet<Value, 8> *sequentialLoops) {
  forOp->walk([&](Operation *op) {
    if (auto innerFor = dyn_cast<AffineForOp>(op))
      if (!isLoopParallel(innerFor))
        sequentialLoops->insert(innerFor.getInductionVar());
  });
}

IntegerSet mlir::simplifyIntegerSet(IntegerSet set) {
  FlatAffineValueConstraints fac(set);
  if (fac.isEmpty())
    return IntegerSet::getEmptySet(set.getNumDims(), set.getNumSymbols(),
                                   set.getContext());
  fac.removeTrivialRedundancy();

  auto simplifiedSet = fac.getAsIntegerSet(set.getContext());
  assert(simplifiedSet && "guaranteed to succeed while roundtripping");
  return simplifiedSet;
}

static void unpackOptionalValues(ArrayRef<std::optional<Value>> source,
                                 SmallVector<Value> &target) {
  target =
      llvm::to_vector<4>(llvm::map_range(source, [](std::optional<Value> val) {
        return val.has_value() ? *val : Value();
      }));
}

/// Bound an identifier `pos` in a given FlatAffineValueConstraints with
/// constraints drawn from an affine map. Before adding the constraint, the
/// dimensions/symbols of the affine map are aligned with `constraints`.
/// `operands` are the SSA Value operands used with the affine map.
/// Note: This function adds a new symbol column to the `constraints` for each
/// dimension/symbol that exists in the affine map but not in `constraints`.
static LogicalResult alignAndAddBound(FlatAffineValueConstraints &constraints,
                                      BoundType type, unsigned pos,
                                      AffineMap map, ValueRange operands) {
  SmallVector<Value> dims, syms, newSyms;
  unpackOptionalValues(constraints.getMaybeValues(VarKind::SetDim), dims);
  unpackOptionalValues(constraints.getMaybeValues(VarKind::Symbol), syms);

  AffineMap alignedMap =
      alignAffineMapWithValues(map, operands, dims, syms, &newSyms);
  for (unsigned i = syms.size(); i < newSyms.size(); ++i)
    constraints.appendSymbolVar(newSyms[i]);
  return constraints.addBound(type, pos, alignedMap);
}

/// Add `val` to each result of `map`.
static AffineMap addConstToResults(AffineMap map, int64_t val) {
  SmallVector<AffineExpr> newResults;
  for (AffineExpr r : map.getResults())
    newResults.push_back(r + val);
  return AffineMap::get(map.getNumDims(), map.getNumSymbols(), newResults,
                        map.getContext());
}

// Attempt to simplify the given min/max operation by proving that its value is
// bounded by the same lower and upper bound.
//
// Bounds are computed by FlatAffineValueConstraints. Invariants required for
// finding/proving bounds should be supplied via `constraints`.
//
// 1. Add dimensions for `op` and `opBound` (lower or upper bound of `op`).
// 2. Compute an upper bound of `op` (in case of `isMin`) or a lower bound (in
//    case of `!isMin`) and bind it to `opBound`. SSA values that are used in
//    `op` but are not part of `constraints`, are added as extra symbols.
// 3. For each result of `op`: Add result as a dimension `r_i`. Prove that:
//    * If `isMin`: r_i >= opBound
//    * If `isMax`: r_i <= opBound
//    If this is the case, ub(op) == lb(op).
// 4. Replace `op` with `opBound`.
//
// In summary, the following constraints are added throughout this function.
// Note: `invar` are dimensions added by the caller to express the invariants.
// (Showing only the case where `isMin`.)
//
//  invar |    op | opBound | r_i | extra syms... | const |           eq/ineq
//  ------+-------+---------+-----+---------------+-------+-------------------
//   (various eq./ineq. constraining `invar`, added by the caller)
//    ... |     0 |       0 |   0 |             0 |   ... |               ...
//  ------+-------+---------+-----+---------------+-------+-------------------
//  (various ineq. constraining `op` in terms of `op` operands (`invar` and
//    extra `op` operands "extra syms" that are not in `invar`)).
//    ... |    -1 |       0 |   0 |           ... |   ... |              >= 0
//  ------+-------+---------+-----+---------------+-------+-------------------
//   (set `opBound` to `op` upper bound in terms of `invar` and "extra syms")
//    ... |     0 |      -1 |   0 |           ... |   ... |               = 0
//  ------+-------+---------+-----+---------------+-------+-------------------
//   (for each `op` map result r_i: set r_i to corresponding map result,
//    prove that r_i >= minOpUb via contradiction)
//    ... |     0 |       0 |  -1 |           ... |   ... |               = 0
//      0 |     0 |       1 |  -1 |             0 |    -1 |              >= 0
//
FailureOr<AffineValueMap>
mlir::simplifyConstrainedMinMaxOp(Operation *op,
                                  FlatAffineValueConstraints constraints) {
  bool isMin = isa<AffineMinOp>(op);
  assert((isMin || isa<AffineMaxOp>(op)) && "expect AffineMin/MaxOp");
  MLIRContext *ctx = op->getContext();
  Builder builder(ctx);
  AffineMap map =
      isMin ? cast<AffineMinOp>(op).getMap() : cast<AffineMaxOp>(op).getMap();
  ValueRange operands = op->getOperands();
  unsigned numResults = map.getNumResults();

  // Add a few extra dimensions.
  unsigned dimOp = constraints.appendDimVar();      // `op`
  unsigned dimOpBound = constraints.appendDimVar(); // `op` lower/upper bound
  unsigned resultDimStart = constraints.appendDimVar(/*num=*/numResults);

  // Add an inequality for each result expr_i of map:
  // isMin: op <= expr_i, !isMin: op >= expr_i
  auto boundType = isMin ? BoundType::UB : BoundType::LB;
  // Upper bounds are exclusive, so add 1. (`affine.min` ops are inclusive.)
  AffineMap mapLbUb = isMin ? addConstToResults(map, 1) : map;
  if (failed(
          alignAndAddBound(constraints, boundType, dimOp, mapLbUb, operands)))
    return failure();

  // Try to compute a lower/upper bound for op, expressed in terms of the other
  // `dims` and extra symbols.
  SmallVector<AffineMap> opLb(1), opUb(1);
  constraints.getSliceBounds(dimOp, 1, ctx, &opLb, &opUb);
  AffineMap sliceBound = isMin ? opUb[0] : opLb[0];
  // TODO: `getSliceBounds` may return multiple bounds at the moment. This is
  // a TODO of `getSliceBounds` and not handled here.
  if (!sliceBound || sliceBound.getNumResults() != 1)
    return failure(); // No or multiple bounds found.
  // Recover the inclusive UB in the case of an `affine.min`.
  AffineMap boundMap = isMin ? addConstToResults(sliceBound, -1) : sliceBound;

  // Add an equality: Set dimOpBound to computed bound.
  // Add back dimension for op. (Was removed by `getSliceBounds`.)
  AffineMap alignedBoundMap = boundMap.shiftDims(/*shift=*/1, /*offset=*/dimOp);
  if (failed(constraints.addBound(BoundType::EQ, dimOpBound, alignedBoundMap)))
    return failure();

  // If the constraint system is empty, there is an inconsistency. (E.g., this
  // can happen if loop lb > ub.)
  if (constraints.isEmpty())
    return failure();

  // In the case of `isMin` (`!isMin` is inversed):
  // Prove that each result of `map` has a lower bound that is equal to (or
  // greater than) the upper bound of `op` (`dimOpBound`). In that case, `op`
  // can be replaced with the bound. I.e., prove that for each result
  // expr_i (represented by dimension r_i):
  //
  // r_i >= opBound
  //
  // To prove this inequality, add its negation to the constraint set and prove
  // that the constraint set is empty.
  for (unsigned i = resultDimStart; i < resultDimStart + numResults; ++i) {
    FlatAffineValueConstraints newConstr(constraints);

    // Add an equality: r_i = expr_i
    // Note: These equalities could have been added earlier and used to express
    // minOp <= expr_i. However, then we run the risk that `getSliceBounds`
    // computes minOpUb in terms of r_i dims, which is not desired.
    if (failed(alignAndAddBound(newConstr, BoundType::EQ, i,
                                map.getSubMap({i - resultDimStart}), operands)))
      return failure();

    // If `isMin`:  Add inequality: r_i < opBound
    //              equiv.: opBound - r_i - 1 >= 0
    // If `!isMin`: Add inequality: r_i > opBound
    //              equiv.: -opBound + r_i - 1 >= 0
    SmallVector<int64_t> ineq(newConstr.getNumCols(), 0);
    ineq[dimOpBound] = isMin ? 1 : -1;
    ineq[i] = isMin ? -1 : 1;
    ineq[newConstr.getNumCols() - 1] = -1;
    newConstr.addInequality(ineq);
    if (!newConstr.isEmpty())
      return failure();
  }

  // Lower and upper bound of `op` are equal. Replace `minOp` with its bound.
  AffineMap newMap = alignedBoundMap;
  SmallVector<Value> newOperands;
  unpackOptionalValues(constraints.getMaybeValues(), newOperands);
  // If dims/symbols have known constant values, use those in order to simplify
  // the affine map further.
  for (int64_t i = 0, e = constraints.getNumDimAndSymbolVars(); i < e; ++i) {
    // Skip unused operands and operands that are already constants.
    if (!newOperands[i] || getConstantIntValue(newOperands[i]))
      continue;
    if (auto bound = constraints.getConstantBound64(BoundType::EQ, i)) {
      AffineExpr expr =
          i < newMap.getNumDims()
              ? builder.getAffineDimExpr(i)
              : builder.getAffineSymbolExpr(i - newMap.getNumDims());
      newMap = newMap.replace(expr, builder.getAffineConstantExpr(*bound),
                              newMap.getNumDims(), newMap.getNumSymbols());
    }
  }
  mlir::canonicalizeMapAndOperands(&newMap, &newOperands);
  return AffineValueMap(newMap, newOperands);
}
