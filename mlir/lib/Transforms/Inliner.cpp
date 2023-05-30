//===- Inliner.cpp - Pass to inline function calls ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a basic inlining algorithm that operates bottom up over
// the Strongly Connect Components(SCCs) of the CallGraph. This enables a more
// incremental propagation of inlining decisions from the leafs to the roots of
// the callgraph.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Passes.h"

#include "mlir/Analysis/CallGraph.h"
#include "mlir/IR/Threading.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_INLINER
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "inlining"

using namespace mlir;

/// This function implements the default inliner optimization pipeline.
static void defaultInlinerOptPipeline(OpPassManager &pm) {
  pm.addPass(createCanonicalizerPass());
}

//===----------------------------------------------------------------------===//
// Symbol Use Tracking
//===----------------------------------------------------------------------===//

/// Walk all of the used symbol callgraph nodes referenced with the given op.
static void walkReferencedSymbolNodes(
    Operation *op, CallGraph &cg, SymbolTableCollection &symbolTable,
    DenseMap<Attribute, CallGraphNode *> &resolvedRefs,
    function_ref<void(CallGraphNode *, Operation *)> callback) {
  auto symbolUses = SymbolTable::getSymbolUses(op);
  assert(symbolUses && "expected uses to be valid");

  Operation *symbolTableOp = op->getParentOp();
  for (const SymbolTable::SymbolUse &use : *symbolUses) {
    auto refIt = resolvedRefs.insert({use.getSymbolRef(), nullptr});
    CallGraphNode *&node = refIt.first->second;

    // If this is the first instance of this reference, try to resolve a
    // callgraph node for it.
    if (refIt.second) {
      auto *symbolOp = symbolTable.lookupNearestSymbolFrom(symbolTableOp,
                                                           use.getSymbolRef());
      auto callableOp = dyn_cast_or_null<CallableOpInterface>(symbolOp);
      if (!callableOp)
        continue;
      node = cg.lookupNode(callableOp.getCallableRegion());
    }
    if (node)
      callback(node, use.getUser());
  }
}

//===----------------------------------------------------------------------===//
// CGUseList

namespace {
/// This struct tracks the uses of callgraph nodes that can be dropped when
/// use_empty. It directly tracks and manages a use-list for all of the
/// call-graph nodes. This is necessary because many callgraph nodes are
/// referenced by SymbolRefAttr, which has no mechanism akin to the SSA `Use`
/// class.
struct CGUseList {
  /// This struct tracks the uses of callgraph nodes within a specific
  /// operation.
  struct CGUser {
    /// Any nodes referenced in the top-level attribute list of this user. We
    /// use a set here because the number of references does not matter.
    DenseSet<CallGraphNode *> topLevelUses;

    /// Uses of nodes referenced by nested operations.
    DenseMap<CallGraphNode *, int> innerUses;
  };

  CGUseList(Operation *op, CallGraph &cg, SymbolTableCollection &symbolTable);

  /// Drop uses of nodes referred to by the given call operation that resides
  /// within 'userNode'.
  void dropCallUses(CallGraphNode *userNode, Operation *callOp, CallGraph &cg);

  /// Remove the given node from the use list.
  void eraseNode(CallGraphNode *node);

  /// Returns true if the given callgraph node has no uses and can be pruned.
  bool isDead(CallGraphNode *node) const;

  /// Returns true if the given callgraph node has a single use and can be
  /// discarded.
  bool hasOneUseAndDiscardable(CallGraphNode *node) const;

  /// Recompute the uses held by the given callgraph node.
  void recomputeUses(CallGraphNode *node, CallGraph &cg);

  /// Merge the uses of 'lhs' with the uses of the 'rhs' after inlining a copy
  /// of 'lhs' into 'rhs'.
  void mergeUsesAfterInlining(CallGraphNode *lhs, CallGraphNode *rhs);

private:
  /// Decrement the uses of discardable nodes referenced by the given user.
  void decrementDiscardableUses(CGUser &uses);

  /// A mapping between a discardable callgraph node (that is a symbol) and the
  /// number of uses for this node.
  DenseMap<CallGraphNode *, int> discardableSymNodeUses;

  /// A mapping between a callgraph node and the symbol callgraph nodes that it
  /// uses.
  DenseMap<CallGraphNode *, CGUser> nodeUses;

  /// A symbol table to use when resolving call lookups.
  SymbolTableCollection &symbolTable;
};
} // namespace

CGUseList::CGUseList(Operation *op, CallGraph &cg,
                     SymbolTableCollection &symbolTable)
    : symbolTable(symbolTable) {
  /// A set of callgraph nodes that are always known to be live during inlining.
  DenseMap<Attribute, CallGraphNode *> alwaysLiveNodes;

  // Walk each of the symbol tables looking for discardable callgraph nodes.
  auto walkFn = [&](Operation *symbolTableOp, bool allUsesVisible) {
    for (Operation &op : symbolTableOp->getRegion(0).getOps()) {
      // If this is a callgraph operation, check to see if it is discardable.
      if (auto callable = dyn_cast<CallableOpInterface>(&op)) {
        if (auto *node = cg.lookupNode(callable.getCallableRegion())) {
          SymbolOpInterface symbol = dyn_cast<SymbolOpInterface>(&op);
          if (symbol && (allUsesVisible || symbol.isPrivate()) &&
              symbol.canDiscardOnUseEmpty()) {
            discardableSymNodeUses.try_emplace(node, 0);
          }
          continue;
        }
      }
      // Otherwise, check for any referenced nodes. These will be always-live.
      walkReferencedSymbolNodes(&op, cg, symbolTable, alwaysLiveNodes,
                                [](CallGraphNode *, Operation *) {});
    }
  };
  SymbolTable::walkSymbolTables(op, /*allSymUsesVisible=*/!op->getBlock(),
                                walkFn);

  // Drop the use information for any discardable nodes that are always live.
  for (auto &it : alwaysLiveNodes)
    discardableSymNodeUses.erase(it.second);

  // Compute the uses for each of the callable nodes in the graph.
  for (CallGraphNode *node : cg)
    recomputeUses(node, cg);
}

void CGUseList::dropCallUses(CallGraphNode *userNode, Operation *callOp,
                             CallGraph &cg) {
  auto &userRefs = nodeUses[userNode].innerUses;
  auto walkFn = [&](CallGraphNode *node, Operation *user) {
    auto parentIt = userRefs.find(node);
    if (parentIt == userRefs.end())
      return;
    --parentIt->second;
    --discardableSymNodeUses[node];
  };
  DenseMap<Attribute, CallGraphNode *> resolvedRefs;
  walkReferencedSymbolNodes(callOp, cg, symbolTable, resolvedRefs, walkFn);
}

void CGUseList::eraseNode(CallGraphNode *node) {
  // Drop all child nodes.
  for (auto &edge : *node)
    if (edge.isChild())
      eraseNode(edge.getTarget());

  // Drop the uses held by this node and erase it.
  auto useIt = nodeUses.find(node);
  assert(useIt != nodeUses.end() && "expected node to be valid");
  decrementDiscardableUses(useIt->getSecond());
  nodeUses.erase(useIt);
  discardableSymNodeUses.erase(node);
}

bool CGUseList::isDead(CallGraphNode *node) const {
  // If the parent operation isn't a symbol, simply check normal SSA deadness.
  Operation *nodeOp = node->getCallableRegion()->getParentOp();
  if (!isa<SymbolOpInterface>(nodeOp))
    return isMemoryEffectFree(nodeOp) && nodeOp->use_empty();

  // Otherwise, check the number of symbol uses.
  auto symbolIt = discardableSymNodeUses.find(node);
  return symbolIt != discardableSymNodeUses.end() && symbolIt->second == 0;
}

bool CGUseList::hasOneUseAndDiscardable(CallGraphNode *node) const {
  // If this isn't a symbol node, check for side-effects and SSA use count.
  Operation *nodeOp = node->getCallableRegion()->getParentOp();
  if (!isa<SymbolOpInterface>(nodeOp))
    return isMemoryEffectFree(nodeOp) && nodeOp->hasOneUse();

  // Otherwise, check the number of symbol uses.
  auto symbolIt = discardableSymNodeUses.find(node);
  return symbolIt != discardableSymNodeUses.end() && symbolIt->second == 1;
}

void CGUseList::recomputeUses(CallGraphNode *node, CallGraph &cg) {
  Operation *parentOp = node->getCallableRegion()->getParentOp();
  CGUser &uses = nodeUses[node];
  decrementDiscardableUses(uses);

  // Collect the new discardable uses within this node.
  uses = CGUser();
  DenseMap<Attribute, CallGraphNode *> resolvedRefs;
  auto walkFn = [&](CallGraphNode *refNode, Operation *user) {
    auto discardSymIt = discardableSymNodeUses.find(refNode);
    if (discardSymIt == discardableSymNodeUses.end())
      return;

    if (user != parentOp)
      ++uses.innerUses[refNode];
    else if (!uses.topLevelUses.insert(refNode).second)
      return;
    ++discardSymIt->second;
  };
  walkReferencedSymbolNodes(parentOp, cg, symbolTable, resolvedRefs, walkFn);
}

void CGUseList::mergeUsesAfterInlining(CallGraphNode *lhs, CallGraphNode *rhs) {
  auto &lhsUses = nodeUses[lhs], &rhsUses = nodeUses[rhs];
  for (auto &useIt : lhsUses.innerUses) {
    rhsUses.innerUses[useIt.first] += useIt.second;
    discardableSymNodeUses[useIt.first] += useIt.second;
  }
}

void CGUseList::decrementDiscardableUses(CGUser &uses) {
  for (CallGraphNode *node : uses.topLevelUses)
    --discardableSymNodeUses[node];
  for (auto &it : uses.innerUses)
    discardableSymNodeUses[it.first] -= it.second;
}

//===----------------------------------------------------------------------===//
// CallGraph traversal
//===----------------------------------------------------------------------===//

namespace {
/// This class represents a specific callgraph SCC.
class CallGraphSCC {
public:
  CallGraphSCC(llvm::scc_iterator<const CallGraph *> &parentIterator)
      : parentIterator(parentIterator) {}
  /// Return a range over the nodes within this SCC.
  std::vector<CallGraphNode *>::iterator begin() { return nodes.begin(); }
  std::vector<CallGraphNode *>::iterator end() { return nodes.end(); }

  /// Reset the nodes of this SCC with those provided.
  void reset(const std::vector<CallGraphNode *> &newNodes) { nodes = newNodes; }

  /// Remove the given node from this SCC.
  void remove(CallGraphNode *node) {
    auto it = llvm::find(nodes, node);
    if (it != nodes.end()) {
      nodes.erase(it);
      parentIterator.ReplaceNode(node, nullptr);
    }
  }

private:
  std::vector<CallGraphNode *> nodes;
  llvm::scc_iterator<const CallGraph *> &parentIterator;
};
} // namespace

/// Run a given transformation over the SCCs of the callgraph in a bottom up
/// traversal.
static LogicalResult runTransformOnCGSCCs(
    const CallGraph &cg,
    function_ref<LogicalResult(CallGraphSCC &)> sccTransformer) {
  llvm::scc_iterator<const CallGraph *> cgi = llvm::scc_begin(&cg);
  CallGraphSCC currentSCC(cgi);
  while (!cgi.isAtEnd()) {
    // Copy the current SCC and increment so that the transformer can modify the
    // SCC without invalidating our iterator.
    currentSCC.reset(*cgi);
    ++cgi;
    if (failed(sccTransformer(currentSCC)))
      return failure();
  }
  return success();
}

namespace {
/// This struct represents a resolved call to a given callgraph node. Given that
/// the call does not actually contain a direct reference to the
/// Region(CallGraphNode) that it is dispatching to, we need to resolve them
/// explicitly.
struct ResolvedCall {
  ResolvedCall(CallOpInterface call, CallGraphNode *sourceNode,
               CallGraphNode *targetNode)
      : call(call), sourceNode(sourceNode), targetNode(targetNode) {}
  CallOpInterface call;
  CallGraphNode *sourceNode, *targetNode;
};
} // namespace

/// Collect all of the callable operations within the given range of blocks. If
/// `traverseNestedCGNodes` is true, this will also collect call operations
/// inside of nested callgraph nodes.
static void collectCallOps(iterator_range<Region::iterator> blocks,
                           CallGraphNode *sourceNode, CallGraph &cg,
                           SymbolTableCollection &symbolTable,
                           SmallVectorImpl<ResolvedCall> &calls,
                           bool traverseNestedCGNodes) {
  SmallVector<std::pair<Block *, CallGraphNode *>, 8> worklist;
  auto addToWorklist = [&](CallGraphNode *node,
                           iterator_range<Region::iterator> blocks) {
    for (Block &block : blocks)
      worklist.emplace_back(&block, node);
  };

  addToWorklist(sourceNode, blocks);
  while (!worklist.empty()) {
    Block *block;
    std::tie(block, sourceNode) = worklist.pop_back_val();

    for (Operation &op : *block) {
      if (auto call = dyn_cast<CallOpInterface>(op)) {
        // TODO: Support inlining nested call references.
        CallInterfaceCallable callable = call.getCallableForCallee();
        if (SymbolRefAttr symRef = dyn_cast<SymbolRefAttr>(callable)) {
          if (!isa<FlatSymbolRefAttr>(symRef))
            continue;
        }

        CallGraphNode *targetNode = cg.resolveCallable(call, symbolTable);
        if (!targetNode->isExternal())
          calls.emplace_back(call, sourceNode, targetNode);
        continue;
      }

      // If this is not a call, traverse the nested regions. If
      // `traverseNestedCGNodes` is false, then don't traverse nested call graph
      // regions.
      for (auto &nestedRegion : op.getRegions()) {
        CallGraphNode *nestedNode = cg.lookupNode(&nestedRegion);
        if (traverseNestedCGNodes || !nestedNode)
          addToWorklist(nestedNode ? nestedNode : sourceNode, nestedRegion);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Inliner
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
static std::string getNodeName(CallOpInterface op) {
  if (auto sym = llvm::dyn_cast_if_present<SymbolRefAttr>(op.getCallableForCallee()))
    return debugString(op);
  return "_unnamed_callee_";
}
#endif

/// Return true if the specified `inlineHistoryID`  indicates an inline history
/// that already includes `node`.
static bool inlineHistoryIncludes(
    CallGraphNode *node, std::optional<size_t> inlineHistoryID,
    MutableArrayRef<std::pair<CallGraphNode *, std::optional<size_t>>>
        inlineHistory) {
  while (inlineHistoryID.has_value()) {
    assert(*inlineHistoryID < inlineHistory.size() &&
           "Invalid inline history ID");
    if (inlineHistory[*inlineHistoryID].first == node)
      return true;
    inlineHistoryID = inlineHistory[*inlineHistoryID].second;
  }
  return false;
}

namespace {
/// This class provides a specialization of the main inlining interface.
struct Inliner : public InlinerInterface {
  Inliner(MLIRContext *context, CallGraph &cg,
          SymbolTableCollection &symbolTable)
      : InlinerInterface(context), cg(cg), symbolTable(symbolTable) {}

  /// Process a set of blocks that have been inlined. This callback is invoked
  /// *before* inlined terminator operations have been processed.
  void
  processInlinedBlocks(iterator_range<Region::iterator> inlinedBlocks) final {
    // Find the closest callgraph node from the first block.
    CallGraphNode *node;
    Region *region = inlinedBlocks.begin()->getParent();
    while (!(node = cg.lookupNode(region))) {
      region = region->getParentRegion();
      assert(region && "expected valid parent node");
    }

    collectCallOps(inlinedBlocks, node, cg, symbolTable, calls,
                   /*traverseNestedCGNodes=*/true);
  }

  /// Mark the given callgraph node for deletion.
  void markForDeletion(CallGraphNode *node) { deadNodes.insert(node); }

  /// This method properly disposes of callables that became dead during
  /// inlining. This should not be called while iterating over the SCCs.
  void eraseDeadCallables() {
    for (CallGraphNode *node : deadNodes)
      node->getCallableRegion()->getParentOp()->erase();
  }

  /// The set of callables known to be dead.
  SmallPtrSet<CallGraphNode *, 8> deadNodes;

  /// The current set of call instructions to consider for inlining.
  SmallVector<ResolvedCall, 8> calls;

  /// The callgraph being operated on.
  CallGraph &cg;

  /// A symbol table to use when resolving call lookups.
  SymbolTableCollection &symbolTable;
};
} // namespace

/// Returns true if the given call should be inlined.
static bool shouldInline(ResolvedCall &resolvedCall) {
  // Don't allow inlining terminator calls. We currently don't support this
  // case.
  if (resolvedCall.call->hasTrait<OpTrait::IsTerminator>())
    return false;

  // Don't allow inlining if the target is an ancestor of the call. This
  // prevents inlining recursively.
  if (resolvedCall.targetNode->getCallableRegion()->isAncestor(
          resolvedCall.call->getParentRegion()))
    return false;

  // Otherwise, inline.
  return true;
}

/// Attempt to inline calls within the given scc. This function returns
/// success if any calls were inlined, failure otherwise.
static LogicalResult inlineCallsInSCC(Inliner &inliner, CGUseList &useList,
                                      CallGraphSCC &currentSCC) {
  CallGraph &cg = inliner.cg;
  auto &calls = inliner.calls;

  // A set of dead nodes to remove after inlining.
  llvm::SmallSetVector<CallGraphNode *, 1> deadNodes;

  // Collect all of the direct calls within the nodes of the current SCC. We
  // don't traverse nested callgraph nodes, because they are handled separately
  // likely within a different SCC.
  for (CallGraphNode *node : currentSCC) {
    if (node->isExternal())
      continue;

    // Don't collect calls if the node is already dead.
    if (useList.isDead(node)) {
      deadNodes.insert(node);
    } else {
      collectCallOps(*node->getCallableRegion(), node, cg, inliner.symbolTable,
                     calls, /*traverseNestedCGNodes=*/false);
    }
  }

  // When inlining a callee produces new call sites, we want to keep track of
  // the fact that they were inlined from the callee. This allows us to avoid
  // infinite inlining.
  using InlineHistoryT = std::optional<size_t>;
  SmallVector<std::pair<CallGraphNode *, InlineHistoryT>, 8> inlineHistory;
  std::vector<InlineHistoryT> callHistory(calls.size(), InlineHistoryT{});

  LLVM_DEBUG({
    llvm::dbgs() << "* Inliner: Initial calls in SCC are: {\n";
    for (unsigned i = 0, e = calls.size(); i < e; ++i)
      llvm::dbgs() << "  " << i << ". " << calls[i].call << ",\n";
    llvm::dbgs() << "}\n";
  });

  // Try to inline each of the call operations. Don't cache the end iterator
  // here as more calls may be added during inlining.
  bool inlinedAnyCalls = false;
  for (unsigned i = 0; i < calls.size(); ++i) {
    if (deadNodes.contains(calls[i].sourceNode))
      continue;
    ResolvedCall it = calls[i];

    InlineHistoryT inlineHistoryID = callHistory[i];
    bool inHistory =
        inlineHistoryIncludes(it.targetNode, inlineHistoryID, inlineHistory);
    bool doInline = !inHistory && shouldInline(it);
    CallOpInterface call = it.call;
    LLVM_DEBUG({
      if (doInline)
        llvm::dbgs() << "* Inlining call: " << i << ". " << call << "\n";
      else
        llvm::dbgs() << "* Not inlining call: " << i << ". " << call << "\n";
    });
    if (!doInline)
      continue;

    unsigned prevSize = calls.size();
    Region *targetRegion = it.targetNode->getCallableRegion();

    // If this is the last call to the target node and the node is discardable,
    // then inline it in-place and delete the node if successful.
    bool inlineInPlace = useList.hasOneUseAndDiscardable(it.targetNode);

    LogicalResult inlineResult = inlineCall(
        inliner, call, cast<CallableOpInterface>(targetRegion->getParentOp()),
        targetRegion, /*shouldCloneInlinedRegion=*/!inlineInPlace);
    if (failed(inlineResult)) {
      LLVM_DEBUG(llvm::dbgs() << "** Failed to inline\n");
      continue;
    }
    inlinedAnyCalls = true;

    // Create a inline history entry for this inlined call, so that we remember
    // that new callsites came about due to inlining Callee.
    InlineHistoryT newInlineHistoryID{inlineHistory.size()};
    inlineHistory.push_back(std::make_pair(it.targetNode, inlineHistoryID));

    auto historyToString = [](InlineHistoryT h) {
      return h.has_value() ? std::to_string(*h) : "root";
    };
    (void)historyToString;
    LLVM_DEBUG(llvm::dbgs()
               << "* new inlineHistory entry: " << newInlineHistoryID << ". ["
               << getNodeName(call) << ", " << historyToString(inlineHistoryID)
               << "]\n");

    for (unsigned k = prevSize; k != calls.size(); ++k) {
      callHistory.push_back(newInlineHistoryID);
      LLVM_DEBUG(llvm::dbgs() << "* new call " << k << " {" << calls[i].call
                              << "}\n   with historyID = " << newInlineHistoryID
                              << ", added due to inlining of\n  call {" << call
                              << "}\n with historyID = "
                              << historyToString(inlineHistoryID) << "\n");
    }

    // If the inlining was successful, Merge the new uses into the source node.
    useList.dropCallUses(it.sourceNode, call.getOperation(), cg);
    useList.mergeUsesAfterInlining(it.targetNode, it.sourceNode);

    // then erase the call.
    call.erase();

    // If we inlined in place, mark the node for deletion.
    if (inlineInPlace) {
      useList.eraseNode(it.targetNode);
      deadNodes.insert(it.targetNode);
    }
  }

  for (CallGraphNode *node : deadNodes) {
    currentSCC.remove(node);
    inliner.markForDeletion(node);
  }
  calls.clear();
  return success(inlinedAnyCalls);
}

//===----------------------------------------------------------------------===//
// InlinerPass
//===----------------------------------------------------------------------===//

namespace {
class InlinerPass : public impl::InlinerBase<InlinerPass> {
public:
  InlinerPass();
  InlinerPass(const InlinerPass &) = default;
  InlinerPass(std::function<void(OpPassManager &)> defaultPipeline);
  InlinerPass(std::function<void(OpPassManager &)> defaultPipeline,
              llvm::StringMap<OpPassManager> opPipelines);
  void runOnOperation() override;

private:
  /// Attempt to inline calls within the given scc, and run simplifications,
  /// until a fixed point is reached. This allows for the inlining of newly
  /// devirtualized calls. Returns failure if there was a fatal error during
  /// inlining.
  LogicalResult inlineSCC(Inliner &inliner, CGUseList &useList,
                          CallGraphSCC &currentSCC, MLIRContext *context);

  /// Optimize the nodes within the given SCC with one of the held optimization
  /// pass pipelines. Returns failure if an error occurred during the
  /// optimization of the SCC, success otherwise.
  LogicalResult optimizeSCC(CallGraph &cg, CGUseList &useList,
                            CallGraphSCC &currentSCC, MLIRContext *context);

  /// Optimize the nodes within the given SCC in parallel. Returns failure if an
  /// error occurred during the optimization of the SCC, success otherwise.
  LogicalResult optimizeSCCAsync(MutableArrayRef<CallGraphNode *> nodesToVisit,
                                 MLIRContext *context);

  /// Optimize the given callable node with one of the pass managers provided
  /// with `pipelines`, or the default pipeline. Returns failure if an error
  /// occurred during the optimization of the callable, success otherwise.
  LogicalResult optimizeCallable(CallGraphNode *node,
                                 llvm::StringMap<OpPassManager> &pipelines);

  /// Attempt to initialize the options of this pass from the given string.
  /// Derived classes may override this method to hook into the point at which
  /// options are initialized, but should generally always invoke this base
  /// class variant.
  LogicalResult initializeOptions(StringRef options) override;

  /// An optional function that constructs a default optimization pipeline for
  /// a given operation.
  std::function<void(OpPassManager &)> defaultPipeline;
  /// A map of operation names to pass pipelines to use when optimizing
  /// callable operations of these types. This provides a specialized pipeline
  /// instead of the default. The vector size is the number of threads used
  /// during optimization.
  SmallVector<llvm::StringMap<OpPassManager>, 8> opPipelines;
};
} // namespace

InlinerPass::InlinerPass() : InlinerPass(defaultInlinerOptPipeline) {}
InlinerPass::InlinerPass(
    std::function<void(OpPassManager &)> defaultPipelineArg)
    : defaultPipeline(std::move(defaultPipelineArg)) {
  opPipelines.push_back({});
}

InlinerPass::InlinerPass(std::function<void(OpPassManager &)> defaultPipeline,
                         llvm::StringMap<OpPassManager> opPipelines)
    : InlinerPass(std::move(defaultPipeline)) {
  if (opPipelines.empty())
    return;

  // Update the option for the op specific optimization pipelines.
  for (auto &it : opPipelines)
    opPipelineList.addValue(it.second);
  this->opPipelines.emplace_back(std::move(opPipelines));
}

void InlinerPass::runOnOperation() {
  CallGraph &cg = getAnalysis<CallGraph>();
  auto *context = &getContext();

  // The inliner should only be run on operations that define a symbol table,
  // as the callgraph will need to resolve references.
  Operation *op = getOperation();
  if (!op->hasTrait<OpTrait::SymbolTable>()) {
    op->emitOpError() << " was scheduled to run under the inliner, but does "
                         "not define a symbol table";
    return signalPassFailure();
  }

  // Run the inline transform in post-order over the SCCs in the callgraph.
  SymbolTableCollection symbolTable;
  Inliner inliner(context, cg, symbolTable);
  CGUseList useList(getOperation(), cg, symbolTable);
  LogicalResult result = runTransformOnCGSCCs(cg, [&](CallGraphSCC &scc) {
    return inlineSCC(inliner, useList, scc, context);
  });
  if (failed(result))
    return signalPassFailure();

  // After inlining, make sure to erase any callables proven to be dead.
  inliner.eraseDeadCallables();
}

LogicalResult InlinerPass::inlineSCC(Inliner &inliner, CGUseList &useList,
                                     CallGraphSCC &currentSCC,
                                     MLIRContext *context) {
  // Continuously simplify and inline until we either reach a fixed point, or
  // hit the maximum iteration count. Simplifying early helps to refine the cost
  // model, and in future iterations may devirtualize new calls.
  unsigned iterationCount = 0;
  do {
    if (failed(optimizeSCC(inliner.cg, useList, currentSCC, context)))
      return failure();
    if (failed(inlineCallsInSCC(inliner, useList, currentSCC)))
      break;
  } while (++iterationCount < maxInliningIterations);
  return success();
}

LogicalResult InlinerPass::optimizeSCC(CallGraph &cg, CGUseList &useList,
                                       CallGraphSCC &currentSCC,
                                       MLIRContext *context) {
  // Collect the sets of nodes to simplify.
  SmallVector<CallGraphNode *, 4> nodesToVisit;
  for (auto *node : currentSCC) {
    if (node->isExternal())
      continue;

    // Don't simplify nodes with children. Nodes with children require special
    // handling as we may remove the node during simplification. In the future,
    // we should be able to handle this case with proper node deletion tracking.
    if (node->hasChildren())
      continue;

    // We also won't apply simplifications to nodes that can't have passes
    // scheduled on them.
    auto *region = node->getCallableRegion();
    if (!region->getParentOp()->hasTrait<OpTrait::IsIsolatedFromAbove>())
      continue;
    nodesToVisit.push_back(node);
  }
  if (nodesToVisit.empty())
    return success();

  // Optimize each of the nodes within the SCC in parallel.
  if (failed(optimizeSCCAsync(nodesToVisit, context)))
    return failure();

  // Recompute the uses held by each of the nodes.
  for (CallGraphNode *node : nodesToVisit)
    useList.recomputeUses(node, cg);
  return success();
}

LogicalResult
InlinerPass::optimizeSCCAsync(MutableArrayRef<CallGraphNode *> nodesToVisit,
                              MLIRContext *ctx) {
  // We must maintain a fixed pool of pass managers which is at least as large
  // as the maximum parallelism of the failableParallelForEach below.
  // Note: The number of pass managers here needs to remain constant
  // to prevent issues with pass instrumentations that rely on having the same
  // pass manager for the main thread.
  size_t numThreads = ctx->getNumThreads();
  if (opPipelines.size() < numThreads) {
    // Reserve before resizing so that we can use a reference to the first
    // element.
    opPipelines.reserve(numThreads);
    opPipelines.resize(numThreads, opPipelines.front());
  }

  // Ensure an analysis manager has been constructed for each of the nodes.
  // This prevents thread races when running the nested pipelines.
  for (CallGraphNode *node : nodesToVisit)
    getAnalysisManager().nest(node->getCallableRegion()->getParentOp());

  // An atomic failure variable for the async executors.
  std::vector<std::atomic<bool>> activePMs(opPipelines.size());
  std::fill(activePMs.begin(), activePMs.end(), false);
  return failableParallelForEach(ctx, nodesToVisit, [&](CallGraphNode *node) {
    // Find a pass manager for this operation.
    auto it = llvm::find_if(activePMs, [](std::atomic<bool> &isActive) {
      bool expectedInactive = false;
      return isActive.compare_exchange_strong(expectedInactive, true);
    });
    assert(it != activePMs.end() &&
           "could not find inactive pass manager for thread");
    unsigned pmIndex = it - activePMs.begin();

    // Optimize this callable node.
    LogicalResult result = optimizeCallable(node, opPipelines[pmIndex]);

    // Reset the active bit for this pass manager.
    activePMs[pmIndex].store(false);
    return result;
  });
}

LogicalResult
InlinerPass::optimizeCallable(CallGraphNode *node,
                              llvm::StringMap<OpPassManager> &pipelines) {
  Operation *callable = node->getCallableRegion()->getParentOp();
  StringRef opName = callable->getName().getStringRef();
  auto pipelineIt = pipelines.find(opName);
  if (pipelineIt == pipelines.end()) {
    // If a pipeline didn't exist, use the default if possible.
    if (!defaultPipeline)
      return success();

    OpPassManager defaultPM(opName);
    defaultPipeline(defaultPM);
    pipelineIt = pipelines.try_emplace(opName, std::move(defaultPM)).first;
  }
  return runPipeline(pipelineIt->second, callable);
}

LogicalResult InlinerPass::initializeOptions(StringRef options) {
  if (failed(Pass::initializeOptions(options)))
    return failure();

  // Initialize the default pipeline builder to use the option string.
  // TODO: Use a generic pass manager for default pipelines, and remove this.
  if (!defaultPipelineStr.empty()) {
    std::string defaultPipelineCopy = defaultPipelineStr;
    defaultPipeline = [=](OpPassManager &pm) {
      (void)parsePassPipeline(defaultPipelineCopy, pm);
    };
  } else if (defaultPipelineStr.getNumOccurrences()) {
    defaultPipeline = nullptr;
  }

  // Initialize the op specific pass pipelines.
  llvm::StringMap<OpPassManager> pipelines;
  for (OpPassManager pipeline : opPipelineList)
    if (!pipeline.empty())
      pipelines.try_emplace(pipeline.getOpAnchorName(), pipeline);
  opPipelines.assign({std::move(pipelines)});

  return success();
}

std::unique_ptr<Pass> mlir::createInlinerPass() {
  return std::make_unique<InlinerPass>();
}
std::unique_ptr<Pass>
mlir::createInlinerPass(llvm::StringMap<OpPassManager> opPipelines) {
  return std::make_unique<InlinerPass>(defaultInlinerOptPipeline,
                                       std::move(opPipelines));
}
std::unique_ptr<Pass> mlir::createInlinerPass(
    llvm::StringMap<OpPassManager> opPipelines,
    std::function<void(OpPassManager &)> defaultPipelineBuilder) {
  return std::make_unique<InlinerPass>(std::move(defaultPipelineBuilder),
                                       std::move(opPipelines));
}
