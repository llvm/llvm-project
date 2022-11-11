//===- CallGraph.cpp - CallGraph analysis for MLIR ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains interfaces and analyses for defining a nested callgraph.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/CallGraph.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// CallGraphNode
//===----------------------------------------------------------------------===//

/// Returns true if this node refers to the indirect/external node.
bool CallGraphNode::isExternal() const { return !callableRegion; }

/// Return the callable region this node represents. This can only be called
/// on non-external nodes.
Region *CallGraphNode::getCallableRegion() const {
  assert(!isExternal() && "the external node has no callable region");
  return callableRegion;
}

/// Adds an reference edge to the given node. This is only valid on the
/// external node.
void CallGraphNode::addAbstractEdge(CallGraphNode *node) {
  assert(isExternal() && "abstract edges are only valid on external nodes");
  addEdge(node, Edge::Kind::Abstract);
}

/// Add an outgoing call edge from this node.
void CallGraphNode::addCallEdge(CallGraphNode *node) {
  addEdge(node, Edge::Kind::Call);
}

/// Adds a reference edge to the given child node.
void CallGraphNode::addChildEdge(CallGraphNode *child) {
  addEdge(child, Edge::Kind::Child);
}

/// Returns true if this node has any child edges.
bool CallGraphNode::hasChildren() const {
  return llvm::any_of(edges, [](const Edge &edge) { return edge.isChild(); });
}

/// Add an edge to 'node' with the given kind.
void CallGraphNode::addEdge(CallGraphNode *node, Edge::Kind kind) {
  edges.insert({node, kind});
}

//===----------------------------------------------------------------------===//
// CallGraph
//===----------------------------------------------------------------------===//

/// Recursively compute the callgraph edges for the given operation. Computed
/// edges are placed into the given callgraph object.
static void computeCallGraph(Operation *op, CallGraph &cg,
                             SymbolTableCollection &symbolTable,
                             CallGraphNode *parentNode, bool resolveCalls) {
  if (CallOpInterface call = dyn_cast<CallOpInterface>(op)) {
    // If there is no parent node, we ignore this operation. Even if this
    // operation was a call, there would be no callgraph node to attribute it
    // to.
    if (resolveCalls && parentNode)
      parentNode->addCallEdge(cg.resolveCallable(call, symbolTable));
    return;
  }

  // Compute the callgraph nodes and edges for each of the nested operations.
  if (CallableOpInterface callable = dyn_cast<CallableOpInterface>(op)) {
    if (auto *callableRegion = callable.getCallableRegion())
      parentNode = cg.getOrAddNode(callableRegion, parentNode);
    else
      return;
  }

  for (Region &region : op->getRegions())
    for (Operation &nested : region.getOps())
      computeCallGraph(&nested, cg, symbolTable, parentNode, resolveCalls);
}

CallGraph::CallGraph(Operation *op)
    : externalCallerNode(/*callableRegion=*/nullptr),
      unknownCalleeNode(/*callableRegion=*/nullptr) {
  // Make two passes over the graph, one to compute the callables and one to
  // resolve the calls. We split these up as we may have nested callable objects
  // that need to be reserved before the calls.
  SymbolTableCollection symbolTable;
  computeCallGraph(op, *this, symbolTable, /*parentNode=*/nullptr,
                   /*resolveCalls=*/false);
  computeCallGraph(op, *this, symbolTable, /*parentNode=*/nullptr,
                   /*resolveCalls=*/true);
}

/// Get or add a call graph node for the given region.
CallGraphNode *CallGraph::getOrAddNode(Region *region,
                                       CallGraphNode *parentNode) {
  assert(region && isa<CallableOpInterface>(region->getParentOp()) &&
         "expected parent operation to be callable");
  std::unique_ptr<CallGraphNode> &node = nodes[region];
  if (!node) {
    node.reset(new CallGraphNode(region));

    // Add this node to the given parent node if necessary.
    if (parentNode) {
      parentNode->addChildEdge(node.get());
    } else {
      // Otherwise, connect all callable nodes to the external node, this allows
      // for conservatively including all callable nodes within the graph.
      // FIXME This isn't correct, this is only necessary for callable nodes
      // that *could* be called from external sources. This requires extending
      // the interface for callables to check if they may be referenced
      // externally.
      externalCallerNode.addAbstractEdge(node.get());
    }
  }
  return node.get();
}

/// Lookup a call graph node for the given region, or nullptr if none is
/// registered.
CallGraphNode *CallGraph::lookupNode(Region *region) const {
  auto it = nodes.find(region);
  return it == nodes.end() ? nullptr : it->second.get();
}

/// Resolve the callable for given callee to a node in the callgraph, or the
/// unknown callee node if a valid node was not resolved.
CallGraphNode *
CallGraph::resolveCallable(CallOpInterface call,
                           SymbolTableCollection &symbolTable) const {
  Operation *callable = call.resolveCallable(&symbolTable);
  if (auto callableOp = dyn_cast_or_null<CallableOpInterface>(callable))
    if (auto *node = lookupNode(callableOp.getCallableRegion()))
      return node;

  return getUnknownCalleeNode();
}

/// Erase the given node from the callgraph.
void CallGraph::eraseNode(CallGraphNode *node) {
  // Erase any children of this node first.
  if (node->hasChildren()) {
    for (const CallGraphNode::Edge &edge : llvm::make_early_inc_range(*node))
      if (edge.isChild())
        eraseNode(edge.getTarget());
  }
  // Erase any edges to this node from any other nodes.
  for (auto &it : nodes) {
    it.second->edges.remove_if([node](const CallGraphNode::Edge &edge) {
      return edge.getTarget() == node;
    });
  }
  nodes.erase(node->getCallableRegion());
}

//===----------------------------------------------------------------------===//
// Printing

/// Dump the graph in a human readable format.
void CallGraph::dump() const { print(llvm::errs()); }
void CallGraph::print(raw_ostream &os) const {
  os << "// ---- CallGraph ----\n";

  // Functor used to output the name for the given node.
  auto emitNodeName = [&](const CallGraphNode *node) {
    if (node == getExternalCallerNode()) {
      os << "<External-Caller-Node>";
      return;
    }
    if (node == getUnknownCalleeNode()) {
      os << "<Unknown-Callee-Node>";
      return;
    }

    auto *callableRegion = node->getCallableRegion();
    auto *parentOp = callableRegion->getParentOp();
    os << "'" << callableRegion->getParentOp()->getName() << "' - Region #"
       << callableRegion->getRegionNumber();
    auto attrs = parentOp->getAttrDictionary();
    if (!attrs.empty())
      os << " : " << attrs;
  };

  for (auto &nodeIt : nodes) {
    const CallGraphNode *node = nodeIt.second.get();

    // Dump the header for this node.
    os << "// - Node : ";
    emitNodeName(node);
    os << "\n";

    // Emit each of the edges.
    for (auto &edge : *node) {
      os << "// -- ";
      if (edge.isCall())
        os << "Call";
      else if (edge.isChild())
        os << "Child";

      os << "-Edge : ";
      emitNodeName(edge.getTarget());
      os << "\n";
    }
    os << "//\n";
  }

  os << "// -- SCCs --\n";

  for (auto &scc : make_range(llvm::scc_begin(this), llvm::scc_end(this))) {
    os << "// - SCC : \n";
    for (auto &node : scc) {
      os << "// -- Node :";
      emitNodeName(node);
      os << "\n";
    }
    os << "\n";
  }

  os << "// -------------------\n";
}
