//===-- OperandGraph.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "OperandGraph.h"
#include "llvm/MC/MCRegisterInfo.h"

namespace exegesis {
namespace graph {

void Node::dump(const llvm::MCRegisterInfo &RegInfo) const {
  switch (type()) {
  case NodeType::VARIABLE:
    printf(" %d", varValue());
    break;
  case NodeType::REG:
    printf(" %s", RegInfo.getName(regValue()));
    break;
  case NodeType::IN:
    printf(" IN");
    break;
  case NodeType::OUT:
    printf(" OUT");
    break;
  }
}

NodeType Node::type() const { return first; }

int Node::regValue() const {
  assert(first == NodeType::REG && "regValue() called on non-reg");
  return second;
}

int Node::varValue() const {
  assert(first == NodeType::VARIABLE && "varValue() called on non-var");
  return second;
}

void Graph::connect(const Node From, const Node To) {
  AdjacencyLists[From].insert(To);
}

void Graph::disconnect(const Node From, const Node To) {
  AdjacencyLists[From].erase(To);
}

std::vector<Node> Graph::getPathFrom(const Node From, const Node To) const {
  std::vector<Node> Path;
  NodeSet Seen;
  dfs(From, To, Path, Seen);
  return Path;
}

// DFS is implemented recursively, this is fine as graph size is small (~250
// nodes, ~200 edges, longuest path depth < 10).
bool Graph::dfs(const Node Current, const Node Sentinel,
                std::vector<Node> &Path, NodeSet &Seen) const {
  Path.push_back(Current);
  Seen.insert(Current);
  if (Current == Sentinel)
    return true;
  if (AdjacencyLists.count(Current)) {
    for (const Node Next : AdjacencyLists.find(Current)->second) {
      if (Seen.count(Next))
        continue;
      if (dfs(Next, Sentinel, Path, Seen))
        return true;
    }
  }
  Path.pop_back();
  return false;
}

// For each Register Units we walk up their parents.
// Let's take the case of the A register family:
//
//  RAX
//   ^
//   EAX
//    ^
//    AX
//   ^  ^
//  AH  AL
//
// Register Units are AH and AL.
// Walking them up gives the following lists:
// AH->AX->EAX->RAX and AL->AX->EAX->RAX
// When walking the lists we add connect current to parent both ways leading to
// the following connections:
//
// AL<->AX, AH<->AX, AX<->EAX, EAX<->RAX
// We repeat this process for all Unit Registers to cover all connections.
void setupRegisterAliasing(const llvm::MCRegisterInfo &RegInfo,
                           Graph &TheGraph) {
  using SuperItr = llvm::MCSuperRegIterator;
  for (size_t Reg = 0, E = RegInfo.getNumRegUnits(); Reg < E; ++Reg) {
    size_t Current = Reg;
    for (SuperItr Super(Reg, &RegInfo); Super.isValid(); ++Super) {
      const Node A = Node::Reg(Current);
      const Node B = Node::Reg(*Super);
      TheGraph.connect(A, B);
      TheGraph.connect(B, A);
      Current = *Super;
    }
  }
}

} // namespace graph
} // namespace exegesis
