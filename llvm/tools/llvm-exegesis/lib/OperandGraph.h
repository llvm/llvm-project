//===-- OperandGraph.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// A collection of tools to model register aliasing and instruction operand.
/// This is used to find an aliasing between the input and output registers of
/// an instruction. It allows us to repeat an instruction and make sure that
/// successive instances are executed sequentially.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_OPERANDGRAPH_H
#define LLVM_TOOLS_LLVM_EXEGESIS_OPERANDGRAPH_H

#include "llvm/MC/MCRegisterInfo.h"
#include <map>
#include <set>
#include <tuple>
#include <vector>

namespace exegesis {
namespace graph {

enum class NodeType {
  VARIABLE, // An set of "tied together operands" to resolve.
  REG,      // A particular register.
  IN,       // The input node.
  OUT       // The output node.
};

// A Node in the graph, it has a type and an int value.
struct Node : public std::pair<NodeType, int> {
  using std::pair<NodeType, int>::pair;

  static Node Reg(int Value) { return {NodeType::REG, Value}; }
  static Node Var(int Value) { return {NodeType::VARIABLE, Value}; }
  static Node In() { return {NodeType::IN, 0}; }
  static Node Out() { return {NodeType::OUT, 0}; }

  NodeType type() const;
  int regValue() const; // checks that type==REG and returns value.
  int varValue() const; // checks that type==VARIABLE and returns value.

  void dump(const llvm::MCRegisterInfo &RegInfo) const;
};

// Graph represents the connectivity of registers for a particular instruction.
// This object is used to select registers that would create a dependency chain
// between instruction's input and output.
struct Graph {
public:
  void connect(const Node From, const Node To);
  void disconnect(const Node From, const Node To);

  // Tries to find a path between 'From' and 'To' nodes.
  // Returns empty if no path is found.
  std::vector<Node> getPathFrom(const Node From, const Node To) const;

private:
  // We use std::set to keep the implementation simple, using an unordered_set
  // requires the definition of a hasher.
  using NodeSet = std::set<Node>;

  // Performs a Depth First Search from 'current' node up until 'sentinel' node
  // is found. 'path' is the recording of the traversed nodes, 'seen' is the
  // collection of nodes seen so far.
  bool dfs(const Node Current, const Node Sentinel, std::vector<Node> &Path,
           NodeSet &Seen) const;

  // We use std::map to keep the implementation simple, using an unordered_map
  // requires the definition of a hasher.
  std::map<Node, NodeSet> AdjacencyLists;
};

// Add register nodes to graph and connect them when they alias. Connection is
// both ways.
void setupRegisterAliasing(const llvm::MCRegisterInfo &RegInfo,
                           Graph &TheGraph);

} // namespace graph
} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_OPERANDGRAPH_H
