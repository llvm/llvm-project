//===--------------------- LogicCombine.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file attempts to find the simplest expression for a bitwise logic
/// operation chain. We canonicalize all other ops to "&"/"^".
/// For example:
///    a | b --> (a & b) ^ a ^ b
///    c ? a : b --> (c & a) ^ ((c ^ true) & b)
/// We use a set of bitset to represent the expression. Any value that is not a
/// logic operation is a leaf node. Leaf node is 1 bit in the bitset. For
/// example, we have source a, b, c. The bit for a is 1, b is 2, c is 4.
///     a & b & c --> {0b111}
///     a & b ^ c & a --> {0b011, 0b101}
///     a & b ^ c & a ^ b --> {0b011, 0b101, 0b010}
/// Every bitset is an "&" chain. The set of bitset is a "^" chain.
/// Based on boolean ring, we can treat "&" as ring multiplication and "^" as
/// ring addition. After that, any logic value can be represented as a chain of
/// bitsets. For example:
///     r1 = (a | b) & c -> r1 = (a * b * c) + (a * c) + (b * c) ->
///     {0b111, 0b101, 0b110}
/// Finally we need to rebuild the simplest pattern from the expression.
///
/// Reference: https://en.wikipedia.org/wiki/Boolean_ring
///
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LogicCombine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "logic-combine"

STATISTIC(NumLogicalOpsSimplified, "Number of logical operations simplified");

static cl::opt<unsigned> MaxLogicOpLeafsToScan(
    "logic-combine-max-leafs", cl::init(8), cl::Hidden,
    cl::desc("Max leafs of logic ops to scan for logical combine."));

static cl::opt<unsigned> MaxDepthLogicOpsToScan(
    "logic-combine-max-depth", cl::init(8), cl::Hidden,
    cl::desc("Max depth of logic ops to scan for logical combine."));

void LogicalOpNode::printAndChain(raw_ostream &OS, uint64_t LeafBits) const {
  if (LeafBits == LogicalExpr::ExprAllOne) {
    OS << "-1";
    return;
  }

  if (LeafBits == 0)
    return;

  unsigned LeafCnt = popcount(LeafBits);
  if (LeafCnt == 1) {
    Helper->LeafValues[Log2_64(LeafBits)]->printAsOperand(OS, false);
    return;
  }

  unsigned LeafIdx;
  ListSeparator LS(" * ");
  for (unsigned I = 0; I < LeafCnt; I++) {
    LeafIdx = countr_zero(LeafBits);
    OS << LS;
    Helper->LeafValues[LeafIdx]->printAsOperand(OS, false);
    LeafBits -= (1ULL << LeafIdx);
  }
}

void LogicalOpNode::print(raw_ostream &OS) const {
  Val->printAsOperand(OS, false);
  OS << " --> ";
  if (Expr.size() == 0) {
    OS << "0\n";
    return;
  }

  ListSeparator LS(" + ");
  for (auto I = Expr.begin(); I != Expr.end(); I++) {
    OS << LS;
    printAndChain(OS, *I);
  }

  OS << "\n";
}

void LogicCombiner::clear() {
  LogicalOpNodes.clear();
  LeafValues.clear();
}

LogicalOpNode *LogicCombiner::visitLeafNode(Value *Val, unsigned Depth) {
  // Depth is 0 means the root is not logical operation. We can't
  // do anything for that.
  if (Depth == 0 || LeafValues.size() >= MaxLogicOpLeafsToScan)
    return nullptr;

  uint64_t ExprVal = 1ULL << LeafValues.size();
  // Constant Zero,AllOne are special leaf nodes. They involve
  // LogicalExpr's calculation so we must detect them at first.
  if (auto ConstVal = dyn_cast<ConstantInt>(Val)) {
    if (ConstVal->isZero())
      ExprVal = 0;
    else if (ConstVal->isAllOnesValue())
      ExprVal = LogicalExpr::ExprAllOne;
  }
  if (ExprVal != LogicalExpr::ExprAllOne && ExprVal != 0)
    LeafValues.insert(Val);
  LogicalOpNode *Node =
      new (Alloc.Allocate()) LogicalOpNode(this, Val, LogicalExpr(ExprVal));
  LogicalOpNodes[Val] = Node;
  return Node;
}

LogicalOpNode *LogicCombiner::visitBinOp(BinaryOperator *BO, unsigned Depth) {
  if (!BO->isBitwiseLogicOp())
    return visitLeafNode(BO, Depth);

  LogicalOpNode *LHS = getLogicalOpNode(BO->getOperand(0), Depth + 1);
  if (LHS == nullptr)
    return nullptr;

  LogicalOpNode *RHS = getLogicalOpNode(BO->getOperand(1), Depth + 1);
  if (RHS == nullptr)
    return nullptr;

  LogicalOpNode *Node;
  if (BO->getOpcode() == Instruction::And)
    Node = new (Alloc.Allocate())
        LogicalOpNode(this, BO, LHS->getExpr() & RHS->getExpr());
  else if (BO->getOpcode() == Instruction::Or)
    Node = new (Alloc.Allocate())
        LogicalOpNode(this, BO, LHS->getExpr() | RHS->getExpr());
  else
    Node = new (Alloc.Allocate())
        LogicalOpNode(this, BO, LHS->getExpr() ^ RHS->getExpr());
  LogicalOpNodes[BO] = Node;
  return Node;
}

LogicalOpNode *LogicCombiner::getLogicalOpNode(Value *Val, unsigned Depth) {
  if (Depth == MaxDepthLogicOpsToScan)
    return nullptr;

  if (LogicalOpNodes.find(Val) == LogicalOpNodes.end()) {
    LogicalOpNode *Node;

    // TODO: add select instruction support
    if (auto *BO = dyn_cast<BinaryOperator>(Val))
      Node = visitBinOp(BO, Depth);
    else
      Node = visitLeafNode(Val, Depth);

    if (!Node)
      return nullptr;
    LLVM_DEBUG(dbgs() << *Node);
  }
  return LogicalOpNodes[Val];
}

Value *LogicCombiner::logicalOpToValue(LogicalOpNode *Node) {
  const LogicalExpr &Expr = Node->getExpr();
  // Empty when all leaf bits are erased from the set because a ^ a = 0.
  if (Expr.size() == 0)
    return Constant::getNullValue(Node->getValue()->getType());

  if (Expr.size() == 1) {
    uint64_t LeafBits = *Expr.begin();
    if (LeafBits == 0)
      return Constant::getNullValue(Node->getValue()->getType());
    // ExprAllOne is not in the LeafValues
    if (LeafBits == LogicalExpr::ExprAllOne)
      return Constant::getAllOnesValue(Node->getValue()->getType());

    if (popcount(LeafBits) == 1)
      return LeafValues[Log2_64(LeafBits)];
  }

  // TODO: find the simplest form from logical expression when it is not
  // only an "and" chain.

  return nullptr;
}

Value *LogicCombiner::simplify(Value *Root) {
  assert(MaxLogicOpLeafsToScan <= 63 &&
         "Logical leaf node can't be larger than 63.");
  LogicalOpNode *RootNode = getLogicalOpNode(Root);
  if (RootNode == nullptr)
    return nullptr;

  Value *NewRoot = logicalOpToValue(RootNode);
  if (NewRoot == nullptr || NewRoot == Root)
    return nullptr;

  LogicalOpNodes.erase(Root);
  NumLogicalOpsSimplified++;
  return NewRoot;
}
