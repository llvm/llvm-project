//===------------------ LogicCombine.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LOGICCOMBINE_H
#define LLVM_ANALYSIS_LOGICCOMBINE_H

#include "LogicalExpr.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/Allocator.h"

namespace llvm {

class LogicCombiner;

class LogicalOpNode {
private:
  LogicCombiner *Helper;
  Value *Val;
  LogicalExpr Expr;
  // TODO: Add weight to measure cost for more than one use value

  void printAndChain(raw_ostream &OS, uint64_t LeafBits) const;

public:
  LogicalOpNode(LogicCombiner *OpsHelper, Value *SrcVal,
                const LogicalExpr &SrcExpr)
      : Helper(OpsHelper), Val(SrcVal), Expr(SrcExpr) {}
  ~LogicalOpNode() {}

  Value *getValue() const { return Val; }
  const LogicalExpr &getExpr() const { return Expr; }
  void print(raw_ostream &OS) const;
};

class LogicCombiner {
public:
  LogicCombiner() {}
  ~LogicCombiner() { clear(); }

  Value *simplify(Value *Root);

private:
  friend class LogicalOpNode;

  SpecificBumpPtrAllocator<LogicalOpNode> Alloc;
  SmallDenseMap<Value *, LogicalOpNode *, 16> LogicalOpNodes;
  SmallSetVector<Value *, 8> LeafValues;

  void clear();

  LogicalOpNode *visitLeafNode(Value *Val, unsigned Depth);
  LogicalOpNode *visitBinOp(BinaryOperator *BO, unsigned Depth);
  LogicalOpNode *getLogicalOpNode(Value *Val, unsigned Depth = 0);
  Value *logicalOpToValue(LogicalOpNode *Node);
};

inline raw_ostream &operator<<(raw_ostream &OS, const LogicalOpNode &I) {
  I.print(OS);
  return OS;
}

} // namespace llvm

#endif // LLVM_ANALYSIS_LOGICCOMBINE_H
