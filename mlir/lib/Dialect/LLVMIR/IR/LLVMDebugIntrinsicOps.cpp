//===- LLVMDebugIntrinsicOps.cpp - LLVM dialect debug intrinsics ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `DebugIntrinsic` of the LLVM
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDebugIntrinsicOps.h"

#include "IR/LLVMOps.h"
#include "mlir/Dialect/LLVMIR/LLVMConstantOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/MemorySlotInterfaces.h"

using namespace mlir;
using namespace mlir::LLVM;

bool LLVM::DbgDeclareOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  return true;
}

DeletionKind LLVM::DbgDeclareOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder) {
  return DeletionKind::Delete;
}

bool LLVM::DbgValueOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses,
    const DataLayout &dataLayout) {
  // There is only one operand that we can remove the use of.
  if (blockingUses.size() != 1)
    return false;

  return (*blockingUses.begin())->get() == getValue();
}

DeletionKind LLVM::DbgValueOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, OpBuilder &builder) {
  // builder by default is after '*this', but we need it before '*this'.
  builder.setInsertionPoint(*this);

  // Rather than dropping the debug value, replace it with undef to preserve the
  // debug local variable info. This allows the debugger to inform the user that
  // the variable has been optimized out.
  auto undef =
      UndefOp::create(builder, getValue().getLoc(), getValue().getType());
  getValueMutable().assign(undef);
  return DeletionKind::Keep;
}

bool LLVM::DbgDeclareOp::requiresReplacedValues() { return true; }

void LLVM::DbgDeclareOp::visitReplacedValues(
    ArrayRef<std::pair<Operation *, Value>> definitions, OpBuilder &builder) {
  for (auto [op, value] : definitions) {
    builder.setInsertionPointAfter(op);
    LLVM::DbgValueOp::create(builder, getLoc(), value, getVarInfo(),
                             getLocationExpr());
  }
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMDebugIntrinsicOps.cpp.inc"
