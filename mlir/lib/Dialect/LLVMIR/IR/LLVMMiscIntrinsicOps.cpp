//===- LLVMMiscIntrinsicOps.cpp - LLVM dialect other intrinsics -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `MiscIntrinsic` of the LLVM
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMMiscIntrinsicOps.h"

#include "IR/LLVMOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::LLVM;

//===----------------------------------------------------------------------===//
// AssumeOp (intrinsic)
//===----------------------------------------------------------------------===//

void LLVM::AssumeOp::build(OpBuilder &builder, OperationState &state,
                           mlir::Value cond) {
  return build(builder, state, cond, /*op_bundle_operands=*/{},
               /*op_bundle_tags=*/ArrayAttr{});
}

void LLVM::AssumeOp::build(OpBuilder &builder, OperationState &state,
                           Value cond, llvm::StringRef tag, ValueRange args) {
  return build(builder, state, cond, ArrayRef<ValueRange>(args),
               builder.getStrArrayAttr(tag));
}

void LLVM::AssumeOp::build(OpBuilder &builder, OperationState &state,
                           Value cond, AssumeAlignTag, Value ptr, Value align) {
  return build(builder, state, cond, "align", ValueRange{ptr, align});
}

void LLVM::AssumeOp::build(OpBuilder &builder, OperationState &state,
                           Value cond, AssumeSeparateStorageTag, Value ptr1,
                           Value ptr2) {
  return build(builder, state, cond, "separate_storage",
               ValueRange{ptr1, ptr2});
}

LogicalResult LLVM::AssumeOp::verify() { return verifyOperandBundles(*this); }

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMMiscIntrinsicOps.cpp.inc"
