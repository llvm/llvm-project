//===- LLVMArmNeonToLLVMIRTranslation.cpp - LLVMArmNeon to LLVM IR --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR LLVMArmNeon dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/LLVMArmNeon/LLVMArmNeonToLLVMIRTranslation.h"
#include "mlir/Dialect/LLVMIR/LLVMArmNeonDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAArch64.h"

using namespace mlir;
using namespace mlir::LLVM;

LogicalResult
mlir::LLVMArmNeonDialectLLVMIRTranslationInterface::convertOperation(
    Operation *op, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) const {
  Operation &opInst = *op;
#include "mlir/Dialect/LLVMIR/LLVMArmNeonConversions.inc"

  return failure();
}
