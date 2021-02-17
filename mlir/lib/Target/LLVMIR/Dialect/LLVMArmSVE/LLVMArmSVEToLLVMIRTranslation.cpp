//===- LLVMArmSVEToLLVMIRTranslation.cpp - Translate LLVMArmSVE to LLVM IR-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR LLVMArmSVE dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/LLVMArmSVE/LLVMArmSVEToLLVMIRTranslation.h"
#include "mlir/Dialect/LLVMIR/LLVMArmSVEDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAArch64.h"

using namespace mlir;
using namespace mlir::LLVM;

LogicalResult
mlir::LLVMArmSVEDialectLLVMIRTranslationInterface::convertOperation(
    Operation *op, llvm::IRBuilderBase &builder,
    LLVM::ModuleTranslation &moduleTranslation) const {
  Operation &opInst = *op;
#include "mlir/Dialect/LLVMIR/LLVMArmSVEConversions.inc"

  return failure();
}
