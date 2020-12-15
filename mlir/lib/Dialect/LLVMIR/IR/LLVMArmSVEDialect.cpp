//===- LLVMArmSVEDialect.cpp - MLIR LLVMSVE ops implementation ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLVMArmSVE dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/IntrinsicsAArch64.h"

#include "mlir/Dialect/LLVMIR/LLVMArmSVEDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

void LLVM::LLVMArmSVEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LLVMIR/LLVMArmSVE.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMArmSVE.cpp.inc"
