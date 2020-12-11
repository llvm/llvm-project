//===- LLVMArmNeonDialect.cpp - MLIR LLVMArmNeon ops implementation -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLVMArmNeon dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/IntrinsicsAArch64.h"

#include "mlir/Dialect/LLVMIR/LLVMArmNeonDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

void LLVM::LLVMArmNeonDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LLVMIR/LLVMArmNeon.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMArmNeon.cpp.inc"
