//===- LLVMTestBase.h - Test fixure for LLVM dialect tests ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test fixure for LLVM dialect tests.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_UNITTEST_DIALECT_LLVMIR_LLVMTESTBASE_H
#define MLIR_UNITTEST_DIALECT_LLVMIR_LLVMTESTBASE_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

class LLVMIRTest : public ::testing::Test {
protected:
  LLVMIRTest() { context.getOrLoadDialect<mlir::LLVM::LLVMDialect>(); }

  mlir::MLIRContext context;
};

#endif
