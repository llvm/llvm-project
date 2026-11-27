//===- SerializeXeVMTarget.cpp ----------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVM/XeVM/Utils.h"

#include "llvm/Support/Regex.h"

#include "gtest/gtest.h"

using namespace mlir;

// The compiler info is either empty (no compiler found) or a version string.
TEST(MLIRTargetLLVMXeVM, CompilerInfo) {
  std::string info = xevm::SerializeGPUModuleBase::getCompilerInfo();
  if (!info.empty())
    EXPECT_TRUE(llvm::Regex("^[0-9]+(\\.[0-9]+)+$").match(info)) << info;
}
