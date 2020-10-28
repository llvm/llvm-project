//===- llvm/unittest/Transforms/IPO/AttributorTestBase.h -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines an AttributorTestBase class, which provides helpers to
/// parse a LLVM IR string and create Attributor
//===----------------------------------------------------------------------===//
#ifndef LLVM_UNITTESTS_TRANSFORMS_ATTRIBUTOR_TESTBASE_H
#define LLVM_UNITTESTS_TRANSFORMS_ATTRIBUTOR_TESTBASE_H

#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Transforms/IPO/Attributor.h"
#include "llvm/Transforms/Utils/CallGraphUpdater.h"
#include "gtest/gtest.h"
#include <memory>

namespace llvm {

/// Helper class to create a module from assembly string and an Attributor
class AttributorTestBase : public testing::Test {
protected:
  std::unique_ptr<LLVMContext> Ctx;
  std::unique_ptr<Module> M;

  AttributorTestBase() : Ctx(new LLVMContext) {}

  Module &parseModule(const char *ModuleString) {
    SMDiagnostic Err;
    M = parseAssemblyString(ModuleString, Err, *Ctx);
    EXPECT_TRUE(M);
    return *M;
  }
};

} // namespace llvm

#endif