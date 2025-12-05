//===- RippleTestBase.h ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines a RippleTestBase class, which provides helpers to create a
/// Ripple object and LLVM Functions for testing.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_TRANSFORMS_RIPPLE_RIPPLETESTBASE_H
#define LLVM_UNITTESTS_TRANSFORMS_RIPPLE_RIPPLETESTBASE_H

#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

namespace llvm {

class RippleTestBase : public testing::Test {
protected:
  LLVMContext C;

  RippleTestBase() {}
};

} // namespace llvm

#endif // LLVM_UNITTESTS_TRANSFORMS_RIPPLE_RIPPLETESTBASE_H
