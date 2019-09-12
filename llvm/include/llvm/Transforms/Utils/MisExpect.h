//===--- MisExpect.h - Check the use of llvm.expect with PGO data ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit warnings for potentially incorrect usage of the
// llvm.expect intrinsic. This utility extracts the threshold values from
// metadata associated with the instrumented Branch or Switch instruction. The
// threshold values are then used to determine if a warning should be emmited.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"

namespace llvm {
namespace misexpect {

/// verifyMisExpect - compares PGO counters to the thresholds used for
/// llvm.expect and warns if the PGO counters are outside of the expected
/// range.
/// \param I The Instruction being checked
/// \param Weights A vector of profile weights for each target block
/// \param Ctx The current LLVM context
void verifyMisExpect(llvm::Instruction *I,
                     const llvm::SmallVector<uint32_t, 4> &Weights,
                     llvm::LLVMContext &Ctx);

/// checkClangInstrumentation - verify if llvm.expect matches PGO profile
/// This function checks the frontend instrumentation in the backend when
/// lowering llvm.expect intrinsics. It checks for existing metadata, and
/// then validates the use of llvm.expect against the assigned branch weights.
//
/// \param I the Instruction being checked
void checkFrontendInstrumentation(Instruction &I);

} // namespace misexpect
} // namespace llvm
