//===--- Canonicalization.h - Set of canonicalization passes ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_CANONICALIZATION_H
#define POLLY_CANONICALIZATION_H

#include "llvm/Passes/PassBuilder.h"

namespace llvm {
namespace legacy {
class PassManagerBase;
}
} // namespace llvm

namespace polly {

/// Schedule a set of canonicalization passes to prepare for Polly.
///
/// The set of optimization passes was partially taken/copied from the
/// set of default optimization passes in LLVM. It is used to bring the code
/// into a canonical form that simplifies the analysis and optimization passes
/// of Polly. The set of optimization passes scheduled here is probably not yet
/// optimal. TODO: Optimize the set of canonicalization passes.
void registerCanonicalicationPasses(llvm::legacy::PassManagerBase &PM);

llvm::FunctionPassManager
buildCanonicalicationPassesForNPM(llvm::ModulePassManager &MPM,
                                  llvm::PassBuilder::OptimizationLevel Level);

} // namespace polly

#endif
