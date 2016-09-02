//===--- Canonicalization.h - Set of canonicalization passes ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_CANONICALIZATION_H
#define POLLY_CANONICALIZATION_H

#include "llvm/IR/LegacyPassManager.h"

namespace polly {

/// Schedule a set of canonicalization passes to prepare for Polly.
///
/// The set of optimization passes was partially taken/copied from the
/// set of default optimization passes in LLVM. It is used to bring the code
/// into a canonical form that simplifies the analysis and optimization passes
/// of Polly. The set of optimization passes scheduled here is probably not yet
/// optimal. TODO: Optimize the set of canonicalization passes.
void registerCanonicalicationPasses(llvm::legacy::PassManagerBase &PM);
} // namespace polly

#endif
