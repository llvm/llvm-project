//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_IPO_IPOOPTIONS_H
#define LLVM_LIB_TRANSFORMS_IPO_IPOOPTIONS_H

#include "llvm/Analysis/ReplayInlineAdvisor.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include <optional>

namespace llvm {
enum class ExpandVariadicsMode;
enum class PassSummaryAction;

/// Mechanism to add runtime checking of devirtualization decisions, optionally
/// trapping or falling back to indirect call on any that are not correct.
/// Trapping mode is useful for debugging undefined behavior leading to failures
/// with WPD. Fallback mode is useful for ensuring safety when whole program
/// visibility may be compromised.
enum class WPDCheckMode { None, Trap, Fallback };
} // namespace llvm

#define OPTIONS_STRUCT_DECL
#include "IPOOptions.inc"

namespace llvm {

inline const IPOOptions &getIPOOptions(const LLVMContext &C) {
  return C.getLibraryOptions().get<IPOOptions>();
}

inline const IPOOptions &getIPOOptions(const Function &F) {
  return getIPOOptions(F.getContext());
}

inline const IPOOptions &getIPOOptions(const Module &M) {
  return getIPOOptions(M.getContext());
}

} // namespace llvm

#endif // LLVM_LIB_TRANSFORMS_IPO_IPOOPTIONS_H
