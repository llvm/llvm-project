//===- CallPromotionUtils.h - Utilities for call promotion ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares utilities useful for promoting indirect call sites to
// direct call sites.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_CALLPROMOTIONUTILS_H
#define LLVM_TRANSFORMS_UTILS_CALLPROMOTIONUTILS_H

#include "llvm/IR/CallSite.h"

namespace llvm {

/// Return true if the given indirect call site can be made to call \p Callee.
///
/// This function ensures that the number and type of the call site's arguments
/// and return value match those of the given function. If the types do not
/// match exactly, they must at least be bitcast compatible. If \p FailureReason
/// is non-null and the indirect call cannot be promoted, the failure reason
/// will be stored in it.
bool isLegalToPromote(CallSite CS, Function *Callee,
                      const char **FailureReason = nullptr);

/// Promote the given indirect call site to conditionally call \p Callee.
///
/// This function creates an if-then-else structure at the location of the call
/// site. The original call site is promoted and moved into the "then" block. A
/// clone of the indirect call site is placed in the "else" block and returned.
/// If \p BranchWeights is non-null, it will be used to set !prof metadata on
/// the new conditional branch.
Instruction *promoteCallWithIfThenElse(CallSite CS, Function *Callee,
                                       MDNode *BranchWeights = nullptr);

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_CALLPROMOTIONUTILS_H
