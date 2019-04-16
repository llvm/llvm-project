//===- llvm/Transforms/Utils/SizeOpts.h - size optimization -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains some shared code size optimization related code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_SIZEOPTS_H
#define LLVM_TRANSFORMS_UTILS_SIZEOPTS_H

namespace llvm {

class BasicBlock;
class BlockFrequencyInfo;
class Function;
class ProfileSummaryInfo;

/// Returns true if function \p F is suggested to be size-optimized base on the
/// profile.
bool shouldOptimizeForSize(Function *F, ProfileSummaryInfo *PSI,
                           BlockFrequencyInfo *BFI);
/// Returns true if basic block \p BB is suggested to be size-optimized base
/// on the profile.
bool shouldOptimizeForSize(BasicBlock *BB, ProfileSummaryInfo *PSI,
                           BlockFrequencyInfo *BFI);

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_SIZEOPTS_H
