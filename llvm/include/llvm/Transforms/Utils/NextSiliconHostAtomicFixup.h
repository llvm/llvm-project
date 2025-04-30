//===- NextSiliconHostAtomicFixup.h - IR Manipulation For NextSilicon Pass
//-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// NextSiliconHostAtomicFixup is an LLVM pass that bumps the pointers of atomic
// operations in order detect cases where accessing memory that is
// migrated on device.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_NEXTSILICONATOMICFIXUP_H
#define LLVM_TRANSFORMS_UTILS_NEXTSILICONATOMICFIXUP_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Module;
class ModulePass;

class NextSiliconHostAtomicFixupPass
    : public PassInfoMixin<NextSiliconHostAtomicFixupPass> {
public:
  NextSiliconHostAtomicFixupPass() {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_NEXTSILICONATOMICFIXUP_H
