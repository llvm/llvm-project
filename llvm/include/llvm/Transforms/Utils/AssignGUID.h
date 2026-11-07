//===-- AssignGUID.h - Unique identifier assignment pass --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a pass which assigns a a GUID (globally unique identifier)
// to every GlobalValue in the module, according to its current name, linkage,
// and originating file. This way we have a consistent identifier even when
// these inputs to the GUID change (for instance, after externalising a global
// in ThinLTO).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_ASSIGNGUID_H
#define LLVM_TRANSFORMS_UTILS_ASSIGNGUID_H

#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Debug.h"

namespace llvm {

class AssignGUIDPass : public RequiredPassInfoMixin<AssignGUIDPass> {
public:
  AssignGUIDPass() = default;

  LLVM_ABI static void runOnModule(Module &M);

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
    AssignGUIDPass::runOnModule(M);
    return PreservedAnalyses::all();
  }

  // Let GlobalMerge assign a GUID for merged GVs, instead of needing to
  // traverse all the module; or instead of making GlobalValue::assignGUID
  // public.
  LLVM_ABI static void assignGUIDForMergedGV(GlobalVariable &GV);

  /// Record fallback GUIDs for definitions without assigned GUIDs in the
  /// module's GUID map without attaching metadata.
  LLVM_ABI static void recordMissingGUIDs(Module &M);

  /// Transfer the GUID map entry for \p Src to \p Dst, if one exists.
  LLVM_ABI static void transferGUID(const GlobalValue &Src, GlobalValue &Dst);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_ASSIGNGUID_H
