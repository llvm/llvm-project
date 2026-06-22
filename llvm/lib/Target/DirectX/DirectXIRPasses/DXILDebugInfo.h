//===----- DebugInfo.h - analysis and lowering for Debug info -*- C++ -*- -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file Analyze and downgrade debug info metadata to match DXIL (LLVM 3.7).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DIRECTX_DXILDEBUGINFO_H
#define LLVM_LIB_TARGET_DIRECTX_DXILDEBUGINFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/Casting.h"
#include <memory>

namespace llvm {

namespace dxil {

class DXILDebugInfoMap {
  struct InstructionDeleter {
    void operator()(Instruction *I) { I->deleteValue(); }
  };

public:
  using FuncMap = DenseMap<const Function *, std::unique_ptr<Function>>;
  using InstMap = DenseMap<const Instruction *,
                           std::unique_ptr<Instruction, InstructionDeleter>>;
  using MDMap = DenseMap<const Metadata *, const Metadata *>;

  DXILDebugInfoMap() = default;
  DXILDebugInfoMap(const DXILDebugInfoMap &) = delete;
  DXILDebugInfoMap(DXILDebugInfoMap &&) = default;

  /// Completely replace one function with another in ValueEnumerator.
  FuncMap FuncReplace;

  /// Completely replace one instruction with another in ValueEnumerator.
  InstMap InstReplace;

  /// Enumerate extra metadata when Key is encountered in ValueEnumerator.
  MDMap MDExtra;

  /// Completely replace one metadata with another in ValueEnumerator.
  MDMap MDReplace;

  const Function &getDXILFunction(const Function &F) const {
    auto It = FuncReplace.find(&F);
    if (It != FuncReplace.end())
      return *It->second.get();
    return F;
  }

  const Instruction &getDXILInstruction(const Instruction &I) const {
    auto It = InstReplace.find(&I);
    if (It != InstReplace.end())
      return *It->second.get();
    return I;
  }

  const Metadata *getDXILMetadata(const Metadata *M) const {
    return MDReplace.lookup_or(M, M);
  }
};

namespace DXILDebugInfoPass {

DXILDebugInfoMap run(Module &M);

} // namespace DXILDebugInfoPass
} // namespace dxil
} // namespace llvm

#endif // LLVM_LIB_TARGET_DIRECTX_DXILDEBUGINFO_H
