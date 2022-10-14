//===- DXILShaderFlags.h - DXIL Shader Flags helper objects ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains helper objects and APIs for working with DXIL
///       Shader Flags.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_DIRECTX_DXILSHADERFLAGS_H
#define LLVM_TARGET_DIRECTX_DXILSHADERFLAGS_H

#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

namespace llvm {
class Module;
class GlobalVariable;

namespace dxil {

struct ComputedShaderFlags {
#define SHADER_FLAG(bit, FlagName, Str) bool FlagName : 1;
#include "llvm/BinaryFormat/DXContainerConstants.def"

#define SHADER_FLAG(bit, FlagName, Str) FlagName = false;
  ComputedShaderFlags() {
#include "llvm/BinaryFormat/DXContainerConstants.def"
  }

  operator uint64_t() const {
    uint64_t FlagValue = 0;
#define SHADER_FLAG(bit, FlagName, Str)                                        \
  FlagValue |=                                                                 \
      FlagName ? static_cast<uint64_t>(dxbc::FeatureFlags::FlagName) : 0ull;
#include "llvm/BinaryFormat/DXContainerConstants.def"
    return FlagValue;
  }

  static ComputedShaderFlags computeFlags(Module &M);
  void print(raw_ostream &OS = dbgs()) const;
  LLVM_DUMP_METHOD void dump() const { print(); }
};

class ShaderFlagsAnalysis : public AnalysisInfoMixin<ShaderFlagsAnalysis> {
  friend AnalysisInfoMixin<ShaderFlagsAnalysis>;
  static AnalysisKey Key;

public:
  ShaderFlagsAnalysis() = default;

  using Result = ComputedShaderFlags;

  ComputedShaderFlags run(Module &M, ModuleAnalysisManager &AM);
};

/// Printer pass for ShaderFlagsAnalysis results.
class ShaderFlagsAnalysisPrinter
    : public PassInfoMixin<ShaderFlagsAnalysisPrinter> {
  raw_ostream &OS;

public:
  explicit ShaderFlagsAnalysisPrinter(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace dxil
} // namespace llvm

#endif // LLVM_TARGET_DIRECTX_DXILSHADERFLAGS_H
