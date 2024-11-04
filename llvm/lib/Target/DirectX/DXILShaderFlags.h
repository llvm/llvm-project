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

#include "llvm/IR/Function.h"
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
#define SHADER_FEATURE_FLAG(FeatureBit, DxilModuleBit, FlagName, Str)          \
  bool FlagName : 1;
#define DXIL_MODULE_FLAG(DxilModuleBit, FlagName, Str) bool FlagName : 1;
#include "llvm/BinaryFormat/DXContainerConstants.def"

#define SHADER_FEATURE_FLAG(FeatureBit, DxilModuleBit, FlagName, Str)          \
  FlagName = false;
#define DXIL_MODULE_FLAG(DxilModuleBit, FlagName, Str) FlagName = false;
  ComputedShaderFlags() {
#include "llvm/BinaryFormat/DXContainerConstants.def"
  }

  constexpr uint64_t getMask(int Bit) const {
    return Bit != -1 ? 1ull << Bit : 0;
  }
  operator uint64_t() const {
    uint64_t FlagValue = 0;
#define SHADER_FEATURE_FLAG(FeatureBit, DxilModuleBit, FlagName, Str)          \
  FlagValue |= FlagName ? getMask(DxilModuleBit) : 0ull;
#define DXIL_MODULE_FLAG(DxilModuleBit, FlagName, Str)                         \
  FlagValue |= FlagName ? getMask(DxilModuleBit) : 0ull;
#include "llvm/BinaryFormat/DXContainerConstants.def"
    return FlagValue;
  }
  uint64_t getFeatureFlags() const {
    uint64_t FeatureFlags = 0;
#define SHADER_FEATURE_FLAG(FeatureBit, DxilModuleBit, FlagName, Str)          \
  FeatureFlags |= FlagName ? getMask(FeatureBit) : 0ull;
#include "llvm/BinaryFormat/DXContainerConstants.def"
    return FeatureFlags;
  }

  uint64_t getModuleFlags() const {
    uint64_t ModuleFlags = 0;
#define DXIL_MODULE_FLAG(DxilModuleBit, FlagName, Str)                         \
  ModuleFlags |= FlagName ? getMask(DxilModuleBit) : 0ull;
#include "llvm/BinaryFormat/DXContainerConstants.def"
    return ModuleFlags;
  }

  void print(raw_ostream &OS = dbgs()) const;
  LLVM_DUMP_METHOD void dump() const { print(); }
};

struct DXILModuleShaderFlagsInfo {
  Expected<const ComputedShaderFlags &>
  getShaderFlagsMask(const Function *Func) const;
  bool hasShaderFlagsMask(const Function *Func) const;
  const ComputedShaderFlags &getModuleFlags() const;
  const SmallVector<std::pair<Function const *, ComputedShaderFlags>> &
  getFunctionFlags() const;
  void insertInorderFunctionFlags(const Function *, ComputedShaderFlags);

private:
  // Shader Flag mask representing module-level properties. These are
  // represented using the macro DXIL_MODULE_FLAG
  ComputedShaderFlags ModuleFlags;
  // Vector of Function-Shader Flag mask pairs representing properties of each
  // of the functions in the module. Shader Flags of each function are those
  // represented using the macro SHADER_FEATURE_FLAG.
  SmallVector<std::pair<Function const *, ComputedShaderFlags>> FunctionFlags;
};

class ShaderFlagsAnalysis : public AnalysisInfoMixin<ShaderFlagsAnalysis> {
  friend AnalysisInfoMixin<ShaderFlagsAnalysis>;
  static AnalysisKey Key;

public:
  ShaderFlagsAnalysis() = default;

  using Result = DXILModuleShaderFlagsInfo;

  DXILModuleShaderFlagsInfo run(Module &M, ModuleAnalysisManager &AM);
};

/// Printer pass for ShaderFlagsAnalysis results.
class ShaderFlagsAnalysisPrinter
    : public PassInfoMixin<ShaderFlagsAnalysisPrinter> {
  raw_ostream &OS;

public:
  explicit ShaderFlagsAnalysisPrinter(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

/// Wrapper pass for the legacy pass manager.
///
/// This is required because the passes that will depend on this are codegen
/// passes which run through the legacy pass manager.
class ShaderFlagsAnalysisWrapper : public ModulePass {
  DXILModuleShaderFlagsInfo MSFI;

public:
  static char ID;

  ShaderFlagsAnalysisWrapper() : ModulePass(ID) {}

  const DXILModuleShaderFlagsInfo &getShaderFlags() { return MSFI; }

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};

} // namespace dxil
} // namespace llvm

#endif // LLVM_TARGET_DIRECTX_DXILSHADERFLAGS_H
