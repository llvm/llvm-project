//===- DXILShaderFlags.cpp - DXIL Shader Flags helper objects -------------===//
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

#include "DXILShaderFlags.h"
#include "DirectX.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::dxil;

void ModuleShaderFlags::updateFunctionFlags(ComputedShaderFlags &CSF,
                                            const Instruction &I) {
  if (!CSF.Doubles)
    CSF.Doubles = I.getType()->isDoubleTy();

  if (!CSF.Doubles) {
    for (Value *Op : I.operands())
      CSF.Doubles |= Op->getType()->isDoubleTy();
  }
  if (CSF.Doubles) {
    switch (I.getOpcode()) {
    case Instruction::FDiv:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
      // TODO: To be set if I is a call to DXIL intrinsic DXIL::Opcode::Fma
      // https://github.com/llvm/llvm-project/issues/114554
      CSF.DX11_1_DoubleExtensions = true;
      break;
    }
  }
}

void ModuleShaderFlags::initialize(const Module &M) {
  // Collect shader flags for each of the functions
  for (const auto &F : M.getFunctionList()) {
    if (F.isDeclaration())
      continue;
    ComputedShaderFlags CSF;
    for (const auto &BB : F)
      for (const auto &I : BB)
        updateFunctionFlags(CSF, I);
    // Insert shader flag mask for function F
    FunctionFlags.push_back({&F, CSF});
    // Update combined shader flags mask
    CombinedSFMask.merge(CSF);
  }
  llvm::sort(FunctionFlags);
}

void ComputedShaderFlags::print(raw_ostream &OS) const {
  uint64_t FlagVal = (uint64_t) * this;
  OS << formatv("; Shader Flags Value: {0:x8}\n;\n", FlagVal);
  if (FlagVal == 0)
    return;
  OS << "; Note: shader requires additional functionality:\n";
#define SHADER_FEATURE_FLAG(FeatureBit, DxilModuleNum, FlagName, Str)          \
  if (FlagName)                                                                \
    (OS << ";").indent(7) << Str << "\n";
#include "llvm/BinaryFormat/DXContainerConstants.def"
  OS << "; Note: extra DXIL module flags:\n";
#define DXIL_MODULE_FLAG(DxilModuleBit, FlagName, Str)                         \
  if (FlagName)                                                                \
    (OS << ";").indent(7) << Str << "\n";
#include "llvm/BinaryFormat/DXContainerConstants.def"
  OS << ";\n";
}

/// Return the shader flags mask of the specified function Func.
const ComputedShaderFlags &
ModuleShaderFlags::getShaderFlagsMask(const Function *Func) const {
  std::pair<Function const *, ComputedShaderFlags> V{Func, {}};
  const auto Iter = llvm::lower_bound(FunctionFlags, V);
  assert((Iter != FunctionFlags.end() && Iter->first == Func) &&
         "No Shader Flags Mask exists for function");
  return Iter->second;
}

//===----------------------------------------------------------------------===//
// ShaderFlagsAnalysis and ShaderFlagsAnalysisPrinterPass

// Provide an explicit template instantiation for the static ID.
AnalysisKey ShaderFlagsAnalysis::Key;

ModuleShaderFlags ShaderFlagsAnalysis::run(Module &M,
                                           ModuleAnalysisManager &AM) {
  ModuleShaderFlags MSFI;
  MSFI.initialize(M);
  return MSFI;
}

PreservedAnalyses ShaderFlagsAnalysisPrinter::run(Module &M,
                                                  ModuleAnalysisManager &AM) {
  const ModuleShaderFlags &FlagsInfo = AM.getResult<ShaderFlagsAnalysis>(M);
  for (const auto &F : M.getFunctionList()) {
    if (F.isDeclaration())
      continue;
    OS << "; Shader Flags mask for Function: " << F.getName() << "\n";
    auto SFMask = FlagsInfo.getShaderFlagsMask(&F);
    SFMask.print(OS);
  }

  return PreservedAnalyses::all();
}

//===----------------------------------------------------------------------===//
// ShaderFlagsAnalysis and ShaderFlagsAnalysisPrinterPass

bool ShaderFlagsAnalysisWrapper::runOnModule(Module &M) {
  MSFI.initialize(M);
  return false;
}

char ShaderFlagsAnalysisWrapper::ID = 0;

INITIALIZE_PASS(ShaderFlagsAnalysisWrapper, "dx-shader-flag-analysis",
                "DXIL Shader Flag Analysis", true, true)
