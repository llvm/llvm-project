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
#include "llvm/Analysis/DXILResource.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::dxil;

static void updateFunctionFlags(ComputedShaderFlags &CSF, const Instruction &I,
                                DXILResourceTypeMap &DRTM) {
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

  if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
    switch (II->getIntrinsicID()) {
    default:
      break;
    case Intrinsic::dx_resource_load_typedbuffer: {
      dxil::ResourceTypeInfo &RTI =
          DRTM[cast<TargetExtType>(II->getArgOperand(0)->getType())];
      if (RTI.isTyped())
        CSF.TypedUAVLoadAdditionalFormats |= RTI.getTyped().ElementCount > 1;
    }
    }
  }
}

void ModuleShaderFlags::initialize(const Module &M, DXILResourceTypeMap &DRTM) {

  // Collect shader flags for each of the functions
  for (const auto &F : M.getFunctionList()) {
    if (F.isDeclaration()) {
      assert(!F.getName().starts_with("dx.op.") &&
             "DXIL Shader Flag analysis should not be run post-lowering.");
      continue;
    }
    ComputedShaderFlags CSF;
    for (const auto &BB : F)
      for (const auto &I : BB)
        updateFunctionFlags(CSF, I, DRTM);
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
ModuleShaderFlags::getFunctionFlags(const Function *Func) const {
  const auto Iter = llvm::lower_bound(
      FunctionFlags, Func,
      [](const std::pair<const Function *, ComputedShaderFlags> FSM,
         const Function *FindFunc) { return (FSM.first < FindFunc); });
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
  DXILResourceTypeMap &DRTM = AM.getResult<DXILResourceTypeAnalysis>(M);

  ModuleShaderFlags MSFI;
  MSFI.initialize(M, DRTM);

  return MSFI;
}

PreservedAnalyses ShaderFlagsAnalysisPrinter::run(Module &M,
                                                  ModuleAnalysisManager &AM) {
  const ModuleShaderFlags &FlagsInfo = AM.getResult<ShaderFlagsAnalysis>(M);
  // Print description of combined shader flags for all module functions
  OS << "; Combined Shader Flags for Module\n";
  FlagsInfo.getCombinedFlags().print(OS);
  // Print shader flags mask for each of the module functions
  OS << "; Shader Flags for Module Functions\n";
  for (const auto &F : M.getFunctionList()) {
    if (F.isDeclaration())
      continue;
    auto SFMask = FlagsInfo.getFunctionFlags(&F);
    OS << formatv("; Function {0} : {1:x8}\n;\n", F.getName(),
                  (uint64_t)(SFMask));
  }

  return PreservedAnalyses::all();
}

//===----------------------------------------------------------------------===//
// ShaderFlagsAnalysis and ShaderFlagsAnalysisPrinterPass

bool ShaderFlagsAnalysisWrapper::runOnModule(Module &M) {
  DXILResourceTypeMap &DRTM =
      getAnalysis<DXILResourceTypeWrapperPass>().getResourceTypeMap();

  MSFI.initialize(M, DRTM);
  return false;
}

void ShaderFlagsAnalysisWrapper::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<DXILResourceTypeWrapperPass>();
}

char ShaderFlagsAnalysisWrapper::ID = 0;

INITIALIZE_PASS_BEGIN(ShaderFlagsAnalysisWrapper, "dx-shader-flag-analysis",
                      "DXIL Shader Flag Analysis", true, true)
INITIALIZE_PASS_DEPENDENCY(DXILResourceTypeWrapperPass)
INITIALIZE_PASS_END(ShaderFlagsAnalysisWrapper, "dx-shader-flag-analysis",
                    "DXIL Shader Flag Analysis", true, true)
