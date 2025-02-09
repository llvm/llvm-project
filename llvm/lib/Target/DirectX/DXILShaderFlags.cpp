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
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::dxil;

/// Update the shader flags mask based on the given instruction.
/// \param CSF Shader flags mask to update.
/// \param I Instruction to check.
void ModuleShaderFlags::updateFunctionFlags(ComputedShaderFlags &CSF,
                                            const Instruction &I,
                                            DXILResourceTypeMap &DRTM) {
  if (!CSF.Doubles)
    CSF.Doubles = I.getType()->isDoubleTy();

  if (!CSF.Doubles) {
    for (const Value *Op : I.operands()) {
      if (Op->getType()->isDoubleTy()) {
        CSF.Doubles = true;
        break;
      }
    }
  }

  if (CSF.Doubles) {
    switch (I.getOpcode()) {
    case Instruction::FDiv:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
      CSF.DX11_1_DoubleExtensions = true;
      break;
    }
  }

  if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
    switch (II->getIntrinsicID()) {
    default:
      break;
    case Intrinsic::dx_resource_handlefrombinding:
      switch (DRTM[cast<TargetExtType>(II->getType())].getResourceKind()) {
      case dxil::ResourceKind::StructuredBuffer:
      case dxil::ResourceKind::RawBuffer:
        CSF.EnableRawAndStructuredBuffers = true;
        break;
      default:
        break;
      }
      break;
    case Intrinsic::dx_resource_load_typedbuffer: {
      dxil::ResourceTypeInfo &RTI =
          DRTM[cast<TargetExtType>(II->getArgOperand(0)->getType())];
      if (RTI.isTyped())
        CSF.TypedUAVLoadAdditionalFormats |= RTI.getTyped().ElementCount > 1;
      break;
    }
    }
  }
  // Handle call instructions
  if (auto *CI = dyn_cast<CallInst>(&I)) {
    const Function *CF = CI->getCalledFunction();
    // Merge-in shader flags mask of the called function in the current module
    if (FunctionFlags.contains(CF))
      CSF.merge(FunctionFlags[CF]);

    // TODO: Set DX11_1_DoubleExtensions if I is a call to DXIL intrinsic
    // DXIL::Opcode::Fma https://github.com/llvm/llvm-project/issues/114554
  }
}

/// Construct ModuleShaderFlags for module Module M
void ModuleShaderFlags::initialize(Module &M, DXILResourceTypeMap &DRTM) {
  CallGraph CG(M);

  // Compute Shader Flags Mask for all functions using post-order visit of SCC
  // of the call graph.
  for (scc_iterator<CallGraph *> SCCI = scc_begin(&CG); !SCCI.isAtEnd();
       ++SCCI) {
    const std::vector<CallGraphNode *> &CurSCC = *SCCI;

    // Union of shader masks of all functions in CurSCC
    ComputedShaderFlags SCCSF;
    // List of functions in CurSCC that are neither external nor declarations
    // and hence whose flags are collected
    SmallVector<Function *> CurSCCFuncs;
    for (CallGraphNode *CGN : CurSCC) {
      Function *F = CGN->getFunction();
      if (!F)
        continue;

      if (F->isDeclaration()) {
        assert(!F->getName().starts_with("dx.op.") &&
               "DXIL Shader Flag analysis should not be run post-lowering.");
        continue;
      }

      ComputedShaderFlags CSF;
      for (const auto &BB : *F)
        for (const auto &I : BB)
          updateFunctionFlags(CSF, I, DRTM);
      // Update combined shader flags mask for all functions in this SCC
      SCCSF.merge(CSF);

      CurSCCFuncs.push_back(F);
    }

    // Update combined shader flags mask for all functions of the module
    CombinedSFMask.merge(SCCSF);

    // Shader flags mask of each of the functions in an SCC of the call graph is
    // the union of all functions in the SCC. Update shader flags masks of
    // functions in CurSCC accordingly. This is trivially true if SCC contains
    // one function.
    for (Function *F : CurSCCFuncs)
      // Merge SCCSF with that of F
      FunctionFlags[F].merge(SCCSF);
  }
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
  auto Iter = FunctionFlags.find(Func);
  assert((Iter != FunctionFlags.end() && Iter->first == Func) &&
         "Get Shader Flags : No Shader Flags Mask exists for function");
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
    const ComputedShaderFlags &SFMask = FlagsInfo.getFunctionFlags(&F);
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
