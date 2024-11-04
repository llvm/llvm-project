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
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::dxil;

namespace {
/// A simple Wrapper DiagnosticInfo that generates Module-level diagnostic
/// for Shader Flags Analysis pass
class DiagnosticInfoShaderFlags : public DiagnosticInfo {
private:
  const Twine &Msg;
  const Module &Mod;

public:
  /// \p M is the module for which the diagnostic is being emitted. \p Msg is
  /// the message to show. Note that this class does not copy this message, so
  /// this reference must be valid for the whole life time of the diagnostic.
  DiagnosticInfoShaderFlags(const Module &M, const Twine &Msg,
                            DiagnosticSeverity Severity = DS_Error)
      : DiagnosticInfo(DK_Unsupported, Severity), Msg(Msg), Mod(M) {}

  void print(DiagnosticPrinter &DP) const override {
    DP << Mod.getName() << ": " << Msg << '\n';
  }
};
} // namespace

static void updateFlags(ComputedShaderFlags &CSF, const Instruction &I) {
  if (!CSF.Doubles) {
    CSF.Doubles = I.getType()->isDoubleTy();
  }
  if (!CSF.Doubles) {
    for (Value *Op : I.operands()) {
      CSF.Doubles |= Op->getType()->isDoubleTy();
    }
  }
  if (CSF.Doubles) {
    switch (I.getOpcode()) {
    case Instruction::FDiv:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
      // TODO: To be set if I is a call to DXIL intrinsic DXIL::Opcode::Fma
      CSF.DX11_1_DoubleExtensions = true;
      break;
    }
  }
}

static DXILModuleShaderFlagsInfo computeFlags(const Module &M) {
  // Construct a sorted list of functions in the module. Walk the sorted list to
  // create a list of <Function, Shader Flags Mask> pairs. This list is thus
  // sorted at construction time and may be looked up using binary search.
  SmallVector<const Function *> FuncList;
  for (const auto &F : M.getFunctionList()) {
    if (F.isDeclaration())
      continue;
    FuncList.push_back(&F);
  }
  llvm::sort(FuncList);

  DXILModuleShaderFlagsInfo MSFI;

  // Collect shader flags for each of the functions
  for (const Function *F : FuncList) {
    ComputedShaderFlags CSF{};
    for (const auto &BB : *F)
      for (const auto &I : BB)
        updateFlags(CSF, I);
    // Insert shader flag mask for function F
    MSFI.insertInorderFunctionFlags(F, CSF);
  }
  return MSFI;
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

/// Insert the pair <Func, FlagMask> into the sorted vector
/// FunctionFlags. The insertion is expected to be in-order and hence
/// is done at the end of the already sorted list.
void DXILModuleShaderFlagsInfo::insertInorderFunctionFlags(
    const Function *Func, ComputedShaderFlags FlagMask) {
  FunctionFlags.push_back({Func, FlagMask});
}

const SmallVector<std::pair<Function const *, ComputedShaderFlags>> &
DXILModuleShaderFlagsInfo::getFunctionFlags() const {
  return FunctionFlags;
}

const ComputedShaderFlags &DXILModuleShaderFlagsInfo::getModuleFlags() const {
  return ModuleFlags;
}

Expected<const ComputedShaderFlags &>
DXILModuleShaderFlagsInfo::getShaderFlagsMask(const Function *Func) const {
  std::pair<Function const *, ComputedShaderFlags> V{Func, {}};
  const auto *Iter = llvm::lower_bound(FunctionFlags, V);
  if (Iter == FunctionFlags.end() || Iter->first != Func) {
    return createStringError("Shader Flags information of Function '" +
                             Func->getName() + "' not found");
  }
  return Iter->second;
}

AnalysisKey ShaderFlagsAnalysis::Key;

DXILModuleShaderFlagsInfo ShaderFlagsAnalysis::run(Module &M,
                                                   ModuleAnalysisManager &AM) {
  return computeFlags(M);
}

bool ShaderFlagsAnalysisWrapper::runOnModule(Module &M) {
  MSFI = computeFlags(M);
  return false;
}

PreservedAnalyses ShaderFlagsAnalysisPrinter::run(Module &M,
                                                  ModuleAnalysisManager &AM) {
  DXILModuleShaderFlagsInfo FlagsInfo = AM.getResult<ShaderFlagsAnalysis>(M);
  OS << "; Shader Flags mask for Module:\n";
  FlagsInfo.getModuleFlags().print(OS);
  for (const auto &F : M.getFunctionList()) {
    if (F.isDeclaration())
      continue;
    OS << "; Shader Flags mask for Function: " << F.getName() << "\n";
    auto SFMask = FlagsInfo.getShaderFlagsMask(&F);
    if (Error E = SFMask.takeError()) {
      M.getContext().diagnose(
          DiagnosticInfoShaderFlags(M, toString(std::move(E))));
    }
    SFMask->print(OS);
  }

  return PreservedAnalyses::all();
}

char ShaderFlagsAnalysisWrapper::ID = 0;

INITIALIZE_PASS(ShaderFlagsAnalysisWrapper, "dx-shader-flag-analysis",
                "DXIL Shader Flag Analysis", true, true)
