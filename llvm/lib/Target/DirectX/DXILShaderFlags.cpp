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
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::dxil;

namespace {
/// A simple Wrapper DiagnosticInfo that generates Module-level diagnostic
/// for ShaderFlagsAnalysis pass
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
  Type *Ty = I.getType();
  bool DoubleTyInUse = Ty->isDoubleTy();
  for (Value *Op : I.operands()) {
    DoubleTyInUse |= Op->getType()->isDoubleTy();
  }

  if (DoubleTyInUse) {
    CSF.Doubles = true;
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

static bool compareFunctions(Function const *F1, Function const *F2) {
  return (F1->getName().compare(F2->getName()) < 0);
}

static bool compareFuncSFPairs(const FuncShaderFlagsMask &First,
                               const FuncShaderFlagsMask &Second) {
  return compareFunctions(First.first, Second.first);
}

static DXILModuleShaderFlagsInfo computeFlags(Module &M) {
  DXILModuleShaderFlagsInfo MSFI;
  // Create a sorted list of functions in the module
  SmallVector<Function const *> FuncList;
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;
    FuncList.push_back(&F);
  }
  llvm::sort(FuncList, compareFunctions);

  MSFI.FuncShaderFlagsVec.clear();

  // Collect shader flags for each of the functions
  for (auto F : FuncList) {
    ComputedShaderFlags CSF{};
    for (const auto &BB : *F)
      for (const auto &I : BB)
        updateFlags(CSF, I);
    // Insert shader flag mask for function F
    MSFI.FuncShaderFlagsVec.push_back({F, CSF});
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

void DXILModuleShaderFlagsInfo::print(raw_ostream &OS) const {
  OS << "; Shader Flags mask for Module:\n";
  ModuleFlags.print(OS);
  for (auto SF : FuncShaderFlagsVec) {
    OS << "; Shader Flags mask for Function: " << SF.first->getName() << "\n";
    SF.second.print(OS);
  }
}

const ComputedShaderFlags
DXILModuleShaderFlagsInfo::getShaderFlagsMask(const Function *Func) const {
  FuncShaderFlagsMask V{Func, {}};
  auto Iter = llvm::lower_bound(FuncShaderFlagsVec, V, compareFuncSFPairs);
  if (Iter == FuncShaderFlagsVec.end()) {
    Func->getContext().diagnose(DiagnosticInfoShaderFlags(
        *(Func->getParent()), "Shader Flags information of Function '" +
                                  Twine(Func->getName()) + "' not found"));
  }
  return Iter->second;
}

bool DXILModuleShaderFlagsInfo::hasShaderFlagsMask(const Function *Func) const {
  FuncShaderFlagsMask V{Func, {}};
  return llvm::binary_search(FuncShaderFlagsVec, V);
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
  DXILModuleShaderFlagsInfo Flags = AM.getResult<ShaderFlagsAnalysis>(M);
  Flags.print(OS);
  return PreservedAnalyses::all();
}

char ShaderFlagsAnalysisWrapper::ID = 0;

INITIALIZE_PASS(ShaderFlagsAnalysisWrapper, "dx-shader-flag-analysis",
                "DXIL Shader Flag Analysis", true, true)
