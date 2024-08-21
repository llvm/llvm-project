//=- DXILMetadataAnalysis.cpp - Representation of Module metadata -*- C++ -*=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DXILMetadataAnalysis.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/ErrorHandling.h"
#include <memory>
#include <utility>

#define DEBUG_TYPE "dxil-metadata-analysis"

using namespace llvm;
using namespace dxil;

static ModuleMetadataInfo collectMetadataInfo(Module &M) {
  ModuleMetadataInfo MMDAI;
  Triple TT(Triple(M.getTargetTriple()));
  MMDAI.DXILVersion = TT.getDXILVersion();
  MMDAI.ShaderModelVersion = TT.getOSVersion();
  MMDAI.ShaderStage = TT.getEnvironment();
  NamedMDNode *ValidatorVerNode = M.getNamedMetadata("dx.valver");
  if (ValidatorVerNode) {
    auto *ValVerMD = cast<MDNode>(ValidatorVerNode->getOperand(0));
    auto *MajorMD = mdconst::extract<ConstantInt>(ValVerMD->getOperand(0));
    auto *MinorMD = mdconst::extract<ConstantInt>(ValVerMD->getOperand(1));
    MMDAI.ValidatorVersion =
        VersionTuple(MajorMD->getZExtValue(), MinorMD->getZExtValue());
  }

  if (MMDAI.ShaderStage == Triple::EnvironmentType::Compute) {
    // For all HLSL Shader functions
    for (auto &F : M.functions()) {
      if (!F.hasFnAttribute("hlsl.shader"))
        continue;

      // Get numthreads attribute value
      StringRef NumThreadsStr =
          F.getFnAttribute("hlsl.numthreads").getValueAsString();
      SmallVector<StringRef> NumThreadsVec;
      NumThreadsStr.split(NumThreadsVec, ',');
      if (NumThreadsVec.size() != 3) {
        report_fatal_error(Twine(F.getName()) +
                               ": Invalid numthreads specified",
                           /* gen_crash_diag */ false);
      }
      FunctionProperties EFP;
      auto Zip =
          llvm::zip(NumThreadsVec, MutableArrayRef<unsigned>(EFP.NumThreads));
      for (auto It : Zip) {
        StringRef Str = std::get<0>(It);
        APInt V;
        assert(!Str.getAsInteger(10, V) &&
               "Failed to parse numthreads components as integer values");
        unsigned &Num = std::get<1>(It);
        Num = V.getLimitedValue();
      }
      MMDAI.FunctionPropertyMap.emplace(std::make_pair(std::addressof(F), EFP));
    }
  }
  return MMDAI;
}

void ModuleMetadataInfo::print(raw_ostream &OS) const {
  OS << "Shader Model Version : " << ShaderModelVersion.getAsString() << "\n";
  OS << "DXIL Version : " << DXILVersion.getAsString() << "\n";
  OS << "Shader Stage : " << Triple::getEnvironmentTypeName(ShaderStage)
     << "\n";
  OS << "Validator Version : " << ValidatorVersion.getAsString() << "\n";
  for (auto MapItem : FunctionPropertyMap) {
    MapItem.first->getReturnType()->print(OS, false, true);
    OS << " " << MapItem.first->getName() << "(";
    FunctionType *FT = MapItem.first->getFunctionType();
    for (unsigned I = 0, Sz = FT->getNumParams(); I < Sz; ++I) {
      if (I)
        OS << ",";
      FT->getParamType(I)->print(OS);
    }
    OS << ")\n";
    OS << "  NumThreads: " << MapItem.second.NumThreads[0] << ","
       << MapItem.second.NumThreads[1] << "," << MapItem.second.NumThreads[2]
       << "\n";
  }
}

//===----------------------------------------------------------------------===//
// DXILMetadataAnalysis and DXILMetadataAnalysisPrinterPass

// Provide an explicit template instantiation for the static ID.
AnalysisKey DXILMetadataAnalysis::Key;

llvm::dxil::ModuleMetadataInfo
DXILMetadataAnalysis::run(Module &M, ModuleAnalysisManager &AM) {
  return collectMetadataInfo(M);
}

PreservedAnalyses
DXILMetadataAnalysisPrinterPass::run(Module &M, ModuleAnalysisManager &AM) {
  llvm::dxil::ModuleMetadataInfo &Data = AM.getResult<DXILMetadataAnalysis>(M);

  Data.print(OS);
  return PreservedAnalyses::all();
}

//===----------------------------------------------------------------------===//
// DXILMetadataAnalysisWrapperPass

DXILMetadataAnalysisWrapperPass::DXILMetadataAnalysisWrapperPass()
    : ModulePass(ID) {
  initializeDXILMetadataAnalysisWrapperPassPass(
      *PassRegistry::getPassRegistry());
}

DXILMetadataAnalysisWrapperPass::~DXILMetadataAnalysisWrapperPass() = default;

void DXILMetadataAnalysisWrapperPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

bool DXILMetadataAnalysisWrapperPass::runOnModule(Module &M) {
  MetadataInfo.reset(new ModuleMetadataInfo(collectMetadataInfo(M)));
  return false;
}

void DXILMetadataAnalysisWrapperPass::releaseMemory() { MetadataInfo.reset(); }

void DXILMetadataAnalysisWrapperPass::print(raw_ostream &OS,
                                            const Module *) const {
  if (!MetadataInfo) {
    OS << "No module metadata info has been built!\n";
    return;
  }
  MetadataInfo->print(dbgs());
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD
void DXILMetadataAnalysisWrapperPass::dump() const { print(dbgs(), nullptr); }
#endif

INITIALIZE_PASS(DXILMetadataAnalysisWrapperPass, "dxil-metadata-analysis",
                "DXIL Module Metadata analysis", false, true)
char DXILMetadataAnalysisWrapperPass::ID = 0;
