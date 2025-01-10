//=- DXILMetadataAnalysis.cpp - Representation of Module metadata -*- C++ -*=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DXILMetadataAnalysis.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "dxil-metadata-analysis"

using namespace llvm;
using namespace dxil;

static ModuleMetadataInfo collectMetadataInfo(Module &M) {
  ModuleMetadataInfo MMDAI;
  Triple TT(Triple(M.getTargetTriple()));
  MMDAI.DXILVersion = TT.getDXILVersion();
  MMDAI.ShaderModelVersion = TT.getOSVersion();
  MMDAI.ShaderProfile = TT.getEnvironment();
  NamedMDNode *ValidatorVerNode = M.getNamedMetadata("dx.valver");
  if (ValidatorVerNode) {
    auto *ValVerMD = cast<MDNode>(ValidatorVerNode->getOperand(0));
    auto *MajorMD = mdconst::extract<ConstantInt>(ValVerMD->getOperand(0));
    auto *MinorMD = mdconst::extract<ConstantInt>(ValVerMD->getOperand(1));
    MMDAI.ValidatorVersion =
        VersionTuple(MajorMD->getZExtValue(), MinorMD->getZExtValue());
  }

  // For all HLSL Shader functions
  for (auto &F : M.functions()) {
    if (!F.hasFnAttribute("hlsl.shader"))
      continue;

    EntryProperties EFP(&F);
    // Get "hlsl.shader" attribute
    Attribute EntryAttr = F.getFnAttribute("hlsl.shader");
    assert(EntryAttr.isValid() &&
           "Invalid value specified for HLSL function attribute hlsl.shader");
    StringRef EntryProfile = EntryAttr.getValueAsString();
    Triple T("", "", "", EntryProfile);
    EFP.ShaderStage = T.getEnvironment();
    // Get numthreads attribute value, if one exists
    StringRef NumThreadsStr =
        F.getFnAttribute("hlsl.numthreads").getValueAsString();
    if (!NumThreadsStr.empty()) {
      SmallVector<StringRef> NumThreadsVec;
      NumThreadsStr.split(NumThreadsVec, ',');
      assert(NumThreadsVec.size() == 3 && "Invalid numthreads specified");
      // Read in the three component values of numthreads
      [[maybe_unused]] bool Success =
          llvm::to_integer(NumThreadsVec[0], EFP.NumThreadsX, 10);
      assert(Success && "Failed to parse X component of numthreads");
      Success = llvm::to_integer(NumThreadsVec[1], EFP.NumThreadsY, 10);
      assert(Success && "Failed to parse Y component of numthreads");
      Success = llvm::to_integer(NumThreadsVec[2], EFP.NumThreadsZ, 10);
      assert(Success && "Failed to parse Z component of numthreads");
    }
    MMDAI.EntryPropertyVec.push_back(EFP);
  }
  return MMDAI;
}

void ModuleMetadataInfo::print(raw_ostream &OS) const {
  OS << "Shader Model Version : " << ShaderModelVersion.getAsString() << "\n";
  OS << "DXIL Version : " << DXILVersion.getAsString() << "\n";
  OS << "Target Shader Stage : "
     << Triple::getEnvironmentTypeName(ShaderProfile) << "\n";
  OS << "Validator Version : " << ValidatorVersion.getAsString() << "\n";
  for (const auto &EP : EntryPropertyVec) {
    OS << " " << EP.Entry->getName() << "\n";
    OS << "  Function Shader Stage : "
       << Triple::getEnvironmentTypeName(EP.ShaderStage) << "\n";
    OS << "  NumThreads: " << EP.NumThreadsX << "," << EP.NumThreadsY << ","
       << EP.NumThreadsZ << "\n";
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
