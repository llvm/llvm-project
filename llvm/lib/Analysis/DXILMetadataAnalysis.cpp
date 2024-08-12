//=- DXILMetadataAnalysis.cpp - Representation of Module metadata -*- C++ -*=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DXILMetadataAnalysis.h"
#include "llvm/ADT/APInt.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "dxil-metadata-analysis"

using namespace llvm;
using namespace dxil;

static ModuleMetadataInfo collectMetadataInfo(Module &M) {
  ModuleMetadataInfo MMDAI;
  Triple TT(Triple(M.getTargetTriple()));
  MMDAI.DXILVersion = TT.getDXILVersion();
  MMDAI.ShaderModelVersion = TT.getOSVersion();
  MMDAI.ShaderStage = TT.getEnvironment();
  return MMDAI;
}

void ModuleMetadataInfo::print(raw_ostream &OS) const {
  OS << "Shader Model Version : " << ShaderModelVersion.getAsString() << "\n";
  OS << "DXIL Version : " << DXILVersion.getAsString() << "\n";
  OS << "Shader Stage : " << Triple::getEnvironmentTypeName(ShaderStage)
     << "\n";
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
