//===- DXILResourceAnalysis.cpp - DXIL Resource analysis-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains Analysis for information about DXIL resources.
///
//===----------------------------------------------------------------------===//

#include "DXILResourceAnalysis.h"
#include "DirectX.h"
#include "llvm/IR/PassManager.h"

using namespace llvm;

#define DEBUG_TYPE "dxil-resource-analysis"

dxil::Resources DXILResourceMDAnalysis::run(Module &M,
                                            ModuleAnalysisManager &AM) {
  dxil::Resources R;
  R.collect(M);
  return R;
}

AnalysisKey DXILResourceMDAnalysis::Key;

PreservedAnalyses DXILResourceMDPrinterPass::run(Module &M,
                                                 ModuleAnalysisManager &AM) {
  dxil::Resources Res = AM.getResult<DXILResourceMDAnalysis>(M);
  Res.print(OS);
  return PreservedAnalyses::all();
}

char DXILResourceMDWrapper::ID = 0;
INITIALIZE_PASS_BEGIN(DXILResourceMDWrapper, DEBUG_TYPE,
                      "DXIL resource Information", true, true)
INITIALIZE_PASS_END(DXILResourceMDWrapper, DEBUG_TYPE,
                    "DXIL resource Information", true, true)

bool DXILResourceMDWrapper::runOnModule(Module &M) {
  Resources.collect(M);
  return false;
}

DXILResourceMDWrapper::DXILResourceMDWrapper() : ModulePass(ID) {}

void DXILResourceMDWrapper::print(raw_ostream &OS, const Module *) const {
  Resources.print(OS);
}
