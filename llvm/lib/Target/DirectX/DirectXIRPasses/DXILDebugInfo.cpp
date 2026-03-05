//===--- DXILDebugInfo.cpp - analysis&lowering for Debug info -*- C++ -*- ---=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXILDebugInfo.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Module.h"

#include <map>

#define DEBUG_TYPE "dx-debug-info"

using namespace llvm;
using namespace llvm::dxil;

DXILDebugInfoMap DXILDebugInfoPass::run(Module &M) {
  DXILDebugInfoMap Res;
  DebugInfoFinder DIF;
  DIF.processModule(M);

  std::multimap<const DICompileUnit *, const Metadata *> CUSubprograms;

  for (const Function &F : M) {
    if (const DISubprogram *SP = F.getSubprogram()) {
      auto *FunctionMD = ConstantAsMetadata::get(const_cast<Function *>(&F));
      Res.MDExtra.insert({SP, FunctionMD});
    }
  }

  for (const DISubprogram *SP : DIF.subprograms())
    if (SP->getUnit())
      CUSubprograms.insert({SP->getUnit(), SP});

  for (auto It = CUSubprograms.begin(), End = CUSubprograms.end(); It != End;) {
    auto *CU = It->first;
    auto CUEnd = CUSubprograms.upper_bound(CU);
    SmallVector<Metadata *, 16> Subprograms;
    do
      Subprograms.push_back(const_cast<Metadata *>(It->second));
    while (++It != CUEnd);
    auto *SubprogramMD = MDTuple::get(M.getContext(), Subprograms);
    Res.MDExtra.insert({CU, SubprogramMD});
  }

  return Res;
}
