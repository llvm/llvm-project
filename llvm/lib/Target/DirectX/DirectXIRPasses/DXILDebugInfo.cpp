//===--- DXILDebugInfo.cpp - analysis&lowering for Debug info -*- C++ -*- ---=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXILDebugInfo.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Module.h"

#define DEBUG_TYPE "dx-debug-info"

using namespace llvm;
using namespace llvm::dxil;

DXILDebugInfoMap DXILDebugInfoPass::run(Module &M) {
  DXILDebugInfoMap Res;
  DebugInfoFinder DIF;
  DIF.processModule(M);

  for (DICompileUnit *CU : DIF.compile_units()) {
    DISourceLanguageName Lang = CU->getSourceLanguage();
    if (Lang.hasVersionedName()) {
      auto LangName = static_cast<dwarf::SourceLanguageName>(Lang.getName());
      Lang = dwarf::toDW_LANG(LangName, Lang.getVersion())
                 .value_or(dwarf::SourceLanguage{});
      auto *NewCU = DICompileUnit::getDistinct(
          M.getContext(), Lang, CU->getFile(), CU->getProducer(),
          CU->isOptimized(), CU->getFlags(), CU->getRuntimeVersion(),
          CU->getSplitDebugFilename(), CU->getEmissionKind(),
          CU->getEnumTypes(), CU->getRetainedTypes(), CU->getGlobalVariables(),
          CU->getImportedEntities(), CU->getMacros(), CU->getDWOId(),
          CU->getSplitDebugInlining(), CU->getDebugInfoForProfiling(),
          CU->getNameTableKind(), CU->getRangesBaseAddress(), CU->getSysRoot(),
          CU->getSDK());
      Res.MDReplace.insert({CU, NewCU});
    }
  }

  return Res;
}
