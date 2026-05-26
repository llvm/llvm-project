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

  std::vector<std::pair<const DICompileUnit *, const Metadata *>> CUSubprograms;

  for (const Function &F : M) {
    if (const DISubprogram *SP = F.getSubprogram()) {
      auto *FunctionMD = ConstantAsMetadata::get(const_cast<Function *>(&F));
      Res.MDExtra.insert({SP, FunctionMD});
    }
  }

  for (const DISubprogram *SP : DIF.subprograms()) {
    const DISubprogram *NewSP = SP;

    static constexpr auto SupportedDIFlags =
        static_cast<DISubprogram::DIFlags>(DISubprogram::FlagExportSymbols - 1);
    static constexpr auto SupportedDISPFlags =
        static_cast<DISubprogram::DISPFlags>(DISubprogram::SPFlagPure - 1);
    if (SP->isDistinct() || SP->getFlags() & ~SupportedDIFlags ||
        SP->getSPFlags() & ~SupportedDISPFlags) {
      NewSP = DISubprogram::get(
          M.getContext(), SP->getScope(), SP->getName(), SP->getLinkageName(),
          SP->getFile(), SP->getLine(), SP->getType(), SP->getScopeLine(),
          SP->getContainingType(), SP->getVirtualIndex(),
          SP->getThisAdjustment(), SP->getFlags() & SupportedDIFlags,
          SP->getSPFlags() & SupportedDISPFlags, SP->getUnit(),
          SP->getTemplateParams(), SP->getDeclaration(), SP->getRetainedNodes(),
          SP->getThrownTypes(), SP->getAnnotations(), SP->getTargetFuncName(),
          SP->getKeyInstructionsEnabled());

      Res.MDReplace.insert({SP, NewSP});

      if (auto It = Res.MDExtra.find(SP); It != Res.MDExtra.end()) {
        const Metadata *FunctionMD = It->second;
        Res.MDExtra.erase(It);
        Res.MDExtra.insert({NewSP, FunctionMD});
      }
    }

    if (SP->getUnit())
      CUSubprograms.push_back(
          {SP->getUnit(), static_cast<const Metadata *>(SP)});
  }

  std::stable_sort(
      CUSubprograms.begin(), CUSubprograms.end(), [](auto &&A, auto &&B) {
        return std::less<const DICompileUnit *>()(A.first, B.first);
      });
  for (auto It = CUSubprograms.begin(), End = CUSubprograms.end(); It != End;) {
    const DICompileUnit *CU = It->first;
    const DICompileUnit *NewCU =
        cast<DICompileUnit>(Res.MDReplace.lookup_or(CU, CU));
    SmallVector<Metadata *, 16> Subprograms;
    do {
      Subprograms.push_back(const_cast<Metadata *>(It->second));
    } while (++It != End && It->first == CU);
    const auto *SubprogramsMD = MDTuple::get(M.getContext(), Subprograms);
    Res.MDExtra.insert({NewCU, SubprogramsMD});
  }

  for (DIType *T : DIF.types()) {
    if (auto *SR = dyn_cast<DISubrangeType>(T)) {
      DIType *BT = SR->getBaseType();
      if (!BT)
        BT = DIBasicType::get(
            SR->getContext(), dwarf::DW_TAG_base_type, SR->getName(),
            SR->getSizeInBits(), SR->getAlignInBits(), dwarf::DW_ATE_unsigned,
            SR->getNumExtraInhabitants(), /*DataSizeInBits=*/0, SR->getFlags());
      Res.MDReplace.insert({T, BT});
    }
  }

  return Res;
}
