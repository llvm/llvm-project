//===- AssignSectionsToGlobals.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// For each global which doesn't have a section, set it's section to the
// default as computed by the backend.  For LTO with a linker script, this is
// used by the linker before LTO runs to compute the output sections for each
// global.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/AssignSectionsToGlobals.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#define DEBUG_TYPE "assign-sections-globals"

using namespace llvm;

namespace {

class AssignSections {
public:
  bool run(Module &M, TargetMachine *TM);

private:
  TargetMachine *TM;
  Module *M;

  void collectGlobalObjectSections(
      SmallVectorImpl<std::pair<StringRef, StringRef>> &GOSectionPairs);
  void collectAsmAliasSections(
      const Triple &TT, const Target &T, MCContext &MCCtx,
      SmallVectorImpl<std::pair<StringRef, StringRef>> &GOSectionPairs);
  StringRef getDefaultSectionNameForGlobal(const GlobalObject &GO);
};

} // namespace

StringRef
AssignSections::getDefaultSectionNameForGlobal(const GlobalObject &GO) {
  TargetLoweringObjectFile &TLOF = *TM->getObjFileLowering();

  SectionKind GVKind = TLOF.getKindForGlobal(&GO, *TM);

  // Section selection might depend on module flags like SmallDataLimit.
  TLOF.getModuleMetadata(*M);

  auto Section = static_cast<const MCSectionELF *>(
      TLOF.SectionForGlobal(&GO, GVKind, *TM));

  if (Section)
    return Section->getName();
  else
    return "";
}

bool AssignSections::run(Module &Mod, TargetMachine *TMIn) {
  M = &Mod;
  TM = TMIn;

  if (!TM)
    return false;

  if (Mod.alias_empty() && Mod.global_empty() && Mod.empty())
    return false;

  // We currently only support ELF for LTO with linker scripts.
  const Triple TT(Mod.getTargetTriple());
  if (!TT.isOSBinFormatELF())
    return false;

  // Initialize the target so we can query sections.
  const Target &T = TM->getTarget();
  assert(T.hasMCAsmParser());
  std::unique_ptr<MCRegisterInfo> MRI(T.createMCRegInfo(TT));
  if (!MRI)
    return false;
  llvm::MCTargetOptions MCOptions;
  std::unique_ptr<MCAsmInfo> MAI(T.createMCAsmInfo(*MRI, TT, MCOptions));
  if (!MAI)
    return false;
  std::unique_ptr<llvm::MCSubtargetInfo> STI(
      T.createMCSubtargetInfo(TT, "", ""));
  MCObjectFileInfo MOFI;
  MCContext MCCtx(Triple(TT), *MAI, *MRI, *STI);
  MCCtx.setObjectFileInfo(&MOFI);
  MOFI.initMCObjectFileInfo(MCCtx, /*PIC*/ false);
  TM->getObjFileLowering()->Initialize(MCCtx, *TM);

  // Iterate over objects, and assign sections
  for (GlobalObject &GO : M->global_objects()) {
    if (GO.isDeclarationForLinker() || GO.hasSection() ||
        GO.hasCommonLinkage() || GO.getName().starts_with("llvm."))
      continue;

    GO.setSection(getDefaultSectionNameForGlobal(GO));
  }

  return true;
}

PreservedAnalyses AssignSectionsToGlobalsPass::run(Module &M,
                                                   ModuleAnalysisManager &MAM) {
  AssignSections O;

  if (O.run(M, TM))
    return PreservedAnalyses::none();

  return PreservedAnalyses::all();
}
