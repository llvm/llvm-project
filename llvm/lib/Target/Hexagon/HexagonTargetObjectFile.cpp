//===-- HexagonTargetObjectFile.cpp - Hexagon asm properties --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the HexagonTargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "HexagonTargetObjectFile.h"
#include "HexagonSubtarget.h"
#include "HexagonTargetMachine.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ELF.h"

using namespace llvm;

static cl::opt<int> SmallDataThreshold("hexagon-small-data-threshold",
                                cl::init(8), cl::Hidden,
                cl::desc("The maximum size of an object in the sdata section"));

void HexagonTargetObjectFile::Initialize(MCContext &Ctx,
                                         const TargetMachine &TM) {
  TargetLoweringObjectFileELF::Initialize(Ctx, TM);
  InitializeELF(TM.Options.UseInitArray);

  SmallDataSection = getContext().getELFSection(
      ".sdata", ELF::SHT_PROGBITS, ELF::SHF_WRITE | ELF::SHF_ALLOC);
  SmallBSSSection = getContext().getELFSection(".sbss", ELF::SHT_NOBITS,
                                               ELF::SHF_WRITE | ELF::SHF_ALLOC);
}

// sdata/sbss support taken largely from the MIPS Backend.
static bool IsInSmallSection(uint64_t Size) {
  return Size > 0 && Size <= (uint64_t)SmallDataThreshold;
}

bool HexagonTargetObjectFile::IsSmallDataEnabled () const {
  return SmallDataThreshold > 0;
}

/// IsGlobalInSmallSection - Return true if this global value should be
/// placed into small data/bss section.
bool HexagonTargetObjectFile::IsGlobalInSmallSection(const GlobalValue *GV,
                                                const TargetMachine &TM) const {
  // If the primary definition of this global value is outside the current
  // translation unit or the global value is available for inspection but not
  // emission, then do nothing.
  if (GV->isDeclaration() || GV->hasAvailableExternallyLinkage())
    return false;

  // Otherwise, Check if GV should be in sdata/sbss, when normally it would end
  // up in getKindForGlobal(GV, TM).
  return IsGlobalInSmallSection(GV, TM, getKindForGlobal(GV, TM));
}

/// IsGlobalInSmallSection - Return true if this global value should be
/// placed into small data/bss section.
bool HexagonTargetObjectFile::
IsGlobalInSmallSection(const GlobalValue *GV, const TargetMachine &TM,
                       SectionKind Kind) const {
  // Only global variables, not functions.
  const GlobalVariable *GVA = dyn_cast<GlobalVariable>(GV);
  if (!GVA)
    return false;

  if (Kind.isBSS() || Kind.isDataNoRel() || Kind.isCommon()) {
    Type *Ty = GV->getType()->getElementType();
    return IsInSmallSection(
        GV->getParent()->getDataLayout().getTypeAllocSize(Ty));
  }

  return false;
}

MCSection *
HexagonTargetObjectFile::SelectSectionForGlobal(const GlobalValue *GV,
                                                SectionKind Kind, Mangler &Mang,
                                                const TargetMachine &TM) const {

  // Handle Small Section classification here.
  if (Kind.isBSS() && IsGlobalInSmallSection(GV, TM, Kind))
    return SmallBSSSection;
  if (Kind.isDataNoRel() && IsGlobalInSmallSection(GV, TM, Kind))
    return SmallDataSection;

  // Otherwise, we work the same as ELF.
  return TargetLoweringObjectFileELF::SelectSectionForGlobal(GV, Kind, Mang,TM);
}
