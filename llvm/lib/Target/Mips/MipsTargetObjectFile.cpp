//===-- MipsTargetObjectFile.cpp - Mips Object Files ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MipsTargetObjectFile.h"
#include "MipsSubtarget.h"
#include "MipsTargetMachine.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ELF.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

static cl::opt<unsigned>
SSThreshold("mips-ssection-threshold", cl::Hidden,
            cl::desc("Small data and bss section threshold size (default=8)"),
            cl::init(8));

static cl::opt<bool>
LocalSData("mlocal-sdata", cl::Hidden,
           cl::desc("MIPS: Use gp_rel for object-local data."),
           cl::init(true));

static cl::opt<bool>
ExternSData("mextern-sdata", cl::Hidden,
            cl::desc("MIPS: Use gp_rel for data that is not defined by the "
                     "current object."),
            cl::init(true));

void MipsTargetObjectFile::Initialize(MCContext &Ctx, const TargetMachine &TM){
  TargetLoweringObjectFileELF::Initialize(Ctx, TM);
  InitializeELF(TM.Options.UseInitArray);

  SmallDataSection = getContext().getELFSection(
      ".sdata", ELF::SHT_PROGBITS, ELF::SHF_WRITE | ELF::SHF_ALLOC);

  SmallBSSSection = getContext().getELFSection(".sbss", ELF::SHT_NOBITS,
                                               ELF::SHF_WRITE | ELF::SHF_ALLOC);
  this->TM = &static_cast<const MipsTargetMachine &>(TM);
}

// A address must be loaded from a small section if its size is less than the
// small section size threshold. Data in this section must be addressed using
// gp_rel operator.
static bool IsInSmallSection(uint64_t Size) {
  // gcc has traditionally not treated zero-sized objects as small data, so this
  // is effectively part of the ABI.
  return Size > 0 && Size <= SSThreshold;
}

/// Return true if this global address should be placed into small data/bss
/// section.
bool MipsTargetObjectFile::
IsGlobalInSmallSection(const GlobalValue *GV, const TargetMachine &TM) const {
  // We first check the case where global is a declaration, because finding
  // section kind using getKindForGlobal() is only allowed for global
  // definitions.
  if (GV->isDeclaration() || GV->hasAvailableExternallyLinkage())
    return IsGlobalInSmallSectionImpl(GV, TM);

  return IsGlobalInSmallSection(GV, TM, getKindForGlobal(GV, TM));
}

/// Return true if this global address should be placed into small data/bss
/// section.
bool MipsTargetObjectFile::
IsGlobalInSmallSection(const GlobalValue *GV, const TargetMachine &TM,
                       SectionKind Kind) const {
  return (IsGlobalInSmallSectionImpl(GV, TM) &&
          (Kind.isDataRel() || Kind.isBSS() || Kind.isCommon()));
}

/// Return true if this global address should be placed into small data/bss
/// section. This method does all the work, except for checking the section
/// kind.
bool MipsTargetObjectFile::
IsGlobalInSmallSectionImpl(const GlobalValue *GV,
                           const TargetMachine &TM) const {
  const MipsSubtarget &Subtarget =
      *static_cast<const MipsTargetMachine &>(TM).getSubtargetImpl();

  // Return if small section is not available.
  if (!Subtarget.useSmallSection())
    return false;

  // Only global variables, not functions.
  const GlobalVariable *GVA = dyn_cast<GlobalVariable>(GV);
  if (!GVA)
    return false;

  // Enforce -mlocal-sdata.
  if (!LocalSData && GV->hasLocalLinkage())
    return false;

  // Enforce -mextern-sdata.
  if (!ExternSData && ((GV->hasExternalLinkage() && GV->isDeclaration()) ||
                       GV->hasCommonLinkage()))
    return false;

  Type *Ty = GV->getType()->getElementType();
  return IsInSmallSection(
      GV->getParent()->getDataLayout().getTypeAllocSize(Ty));
}

MCSection *
MipsTargetObjectFile::SelectSectionForGlobal(const GlobalValue *GV,
                                             SectionKind Kind, Mangler &Mang,
                                             const TargetMachine &TM) const {
  // TODO: Could also support "weak" symbols as well with ".gnu.linkonce.s.*"
  // sections?

  // Handle Small Section classification here.
  if (Kind.isBSS() && IsGlobalInSmallSection(GV, TM, Kind))
    return SmallBSSSection;
  if (Kind.isDataRel() && IsGlobalInSmallSection(GV, TM, Kind))
    return SmallDataSection;

  // Otherwise, we work the same as ELF.
  return TargetLoweringObjectFileELF::SelectSectionForGlobal(GV, Kind, Mang,TM);
}

/// Return true if this constant should be placed into small data section.
bool MipsTargetObjectFile::IsConstantInSmallSection(
    const DataLayout &DL, const Constant *CN, const TargetMachine &TM) const {
  return (static_cast<const MipsTargetMachine &>(TM)
              .getSubtargetImpl()
              ->useSmallSection() &&
          LocalSData && IsInSmallSection(DL.getTypeAllocSize(CN->getType())));
}

/// Return true if this constant should be placed into small data section.
MCSection *MipsTargetObjectFile::getSectionForConstant(
    const DataLayout &DL, SectionKind Kind, const Constant *C) const {
  if (IsConstantInSmallSection(DL, C, *TM))
    return SmallDataSection;

  // Otherwise, we work the same as ELF.
  return TargetLoweringObjectFileELF::getSectionForConstant(DL, Kind, C);
}
