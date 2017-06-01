//===-- X86TargetObjectFile.cpp - X86 Object Info -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86TargetObjectFile.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Operator.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Target/TargetLowering.h"

using namespace llvm;
using namespace dwarf;

const MCExpr *X86_64MachoTargetObjectFile::getTTypeGlobalReference(
    const GlobalValue *GV, unsigned Encoding, const TargetMachine &TM,
    MachineModuleInfo *MMI, MCStreamer &Streamer) const {

  // On Darwin/X86-64, we can reference dwarf symbols with foo@GOTPCREL+4, which
  // is an indirect pc-relative reference.
  if ((Encoding & DW_EH_PE_indirect) && (Encoding & DW_EH_PE_pcrel)) {
    const MCSymbol *Sym = TM.getSymbol(GV);
    const MCExpr *Res =
      MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_GOTPCREL, getContext());
    const MCExpr *Four = MCConstantExpr::create(4, getContext());
    return MCBinaryExpr::createAdd(Res, Four, getContext());
  }

  return TargetLoweringObjectFileMachO::getTTypeGlobalReference(
      GV, Encoding, TM, MMI, Streamer);
}

MCSymbol *X86_64MachoTargetObjectFile::getCFIPersonalitySymbol(
    const GlobalValue *GV, const TargetMachine &TM,
    MachineModuleInfo *MMI) const {
  return TM.getSymbol(GV);
}

const MCExpr *X86_64MachoTargetObjectFile::getIndirectSymViaGOTPCRel(
    const MCSymbol *Sym, const MCValue &MV, int64_t Offset,
    MachineModuleInfo *MMI, MCStreamer &Streamer) const {
  // On Darwin/X86-64, we need to use foo@GOTPCREL+4 to access the got entry
  // from a data section. In case there's an additional offset, then use
  // foo@GOTPCREL+4+<offset>.
  unsigned FinalOff = Offset+MV.getConstant()+4;
  const MCExpr *Res =
    MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_GOTPCREL, getContext());
  const MCExpr *Off = MCConstantExpr::create(FinalOff, getContext());
  return MCBinaryExpr::createAdd(Res, Off, getContext());
}

const MCExpr *X86ELFTargetObjectFile::getDebugThreadLocalSymbol(
    const MCSymbol *Sym) const {
  return MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_DTPOFF, getContext());
}

void
X86FreeBSDTargetObjectFile::Initialize(MCContext &Ctx,
                                       const TargetMachine &TM) {
  TargetLoweringObjectFileELF::Initialize(Ctx, TM);
  InitializeELF(TM.Options.UseInitArray);
}

void
X86FuchsiaTargetObjectFile::Initialize(MCContext &Ctx,
                                       const TargetMachine &TM) {
  TargetLoweringObjectFileELF::Initialize(Ctx, TM);
  InitializeELF(TM.Options.UseInitArray);
}

void
X86LinuxNaClTargetObjectFile::Initialize(MCContext &Ctx,
                                         const TargetMachine &TM) {
  TargetLoweringObjectFileELF::Initialize(Ctx, TM);
  InitializeELF(TM.Options.UseInitArray);
}

const MCExpr *X86WindowsTargetObjectFile::lowerRelativeReference(
    const GlobalValue *LHS, const GlobalValue *RHS,
    const TargetMachine &TM) const {
  // Our symbols should exist in address space zero, cowardly no-op if
  // otherwise.
  if (LHS->getType()->getPointerAddressSpace() != 0 ||
      RHS->getType()->getPointerAddressSpace() != 0)
    return nullptr;

  // Both ptrtoint instructions must wrap global objects:
  // - Only global variables are eligible for image relative relocations.
  // - The subtrahend refers to the special symbol __ImageBase, a GlobalVariable.
  // We expect __ImageBase to be a global variable without a section, externally
  // defined.
  //
  // It should look something like this: @__ImageBase = external constant i8
  if (!isa<GlobalObject>(LHS) || !isa<GlobalVariable>(RHS) ||
      LHS->isThreadLocal() || RHS->isThreadLocal() ||
      RHS->getName() != "__ImageBase" || !RHS->hasExternalLinkage() ||
      cast<GlobalVariable>(RHS)->hasInitializer() || RHS->hasSection())
    return nullptr;

  return MCSymbolRefExpr::create(TM.getSymbol(LHS),
                                 MCSymbolRefExpr::VK_COFF_IMGREL32,
                                 getContext());
}

static std::string APIntToHexString(const APInt &AI) {
  unsigned Width = (AI.getBitWidth() / 8) * 2;
  std::string HexString = utohexstr(AI.getLimitedValue(), /*LowerCase=*/true);
  unsigned Size = HexString.size();
  assert(Width >= Size && "hex string is too large!");
  HexString.insert(HexString.begin(), Width - Size, '0');

  return HexString;
}

static std::string scalarConstantToHexString(const Constant *C) {
  Type *Ty = C->getType();
  if (isa<UndefValue>(C)) {
    return APIntToHexString(APInt::getNullValue(Ty->getPrimitiveSizeInBits()));
  } else if (const auto *CFP = dyn_cast<ConstantFP>(C)) {
    return APIntToHexString(CFP->getValueAPF().bitcastToAPInt());
  } else if (const auto *CI = dyn_cast<ConstantInt>(C)) {
    return APIntToHexString(CI->getValue());
  } else {
    unsigned NumElements;
    if (isa<VectorType>(Ty))
      NumElements = Ty->getVectorNumElements();
    else
      NumElements = Ty->getArrayNumElements();
    std::string HexString;
    for (int I = NumElements - 1, E = -1; I != E; --I)
      HexString += scalarConstantToHexString(C->getAggregateElement(I));
    return HexString;
  }
}

MCSection *X86WindowsTargetObjectFile::getSectionForConstant(
    const DataLayout &DL, SectionKind Kind, const Constant *C,
    unsigned &Align) const {
  if (Kind.isMergeableConst() && C) {
    const unsigned Characteristics = COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                                     COFF::IMAGE_SCN_MEM_READ |
                                     COFF::IMAGE_SCN_LNK_COMDAT;
    std::string COMDATSymName;
    if (Kind.isMergeableConst4()) {
      if (Align <= 4) {
        COMDATSymName = "__real@" + scalarConstantToHexString(C);
        Align = 4;
      }
    } else if (Kind.isMergeableConst8()) {
      if (Align <= 8) {
        COMDATSymName = "__real@" + scalarConstantToHexString(C);
        Align = 8;
      }
    } else if (Kind.isMergeableConst16()) {
      if (Align <= 16) {
        COMDATSymName = "__xmm@" + scalarConstantToHexString(C);
        Align = 16;
      }
    } else if (Kind.isMergeableConst32()) {
      if (Align <= 32) {
        COMDATSymName = "__ymm@" + scalarConstantToHexString(C);
        Align = 32;
      }
    }

    if (!COMDATSymName.empty())
      return getContext().getCOFFSection(".rdata", Characteristics, Kind,
                                         COMDATSymName,
                                         COFF::IMAGE_COMDAT_SELECT_ANY);
  }

  return TargetLoweringObjectFile::getSectionForConstant(DL, Kind, C, Align);
}
