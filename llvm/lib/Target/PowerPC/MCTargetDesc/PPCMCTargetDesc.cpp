//===-- PPCMCTargetDesc.cpp - PowerPC Target Descriptions -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides PowerPC specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/PPCMCTargetDesc.h"
#include "MCTargetDesc/PPCInstPrinter.h"
#include "MCTargetDesc/PPCMCAsmInfo.h"
#include "PPCELFStreamer.h"
#include "PPCTargetStreamer.h"
#include "TargetInfo/PowerPCTargetInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSectionXCOFF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/MCSymbolXCOFF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define GET_INSTRINFO_MC_DESC
#include "PPCGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "PPCGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "PPCGenRegisterInfo.inc"

PPCTargetStreamer::PPCTargetStreamer(MCStreamer &S) : MCTargetStreamer(S) {}

// Pin the vtable to this file.
PPCTargetStreamer::~PPCTargetStreamer() = default;

static MCInstrInfo *createPPCMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitPPCMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createPPCMCRegisterInfo(const Triple &TT) {
  bool isPPC64 =
      (TT.getArch() == Triple::ppc64 || TT.getArch() == Triple::ppc64le);
  unsigned Flavour = isPPC64 ? 0 : 1;
  unsigned RA = isPPC64 ? PPC::LR8 : PPC::LR;

  MCRegisterInfo *X = new MCRegisterInfo();
  InitPPCMCRegisterInfo(X, RA, Flavour, Flavour);
  return X;
}

static MCSubtargetInfo *createPPCMCSubtargetInfo(const Triple &TT,
                                                 StringRef CPU, StringRef FS) {
  return createPPCMCSubtargetInfoImpl(TT, CPU, /*TuneCPU*/ CPU, FS);
}

static MCAsmInfo *createPPCMCAsmInfo(const MCRegisterInfo &MRI,
                                     const Triple &TheTriple,
                                     const MCTargetOptions &Options) {
  bool isPPC64 = (TheTriple.getArch() == Triple::ppc64 ||
                  TheTriple.getArch() == Triple::ppc64le);

  MCAsmInfo *MAI;
  if (TheTriple.isOSBinFormatXCOFF())
    MAI = new PPCXCOFFMCAsmInfo(isPPC64, TheTriple);
  else
    MAI = new PPCELFMCAsmInfo(isPPC64, TheTriple);

  // Initial state of the frame pointer is R1.
  unsigned Reg = isPPC64 ? PPC::X1 : PPC::R1;
  MCCFIInstruction Inst =
      MCCFIInstruction::cfiDefCfa(nullptr, MRI.getDwarfRegNum(Reg, true), 0);
  MAI->addInitialFrameState(Inst);

  return MAI;
}

static MCStreamer *createPPCMCStreamer(const Triple &T, MCContext &Context,
                                       std::unique_ptr<MCAsmBackend> &&MAB,
                                       std::unique_ptr<MCObjectWriter> &&OW,
                                       std::unique_ptr<MCCodeEmitter> &&Emitter,
                                       bool RelaxAll) {
  return createPPCELFStreamer(Context, std::move(MAB), std::move(OW),
                              std::move(Emitter));
}

namespace {

class PPCTargetAsmStreamer : public PPCTargetStreamer {
  formatted_raw_ostream &OS;

public:
  PPCTargetAsmStreamer(MCStreamer &S, formatted_raw_ostream &OS)
      : PPCTargetStreamer(S), OS(OS) {}

  void emitTCEntry(const MCSymbol &S) override {
    if (const MCSymbolXCOFF *XSym = dyn_cast<MCSymbolXCOFF>(&S)) {
      MCSymbolXCOFF *TCSym =
          cast<MCSectionXCOFF>(Streamer.getCurrentSectionOnly())
              ->getQualNameSymbol();
      OS << "\t.tc " << TCSym->getName() << "," << XSym->getName() << '\n';

      if (TCSym->hasRename())
        Streamer.emitXCOFFRenameDirective(TCSym, TCSym->getSymbolTableName());
      return;
    }

    OS << "\t.tc " << S.getName() << "[TC]," << S.getName() << '\n';
  }

  void emitMachine(StringRef CPU) override {
    OS << "\t.machine " << CPU << '\n';
  }

  void emitAbiVersion(int AbiVersion) override {
    OS << "\t.abiversion " << AbiVersion << '\n';
  }

  void emitLocalEntry(MCSymbolELF *S, const MCExpr *LocalOffset) override {
    const MCAsmInfo *MAI = Streamer.getContext().getAsmInfo();

    OS << "\t.localentry\t";
    S->print(OS, MAI);
    OS << ", ";
    LocalOffset->print(OS, MAI);
    OS << '\n';
  }
};

class PPCTargetELFStreamer : public PPCTargetStreamer {
public:
  PPCTargetELFStreamer(MCStreamer &S) : PPCTargetStreamer(S) {}

  MCELFStreamer &getStreamer() {
    return static_cast<MCELFStreamer &>(Streamer);
  }

  void emitTCEntry(const MCSymbol &S) override {
    // Creates a R_PPC64_TOC relocation
    Streamer.emitValueToAlignment(8);
    Streamer.emitSymbolValue(&S, 8);
  }

  void emitMachine(StringRef CPU) override {
    // FIXME: Is there anything to do in here or does this directive only
    // limit the parser?
  }

  void emitAbiVersion(int AbiVersion) override {
    MCAssembler &MCA = getStreamer().getAssembler();
    unsigned Flags = MCA.getELFHeaderEFlags();
    Flags &= ~ELF::EF_PPC64_ABI;
    Flags |= (AbiVersion & ELF::EF_PPC64_ABI);
    MCA.setELFHeaderEFlags(Flags);
  }

  void emitLocalEntry(MCSymbolELF *S, const MCExpr *LocalOffset) override {
    MCAssembler &MCA = getStreamer().getAssembler();

    // encodePPC64LocalEntryOffset will report an error if it cannot
    // encode LocalOffset.
    unsigned Encoded = encodePPC64LocalEntryOffset(LocalOffset);

    unsigned Other = S->getOther();
    Other &= ~ELF::STO_PPC64_LOCAL_MASK;
    Other |= Encoded;
    S->setOther(Other);

    // For GAS compatibility, unless we already saw a .abiversion directive,
    // set e_flags to indicate ELFv2 ABI.
    unsigned Flags = MCA.getELFHeaderEFlags();
    if ((Flags & ELF::EF_PPC64_ABI) == 0)
      MCA.setELFHeaderEFlags(Flags | 2);
  }

  void emitAssignment(MCSymbol *S, const MCExpr *Value) override {
    auto *Symbol = cast<MCSymbolELF>(S);

    // When encoding an assignment to set symbol A to symbol B, also copy
    // the st_other bits encoding the local entry point offset.
    if (copyLocalEntry(Symbol, Value))
      UpdateOther.insert(Symbol);
    else
      UpdateOther.erase(Symbol);
  }

  void finish() override {
    for (auto *Sym : UpdateOther)
      if (Sym->isVariable())
        copyLocalEntry(Sym, Sym->getVariableValue());

    // Clear the set of symbols that needs to be updated so the streamer can
    // be reused without issues.
    UpdateOther.clear();
  }

private:
  SmallPtrSet<MCSymbolELF *, 32> UpdateOther;

  bool copyLocalEntry(MCSymbolELF *D, const MCExpr *S) {
    auto *Ref = dyn_cast<const MCSymbolRefExpr>(S);
    if (!Ref)
      return false;
    const auto &RhsSym = cast<MCSymbolELF>(Ref->getSymbol());
    unsigned Other = D->getOther();
    Other &= ~ELF::STO_PPC64_LOCAL_MASK;
    Other |= RhsSym.getOther() & ELF::STO_PPC64_LOCAL_MASK;
    D->setOther(Other);
    return true;
  }

  unsigned encodePPC64LocalEntryOffset(const MCExpr *LocalOffset) {
    MCAssembler &MCA = getStreamer().getAssembler();
    int64_t Offset;
    if (!LocalOffset->evaluateAsAbsolute(Offset, MCA))
      MCA.getContext().reportFatalError(
          LocalOffset->getLoc(), ".localentry expression must be absolute.");

    switch (Offset) {
    default:
      MCA.getContext().reportFatalError(
          LocalOffset->getLoc(),
          ".localentry expression is not a valid power of 2.");
    case 0:
      return 0;
    case 1:
      return 1 << ELF::STO_PPC64_LOCAL_BIT;
    case 4:
    case 8:
    case 16:
    case 32:
    case 64:
      return (int)Log2(Offset) << (int)ELF::STO_PPC64_LOCAL_BIT;
    }
  }
};

class PPCTargetMachOStreamer : public PPCTargetStreamer {
public:
  PPCTargetMachOStreamer(MCStreamer &S) : PPCTargetStreamer(S) {}

  void emitTCEntry(const MCSymbol &S) override {
    llvm_unreachable("Unknown pseudo-op: .tc");
  }

  void emitMachine(StringRef CPU) override {
    // FIXME: We should update the CPUType, CPUSubType in the Object file if
    // the new values are different from the defaults.
  }

  void emitAbiVersion(int AbiVersion) override {
    llvm_unreachable("Unknown pseudo-op: .abiversion");
  }

  void emitLocalEntry(MCSymbolELF *S, const MCExpr *LocalOffset) override {
    llvm_unreachable("Unknown pseudo-op: .localentry");
  }
};

class PPCTargetXCOFFStreamer : public PPCTargetStreamer {
public:
  PPCTargetXCOFFStreamer(MCStreamer &S) : PPCTargetStreamer(S) {}

  void emitTCEntry(const MCSymbol &S) override {
    const MCAsmInfo *MAI = Streamer.getContext().getAsmInfo();
    const unsigned PointerSize = MAI->getCodePointerSize();
    Streamer.emitValueToAlignment(PointerSize);
    Streamer.emitSymbolValue(&S, PointerSize);
  }

  void emitMachine(StringRef CPU) override {
    llvm_unreachable("Machine pseudo-ops are invalid for XCOFF.");
  }

  void emitAbiVersion(int AbiVersion) override {
    llvm_unreachable("ABI-version pseudo-ops are invalid for XCOFF.");
  }

  void emitLocalEntry(MCSymbolELF *S, const MCExpr *LocalOffset) override {
    llvm_unreachable("Local-entry pseudo-ops are invalid for XCOFF.");
  }
};

} // end anonymous namespace

static MCTargetStreamer *createAsmTargetStreamer(MCStreamer &S,
                                                 formatted_raw_ostream &OS,
                                                 MCInstPrinter *InstPrint,
                                                 bool isVerboseAsm) {
  return new PPCTargetAsmStreamer(S, OS);
}

static MCTargetStreamer *
createObjectTargetStreamer(MCStreamer &S, const MCSubtargetInfo &STI) {
  const Triple &TT = STI.getTargetTriple();
  if (TT.isOSBinFormatELF())
    return new PPCTargetELFStreamer(S);
  if (TT.isOSBinFormatXCOFF())
    return new PPCTargetXCOFFStreamer(S);
  return new PPCTargetMachOStreamer(S);
}

static MCInstPrinter *createPPCMCInstPrinter(const Triple &T,
                                             unsigned SyntaxVariant,
                                             const MCAsmInfo &MAI,
                                             const MCInstrInfo &MII,
                                             const MCRegisterInfo &MRI) {
  return new PPCInstPrinter(MAI, MII, MRI, T);
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializePowerPCTargetMC() {
  for (Target *T : {&getThePPC32Target(), &getThePPC32LETarget(),
                    &getThePPC64Target(), &getThePPC64LETarget()}) {
    // Register the MC asm info.
    RegisterMCAsmInfoFn C(*T, createPPCMCAsmInfo);

    // Register the MC instruction info.
    TargetRegistry::RegisterMCInstrInfo(*T, createPPCMCInstrInfo);

    // Register the MC register info.
    TargetRegistry::RegisterMCRegInfo(*T, createPPCMCRegisterInfo);

    // Register the MC subtarget info.
    TargetRegistry::RegisterMCSubtargetInfo(*T, createPPCMCSubtargetInfo);

    // Register the MC Code Emitter
    TargetRegistry::RegisterMCCodeEmitter(*T, createPPCMCCodeEmitter);

    // Register the asm backend.
    TargetRegistry::RegisterMCAsmBackend(*T, createPPCAsmBackend);

    // Register the elf streamer.
    TargetRegistry::RegisterELFStreamer(*T, createPPCMCStreamer);

    // Register the object target streamer.
    TargetRegistry::RegisterObjectTargetStreamer(*T,
                                                 createObjectTargetStreamer);

    // Register the asm target streamer.
    TargetRegistry::RegisterAsmTargetStreamer(*T, createAsmTargetStreamer);

    // Register the MCInstPrinter.
    TargetRegistry::RegisterMCInstPrinter(*T, createPPCMCInstPrinter);
  }
}
