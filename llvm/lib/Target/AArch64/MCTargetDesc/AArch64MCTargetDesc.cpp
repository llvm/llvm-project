//===-- AArch64MCTargetDesc.cpp - AArch64 Target Descriptions ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides AArch64 specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "AArch64MCTargetDesc.h"
#include "AArch64ELFStreamer.h"
#include "AArch64MCAsmInfo.h"
#include "AArch64WinCOFFStreamer.h"
#include "MCTargetDesc/AArch64AddressingModes.h"
#include "MCTargetDesc/AArch64InstPrinter.h"
#include "TargetInfo/AArch64TargetInfo.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define GET_INSTRINFO_MC_DESC
#define GET_INSTRINFO_MC_HELPERS
#include "AArch64GenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "AArch64GenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "AArch64GenRegisterInfo.inc"

static MCInstrInfo *createAArch64MCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitAArch64MCInstrInfo(X);
  return X;
}

static MCSubtargetInfo *
createAArch64MCSubtargetInfo(const Triple &TT, StringRef CPU, StringRef FS) {
  if (CPU.empty())
    CPU = "generic";

  return createAArch64MCSubtargetInfoImpl(TT, CPU, FS);
}

void AArch64_MC::initLLVMToCVRegMapping(MCRegisterInfo *MRI) {
  // Mapping from CodeView to MC register id.
  static const struct {
    codeview::RegisterId CVReg;
    MCPhysReg Reg;
  } RegMap[] = {
      {codeview::RegisterId::ARM64_W0, AArch64::W0},
      {codeview::RegisterId::ARM64_W1, AArch64::W1},
      {codeview::RegisterId::ARM64_W2, AArch64::W2},
      {codeview::RegisterId::ARM64_W3, AArch64::W3},
      {codeview::RegisterId::ARM64_W4, AArch64::W4},
      {codeview::RegisterId::ARM64_W5, AArch64::W5},
      {codeview::RegisterId::ARM64_W6, AArch64::W6},
      {codeview::RegisterId::ARM64_W7, AArch64::W7},
      {codeview::RegisterId::ARM64_W8, AArch64::W8},
      {codeview::RegisterId::ARM64_W9, AArch64::W9},
      {codeview::RegisterId::ARM64_W10, AArch64::W10},
      {codeview::RegisterId::ARM64_W11, AArch64::W11},
      {codeview::RegisterId::ARM64_W12, AArch64::W12},
      {codeview::RegisterId::ARM64_W13, AArch64::W13},
      {codeview::RegisterId::ARM64_W14, AArch64::W14},
      {codeview::RegisterId::ARM64_W15, AArch64::W15},
      {codeview::RegisterId::ARM64_W16, AArch64::W16},
      {codeview::RegisterId::ARM64_W17, AArch64::W17},
      {codeview::RegisterId::ARM64_W18, AArch64::W18},
      {codeview::RegisterId::ARM64_W19, AArch64::W19},
      {codeview::RegisterId::ARM64_W20, AArch64::W20},
      {codeview::RegisterId::ARM64_W21, AArch64::W21},
      {codeview::RegisterId::ARM64_W22, AArch64::W22},
      {codeview::RegisterId::ARM64_W23, AArch64::W23},
      {codeview::RegisterId::ARM64_W24, AArch64::W24},
      {codeview::RegisterId::ARM64_W25, AArch64::W25},
      {codeview::RegisterId::ARM64_W26, AArch64::W26},
      {codeview::RegisterId::ARM64_W27, AArch64::W27},
      {codeview::RegisterId::ARM64_W28, AArch64::W28},
      {codeview::RegisterId::ARM64_W29, AArch64::W29},
      {codeview::RegisterId::ARM64_W30, AArch64::W30},
      {codeview::RegisterId::ARM64_WZR, AArch64::WZR},
      {codeview::RegisterId::ARM64_X0, AArch64::X0},
      {codeview::RegisterId::ARM64_X1, AArch64::X1},
      {codeview::RegisterId::ARM64_X2, AArch64::X2},
      {codeview::RegisterId::ARM64_X3, AArch64::X3},
      {codeview::RegisterId::ARM64_X4, AArch64::X4},
      {codeview::RegisterId::ARM64_X5, AArch64::X5},
      {codeview::RegisterId::ARM64_X6, AArch64::X6},
      {codeview::RegisterId::ARM64_X7, AArch64::X7},
      {codeview::RegisterId::ARM64_X8, AArch64::X8},
      {codeview::RegisterId::ARM64_X9, AArch64::X9},
      {codeview::RegisterId::ARM64_X10, AArch64::X10},
      {codeview::RegisterId::ARM64_X11, AArch64::X11},
      {codeview::RegisterId::ARM64_X12, AArch64::X12},
      {codeview::RegisterId::ARM64_X13, AArch64::X13},
      {codeview::RegisterId::ARM64_X14, AArch64::X14},
      {codeview::RegisterId::ARM64_X15, AArch64::X15},
      {codeview::RegisterId::ARM64_X16, AArch64::X16},
      {codeview::RegisterId::ARM64_X17, AArch64::X17},
      {codeview::RegisterId::ARM64_X18, AArch64::X18},
      {codeview::RegisterId::ARM64_X19, AArch64::X19},
      {codeview::RegisterId::ARM64_X20, AArch64::X20},
      {codeview::RegisterId::ARM64_X21, AArch64::X21},
      {codeview::RegisterId::ARM64_X22, AArch64::X22},
      {codeview::RegisterId::ARM64_X23, AArch64::X23},
      {codeview::RegisterId::ARM64_X24, AArch64::X24},
      {codeview::RegisterId::ARM64_X25, AArch64::X25},
      {codeview::RegisterId::ARM64_X26, AArch64::X26},
      {codeview::RegisterId::ARM64_X27, AArch64::X27},
      {codeview::RegisterId::ARM64_X28, AArch64::X28},
      {codeview::RegisterId::ARM64_FP, AArch64::FP},
      {codeview::RegisterId::ARM64_LR, AArch64::LR},
      {codeview::RegisterId::ARM64_SP, AArch64::SP},
      {codeview::RegisterId::ARM64_ZR, AArch64::XZR},
      {codeview::RegisterId::ARM64_NZCV, AArch64::NZCV},
      {codeview::RegisterId::ARM64_S0, AArch64::S0},
      {codeview::RegisterId::ARM64_S1, AArch64::S1},
      {codeview::RegisterId::ARM64_S2, AArch64::S2},
      {codeview::RegisterId::ARM64_S3, AArch64::S3},
      {codeview::RegisterId::ARM64_S4, AArch64::S4},
      {codeview::RegisterId::ARM64_S5, AArch64::S5},
      {codeview::RegisterId::ARM64_S6, AArch64::S6},
      {codeview::RegisterId::ARM64_S7, AArch64::S7},
      {codeview::RegisterId::ARM64_S8, AArch64::S8},
      {codeview::RegisterId::ARM64_S9, AArch64::S9},
      {codeview::RegisterId::ARM64_S10, AArch64::S10},
      {codeview::RegisterId::ARM64_S11, AArch64::S11},
      {codeview::RegisterId::ARM64_S12, AArch64::S12},
      {codeview::RegisterId::ARM64_S13, AArch64::S13},
      {codeview::RegisterId::ARM64_S14, AArch64::S14},
      {codeview::RegisterId::ARM64_S15, AArch64::S15},
      {codeview::RegisterId::ARM64_S16, AArch64::S16},
      {codeview::RegisterId::ARM64_S17, AArch64::S17},
      {codeview::RegisterId::ARM64_S18, AArch64::S18},
      {codeview::RegisterId::ARM64_S19, AArch64::S19},
      {codeview::RegisterId::ARM64_S20, AArch64::S20},
      {codeview::RegisterId::ARM64_S21, AArch64::S21},
      {codeview::RegisterId::ARM64_S22, AArch64::S22},
      {codeview::RegisterId::ARM64_S23, AArch64::S23},
      {codeview::RegisterId::ARM64_S24, AArch64::S24},
      {codeview::RegisterId::ARM64_S25, AArch64::S25},
      {codeview::RegisterId::ARM64_S26, AArch64::S26},
      {codeview::RegisterId::ARM64_S27, AArch64::S27},
      {codeview::RegisterId::ARM64_S28, AArch64::S28},
      {codeview::RegisterId::ARM64_S29, AArch64::S29},
      {codeview::RegisterId::ARM64_S30, AArch64::S30},
      {codeview::RegisterId::ARM64_S31, AArch64::S31},
      {codeview::RegisterId::ARM64_D0, AArch64::D0},
      {codeview::RegisterId::ARM64_D1, AArch64::D1},
      {codeview::RegisterId::ARM64_D2, AArch64::D2},
      {codeview::RegisterId::ARM64_D3, AArch64::D3},
      {codeview::RegisterId::ARM64_D4, AArch64::D4},
      {codeview::RegisterId::ARM64_D5, AArch64::D5},
      {codeview::RegisterId::ARM64_D6, AArch64::D6},
      {codeview::RegisterId::ARM64_D7, AArch64::D7},
      {codeview::RegisterId::ARM64_D8, AArch64::D8},
      {codeview::RegisterId::ARM64_D9, AArch64::D9},
      {codeview::RegisterId::ARM64_D10, AArch64::D10},
      {codeview::RegisterId::ARM64_D11, AArch64::D11},
      {codeview::RegisterId::ARM64_D12, AArch64::D12},
      {codeview::RegisterId::ARM64_D13, AArch64::D13},
      {codeview::RegisterId::ARM64_D14, AArch64::D14},
      {codeview::RegisterId::ARM64_D15, AArch64::D15},
      {codeview::RegisterId::ARM64_D16, AArch64::D16},
      {codeview::RegisterId::ARM64_D17, AArch64::D17},
      {codeview::RegisterId::ARM64_D18, AArch64::D18},
      {codeview::RegisterId::ARM64_D19, AArch64::D19},
      {codeview::RegisterId::ARM64_D20, AArch64::D20},
      {codeview::RegisterId::ARM64_D21, AArch64::D21},
      {codeview::RegisterId::ARM64_D22, AArch64::D22},
      {codeview::RegisterId::ARM64_D23, AArch64::D23},
      {codeview::RegisterId::ARM64_D24, AArch64::D24},
      {codeview::RegisterId::ARM64_D25, AArch64::D25},
      {codeview::RegisterId::ARM64_D26, AArch64::D26},
      {codeview::RegisterId::ARM64_D27, AArch64::D27},
      {codeview::RegisterId::ARM64_D28, AArch64::D28},
      {codeview::RegisterId::ARM64_D29, AArch64::D29},
      {codeview::RegisterId::ARM64_D30, AArch64::D30},
      {codeview::RegisterId::ARM64_D31, AArch64::D31},
      {codeview::RegisterId::ARM64_Q0, AArch64::Q0},
      {codeview::RegisterId::ARM64_Q1, AArch64::Q1},
      {codeview::RegisterId::ARM64_Q2, AArch64::Q2},
      {codeview::RegisterId::ARM64_Q3, AArch64::Q3},
      {codeview::RegisterId::ARM64_Q4, AArch64::Q4},
      {codeview::RegisterId::ARM64_Q5, AArch64::Q5},
      {codeview::RegisterId::ARM64_Q6, AArch64::Q6},
      {codeview::RegisterId::ARM64_Q7, AArch64::Q7},
      {codeview::RegisterId::ARM64_Q8, AArch64::Q8},
      {codeview::RegisterId::ARM64_Q9, AArch64::Q9},
      {codeview::RegisterId::ARM64_Q10, AArch64::Q10},
      {codeview::RegisterId::ARM64_Q11, AArch64::Q11},
      {codeview::RegisterId::ARM64_Q12, AArch64::Q12},
      {codeview::RegisterId::ARM64_Q13, AArch64::Q13},
      {codeview::RegisterId::ARM64_Q14, AArch64::Q14},
      {codeview::RegisterId::ARM64_Q15, AArch64::Q15},
      {codeview::RegisterId::ARM64_Q16, AArch64::Q16},
      {codeview::RegisterId::ARM64_Q17, AArch64::Q17},
      {codeview::RegisterId::ARM64_Q18, AArch64::Q18},
      {codeview::RegisterId::ARM64_Q19, AArch64::Q19},
      {codeview::RegisterId::ARM64_Q20, AArch64::Q20},
      {codeview::RegisterId::ARM64_Q21, AArch64::Q21},
      {codeview::RegisterId::ARM64_Q22, AArch64::Q22},
      {codeview::RegisterId::ARM64_Q23, AArch64::Q23},
      {codeview::RegisterId::ARM64_Q24, AArch64::Q24},
      {codeview::RegisterId::ARM64_Q25, AArch64::Q25},
      {codeview::RegisterId::ARM64_Q26, AArch64::Q26},
      {codeview::RegisterId::ARM64_Q27, AArch64::Q27},
      {codeview::RegisterId::ARM64_Q28, AArch64::Q28},
      {codeview::RegisterId::ARM64_Q29, AArch64::Q29},
      {codeview::RegisterId::ARM64_Q30, AArch64::Q30},
      {codeview::RegisterId::ARM64_Q31, AArch64::Q31},

  };
  for (unsigned I = 0; I < array_lengthof(RegMap); ++I)
    MRI->mapLLVMRegToCVReg(RegMap[I].Reg, static_cast<int>(RegMap[I].CVReg));
}

static MCRegisterInfo *createAArch64MCRegisterInfo(const Triple &Triple) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitAArch64MCRegisterInfo(X, AArch64::LR);
  AArch64_MC::initLLVMToCVRegMapping(X);
  return X;
}

static MCAsmInfo *createAArch64MCAsmInfo(const MCRegisterInfo &MRI,
                                         const Triple &TheTriple,
                                         const MCTargetOptions &Options) {
  MCAsmInfo *MAI;
  if (TheTriple.isOSBinFormatMachO())
    MAI = new AArch64MCAsmInfoDarwin(TheTriple.getArch() == Triple::aarch64_32);
  else if (TheTriple.isWindowsMSVCEnvironment())
    MAI = new AArch64MCAsmInfoMicrosoftCOFF();
  else if (TheTriple.isOSBinFormatCOFF())
    MAI = new AArch64MCAsmInfoGNUCOFF();
  else {
    assert(TheTriple.isOSBinFormatELF() && "Invalid target");
    MAI = new AArch64MCAsmInfoELF(TheTriple);
  }

  // Initial state of the frame pointer is SP.
  unsigned Reg = MRI.getDwarfRegNum(AArch64::SP, true);
  MCCFIInstruction Inst = MCCFIInstruction::createDefCfa(nullptr, Reg, 0);
  MAI->addInitialFrameState(Inst);

  return MAI;
}

static MCInstPrinter *createAArch64MCInstPrinter(const Triple &T,
                                                 unsigned SyntaxVariant,
                                                 const MCAsmInfo &MAI,
                                                 const MCInstrInfo &MII,
                                                 const MCRegisterInfo &MRI) {
  if (SyntaxVariant == 0)
    return new AArch64InstPrinter(MAI, MII, MRI);
  if (SyntaxVariant == 1)
    return new AArch64AppleInstPrinter(MAI, MII, MRI);

  return nullptr;
}

static MCStreamer *createELFStreamer(const Triple &T, MCContext &Ctx,
                                     std::unique_ptr<MCAsmBackend> &&TAB,
                                     std::unique_ptr<MCObjectWriter> &&OW,
                                     std::unique_ptr<MCCodeEmitter> &&Emitter,
                                     bool RelaxAll) {
  return createAArch64ELFStreamer(Ctx, std::move(TAB), std::move(OW),
                                  std::move(Emitter), RelaxAll);
}

static MCStreamer *createMachOStreamer(MCContext &Ctx,
                                       std::unique_ptr<MCAsmBackend> &&TAB,
                                       std::unique_ptr<MCObjectWriter> &&OW,
                                       std::unique_ptr<MCCodeEmitter> &&Emitter,
                                       bool RelaxAll,
                                       bool DWARFMustBeAtTheEnd) {
  return createMachOStreamer(Ctx, std::move(TAB), std::move(OW),
                             std::move(Emitter), RelaxAll, DWARFMustBeAtTheEnd,
                             /*LabelSections*/ true);
}

static MCStreamer *
createWinCOFFStreamer(MCContext &Ctx, std::unique_ptr<MCAsmBackend> &&TAB,
                      std::unique_ptr<MCObjectWriter> &&OW,
                      std::unique_ptr<MCCodeEmitter> &&Emitter, bool RelaxAll,
                      bool IncrementalLinkerCompatible) {
  return createAArch64WinCOFFStreamer(Ctx, std::move(TAB), std::move(OW),
                                      std::move(Emitter), RelaxAll,
                                      IncrementalLinkerCompatible);
}

namespace {

class AArch64MCInstrAnalysis : public MCInstrAnalysis {
public:
  AArch64MCInstrAnalysis(const MCInstrInfo *Info) : MCInstrAnalysis(Info) {}

  bool evaluateBranch(const MCInst &Inst, uint64_t Addr, uint64_t Size,
                      uint64_t &Target) const override {
    // Search for a PC-relative argument.
    // This will handle instructions like bcc (where the first argument is the
    // condition code) and cbz (where it is a register).
    const auto &Desc = Info->get(Inst.getOpcode());
    for (unsigned i = 0, e = Inst.getNumOperands(); i != e; i++) {
      if (Desc.OpInfo[i].OperandType == MCOI::OPERAND_PCREL) {
        int64_t Imm = Inst.getOperand(i).getImm() * 4;
        Target = Addr + Imm;
        return true;
      }
    }
    return false;
  }

  std::vector<std::pair<uint64_t, uint64_t>>
  findPltEntries(uint64_t PltSectionVA, ArrayRef<uint8_t> PltContents,
                 uint64_t GotPltSectionVA,
                 const Triple &TargetTriple) const override {
    // Do a lightweight parsing of PLT entries.
    std::vector<std::pair<uint64_t, uint64_t>> Result;
    for (uint64_t Byte = 0, End = PltContents.size(); Byte + 7 < End;
         Byte += 4) {
      uint32_t Insn = support::endian::read32le(PltContents.data() + Byte);
      uint64_t Off = 0;
      // Check for optional bti c that prefixes adrp in BTI enabled entries
      if (Insn == 0xd503245f) {
         Off = 4;
         Insn = support::endian::read32le(PltContents.data() + Byte + Off);
      }
      // Check for adrp.
      if ((Insn & 0x9f000000) != 0x90000000)
        continue;
      Off += 4;
      uint64_t Imm = (((PltSectionVA + Byte) >> 12) << 12) +
            (((Insn >> 29) & 3) << 12) + (((Insn >> 5) & 0x3ffff) << 14);
      uint32_t Insn2 =
          support::endian::read32le(PltContents.data() + Byte + Off);
      // Check for: ldr Xt, [Xn, #pimm].
      if (Insn2 >> 22 == 0x3e5) {
        Imm += ((Insn2 >> 10) & 0xfff) << 3;
        Result.push_back(std::make_pair(PltSectionVA + Byte, Imm));
        Byte += 4;
      }
    }
    return Result;
  }
};

} // end anonymous namespace

static MCInstrAnalysis *createAArch64InstrAnalysis(const MCInstrInfo *Info) {
  return new AArch64MCInstrAnalysis(Info);
}

// Force static initialization.
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeAArch64TargetMC() {
  for (Target *T : {&getTheAArch64leTarget(), &getTheAArch64beTarget(),
                    &getTheAArch64_32Target(), &getTheARM64Target(),
                    &getTheARM64_32Target()}) {
    // Register the MC asm info.
    RegisterMCAsmInfoFn X(*T, createAArch64MCAsmInfo);

    // Register the MC instruction info.
    TargetRegistry::RegisterMCInstrInfo(*T, createAArch64MCInstrInfo);

    // Register the MC register info.
    TargetRegistry::RegisterMCRegInfo(*T, createAArch64MCRegisterInfo);

    // Register the MC subtarget info.
    TargetRegistry::RegisterMCSubtargetInfo(*T, createAArch64MCSubtargetInfo);

    // Register the MC instruction analyzer.
    TargetRegistry::RegisterMCInstrAnalysis(*T, createAArch64InstrAnalysis);

    // Register the MC Code Emitter
    TargetRegistry::RegisterMCCodeEmitter(*T, createAArch64MCCodeEmitter);

    // Register the obj streamers.
    TargetRegistry::RegisterELFStreamer(*T, createELFStreamer);
    TargetRegistry::RegisterMachOStreamer(*T, createMachOStreamer);
    TargetRegistry::RegisterCOFFStreamer(*T, createWinCOFFStreamer);

    // Register the obj target streamer.
    TargetRegistry::RegisterObjectTargetStreamer(
        *T, createAArch64ObjectTargetStreamer);

    // Register the asm streamer.
    TargetRegistry::RegisterAsmTargetStreamer(*T,
                                              createAArch64AsmTargetStreamer);
    // Register the MCInstPrinter.
    TargetRegistry::RegisterMCInstPrinter(*T, createAArch64MCInstPrinter);
  }

  // Register the asm backend.
  for (Target *T : {&getTheAArch64leTarget(), &getTheAArch64_32Target(),
                    &getTheARM64Target(), &getTheARM64_32Target()})
    TargetRegistry::RegisterMCAsmBackend(*T, createAArch64leAsmBackend);
  TargetRegistry::RegisterMCAsmBackend(getTheAArch64beTarget(),
                                       createAArch64beAsmBackend);
}
