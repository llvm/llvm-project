//===- lib/MC/MCSQELFStreamer.cpp ----------------------------------------===//
// SQLite ELF Object Output
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file assembles .s files and emits SQLite ELF object files.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSQELFStreamer.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sqelf"

using namespace llvm;

MCSQELFStreamer::MCSQELFStreamer(MCContext &Context,
                                 std::unique_ptr<MCAsmBackend> TAB,
                                 std::unique_ptr<MCObjectWriter> OW,
                                 std::unique_ptr<MCCodeEmitter> Emitter)
    : MCObjectStreamer(Context, std::move(TAB), std::move(OW),
                       std::move(Emitter)) {}

bool MCSQELFStreamer::emitSymbolAttribute(MCSymbol *Symbol,
                                          MCSymbolAttr Attribute) {
  return true;
}

void MCSQELFStreamer::emitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                       Align ByteAlignment) {}

void MCSQELFStreamer::emitZerofill(MCSection *Section, MCSymbol *Symbol,
                                   uint64_t Size, Align ByteAlignment,
                                   SMLoc Loc) {}

void MCSQELFStreamer::emitInstToData(const MCInst &Inst,
                                     const MCSubtargetInfo &STI) {
  MCAssembler &Assembler = getAssembler();
  SmallVector<MCFixup, 4> Fixups;

  // Fetch the raw representation of the code
  SmallString<256> Code;
  Assembler.getEmitter().encodeInstruction(Inst, Code, Fixups, STI);

  const llvm::MCContext &Ctx = getContext();
  const llvm::MCAsmInfo *MAI = Ctx.getAsmInfo();
  const llvm::MCRegisterInfo *MRI = Ctx.getRegisterInfo();
  const llvm::Triple &TheTriple = Ctx.getTargetTriple();

  std::string Error;
  const Target *Target = TargetRegistry::lookupTarget(TheTriple.str(), Error);
  unsigned AsmVariant = 0; // typically 0 for the default assembly variant

  std::unique_ptr<const MCInstrInfo> MII(Target->createMCInstrInfo());
  std::unique_ptr<llvm::MCInstPrinter> Printer(
      Target->createMCInstPrinter(TheTriple, AsmVariant, *MAI, *MII, *MRI));

  const char *Mnemonic = Printer->getMnemonic(&Inst).first;
  std::string InstructionStr;
  raw_string_ostream SS(InstructionStr);
  Printer->printInst(&Inst, 0, "", STI, SS);

  // TODO(fzakaria): really bad way to get the OpStr. Find better way?
  // idealy we should have Printer return a string rather than write to an
  // ostream
  StringRef OpStr =
      StringRef(InstructionStr).split("\t").second.split("\t").second;

  LLVM_DEBUG(dbgs() << Mnemonic << "\n");
  LLVM_DEBUG(dbgs() << OpStr << "\n");
}

MCStreamer *llvm::createSQELFStreamer(MCContext &Context,
                                      std::unique_ptr<MCAsmBackend> &&MAB,
                                      std::unique_ptr<MCObjectWriter> &&OW,
                                      std::unique_ptr<MCCodeEmitter> &&CE,
                                      bool RelaxAll) {
  MCSQELFStreamer *S = new MCSQELFStreamer(Context, std::move(MAB),
                                           std::move(OW), std::move(CE));
  if (RelaxAll)
    S->getAssembler().setRelaxAll(true);
  return S;
}
