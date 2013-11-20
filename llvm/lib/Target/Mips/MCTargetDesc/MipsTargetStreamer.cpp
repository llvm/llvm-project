//===-- MipsTargetStreamer.cpp - Mips Target Streamer Methods -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Mips specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "MipsTargetStreamer.h"
#include "llvm/MC/MCELF.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;

static cl::opt<bool> PrintHackDirectives("print-hack-directives",
                                         cl::init(false), cl::Hidden);

// pin vtable to this file
void MipsTargetStreamer::anchor() {}

MipsTargetAsmStreamer::MipsTargetAsmStreamer(formatted_raw_ostream &OS)
    : OS(OS) {}

void MipsTargetAsmStreamer::emitMipsHackELFFlags(unsigned Flags) {
  if (!PrintHackDirectives)
    return;

  OS << "\t.mips_hack_elf_flags 0x";
  OS.write_hex(Flags);
  OS << '\n';
}
void MipsTargetAsmStreamer::emitMipsHackSTOCG(MCSymbol *Sym, unsigned Val) {
  if (!PrintHackDirectives)
    return;

  OS << "\t.mips_hack_stocg ";
  OS << Sym->getName();
  OS << ", ";
  OS << Val;
  OS << '\n';
}

MCELFStreamer &MipsTargetELFStreamer::getStreamer() {
  return static_cast<MCELFStreamer &>(*Streamer);
}

void MipsTargetELFStreamer::emitMipsHackELFFlags(unsigned Flags) {
  MCAssembler &MCA = getStreamer().getAssembler();
  MCA.setELFHeaderEFlags(Flags);
}

// Set a symbol's STO flags
void MipsTargetELFStreamer::emitMipsHackSTOCG(MCSymbol *Sym, unsigned Val) {
  MCSymbolData &Data = getStreamer().getOrCreateSymbolData(Sym);
  // The "other" values are stored in the last 6 bits of the second byte
  // The traditional defines for STO values assume the full byte and thus
  // the shift to pack it.
  MCELF::setOther(Data, Val >> 2);
}
