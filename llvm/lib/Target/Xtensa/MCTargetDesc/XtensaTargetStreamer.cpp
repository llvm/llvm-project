//===-- XtensaTargetStreamer.cpp - Xtensa Target Streamer Methods ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides Xtensa specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "XtensaTargetStreamer.h"
#include "XtensaInstPrinter.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;

XtensaTargetStreamer::XtensaTargetStreamer(MCStreamer &S)
    : MCTargetStreamer(S) {}

XtensaTargetAsmStreamer::XtensaTargetAsmStreamer(MCStreamer &S,
                                                 formatted_raw_ostream &OS)
    : XtensaTargetStreamer(S), OS(OS) {}

void XtensaTargetAsmStreamer::emitLiteral(MCSymbol *LblSym, const MCExpr *Value,
                                          SMLoc L) {
  const MCAsmInfo *MAI = Streamer.getContext().getAsmInfo();

  OS << "\t.literal\t";
  LblSym->print(OS, MAI);
  OS << ", ";
  Value->print(OS, MAI);
  OS << '\n';
}

void XtensaTargetAsmStreamer::emitLiteralPosition() {
  OS << "\t.literal_position\n";
}

XtensaTargetELFStreamer::XtensaTargetELFStreamer(MCStreamer &S)
    : XtensaTargetStreamer(S) {}

static std::string getLiteralSectionName(std::string CSectionName) {
  std::size_t Pos = CSectionName.find(".text");
  std::string SectionName;
  if (Pos != std::string::npos) {
    if (Pos > 0)
      SectionName = CSectionName.substr(0, Pos + 5);
    else
      SectionName = "";
    SectionName += ".literal";
    SectionName += CSectionName.substr(Pos + 5);
  } else {
    SectionName = CSectionName;
    SectionName += ".literal";
  }
  return SectionName;
}

void XtensaTargetELFStreamer::emitLiteral(MCSymbol *LblSym, const MCExpr *Value,
                                          SMLoc L) {
  MCContext &Context = getStreamer().getContext();
  MCStreamer &OutStreamer = getStreamer();
  MCSectionELF *CS = (MCSectionELF *)OutStreamer.getCurrentSectionOnly();
  std::string SectionName = getLiteralSectionName(CS->getName().str());

  MCSection *ConstSection = Context.getELFSection(
      SectionName, ELF::SHT_PROGBITS, ELF::SHF_EXECINSTR | ELF::SHF_ALLOC);

  OutStreamer.pushSection();
  OutStreamer.switchSection(ConstSection);
  OutStreamer.emitLabel(LblSym, L);
  OutStreamer.emitValue(Value, 4, L);
  OutStreamer.popSection();
}

MCELFStreamer &XtensaTargetELFStreamer::getStreamer() {
  return static_cast<MCELFStreamer &>(Streamer);
}
