//===-- LoongArchTargetStreamer.cpp - LoongArch Target Streamer Methods ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides LoongArch specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "LoongArchTargetStreamer.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"

using namespace llvm;

LoongArchTargetStreamer::LoongArchTargetStreamer(MCStreamer &S)
    : MCTargetStreamer(S) {}

void LoongArchTargetStreamer::setTargetABI(LoongArchABI::ABI ABI) {
  assert(ABI != LoongArchABI::ABI_Unknown &&
         "Improperly initialized target ABI");
  TargetABI = ABI;
}

void LoongArchTargetStreamer::emitDirectiveOptionPush() {}
void LoongArchTargetStreamer::emitDirectiveOptionPop() {}
void LoongArchTargetStreamer::emitDirectiveOptionRelax() {}
void LoongArchTargetStreamer::emitDirectiveOptionNoRelax() {}
void LoongArchTargetStreamer::emitDTPRel32Value(const MCExpr *) {}
void LoongArchTargetStreamer::emitDTPRel64Value(const MCExpr *) {}

// This part is for ascii assembly output.
LoongArchTargetAsmStreamer::LoongArchTargetAsmStreamer(
    MCStreamer &S, formatted_raw_ostream &OS)
    : LoongArchTargetStreamer(S), OS(OS) {}

void LoongArchTargetAsmStreamer::emitDirectiveOptionPush() {
  OS << "\t.option\tpush\n";
}

void LoongArchTargetAsmStreamer::emitDirectiveOptionPop() {
  OS << "\t.option\tpop\n";
}

void LoongArchTargetAsmStreamer::emitDirectiveOptionRelax() {
  OS << "\t.option\trelax\n";
}

void LoongArchTargetAsmStreamer::emitDirectiveOptionNoRelax() {
  OS << "\t.option\tnorelax\n";
}

void LoongArchTargetAsmStreamer::emitDTPRel32Value(const MCExpr *Value) {
  auto &MAI = getStreamer().getContext().getAsmInfo();
  OS << "\t.dtprelword\t";
  MAI.printExpr(OS, *Value);
  OS << '\n';
}

void LoongArchTargetAsmStreamer::emitDTPRel64Value(const MCExpr *Value) {
  auto &MAI = getStreamer().getContext().getAsmInfo();
  OS << "\t.dtpreldword\t";
  MAI.printExpr(OS, *Value);
  OS << '\n';
}
