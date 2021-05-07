//===- AArch64TargetStreamer.cpp - AArch64TargetStreamer class ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AArch64TargetStreamer class.
//
//===----------------------------------------------------------------------===//

#include "AArch64TargetStreamer.h"
#include "AArch64MCAsmInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/ConstantPools.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

static cl::opt<bool> MarkBTIProperty(
    "aarch64-mark-bti-property", cl::Hidden,
    cl::desc("Add .note.gnu.property with BTI to assembly files"),
    cl::init(false));

//
// AArch64TargetStreamer Implemenation
//
AArch64TargetStreamer::AArch64TargetStreamer(MCStreamer &S)
    : MCTargetStreamer(S), ConstantPools(new AssemblerConstantPools()) {}

AArch64TargetStreamer::~AArch64TargetStreamer() = default;

// The constant pool handling is shared by all AArch64TargetStreamer
// implementations.
const MCExpr *AArch64TargetStreamer::addConstantPoolEntry(const MCExpr *Expr,
                                                          unsigned Size,
                                                          SMLoc Loc) {
  return ConstantPools->addEntry(Streamer, Expr, Size, Loc);
}

void AArch64TargetStreamer::emitCurrentConstantPool() {
  ConstantPools->emitForCurrentSection(Streamer);
}

// finish() - write out any non-empty assembler constant pools and
//   write out note.gnu.properties if need.
void AArch64TargetStreamer::finish() {
  ConstantPools->emitAll(Streamer);

  if (MarkBTIProperty)
    emitNoteSection(ELF::GNU_PROPERTY_AARCH64_FEATURE_1_BTI);
}

void AArch64TargetStreamer::emitNoteSection(unsigned Flags) {
  if (Flags == 0)
    return;

  MCStreamer &OutStreamer = getStreamer();
  MCContext &Context = OutStreamer.getContext();
  // Emit a .note.gnu.property section with the flags.
  MCSectionELF *Nt = Context.getELFSection(".note.gnu.property", ELF::SHT_NOTE,
                                           ELF::SHF_ALLOC);
  if (Nt->isRegistered()) {
    SMLoc Loc;
    Context.reportWarning(
        Loc,
        "The .note.gnu.property is not emitted because it is already present.");
    return;
  }
  MCSection *Cur = OutStreamer.getCurrentSectionOnly();
  OutStreamer.SwitchSection(Nt);

  // Emit the note header.
  OutStreamer.emitValueToAlignment(Align(8).value());
  OutStreamer.emitIntValue(4, 4);     // data size for "GNU\0"
  OutStreamer.emitIntValue(4 * 4, 4); // Elf_Prop size
  OutStreamer.emitIntValue(ELF::NT_GNU_PROPERTY_TYPE_0, 4);
  OutStreamer.emitBytes(StringRef("GNU", 4)); // note name

  // Emit the PAC/BTI properties.
  OutStreamer.emitIntValue(ELF::GNU_PROPERTY_AARCH64_FEATURE_1_AND, 4);
  OutStreamer.emitIntValue(4, 4);     // data size
  OutStreamer.emitIntValue(Flags, 4); // data
  OutStreamer.emitIntValue(0, 4);     // pad

  OutStreamer.endSection(Nt);
  OutStreamer.SwitchSection(Cur);
}

void AArch64TargetStreamer::emitInst(uint32_t Inst) {
  char Buffer[4];

  // We can't just use EmitIntValue here, as that will swap the
  // endianness on big-endian systems (instructions are always
  // little-endian).
  for (unsigned I = 0; I < 4; ++I) {
    Buffer[I] = uint8_t(Inst);
    Inst >>= 8;
  }

  getStreamer().emitBytes(StringRef(Buffer, 4));
}

MCTargetStreamer *
llvm::createAArch64ObjectTargetStreamer(MCStreamer &S,
                                        const MCSubtargetInfo &STI) {
  const Triple &TT = STI.getTargetTriple();
  if (TT.isOSBinFormatELF())
    return new AArch64TargetELFStreamer(S);
  if (TT.isOSBinFormatCOFF())
    return new AArch64TargetWinCOFFStreamer(S);
  return nullptr;
}
