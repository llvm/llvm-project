//===- MCEncodingCommentHelper.cpp - Encoding Comment Helper ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCEncodingCommentHelper.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

void llvm::mc::emitEncodingComment(raw_ostream &OS, const MCInst &Inst,
                                   const MCSubtargetInfo &STI,
                                   MCAssembler &Assembler, const MCAsmInfo &MAI,
                                   bool ForceLE) {
  SmallString<256> Code;
  SmallVector<MCFixup, 4> Fixups;

  // If we have no code emitter, don't emit code.
  if (!Assembler.getEmitterPtr())
    return;

  Assembler.getEmitter().encodeInstruction(Inst, Code, Fixups, STI);

  // If we are showing fixups, create symbolic markers in the encoded
  // representation. We do this by making a per-bit map to the fixup item index,
  // then trying to display it as nicely as possible.
  SmallVector<uint8_t, 64> FixupMap;
  FixupMap.resize(Code.size() * 8);
  for (unsigned I = 0, E = Code.size() * 8; I != E; ++I)
    FixupMap[I] = 0;

  for (unsigned I = 0, E = Fixups.size(); I != E; ++I) {
    MCFixup &F = Fixups[I];
    MCFixupKindInfo Info = Assembler.getBackend().getFixupKindInfo(F.getKind());
    for (unsigned J = 0; J != Info.TargetSize; ++J) {
      unsigned Index = F.getOffset() * 8 + Info.TargetOffset + J;
      assert(Index < Code.size() * 8 && "Invalid offset in fixup!");
      FixupMap[Index] = 1 + I;
    }
  }

  // FIXME: Note the fixup comments for Thumb2 are completely bogus since the
  // high order halfword of a 32-bit Thumb2 instruction is emitted first.
  OS << "encoding: [";
  for (unsigned I = 0, E = Code.size(); I != E; ++I) {
    if (I)
      OS << ',';

    // See if all bits are the same map entry.
    uint8_t MapEntry = FixupMap[I * 8 + 0];
    for (unsigned J = 1; J != 8; ++J) {
      if (FixupMap[I * 8 + J] == MapEntry)
        continue;

      MapEntry = uint8_t(~0U);
      break;
    }

    if (MapEntry != uint8_t(~0U)) {
      if (MapEntry == 0) {
        OS << format("0x%02x", uint8_t(Code[I]));
      } else {
        if (Code[I]) {
          // FIXME: Some of the 8 bits require fix up.
          OS << format("0x%02x", uint8_t(Code[I])) << '\''
             << char('A' + MapEntry - 1) << '\'';
        } else
          OS << char('A' + MapEntry - 1);
      }
    } else {
      // Otherwise, write out in binary.
      OS << "0b";
      for (unsigned J = 8; J--;) {
        unsigned Bit = (Code[I] >> J) & 1;

        unsigned FixupBit;
        // RISC-V instructions are always little-endian.
        // The FixupMap is indexed by actual bit positions in the LE
        // instruction.
        if (MAI.isLittleEndian() || ForceLE)
          FixupBit = I * 8 + J;
        else
          FixupBit = I * 8 + (7 - J);

        if (uint8_t MapEntry = FixupMap[FixupBit]) {
          assert(Bit == 0 && "Encoder wrote into fixed up bit!");
          OS << char('A' + MapEntry - 1);
        } else
          OS << Bit;
      }
    }
  }
  OS << "]\n";

  for (unsigned I = 0, E = Fixups.size(); I != E; ++I) {
    MCFixup &F = Fixups[I];
    OS << "  fixup " << char('A' + I) << " - "
       << "offset: " << F.getOffset() << ", value: ";
    MAI.printExpr(OS, *F.getValue());
    auto Kind = F.getKind();
    if (mc::isRelocation(Kind))
      OS << ", relocation type: " << Kind;
    else {
      OS << ", kind: ";
      auto Info = Assembler.getBackend().getFixupKindInfo(Kind);
      if (F.isPCRel() && StringRef(Info.Name).starts_with("FK_Data_"))
        OS << "FK_PCRel_" << (Info.TargetSize / 8);
      else
        OS << Info.Name;
    }
    OS << '\n';
  }
}
