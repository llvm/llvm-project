//===-------- Next32ELFStreamer.cpp - ELF Object Output -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Next32ELFStreamer.h"
#include "Next32MCExpr.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCCodeView.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

Next32ELFStreamer::Next32ELFStreamer(MCContext &Context,
                                     std::unique_ptr<MCAsmBackend> MAB,
                                     std::unique_ptr<MCObjectWriter> OW,
                                     std::unique_ptr<MCCodeEmitter> Emitter)
    : MCELFStreamer(Context, std::move(MAB), std::move(OW),
                    std::move(Emitter)) {}

void Next32ELFStreamer::CreateFixup(const MCExpr *Value, SMLoc Loc,
                                    MCDataFragment *DF, Next32::Fixups Kind) {
  MCStreamer::emitValueImpl(Value, 4, Loc);
  DF->getFixups().push_back(
      MCFixup::create(DF->getContents().size(), Value, (MCFixupKind)Kind, Loc));
  DF->getContents().resize(DF->getContents().size() + 4, 0);
}

void Next32ELFStreamer::emitValueImpl(const MCExpr *Value, unsigned Size,
                                      SMLoc Loc) {
  if (!isa<MCSymbolRefExpr>(Value) && !isa<MCBinaryExpr>(Value)) {
    MCELFStreamer::emitValueImpl(Value, Size, Loc);
    return;
  }

  // Hack: detect whether this is a DWARF debug section according to SHF_ALLOC
  MCSectionELF *sec = static_cast<MCSectionELF *>(getCurrentSectionOnly());
  if (!(sec->getFlags() & ELF::SHF_ALLOC)) {
    MCELFStreamer::emitValueImpl(Value, Size, Loc);
    return;
  }

  MCDataFragment *DF = getOrCreateDataFragment();
  MCDwarfLineEntry::make(this, getCurrentSectionOnly());

  switch (Size) {
  case 4:
    CreateFixup(Value, Loc, DF, Next32::reloc_4byte_sym_function);
    break;
  case 8: {
    Next32::Fixups HighType, LowType;
    if (isa<MCBinaryExpr>(Value)) {
      HighType = Next32::reloc_4byte_mem_high;
      LowType = Next32::reloc_4byte_mem_low;
    } else if (cast<MCSymbolRefExpr>(Value)->getKind() ==
               MCSymbolRefExpr::VK_Next32_FUNC_PTR) {
      HighType = Next32::reloc_4byte_func_high;
      LowType = Next32::reloc_4byte_func_low;
    } else {
      HighType = Next32::reloc_4byte_mem_high;
      LowType = Next32::reloc_4byte_mem_low;
    }
    CreateFixup(Value, Loc, DF, LowType);
    CreateFixup(Value, Loc, DF, HighType);
    break;
  }
  default:
    llvm_unreachable("Unsupported value");
  }
}

MCELFStreamer *llvm::createNext32ELFStreamer(
    MCContext &Context, std::unique_ptr<MCAsmBackend> MAB,
    std::unique_ptr<MCObjectWriter> OW, std::unique_ptr<MCCodeEmitter> Emitter) {
  return new Next32ELFStreamer(Context, std::move(MAB), std::move(OW),
                               std::move(Emitter));
}
