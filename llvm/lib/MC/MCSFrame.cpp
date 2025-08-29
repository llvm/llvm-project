//===- lib/MC/MCSFrame.cpp - MCSFrame implementation ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSFrame.h"
#include "llvm/BinaryFormat/SFrame.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/EndianStream.h"

using namespace llvm;
using namespace sframe;

namespace {

// High-level structure to track info needed to emit a sframe_func_desc_entry
// and its associated FREs.
struct SFrameFDE {
  // Reference to the original dwarf frame to avoid copying.
  const MCDwarfFrameInfo &DFrame;
  // Label where this FDE's FREs start.
  MCSymbol *FREStart;

  SFrameFDE(const MCDwarfFrameInfo &DF, MCSymbol *FRES)
      : DFrame(DF), FREStart(FRES) {}

  void emit(MCObjectStreamer &S, const MCSymbol *FRESubSectionStart) {
    MCContext &C = S.getContext();

    // sfde_func_start_address
    const MCExpr *V = C.getAsmInfo()->getExprForFDESymbol(
        &(*DFrame.Begin), C.getObjectFileInfo()->getFDEEncoding(), S);
    S.emitValue(V, sizeof(int32_t));

    // sfde_func_size
    S.emitAbsoluteSymbolDiff(DFrame.End, DFrame.Begin, sizeof(uint32_t));

    // sfde_func_start_fre_off
    auto *F = S.getCurrentFragment();
    const MCExpr *Diff = MCBinaryExpr::createSub(
        MCSymbolRefExpr::create(FREStart, C),
        MCSymbolRefExpr::create(FRESubSectionStart, C), C);

    F->addFixup(MCFixup::create(F->getContents().size(), Diff,
                                MCFixup::getDataKindForSize(4)));
    S.emitInt32(0);

    // sfde_func_start_num_fres
    S.emitInt32(0);

    // sfde_func_info word
    FDEInfo<endianness::native> I;
    I.setFuncInfo(0 /* No pauth key */, FDEType::PCInc, FREType::Addr1);
    S.emitInt8(I.Info);

    // sfde_func_rep_size. Not relevant in non-PCMASK fdes.
    S.emitInt8(0);

    // sfde_func_padding2
    S.emitInt16(0);
  }
};

// Emitting these field-by-field, instead of constructing the actual structures
// lets Streamer do target endian-fixups for free.

class SFrameEmitterImpl {
  MCObjectStreamer &Streamer;
  SmallVector<SFrameFDE> FDEs;
  ABI SFrameABI;
  MCSymbol *FDESubSectionStart;
  MCSymbol *FRESubSectionStart;
  MCSymbol *FRESubSectionEnd;

public:
  SFrameEmitterImpl(MCObjectStreamer &Streamer) : Streamer(Streamer) {
    assert(Streamer.getContext()
               .getObjectFileInfo()
               ->getSFrameABIArch()
               .has_value());
    FDEs.reserve(Streamer.getDwarfFrameInfos().size());
    SFrameABI = *Streamer.getContext().getObjectFileInfo()->getSFrameABIArch();
    FDESubSectionStart = Streamer.getContext().createTempSymbol();
    FRESubSectionStart = Streamer.getContext().createTempSymbol();
    FRESubSectionEnd = Streamer.getContext().createTempSymbol();
  }

  void BuildSFDE(const MCDwarfFrameInfo &DF) {
    FDEs.emplace_back(DF, Streamer.getContext().createTempSymbol());
  }

  void emitPreamble() {
    Streamer.emitInt16(Magic);
    Streamer.emitInt8(static_cast<uint8_t>(Version::V2));
    Streamer.emitInt8(static_cast<uint8_t>(Flags::FDEFuncStartPCRel));
  }

  void emitHeader() {
    emitPreamble();
    // sfh_abi_arch
    Streamer.emitInt8(static_cast<uint8_t>(SFrameABI));
    // sfh_cfa_fixed_fp_offset
    Streamer.emitInt8(0);
    // sfh_cfa_fixed_ra_offset
    Streamer.emitInt8(0);
    // sfh_auxhdr_len
    Streamer.emitInt8(0);
    // shf_num_fdes
    Streamer.emitInt32(FDEs.size());
    // shf_num_fres
    Streamer.emitInt32(0);
    // shf_fre_len
    Streamer.emitAbsoluteSymbolDiff(FRESubSectionEnd, FRESubSectionStart,
                                    sizeof(int32_t));
    // shf_fdeoff. With no sfh_auxhdr, these immediately follow this header.
    Streamer.emitInt32(0);
    // shf_freoff
    Streamer.emitAbsoluteSymbolDiff(FRESubSectionStart, FDESubSectionStart,
                                    sizeof(uint32_t));
  }

  void emitFDEs() {
    Streamer.emitLabel(FDESubSectionStart);
    for (auto &FDE : FDEs) {
      FDE.emit(Streamer, FRESubSectionStart);
    }
  }

  void emitFREs() {
    Streamer.emitLabel(FRESubSectionStart);
    for (auto &FDE : FDEs)
      Streamer.emitLabel(FDE.FREStart);
    Streamer.emitLabel(FRESubSectionEnd);
  }
};

} // end anonymous namespace

void MCSFrameEmitter::emit(MCObjectStreamer &Streamer) {
  MCContext &Context = Streamer.getContext();
  // If this target doesn't support sframes, return now. Gas doesn't warn in
  // this case, but if we want to, it should be done at option-parsing time,
  // rather than here.
  if (!Streamer.getContext()
           .getObjectFileInfo()
           ->getSFrameABIArch()
           .has_value())
    return;

  SFrameEmitterImpl Emitter(Streamer);
  ArrayRef<MCDwarfFrameInfo> FrameArray = Streamer.getDwarfFrameInfos();

  // Both the header itself and the FDEs include various offsets and counts.
  // Therefore, all of this must be precomputed.
  for (const auto &DFrame : FrameArray)
    Emitter.BuildSFDE(DFrame);

  MCSection *Section = Context.getObjectFileInfo()->getSFrameSection();
  // Not strictly necessary, but gas always aligns to 8, so match that.
  Section->ensureMinAlignment(Align(8));
  Streamer.switchSection(Section);
  MCSymbol *SectionStart = Context.createTempSymbol();
  Streamer.emitLabel(SectionStart);
  Emitter.emitHeader();
  Emitter.emitFDEs();
  Emitter.emitFREs();
}
