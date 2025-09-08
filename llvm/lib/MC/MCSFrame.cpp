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

// High-level structure to track info needed to emit a
// sframe_frame_row_entry_addrX. On disk these have both a fixed portion of type
// sframe_frame_row_entry_addrX and trailing data of X * S bytes, where X is the
// datum size, and S is 1, 2, or 3 depending on which of CFA, SP, and FP are
// being tracked.
struct SFrameFRE {
  // An FRE describes how to find the registers when the PC is at this
  // Label from function start.
  const MCSymbol *Label = nullptr;
  size_t CFAOffset = 0;
  size_t FPOffset = 0;
  size_t RAOffset = 0;
  bool FromFP = false;
  bool CFARegSet = false;

  SFrameFRE(const MCSymbol *Start) : Label(Start) {}
};

// High-level structure to track info needed to emit a sframe_func_desc_entry
// and its associated FREs.
struct SFrameFDE {
  // Reference to the original dwarf frame to avoid copying.
  const MCDwarfFrameInfo &DFrame;
  // Label where this FDE's FREs start.
  MCSymbol *FREStart;
  // Unwinding fres
  SmallVector<SFrameFRE> FREs;

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

    // sfde_func_num_fres
    // TODO: When we actually emit fres, replace 0 with FREs.size()
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
  // Target-specific convenience variables to detect when a CFI instruction
  // references these registers. Unlike in dwarf frame descriptions, they never
  // escape into the sframe section itself.
  unsigned SPReg;
  unsigned FPReg;
  unsigned RAReg;
  MCSymbol *FDESubSectionStart;
  MCSymbol *FRESubSectionStart;
  MCSymbol *FRESubSectionEnd;

  bool setCFARegister(SFrameFRE &FRE, const MCCFIInstruction &I) {
    if (I.getRegister() == SPReg) {
      FRE.CFARegSet = true;
      FRE.FromFP = false;
      return true;
    }
    if (I.getRegister() == FPReg) {
      FRE.CFARegSet = true;
      FRE.FromFP = true;
      return true;
    }
    Streamer.getContext().reportWarning(
        I.getLoc(), "canonical Frame Address not in stack- or frame-pointer. "
                    "Omitting SFrame unwind info for this function");
    return false;
  }

  bool setCFAOffset(SFrameFRE &FRE, const SMLoc &Loc, size_t Offset) {
    if (!FRE.CFARegSet) {
      Streamer.getContext().reportWarning(
          Loc, "adjusting CFA offset without a base register. "
               "Omitting SFrame unwind info for this function");
      return false;
    }
    FRE.CFAOffset = Offset;
    return true;
  }

  // Add the effects of CFI to the current FDE, creating a new FRE when
  // necessary.
  bool handleCFI(SFrameFDE &FDE, SFrameFRE &FRE, const MCCFIInstruction &CFI) {
    switch (CFI.getOperation()) {
    case MCCFIInstruction::OpDefCfaRegister:
      return setCFARegister(FRE, CFI);
    case MCCFIInstruction::OpDefCfa:
    case MCCFIInstruction::OpLLVMDefAspaceCfa:
      if (!setCFARegister(FRE, CFI))
        return false;
      return setCFAOffset(FRE, CFI.getLoc(), CFI.getOffset());
    case MCCFIInstruction::OpOffset:
      if (CFI.getRegister() == FPReg)
        FRE.FPOffset = CFI.getOffset();
      else if (CFI.getRegister() == RAReg)
        FRE.RAOffset = CFI.getOffset();
      return true;
    case MCCFIInstruction::OpRelOffset:
      if (CFI.getRegister() == FPReg)
        FRE.FPOffset += CFI.getOffset();
      else if (CFI.getRegister() == RAReg)
        FRE.RAOffset += CFI.getOffset();
      return true;
    case MCCFIInstruction::OpDefCfaOffset:
      return setCFAOffset(FRE, CFI.getLoc(), CFI.getOffset());
    case MCCFIInstruction::OpAdjustCfaOffset:
      return setCFAOffset(FRE, CFI.getLoc(), FRE.CFAOffset + CFI.getOffset());
    case MCCFIInstruction::OpRememberState:
      // TODO: Implement. Will use FDE.
      return true;
    case MCCFIInstruction::OpRestore:
      // TODO: Implement. Will use FDE.
      return true;
    case MCCFIInstruction::OpRestoreState:
      // TODO: Implement. Will use FDE.
      return true;
    case MCCFIInstruction::OpEscape:
      // TODO: Implement. Will use FDE.
      return true;
    default:
      // Instructions that don't affect the CFA, RA, and SP can be safely
      // ignored.
      return true;
    }
  }

public:
  SFrameEmitterImpl(MCObjectStreamer &Streamer) : Streamer(Streamer) {
    assert(Streamer.getContext()
               .getObjectFileInfo()
               ->getSFrameABIArch()
               .has_value());
    FDEs.reserve(Streamer.getDwarfFrameInfos().size());
    SFrameABI = *Streamer.getContext().getObjectFileInfo()->getSFrameABIArch();
    switch (SFrameABI) {
    case ABI::AArch64EndianBig:
    case ABI::AArch64EndianLittle:
      SPReg = 31;
      RAReg = 29;
      FPReg = 30;
      break;
    case ABI::AMD64EndianLittle:
      SPReg = 7;
      // RARegister untracked in this abi. Value chosen to match
      // MCDwarfFrameInfo constructor.
      RAReg = static_cast<unsigned>(INT_MAX);
      FPReg = 6;
      break;
    }

    FDESubSectionStart = Streamer.getContext().createTempSymbol();
    FRESubSectionStart = Streamer.getContext().createTempSymbol();
    FRESubSectionEnd = Streamer.getContext().createTempSymbol();
  }

  bool atSameLocation(const MCSymbol *Left, const MCSymbol *Right) {
    return Left != nullptr && Right != nullptr &&
           Left->getFragment() == Right->getFragment() &&
           Left->getOffset() == Right->getOffset();
  }

  bool equalIgnoringLocation(const SFrameFRE &Left, const SFrameFRE &Right) {
    return Left.CFAOffset == Right.CFAOffset &&
           Left.FPOffset == Right.FPOffset && Left.RAOffset == Right.RAOffset &&
           Left.FromFP == Right.FromFP && Left.CFARegSet == Right.CFARegSet;
  }

  void buildSFDE(const MCDwarfFrameInfo &DF) {
    bool Valid = true;
    SFrameFDE FDE(DF, Streamer.getContext().createTempSymbol());
    // This would have been set via ".cfi_return_column", but
    // MCObjectStreamer doesn't emit an MCCFIInstruction for that. It just
    // sets the DF.RAReg.
    // FIXME: This also prevents providing a proper location for the error.
    // LLVM doesn't change the return column itself, so this was
    // hand-written assembly.
    if (DF.RAReg != RAReg) {
      Streamer.getContext().reportWarning(
          SMLoc(), "non-default RA register in .cfi_return_column " +
                       Twine(DF.RAReg) +
                       ". Omitting SFrame unwind info for this function");
      Valid = false;
    }
    MCSymbol *LastLabel = DF.Begin;
    SFrameFRE BaseFRE(LastLabel);
    if (!DF.IsSimple) {
      for (const auto &CFI :
           Streamer.getContext().getAsmInfo()->getInitialFrameState())
        if (!handleCFI(FDE, BaseFRE, CFI))
          Valid = false;
    }
    FDE.FREs.push_back(BaseFRE);

    for (const auto &CFI : DF.Instructions) {
      // Instructions from InitialFrameState may not have a label, but if these
      // instructions don't, then they are in dead code or otherwise unused.
      // TODO: This check follows MCDwarf.cpp
      // FrameEmitterImplementation::emitCFIInstructions, but nothing in the
      // testsuite triggers it. We should see if it can be removed in both
      // places, or alternately, add a test to exercise it.
      auto *L = CFI.getLabel();
      if (L && !L->isDefined())
        continue;

      SFrameFRE FRE = FDE.FREs.back();
      if (!handleCFI(FDE, FRE, CFI))
        Valid = false;

      // If nothing relevant but the location changed, don't add the FRE.
      if (equalIgnoringLocation(FRE, FDE.FREs.back()))
        continue;

      // If the location stayed the same, then update the current
      // row. Otherwise, add a new one.
      if (atSameLocation(LastLabel, L))
        FDE.FREs.back() = FRE;
      else {
        FDE.FREs.push_back(FRE);
        FDE.FREs.back().Label = L;
        LastLabel = L;
      }
    }
    if (Valid)
      FDEs.push_back(FDE);
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
    uint32_t TotalFREs = 0;
    Streamer.emitInt32(TotalFREs);

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
    Emitter.buildSFDE(DFrame);

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
