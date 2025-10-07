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
#include "llvm/Support/Endian.h"
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
  FREInfo<endianness::native> Info;
  bool CFARegSet = false;

  SFrameFRE(const MCSymbol *Start) : Label(Start) { Info.Info = 0; }

  void emitOffset(MCObjectStreamer &S, FREOffset OffsetSize, size_t Offset) {
    switch (OffsetSize) {
    case (FREOffset::B1):
      S.emitInt8(Offset);
      return;
    case (FREOffset::B2):
      S.emitInt16(Offset);
      return;
    case (FREOffset::B4):
      S.emitInt32(Offset);
      return;
    }
  }

  void emit(MCObjectStreamer &S, const MCSymbol *FuncBegin,
            MCFragment *FDEFrag) {
    S.emitSFrameCalculateFuncOffset(FuncBegin, Label, FDEFrag, SMLoc());

    // fre_cfa_base_reg_id already set during parsing

    // fre_offset_count
    unsigned RegsTracked = 1; // always track the cfa.
    if (FPOffset != 0)
      ++RegsTracked;
    if (RAOffset != 0)
      ++RegsTracked;
    Info.setOffsetCount(RegsTracked);

    // fre_offset_size
    if (isInt<8>(CFAOffset) && isInt<8>(FPOffset) && isInt<8>(RAOffset))
      Info.setOffsetSize(FREOffset::B1);
    else if (isInt<16>(CFAOffset) && isInt<16>(FPOffset) && isInt<16>(RAOffset))
      Info.setOffsetSize(FREOffset::B2);
    else {
      assert(isInt<32>(CFAOffset) && isInt<32>(FPOffset) &&
             isInt<32>(RAOffset) && "Offset too big for sframe");
      Info.setOffsetSize(FREOffset::B4);
    }

    // No support for fre_mangled_ra_p yet.
    Info.setReturnAddressSigned(false);

    // sframe_fre_info_word
    S.emitInt8(Info.getFREInfo());

    // FRE Offsets
    [[maybe_unused]] unsigned OffsetsEmitted = 1;
    emitOffset(S, Info.getOffsetSize(), CFAOffset);
    if (FPOffset) {
      ++OffsetsEmitted;
      emitOffset(S, Info.getOffsetSize(), FPOffset);
    }
    if (RAOffset) {
      ++OffsetsEmitted;
      emitOffset(S, Info.getOffsetSize(), RAOffset);
    }
    assert(OffsetsEmitted == RegsTracked &&
           "Didn't emit the right number of offsets");
  }
};

// High-level structure to track info needed to emit a sframe_func_desc_entry
// and its associated FREs.
struct SFrameFDE {
  // Reference to the original dwarf frame to avoid copying.
  const MCDwarfFrameInfo &DFrame;
  // Label where this FDE's FREs start.
  MCSymbol *FREStart;
  // Frag where this FDE is emitted.
  MCFragment *Frag;
  // Unwinding fres
  SmallVector<SFrameFRE> FREs;
  // .cfi_remember_state stack
  SmallVector<SFrameFRE> SaveState;

  SFrameFDE(const MCDwarfFrameInfo &DF, MCSymbol *FRES)
      : DFrame(DF), FREStart(FRES), Frag(nullptr) {}

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
    S.emitInt32(FREs.size());

    // sfde_func_info word

    // All FREs within an FDE share the same sframe::FREType::AddrX. The value
    // of 'X' is determined by the FRE with the largest offset, which is the
    // last. This offset isn't known until relax time, so emit a frag which can
    // calculate that now.
    //
    // At relax time, this FDE frag calculates the proper AddrX value (as well
    // as the rest of the FDE FuncInfo word). Subsequent FRE frags will read it
    // from this frag and emit the proper number of bytes.
    Frag = S.getCurrentFragment();
    S.emitSFrameCalculateFuncOffset(DFrame.Begin, FREs.back().Label, nullptr,
                                    SMLoc());

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
  uint32_t TotalFREs;
  ABI SFrameABI;
  // Target-specific convenience variables to detect when a CFI instruction
  // references these registers. Unlike in dwarf frame descriptions, they never
  // escape into the sframe section itself. TODO: These should be retrieved from
  // the target.
  unsigned SPReg;
  unsigned FPReg;
  unsigned RAReg;
  int8_t FixedRAOffset;
  MCSymbol *FDESubSectionStart;
  MCSymbol *FRESubSectionStart;
  MCSymbol *FRESubSectionEnd;

  bool setCFARegister(SFrameFRE &FRE, const MCCFIInstruction &I) {
    if (I.getRegister() == SPReg) {
      FRE.CFARegSet = true;
      FRE.Info.setBaseRegister(BaseReg::SP);
      return true;
    }
    if (I.getRegister() == FPReg) {
      FRE.CFARegSet = true;
      FRE.Info.setBaseRegister(BaseReg::FP);
      return true;
    }
    Streamer.getContext().reportWarning(
        I.getLoc(), "canonical Frame Address not in stack- or frame-pointer. "
                    "Omitting SFrame unwind info for this function");
    return false;
  }

  bool setCFAOffset(SFrameFRE &FRE, SMLoc Loc, size_t Offset) {
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
      if (FDE.FREs.size() == 1) {
        // Error for gas compatibility: If the initial FRE isn't complete,
        // then any state is incomplete.  FIXME: Dwarf doesn't error here.
        // Why should sframe?
        Streamer.getContext().reportWarning(
            CFI.getLoc(), "skipping SFrame FDE; .cfi_remember_state without "
                          "prior SFrame FRE state");
        return false;
      }
      FDE.SaveState.push_back(FRE);
      return true;
    case MCCFIInstruction::OpRestore:
      // The first FRE generated has the original state.
      if (CFI.getRegister() == FPReg)
        FRE.FPOffset = FDE.FREs.front().FPOffset;
      else if (CFI.getRegister() == RAReg)
        FRE.RAOffset = FDE.FREs.front().RAOffset;
      return true;
    case MCCFIInstruction::OpRestoreState:
      // The cfi parser will have caught unbalanced directives earlier, so a
      // mismatch here is an implementation error.
      assert(!FDE.SaveState.empty() &&
             "cfi_restore_state without cfi_save_state");
      FRE = FDE.SaveState.pop_back_val();
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
  SFrameEmitterImpl(MCObjectStreamer &Streamer)
      : Streamer(Streamer), TotalFREs(0) {
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
      FixedRAOffset = 0;
      break;
    case ABI::AMD64EndianLittle:
      SPReg = 7;
      // RARegister untracked in this abi. Value chosen to match
      // MCDwarfFrameInfo constructor.
      RAReg = static_cast<unsigned>(INT_MAX);
      FPReg = 6;
      FixedRAOffset = -8;
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
           Left.Info.getFREInfo() == Right.Info.getFREInfo() &&
           Left.CFARegSet == Right.CFARegSet;
  }

  void buildSFDE(const MCDwarfFrameInfo &DF) {
    // Functions with zero size can happen with assembler macros and
    // machine-generated code. They don't need unwind info at all, so
    // no need to warn.
    if (atSameLocation(DF.Begin, DF.End))
      return;
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

    if (Valid) {
      FDEs.push_back(FDE);
      TotalFREs += FDE.FREs.size();
    }
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
    Streamer.emitInt8(FixedRAOffset);
    // sfh_auxhdr_len
    Streamer.emitInt8(0);
    // shf_num_fdes
    Streamer.emitInt32(FDEs.size());
    // shf_num_fres
    Streamer.emitInt32(TotalFREs);

    // shf_fre_len
    Streamer.emitAbsoluteSymbolDiff(FRESubSectionEnd, FRESubSectionStart,
                                    sizeof(int32_t));
    // shf_fdeoff. With no sfh_auxhdr, these immediately follow this header.
    Streamer.emitInt32(0);
    // shf_freoff
    Streamer.emitInt32(FDEs.size() *
                       sizeof(sframe::FuncDescEntry<endianness::native>));
  }

  void emitFDEs() {
    Streamer.emitLabel(FDESubSectionStart);
    for (auto &FDE : FDEs) {
      FDE.emit(Streamer, FRESubSectionStart);
    }
  }

  void emitFREs() {
    Streamer.emitLabel(FRESubSectionStart);
    for (auto &FDE : FDEs) {
      Streamer.emitLabel(FDE.FREStart);
      for (auto &FRE : FDE.FREs)
        FRE.emit(Streamer, FDE.DFrame.Begin, FDE.Frag);
    }
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

void MCSFrameEmitter::encodeFuncOffset(MCContext &C, uint64_t Offset,
                                       SmallVectorImpl<char> &Out,
                                       MCFragment *FDEFrag) {
  // If encoding into the FDE Frag itself, generate the sfde_func_info.
  if (FDEFrag == nullptr) {
    // sfde_func_info

    // Offset is the difference between the function start label and the final
    // FRE's offset, which is the max offset for this FDE.
    FDEInfo<endianness::native> I;
    I.Info = 0;
    if (isUInt<8>(Offset))
      I.setFREType(FREType::Addr1);
    else if (isUInt<16>(Offset))
      I.setFREType(FREType::Addr2);
    else {
      assert(isUInt<32>(Offset));
      I.setFREType(FREType::Addr4);
    }
    I.setFDEType(FDEType::PCInc);
    // TODO: When we support pauth keys, this will need to be retrieved
    // from the frag itself.
    I.setPAuthKey(0);

    Out.push_back(I.getFuncInfo());
    return;
  }

  const auto &FDEData = FDEFrag->getVarContents();
  FDEInfo<endianness::native> I;
  I.Info = FDEData.back();
  FREType T = I.getFREType();
  llvm::endianness E = C.getAsmInfo()->isLittleEndian()
                           ? llvm::endianness::little
                           : llvm::endianness::big;
  // sfre_start_address
  switch (T) {
  case FREType::Addr1:
    assert(isUInt<8>(Offset) && "Miscalculated Sframe FREType");
    support::endian::write<uint8_t>(Out, Offset, E);
    break;
  case FREType::Addr2:
    assert(isUInt<16>(Offset) && "Miscalculated Sframe FREType");
    support::endian::write<uint16_t>(Out, Offset, E);
    break;
  case FREType::Addr4:
    assert(isUInt<32>(Offset) && "Miscalculated Sframe FREType");
    support::endian::write<uint32_t>(Out, Offset, E);
    break;
  }
}
