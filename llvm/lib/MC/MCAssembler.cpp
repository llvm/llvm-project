//===- lib/MC/MCAssembler.cpp - Assembler Backend Implementation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAssembler.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCCodeView.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <tuple>
#include <utility>

using namespace llvm;

namespace llvm {
class MCSubtargetInfo;
}

#define DEBUG_TYPE "assembler"

namespace {
namespace stats {

STATISTIC(EmittedFragments, "Number of emitted assembler fragments - total");
STATISTIC(EmittedRelaxableFragments,
          "Number of emitted assembler fragments - relaxable");
STATISTIC(EmittedDataFragments,
          "Number of emitted assembler fragments - data");
STATISTIC(EmittedAlignFragments,
          "Number of emitted assembler fragments - align");
STATISTIC(EmittedFillFragments,
          "Number of emitted assembler fragments - fill");
STATISTIC(EmittedNopsFragments, "Number of emitted assembler fragments - nops");
STATISTIC(EmittedOrgFragments, "Number of emitted assembler fragments - org");
STATISTIC(evaluateFixup, "Number of evaluated fixups");
STATISTIC(ObjectBytes, "Number of emitted object file bytes");
STATISTIC(RelaxationSteps, "Number of assembler layout and relaxation steps");
STATISTIC(RelaxedInstructions, "Number of relaxed instructions");

} // end namespace stats
} // end anonymous namespace

// FIXME FIXME FIXME: There are number of places in this file where we convert
// what is a 64-bit assembler value used for computation into a value in the
// object file, which may truncate it. We should detect that truncation where
// invalid and report errors back.

/* *** */

MCAssembler::MCAssembler(MCContext &Context,
                         std::unique_ptr<MCAsmBackend> Backend,
                         std::unique_ptr<MCCodeEmitter> Emitter,
                         std::unique_ptr<MCObjectWriter> Writer)
    : Context(Context), Backend(std::move(Backend)),
      Emitter(std::move(Emitter)), Writer(std::move(Writer)) {
  if (this->Backend)
    this->Backend->setAssembler(this);
  if (this->Writer)
    this->Writer->setAssembler(this);
}

void MCAssembler::reset() {
  HasLayout = false;
  HasFinalLayout = false;
  RelaxAll = false;
  Sections.clear();
  Symbols.clear();
  ThumbFuncs.clear();
  BundleAlignSize = 0;

  // reset objects owned by us
  if (getBackendPtr())
    getBackendPtr()->reset();
  if (getEmitterPtr())
    getEmitterPtr()->reset();
  if (Writer)
    Writer->reset();
}

bool MCAssembler::registerSection(MCSection &Section) {
  if (Section.isRegistered())
    return false;
  assert(Section.curFragList()->Head && "allocInitialFragment not called");
  Sections.push_back(&Section);
  Section.setIsRegistered(true);
  return true;
}

bool MCAssembler::isThumbFunc(const MCSymbol *Symbol) const {
  if (ThumbFuncs.count(Symbol))
    return true;

  if (!Symbol->isVariable())
    return false;

  const MCExpr *Expr = Symbol->getVariableValue();

  MCValue V;
  if (!Expr->evaluateAsRelocatable(V, nullptr))
    return false;

  if (V.getSubSym() || V.getSpecifier())
    return false;

  auto *Sym = V.getAddSym();
  if (!Sym || V.getSpecifier())
    return false;

  if (!isThumbFunc(Sym))
    return false;

  ThumbFuncs.insert(Symbol); // Cache it.
  return true;
}

bool MCAssembler::evaluateFixup(const MCFragment &F, MCFixup &Fixup,
                                MCValue &Target, uint64_t &Value,
                                bool RecordReloc,
                                MutableArrayRef<char> Contents) const {
  ++stats::evaluateFixup;

  // FIXME: This code has some duplication with recordRelocation. We should
  // probably merge the two into a single callback that tries to evaluate a
  // fixup and records a relocation if one is needed.

  // On error claim to have completely evaluated the fixup, to prevent any
  // further processing from being done.
  const MCExpr *Expr = Fixup.getValue();
  Value = 0;
  if (!Expr->evaluateAsRelocatable(Target, this)) {
    reportError(Fixup.getLoc(), "expected relocatable expression");
    return true;
  }

  bool IsResolved = false;
  unsigned FixupFlags = getBackend().getFixupKindInfo(Fixup.getKind()).Flags;
  bool IsPCRel = FixupFlags & MCFixupKindInfo::FKF_IsPCRel;
  if (FixupFlags & MCFixupKindInfo::FKF_IsTarget) {
    IsResolved = getBackend().evaluateTargetFixup(Fixup, Target, Value);
  } else {
    const MCSymbol *Add = Target.getAddSym();
    const MCSymbol *Sub = Target.getSubSym();
    Value = Target.getConstant();
    if (Add && Add->isDefined())
      Value += getSymbolOffset(*Add);
    if (Sub && Sub->isDefined())
      Value -= getSymbolOffset(*Sub);

    bool ShouldAlignPC =
        FixupFlags & MCFixupKindInfo::FKF_IsAlignedDownTo32Bits;
    if (IsPCRel) {
      uint64_t Offset = getFragmentOffset(F) + Fixup.getOffset();

      // A number of ARM fixups in Thumb mode require that the effective PC
      // address be determined as the 32-bit aligned version of the actual
      // offset.
      if (ShouldAlignPC)
        Offset &= ~0x3;
      Value -= Offset;

      if (Add && !Sub && !Add->isUndefined() && !Add->isAbsolute()) {
        IsResolved = getWriter().isSymbolRefDifferenceFullyResolvedImpl(
            *Add, F, false, true);
      }
    } else {
      IsResolved = Target.isAbsolute();
      assert(!ShouldAlignPC && "FKF_IsAlignedDownTo32Bits must be PC-relative");
    }
  }

  if (!RecordReloc)
    return IsResolved;

  if (IsResolved && mc::isRelocRelocation(Fixup.getKind()))
    IsResolved = false;
  if (IsPCRel)
    Fixup.setPCRel();
  getBackend().applyFixup(F, Fixup, Target, Contents, Value, IsResolved);
  return true;
}

uint64_t MCAssembler::computeFragmentSize(const MCFragment &F) const {
  assert(getBackendPtr() && "Requires assembler backend");
  switch (F.getKind()) {
  case MCFragment::FT_Data:
    return cast<MCDataFragment>(F).getContents().size();
  case MCFragment::FT_Relaxable:
    return cast<MCRelaxableFragment>(F).getContents().size();
  case MCFragment::FT_Fill: {
    auto &FF = cast<MCFillFragment>(F);
    int64_t NumValues = 0;
    if (!FF.getNumValues().evaluateKnownAbsolute(NumValues, *this)) {
      recordError(FF.getLoc(), "expected assembly-time absolute expression");
      return 0;
    }
    int64_t Size = NumValues * FF.getValueSize();
    if (Size < 0) {
      recordError(FF.getLoc(), "invalid number of bytes");
      return 0;
    }
    return Size;
  }

  case MCFragment::FT_Nops:
    return cast<MCNopsFragment>(F).getNumBytes();

  case MCFragment::FT_LEB:
    return cast<MCLEBFragment>(F).getContents().size();

  case MCFragment::FT_BoundaryAlign:
    return cast<MCBoundaryAlignFragment>(F).getSize();

  case MCFragment::FT_SymbolId:
    return 4;

  case MCFragment::FT_Align: {
    const MCAlignFragment &AF = cast<MCAlignFragment>(F);
    unsigned Offset = getFragmentOffset(AF);
    unsigned Size = offsetToAlignment(Offset, AF.getAlignment());

    // Insert extra Nops for code alignment if the target define
    // shouldInsertExtraNopBytesForCodeAlign target hook.
    if (AF.getParent()->useCodeAlign() && AF.hasEmitNops() &&
        getBackend().shouldInsertExtraNopBytesForCodeAlign(AF, Size))
      return Size;

    // If we are padding with nops, force the padding to be larger than the
    // minimum nop size.
    if (Size > 0 && AF.hasEmitNops()) {
      while (Size % getBackend().getMinimumNopSize())
        Size += AF.getAlignment().value();
    }
    if (Size > AF.getMaxBytesToEmit())
      return 0;
    return Size;
  }

  case MCFragment::FT_Org: {
    const MCOrgFragment &OF = cast<MCOrgFragment>(F);
    MCValue Value;
    if (!OF.getOffset().evaluateAsValue(Value, *this)) {
      recordError(OF.getLoc(), "expected assembly-time absolute expression");
      return 0;
    }

    uint64_t FragmentOffset = getFragmentOffset(OF);
    int64_t TargetLocation = Value.getConstant();
    if (const auto *SA = Value.getAddSym()) {
      uint64_t Val;
      if (!getSymbolOffset(*SA, Val)) {
        recordError(OF.getLoc(), "expected absolute expression");
        return 0;
      }
      TargetLocation += Val;
    }
    int64_t Size = TargetLocation - FragmentOffset;
    if (Size < 0 || Size >= 0x40000000) {
      recordError(OF.getLoc(), "invalid .org offset '" + Twine(TargetLocation) +
                                   "' (at offset '" + Twine(FragmentOffset) +
                                   "')");
      return 0;
    }
    return Size;
  }

  case MCFragment::FT_Dwarf:
    return cast<MCDwarfLineAddrFragment>(F).getContents().size();
  case MCFragment::FT_DwarfFrame:
    return cast<MCDwarfCallFrameFragment>(F).getContents().size();
  case MCFragment::FT_CVInlineLines:
    return cast<MCCVInlineLineTableFragment>(F).getContents().size();
  case MCFragment::FT_CVDefRange:
    return cast<MCCVDefRangeFragment>(F).getContents().size();
  case MCFragment::FT_PseudoProbe:
    return cast<MCPseudoProbeAddrFragment>(F).getContents().size();
  }

  llvm_unreachable("invalid fragment kind");
}

// Compute the amount of padding required before the fragment \p F to
// obey bundling restrictions, where \p FOffset is the fragment's offset in
// its section and \p FSize is the fragment's size.
static uint64_t computeBundlePadding(unsigned BundleSize,
                                     const MCEncodedFragment *F,
                                     uint64_t FOffset, uint64_t FSize) {
  uint64_t OffsetInBundle = FOffset & (BundleSize - 1);
  uint64_t EndOfFragment = OffsetInBundle + FSize;

  // There are two kinds of bundling restrictions:
  //
  // 1) For alignToBundleEnd(), add padding to ensure that the fragment will
  //    *end* on a bundle boundary.
  // 2) Otherwise, check if the fragment would cross a bundle boundary. If it
  //    would, add padding until the end of the bundle so that the fragment
  //    will start in a new one.
  if (F->alignToBundleEnd()) {
    // Three possibilities here:
    //
    // A) The fragment just happens to end at a bundle boundary, so we're good.
    // B) The fragment ends before the current bundle boundary: pad it just
    //    enough to reach the boundary.
    // C) The fragment ends after the current bundle boundary: pad it until it
    //    reaches the end of the next bundle boundary.
    //
    // Note: this code could be made shorter with some modulo trickery, but it's
    // intentionally kept in its more explicit form for simplicity.
    if (EndOfFragment == BundleSize)
      return 0;
    else if (EndOfFragment < BundleSize)
      return BundleSize - EndOfFragment;
    else { // EndOfFragment > BundleSize
      return 2 * BundleSize - EndOfFragment;
    }
  } else if (OffsetInBundle > 0 && EndOfFragment > BundleSize)
    return BundleSize - OffsetInBundle;
  else
    return 0;
}

void MCAssembler::layoutBundle(MCFragment *Prev, MCFragment *F) const {
  // If bundling is enabled and this fragment has instructions in it, it has to
  // obey the bundling restrictions. With padding, we'll have:
  //
  //
  //        BundlePadding
  //             |||
  // -------------------------------------
  //   Prev  |##########|       F        |
  // -------------------------------------
  //                    ^
  //                    |
  //                    F->Offset
  //
  // The fragment's offset will point to after the padding, and its computed
  // size won't include the padding.
  //
  // ".align N" is an example of a directive that introduces multiple
  // fragments. We could add a special case to handle ".align N" by emitting
  // within-fragment padding (which would produce less padding when N is less
  // than the bundle size), but for now we don't.
  //
  assert(isa<MCEncodedFragment>(F) &&
         "Only MCEncodedFragment implementations have instructions");
  MCEncodedFragment *EF = cast<MCEncodedFragment>(F);
  uint64_t FSize = computeFragmentSize(*EF);

  if (FSize > getBundleAlignSize())
    report_fatal_error("Fragment can't be larger than a bundle size");

  uint64_t RequiredBundlePadding =
      computeBundlePadding(getBundleAlignSize(), EF, EF->Offset, FSize);
  if (RequiredBundlePadding > UINT8_MAX)
    report_fatal_error("Padding cannot exceed 255 bytes");
  EF->setBundlePadding(static_cast<uint8_t>(RequiredBundlePadding));
  EF->Offset += RequiredBundlePadding;
  if (auto *DF = dyn_cast_or_null<MCDataFragment>(Prev))
    if (DF->getContents().empty())
      DF->Offset = EF->Offset;
}

// Simple getSymbolOffset helper for the non-variable case.
static bool getLabelOffset(const MCAssembler &Asm, const MCSymbol &S,
                           bool ReportError, uint64_t &Val) {
  if (!S.getFragment()) {
    if (ReportError)
      reportFatalUsageError("cannot evaluate undefined symbol '" + S.getName() +
                            "'");
    return false;
  }
  Val = Asm.getFragmentOffset(*S.getFragment()) + S.getOffset();
  return true;
}

static bool getSymbolOffsetImpl(const MCAssembler &Asm, const MCSymbol &S,
                                bool ReportError, uint64_t &Val) {
  if (!S.isVariable())
    return getLabelOffset(Asm, S, ReportError, Val);

  // If SD is a variable, evaluate it.
  MCValue Target;
  if (!S.getVariableValue()->evaluateAsValue(Target, Asm))
    reportFatalUsageError("cannot evaluate equated symbol '" + S.getName() +
                          "'");

  uint64_t Offset = Target.getConstant();

  const MCSymbol *A = Target.getAddSym();
  if (A) {
    uint64_t ValA;
    // FIXME: On most platforms, `Target`'s component symbols are labels from
    // having been simplified during evaluation, but on Mach-O they can be
    // variables due to PR19203. This, and the line below for `B` can be
    // restored to call `getLabelOffset` when PR19203 is fixed.
    if (!getSymbolOffsetImpl(Asm, *A, ReportError, ValA))
      return false;
    Offset += ValA;
  }

  const MCSymbol *B = Target.getSubSym();
  if (B) {
    uint64_t ValB;
    if (!getSymbolOffsetImpl(Asm, *B, ReportError, ValB))
      return false;
    Offset -= ValB;
  }

  Val = Offset;
  return true;
}

bool MCAssembler::getSymbolOffset(const MCSymbol &S, uint64_t &Val) const {
  return getSymbolOffsetImpl(*this, S, false, Val);
}

uint64_t MCAssembler::getSymbolOffset(const MCSymbol &S) const {
  uint64_t Val;
  getSymbolOffsetImpl(*this, S, true, Val);
  return Val;
}

const MCSymbol *MCAssembler::getBaseSymbol(const MCSymbol &Symbol) const {
  assert(HasLayout);
  if (!Symbol.isVariable())
    return &Symbol;

  const MCExpr *Expr = Symbol.getVariableValue();
  MCValue Value;
  if (!Expr->evaluateAsValue(Value, *this)) {
    reportError(Expr->getLoc(), "expression could not be evaluated");
    return nullptr;
  }

  const MCSymbol *SymB = Value.getSubSym();
  if (SymB) {
    reportError(Expr->getLoc(),
                Twine("symbol '") + SymB->getName() +
                    "' could not be evaluated in a subtraction expression");
    return nullptr;
  }

  const MCSymbol *A = Value.getAddSym();
  if (!A)
    return nullptr;

  const MCSymbol &ASym = *A;
  if (ASym.isCommon()) {
    reportError(Expr->getLoc(), "Common symbol '" + ASym.getName() +
                                    "' cannot be used in assignment expr");
    return nullptr;
  }

  return &ASym;
}

uint64_t MCAssembler::getSectionAddressSize(const MCSection &Sec) const {
  assert(HasLayout);
  // The size is the last fragment's end offset.
  const MCFragment &F = *Sec.curFragList()->Tail;
  return getFragmentOffset(F) + computeFragmentSize(F);
}

uint64_t MCAssembler::getSectionFileSize(const MCSection &Sec) const {
  // Virtual sections have no file size.
  if (Sec.isVirtualSection())
    return 0;
  return getSectionAddressSize(Sec);
}

bool MCAssembler::registerSymbol(const MCSymbol &Symbol) {
  bool Changed = !Symbol.isRegistered();
  if (Changed) {
    Symbol.setIsRegistered(true);
    Symbols.push_back(&Symbol);
  }
  return Changed;
}

void MCAssembler::writeFragmentPadding(raw_ostream &OS,
                                       const MCEncodedFragment &EF,
                                       uint64_t FSize) const {
  assert(getBackendPtr() && "Expected assembler backend");
  // Should NOP padding be written out before this fragment?
  unsigned BundlePadding = EF.getBundlePadding();
  if (BundlePadding > 0) {
    assert(isBundlingEnabled() &&
           "Writing bundle padding with disabled bundling");
    assert(EF.hasInstructions() &&
           "Writing bundle padding for a fragment without instructions");

    unsigned TotalLength = BundlePadding + static_cast<unsigned>(FSize);
    const MCSubtargetInfo *STI = EF.getSubtargetInfo();
    if (EF.alignToBundleEnd() && TotalLength > getBundleAlignSize()) {
      // If the padding itself crosses a bundle boundary, it must be emitted
      // in 2 pieces, since even nop instructions must not cross boundaries.
      //             v--------------v   <- BundleAlignSize
      //        v---------v             <- BundlePadding
      // ----------------------------
      // | Prev |####|####|    F    |
      // ----------------------------
      //        ^-------------------^   <- TotalLength
      unsigned DistanceToBoundary = TotalLength - getBundleAlignSize();
      if (!getBackend().writeNopData(OS, DistanceToBoundary, STI))
        report_fatal_error("unable to write NOP sequence of " +
                           Twine(DistanceToBoundary) + " bytes");
      BundlePadding -= DistanceToBoundary;
    }
    if (!getBackend().writeNopData(OS, BundlePadding, STI))
      report_fatal_error("unable to write NOP sequence of " +
                         Twine(BundlePadding) + " bytes");
  }
}

/// Write the fragment \p F to the output file.
static void writeFragment(raw_ostream &OS, const MCAssembler &Asm,
                          const MCFragment &F) {
  // FIXME: Embed in fragments instead?
  uint64_t FragmentSize = Asm.computeFragmentSize(F);

  llvm::endianness Endian = Asm.getBackend().Endian;

  if (const MCEncodedFragment *EF = dyn_cast<MCEncodedFragment>(&F))
    Asm.writeFragmentPadding(OS, *EF, FragmentSize);

  // This variable (and its dummy usage) is to participate in the assert at
  // the end of the function.
  uint64_t Start = OS.tell();
  (void) Start;

  ++stats::EmittedFragments;

  switch (F.getKind()) {
  case MCFragment::FT_Align: {
    ++stats::EmittedAlignFragments;
    const MCAlignFragment &AF = cast<MCAlignFragment>(F);
    assert(AF.getValueSize() && "Invalid virtual align in concrete fragment!");

    uint64_t Count = FragmentSize / AF.getValueSize();

    // FIXME: This error shouldn't actually occur (the front end should emit
    // multiple .align directives to enforce the semantics it wants), but is
    // severe enough that we want to report it. How to handle this?
    if (Count * AF.getValueSize() != FragmentSize)
      report_fatal_error("undefined .align directive, value size '" +
                        Twine(AF.getValueSize()) +
                        "' is not a divisor of padding size '" +
                        Twine(FragmentSize) + "'");

    // See if we are aligning with nops, and if so do that first to try to fill
    // the Count bytes.  Then if that did not fill any bytes or there are any
    // bytes left to fill use the Value and ValueSize to fill the rest.
    // If we are aligning with nops, ask that target to emit the right data.
    if (AF.hasEmitNops()) {
      if (!Asm.getBackend().writeNopData(OS, Count, AF.getSubtargetInfo()))
        report_fatal_error("unable to write nop sequence of " +
                          Twine(Count) + " bytes");
      break;
    }

    // Otherwise, write out in multiples of the value size.
    for (uint64_t i = 0; i != Count; ++i) {
      switch (AF.getValueSize()) {
      default: llvm_unreachable("Invalid size!");
      case 1: OS << char(AF.getValue()); break;
      case 2:
        support::endian::write<uint16_t>(OS, AF.getValue(), Endian);
        break;
      case 4:
        support::endian::write<uint32_t>(OS, AF.getValue(), Endian);
        break;
      case 8:
        support::endian::write<uint64_t>(OS, AF.getValue(), Endian);
        break;
      }
    }
    break;
  }

  case MCFragment::FT_Data:
    ++stats::EmittedDataFragments;
    OS << StringRef(cast<MCDataFragment>(F).getContents().data(),
                    cast<MCDataFragment>(F).getContents().size());
    break;

  case MCFragment::FT_Relaxable:
    ++stats::EmittedRelaxableFragments;
    OS << StringRef(cast<MCRelaxableFragment>(F).getContents().data(),
                    cast<MCRelaxableFragment>(F).getContents().size());
    break;

  case MCFragment::FT_Fill: {
    ++stats::EmittedFillFragments;
    const MCFillFragment &FF = cast<MCFillFragment>(F);
    uint64_t V = FF.getValue();
    unsigned VSize = FF.getValueSize();
    const unsigned MaxChunkSize = 16;
    char Data[MaxChunkSize];
    assert(0 < VSize && VSize <= MaxChunkSize && "Illegal fragment fill size");
    // Duplicate V into Data as byte vector to reduce number of
    // writes done. As such, do endian conversion here.
    for (unsigned I = 0; I != VSize; ++I) {
      unsigned index = Endian == llvm::endianness::little ? I : (VSize - I - 1);
      Data[I] = uint8_t(V >> (index * 8));
    }
    for (unsigned I = VSize; I < MaxChunkSize; ++I)
      Data[I] = Data[I - VSize];

    // Set to largest multiple of VSize in Data.
    const unsigned NumPerChunk = MaxChunkSize / VSize;
    // Set ChunkSize to largest multiple of VSize in Data
    const unsigned ChunkSize = VSize * NumPerChunk;

    // Do copies by chunk.
    StringRef Ref(Data, ChunkSize);
    for (uint64_t I = 0, E = FragmentSize / ChunkSize; I != E; ++I)
      OS << Ref;

    // do remainder if needed.
    unsigned TrailingCount = FragmentSize % ChunkSize;
    if (TrailingCount)
      OS.write(Data, TrailingCount);
    break;
  }

  case MCFragment::FT_Nops: {
    ++stats::EmittedNopsFragments;
    const MCNopsFragment &NF = cast<MCNopsFragment>(F);

    int64_t NumBytes = NF.getNumBytes();
    int64_t ControlledNopLength = NF.getControlledNopLength();
    int64_t MaximumNopLength =
        Asm.getBackend().getMaximumNopSize(*NF.getSubtargetInfo());

    assert(NumBytes > 0 && "Expected positive NOPs fragment size");
    assert(ControlledNopLength >= 0 && "Expected non-negative NOP size");

    if (ControlledNopLength > MaximumNopLength) {
      Asm.reportError(NF.getLoc(), "illegal NOP size " +
                                       std::to_string(ControlledNopLength) +
                                       ". (expected within [0, " +
                                       std::to_string(MaximumNopLength) + "])");
      // Clamp the NOP length as reportError does not stop the execution
      // immediately.
      ControlledNopLength = MaximumNopLength;
    }

    // Use maximum value if the size of each NOP is not specified
    if (!ControlledNopLength)
      ControlledNopLength = MaximumNopLength;

    while (NumBytes) {
      uint64_t NumBytesToEmit =
          (uint64_t)std::min(NumBytes, ControlledNopLength);
      assert(NumBytesToEmit && "try to emit empty NOP instruction");
      if (!Asm.getBackend().writeNopData(OS, NumBytesToEmit,
                                         NF.getSubtargetInfo())) {
        report_fatal_error("unable to write nop sequence of the remaining " +
                           Twine(NumBytesToEmit) + " bytes");
        break;
      }
      NumBytes -= NumBytesToEmit;
    }
    break;
  }

  case MCFragment::FT_LEB: {
    const MCLEBFragment &LF = cast<MCLEBFragment>(F);
    OS << StringRef(LF.getContents().data(), LF.getContents().size());
    break;
  }

  case MCFragment::FT_BoundaryAlign: {
    const MCBoundaryAlignFragment &BF = cast<MCBoundaryAlignFragment>(F);
    if (!Asm.getBackend().writeNopData(OS, FragmentSize, BF.getSubtargetInfo()))
      report_fatal_error("unable to write nop sequence of " +
                         Twine(FragmentSize) + " bytes");
    break;
  }

  case MCFragment::FT_SymbolId: {
    const MCSymbolIdFragment &SF = cast<MCSymbolIdFragment>(F);
    support::endian::write<uint32_t>(OS, SF.getSymbol()->getIndex(), Endian);
    break;
  }

  case MCFragment::FT_Org: {
    ++stats::EmittedOrgFragments;
    const MCOrgFragment &OF = cast<MCOrgFragment>(F);

    for (uint64_t i = 0, e = FragmentSize; i != e; ++i)
      OS << char(OF.getValue());

    break;
  }

  case MCFragment::FT_Dwarf: {
    const MCDwarfLineAddrFragment &OF = cast<MCDwarfLineAddrFragment>(F);
    OS << StringRef(OF.getContents().data(), OF.getContents().size());
    break;
  }
  case MCFragment::FT_DwarfFrame: {
    const MCDwarfCallFrameFragment &CF = cast<MCDwarfCallFrameFragment>(F);
    OS << StringRef(CF.getContents().data(), CF.getContents().size());
    break;
  }
  case MCFragment::FT_CVInlineLines: {
    const auto &OF = cast<MCCVInlineLineTableFragment>(F);
    OS << StringRef(OF.getContents().data(), OF.getContents().size());
    break;
  }
  case MCFragment::FT_CVDefRange: {
    const auto &DRF = cast<MCCVDefRangeFragment>(F);
    OS << StringRef(DRF.getContents().data(), DRF.getContents().size());
    break;
  }
  case MCFragment::FT_PseudoProbe: {
    const MCPseudoProbeAddrFragment &PF = cast<MCPseudoProbeAddrFragment>(F);
    OS << StringRef(PF.getContents().data(), PF.getContents().size());
    break;
  }
  }

  assert(OS.tell() - Start == FragmentSize &&
         "The stream should advance by fragment size");
}

void MCAssembler::writeSectionData(raw_ostream &OS,
                                   const MCSection *Sec) const {
  assert(getBackendPtr() && "Expected assembler backend");

  // Ignore virtual sections.
  if (Sec->isVirtualSection()) {
    assert(getSectionFileSize(*Sec) == 0 && "Invalid size for section!");

    // Check that contents are only things legal inside a virtual section.
    for (const MCFragment &F : *Sec) {
      switch (F.getKind()) {
      default: llvm_unreachable("Invalid fragment in virtual section!");
      case MCFragment::FT_Data: {
        // Check that we aren't trying to write a non-zero contents (or fixups)
        // into a virtual section. This is to support clients which use standard
        // directives to fill the contents of virtual sections.
        const MCDataFragment &DF = cast<MCDataFragment>(F);
        if (DF.getFixups().size())
          reportError(SMLoc(), Sec->getVirtualSectionKind() + " section '" +
                                   Sec->getName() + "' cannot have fixups");
        for (char C : DF.getContents())
          if (C) {
            reportError(SMLoc(), Sec->getVirtualSectionKind() + " section '" +
                                     Sec->getName() +
                                     "' cannot have non-zero initializers");
            break;
          }
        break;
      }
      case MCFragment::FT_Align:
        // Check that we aren't trying to write a non-zero value into a virtual
        // section.
        assert((cast<MCAlignFragment>(F).getValueSize() == 0 ||
                cast<MCAlignFragment>(F).getValue() == 0) &&
               "Invalid align in virtual section!");
        break;
      case MCFragment::FT_Fill:
        assert((cast<MCFillFragment>(F).getValue() == 0) &&
               "Invalid fill in virtual section!");
        break;
      case MCFragment::FT_Org:
        break;
      }
    }

    return;
  }

  uint64_t Start = OS.tell();
  (void)Start;

  for (const MCFragment &F : *Sec)
    writeFragment(OS, *this, F);

  flushPendingErrors();
  assert(getContext().hadError() ||
         OS.tell() - Start == getSectionAddressSize(*Sec));
}

void MCAssembler::layout() {
  assert(getBackendPtr() && "Expected assembler backend");
  DEBUG_WITH_TYPE("mc-dump-pre", {
    errs() << "assembler backend - pre-layout\n--\n";
    dump();
  });

  // Assign section ordinals.
  unsigned SectionIndex = 0;
  for (MCSection &Sec : *this) {
    Sec.setOrdinal(SectionIndex++);

    // Chain together fragments from all subsections.
    if (Sec.Subsections.size() > 1) {
      MCDataFragment Dummy;
      MCFragment *Tail = &Dummy;
      for (auto &[_, List] : Sec.Subsections) {
        assert(List.Head);
        Tail->Next = List.Head;
        Tail = List.Tail;
      }
      Sec.Subsections.clear();
      Sec.Subsections.push_back({0u, {Dummy.getNext(), Tail}});
      Sec.CurFragList = &Sec.Subsections[0].second;

      unsigned FragmentIndex = 0;
      for (MCFragment &Frag : Sec)
        Frag.setLayoutOrder(FragmentIndex++);
    }
  }

  // Layout until everything fits.
  this->HasLayout = true;
  for (MCSection &Sec : *this)
    layoutSection(Sec);
  while (relaxOnce())
    if (getContext().hadError())
      return;

  // Some targets might want to adjust fragment offsets. If so, perform another
  // layout iteration.
  if (getBackend().finishLayout(*this))
    for (MCSection &Sec : *this)
      layoutSection(Sec);

  flushPendingErrors();

  DEBUG_WITH_TYPE("mc-dump", {
      errs() << "assembler backend - final-layout\n--\n";
      dump(); });

  // Allow the object writer a chance to perform post-layout binding (for
  // example, to set the index fields in the symbol data).
  getWriter().executePostLayoutBinding();

  // Fragment sizes are finalized. For RISC-V linker relaxation, this flag
  // helps check whether a PC-relative fixup is fully resolved.
  this->HasFinalLayout = true;

  // Evaluate and apply the fixups, generating relocation entries as necessary.
  for (MCSection &Sec : *this) {
    for (MCFragment &Frag : Sec) {
      // Process fragments with fixups here.
      if (auto *F = dyn_cast<MCEncodedFragment>(&Frag)) {
        auto Contents = F->getContents();
        for (MCFixup &Fixup : F->getFixups()) {
          uint64_t FixedValue;
          MCValue Target;
          evaluateFixup(Frag, Fixup, Target, FixedValue,
                        /*RecordReloc=*/true, Contents);
        }
      } else if (auto *AF = dyn_cast<MCAlignFragment>(&Frag)) {
        // For RISC-V linker relaxation, an alignment relocation might be
        // needed.
        if (AF->hasEmitNops())
          getBackend().shouldInsertFixupForCodeAlign(*this, *AF);
      }
    }
  }
}

void MCAssembler::Finish() {
  layout();

  // Write the object file.
  stats::ObjectBytes += getWriter().writeObject();

  HasLayout = false;
  assert(PendingErrors.empty());
}

bool MCAssembler::fixupNeedsRelaxation(const MCRelaxableFragment &F,
                                       const MCFixup &Fixup) const {
  assert(getBackendPtr() && "Expected assembler backend");
  MCValue Target;
  uint64_t Value;
  bool Resolved = evaluateFixup(F, const_cast<MCFixup &>(Fixup), Target, Value,
                                /*RecordReloc=*/false, {});
  return getBackend().fixupNeedsRelaxationAdvanced(Fixup, Target, Value,
                                                   Resolved);
}

bool MCAssembler::fragmentNeedsRelaxation(const MCRelaxableFragment &F) const {
  assert(getBackendPtr() && "Expected assembler backend");
  // If this inst doesn't ever need relaxation, ignore it. This occurs when we
  // are intentionally pushing out inst fragments, or because we relaxed a
  // previous instruction to one that doesn't need relaxation.
  if (!getBackend().mayNeedRelaxation(F.getInst(), *F.getSubtargetInfo()))
    return false;

  for (const MCFixup &Fixup : F.getFixups())
    if (fixupNeedsRelaxation(F, Fixup))
      return true;

  return false;
}

bool MCAssembler::relaxInstruction(MCRelaxableFragment &F) {
  assert(getEmitterPtr() &&
         "Expected CodeEmitter defined for relaxInstruction");
  if (!fragmentNeedsRelaxation(F))
    return false;

  ++stats::RelaxedInstructions;

  // FIXME-PERF: We could immediately lower out instructions if we can tell
  // they are fully resolved, to avoid retesting on later passes.

  // Relax the fragment.

  MCInst Relaxed = F.getInst();
  getBackend().relaxInstruction(Relaxed, *F.getSubtargetInfo());

  // Encode the new instruction.
  F.setInst(Relaxed);
  SmallVector<char, 16> Data;
  SmallVector<MCFixup, 1> Fixups;
  getEmitter().encodeInstruction(Relaxed, Data, Fixups, *F.getSubtargetInfo());
  F.setContents(Data);
  F.setFixups(Fixups);
  return true;
}

bool MCAssembler::relaxLEB(MCLEBFragment &LF) {
  const unsigned OldSize = static_cast<unsigned>(LF.getContents().size());
  unsigned PadTo = OldSize;
  int64_t Value;
  LF.clearFixups();
  // Use evaluateKnownAbsolute for Mach-O as a hack: .subsections_via_symbols
  // requires that .uleb128 A-B is foldable where A and B reside in different
  // fragments. This is used by __gcc_except_table.
  bool Abs = getWriter().getSubsectionsViaSymbols()
                 ? LF.getValue().evaluateKnownAbsolute(Value, *this)
                 : LF.getValue().evaluateAsAbsolute(Value, *this);
  if (!Abs) {
    bool Relaxed, UseZeroPad;
    std::tie(Relaxed, UseZeroPad) = getBackend().relaxLEB128(LF, Value);
    if (!Relaxed) {
      reportError(LF.getValue().getLoc(),
                  Twine(LF.isSigned() ? ".s" : ".u") +
                      "leb128 expression is not absolute");
      LF.setValue(MCConstantExpr::create(0, Context));
    }
    uint8_t Tmp[10]; // maximum size: ceil(64/7)
    PadTo = std::max(PadTo, encodeULEB128(uint64_t(Value), Tmp));
    if (UseZeroPad)
      Value = 0;
  }
  uint8_t Data[16];
  size_t Size = 0;
  // The compiler can generate EH table assembly that is impossible to assemble
  // without either adding padding to an LEB fragment or adding extra padding
  // to a later alignment fragment. To accommodate such tables, relaxation can
  // only increase an LEB fragment size here, not decrease it. See PR35809.
  if (LF.isSigned())
    Size = encodeSLEB128(Value, Data, PadTo);
  else
    Size = encodeULEB128(Value, Data, PadTo);
  LF.setContents({reinterpret_cast<char *>(Data), Size});
  return OldSize != Size;
}

/// Check if the branch crosses the boundary.
///
/// \param StartAddr start address of the fused/unfused branch.
/// \param Size size of the fused/unfused branch.
/// \param BoundaryAlignment alignment requirement of the branch.
/// \returns true if the branch cross the boundary.
static bool mayCrossBoundary(uint64_t StartAddr, uint64_t Size,
                             Align BoundaryAlignment) {
  uint64_t EndAddr = StartAddr + Size;
  return (StartAddr >> Log2(BoundaryAlignment)) !=
         ((EndAddr - 1) >> Log2(BoundaryAlignment));
}

/// Check if the branch is against the boundary.
///
/// \param StartAddr start address of the fused/unfused branch.
/// \param Size size of the fused/unfused branch.
/// \param BoundaryAlignment alignment requirement of the branch.
/// \returns true if the branch is against the boundary.
static bool isAgainstBoundary(uint64_t StartAddr, uint64_t Size,
                              Align BoundaryAlignment) {
  uint64_t EndAddr = StartAddr + Size;
  return (EndAddr & (BoundaryAlignment.value() - 1)) == 0;
}

/// Check if the branch needs padding.
///
/// \param StartAddr start address of the fused/unfused branch.
/// \param Size size of the fused/unfused branch.
/// \param BoundaryAlignment alignment requirement of the branch.
/// \returns true if the branch needs padding.
static bool needPadding(uint64_t StartAddr, uint64_t Size,
                        Align BoundaryAlignment) {
  return mayCrossBoundary(StartAddr, Size, BoundaryAlignment) ||
         isAgainstBoundary(StartAddr, Size, BoundaryAlignment);
}

bool MCAssembler::relaxBoundaryAlign(MCBoundaryAlignFragment &BF) {
  // BoundaryAlignFragment that doesn't need to align any fragment should not be
  // relaxed.
  if (!BF.getLastFragment())
    return false;

  uint64_t AlignedOffset = getFragmentOffset(BF);
  uint64_t AlignedSize = 0;
  for (const MCFragment *F = BF.getNext();; F = F->getNext()) {
    AlignedSize += computeFragmentSize(*F);
    if (F == BF.getLastFragment())
      break;
  }

  Align BoundaryAlignment = BF.getAlignment();
  uint64_t NewSize = needPadding(AlignedOffset, AlignedSize, BoundaryAlignment)
                         ? offsetToAlignment(AlignedOffset, BoundaryAlignment)
                         : 0U;
  if (NewSize == BF.getSize())
    return false;
  BF.setSize(NewSize);
  return true;
}

bool MCAssembler::relaxDwarfLineAddr(MCDwarfLineAddrFragment &DF) {
  bool WasRelaxed;
  if (getBackend().relaxDwarfLineAddr(DF, WasRelaxed))
    return WasRelaxed;

  MCContext &Context = getContext();
  auto OldSize = DF.getContents().size();
  int64_t AddrDelta;
  bool Abs = DF.getAddrDelta().evaluateKnownAbsolute(AddrDelta, *this);
  assert(Abs && "We created a line delta with an invalid expression");
  (void)Abs;
  int64_t LineDelta;
  LineDelta = DF.getLineDelta();
  SmallVector<char, 8> Data;

  MCDwarfLineAddr::encode(Context, getDWARFLinetableParams(), LineDelta,
                          AddrDelta, Data);
  DF.setContents(Data);
  DF.clearFixups();
  return OldSize != Data.size();
}

bool MCAssembler::relaxDwarfCallFrameFragment(MCDwarfCallFrameFragment &DF) {
  bool WasRelaxed;
  if (getBackend().relaxDwarfCFA(DF, WasRelaxed))
    return WasRelaxed;

  MCContext &Context = getContext();
  int64_t Value;
  bool Abs = DF.getAddrDelta().evaluateAsAbsolute(Value, *this);
  if (!Abs) {
    reportError(DF.getAddrDelta().getLoc(),
                "invalid CFI advance_loc expression");
    DF.setAddrDelta(MCConstantExpr::create(0, Context));
    return false;
  }

  auto OldSize = DF.getContents().size();
  SmallVector<char, 8> Data;
  MCDwarfFrameEmitter::encodeAdvanceLoc(Context, Value, Data);
  DF.setContents(Data);
  DF.clearFixups();
  return OldSize != Data.size();
}

bool MCAssembler::relaxCVInlineLineTable(MCCVInlineLineTableFragment &F) {
  unsigned OldSize = F.getContents().size();
  getContext().getCVContext().encodeInlineLineTable(*this, F);
  return OldSize != F.getContents().size();
}

bool MCAssembler::relaxCVDefRange(MCCVDefRangeFragment &F) {
  unsigned OldSize = F.getContents().size();
  getContext().getCVContext().encodeDefRange(*this, F);
  return OldSize != F.getContents().size();
}

bool MCAssembler::relaxFill(MCFillFragment &F) {
  uint64_t Size = computeFragmentSize(F);
  if (F.getSize() == Size)
    return false;
  F.setSize(Size);
  return true;
}

bool MCAssembler::relaxPseudoProbeAddr(MCPseudoProbeAddrFragment &PF) {
  uint64_t OldSize = PF.getContents().size();
  int64_t AddrDelta;
  bool Abs = PF.getAddrDelta().evaluateKnownAbsolute(AddrDelta, *this);
  assert(Abs && "We created a pseudo probe with an invalid expression");
  (void)Abs;
  SmallVector<char, 8> Data;
  raw_svector_ostream OSE(Data);

  // AddrDelta is a signed integer
  encodeSLEB128(AddrDelta, OSE, OldSize);
  PF.setContents(Data);
  PF.clearFixups();
  return OldSize != Data.size();
}

bool MCAssembler::relaxFragment(MCFragment &F) {
  switch(F.getKind()) {
  default:
    return false;
  case MCFragment::FT_Relaxable:
    assert(!getRelaxAll() &&
           "Did not expect a MCRelaxableFragment in RelaxAll mode");
    return relaxInstruction(cast<MCRelaxableFragment>(F));
  case MCFragment::FT_Dwarf:
    return relaxDwarfLineAddr(cast<MCDwarfLineAddrFragment>(F));
  case MCFragment::FT_DwarfFrame:
    return relaxDwarfCallFrameFragment(cast<MCDwarfCallFrameFragment>(F));
  case MCFragment::FT_LEB:
    return relaxLEB(cast<MCLEBFragment>(F));
  case MCFragment::FT_BoundaryAlign:
    return relaxBoundaryAlign(cast<MCBoundaryAlignFragment>(F));
  case MCFragment::FT_CVInlineLines:
    return relaxCVInlineLineTable(cast<MCCVInlineLineTableFragment>(F));
  case MCFragment::FT_CVDefRange:
    return relaxCVDefRange(cast<MCCVDefRangeFragment>(F));
  case MCFragment::FT_Fill:
    return relaxFill(cast<MCFillFragment>(F));
  case MCFragment::FT_PseudoProbe:
    return relaxPseudoProbeAddr(cast<MCPseudoProbeAddrFragment>(F));
  }
}

void MCAssembler::layoutSection(MCSection &Sec) {
  MCFragment *Prev = nullptr;
  uint64_t Offset = 0;
  for (MCFragment &F : Sec) {
    F.Offset = Offset;
    if (LLVM_UNLIKELY(isBundlingEnabled())) {
      if (F.hasInstructions()) {
        layoutBundle(Prev, &F);
        Offset = F.Offset;
      }
      Prev = &F;
    }
    Offset += computeFragmentSize(F);
  }
}

bool MCAssembler::relaxOnce() {
  ++stats::RelaxationSteps;
  PendingErrors.clear();

  // Size of fragments in one section can depend on the size of fragments in
  // another. If any fragment has changed size, we have to re-layout (and
  // as a result possibly further relax) all sections.
  bool ChangedAny = false;
  for (MCSection &Sec : *this) {
    // Assume each iteration finalizes at least one extra fragment. If the
    // layout does not converge after N+1 iterations, bail out.
    auto MaxIter = Sec.curFragList()->Tail->getLayoutOrder() + 1;
    for (;;) {
      bool Changed = false;
      for (MCFragment &F : Sec)
        if (relaxFragment(F))
          Changed = true;

      ChangedAny |= Changed;
      if (!Changed || --MaxIter == 0)
        break;
      layoutSection(Sec);
    }
  }
  return ChangedAny;
}

void MCAssembler::reportError(SMLoc L, const Twine &Msg) const {
  getContext().reportError(L, Msg);
}

void MCAssembler::recordError(SMLoc Loc, const Twine &Msg) const {
  PendingErrors.emplace_back(Loc, Msg.str());
}

void MCAssembler::flushPendingErrors() const {
  for (auto &Err : PendingErrors)
    reportError(Err.first, Err.second);
  PendingErrors.clear();
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void MCAssembler::dump() const{
  raw_ostream &OS = errs();
  DenseMap<const MCFragment *, SmallVector<const MCSymbol *, 0>> FragToSyms;
  // Scan symbols and build a map of fragments to their corresponding symbols.
  // For variable symbols, we don't want to call their getFragment, which might
  // modify `Fragment`.
  for (const MCSymbol &Sym : symbols())
    if (!Sym.isVariable())
      if (auto *F = Sym.getFragment())
        FragToSyms.try_emplace(F).first->second.push_back(&Sym);

  OS << "Sections:[";
  for (const MCSection &Sec : *this) {
    OS << '\n';
    Sec.dump(&FragToSyms);
  }
  OS << "\n]\n";
}
#endif
