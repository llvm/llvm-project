//===- lib/MC/MCAssembler.cpp - Assembler Backend Implementation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAssembler.h"
#include "llvm/ADT/ArrayRef.h"
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
STATISTIC(Fixups, "Number of fixups");
STATISTIC(FixupEvalForRelax, "Number of fixup evaluations for relaxation");
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
                                bool RecordReloc, uint8_t *Data) const {
  if (RecordReloc)
    ++stats::Fixups;

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
  if (auto State = getBackend().evaluateFixup(F, Fixup, Target, Value)) {
    IsResolved = *State;
  } else {
    const MCSymbol *Add = Target.getAddSym();
    const MCSymbol *Sub = Target.getSubSym();
    Value += Target.getConstant();
    if (Add && Add->isDefined())
      Value += getSymbolOffset(*Add);
    if (Sub && Sub->isDefined())
      Value -= getSymbolOffset(*Sub);

    if (Fixup.isPCRel()) {
      Value -= getFragmentOffset(F) + Fixup.getOffset();
      if (Add && !Sub && !Add->isUndefined() && !Add->isAbsolute()) {
        IsResolved = getWriter().isSymbolRefDifferenceFullyResolvedImpl(
            *Add, F, false, true);
      }
    } else {
      IsResolved = Target.isAbsolute();
    }
  }

  if (!RecordReloc)
    return IsResolved;

  if (IsResolved && mc::isRelocRelocation(Fixup.getKind()))
    IsResolved = false;
  getBackend().applyFixup(F, Fixup, Target, Data, Value, IsResolved);
  return true;
}

uint64_t MCAssembler::computeFragmentSize(const MCFragment &F) const {
  assert(getBackendPtr() && "Requires assembler backend");
  switch (F.getKind()) {
  case MCFragment::FT_Data:
  case MCFragment::FT_Relaxable:
  case MCFragment::FT_Align:
  case MCFragment::FT_LEB:
  case MCFragment::FT_Dwarf:
  case MCFragment::FT_DwarfFrame:
  case MCFragment::FT_CVInlineLines:
  case MCFragment::FT_CVDefRange:
    return F.getSize();
  case MCFragment::FT_Fill: {
    auto &FF = static_cast<const MCFillFragment &>(F);
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

  case MCFragment::FT_BoundaryAlign:
    return cast<MCBoundaryAlignFragment>(F).getSize();

  case MCFragment::FT_SymbolId:
    return 4;

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
  }

  llvm_unreachable("invalid fragment kind");
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
  const MCFragment &F = *Sec.curFragList()->Tail;
  assert(HasLayout && F.getKind() == MCFragment::FT_Data);
  return getFragmentOffset(F) + F.getSize();
}

uint64_t MCAssembler::getSectionFileSize(const MCSection &Sec) const {
  // Virtual sections have no file size.
  if (Sec.isBssSection())
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

void MCAssembler::addRelocDirective(RelocDirective RD) {
  relocDirectives.push_back(RD);
}

/// Write the fragment \p F to the output file.
static void writeFragment(raw_ostream &OS, const MCAssembler &Asm,
                          const MCFragment &F) {
  // FIXME: Embed in fragments instead?
  uint64_t FragmentSize = Asm.computeFragmentSize(F);

  llvm::endianness Endian = Asm.getBackend().Endian;

  // This variable (and its dummy usage) is to participate in the assert at
  // the end of the function.
  uint64_t Start = OS.tell();
  (void) Start;

  ++stats::EmittedFragments;

  switch (F.getKind()) {
  case MCFragment::FT_Data:
  case MCFragment::FT_Relaxable:
  case MCFragment::FT_LEB:
  case MCFragment::FT_Dwarf:
  case MCFragment::FT_DwarfFrame:
  case MCFragment::FT_CVInlineLines:
  case MCFragment::FT_CVDefRange: {
    if (F.getKind() == MCFragment::FT_Data)
      ++stats::EmittedDataFragments;
    else if (F.getKind() == MCFragment::FT_Relaxable)
      ++stats::EmittedRelaxableFragments;
    const auto &EF = cast<MCFragment>(F);
    OS << StringRef(EF.getContents().data(), EF.getContents().size());
    OS << StringRef(EF.getVarContents().data(), EF.getVarContents().size());
  } break;

  case MCFragment::FT_Align: {
    ++stats::EmittedAlignFragments;
    OS << StringRef(F.getContents().data(), F.getContents().size());
    assert(F.getAlignFillLen() &&
           "Invalid virtual align in concrete fragment!");

    uint64_t Count = (FragmentSize - F.getFixedSize()) / F.getAlignFillLen();
    assert((FragmentSize - F.getFixedSize()) % F.getAlignFillLen() == 0 &&
           "computeFragmentSize computed size is incorrect");

    // In the nops mode, call the backend hook to write `Count` nops.
    if (F.hasAlignEmitNops()) {
      if (!Asm.getBackend().writeNopData(OS, Count, F.getSubtargetInfo()))
        reportFatalInternalError("unable to write nop sequence of " +
                                 Twine(Count) + " bytes");
    } else {
      // Otherwise, write out in multiples of the value size.
      for (uint64_t i = 0; i != Count; ++i) {
        switch (F.getAlignFillLen()) {
        default:
          llvm_unreachable("Invalid size!");
        case 1:
          OS << char(F.getAlignFill());
          break;
        case 2:
          support::endian::write<uint16_t>(OS, F.getAlignFill(), Endian);
          break;
        case 4:
          support::endian::write<uint32_t>(OS, F.getAlignFill(), Endian);
          break;
        case 8:
          support::endian::write<uint64_t>(OS, F.getAlignFill(), Endian);
          break;
        }
      }
    }
  } break;

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

  }

  assert(OS.tell() - Start == FragmentSize &&
         "The stream should advance by fragment size");
}

void MCAssembler::writeSectionData(raw_ostream &OS,
                                   const MCSection *Sec) const {
  assert(getBackendPtr() && "Expected assembler backend");

  if (Sec->isBssSection()) {
    assert(getSectionFileSize(*Sec) == 0 && "Invalid size for section!");

    // Ensure no fixups or non-zero bytes are written to BSS sections, catching
    // errors in both input assembly code and MCStreamer API usage. Location is
    // not tracked for efficiency.
    auto Fn = [](char c) { return c != 0; };
    for (const MCFragment &F : *Sec) {
      bool HasNonZero = false;
      switch (F.getKind()) {
      default:
        reportFatalInternalError("BSS section '" + Sec->getName() +
                                 "' contains invalid fragment");
        break;
      case MCFragment::FT_Data:
      case MCFragment::FT_Relaxable:
        HasNonZero =
            any_of(F.getContents(), Fn) || any_of(F.getVarContents(), Fn);
        break;
      case MCFragment::FT_Align:
        // Disallowed for API usage. AsmParser changes non-zero fill values to
        // 0.
        assert(F.getAlignFill() == 0 && "Invalid align in virtual section!");
        break;
      case MCFragment::FT_Fill:
        HasNonZero = cast<MCFillFragment>(F).getValue() != 0;
        break;
      case MCFragment::FT_Org:
        HasNonZero = cast<MCOrgFragment>(F).getValue() != 0;
        break;
      }
      if (HasNonZero) {
        reportError(SMLoc(), "BSS section '" + Sec->getName() +
                                 "' cannot have non-zero bytes");
        break;
      }
      if (F.getFixups().size() || F.getVarFixups().size()) {
        reportError(SMLoc(),
                    "BSS section '" + Sec->getName() + "' cannot have fixups");
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
      MCFragment Dummy;
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
  unsigned FirstStable = Sections.size();
  while ((FirstStable = relaxOnce(FirstStable)) > 0)
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

  // Resolve .reloc offsets and add fixups.
  for (auto &PF : relocDirectives) {
    MCValue Res;
    auto &O = PF.Offset;
    if (!O.evaluateAsValue(Res, *this)) {
      getContext().reportError(O.getLoc(), ".reloc offset is not relocatable");
      continue;
    }
    auto *Sym = Res.getAddSym();
    auto *F = Sym ? Sym->getFragment() : nullptr;
    auto *Sec = F ? F->getParent() : nullptr;
    if (Res.getSubSym() || !Sec) {
      getContext().reportError(O.getLoc(),
                               ".reloc offset is not relative to a section");
      continue;
    }

    uint64_t Offset = Sym ? Sym->getOffset() + Res.getConstant() : 0;
    F->addFixup(MCFixup::create(Offset, PF.Expr, PF.Kind));
  }

  // Evaluate and apply the fixups, generating relocation entries as necessary.
  for (MCSection &Sec : *this) {
    for (MCFragment &F : Sec) {
      // Process fragments with fixups here.
      auto Contents = F.getContents();
      for (MCFixup &Fixup : F.getFixups()) {
        uint64_t FixedValue;
        MCValue Target;
        assert(mc::isRelocRelocation(Fixup.getKind()) ||
               Fixup.getOffset() <= F.getFixedSize());
        auto *Data =
            reinterpret_cast<uint8_t *>(Contents.data() + Fixup.getOffset());
        evaluateFixup(F, Fixup, Target, FixedValue,
                      /*RecordReloc=*/true, Data);
      }
      // In the variable part, fixup offsets are relative to the fixed part's
      // start.
      for (MCFixup &Fixup : F.getVarFixups()) {
        uint64_t FixedValue;
        MCValue Target;
        assert(mc::isRelocRelocation(Fixup.getKind()) ||
               (Fixup.getOffset() >= F.getFixedSize() &&
                Fixup.getOffset() <= F.getSize()));
        auto *Data = reinterpret_cast<uint8_t *>(
            F.getVarContents().data() + (Fixup.getOffset() - F.getFixedSize()));
        evaluateFixup(F, Fixup, Target, FixedValue,
                      /*RecordReloc=*/true, Data);
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

bool MCAssembler::fixupNeedsRelaxation(const MCFragment &F,
                                       const MCFixup &Fixup) const {
  ++stats::FixupEvalForRelax;
  MCValue Target;
  uint64_t Value;
  bool Resolved = evaluateFixup(F, const_cast<MCFixup &>(Fixup), Target, Value,
                                /*RecordReloc=*/false, {});
  return getBackend().fixupNeedsRelaxationAdvanced(F, Fixup, Target, Value,
                                                   Resolved);
}

void MCAssembler::relaxInstruction(MCFragment &F) {
  assert(getEmitterPtr() &&
         "Expected CodeEmitter defined for relaxInstruction");
  // If this inst doesn't ever need relaxation, ignore it. This occurs when we
  // are intentionally pushing out inst fragments, or because we relaxed a
  // previous instruction to one that doesn't need relaxation.
  if (!getBackend().mayNeedRelaxation(F.getOpcode(), F.getOperands(),
                                      *F.getSubtargetInfo()))
    return;

  bool DoRelax = false;
  for (const MCFixup &Fixup : F.getVarFixups())
    if ((DoRelax = fixupNeedsRelaxation(F, Fixup)))
      break;
  if (!DoRelax)
    return;

  ++stats::RelaxedInstructions;

  // TODO Refactor relaxInstruction to accept MCFragment and remove
  // `setInst`.
  MCInst Relaxed = F.getInst();
  getBackend().relaxInstruction(Relaxed, *F.getSubtargetInfo());

  // Encode the new instruction.
  F.setInst(Relaxed);
  SmallVector<char, 16> Data;
  SmallVector<MCFixup, 1> Fixups;
  getEmitter().encodeInstruction(Relaxed, Data, Fixups, *F.getSubtargetInfo());
  F.setVarContents(Data);
  F.setVarFixups(Fixups);
}

void MCAssembler::relaxLEB(MCFragment &F) {
  unsigned PadTo = F.getVarSize();
  int64_t Value;
  F.clearVarFixups();
  // Use evaluateKnownAbsolute for Mach-O as a hack: .subsections_via_symbols
  // requires that .uleb128 A-B is foldable where A and B reside in different
  // fragments. This is used by __gcc_except_table.
  bool Abs = getWriter().getSubsectionsViaSymbols()
                 ? F.getLEBValue().evaluateKnownAbsolute(Value, *this)
                 : F.getLEBValue().evaluateAsAbsolute(Value, *this);
  if (!Abs) {
    bool Relaxed, UseZeroPad;
    std::tie(Relaxed, UseZeroPad) = getBackend().relaxLEB128(F, Value);
    if (!Relaxed) {
      reportError(F.getLEBValue().getLoc(),
                  Twine(F.isLEBSigned() ? ".s" : ".u") +
                      "leb128 expression is not absolute");
      F.setLEBValue(MCConstantExpr::create(0, Context));
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
  if (F.isLEBSigned())
    Size = encodeSLEB128(Value, Data, PadTo);
  else
    Size = encodeULEB128(Value, Data, PadTo);
  F.setVarContents({reinterpret_cast<char *>(Data), Size});
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

void MCAssembler::relaxBoundaryAlign(MCBoundaryAlignFragment &BF) {
  // BoundaryAlignFragment that doesn't need to align any fragment should not be
  // relaxed.
  if (!BF.getLastFragment())
    return;

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
    return;
  BF.setSize(NewSize);
}

void MCAssembler::relaxDwarfLineAddr(MCFragment &F) {
  if (getBackend().relaxDwarfLineAddr(F))
    return;

  MCContext &Context = getContext();
  int64_t AddrDelta;
  bool Abs = F.getDwarfAddrDelta().evaluateKnownAbsolute(AddrDelta, *this);
  assert(Abs && "We created a line delta with an invalid expression");
  (void)Abs;
  SmallVector<char, 8> Data;
  MCDwarfLineAddr::encode(Context, getDWARFLinetableParams(),
                          F.getDwarfLineDelta(), AddrDelta, Data);
  F.setVarContents(Data);
  F.clearVarFixups();
}

void MCAssembler::relaxDwarfCallFrameFragment(MCFragment &F) {
  if (getBackend().relaxDwarfCFA(F))
    return;

  MCContext &Context = getContext();
  int64_t Value;
  bool Abs = F.getDwarfAddrDelta().evaluateAsAbsolute(Value, *this);
  if (!Abs) {
    reportError(F.getDwarfAddrDelta().getLoc(),
                "invalid CFI advance_loc expression");
    F.setDwarfAddrDelta(MCConstantExpr::create(0, Context));
    return;
  }

  SmallVector<char, 8> Data;
  MCDwarfFrameEmitter::encodeAdvanceLoc(Context, Value, Data);
  F.setVarContents(Data);
  F.clearVarFixups();
}

bool MCAssembler::relaxFragment(MCFragment &F) {
  size_t Size = computeFragmentSize(F);
  switch (F.getKind()) {
  default:
    return false;
  case MCFragment::FT_Relaxable:
    assert(!getRelaxAll() && "Did not expect a FT_Relaxable in RelaxAll mode");
    relaxInstruction(F);
    break;
  case MCFragment::FT_LEB:
    relaxLEB(F);
    break;
  case MCFragment::FT_Dwarf:
    relaxDwarfLineAddr(F);
    break;
  case MCFragment::FT_DwarfFrame:
    relaxDwarfCallFrameFragment(F);
    break;
  case MCFragment::FT_BoundaryAlign:
    relaxBoundaryAlign(static_cast<MCBoundaryAlignFragment &>(F));
    break;
  case MCFragment::FT_CVInlineLines:
    getContext().getCVContext().encodeInlineLineTable(
        *this, static_cast<MCCVInlineLineTableFragment &>(F));
    break;
  case MCFragment::FT_CVDefRange:
    getContext().getCVContext().encodeDefRange(
        *this, static_cast<MCCVDefRangeFragment &>(F));
    break;
  case MCFragment::FT_Fill: {
    auto &FF = static_cast<MCFillFragment &>(F);
    if (FF.getSize() == Size)
      return false;
    FF.setSize(Size);
    return true;
  }
  case MCFragment::FT_Org: {
    auto &FF = static_cast<MCOrgFragment &>(F);
    if (FF.getSize() == Size)
      return false;
    FF.setSize(Size);
    return true;
  }
  }
  return computeFragmentSize(F) != Size;
}

void MCAssembler::layoutSection(MCSection &Sec) {
  uint64_t Offset = 0;
  for (MCFragment &F : Sec) {
    F.Offset = Offset;
    if (F.getKind() == MCFragment::FT_Align) {
      Offset += F.getFixedSize();
      unsigned Size = offsetToAlignment(Offset, F.getAlignment());
      // In the nops mode, RISC-V style linker relaxation might adjust the size
      // and add a fixup, even if `Size` is originally 0.
      bool AlignFixup = false;
      if (F.hasAlignEmitNops()) {
        AlignFixup = getBackend().relaxAlign(F, Size);
        // If the backend does not handle the fragment specially, pad with nops,
        // but ensure that the padding is larger than the minimum nop size.
        if (!AlignFixup)
          while (Size % getBackend().getMinimumNopSize())
            Size += F.getAlignment().value();
      }
      if (!AlignFixup && Size > F.getAlignMaxBytesToEmit())
        Size = 0;
      // Update the variable tail size, offset by FixedSize to prevent ubsan
      // pointer-overflow in evaluateFixup. The content is ignored.
      F.VarContentStart = F.getFixedSize();
      F.VarContentEnd = F.VarContentStart + Size;
      if (F.VarContentEnd > F.getParent()->ContentStorage.size())
        F.getParent()->ContentStorage.resize(F.VarContentEnd);
      Offset += Size;
    } else {
      Offset += computeFragmentSize(F);
    }
  }
}

unsigned MCAssembler::relaxOnce(unsigned FirstStable) {
  ++stats::RelaxationSteps;
  PendingErrors.clear();

  unsigned Res = 0;
  for (unsigned I = 0; I != FirstStable; ++I) {
    // Assume each iteration finalizes at least one extra fragment. If the
    // layout does not converge after N+1 iterations, bail out.
    auto &Sec = *Sections[I];
    auto MaxIter = Sec.curFragList()->Tail->getLayoutOrder() + 1;
    for (;;) {
      bool Changed = false;
      for (MCFragment &F : Sec)
        if (F.getKind() != MCFragment::FT_Data && relaxFragment(F))
          Changed = true;

      if (!Changed)
        break;
      // If any fragment changed size, it might impact the layout of subsequent
      // sections. Therefore, we must re-evaluate all sections.
      FirstStable = Sections.size();
      Res = I;
      if (--MaxIter == 0)
        break;
      layoutSection(Sec);
    }
  }
  // The subsequent relaxOnce call only needs to visit Sections [0,Res) if no
  // change occurred.
  return Res;
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

SMLoc MCFixup::getLoc() const {
  if (auto *E = getValue())
    return E->getLoc();
  return {};
}
