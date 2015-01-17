//===- lib/MC/MCAssembler.cpp - Assembler Backend Implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAssembler.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include <tuple>
using namespace llvm;

#define DEBUG_TYPE "assembler"

namespace {
namespace stats {
STATISTIC(EmittedFragments, "Number of emitted assembler fragments - total");
STATISTIC(EmittedRelaxableFragments,
          "Number of emitted assembler fragments - relaxable");
STATISTIC(EmittedDataFragments,
          "Number of emitted assembler fragments - data");
STATISTIC(EmittedCompactEncodedInstFragments,
          "Number of emitted assembler fragments - compact encoded inst");
STATISTIC(EmittedAlignFragments,
          "Number of emitted assembler fragments - align");
STATISTIC(EmittedFillFragments,
          "Number of emitted assembler fragments - fill");
STATISTIC(EmittedOrgFragments,
          "Number of emitted assembler fragments - org");
STATISTIC(evaluateFixup, "Number of evaluated fixups");
STATISTIC(FragmentLayouts, "Number of fragment layouts");
STATISTIC(ObjectBytes, "Number of emitted object file bytes");
STATISTIC(RelaxationSteps, "Number of assembler layout and relaxation steps");
STATISTIC(RelaxedInstructions, "Number of relaxed instructions");
}
}

// FIXME FIXME FIXME: There are number of places in this file where we convert
// what is a 64-bit assembler value used for computation into a value in the
// object file, which may truncate it. We should detect that truncation where
// invalid and report errors back.

/* *** */

MCAsmLayout::MCAsmLayout(MCAssembler &Asm)
  : Assembler(Asm), LastValidFragment()
 {
  // Compute the section layout order. Virtual sections must go last.
  for (MCAssembler::iterator it = Asm.begin(), ie = Asm.end(); it != ie; ++it)
    if (!it->getSection().isVirtualSection())
      SectionOrder.push_back(&*it);
  for (MCAssembler::iterator it = Asm.begin(), ie = Asm.end(); it != ie; ++it)
    if (it->getSection().isVirtualSection())
      SectionOrder.push_back(&*it);
}

bool MCAsmLayout::isFragmentValid(const MCFragment *F) const {
  const MCSectionData &SD = *F->getParent();
  const MCFragment *LastValid = LastValidFragment.lookup(&SD);
  if (!LastValid)
    return false;
  assert(LastValid->getParent() == F->getParent());
  return F->getLayoutOrder() <= LastValid->getLayoutOrder();
}

void MCAsmLayout::invalidateFragmentsFrom(MCFragment *F) {
  // If this fragment wasn't already valid, we don't need to do anything.
  if (!isFragmentValid(F))
    return;

  // Otherwise, reset the last valid fragment to the previous fragment
  // (if this is the first fragment, it will be NULL).
  const MCSectionData &SD = *F->getParent();
  LastValidFragment[&SD] = F->getPrevNode();
}

void MCAsmLayout::ensureValid(const MCFragment *F) const {
  MCSectionData &SD = *F->getParent();

  MCFragment *Cur = LastValidFragment[&SD];
  if (!Cur)
    Cur = &*SD.begin();
  else
    Cur = Cur->getNextNode();

  // Advance the layout position until the fragment is valid.
  while (!isFragmentValid(F)) {
    assert(Cur && "Layout bookkeeping error");
    const_cast<MCAsmLayout*>(this)->layoutFragment(Cur);
    Cur = Cur->getNextNode();
  }
}

uint64_t MCAsmLayout::getFragmentOffset(const MCFragment *F) const {
  ensureValid(F);
  assert(F->Offset != ~UINT64_C(0) && "Address not set!");
  return F->Offset;
}

// Simple getSymbolOffset helper for the non-varibale case.
static bool getLabelOffset(const MCAsmLayout &Layout, const MCSymbolData &SD,
                           bool ReportError, uint64_t &Val) {
  if (!SD.getFragment()) {
    if (ReportError)
      report_fatal_error("unable to evaluate offset to undefined symbol '" +
                         SD.getSymbol().getName() + "'");
    return false;
  }
  Val = Layout.getFragmentOffset(SD.getFragment()) + SD.getOffset();
  return true;
}

static bool getSymbolOffsetImpl(const MCAsmLayout &Layout,
                                const MCSymbolData *SD, bool ReportError,
                                uint64_t &Val) {
  const MCSymbol &S = SD->getSymbol();

  if (!S.isVariable())
    return getLabelOffset(Layout, *SD, ReportError, Val);

  // If SD is a variable, evaluate it.
  MCValue Target;
  if (!S.getVariableValue()->EvaluateAsValue(Target, &Layout, nullptr))
    report_fatal_error("unable to evaluate offset for variable '" +
                       S.getName() + "'");

  uint64_t Offset = Target.getConstant();

  const MCAssembler &Asm = Layout.getAssembler();

  const MCSymbolRefExpr *A = Target.getSymA();
  if (A) {
    uint64_t ValA;
    if (!getLabelOffset(Layout, Asm.getSymbolData(A->getSymbol()), ReportError,
                        ValA))
      return false;
    Offset += ValA;
  }

  const MCSymbolRefExpr *B = Target.getSymB();
  if (B) {
    uint64_t ValB;
    if (!getLabelOffset(Layout, Asm.getSymbolData(B->getSymbol()), ReportError,
                        ValB))
      return false;
    Offset -= ValB;
  }

  Val = Offset;
  return true;
}

bool MCAsmLayout::getSymbolOffset(const MCSymbolData *SD, uint64_t &Val) const {
  return getSymbolOffsetImpl(*this, SD, false, Val);
}

uint64_t MCAsmLayout::getSymbolOffset(const MCSymbolData *SD) const {
  uint64_t Val;
  getSymbolOffsetImpl(*this, SD, true, Val);
  return Val;
}

const MCSymbol *MCAsmLayout::getBaseSymbol(const MCSymbol &Symbol) const {
  if (!Symbol.isVariable())
    return &Symbol;

  const MCExpr *Expr = Symbol.getVariableValue();
  MCValue Value;
  if (!Expr->EvaluateAsValue(Value, this, nullptr))
    llvm_unreachable("Invalid Expression");

  const MCSymbolRefExpr *RefB = Value.getSymB();
  if (RefB)
    Assembler.getContext().FatalError(
        SMLoc(), Twine("symbol '") + RefB->getSymbol().getName() +
                     "' could not be evaluated in a subtraction expression");

  const MCSymbolRefExpr *A = Value.getSymA();
  if (!A)
    return nullptr;

  return &A->getSymbol();
}

uint64_t MCAsmLayout::getSectionAddressSize(const MCSectionData *SD) const {
  // The size is the last fragment's end offset.
  const MCFragment &F = SD->getFragmentList().back();
  return getFragmentOffset(&F) + getAssembler().computeFragmentSize(*this, F);
}

uint64_t MCAsmLayout::getSectionFileSize(const MCSectionData *SD) const {
  // Virtual sections have no file size.
  if (SD->getSection().isVirtualSection())
    return 0;

  // Otherwise, the file size is the same as the address space size.
  return getSectionAddressSize(SD);
}

uint64_t MCAsmLayout::computeBundlePadding(const MCFragment *F,
                                           uint64_t FOffset, uint64_t FSize) {
  uint64_t BundleSize = Assembler.getBundleAlignSize();
  assert(BundleSize > 0 &&
         "computeBundlePadding should only be called if bundling is enabled");
  uint64_t BundleMask = BundleSize - 1;
  uint64_t OffsetInBundle = FOffset & BundleMask;
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
  } else if (EndOfFragment > BundleSize)
    return BundleSize - OffsetInBundle;
  else
    return 0;
}

/* *** */

MCFragment::MCFragment() : Kind(FragmentType(~0)) {
}

MCFragment::~MCFragment() {
}

MCFragment::MCFragment(FragmentType _Kind, MCSectionData *_Parent)
  : Kind(_Kind), Parent(_Parent), Atom(nullptr), Offset(~UINT64_C(0))
{
  if (Parent)
    Parent->getFragmentList().push_back(this);
}

/* *** */

MCEncodedFragment::~MCEncodedFragment() {
}

/* *** */

MCEncodedFragmentWithFixups::~MCEncodedFragmentWithFixups() {
}

/* *** */

MCSectionData::MCSectionData() : Section(nullptr) {}

MCSectionData::MCSectionData(const MCSection &_Section, MCAssembler *A)
  : Section(&_Section),
    Ordinal(~UINT32_C(0)),
    Alignment(1),
    BundleLockState(NotBundleLocked),
    BundleLockNestingDepth(0),
    BundleGroupBeforeFirstInst(false),
    HasInstructions(false)
{
  if (A)
    A->getSectionList().push_back(this);
}

MCSectionData::iterator
MCSectionData::getSubsectionInsertionPoint(unsigned Subsection) {
  if (Subsection == 0 && SubsectionFragmentMap.empty())
    return end();

  SmallVectorImpl<std::pair<unsigned, MCFragment *> >::iterator MI =
    std::lower_bound(SubsectionFragmentMap.begin(), SubsectionFragmentMap.end(),
                     std::make_pair(Subsection, (MCFragment *)nullptr));
  bool ExactMatch = false;
  if (MI != SubsectionFragmentMap.end()) {
    ExactMatch = MI->first == Subsection;
    if (ExactMatch)
      ++MI;
  }
  iterator IP;
  if (MI == SubsectionFragmentMap.end())
    IP = end();
  else
    IP = MI->second;
  if (!ExactMatch && Subsection != 0) {
    // The GNU as documentation claims that subsections have an alignment of 4,
    // although this appears not to be the case.
    MCFragment *F = new MCDataFragment();
    SubsectionFragmentMap.insert(MI, std::make_pair(Subsection, F));
    getFragmentList().insert(IP, F);
    F->setParent(this);
  }
  return IP;
}

void MCSectionData::setBundleLockState(BundleLockStateType NewState) {
  if (NewState == NotBundleLocked) {
    if (BundleLockNestingDepth == 0) {
      report_fatal_error("Mismatched bundle_lock/unlock directives");
    }
    if (--BundleLockNestingDepth == 0) {
      BundleLockState = NotBundleLocked;
    }
    return;
  }

  // If any of the directives is an align_to_end directive, the whole nested
  // group is align_to_end. So don't downgrade from align_to_end to just locked.
  if (BundleLockState != BundleLockedAlignToEnd) {
    BundleLockState = NewState;
  }
  ++BundleLockNestingDepth;
}

/* *** */

MCSymbolData::MCSymbolData() : Symbol(nullptr) {}

MCSymbolData::MCSymbolData(const MCSymbol &_Symbol, MCFragment *_Fragment,
                           uint64_t _Offset, MCAssembler *A)
    : Symbol(&_Symbol), Fragment(_Fragment), Offset(_Offset),
      SymbolSize(nullptr), CommonAlign(-1U), Flags(0), Index(0) {
  if (A)
    A->getSymbolList().push_back(this);
}

/* *** */

MCAssembler::MCAssembler(MCContext &Context_, MCAsmBackend &Backend_,
                         MCCodeEmitter &Emitter_, MCObjectWriter &Writer_,
                         raw_ostream &OS_)
    : Context(Context_), Backend(Backend_), Emitter(Emitter_), Writer(Writer_),
      OS(OS_), BundleAlignSize(0), RelaxAll(false),
      SubsectionsViaSymbols(false), ELFHeaderEFlags(0) {
  VersionMinInfo.Major = 0; // Major version == 0 for "none specified"
}

MCAssembler::~MCAssembler() {
}

void MCAssembler::reset() {
  Sections.clear();
  Symbols.clear();
  SectionMap.clear();
  SymbolMap.clear();
  IndirectSymbols.clear();
  DataRegions.clear();
  LinkerOptions.clear();
  FileNames.clear();
  ThumbFuncs.clear();
  BundleAlignSize = 0;
  RelaxAll = false;
  SubsectionsViaSymbols = false;
  ELFHeaderEFlags = 0;
  LOHContainer.reset();
  VersionMinInfo.Major = 0;

  // reset objects owned by us
  getBackend().reset();
  getEmitter().reset();
  getWriter().reset();
  getLOHContainer().reset();
}

bool MCAssembler::isThumbFunc(const MCSymbol *Symbol) const {
  if (ThumbFuncs.count(Symbol))
    return true;

  if (!Symbol->isVariable())
    return false;

  // FIXME: It looks like gas supports some cases of the form "foo + 2". It
  // is not clear if that is a bug or a feature.
  const MCExpr *Expr = Symbol->getVariableValue();
  const MCSymbolRefExpr *Ref = dyn_cast<MCSymbolRefExpr>(Expr);
  if (!Ref)
    return false;

  if (Ref->getKind() != MCSymbolRefExpr::VK_None)
    return false;

  const MCSymbol &Sym = Ref->getSymbol();
  if (!isThumbFunc(&Sym))
    return false;

  ThumbFuncs.insert(Symbol); // Cache it.
  return true;
}

bool MCAssembler::isSymbolLinkerVisible(const MCSymbol &Symbol) const {
  // Non-temporary labels should always be visible to the linker.
  if (!Symbol.isTemporary())
    return true;

  // Absolute temporary labels are never visible.
  if (!Symbol.isInSection())
    return false;

  // Otherwise, check if the section requires symbols even for temporary labels.
  return getBackend().doesSectionRequireSymbols(Symbol.getSection());
}

const MCSymbolData *MCAssembler::getAtom(const MCSymbolData *SD) const {
  // Linker visible symbols define atoms.
  if (isSymbolLinkerVisible(SD->getSymbol()))
    return SD;

  // Absolute and undefined symbols have no defining atom.
  if (!SD->getFragment())
    return nullptr;

  // Non-linker visible symbols in sections which can't be atomized have no
  // defining atom.
  if (!getContext().getAsmInfo()->isSectionAtomizableBySymbols(
          SD->getFragment()->getParent()->getSection()))
    return nullptr;

  // Otherwise, return the atom for the containing fragment.
  return SD->getFragment()->getAtom();
}

// Try to fully compute Expr to an absolute value and if that fails produce
// a relocatable expr.
// FIXME: Should this be the behavior of EvaluateAsRelocatable itself?
static bool evaluate(const MCExpr &Expr, const MCAsmLayout &Layout,
                     const MCFixup &Fixup, MCValue &Target) {
  if (Expr.EvaluateAsValue(Target, &Layout, &Fixup)) {
    if (Target.isAbsolute())
      return true;
  }
  return Expr.EvaluateAsRelocatable(Target, &Layout, &Fixup);
}

bool MCAssembler::evaluateFixup(const MCAsmLayout &Layout,
                                const MCFixup &Fixup, const MCFragment *DF,
                                MCValue &Target, uint64_t &Value) const {
  ++stats::evaluateFixup;

  // FIXME: This code has some duplication with RecordRelocation. We should
  // probably merge the two into a single callback that tries to evaluate a
  // fixup and records a relocation if one is needed.
  const MCExpr *Expr = Fixup.getValue();
  if (!evaluate(*Expr, Layout, Fixup, Target))
    getContext().FatalError(Fixup.getLoc(), "expected relocatable expression");

  bool IsPCRel = Backend.getFixupKindInfo(
    Fixup.getKind()).Flags & MCFixupKindInfo::FKF_IsPCRel;

  bool IsResolved;
  if (IsPCRel) {
    if (Target.getSymB()) {
      IsResolved = false;
    } else if (!Target.getSymA()) {
      IsResolved = false;
    } else {
      const MCSymbolRefExpr *A = Target.getSymA();
      const MCSymbol &SA = A->getSymbol();
      if (A->getKind() != MCSymbolRefExpr::VK_None ||
          SA.AliasedSymbol().isUndefined()) {
        IsResolved = false;
      } else {
        const MCSymbolData &DataA = getSymbolData(SA);
        IsResolved =
          getWriter().IsSymbolRefDifferenceFullyResolvedImpl(*this, DataA,
                                                             *DF, false, true);
      }
    }
  } else {
    IsResolved = Target.isAbsolute();
  }

  Value = Target.getConstant();

  if (const MCSymbolRefExpr *A = Target.getSymA()) {
    const MCSymbol &Sym = A->getSymbol().AliasedSymbol();
    if (Sym.isDefined())
      Value += Layout.getSymbolOffset(&getSymbolData(Sym));
  }
  if (const MCSymbolRefExpr *B = Target.getSymB()) {
    const MCSymbol &Sym = B->getSymbol().AliasedSymbol();
    if (Sym.isDefined())
      Value -= Layout.getSymbolOffset(&getSymbolData(Sym));
  }


  bool ShouldAlignPC = Backend.getFixupKindInfo(Fixup.getKind()).Flags &
                         MCFixupKindInfo::FKF_IsAlignedDownTo32Bits;
  assert((ShouldAlignPC ? IsPCRel : true) &&
    "FKF_IsAlignedDownTo32Bits is only allowed on PC-relative fixups!");

  if (IsPCRel) {
    uint32_t Offset = Layout.getFragmentOffset(DF) + Fixup.getOffset();

    // A number of ARM fixups in Thumb mode require that the effective PC
    // address be determined as the 32-bit aligned version of the actual offset.
    if (ShouldAlignPC) Offset &= ~0x3;
    Value -= Offset;
  }

  // Let the backend adjust the fixup value if necessary, including whether
  // we need a relocation.
  Backend.processFixupValue(*this, Layout, Fixup, DF, Target, Value,
                            IsResolved);

  return IsResolved;
}

uint64_t MCAssembler::computeFragmentSize(const MCAsmLayout &Layout,
                                          const MCFragment &F) const {
  switch (F.getKind()) {
  case MCFragment::FT_Data:
  case MCFragment::FT_Relaxable:
  case MCFragment::FT_CompactEncodedInst:
    return cast<MCEncodedFragment>(F).getContents().size();
  case MCFragment::FT_Fill:
    return cast<MCFillFragment>(F).getSize();

  case MCFragment::FT_LEB:
    return cast<MCLEBFragment>(F).getContents().size();

  case MCFragment::FT_Align: {
    const MCAlignFragment &AF = cast<MCAlignFragment>(F);
    unsigned Offset = Layout.getFragmentOffset(&AF);
    unsigned Size = OffsetToAlignment(Offset, AF.getAlignment());
    // If we are padding with nops, force the padding to be larger than the
    // minimum nop size.
    if (Size > 0 && AF.hasEmitNops()) {
      while (Size % getBackend().getMinimumNopSize())
        Size += AF.getAlignment();
    }
    if (Size > AF.getMaxBytesToEmit())
      return 0;
    return Size;
  }

  case MCFragment::FT_Org: {
    const MCOrgFragment &OF = cast<MCOrgFragment>(F);
    int64_t TargetLocation;
    if (!OF.getOffset().EvaluateAsAbsolute(TargetLocation, Layout))
      report_fatal_error("expected assembly-time absolute expression");

    // FIXME: We need a way to communicate this error.
    uint64_t FragmentOffset = Layout.getFragmentOffset(&OF);
    int64_t Size = TargetLocation - FragmentOffset;
    if (Size < 0 || Size >= 0x40000000)
      report_fatal_error("invalid .org offset '" + Twine(TargetLocation) +
                         "' (at offset '" + Twine(FragmentOffset) + "')");
    return Size;
  }

  case MCFragment::FT_Dwarf:
    return cast<MCDwarfLineAddrFragment>(F).getContents().size();
  case MCFragment::FT_DwarfFrame:
    return cast<MCDwarfCallFrameFragment>(F).getContents().size();
  }

  llvm_unreachable("invalid fragment kind");
}

void MCAsmLayout::layoutFragment(MCFragment *F) {
  MCFragment *Prev = F->getPrevNode();

  // We should never try to recompute something which is valid.
  assert(!isFragmentValid(F) && "Attempt to recompute a valid fragment!");
  // We should never try to compute the fragment layout if its predecessor
  // isn't valid.
  assert((!Prev || isFragmentValid(Prev)) &&
         "Attempt to compute fragment before its predecessor!");

  ++stats::FragmentLayouts;

  // Compute fragment offset and size.
  if (Prev)
    F->Offset = Prev->Offset + getAssembler().computeFragmentSize(*this, *Prev);
  else
    F->Offset = 0;
  LastValidFragment[F->getParent()] = F;

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
  if (Assembler.isBundlingEnabled() && F->hasInstructions()) {
    assert(isa<MCEncodedFragment>(F) &&
           "Only MCEncodedFragment implementations have instructions");
    uint64_t FSize = Assembler.computeFragmentSize(*this, *F);

    if (FSize > Assembler.getBundleAlignSize())
      report_fatal_error("Fragment can't be larger than a bundle size");

    uint64_t RequiredBundlePadding = computeBundlePadding(F, F->Offset, FSize);
    if (RequiredBundlePadding > UINT8_MAX)
      report_fatal_error("Padding cannot exceed 255 bytes");
    F->setBundlePadding(static_cast<uint8_t>(RequiredBundlePadding));
    F->Offset += RequiredBundlePadding;
  }
}

/// \brief Write the contents of a fragment to the given object writer. Expects
///        a MCEncodedFragment.
static void writeFragmentContents(const MCFragment &F, MCObjectWriter *OW) {
  const MCEncodedFragment &EF = cast<MCEncodedFragment>(F);
  OW->WriteBytes(EF.getContents());
}

/// \brief Write the fragment \p F to the output file.
static void writeFragment(const MCAssembler &Asm, const MCAsmLayout &Layout,
                          const MCFragment &F) {
  MCObjectWriter *OW = &Asm.getWriter();

  // FIXME: Embed in fragments instead?
  uint64_t FragmentSize = Asm.computeFragmentSize(Layout, F);

  // Should NOP padding be written out before this fragment?
  unsigned BundlePadding = F.getBundlePadding();
  if (BundlePadding > 0) {
    assert(Asm.isBundlingEnabled() &&
           "Writing bundle padding with disabled bundling");
    assert(F.hasInstructions() &&
           "Writing bundle padding for a fragment without instructions");

    unsigned TotalLength = BundlePadding + static_cast<unsigned>(FragmentSize);
    if (F.alignToBundleEnd() && TotalLength > Asm.getBundleAlignSize()) {
      // If the padding itself crosses a bundle boundary, it must be emitted
      // in 2 pieces, since even nop instructions must not cross boundaries.
      //             v--------------v   <- BundleAlignSize
      //        v---------v             <- BundlePadding
      // ----------------------------
      // | Prev |####|####|    F    |
      // ----------------------------
      //        ^-------------------^   <- TotalLength
      unsigned DistanceToBoundary = TotalLength - Asm.getBundleAlignSize();
      if (!Asm.getBackend().writeNopData(DistanceToBoundary, OW))
          report_fatal_error("unable to write NOP sequence of " +
                             Twine(DistanceToBoundary) + " bytes");
      BundlePadding -= DistanceToBoundary;
    }
    if (!Asm.getBackend().writeNopData(BundlePadding, OW))
      report_fatal_error("unable to write NOP sequence of " +
                         Twine(BundlePadding) + " bytes");
  }

  // This variable (and its dummy usage) is to participate in the assert at
  // the end of the function.
  uint64_t Start = OW->getStream().tell();
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
      if (!Asm.getBackend().writeNopData(Count, OW))
        report_fatal_error("unable to write nop sequence of " +
                          Twine(Count) + " bytes");
      break;
    }

    // Otherwise, write out in multiples of the value size.
    for (uint64_t i = 0; i != Count; ++i) {
      switch (AF.getValueSize()) {
      default: llvm_unreachable("Invalid size!");
      case 1: OW->Write8 (uint8_t (AF.getValue())); break;
      case 2: OW->Write16(uint16_t(AF.getValue())); break;
      case 4: OW->Write32(uint32_t(AF.getValue())); break;
      case 8: OW->Write64(uint64_t(AF.getValue())); break;
      }
    }
    break;
  }

  case MCFragment::FT_Data: 
    ++stats::EmittedDataFragments;
    writeFragmentContents(F, OW);
    break;

  case MCFragment::FT_Relaxable:
    ++stats::EmittedRelaxableFragments;
    writeFragmentContents(F, OW);
    break;

  case MCFragment::FT_CompactEncodedInst:
    ++stats::EmittedCompactEncodedInstFragments;
    writeFragmentContents(F, OW);
    break;

  case MCFragment::FT_Fill: {
    ++stats::EmittedFillFragments;
    const MCFillFragment &FF = cast<MCFillFragment>(F);

    assert(FF.getValueSize() && "Invalid virtual align in concrete fragment!");

    for (uint64_t i = 0, e = FF.getSize() / FF.getValueSize(); i != e; ++i) {
      switch (FF.getValueSize()) {
      default: llvm_unreachable("Invalid size!");
      case 1: OW->Write8 (uint8_t (FF.getValue())); break;
      case 2: OW->Write16(uint16_t(FF.getValue())); break;
      case 4: OW->Write32(uint32_t(FF.getValue())); break;
      case 8: OW->Write64(uint64_t(FF.getValue())); break;
      }
    }
    break;
  }

  case MCFragment::FT_LEB: {
    const MCLEBFragment &LF = cast<MCLEBFragment>(F);
    OW->WriteBytes(LF.getContents().str());
    break;
  }

  case MCFragment::FT_Org: {
    ++stats::EmittedOrgFragments;
    const MCOrgFragment &OF = cast<MCOrgFragment>(F);

    for (uint64_t i = 0, e = FragmentSize; i != e; ++i)
      OW->Write8(uint8_t(OF.getValue()));

    break;
  }

  case MCFragment::FT_Dwarf: {
    const MCDwarfLineAddrFragment &OF = cast<MCDwarfLineAddrFragment>(F);
    OW->WriteBytes(OF.getContents().str());
    break;
  }
  case MCFragment::FT_DwarfFrame: {
    const MCDwarfCallFrameFragment &CF = cast<MCDwarfCallFrameFragment>(F);
    OW->WriteBytes(CF.getContents().str());
    break;
  }
  }

  assert(OW->getStream().tell() - Start == FragmentSize &&
         "The stream should advance by fragment size");
}

void MCAssembler::writeSectionData(const MCSectionData *SD,
                                   const MCAsmLayout &Layout) const {
  // Ignore virtual sections.
  if (SD->getSection().isVirtualSection()) {
    assert(Layout.getSectionFileSize(SD) == 0 && "Invalid size for section!");

    // Check that contents are only things legal inside a virtual section.
    for (MCSectionData::const_iterator it = SD->begin(),
           ie = SD->end(); it != ie; ++it) {
      switch (it->getKind()) {
      default: llvm_unreachable("Invalid fragment in virtual section!");
      case MCFragment::FT_Data: {
        // Check that we aren't trying to write a non-zero contents (or fixups)
        // into a virtual section. This is to support clients which use standard
        // directives to fill the contents of virtual sections.
        const MCDataFragment &DF = cast<MCDataFragment>(*it);
        assert(DF.fixup_begin() == DF.fixup_end() &&
               "Cannot have fixups in virtual section!");
        for (unsigned i = 0, e = DF.getContents().size(); i != e; ++i)
          if (DF.getContents()[i]) {
            if (auto *ELFSec = dyn_cast<const MCSectionELF>(&SD->getSection()))
              report_fatal_error("non-zero initializer found in section '" +
                  ELFSec->getSectionName() + "'");
            else
              report_fatal_error("non-zero initializer found in virtual section");
          }
        break;
      }
      case MCFragment::FT_Align:
        // Check that we aren't trying to write a non-zero value into a virtual
        // section.
        assert((cast<MCAlignFragment>(it)->getValueSize() == 0 ||
                cast<MCAlignFragment>(it)->getValue() == 0) &&
               "Invalid align in virtual section!");
        break;
      case MCFragment::FT_Fill:
        assert((cast<MCFillFragment>(it)->getValueSize() == 0 ||
                cast<MCFillFragment>(it)->getValue() == 0) &&
               "Invalid fill in virtual section!");
        break;
      }
    }

    return;
  }

  uint64_t Start = getWriter().getStream().tell();
  (void)Start;

  for (MCSectionData::const_iterator it = SD->begin(), ie = SD->end();
       it != ie; ++it)
    writeFragment(*this, Layout, *it);

  assert(getWriter().getStream().tell() - Start ==
         Layout.getSectionAddressSize(SD));
}

std::pair<uint64_t, bool> MCAssembler::handleFixup(const MCAsmLayout &Layout,
                                                   MCFragment &F,
                                                   const MCFixup &Fixup) {
  // Evaluate the fixup.
  MCValue Target;
  uint64_t FixedValue;
  bool IsPCRel = Backend.getFixupKindInfo(Fixup.getKind()).Flags &
                 MCFixupKindInfo::FKF_IsPCRel;
  if (!evaluateFixup(Layout, Fixup, &F, Target, FixedValue)) {
    // The fixup was unresolved, we need a relocation. Inform the object
    // writer of the relocation, and give it an opportunity to adjust the
    // fixup value if need be.
    getWriter().RecordRelocation(*this, Layout, &F, Fixup, Target, IsPCRel,
                                 FixedValue);
  }
  return std::make_pair(FixedValue, IsPCRel);
}

void MCAssembler::Finish() {
  DEBUG_WITH_TYPE("mc-dump", {
      llvm::errs() << "assembler backend - pre-layout\n--\n";
      dump(); });

  // Create the layout object.
  MCAsmLayout Layout(*this);

  // Create dummy fragments and assign section ordinals.
  unsigned SectionIndex = 0;
  for (MCAssembler::iterator it = begin(), ie = end(); it != ie; ++it) {
    // Create dummy fragments to eliminate any empty sections, this simplifies
    // layout.
    if (it->getFragmentList().empty())
      new MCDataFragment(it);

    it->setOrdinal(SectionIndex++);
  }

  // Assign layout order indices to sections and fragments.
  for (unsigned i = 0, e = Layout.getSectionOrder().size(); i != e; ++i) {
    MCSectionData *SD = Layout.getSectionOrder()[i];
    SD->setLayoutOrder(i);

    unsigned FragmentIndex = 0;
    for (MCSectionData::iterator iFrag = SD->begin(), iFragEnd = SD->end();
         iFrag != iFragEnd; ++iFrag)
      iFrag->setLayoutOrder(FragmentIndex++);
  }

  // Layout until everything fits.
  while (layoutOnce(Layout))
    continue;

  DEBUG_WITH_TYPE("mc-dump", {
      llvm::errs() << "assembler backend - post-relaxation\n--\n";
      dump(); });

  // Finalize the layout, including fragment lowering.
  finishLayout(Layout);

  DEBUG_WITH_TYPE("mc-dump", {
      llvm::errs() << "assembler backend - final-layout\n--\n";
      dump(); });

  uint64_t StartOffset = OS.tell();

  // Allow the object writer a chance to perform post-layout binding (for
  // example, to set the index fields in the symbol data).
  getWriter().ExecutePostLayoutBinding(*this, Layout);

  // Evaluate and apply the fixups, generating relocation entries as necessary.
  for (MCAssembler::iterator it = begin(), ie = end(); it != ie; ++it) {
    for (MCSectionData::iterator it2 = it->begin(),
           ie2 = it->end(); it2 != ie2; ++it2) {
      MCEncodedFragmentWithFixups *F =
        dyn_cast<MCEncodedFragmentWithFixups>(it2);
      if (F) {
        for (MCEncodedFragmentWithFixups::fixup_iterator it3 = F->fixup_begin(),
             ie3 = F->fixup_end(); it3 != ie3; ++it3) {
          MCFixup &Fixup = *it3;
          uint64_t FixedValue;
          bool IsPCRel;
          std::tie(FixedValue, IsPCRel) = handleFixup(Layout, *F, Fixup);
          getBackend().applyFixup(Fixup, F->getContents().data(),
                                  F->getContents().size(), FixedValue, IsPCRel);
        }
      }
    }
  }

  // Write the object file.
  getWriter().WriteObject(*this, Layout);

  stats::ObjectBytes += OS.tell() - StartOffset;
}

bool MCAssembler::fixupNeedsRelaxation(const MCFixup &Fixup,
                                       const MCRelaxableFragment *DF,
                                       const MCAsmLayout &Layout) const {
  // If we cannot resolve the fixup value, it requires relaxation.
  MCValue Target;
  uint64_t Value;
  if (!evaluateFixup(Layout, Fixup, DF, Target, Value))
    return true;

  return getBackend().fixupNeedsRelaxation(Fixup, Value, DF, Layout);
}

bool MCAssembler::fragmentNeedsRelaxation(const MCRelaxableFragment *F,
                                          const MCAsmLayout &Layout) const {
  // If this inst doesn't ever need relaxation, ignore it. This occurs when we
  // are intentionally pushing out inst fragments, or because we relaxed a
  // previous instruction to one that doesn't need relaxation.
  if (!getBackend().mayNeedRelaxation(F->getInst()))
    return false;

  for (MCRelaxableFragment::const_fixup_iterator it = F->fixup_begin(),
       ie = F->fixup_end(); it != ie; ++it)
    if (fixupNeedsRelaxation(*it, F, Layout))
      return true;

  return false;
}

bool MCAssembler::relaxInstruction(MCAsmLayout &Layout,
                                   MCRelaxableFragment &F) {
  if (!fragmentNeedsRelaxation(&F, Layout))
    return false;

  ++stats::RelaxedInstructions;

  // FIXME-PERF: We could immediately lower out instructions if we can tell
  // they are fully resolved, to avoid retesting on later passes.

  // Relax the fragment.

  MCInst Relaxed;
  getBackend().relaxInstruction(F.getInst(), Relaxed);

  // Encode the new instruction.
  //
  // FIXME-PERF: If it matters, we could let the target do this. It can
  // probably do so more efficiently in many cases.
  SmallVector<MCFixup, 4> Fixups;
  SmallString<256> Code;
  raw_svector_ostream VecOS(Code);
  getEmitter().EncodeInstruction(Relaxed, VecOS, Fixups, F.getSubtargetInfo());
  VecOS.flush();

  // Update the fragment.
  F.setInst(Relaxed);
  F.getContents() = Code;
  F.getFixups() = Fixups;

  return true;
}

bool MCAssembler::relaxLEB(MCAsmLayout &Layout, MCLEBFragment &LF) {
  uint64_t OldSize = LF.getContents().size();
  int64_t Value = LF.getValue().evaluateKnownAbsolute(Layout);
  SmallString<8> &Data = LF.getContents();
  Data.clear();
  raw_svector_ostream OSE(Data);
  if (LF.isSigned())
    encodeSLEB128(Value, OSE);
  else
    encodeULEB128(Value, OSE);
  OSE.flush();
  return OldSize != LF.getContents().size();
}

bool MCAssembler::relaxDwarfLineAddr(MCAsmLayout &Layout,
                                     MCDwarfLineAddrFragment &DF) {
  MCContext &Context = Layout.getAssembler().getContext();
  uint64_t OldSize = DF.getContents().size();
  int64_t AddrDelta = DF.getAddrDelta().evaluateKnownAbsolute(Layout);
  int64_t LineDelta;
  LineDelta = DF.getLineDelta();
  SmallString<8> &Data = DF.getContents();
  Data.clear();
  raw_svector_ostream OSE(Data);
  MCDwarfLineAddr::Encode(Context, LineDelta, AddrDelta, OSE);
  OSE.flush();
  return OldSize != Data.size();
}

bool MCAssembler::relaxDwarfCallFrameFragment(MCAsmLayout &Layout,
                                              MCDwarfCallFrameFragment &DF) {
  MCContext &Context = Layout.getAssembler().getContext();
  uint64_t OldSize = DF.getContents().size();
  int64_t AddrDelta = DF.getAddrDelta().evaluateKnownAbsolute(Layout);
  SmallString<8> &Data = DF.getContents();
  Data.clear();
  raw_svector_ostream OSE(Data);
  MCDwarfFrameEmitter::EncodeAdvanceLoc(Context, AddrDelta, OSE);
  OSE.flush();
  return OldSize != Data.size();
}

bool MCAssembler::layoutSectionOnce(MCAsmLayout &Layout, MCSectionData &SD) {
  // Holds the first fragment which needed relaxing during this layout. It will
  // remain NULL if none were relaxed.
  // When a fragment is relaxed, all the fragments following it should get
  // invalidated because their offset is going to change.
  MCFragment *FirstRelaxedFragment = nullptr;

  // Attempt to relax all the fragments in the section.
  for (MCSectionData::iterator I = SD.begin(), IE = SD.end(); I != IE; ++I) {
    // Check if this is a fragment that needs relaxation.
    bool RelaxedFrag = false;
    switch(I->getKind()) {
    default:
      break;
    case MCFragment::FT_Relaxable:
      assert(!getRelaxAll() &&
             "Did not expect a MCRelaxableFragment in RelaxAll mode");
      RelaxedFrag = relaxInstruction(Layout, *cast<MCRelaxableFragment>(I));
      break;
    case MCFragment::FT_Dwarf:
      RelaxedFrag = relaxDwarfLineAddr(Layout,
                                       *cast<MCDwarfLineAddrFragment>(I));
      break;
    case MCFragment::FT_DwarfFrame:
      RelaxedFrag =
        relaxDwarfCallFrameFragment(Layout,
                                    *cast<MCDwarfCallFrameFragment>(I));
      break;
    case MCFragment::FT_LEB:
      RelaxedFrag = relaxLEB(Layout, *cast<MCLEBFragment>(I));
      break;
    }
    if (RelaxedFrag && !FirstRelaxedFragment)
      FirstRelaxedFragment = I;
  }
  if (FirstRelaxedFragment) {
    Layout.invalidateFragmentsFrom(FirstRelaxedFragment);
    return true;
  }
  return false;
}

bool MCAssembler::layoutOnce(MCAsmLayout &Layout) {
  ++stats::RelaxationSteps;

  bool WasRelaxed = false;
  for (iterator it = begin(), ie = end(); it != ie; ++it) {
    MCSectionData &SD = *it;
    while (layoutSectionOnce(Layout, SD))
      WasRelaxed = true;
  }

  return WasRelaxed;
}

void MCAssembler::finishLayout(MCAsmLayout &Layout) {
  // The layout is done. Mark every fragment as valid.
  for (unsigned int i = 0, n = Layout.getSectionOrder().size(); i != n; ++i) {
    Layout.getFragmentOffset(&*Layout.getSectionOrder()[i]->rbegin());
  }
}

// Debugging methods

namespace llvm {

raw_ostream &operator<<(raw_ostream &OS, const MCFixup &AF) {
  OS << "<MCFixup" << " Offset:" << AF.getOffset()
     << " Value:" << *AF.getValue()
     << " Kind:" << AF.getKind() << ">";
  return OS;
}

}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void MCFragment::dump() {
  raw_ostream &OS = llvm::errs();

  OS << "<";
  switch (getKind()) {
  case MCFragment::FT_Align: OS << "MCAlignFragment"; break;
  case MCFragment::FT_Data:  OS << "MCDataFragment"; break;
  case MCFragment::FT_CompactEncodedInst:
    OS << "MCCompactEncodedInstFragment"; break;
  case MCFragment::FT_Fill:  OS << "MCFillFragment"; break;
  case MCFragment::FT_Relaxable:  OS << "MCRelaxableFragment"; break;
  case MCFragment::FT_Org:   OS << "MCOrgFragment"; break;
  case MCFragment::FT_Dwarf: OS << "MCDwarfFragment"; break;
  case MCFragment::FT_DwarfFrame: OS << "MCDwarfCallFrameFragment"; break;
  case MCFragment::FT_LEB:   OS << "MCLEBFragment"; break;
  }

  OS << "<MCFragment " << (void*) this << " LayoutOrder:" << LayoutOrder
     << " Offset:" << Offset
     << " HasInstructions:" << hasInstructions() 
     << " BundlePadding:" << static_cast<unsigned>(getBundlePadding()) << ">";

  switch (getKind()) {
  case MCFragment::FT_Align: {
    const MCAlignFragment *AF = cast<MCAlignFragment>(this);
    if (AF->hasEmitNops())
      OS << " (emit nops)";
    OS << "\n       ";
    OS << " Alignment:" << AF->getAlignment()
       << " Value:" << AF->getValue() << " ValueSize:" << AF->getValueSize()
       << " MaxBytesToEmit:" << AF->getMaxBytesToEmit() << ">";
    break;
  }
  case MCFragment::FT_Data:  {
    const MCDataFragment *DF = cast<MCDataFragment>(this);
    OS << "\n       ";
    OS << " Contents:[";
    const SmallVectorImpl<char> &Contents = DF->getContents();
    for (unsigned i = 0, e = Contents.size(); i != e; ++i) {
      if (i) OS << ",";
      OS << hexdigit((Contents[i] >> 4) & 0xF) << hexdigit(Contents[i] & 0xF);
    }
    OS << "] (" << Contents.size() << " bytes)";

    if (DF->fixup_begin() != DF->fixup_end()) {
      OS << ",\n       ";
      OS << " Fixups:[";
      for (MCDataFragment::const_fixup_iterator it = DF->fixup_begin(),
             ie = DF->fixup_end(); it != ie; ++it) {
        if (it != DF->fixup_begin()) OS << ",\n                ";
        OS << *it;
      }
      OS << "]";
    }
    break;
  }
  case MCFragment::FT_CompactEncodedInst: {
    const MCCompactEncodedInstFragment *CEIF =
      cast<MCCompactEncodedInstFragment>(this);
    OS << "\n       ";
    OS << " Contents:[";
    const SmallVectorImpl<char> &Contents = CEIF->getContents();
    for (unsigned i = 0, e = Contents.size(); i != e; ++i) {
      if (i) OS << ",";
      OS << hexdigit((Contents[i] >> 4) & 0xF) << hexdigit(Contents[i] & 0xF);
    }
    OS << "] (" << Contents.size() << " bytes)";
    break;
  }
  case MCFragment::FT_Fill:  {
    const MCFillFragment *FF = cast<MCFillFragment>(this);
    OS << " Value:" << FF->getValue() << " ValueSize:" << FF->getValueSize()
       << " Size:" << FF->getSize();
    break;
  }
  case MCFragment::FT_Relaxable:  {
    const MCRelaxableFragment *F = cast<MCRelaxableFragment>(this);
    OS << "\n       ";
    OS << " Inst:";
    F->getInst().dump_pretty(OS);
    break;
  }
  case MCFragment::FT_Org:  {
    const MCOrgFragment *OF = cast<MCOrgFragment>(this);
    OS << "\n       ";
    OS << " Offset:" << OF->getOffset() << " Value:" << OF->getValue();
    break;
  }
  case MCFragment::FT_Dwarf:  {
    const MCDwarfLineAddrFragment *OF = cast<MCDwarfLineAddrFragment>(this);
    OS << "\n       ";
    OS << " AddrDelta:" << OF->getAddrDelta()
       << " LineDelta:" << OF->getLineDelta();
    break;
  }
  case MCFragment::FT_DwarfFrame:  {
    const MCDwarfCallFrameFragment *CF = cast<MCDwarfCallFrameFragment>(this);
    OS << "\n       ";
    OS << " AddrDelta:" << CF->getAddrDelta();
    break;
  }
  case MCFragment::FT_LEB: {
    const MCLEBFragment *LF = cast<MCLEBFragment>(this);
    OS << "\n       ";
    OS << " Value:" << LF->getValue() << " Signed:" << LF->isSigned();
    break;
  }
  }
  OS << ">";
}

void MCSectionData::dump() {
  raw_ostream &OS = llvm::errs();

  OS << "<MCSectionData";
  OS << " Alignment:" << getAlignment()
     << " Fragments:[\n      ";
  for (iterator it = begin(), ie = end(); it != ie; ++it) {
    if (it != begin()) OS << ",\n      ";
    it->dump();
  }
  OS << "]>";
}

void MCSymbolData::dump() const {
  raw_ostream &OS = llvm::errs();

  OS << "<MCSymbolData Symbol:" << getSymbol()
     << " Fragment:" << getFragment();
  if (!isCommon())
    OS << " Offset:" << getOffset();
  OS << " Flags:" << getFlags() << " Index:" << getIndex();
  if (isCommon())
    OS << " (common, size:" << getCommonSize()
       << " align: " << getCommonAlignment() << ")";
  if (isExternal())
    OS << " (external)";
  if (isPrivateExtern())
    OS << " (private extern)";
  OS << ">";
}

void MCAssembler::dump() {
  raw_ostream &OS = llvm::errs();

  OS << "<MCAssembler\n";
  OS << "  Sections:[\n    ";
  for (iterator it = begin(), ie = end(); it != ie; ++it) {
    if (it != begin()) OS << ",\n    ";
    it->dump();
  }
  OS << "],\n";
  OS << "  Symbols:[";

  for (symbol_iterator it = symbol_begin(), ie = symbol_end(); it != ie; ++it) {
    if (it != symbol_begin()) OS << ",\n           ";
    it->dump();
  }
  OS << "]>\n";
}
#endif

// anchors for MC*Fragment vtables
void MCEncodedFragment::anchor() { }
void MCEncodedFragmentWithFixups::anchor() { }
void MCDataFragment::anchor() { }
void MCCompactEncodedInstFragment::anchor() { }
void MCRelaxableFragment::anchor() { }
void MCAlignFragment::anchor() { }
void MCFillFragment::anchor() { }
void MCOrgFragment::anchor() { }
void MCLEBFragment::anchor() { }
void MCDwarfLineAddrFragment::anchor() { }
void MCDwarfCallFrameFragment::anchor() { }
