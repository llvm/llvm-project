//===- lib/MC/MCAssembler.cpp - Assembler Backend Implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "assembler"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/LEB128.h"

using namespace llvm;

namespace {
namespace stats {
STATISTIC(EmittedFragments, "Number of emitted assembler fragments");
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

bool MCAsmLayout::isFragmentUpToDate(const MCFragment *F) const {
  const MCSectionData &SD = *F->getParent();
  const MCFragment *LastValid = LastValidFragment.lookup(&SD);
  if (!LastValid)
    return false;
  assert(LastValid->getParent() == F->getParent());
  return F->getLayoutOrder() <= LastValid->getLayoutOrder();
}

void MCAsmLayout::Invalidate(MCFragment *F) {
  // If this fragment wasn't already up-to-date, we don't need to do anything.
  if (!isFragmentUpToDate(F))
    return;

  // Otherwise, reset the last valid fragment to this fragment.
  const MCSectionData &SD = *F->getParent();
  LastValidFragment[&SD] = F;
}

void MCAsmLayout::EnsureValid(const MCFragment *F) const {
  MCSectionData &SD = *F->getParent();

  MCFragment *Cur = LastValidFragment[&SD];
  if (!Cur)
    Cur = &*SD.begin();
  else
    Cur = Cur->getNextNode();

  // Advance the layout position until the fragment is up-to-date.
  while (!isFragmentUpToDate(F)) {
    const_cast<MCAsmLayout*>(this)->LayoutFragment(Cur);
    Cur = Cur->getNextNode();
  }
}

uint64_t MCAsmLayout::getFragmentOffset(const MCFragment *F) const {
  EnsureValid(F);
  assert(F->Offset != ~UINT64_C(0) && "Address not set!");
  return F->Offset;
}

uint64_t MCAsmLayout::getSymbolOffset(const MCSymbolData *SD) const {
  const MCSymbol &S = SD->getSymbol();

  // If this is a variable, then recursively evaluate now.
  if (S.isVariable()) {
    MCValue Target;
    if (!S.getVariableValue()->EvaluateAsRelocatable(Target, *this))
      report_fatal_error("unable to evaluate offset for variable '" +
                         S.getName() + "'");

    // Verify that any used symbols are defined.
    if (Target.getSymA() && Target.getSymA()->getSymbol().isUndefined())
      report_fatal_error("unable to evaluate offset to undefined symbol '" +
                         Target.getSymA()->getSymbol().getName() + "'");
    if (Target.getSymB() && Target.getSymB()->getSymbol().isUndefined())
      report_fatal_error("unable to evaluate offset to undefined symbol '" +
                         Target.getSymB()->getSymbol().getName() + "'");

    uint64_t Offset = Target.getConstant();
    if (Target.getSymA())
      Offset += getSymbolOffset(&Assembler.getSymbolData(
                                  Target.getSymA()->getSymbol()));
    if (Target.getSymB())
      Offset -= getSymbolOffset(&Assembler.getSymbolData(
                                  Target.getSymB()->getSymbol()));
    return Offset;
  }

  assert(SD->getFragment() && "Invalid getOffset() on undefined symbol!");
  return getFragmentOffset(SD->getFragment()) + SD->getOffset();
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

/* *** */

MCFragment::MCFragment() : Kind(FragmentType(~0)) {
}

MCFragment::~MCFragment() {
}

MCFragment::MCFragment(FragmentType _Kind, MCSectionData *_Parent)
  : Kind(_Kind), Parent(_Parent), Atom(0), Offset(~UINT64_C(0))
{
  if (Parent)
    Parent->getFragmentList().push_back(this);
}

/* *** */

MCSectionData::MCSectionData() : Section(0) {}

MCSectionData::MCSectionData(const MCSection &_Section, MCAssembler *A)
  : Section(&_Section),
    Ordinal(~UINT32_C(0)),
    Alignment(1),
    HasInstructions(false)
{
  if (A)
    A->getSectionList().push_back(this);
}

/* *** */

MCSymbolData::MCSymbolData() : Symbol(0) {}

MCSymbolData::MCSymbolData(const MCSymbol &_Symbol, MCFragment *_Fragment,
                           uint64_t _Offset, MCAssembler *A)
  : Symbol(&_Symbol), Fragment(_Fragment), Offset(_Offset),
    IsExternal(false), IsPrivateExtern(false),
    CommonSize(0), SymbolSize(0), CommonAlign(0),
    Flags(0), Index(0)
{
  if (A)
    A->getSymbolList().push_back(this);
}

/* *** */

MCAssembler::MCAssembler(MCContext &Context_, MCAsmBackend &Backend_,
                         MCCodeEmitter &Emitter_, MCObjectWriter &Writer_,
                         raw_ostream &OS_)
  : Context(Context_), Backend(Backend_), Emitter(Emitter_), Writer(Writer_),
    OS(OS_), RelaxAll(false), NoExecStack(false), SubsectionsViaSymbols(false) {
}

MCAssembler::~MCAssembler() {
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
    return 0;

  // Non-linker visible symbols in sections which can't be atomized have no
  // defining atom.
  if (!getBackend().isSectionAtomizable(
        SD->getFragment()->getParent()->getSection()))
    return 0;

  // Otherwise, return the atom for the containing fragment.
  return SD->getFragment()->getAtom();
}

bool MCAssembler::evaluateFixup(const MCAsmLayout &Layout,
                                const MCFixup &Fixup, const MCFragment *DF,
                                MCValue &Target, uint64_t &Value) const {
  ++stats::evaluateFixup;

  if (!Fixup.getValue()->EvaluateAsRelocatable(Target, Layout))
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
    return cast<MCDataFragment>(F).getContents().size();
  case MCFragment::FT_Fill:
    return cast<MCFillFragment>(F).getSize();
  case MCFragment::FT_Inst:
    return cast<MCInstFragment>(F).getInstSize();

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
    MCOrgFragment &OF = cast<MCOrgFragment>(F);
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

void MCAsmLayout::LayoutFragment(MCFragment *F) {
  MCFragment *Prev = F->getPrevNode();

  // We should never try to recompute something which is up-to-date.
  assert(!isFragmentUpToDate(F) && "Attempt to recompute up-to-date fragment!");
  // We should never try to compute the fragment layout if it's predecessor
  // isn't up-to-date.
  assert((!Prev || isFragmentUpToDate(Prev)) &&
         "Attempt to compute fragment before it's predecessor!");

  ++stats::FragmentLayouts;

  // Compute fragment offset and size.
  uint64_t Offset = 0;
  if (Prev)
    Offset += Prev->Offset + getAssembler().computeFragmentSize(*this, *Prev);

  F->Offset = Offset;
  LastValidFragment[F->getParent()] = F;
}

/// WriteFragmentData - Write the \p F data to the output file.
static void WriteFragmentData(const MCAssembler &Asm, const MCAsmLayout &Layout,
                              const MCFragment &F) {
  MCObjectWriter *OW = &Asm.getWriter();
  uint64_t Start = OW->getStream().tell();
  (void) Start;

  ++stats::EmittedFragments;

  // FIXME: Embed in fragments instead?
  uint64_t FragmentSize = Asm.computeFragmentSize(Layout, F);
  switch (F.getKind()) {
  case MCFragment::FT_Align: {
    MCAlignFragment &AF = cast<MCAlignFragment>(F);
    uint64_t Count = FragmentSize / AF.getValueSize();

    assert(AF.getValueSize() && "Invalid virtual align in concrete fragment!");

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

  case MCFragment::FT_Data: {
    MCDataFragment &DF = cast<MCDataFragment>(F);
    assert(FragmentSize == DF.getContents().size() && "Invalid size!");
    OW->WriteBytes(DF.getContents().str());
    break;
  }

  case MCFragment::FT_Fill: {
    MCFillFragment &FF = cast<MCFillFragment>(F);

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

  case MCFragment::FT_Inst: {
    MCInstFragment &IF = cast<MCInstFragment>(F);
    OW->WriteBytes(StringRef(IF.getCode().begin(), IF.getCode().size()));
    break;
  }

  case MCFragment::FT_LEB: {
    MCLEBFragment &LF = cast<MCLEBFragment>(F);
    OW->WriteBytes(LF.getContents().str());
    break;
  }

  case MCFragment::FT_Org: {
    MCOrgFragment &OF = cast<MCOrgFragment>(F);

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

  assert(OW->getStream().tell() - Start == FragmentSize);
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
        MCDataFragment &DF = cast<MCDataFragment>(*it);
        assert(DF.fixup_begin() == DF.fixup_end() &&
               "Cannot have fixups in virtual section!");
        for (unsigned i = 0, e = DF.getContents().size(); i != e; ++i)
          assert(DF.getContents()[i] == 0 &&
                 "Invalid data value for virtual section!");
        break;
      }
      case MCFragment::FT_Align:
        // Check that we aren't trying to write a non-zero value into a virtual
        // section.
        assert((!cast<MCAlignFragment>(it)->getValueSize() ||
                !cast<MCAlignFragment>(it)->getValue()) &&
               "Invalid align in virtual section!");
        break;
      case MCFragment::FT_Fill:
        assert(!cast<MCFillFragment>(it)->getValueSize() &&
               "Invalid fill in virtual section!");
        break;
      }
    }

    return;
  }

  uint64_t Start = getWriter().getStream().tell();
  (void)Start;

  for (MCSectionData::const_iterator it = SD->begin(),
         ie = SD->end(); it != ie; ++it)
    WriteFragmentData(*this, Layout, *it);

  assert(getWriter().getStream().tell() - Start ==
         Layout.getSectionAddressSize(SD));
}


uint64_t MCAssembler::handleFixup(const MCAsmLayout &Layout,
                                  MCFragment &F,
                                  const MCFixup &Fixup) {
   // Evaluate the fixup.
   MCValue Target;
   uint64_t FixedValue;
   if (!evaluateFixup(Layout, Fixup, &F, Target, FixedValue)) {
     // The fixup was unresolved, we need a relocation. Inform the object
     // writer of the relocation, and give it an opportunity to adjust the
     // fixup value if need be.
     getWriter().RecordRelocation(*this, Layout, &F, Fixup, Target, FixedValue);
   }
   return FixedValue;
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
    for (MCSectionData::iterator it2 = SD->begin(),
           ie2 = SD->end(); it2 != ie2; ++it2)
      it2->setLayoutOrder(FragmentIndex++);
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
      MCDataFragment *DF = dyn_cast<MCDataFragment>(it2);
      if (DF) {
        for (MCDataFragment::fixup_iterator it3 = DF->fixup_begin(),
               ie3 = DF->fixup_end(); it3 != ie3; ++it3) {
          MCFixup &Fixup = *it3;
          uint64_t FixedValue = handleFixup(Layout, *DF, Fixup);
          getBackend().applyFixup(Fixup, DF->getContents().data(),
                                  DF->getContents().size(), FixedValue);
        }
      }
      MCInstFragment *IF = dyn_cast<MCInstFragment>(it2);
      if (IF) {
        for (MCInstFragment::fixup_iterator it3 = IF->fixup_begin(),
               ie3 = IF->fixup_end(); it3 != ie3; ++it3) {
          MCFixup &Fixup = *it3;
          uint64_t FixedValue = handleFixup(Layout, *IF, Fixup);
          getBackend().applyFixup(Fixup, IF->getCode().data(),
                                  IF->getCode().size(), FixedValue);
        }
      }
    }
  }

  // Write the object file.
  getWriter().WriteObject(*this, Layout);

  stats::ObjectBytes += OS.tell() - StartOffset;
}

bool MCAssembler::fixupNeedsRelaxation(const MCFixup &Fixup,
                                       const MCInstFragment *DF,
                                       const MCAsmLayout &Layout) const {
  if (getRelaxAll())
    return true;

  // If we cannot resolve the fixup value, it requires relaxation.
  MCValue Target;
  uint64_t Value;
  if (!evaluateFixup(Layout, Fixup, DF, Target, Value))
    return true;

  return getBackend().fixupNeedsRelaxation(Fixup, Value, DF, Layout);
}

bool MCAssembler::fragmentNeedsRelaxation(const MCInstFragment *IF,
                                          const MCAsmLayout &Layout) const {
  // If this inst doesn't ever need relaxation, ignore it. This occurs when we
  // are intentionally pushing out inst fragments, or because we relaxed a
  // previous instruction to one that doesn't need relaxation.
  if (!getBackend().mayNeedRelaxation(IF->getInst()))
    return false;

  for (MCInstFragment::const_fixup_iterator it = IF->fixup_begin(),
         ie = IF->fixup_end(); it != ie; ++it)
    if (fixupNeedsRelaxation(*it, IF, Layout))
      return true;

  return false;
}

bool MCAssembler::relaxInstruction(MCAsmLayout &Layout,
                                   MCInstFragment &IF) {
  if (!fragmentNeedsRelaxation(&IF, Layout))
    return false;

  ++stats::RelaxedInstructions;

  // FIXME-PERF: We could immediately lower out instructions if we can tell
  // they are fully resolved, to avoid retesting on later passes.

  // Relax the fragment.

  MCInst Relaxed;
  getBackend().relaxInstruction(IF.getInst(), Relaxed);

  // Encode the new instruction.
  //
  // FIXME-PERF: If it matters, we could let the target do this. It can
  // probably do so more efficiently in many cases.
  SmallVector<MCFixup, 4> Fixups;
  SmallString<256> Code;
  raw_svector_ostream VecOS(Code);
  getEmitter().EncodeInstruction(Relaxed, VecOS, Fixups);
  VecOS.flush();

  // Update the instruction fragment.
  IF.setInst(Relaxed);
  IF.getCode() = Code;
  IF.getFixups().clear();
  // FIXME: Eliminate copy.
  for (unsigned i = 0, e = Fixups.size(); i != e; ++i)
    IF.getFixups().push_back(Fixups[i]);

  return true;
}

bool MCAssembler::relaxLEB(MCAsmLayout &Layout, MCLEBFragment &LF) {
  int64_t Value = 0;
  uint64_t OldSize = LF.getContents().size();
  bool IsAbs = LF.getValue().EvaluateAsAbsolute(Value, Layout);
  (void)IsAbs;
  assert(IsAbs);
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
  int64_t AddrDelta = 0;
  uint64_t OldSize = DF.getContents().size();
  bool IsAbs = DF.getAddrDelta().EvaluateAsAbsolute(AddrDelta, Layout);
  (void)IsAbs;
  assert(IsAbs);
  int64_t LineDelta;
  LineDelta = DF.getLineDelta();
  SmallString<8> &Data = DF.getContents();
  Data.clear();
  raw_svector_ostream OSE(Data);
  MCDwarfLineAddr::Encode(LineDelta, AddrDelta, OSE);
  OSE.flush();
  return OldSize != Data.size();
}

bool MCAssembler::relaxDwarfCallFrameFragment(MCAsmLayout &Layout,
                                              MCDwarfCallFrameFragment &DF) {
  int64_t AddrDelta = 0;
  uint64_t OldSize = DF.getContents().size();
  bool IsAbs = DF.getAddrDelta().EvaluateAsAbsolute(AddrDelta, Layout);
  (void)IsAbs;
  assert(IsAbs);
  SmallString<8> &Data = DF.getContents();
  Data.clear();
  raw_svector_ostream OSE(Data);
  MCDwarfFrameEmitter::EncodeAdvanceLoc(AddrDelta, OSE);
  OSE.flush();
  return OldSize != Data.size();
}

bool MCAssembler::layoutSectionOnce(MCAsmLayout &Layout,
                                    MCSectionData &SD) {
  MCFragment *FirstInvalidFragment = NULL;
  // Scan for fragments that need relaxation.
  for (MCSectionData::iterator it2 = SD.begin(),
         ie2 = SD.end(); it2 != ie2; ++it2) {
    // Check if this is an fragment that needs relaxation.
    bool relaxedFrag = false;
    switch(it2->getKind()) {
    default:
          break;
    case MCFragment::FT_Inst:
      relaxedFrag = relaxInstruction(Layout, *cast<MCInstFragment>(it2));
      break;
    case MCFragment::FT_Dwarf:
      relaxedFrag = relaxDwarfLineAddr(Layout,
                                       *cast<MCDwarfLineAddrFragment>(it2));
      break;
    case MCFragment::FT_DwarfFrame:
      relaxedFrag =
        relaxDwarfCallFrameFragment(Layout,
                                    *cast<MCDwarfCallFrameFragment>(it2));
      break;
    case MCFragment::FT_LEB:
      relaxedFrag = relaxLEB(Layout, *cast<MCLEBFragment>(it2));
      break;
    }
    // Update the layout, and remember that we relaxed.
    if (relaxedFrag && !FirstInvalidFragment)
      FirstInvalidFragment = it2;
  }
  if (FirstInvalidFragment) {
    Layout.Invalidate(FirstInvalidFragment);
    return true;
  }
  return false;
}

bool MCAssembler::layoutOnce(MCAsmLayout &Layout) {
  ++stats::RelaxationSteps;

  bool WasRelaxed = false;
  for (iterator it = begin(), ie = end(); it != ie; ++it) {
    MCSectionData &SD = *it;
    while(layoutSectionOnce(Layout, SD))
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
  case MCFragment::FT_Fill:  OS << "MCFillFragment"; break;
  case MCFragment::FT_Inst:  OS << "MCInstFragment"; break;
  case MCFragment::FT_Org:   OS << "MCOrgFragment"; break;
  case MCFragment::FT_Dwarf: OS << "MCDwarfFragment"; break;
  case MCFragment::FT_DwarfFrame: OS << "MCDwarfCallFrameFragment"; break;
  case MCFragment::FT_LEB:   OS << "MCLEBFragment"; break;
  }

  OS << "<MCFragment " << (void*) this << " LayoutOrder:" << LayoutOrder
     << " Offset:" << Offset << ">";

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

    if (!DF->getFixups().empty()) {
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
  case MCFragment::FT_Fill:  {
    const MCFillFragment *FF = cast<MCFillFragment>(this);
    OS << " Value:" << FF->getValue() << " ValueSize:" << FF->getValueSize()
       << " Size:" << FF->getSize();
    break;
  }
  case MCFragment::FT_Inst:  {
    const MCInstFragment *IF = cast<MCInstFragment>(this);
    OS << "\n       ";
    OS << " Inst:";
    IF->getInst().dump_pretty(OS);
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
  OS << " Alignment:" << getAlignment() << " Fragments:[\n      ";
  for (iterator it = begin(), ie = end(); it != ie; ++it) {
    if (it != begin()) OS << ",\n      ";
    it->dump();
  }
  OS << "]>";
}

void MCSymbolData::dump() {
  raw_ostream &OS = llvm::errs();

  OS << "<MCSymbolData Symbol:" << getSymbol()
     << " Fragment:" << getFragment() << " Offset:" << getOffset()
     << " Flags:" << getFlags() << " Index:" << getIndex();
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
void MCDataFragment::anchor() { }
void MCInstFragment::anchor() { }
void MCAlignFragment::anchor() { }
void MCFillFragment::anchor() { }
void MCOrgFragment::anchor() { }
void MCLEBFragment::anchor() { }
void MCDwarfLineAddrFragment::anchor() { }
void MCDwarfCallFrameFragment::anchor() { }
