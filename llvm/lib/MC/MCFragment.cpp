//===- lib/MC/MCFragment.cpp - Assembler Fragment Implementation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCFragment.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <utility>

using namespace llvm;

MCAsmLayout::MCAsmLayout(MCAssembler &Asm) : Assembler(Asm) {
  // Compute the section layout order. Virtual sections must go last.
  for (MCSection &Sec : Asm)
    if (!Sec.isVirtualSection())
      SectionOrder.push_back(&Sec);
  for (MCSection &Sec : Asm)
    if (Sec.isVirtualSection())
      SectionOrder.push_back(&Sec);
}

void MCAsmLayout::invalidateFragmentsFrom(MCFragment *F) {
  F->getParent()->setHasLayout(false);
}

// Simple getSymbolOffset helper for the non-variable case.
static bool getLabelOffset(const MCAsmLayout &Layout, const MCSymbol &S,
                           bool ReportError, uint64_t &Val) {
  if (!S.getFragment()) {
    if (ReportError)
      report_fatal_error("unable to evaluate offset to undefined symbol '" +
                         S.getName() + "'");
    return false;
  }
  Val = Layout.getFragmentOffset(S.getFragment()) + S.getOffset();
  return true;
}

static bool getSymbolOffsetImpl(const MCAsmLayout &Layout, const MCSymbol &S,
                                bool ReportError, uint64_t &Val) {
  if (!S.isVariable())
    return getLabelOffset(Layout, S, ReportError, Val);

  // If SD is a variable, evaluate it.
  MCValue Target;
  if (!S.getVariableValue()->evaluateAsValue(Target, Layout))
    report_fatal_error("unable to evaluate offset for variable '" +
                       S.getName() + "'");

  uint64_t Offset = Target.getConstant();

  const MCSymbolRefExpr *A = Target.getSymA();
  if (A) {
    uint64_t ValA;
    // FIXME: On most platforms, `Target`'s component symbols are labels from
    // having been simplified during evaluation, but on Mach-O they can be
    // variables due to PR19203. This, and the line below for `B` can be
    // restored to call `getLabelOffset` when PR19203 is fixed.
    if (!getSymbolOffsetImpl(Layout, A->getSymbol(), ReportError, ValA))
      return false;
    Offset += ValA;
  }

  const MCSymbolRefExpr *B = Target.getSymB();
  if (B) {
    uint64_t ValB;
    if (!getSymbolOffsetImpl(Layout, B->getSymbol(), ReportError, ValB))
      return false;
    Offset -= ValB;
  }

  Val = Offset;
  return true;
}

bool MCAsmLayout::getSymbolOffset(const MCSymbol &S, uint64_t &Val) const {
  return getSymbolOffsetImpl(*this, S, false, Val);
}

uint64_t MCAsmLayout::getSymbolOffset(const MCSymbol &S) const {
  uint64_t Val;
  getSymbolOffsetImpl(*this, S, true, Val);
  return Val;
}

const MCSymbol *MCAsmLayout::getBaseSymbol(const MCSymbol &Symbol) const {
  if (!Symbol.isVariable())
    return &Symbol;

  const MCExpr *Expr = Symbol.getVariableValue();
  MCValue Value;
  if (!Expr->evaluateAsValue(Value, *this)) {
    Assembler.getContext().reportError(
        Expr->getLoc(), "expression could not be evaluated");
    return nullptr;
  }

  const MCSymbolRefExpr *RefB = Value.getSymB();
  if (RefB) {
    Assembler.getContext().reportError(
        Expr->getLoc(), Twine("symbol '") + RefB->getSymbol().getName() +
                     "' could not be evaluated in a subtraction expression");
    return nullptr;
  }

  const MCSymbolRefExpr *A = Value.getSymA();
  if (!A)
    return nullptr;

  const MCSymbol &ASym = A->getSymbol();
  const MCAssembler &Asm = getAssembler();
  if (ASym.isCommon()) {
    Asm.getContext().reportError(Expr->getLoc(),
                                 "Common symbol '" + ASym.getName() +
                                     "' cannot be used in assignment expr");
    return nullptr;
  }

  return &ASym;
}

uint64_t MCAsmLayout::getSectionAddressSize(const MCSection *Sec) const {
  // The size is the last fragment's end offset.
  const MCFragment &F = *Sec->curFragList()->Tail;
  return getFragmentOffset(&F) + getAssembler().computeFragmentSize(*this, F);
}

uint64_t MCAsmLayout::getSectionFileSize(const MCSection *Sec) const {
  // Virtual sections have no file size.
  if (Sec->isVirtualSection())
    return 0;

  // Otherwise, the file size is the same as the address space size.
  return getSectionAddressSize(Sec);
}

/* *** */

MCFragment::MCFragment(FragmentType Kind, bool HasInstructions)
    : Kind(Kind), HasInstructions(HasInstructions), LinkerRelaxable(false) {}

void MCFragment::destroy() {
  switch (Kind) {
    case FT_Align:
      cast<MCAlignFragment>(this)->~MCAlignFragment();
      return;
    case FT_Data:
      cast<MCDataFragment>(this)->~MCDataFragment();
      return;
    case FT_CompactEncodedInst:
      cast<MCCompactEncodedInstFragment>(this)->~MCCompactEncodedInstFragment();
      return;
    case FT_Fill:
      cast<MCFillFragment>(this)->~MCFillFragment();
      return;
    case FT_Nops:
      cast<MCNopsFragment>(this)->~MCNopsFragment();
      return;
    case FT_Relaxable:
      cast<MCRelaxableFragment>(this)->~MCRelaxableFragment();
      return;
    case FT_Org:
      cast<MCOrgFragment>(this)->~MCOrgFragment();
      return;
    case FT_Dwarf:
      cast<MCDwarfLineAddrFragment>(this)->~MCDwarfLineAddrFragment();
      return;
    case FT_DwarfFrame:
      cast<MCDwarfCallFrameFragment>(this)->~MCDwarfCallFrameFragment();
      return;
    case FT_LEB:
      cast<MCLEBFragment>(this)->~MCLEBFragment();
      return;
    case FT_BoundaryAlign:
      cast<MCBoundaryAlignFragment>(this)->~MCBoundaryAlignFragment();
      return;
    case FT_SymbolId:
      cast<MCSymbolIdFragment>(this)->~MCSymbolIdFragment();
      return;
    case FT_CVInlineLines:
      cast<MCCVInlineLineTableFragment>(this)->~MCCVInlineLineTableFragment();
      return;
    case FT_CVDefRange:
      cast<MCCVDefRangeFragment>(this)->~MCCVDefRangeFragment();
      return;
    case FT_PseudoProbe:
      cast<MCPseudoProbeAddrFragment>(this)->~MCPseudoProbeAddrFragment();
      return;
    case FT_Dummy:
      cast<MCDummyFragment>(this)->~MCDummyFragment();
      return;
  }
}

const MCSymbol *MCFragment::getAtom() const {
  return cast<MCSectionMachO>(Parent)->getAtom(LayoutOrder);
}

// Debugging methods

namespace llvm {

raw_ostream &operator<<(raw_ostream &OS, const MCFixup &AF) {
  OS << "<MCFixup" << " Offset:" << AF.getOffset()
     << " Value:" << *AF.getValue()
     << " Kind:" << AF.getKind() << ">";
  return OS;
}

} // end namespace llvm

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void MCFragment::dump() const {
  raw_ostream &OS = errs();

  OS << "<";
  switch (getKind()) {
  case MCFragment::FT_Align: OS << "MCAlignFragment"; break;
  case MCFragment::FT_Data:  OS << "MCDataFragment"; break;
  case MCFragment::FT_CompactEncodedInst:
    OS << "MCCompactEncodedInstFragment"; break;
  case MCFragment::FT_Fill:  OS << "MCFillFragment"; break;
  case MCFragment::FT_Nops:
    OS << "MCFNopsFragment";
    break;
  case MCFragment::FT_Relaxable:  OS << "MCRelaxableFragment"; break;
  case MCFragment::FT_Org:   OS << "MCOrgFragment"; break;
  case MCFragment::FT_Dwarf: OS << "MCDwarfFragment"; break;
  case MCFragment::FT_DwarfFrame: OS << "MCDwarfCallFrameFragment"; break;
  case MCFragment::FT_LEB:   OS << "MCLEBFragment"; break;
  case MCFragment::FT_BoundaryAlign: OS<<"MCBoundaryAlignFragment"; break;
  case MCFragment::FT_SymbolId:    OS << "MCSymbolIdFragment"; break;
  case MCFragment::FT_CVInlineLines: OS << "MCCVInlineLineTableFragment"; break;
  case MCFragment::FT_CVDefRange: OS << "MCCVDefRangeTableFragment"; break;
  case MCFragment::FT_PseudoProbe:
    OS << "MCPseudoProbe";
    break;
  case MCFragment::FT_Dummy: OS << "MCDummyFragment"; break;
  }

  OS << "<MCFragment " << (const void *)this << " LayoutOrder:" << LayoutOrder
     << " Offset:" << Offset << " HasInstructions:" << hasInstructions();
  if (const auto *EF = dyn_cast<MCEncodedFragment>(this))
    OS << " BundlePadding:" << static_cast<unsigned>(EF->getBundlePadding());
  OS << ">";

  switch (getKind()) {
  case MCFragment::FT_Align: {
    const auto *AF = cast<MCAlignFragment>(this);
    if (AF->hasEmitNops())
      OS << " (emit nops)";
    OS << "\n       ";
    OS << " Alignment:" << AF->getAlignment().value()
       << " Value:" << AF->getValue() << " ValueSize:" << AF->getValueSize()
       << " MaxBytesToEmit:" << AF->getMaxBytesToEmit() << ">";
    break;
  }
  case MCFragment::FT_Data:  {
    const auto *DF = cast<MCDataFragment>(this);
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
    const auto *CEIF =
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
    const auto *FF = cast<MCFillFragment>(this);
    OS << " Value:" << static_cast<unsigned>(FF->getValue())
       << " ValueSize:" << static_cast<unsigned>(FF->getValueSize())
       << " NumValues:" << FF->getNumValues();
    break;
  }
  case MCFragment::FT_Nops: {
    const auto *NF = cast<MCNopsFragment>(this);
    OS << " NumBytes:" << NF->getNumBytes()
       << " ControlledNopLength:" << NF->getControlledNopLength();
    break;
  }
  case MCFragment::FT_Relaxable:  {
    const auto *F = cast<MCRelaxableFragment>(this);
    OS << "\n       ";
    OS << " Inst:";
    F->getInst().dump_pretty(OS);
    OS << " (" << F->getContents().size() << " bytes)";
    break;
  }
  case MCFragment::FT_Org:  {
    const auto *OF = cast<MCOrgFragment>(this);
    OS << "\n       ";
    OS << " Offset:" << OF->getOffset()
       << " Value:" << static_cast<unsigned>(OF->getValue());
    break;
  }
  case MCFragment::FT_Dwarf:  {
    const auto *OF = cast<MCDwarfLineAddrFragment>(this);
    OS << "\n       ";
    OS << " AddrDelta:" << OF->getAddrDelta()
       << " LineDelta:" << OF->getLineDelta();
    break;
  }
  case MCFragment::FT_DwarfFrame:  {
    const auto *CF = cast<MCDwarfCallFrameFragment>(this);
    OS << "\n       ";
    OS << " AddrDelta:" << CF->getAddrDelta();
    break;
  }
  case MCFragment::FT_LEB: {
    const auto *LF = cast<MCLEBFragment>(this);
    OS << "\n       ";
    OS << " Value:" << LF->getValue() << " Signed:" << LF->isSigned();
    break;
  }
  case MCFragment::FT_BoundaryAlign: {
    const auto *BF = cast<MCBoundaryAlignFragment>(this);
    OS << "\n       ";
    OS << " BoundarySize:" << BF->getAlignment().value()
       << " LastFragment:" << BF->getLastFragment()
       << " Size:" << BF->getSize();
    break;
  }
  case MCFragment::FT_SymbolId: {
    const auto *F = cast<MCSymbolIdFragment>(this);
    OS << "\n       ";
    OS << " Sym:" << F->getSymbol();
    break;
  }
  case MCFragment::FT_CVInlineLines: {
    const auto *F = cast<MCCVInlineLineTableFragment>(this);
    OS << "\n       ";
    OS << " Sym:" << *F->getFnStartSym();
    break;
  }
  case MCFragment::FT_CVDefRange: {
    const auto *F = cast<MCCVDefRangeFragment>(this);
    OS << "\n       ";
    for (std::pair<const MCSymbol *, const MCSymbol *> RangeStartEnd :
         F->getRanges()) {
      OS << " RangeStart:" << RangeStartEnd.first;
      OS << " RangeEnd:" << RangeStartEnd.second;
    }
    break;
  }
  case MCFragment::FT_PseudoProbe: {
    const auto *OF = cast<MCPseudoProbeAddrFragment>(this);
    OS << "\n       ";
    OS << " AddrDelta:" << OF->getAddrDelta();
    break;
  }
  case MCFragment::FT_Dummy:
    break;
  }
  OS << ">";
}
#endif
