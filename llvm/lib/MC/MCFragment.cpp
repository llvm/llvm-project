//===- lib/MC/MCFragment.cpp - Assembler Fragment Implementation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <type_traits>
#include <utility>

using namespace llvm;

static_assert(std::is_trivially_destructible_v<MCDataFragment>,
              "fragment classes must be trivially destructible");

MCFragment::MCFragment(FragmentType Kind, bool HasInstructions)
    : Kind(Kind), HasInstructions(HasInstructions), AlignToBundleEnd(false),
      LinkerRelaxable(false), AllowAutoPadding(false) {}

const MCSymbol *MCFragment::getAtom() const {
  return cast<MCSectionMachO>(Parent)->getAtom(LayoutOrder);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void MCFragment::dump() const {
  raw_ostream &OS = errs();

  OS << Offset << ' ';
  switch (getKind()) {
    // clang-format off
  case MCFragment::FT_Align:         OS << "Align"; break;
  case MCFragment::FT_Data:          OS << "Data"; break;
  case MCFragment::FT_Fill:          OS << "Fill"; break;
  case MCFragment::FT_Nops:          OS << "Nops"; break;
  case MCFragment::FT_Relaxable:     OS << "Relaxable"; break;
  case MCFragment::FT_Org:           OS << "Org"; break;
  case MCFragment::FT_Dwarf:         OS << "Dwarf"; break;
  case MCFragment::FT_DwarfFrame:    OS << "DwarfCallFrame"; break;
  case MCFragment::FT_LEB:           OS << "LEB"; break;
  case MCFragment::FT_BoundaryAlign: OS<<"BoundaryAlign"; break;
  case MCFragment::FT_SymbolId:      OS << "SymbolId"; break;
  case MCFragment::FT_CVInlineLines: OS << "CVInlineLineTable"; break;
  case MCFragment::FT_CVDefRange:    OS << "CVDefRangeTable"; break;
  case MCFragment::FT_PseudoProbe:   OS << "PseudoProbe"; break;
    // clang-format on
  }

  if (const auto *EF = dyn_cast<MCEncodedFragment>(this))
    if (auto Pad = static_cast<unsigned>(EF->getBundlePadding()))
      OS << " BundlePadding:" << Pad;

  auto printFixups = [&](llvm::ArrayRef<MCFixup> Fixups) {
    if (Fixups.empty())
      return;
    for (auto [I, F] : llvm::enumerate(Fixups)) {
      OS << "\n  Fixup @" << F.getOffset() << " Value:";
      F.getValue()->print(OS, nullptr);
      OS << " Kind:" << F.getKind();
      if (F.isLinkerRelaxable())
        OS << " LinkerRelaxable";
    }
  };

  switch (getKind()) {
  case MCFragment::FT_Align: {
    const auto *AF = cast<MCAlignFragment>(this);
    OS << " Align:" << AF->getAlignment().value() << " Fill:" << AF->getFill()
       << " FillLen:" << unsigned(AF->getFillLen())
       << " MaxBytesToEmit:" << AF->getMaxBytesToEmit();
    if (AF->hasEmitNops())
      OS << " Nops";
    break;
  }
  case MCFragment::FT_Data:  {
    const auto *F = cast<MCDataFragment>(this);
    if (F->isLinkerRelaxable())
      OS << " LinkerRelaxable";
    auto Contents = F->getContents();
    OS << " Size:" << Contents.size() << " [";
    for (unsigned i = 0, e = Contents.size(); i != e; ++i) {
      if (i) OS << ",";
      OS << format("%02x", uint8_t(Contents[i]));
    }
    OS << ']';
    printFixups(F->getFixups());
    break;
  }
  case MCFragment::FT_Fill:  {
    const auto *FF = cast<MCFillFragment>(this);
    OS << " Value:" << static_cast<unsigned>(FF->getValue())
       << " ValueSize:" << static_cast<unsigned>(FF->getValueSize())
       << " NumValues:";
    FF->getNumValues().print(OS, nullptr);
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
    if (F->isLinkerRelaxable())
      OS << " LinkerRelaxable";
    OS << " Size:" << F->getContents().size() << ' ';
    F->getInst().dump_pretty(OS);
    printFixups(F->getFixups());
    break;
  }
  case MCFragment::FT_Org:  {
    const auto *OF = cast<MCOrgFragment>(this);
    OS << " Offset:";
    OF->getOffset().print(OS, nullptr);
    OS << " Value:" << static_cast<unsigned>(OF->getValue());
    break;
  }
  case MCFragment::FT_Dwarf:  {
    const auto *OF = cast<MCDwarfLineAddrFragment>(this);
    OS << " AddrDelta:";
    OF->getAddrDelta().print(OS, nullptr);
    OS << " LineDelta:" << OF->getLineDelta();
    break;
  }
  case MCFragment::FT_DwarfFrame:  {
    const auto *CF = cast<MCDwarfCallFrameFragment>(this);
    OS << " AddrDelta:";
    CF->getAddrDelta().print(OS, nullptr);
    break;
  }
  case MCFragment::FT_LEB: {
    const auto *LF = cast<MCLEBFragment>(this);
    OS << " Value:";
    LF->getValue().print(OS, nullptr);
    OS << " Signed:" << LF->isSigned();
    break;
  }
  case MCFragment::FT_BoundaryAlign: {
    const auto *BF = cast<MCBoundaryAlignFragment>(this);
    OS << " BoundarySize:" << BF->getAlignment().value()
       << " LastFragment:" << BF->getLastFragment()
       << " Size:" << BF->getSize();
    break;
  }
  case MCFragment::FT_SymbolId: {
    const auto *F = cast<MCSymbolIdFragment>(this);
    OS << " Sym:" << F->getSymbol();
    break;
  }
  case MCFragment::FT_CVInlineLines: {
    const auto *F = cast<MCCVInlineLineTableFragment>(this);
    OS << " Sym:" << *F->getFnStartSym();
    break;
  }
  case MCFragment::FT_CVDefRange: {
    const auto *F = cast<MCCVDefRangeFragment>(this);
    OS << "\n   ";
    for (std::pair<const MCSymbol *, const MCSymbol *> RangeStartEnd :
         F->getRanges()) {
      OS << " RangeStart:" << RangeStartEnd.first;
      OS << " RangeEnd:" << RangeStartEnd.second;
    }
    break;
  }
  case MCFragment::FT_PseudoProbe: {
    const auto *OF = cast<MCPseudoProbeAddrFragment>(this);
    OS << " AddrDelta:";
    OF->getAddrDelta().print(OS, nullptr);
    break;
  }
  }
}
#endif
