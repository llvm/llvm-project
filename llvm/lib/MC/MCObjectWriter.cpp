//===- lib/MC/MCObjectWriter.cpp - MCObjectWriter implementation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFragment.h"
#include "llvm/MC/MCSymbol.h"
namespace llvm {
class MCSection;
}

using namespace llvm;

MCObjectWriter::~MCObjectWriter() = default;

MCContext &MCObjectWriter::getContext() const { return Asm->getContext(); }

void MCObjectWriter::reset() {
  FileNames.clear();
  AddrsigSyms.clear();
  EmitAddrsigSection = false;
  SubsectionsViaSymbols = false;
  CGProfile.clear();
}

bool MCObjectWriter::isSymbolRefDifferenceFullyResolved(const MCSymbol &SA,
                                                        const MCSymbol &SB,
                                                        bool InSet) const {
  assert(!SA.isUndefined() && !SB.isUndefined());
  return isSymbolRefDifferenceFullyResolvedImpl(SA, *SB.getFragment(), InSet,
                                                /*IsPCRel=*/false);
}

bool MCObjectWriter::isSymbolRefDifferenceFullyResolvedImpl(
    const MCSymbol &SymA, const MCFragment &FB, bool InSet,
    bool IsPCRel) const {
  const MCSection &SecA = SymA.getSection();
  const MCSection &SecB = *FB.getParent();
  // On ELF and COFF  A - B is absolute if A and B are in the same section.
  return &SecA == &SecB;
}

void MCObjectWriter::addFileName(MCAssembler &Asm, StringRef FileName) {
  FileNames.emplace_back(std::string(FileName), Asm.Symbols.size());
}
