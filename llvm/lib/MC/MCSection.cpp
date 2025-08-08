//===- lib/MC/MCSection.cpp - Machine Code Section Representation ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSection.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

using namespace llvm;

MCSection::MCSection(StringRef Name, bool IsText, bool IsBss, MCSymbol *Begin)
    : Begin(Begin), HasInstructions(false), IsRegistered(false), IsText(IsText),
      IsBss(IsBss), Name(Name) {
  DummyFragment.setParent(this);
}

MCSymbol *MCSection::getEndSymbol(MCContext &Ctx) {
  if (!End)
    End = Ctx.createTempSymbol("sec_end");
  return End;
}

bool MCSection::hasEnded() const { return End && End->isInSection(); }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void MCSection::dump(
    DenseMap<const MCFragment *, SmallVector<const MCSymbol *, 0>> *FragToSyms)
    const {
  raw_ostream &OS = errs();

  OS << "MCSection Name:" << getName();
  for (auto &F : *this) {
    OS << '\n';
    F.dump();
    if (!FragToSyms)
      continue;
    auto It = FragToSyms->find(&F);
    if (It == FragToSyms->end())
      continue;
    for (auto *Sym : It->second) {
      OS << "\n  Symbol @" << Sym->getOffset() << ' ' << Sym->getName();
      if (Sym->isTemporary())
        OS << " Temporary";
    }
  }
}
#endif

void MCFragment::setVarContents(ArrayRef<char> Contents) {
  auto &S = getParent()->ContentStorage;
  if (VarContentStart + Contents.size() > VarContentEnd) {
    VarContentStart = S.size();
    S.resize_for_overwrite(S.size() + Contents.size());
  }
  VarContentEnd = VarContentStart + Contents.size();
  llvm::copy(Contents, S.begin() + VarContentStart);
}

void MCFragment::addFixup(MCFixup Fixup) { appendFixups({Fixup}); }

void MCFragment::appendFixups(ArrayRef<MCFixup> Fixups) {
  auto &S = getParent()->FixupStorage;
  if (LLVM_UNLIKELY(FixupEnd != S.size())) {
    // Move the elements to the end. Reserve space to avoid invalidating
    // S.begin()+I for `append`.
    auto Size = FixupEnd - FixupStart;
    auto I = std::exchange(FixupStart, S.size());
    S.reserve(S.size() + Size);
    S.append(S.begin() + I, S.begin() + I + Size);
  }
  S.append(Fixups.begin(), Fixups.end());
  FixupEnd = S.size();
}

void MCFragment::setVarFixups(ArrayRef<MCFixup> Fixups) {
  assert(Fixups.size() < 256 &&
         "variable-size tail cannot have more than 256 fixups");
  auto &S = getParent()->FixupStorage;
  if (Fixups.size() > VarFixupSize) {
    VarFixupStart = S.size();
    S.resize_for_overwrite(S.size() + Fixups.size());
  }
  VarFixupSize = Fixups.size();
  // Source fixup offsets are relative to the variable part's start. Add the
  // fixed part size to make them relative to the fixed part's start.
  std::transform(Fixups.begin(), Fixups.end(), S.begin() + VarFixupStart,
                 [Fixed = getFixedSize()](MCFixup F) {
                   F.setOffset(Fixed + F.getOffset());
                   return F;
                 });
}
