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

MCSection::MCSection(SectionVariant V, StringRef Name, bool IsText,
                     bool IsVirtual, MCSymbol *Begin)
    : Begin(Begin), BundleGroupBeforeFirstInst(false), HasInstructions(false),
      IsRegistered(false), IsText(IsText), IsVirtual(IsVirtual),
      LinkerRelaxable(false), Name(Name), Variant(V) {
  // The initial subsection number is 0. Create a fragment list.
  CurFragList = &Subsections.emplace_back(0u, FragList{}).second;
}

MCSymbol *MCSection::getEndSymbol(MCContext &Ctx) {
  if (!End)
    End = Ctx.createTempSymbol("sec_end");
  return End;
}

bool MCSection::hasEnded() const { return End && End->isInSection(); }

void MCSection::setBundleLockState(BundleLockStateType NewState) {
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

StringRef MCSection::getVirtualSectionKind() const { return "virtual"; }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void MCSection::dump(
    DenseMap<const MCFragment *, SmallVector<const MCSymbol *, 0>> *FragToSyms)
    const {
  raw_ostream &OS = errs();

  OS << "MCSection Name:" << getName();
  if (isLinkerRelaxable())
    OS << " LinkerRelaxable";
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

void MCEncodedFragment::setContents(ArrayRef<char> Contents) {
  auto &S = getParent()->ContentStorage;
  if (Contents.size() > ContentSize) {
    ContentStart = S.size();
    S.resize_for_overwrite(S.size() + Contents.size());
  }
  ContentSize = Contents.size();
  llvm::copy(Contents, S.begin() + ContentStart);
}

void MCEncodedFragment::addFixup(MCFixup Fixup) { appendFixups({Fixup}); }

void MCEncodedFragment::appendFixups(ArrayRef<MCFixup> Fixups) {
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

void MCEncodedFragment::setFixups(ArrayRef<MCFixup> Fixups) {
  auto &S = getParent()->FixupStorage;
  if (FixupStart + Fixups.size() > FixupEnd) {
    FixupStart = S.size();
    S.resize_for_overwrite(S.size() + Fixups.size());
  }
  FixupEnd = FixupStart + Fixups.size();
  llvm::copy(Fixups, S.begin() + FixupStart);
}
