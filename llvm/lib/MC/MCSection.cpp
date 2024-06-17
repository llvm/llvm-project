//===- lib/MC/MCSection.cpp - Machine Code Section Representation ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCSection.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCFragment.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

using namespace llvm;

MCSection::MCSection(SectionVariant V, StringRef Name, SectionKind K,
                     MCSymbol *Begin)
    : Begin(Begin), BundleGroupBeforeFirstInst(false), HasInstructions(false),
      HasLayout(false), IsRegistered(false), DummyFragment(this), Name(Name),
      Variant(V), Kind(K) {
  // The initial subsection number is 0. Create a fragment list.
  CurFragList = &Subsections.emplace_back(0u, FragList{}).second;
}

MCSymbol *MCSection::getEndSymbol(MCContext &Ctx) {
  if (!End)
    End = Ctx.createTempSymbol("sec_end");
  return End;
}

bool MCSection::hasEnded() const { return End && End->isInSection(); }

MCSection::~MCSection() {
  for (auto &[_, Chain] : Subsections) {
    for (MCFragment *X = Chain.Head, *Y; X; X = Y) {
      Y = X->Next;
      X->destroy();
    }
  }
}

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

void MCSection::switchSubsection(unsigned Subsection) {
  size_t I = 0, E = Subsections.size();
  while (I != E && Subsections[I].first < Subsection)
    ++I;
  // If the subsection number is not in the sorted Subsections list, create a
  // new fragment list.
  if (I == E || Subsections[I].first != Subsection)
    Subsections.insert(Subsections.begin() + I, {Subsection, FragList{}});
  CurFragList = &Subsections[I].second;
}

StringRef MCSection::getVirtualSectionKind() const { return "virtual"; }

void MCSection::addPendingLabel(MCSymbol *label, unsigned Subsection) {
  PendingLabels.push_back(PendingLabel(label, Subsection));
}

void MCSection::flushPendingLabels(MCFragment *F, unsigned Subsection) {
  // Set the fragment and fragment offset for all pending symbols in the
  // specified Subsection, and remove those symbols from the pending list.
  for (auto It = PendingLabels.begin(); It != PendingLabels.end(); ++It) {
    PendingLabel& Label = *It;
    if (Label.Subsection == Subsection) {
      Label.Sym->setFragment(F);
      assert(Label.Sym->getOffset() == 0);
      PendingLabels.erase(It--);
    }
  }
}

void MCSection::flushPendingLabels() {
  // Make sure all remaining pending labels point to data fragments, by
  // creating new empty data fragments for each Subsection with labels pending.
  while (!PendingLabels.empty()) {
    PendingLabel& Label = PendingLabels[0];
    switchSubsection(Label.Subsection);
    MCFragment *F = new MCDataFragment();
    addFragment(*F);
    F->setParent(this);
    flushPendingLabels(F, Label.Subsection);
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void MCSection::dump() const {
  raw_ostream &OS = errs();

  OS << "<MCSection Name:" << getName();
  OS << " Fragments:[\n      ";
  bool First = true;
  for (auto &F : *this) {
    if (First)
      First = false;
    else
      OS << ",\n      ";
    F.dump();
  }
  OS << "]>";
}
#endif
