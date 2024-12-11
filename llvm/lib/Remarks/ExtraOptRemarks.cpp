//===--- ExtraOptRemarks.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* TO_UPSTREAM(BoundsSafety) ON*/
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Remarks/BoundsSafetyOptRemarks.h"

namespace llvm {

void annotateRuntimeChecks(Instruction *I, BoundsSafetyOptRemarkKind Remark) {
  auto BoundsSafetyChecks = getBoundsSafetyRuntimeChecks(I);
  // Early exit if we don't have any -fbounds-safety runtime checks associated with
  // the pointer.
  if (BoundsSafetyChecks.empty())
    return;
  for (auto *CI : BoundsSafetyChecks)
    annotate(CI, Remark);
}

bool isBoundsSafetyAnnotated(Instruction *I) {
  if (!I->hasMetadata(LLVMContext::MD_annotation))
    return false;
  for (const MDOperand &Op :
       I->getMetadata(LLVMContext::MD_annotation)->operands()) {
    StringRef S;
    if (isa<MDString>(Op.get()))
      S = cast<MDString>(Op.get())->getString();
    else {
      auto *AnnotationTuple = dyn_cast<MDTuple>(Op.get());
      S = cast<MDString>(AnnotationTuple->getOperand(0).get())->getString();
    }
    if (S.starts_with("bounds-safety")) {
      return true;
    }
  }
  return false;
}

// Recursively look through the users of I for cmp instructions that contain
// -fbounds-safety annotations.
void searchBoundsSafetyRuntimeChecks(Instruction *I,
                                  SmallVector<Instruction *> &Checks,
                                  int Depth) {
  if (Depth == 0)
    return;
  if (isa<CmpInst>(I) && isBoundsSafetyAnnotated(I))
    Checks.push_back(I);

  for (auto *User : I->users())
    if (auto *UI = dyn_cast<Instruction>(User))
      searchBoundsSafetyRuntimeChecks(UI, Checks, Depth - 1);
}

SmallVector<Instruction *> getBoundsSafetyRuntimeChecks(Instruction *I) {
  SmallVector<Instruction *> Checks;
  searchBoundsSafetyRuntimeChecks(I, Checks, MaxAnalysisRecursionDepth - 1);
  return Checks;
}
} // namespace llvm
/* TO_UPSTREAM(BoundsSafety) OFF*/