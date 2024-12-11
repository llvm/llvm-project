//===--- BoundsSafetyOptRemarks.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* TO_UPSTREAM(BoundsSafety) ON*/
#ifndef LLVM_CODEGEN_BOUNDS_SAFETY_MISSED_OPT_REMARKS_H
#define LLVM_CODEGEN_BOUNDS_SAFETY_MISSED_OPT_REMARKS_H
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"

enum BoundsSafetyOptRemarkKind {
#define BOUNDS_SAFETY_MISSED_REMARK(SUFFIX, ANNOTATION_STR, ACTIONABLE_STR)        \
  BNS_MISSED_REMARK_##SUFFIX,
#include "BoundsSafetyOptRemarks.def"
#undef BOUNDS_SAFETY_MISSED_REMARK
};

namespace llvm {
/// Annotate instruction with -fbounds-safety missed remarks and additonal context
/// based on `BoundsSafetyOptRemarkKind` entry.
inline void annotate(Instruction *I, BoundsSafetyOptRemarkKind Kind) {
  switch (Kind) {
#define BOUNDS_SAFETY_MISSED_REMARK(SUFFIX, ANNOTATION_STR, ACTIONABLE_STR)        \
  case BNS_MISSED_REMARK_##SUFFIX:                                              \
    return I->addAnnotationMetadata({ANNOTATION_STR, ACTIONABLE_STR});
#include "BoundsSafetyOptRemarks.def"
#undef BOUNDS_SAFETY_MISSED_REMARK
  }
  llvm_unreachable("Unhandled BoundsSafetyOptRemarkKind");
}
/// Return the list of runtime checks associated with instruction I.
SmallVector<Instruction *> getBoundsSafetyRuntimeChecks(Instruction *I);

/// Annotate runtime checks associated with instruction I.
void annotateRuntimeChecks(Instruction *I, BoundsSafetyOptRemarkKind Remark);

/// Return true if instruction \p I has bounds-safety-check annotation.
bool isBoundsSafetyAnnotated(Instruction *I);
} // namespace llvm
#endif
/* TO_UPSTREAM(BoundsSafety) OFF*/