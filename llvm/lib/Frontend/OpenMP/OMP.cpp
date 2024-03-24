//===- OMP.cpp ------ Collection of helpers for OpenMP --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/OpenMP/OMP.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
using namespace omp;

#define GEN_DIRECTIVES_IMPL
#include "llvm/Frontend/OpenMP/OMP.inc"

namespace llvm::omp {
bool isCompositeConstruct(Directive D) {
  // OpenMP Spec 5.2: [17.3, 8-9]
  // If directive-name-A and directive-name-B both correspond to loop-
  // associated constructs then directive-name is a composite construct
  size_t numLoopConstructs =
      llvm::count_if(getLeafConstructs(D), [](Directive L) {
        return getDirectiveAssociation(L) == Association::Loop;
      });
  return numLoopConstructs > 1;
}

bool isCombinedConstruct(Directive D) {
  // OpenMP Spec 5.2: [17.3, 9-10]
  // Otherwise directive-name is a combined construct.
  return !getLeafConstructs(D).empty() && !isCompositeConstruct(D);
}

} // namespace llvm::omp
