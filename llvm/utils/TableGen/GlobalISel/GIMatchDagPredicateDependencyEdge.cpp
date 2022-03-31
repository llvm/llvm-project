//===- GIMatchDagPredicateDependencyEdge.cpp - Have inputs before check ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GIMatchDagPredicateDependencyEdge.h"

#include "GIMatchDagInstr.h"
#include "GIMatchDagOperands.h"
#include "GIMatchDagPredicate.h"

#include "llvm/Support/raw_ostream.h"

using namespace llvm;

LLVM_DUMP_METHOD void
GIMatchDagPredicateDependencyEdge::print(raw_ostream &OS) const {
  OS << getRequiredMI()->getName();
  if (getRequiredMO())
    OS << "[" << getRequiredMO()->getName() << "]";
  OS << " ==> " << getPredicate()->getName() << "["
     << getPredicateOp()->getName() << "]";
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void GIMatchDagPredicateDependencyEdge::dump() const {
  print(errs());
}
#endif // if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)

raw_ostream &llvm::operator<<(raw_ostream &OS,
                              const GIMatchDagPredicateDependencyEdge &E) {
  E.print(OS);
  return OS;
}
