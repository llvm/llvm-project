//===-- PPCPredicates.cpp - PPC Branch Predicate Information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PowerPC branch predicates.
//
//===----------------------------------------------------------------------===//

#include "PPCPredicates.h"
#include <cassert>
using namespace llvm;

PPC::Predicate PPC::InvertPredicate(PPC::Predicate Opcode) {
  switch (Opcode) {
  default: assert(0 && "Unknown PPC branch opcode!");
  case PPC::PRED_EQ: return PPC::PRED_NE;
  case PPC::PRED_NE: return PPC::PRED_EQ;
  case PPC::PRED_LT: return PPC::PRED_GE;
  case PPC::PRED_GE: return PPC::PRED_LT;
  case PPC::PRED_GT: return PPC::PRED_LE;
  case PPC::PRED_LE: return PPC::PRED_GT;
  case PPC::PRED_NU: return PPC::PRED_UN;
  case PPC::PRED_UN: return PPC::PRED_NU;
  }
}
