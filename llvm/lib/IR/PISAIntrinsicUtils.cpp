//===-- PISAIntrinsicUtils.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/PISAIntrinsicUtils.h"
#include "llvm/Support/AtomicOrdering.h"

using namespace llvm;
using namespace llvm::pisa;

void pisa::printMemoryOrdering(raw_ostream &OS, const Constant *ImmArgVal) {
  auto *CI = cast<ConstantInt>(ImmArgVal);
  auto AO = static_cast<AtomicOrdering>(CI->getZExtValue());
  if (static_cast<unsigned>(AO) > static_cast<unsigned>(AtomicOrdering::LAST))
    return; // invalid value, print nothing
  OS << toIRString(AO);
}

void pisa::printRoundingMode(raw_ostream &OS, const Constant *ImmArgVal) {
  auto *CI = cast<ConstantInt>(ImmArgVal);
  int64_t Val = CI->getSExtValue();
  switch (static_cast<RoundingMode>(Val)) {
  default:
    // invalid/unsupported value, print nothing
    break;
  case RoundingMode::TowardZero:
    OS << ".rz";
    break;
  case RoundingMode::NearestTiesToEven:
    OS << ".re";
    break;
  case RoundingMode::TowardPositive:
    OS << ".ru";
    break;
  case RoundingMode::TowardNegative:
    OS << ".rd";
    break;
  case RoundingMode::NearestTiesToAway:
    OS << ".rna";
    break;
  case RoundingMode::Invalid:
    OS << "none";
    break;
  }
}

void pisa::printIRedOp(raw_ostream &OS, const Constant *ImmArgVal) {
  auto *CI = cast<ConstantInt>(ImmArgVal);
  int64_t Val = CI->getSExtValue();
  switch (Val) {
  case IRedOp::SUM:
    OS << ".sum";
    break;
  case IRedOp::SMIN:
    OS << ".smin";
    break;
  case IRedOp::SMAX:
    OS << ".smax";
    break;
  case IRedOp::UMIN:
    OS << ".umin";
    break;
  case IRedOp::UMAX:
    OS << ".umax";
    break;
  case IRedOp::AND:
    OS << ".and";
    break;
  case IRedOp::OR:
    OS << ".or";
    break;
  case IRedOp::XOR:
    OS << ".xor";
    break;
  case IRedOp::ABSMAX:
    OS << ".absmax";
    break;
  }
  // invalid value, print nothing
}

void pisa::printFRedOp(raw_ostream &OS, const Constant *ImmArgVal) {
  auto *CI = cast<ConstantInt>(ImmArgVal);
  int64_t Val = CI->getSExtValue();
  switch (Val) {
  case FRedOp::MIN:
    OS << ".min";
    break;
  case FRedOp::MAX:
    OS << ".max";
    break;
  case FRedOp::ABSMAX:
    OS << ".absmax";
    break;
  }
  // invalid value, print nothing
}

void pisa::printSHFLMode(raw_ostream &OS, const Constant *ImmArgVal) {
  auto *CI = cast<ConstantInt>(ImmArgVal);
  int64_t Val = CI->getSExtValue();
  switch (Val) {
  case SHFLMode::UP:
    OS << ".up";
    break;
  case SHFLMode::DOWN:
    OS << ".down";
    break;
  case SHFLMode::XOR:
    OS << ".xor";
    break;
  case SHFLMode::IDX:
    OS << ".idx";
    break;
  }
  // invalid value, print nothing
}
