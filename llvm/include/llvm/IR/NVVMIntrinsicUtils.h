//===--- NVVMIntrinsicUtils.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file contains the definitions of the enumerations and flags
/// associated with NVVM Intrinsics, along with some helper functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_NVVMINTRINSICUTILS_H
#define LLVM_IR_NVVMINTRINSICUTILS_H

#include <stdint.h>

#include "llvm/ADT/APFloat.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"

namespace llvm {
namespace nvvm {

// Reduction Ops supported with TMA Copy from Shared
// to Global Memory for the "cp.reduce.async.bulk.tensor.*"
// family of PTX instructions.
enum class TMAReductionOp : uint8_t {
  ADD = 0,
  MIN = 1,
  MAX = 2,
  INC = 3,
  DEC = 4,
  AND = 5,
  OR = 6,
  XOR = 7,
};

inline bool IntrinsicShouldFTZ(Intrinsic::ID IntrinsicID) {
  switch (IntrinsicID) {
  // Float to i32 / i64 conversion intrinsics:
  case Intrinsic::nvvm_f2i_rm_ftz:
  case Intrinsic::nvvm_f2i_rn_ftz:
  case Intrinsic::nvvm_f2i_rp_ftz:
  case Intrinsic::nvvm_f2i_rz_ftz:

  case Intrinsic::nvvm_f2ui_rm_ftz:
  case Intrinsic::nvvm_f2ui_rn_ftz:
  case Intrinsic::nvvm_f2ui_rp_ftz:
  case Intrinsic::nvvm_f2ui_rz_ftz:

  case Intrinsic::nvvm_f2ll_rm_ftz:
  case Intrinsic::nvvm_f2ll_rn_ftz:
  case Intrinsic::nvvm_f2ll_rp_ftz:
  case Intrinsic::nvvm_f2ll_rz_ftz:

  case Intrinsic::nvvm_f2ull_rm_ftz:
  case Intrinsic::nvvm_f2ull_rn_ftz:
  case Intrinsic::nvvm_f2ull_rp_ftz:
  case Intrinsic::nvvm_f2ull_rz_ftz:
    return true;
  }
  return false;
}

inline bool IntrinsicConvertsToSignedInteger(Intrinsic::ID IntrinsicID) {
  switch (IntrinsicID) {
  // f2i
  case Intrinsic::nvvm_f2i_rm:
  case Intrinsic::nvvm_f2i_rm_ftz:
  case Intrinsic::nvvm_f2i_rn:
  case Intrinsic::nvvm_f2i_rn_ftz:
  case Intrinsic::nvvm_f2i_rp:
  case Intrinsic::nvvm_f2i_rp_ftz:
  case Intrinsic::nvvm_f2i_rz:
  case Intrinsic::nvvm_f2i_rz_ftz:
  // d2i
  case Intrinsic::nvvm_d2i_rm:
  case Intrinsic::nvvm_d2i_rn:
  case Intrinsic::nvvm_d2i_rp:
  case Intrinsic::nvvm_d2i_rz:
  // f2ll
  case Intrinsic::nvvm_f2ll_rm:
  case Intrinsic::nvvm_f2ll_rm_ftz:
  case Intrinsic::nvvm_f2ll_rn:
  case Intrinsic::nvvm_f2ll_rn_ftz:
  case Intrinsic::nvvm_f2ll_rp:
  case Intrinsic::nvvm_f2ll_rp_ftz:
  case Intrinsic::nvvm_f2ll_rz:
  case Intrinsic::nvvm_f2ll_rz_ftz:
  // d2ll
  case Intrinsic::nvvm_d2ll_rm:
  case Intrinsic::nvvm_d2ll_rn:
  case Intrinsic::nvvm_d2ll_rp:
  case Intrinsic::nvvm_d2ll_rz:
    return true;
  }
  return false;
}

inline APFloat::roundingMode
IntrinsicGetRoundingMode(Intrinsic::ID IntrinsicID) {
  switch (IntrinsicID) {
  // RM:
  case Intrinsic::nvvm_f2i_rm:
  case Intrinsic::nvvm_f2ui_rm:
  case Intrinsic::nvvm_f2i_rm_ftz:
  case Intrinsic::nvvm_f2ui_rm_ftz:
  case Intrinsic::nvvm_d2i_rm:
  case Intrinsic::nvvm_d2ui_rm:

  case Intrinsic::nvvm_f2ll_rm:
  case Intrinsic::nvvm_f2ull_rm:
  case Intrinsic::nvvm_f2ll_rm_ftz:
  case Intrinsic::nvvm_f2ull_rm_ftz:
  case Intrinsic::nvvm_d2ll_rm:
  case Intrinsic::nvvm_d2ull_rm:
    return APFloat::rmTowardNegative;

  // RN:
  case Intrinsic::nvvm_f2i_rn:
  case Intrinsic::nvvm_f2ui_rn:
  case Intrinsic::nvvm_f2i_rn_ftz:
  case Intrinsic::nvvm_f2ui_rn_ftz:
  case Intrinsic::nvvm_d2i_rn:
  case Intrinsic::nvvm_d2ui_rn:

  case Intrinsic::nvvm_f2ll_rn:
  case Intrinsic::nvvm_f2ull_rn:
  case Intrinsic::nvvm_f2ll_rn_ftz:
  case Intrinsic::nvvm_f2ull_rn_ftz:
  case Intrinsic::nvvm_d2ll_rn:
  case Intrinsic::nvvm_d2ull_rn:
    return APFloat::rmNearestTiesToEven;

  // RP:
  case Intrinsic::nvvm_f2i_rp:
  case Intrinsic::nvvm_f2ui_rp:
  case Intrinsic::nvvm_f2i_rp_ftz:
  case Intrinsic::nvvm_f2ui_rp_ftz:
  case Intrinsic::nvvm_d2i_rp:
  case Intrinsic::nvvm_d2ui_rp:

  case Intrinsic::nvvm_f2ll_rp:
  case Intrinsic::nvvm_f2ull_rp:
  case Intrinsic::nvvm_f2ll_rp_ftz:
  case Intrinsic::nvvm_f2ull_rp_ftz:
  case Intrinsic::nvvm_d2ll_rp:
  case Intrinsic::nvvm_d2ull_rp:
    return APFloat::rmTowardPositive;

  // RZ:
  case Intrinsic::nvvm_f2i_rz:
  case Intrinsic::nvvm_f2ui_rz:
  case Intrinsic::nvvm_f2i_rz_ftz:
  case Intrinsic::nvvm_f2ui_rz_ftz:
  case Intrinsic::nvvm_d2i_rz:
  case Intrinsic::nvvm_d2ui_rz:

  case Intrinsic::nvvm_f2ll_rz:
  case Intrinsic::nvvm_f2ull_rz:
  case Intrinsic::nvvm_f2ll_rz_ftz:
  case Intrinsic::nvvm_f2ull_rz_ftz:
  case Intrinsic::nvvm_d2ll_rz:
  case Intrinsic::nvvm_d2ull_rz:
    return APFloat::rmTowardZero;
  }
  llvm_unreachable("Invalid f2i/d2i rounding mode intrinsic");
  return APFloat::roundingMode::Invalid;
}

} // namespace nvvm
} // namespace llvm
#endif // LLVM_IR_NVVMINTRINSICUTILS_H
