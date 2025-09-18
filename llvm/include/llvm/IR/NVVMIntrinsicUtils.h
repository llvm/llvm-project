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

// Enum to represent the cta_group::1 and
// cta_group::2 variants in TMA/TCGEN05 family of
// PTX instructions.
enum class CTAGroupKind : uint8_t {
  CG_NONE = 0, // default with no cta_group modifier
  CG_1 = 1,    // cta_group::1 modifier
  CG_2 = 2,    // cta_group::2 modifier
};

inline bool FPToIntegerIntrinsicShouldFTZ(Intrinsic::ID IntrinsicID) {
  switch (IntrinsicID) {
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

  case Intrinsic::nvvm_f2i_rm:
  case Intrinsic::nvvm_f2i_rn:
  case Intrinsic::nvvm_f2i_rp:
  case Intrinsic::nvvm_f2i_rz:

  case Intrinsic::nvvm_f2ui_rm:
  case Intrinsic::nvvm_f2ui_rn:
  case Intrinsic::nvvm_f2ui_rp:
  case Intrinsic::nvvm_f2ui_rz:

  case Intrinsic::nvvm_d2i_rm:
  case Intrinsic::nvvm_d2i_rn:
  case Intrinsic::nvvm_d2i_rp:
  case Intrinsic::nvvm_d2i_rz:

  case Intrinsic::nvvm_d2ui_rm:
  case Intrinsic::nvvm_d2ui_rn:
  case Intrinsic::nvvm_d2ui_rp:
  case Intrinsic::nvvm_d2ui_rz:

  case Intrinsic::nvvm_f2ll_rm:
  case Intrinsic::nvvm_f2ll_rn:
  case Intrinsic::nvvm_f2ll_rp:
  case Intrinsic::nvvm_f2ll_rz:

  case Intrinsic::nvvm_f2ull_rm:
  case Intrinsic::nvvm_f2ull_rn:
  case Intrinsic::nvvm_f2ull_rp:
  case Intrinsic::nvvm_f2ull_rz:

  case Intrinsic::nvvm_d2ll_rm:
  case Intrinsic::nvvm_d2ll_rn:
  case Intrinsic::nvvm_d2ll_rp:
  case Intrinsic::nvvm_d2ll_rz:

  case Intrinsic::nvvm_d2ull_rm:
  case Intrinsic::nvvm_d2ull_rn:
  case Intrinsic::nvvm_d2ull_rp:
  case Intrinsic::nvvm_d2ull_rz:
    return false;
  }
  llvm_unreachable("Checking FTZ flag for invalid f2i/d2i intrinsic");
}

inline bool FPToIntegerIntrinsicResultIsSigned(Intrinsic::ID IntrinsicID) {
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

  // f2ui
  case Intrinsic::nvvm_f2ui_rm:
  case Intrinsic::nvvm_f2ui_rm_ftz:
  case Intrinsic::nvvm_f2ui_rn:
  case Intrinsic::nvvm_f2ui_rn_ftz:
  case Intrinsic::nvvm_f2ui_rp:
  case Intrinsic::nvvm_f2ui_rp_ftz:
  case Intrinsic::nvvm_f2ui_rz:
  case Intrinsic::nvvm_f2ui_rz_ftz:
  // d2ui
  case Intrinsic::nvvm_d2ui_rm:
  case Intrinsic::nvvm_d2ui_rn:
  case Intrinsic::nvvm_d2ui_rp:
  case Intrinsic::nvvm_d2ui_rz:
  // f2ull
  case Intrinsic::nvvm_f2ull_rm:
  case Intrinsic::nvvm_f2ull_rm_ftz:
  case Intrinsic::nvvm_f2ull_rn:
  case Intrinsic::nvvm_f2ull_rn_ftz:
  case Intrinsic::nvvm_f2ull_rp:
  case Intrinsic::nvvm_f2ull_rp_ftz:
  case Intrinsic::nvvm_f2ull_rz:
  case Intrinsic::nvvm_f2ull_rz_ftz:
  // d2ull
  case Intrinsic::nvvm_d2ull_rm:
  case Intrinsic::nvvm_d2ull_rn:
  case Intrinsic::nvvm_d2ull_rp:
  case Intrinsic::nvvm_d2ull_rz:
    return false;
  }
  llvm_unreachable(
      "Checking invalid f2i/d2i intrinsic for signed int conversion");
}

inline APFloat::roundingMode
GetFPToIntegerRoundingMode(Intrinsic::ID IntrinsicID) {
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
  llvm_unreachable("Checking rounding mode for invalid f2i/d2i intrinsic");
}

inline bool FMinFMaxShouldFTZ(Intrinsic::ID IntrinsicID) {
  switch (IntrinsicID) {
  case Intrinsic::nvvm_fmax_ftz_f:
  case Intrinsic::nvvm_fmax_ftz_nan_f:
  case Intrinsic::nvvm_fmax_ftz_nan_xorsign_abs_f:
  case Intrinsic::nvvm_fmax_ftz_xorsign_abs_f:

  case Intrinsic::nvvm_fmin_ftz_f:
  case Intrinsic::nvvm_fmin_ftz_nan_f:
  case Intrinsic::nvvm_fmin_ftz_nan_xorsign_abs_f:
  case Intrinsic::nvvm_fmin_ftz_xorsign_abs_f:
    return true;

  case Intrinsic::nvvm_fmax_d:
  case Intrinsic::nvvm_fmax_f:
  case Intrinsic::nvvm_fmax_nan_f:
  case Intrinsic::nvvm_fmax_nan_xorsign_abs_f:
  case Intrinsic::nvvm_fmax_xorsign_abs_f:

  case Intrinsic::nvvm_fmin_d:
  case Intrinsic::nvvm_fmin_f:
  case Intrinsic::nvvm_fmin_nan_f:
  case Intrinsic::nvvm_fmin_nan_xorsign_abs_f:
  case Intrinsic::nvvm_fmin_xorsign_abs_f:
    return false;
  }
  llvm_unreachable("Checking FTZ flag for invalid fmin/fmax intrinsic");
}

inline bool FMinFMaxPropagatesNaNs(Intrinsic::ID IntrinsicID) {
  switch (IntrinsicID) {
  case Intrinsic::nvvm_fmax_ftz_nan_f:
  case Intrinsic::nvvm_fmax_nan_f:
  case Intrinsic::nvvm_fmax_ftz_nan_xorsign_abs_f:
  case Intrinsic::nvvm_fmax_nan_xorsign_abs_f:

  case Intrinsic::nvvm_fmin_ftz_nan_f:
  case Intrinsic::nvvm_fmin_nan_f:
  case Intrinsic::nvvm_fmin_ftz_nan_xorsign_abs_f:
  case Intrinsic::nvvm_fmin_nan_xorsign_abs_f:
    return true;

  case Intrinsic::nvvm_fmax_d:
  case Intrinsic::nvvm_fmax_f:
  case Intrinsic::nvvm_fmax_ftz_f:
  case Intrinsic::nvvm_fmax_ftz_xorsign_abs_f:
  case Intrinsic::nvvm_fmax_xorsign_abs_f:

  case Intrinsic::nvvm_fmin_d:
  case Intrinsic::nvvm_fmin_f:
  case Intrinsic::nvvm_fmin_ftz_f:
  case Intrinsic::nvvm_fmin_ftz_xorsign_abs_f:
  case Intrinsic::nvvm_fmin_xorsign_abs_f:
    return false;
  }
  llvm_unreachable("Checking NaN flag for invalid fmin/fmax intrinsic");
}

inline bool FMinFMaxIsXorSignAbs(Intrinsic::ID IntrinsicID) {
  switch (IntrinsicID) {
  case Intrinsic::nvvm_fmax_ftz_nan_xorsign_abs_f:
  case Intrinsic::nvvm_fmax_ftz_xorsign_abs_f:
  case Intrinsic::nvvm_fmax_nan_xorsign_abs_f:
  case Intrinsic::nvvm_fmax_xorsign_abs_f:

  case Intrinsic::nvvm_fmin_ftz_nan_xorsign_abs_f:
  case Intrinsic::nvvm_fmin_ftz_xorsign_abs_f:
  case Intrinsic::nvvm_fmin_nan_xorsign_abs_f:
  case Intrinsic::nvvm_fmin_xorsign_abs_f:
    return true;

  case Intrinsic::nvvm_fmax_d:
  case Intrinsic::nvvm_fmax_f:
  case Intrinsic::nvvm_fmax_ftz_f:
  case Intrinsic::nvvm_fmax_ftz_nan_f:
  case Intrinsic::nvvm_fmax_nan_f:

  case Intrinsic::nvvm_fmin_d:
  case Intrinsic::nvvm_fmin_f:
  case Intrinsic::nvvm_fmin_ftz_f:
  case Intrinsic::nvvm_fmin_ftz_nan_f:
  case Intrinsic::nvvm_fmin_nan_f:
    return false;
  }
  llvm_unreachable("Checking XorSignAbs flag for invalid fmin/fmax intrinsic");
}

inline bool UnaryMathIntrinsicShouldFTZ(Intrinsic::ID IntrinsicID) {
  switch (IntrinsicID) {
  case Intrinsic::nvvm_ceil_ftz_f:
  case Intrinsic::nvvm_fabs_ftz:
  case Intrinsic::nvvm_floor_ftz_f:
  case Intrinsic::nvvm_round_ftz_f:
  case Intrinsic::nvvm_saturate_ftz_f:
  case Intrinsic::nvvm_sqrt_rn_ftz_f:
    return true;
  case Intrinsic::nvvm_ceil_f:
  case Intrinsic::nvvm_ceil_d:
  case Intrinsic::nvvm_fabs:
  case Intrinsic::nvvm_floor_f:
  case Intrinsic::nvvm_floor_d:
  case Intrinsic::nvvm_round_f:
  case Intrinsic::nvvm_round_d:
  case Intrinsic::nvvm_saturate_d:
  case Intrinsic::nvvm_saturate_f:
  case Intrinsic::nvvm_sqrt_f:
  case Intrinsic::nvvm_sqrt_rn_d:
  case Intrinsic::nvvm_sqrt_rn_f:
    return false;
  }
  llvm_unreachable("Checking FTZ flag for invalid unary intrinsic");
}

inline bool RCPShouldFTZ(Intrinsic::ID IntrinsicID) {
  switch (IntrinsicID) {
  case Intrinsic::nvvm_rcp_rm_ftz_f:
  case Intrinsic::nvvm_rcp_rn_ftz_f:
  case Intrinsic::nvvm_rcp_rp_ftz_f:
  case Intrinsic::nvvm_rcp_rz_ftz_f:
    return true;
  case Intrinsic::nvvm_rcp_rm_d:
  case Intrinsic::nvvm_rcp_rm_f:
  case Intrinsic::nvvm_rcp_rn_d:
  case Intrinsic::nvvm_rcp_rn_f:
  case Intrinsic::nvvm_rcp_rp_d:
  case Intrinsic::nvvm_rcp_rp_f:
  case Intrinsic::nvvm_rcp_rz_d:
  case Intrinsic::nvvm_rcp_rz_f:
    return false;
  }
  llvm_unreachable("Checking FTZ flag for invalid rcp intrinsic");
}

inline APFloat::roundingMode GetRCPRoundingMode(Intrinsic::ID IntrinsicID) {
  switch (IntrinsicID) {
  case Intrinsic::nvvm_rcp_rm_f:
  case Intrinsic::nvvm_rcp_rm_d:
  case Intrinsic::nvvm_rcp_rm_ftz_f:
    return APFloat::rmTowardNegative;

  case Intrinsic::nvvm_rcp_rn_f:
  case Intrinsic::nvvm_rcp_rn_d:
  case Intrinsic::nvvm_rcp_rn_ftz_f:
    return APFloat::rmNearestTiesToEven;

  case Intrinsic::nvvm_rcp_rp_f:
  case Intrinsic::nvvm_rcp_rp_d:
  case Intrinsic::nvvm_rcp_rp_ftz_f:
    return APFloat::rmTowardPositive;

  case Intrinsic::nvvm_rcp_rz_f:
  case Intrinsic::nvvm_rcp_rz_d:
  case Intrinsic::nvvm_rcp_rz_ftz_f:
    return APFloat::rmTowardZero;
  }
  llvm_unreachable("Checking rounding mode for invalid rcp intrinsic");
}

inline DenormalMode GetNVVMDenormMode(bool ShouldFTZ) {
  if (ShouldFTZ)
    return DenormalMode::getPreserveSign();
  return DenormalMode::getIEEE();
}

inline bool FAddShouldFTZ(Intrinsic::ID IntrinsicID) {
  switch (IntrinsicID) {
  case Intrinsic::nvvm_add_rm_ftz_f:
  case Intrinsic::nvvm_add_rn_ftz_f:
  case Intrinsic::nvvm_add_rp_ftz_f:
  case Intrinsic::nvvm_add_rz_ftz_f:
    return true;

  case Intrinsic::nvvm_add_rm_f:
  case Intrinsic::nvvm_add_rn_f:
  case Intrinsic::nvvm_add_rp_f:
  case Intrinsic::nvvm_add_rz_f:
  case Intrinsic::nvvm_add_rm_d:
  case Intrinsic::nvvm_add_rn_d:
  case Intrinsic::nvvm_add_rp_d:
  case Intrinsic::nvvm_add_rz_d:
    return false;
  }
  llvm_unreachable("Checking FTZ flag for invalid NVVM add intrinsic");
}

inline APFloat::roundingMode GetFAddRoundingMode(Intrinsic::ID IntrinsicID) {
  switch (IntrinsicID) {
  case Intrinsic::nvvm_add_rm_f:
  case Intrinsic::nvvm_add_rm_d:
  case Intrinsic::nvvm_add_rm_ftz_f:
    return APFloat::rmTowardNegative;
  case Intrinsic::nvvm_add_rn_f:
  case Intrinsic::nvvm_add_rn_d:
  case Intrinsic::nvvm_add_rn_ftz_f:
    return APFloat::rmNearestTiesToEven;
  case Intrinsic::nvvm_add_rp_f:
  case Intrinsic::nvvm_add_rp_d:
  case Intrinsic::nvvm_add_rp_ftz_f:
    return APFloat::rmTowardPositive;
  case Intrinsic::nvvm_add_rz_f:
  case Intrinsic::nvvm_add_rz_d:
  case Intrinsic::nvvm_add_rz_ftz_f:
    return APFloat::rmTowardZero;
  }
  llvm_unreachable("Invalid FP instrinsic rounding mode for NVVM add");
}

inline bool FMulShouldFTZ(Intrinsic::ID IntrinsicID) {
  switch (IntrinsicID) {
  case Intrinsic::nvvm_mul_rm_ftz_f:
  case Intrinsic::nvvm_mul_rn_ftz_f:
  case Intrinsic::nvvm_mul_rp_ftz_f:
  case Intrinsic::nvvm_mul_rz_ftz_f:
    return true;

  case Intrinsic::nvvm_mul_rm_f:
  case Intrinsic::nvvm_mul_rn_f:
  case Intrinsic::nvvm_mul_rp_f:
  case Intrinsic::nvvm_mul_rz_f:
  case Intrinsic::nvvm_mul_rm_d:
  case Intrinsic::nvvm_mul_rn_d:
  case Intrinsic::nvvm_mul_rp_d:
  case Intrinsic::nvvm_mul_rz_d:
    return false;
  }
  llvm_unreachable("Checking FTZ flag for invalid NVVM mul intrinsic");
}

inline APFloat::roundingMode GetFMulRoundingMode(Intrinsic::ID IntrinsicID) {
  switch (IntrinsicID) {
  case Intrinsic::nvvm_mul_rm_f:
  case Intrinsic::nvvm_mul_rm_d:
  case Intrinsic::nvvm_mul_rm_ftz_f:
    return APFloat::rmTowardNegative;
  case Intrinsic::nvvm_mul_rn_f:
  case Intrinsic::nvvm_mul_rn_d:
  case Intrinsic::nvvm_mul_rn_ftz_f:
    return APFloat::rmNearestTiesToEven;
  case Intrinsic::nvvm_mul_rp_f:
  case Intrinsic::nvvm_mul_rp_d:
  case Intrinsic::nvvm_mul_rp_ftz_f:
    return APFloat::rmTowardPositive;
  case Intrinsic::nvvm_mul_rz_f:
  case Intrinsic::nvvm_mul_rz_d:
  case Intrinsic::nvvm_mul_rz_ftz_f:
    return APFloat::rmTowardZero;
  }
  llvm_unreachable("Invalid FP instrinsic rounding mode for NVVM mul");
}

inline bool FDivShouldFTZ(Intrinsic::ID IntrinsicID) {
  switch (IntrinsicID) {
  case Intrinsic::nvvm_div_rm_ftz_f:
  case Intrinsic::nvvm_div_rn_ftz_f:
  case Intrinsic::nvvm_div_rp_ftz_f:
  case Intrinsic::nvvm_div_rz_ftz_f:
    return true;

  case Intrinsic::nvvm_div_rm_f:
  case Intrinsic::nvvm_div_rn_f:
  case Intrinsic::nvvm_div_rp_f:
  case Intrinsic::nvvm_div_rz_f:
  case Intrinsic::nvvm_div_rm_d:
  case Intrinsic::nvvm_div_rn_d:
  case Intrinsic::nvvm_div_rp_d:
  case Intrinsic::nvvm_div_rz_d:
    return false;
  }
  llvm_unreachable("Checking FTZ flag for invalid NVVM div intrinsic");
}

inline APFloat::roundingMode GetFDivRoundingMode(Intrinsic::ID IntrinsicID) {
  switch (IntrinsicID) {
  case Intrinsic::nvvm_div_rm_f:
  case Intrinsic::nvvm_div_rm_d:
  case Intrinsic::nvvm_div_rm_ftz_f:
    return APFloat::rmTowardNegative;
  case Intrinsic::nvvm_div_rn_f:
  case Intrinsic::nvvm_div_rn_d:
  case Intrinsic::nvvm_div_rn_ftz_f:
    return APFloat::rmNearestTiesToEven;
  case Intrinsic::nvvm_div_rp_f:
  case Intrinsic::nvvm_div_rp_d:
  case Intrinsic::nvvm_div_rp_ftz_f:
    return APFloat::rmTowardPositive;
  case Intrinsic::nvvm_div_rz_f:
  case Intrinsic::nvvm_div_rz_d:
  case Intrinsic::nvvm_div_rz_ftz_f:
    return APFloat::rmTowardZero;
  }
  llvm_unreachable("Invalid FP instrinsic rounding mode for NVVM div");
}

inline bool FMAShouldFTZ(Intrinsic::ID IntrinsicID) {
  switch (IntrinsicID) {
  case Intrinsic::nvvm_fma_rm_ftz_f:
  case Intrinsic::nvvm_fma_rn_ftz_f:
  case Intrinsic::nvvm_fma_rp_ftz_f:
  case Intrinsic::nvvm_fma_rz_ftz_f:
    return true;

  case Intrinsic::nvvm_fma_rm_f:
  case Intrinsic::nvvm_fma_rn_f:
  case Intrinsic::nvvm_fma_rp_f:
  case Intrinsic::nvvm_fma_rz_f:
  case Intrinsic::nvvm_fma_rm_d:
  case Intrinsic::nvvm_fma_rn_d:
  case Intrinsic::nvvm_fma_rp_d:
  case Intrinsic::nvvm_fma_rz_d:
    return false;
  }
  llvm_unreachable("Checking FTZ flag for invalid NVVM fma intrinsic");
}

inline APFloat::roundingMode GetFMARoundingMode(Intrinsic::ID IntrinsicID) {
  switch (IntrinsicID) {
  case Intrinsic::nvvm_fma_rm_f:
  case Intrinsic::nvvm_fma_rm_d:
  case Intrinsic::nvvm_fma_rm_ftz_f:
    return APFloat::rmTowardNegative;
  case Intrinsic::nvvm_fma_rn_f:
  case Intrinsic::nvvm_fma_rn_d:
  case Intrinsic::nvvm_fma_rn_ftz_f:
    return APFloat::rmNearestTiesToEven;
  case Intrinsic::nvvm_fma_rp_f:
  case Intrinsic::nvvm_fma_rp_d:
  case Intrinsic::nvvm_fma_rp_ftz_f:
    return APFloat::rmTowardPositive;
  case Intrinsic::nvvm_fma_rz_f:
  case Intrinsic::nvvm_fma_rz_d:
  case Intrinsic::nvvm_fma_rz_ftz_f:
    return APFloat::rmTowardZero;
  }
  llvm_unreachable("Invalid FP instrinsic rounding mode for NVVM fma");
}

} // namespace nvvm
} // namespace llvm
#endif // LLVM_IR_NVVMINTRINSICUTILS_H
