//===-- VerifierAArch64.cpp - AArch64-specific IR verification -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains AArch64-specific IR verification logic that was extracted
// from Verifier.cpp for code organization purposes only. These checks are
// always compiled and linked as part of LLVMCore — this is not a target-
// dependent IR verifier.
//
//===----------------------------------------------------------------------===//

#include "VerifierInternal.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IntrinsicsAArch64.h"

using namespace llvm;

#define Check(C, ...)                                                          \
  do {                                                                         \
    if (!(C)) {                                                                \
      VS.CheckFailed(__VA_ARGS__);                                             \
      return;                                                                  \
    }                                                                          \
  } while (false)

static void checkSVEGatherScatterOffsets(VerifierSupport &VS, CallBase &Call,
                                         unsigned OffsetIdx) {
  auto *OffsetTy =
      dyn_cast<VectorType>(Call.getArgOperand(OffsetIdx)->getType());
  Check(OffsetTy && isa<ScalableVectorType>(OffsetTy) &&
            OffsetTy->getPrimitiveSizeInBits().getKnownMinValue() <= 128,
        "SVE gather/scatter offsets must be a scalable vector no wider than "
        "a single Z register",
        &Call);
}

void llvm::verifyAArch64IntrinsicCall(VerifierSupport &VS, Intrinsic::ID ID,
                                      CallBase &Call) {
  if (!VS.TT.isAArch64())
    return;

  switch (ID) {
  default:
    break;

  // Gather loads: (predicate, base, offsets)
  case Intrinsic::aarch64_sve_ld1_gather:
  case Intrinsic::aarch64_sve_ld1_gather_index:
  case Intrinsic::aarch64_sve_ld1_gather_sxtw:
  case Intrinsic::aarch64_sve_ld1_gather_uxtw:
  case Intrinsic::aarch64_sve_ld1_gather_sxtw_index:
  case Intrinsic::aarch64_sve_ld1_gather_uxtw_index:
  case Intrinsic::aarch64_sve_ldff1_gather:
  case Intrinsic::aarch64_sve_ldff1_gather_index:
  case Intrinsic::aarch64_sve_ldff1_gather_sxtw:
  case Intrinsic::aarch64_sve_ldff1_gather_uxtw:
  case Intrinsic::aarch64_sve_ldff1_gather_sxtw_index:
  case Intrinsic::aarch64_sve_ldff1_gather_uxtw_index:
  case Intrinsic::aarch64_sve_ldnt1_gather:
  case Intrinsic::aarch64_sve_ldnt1_gather_index:
  case Intrinsic::aarch64_sve_ldnt1_gather_uxtw:

  // Prefetches: (predicate, base, offsets, prfop)
  case Intrinsic::aarch64_sve_prfb_gather_index:
  case Intrinsic::aarch64_sve_prfb_gather_sxtw_index:
  case Intrinsic::aarch64_sve_prfb_gather_uxtw_index:
  case Intrinsic::aarch64_sve_prfh_gather_index:
  case Intrinsic::aarch64_sve_prfh_gather_sxtw_index:
  case Intrinsic::aarch64_sve_prfh_gather_uxtw_index:
  case Intrinsic::aarch64_sve_prfw_gather_index:
  case Intrinsic::aarch64_sve_prfw_gather_sxtw_index:
  case Intrinsic::aarch64_sve_prfw_gather_uxtw_index:
  case Intrinsic::aarch64_sve_prfd_gather_index:
  case Intrinsic::aarch64_sve_prfd_gather_sxtw_index:
  case Intrinsic::aarch64_sve_prfd_gather_uxtw_index:
    checkSVEGatherScatterOffsets(VS, Call, 2);
    break;

  // Scatter stores: (value, predicate, base, offsets)
  case Intrinsic::aarch64_sve_st1_scatter:
  case Intrinsic::aarch64_sve_st1_scatter_index:
  case Intrinsic::aarch64_sve_st1_scatter_sxtw:
  case Intrinsic::aarch64_sve_st1_scatter_uxtw:
  case Intrinsic::aarch64_sve_st1_scatter_sxtw_index:
  case Intrinsic::aarch64_sve_st1_scatter_uxtw_index:
  case Intrinsic::aarch64_sve_stnt1_scatter:
  case Intrinsic::aarch64_sve_stnt1_scatter_index:
  case Intrinsic::aarch64_sve_stnt1_scatter_uxtw:
    checkSVEGatherScatterOffsets(VS, Call, 3);
    break;
  }
}

#undef Check
