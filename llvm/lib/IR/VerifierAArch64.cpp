//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

/// Returns the operand index of the vector offsets, or -1 if there is none.
static int getVectorOffsetOperandIndex(Intrinsic::ID ID) {
  switch (ID) {
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
    return 2;

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
    return 3;

  default:
    return -1;
  }
}

void llvm::verifyAArch64IntrinsicCall(VerifierSupport &VS, Intrinsic::ID ID,
                                      CallBase &Call) {
  if (!VS.TT.isAArch64())
    return;

  int OffsetIdx = getVectorOffsetOperandIndex(ID);
  if (OffsetIdx < 0)
    return;

  // The offsets must fit in a single Z register.
  auto *OffsetTy =
      dyn_cast<ScalableVectorType>(Call.getArgOperand(OffsetIdx)->getType());
  if (!OffsetTy)
    return;
  Check(OffsetTy->getPrimitiveSizeInBits().getKnownMinValue() <= 128,
        "SVE gather/scatter offsets must fit in a single Z register", &Call);
}

#undef Check
