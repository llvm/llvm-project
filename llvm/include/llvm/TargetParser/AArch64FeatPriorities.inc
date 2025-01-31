//===- AArch64FeatPriorities.inc - AArch64 FMV Priorities enum --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file enumerates the AArch64 FMV features sorted in ascending priority.
//
//===----------------------------------------------------------------------===//

#ifndef AARCH64_FEAT_PRIORITIES_INC_H
#define AARCH64_FEAT_PRIORITIES_INC_H

// Function Multi Versioning feature priorities.
enum FeatPriorities {
  PRIOR_RNG,
  PRIOR_FLAGM,
  PRIOR_FLAGM2,
  PRIOR_LSE,
  PRIOR_FP,
  PRIOR_SIMD,
  PRIOR_DOTPROD,
  PRIOR_SM4,
  PRIOR_RDM,
  PRIOR_CRC,
  PRIOR_SHA2,
  PRIOR_SHA3,
  PRIOR_PMULL,
  PRIOR_FP16,
  PRIOR_FP16FML,
  PRIOR_DIT,
  PRIOR_DPB,
  PRIOR_DPB2,
  PRIOR_JSCVT,
  PRIOR_FCMA,
  PRIOR_RCPC,
  PRIOR_RCPC2,
  PRIOR_RCPC3,
  PRIOR_FRINTTS,
  PRIOR_I8MM,
  PRIOR_BF16,
  PRIOR_SVE,
  PRIOR_SVE_F32MM,
  PRIOR_SVE_F64MM,
  PRIOR_SVE2,
  PRIOR_SVE_PMULL128,
  PRIOR_SVE_BITPERM,
  PRIOR_SVE_SHA3,
  PRIOR_SVE_SM4,
  PRIOR_SME,
  PRIOR_MEMTAG2,
  PRIOR_SB,
  PRIOR_PREDRES,
  PRIOR_SSBS2,
  PRIOR_BTI,
  PRIOR_LS64_ACCDATA,
  PRIOR_WFXT,
  PRIOR_SME_F64,
  PRIOR_SME_I64,
  PRIOR_SME2,
  PRIOR_MOPS
};

#endif
