//===- ArmRunnerUtils.cpp - Utilities for configuring architecture properties //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/MathExtras.h"
#include <iostream>
#include <stdint.h>
#include <string_view>

#if (defined(_WIN32) || defined(__CYGWIN__))
#define MLIR_ARMRUNNERUTILS_EXPORTED __declspec(dllexport)
#else
#define MLIR_ARMRUNNERUTILS_EXPORTED __attribute__((visibility("default")))
#endif

#ifdef __linux__
#include <sys/prctl.h>
#endif

extern "C" {

// Defines for prctl() calls. These may not necessarily exist in the host
// <sys/prctl.h>, but will still be useable under emulation.
//
// https://www.kernel.org/doc/html/v5.3/arm64/sve.html#prctl-extensions
#ifndef PR_SVE_SET_VL
#define PR_SVE_SET_VL 50
#endif
// https://docs.kernel.org/arch/arm64/sme.html#prctl-extensions
#ifndef PR_SME_SET_VL
#define PR_SME_SET_VL 63
#endif
// Note: This mask is the same as both PR_SME_VL_LEN_MASK and
// PR_SVE_VL_LEN_MASK.
#define PR_VL_LEN_MASK 0xffff

static void setArmVectorLength(std::string_view helper_name, int option,
                               uint32_t bits) {
#if defined(__linux__) && defined(__aarch64__)
  if (bits < 128 || bits > 2048 || !llvm::isPowerOf2_32(bits)) {
    std::cerr << "[error] Attempted to set an invalid vector length (" << bits
              << "-bit)" << std::endl;
    abort();
  }
  uint32_t vl = bits / 8;
  if (auto ret = prctl(option, vl & PR_VL_LEN_MASK); ret < 0) {
    std::cerr << "[error] prctl failed (" << ret << ")" << std::endl;
    abort();
  }
#else
  std::cerr << "[error] " << helper_name << " is unsupported" << std::endl;
  abort();
#endif
}

/// Sets the SVE vector length (in bits) to `bits`.
void MLIR_ARMRUNNERUTILS_EXPORTED setArmVLBits(uint32_t bits) {
  setArmVectorLength(__func__, PR_SVE_SET_VL, bits);
}

/// Sets the SME streaming vector length (in bits) to `bits`.
void MLIR_ARMRUNNERUTILS_EXPORTED setArmSVLBits(uint32_t bits) {
  setArmVectorLength(__func__, PR_SME_SET_VL, bits);
}
}
