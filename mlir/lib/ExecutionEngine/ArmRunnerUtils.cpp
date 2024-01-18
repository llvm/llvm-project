//===- ArmRunnerUtils.cpp - Utilities for configuring architecture properties //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

#define PR_VL_LEN_MASK 0xffff

#ifndef PR_SVE_SET_VL
#define PR_SVE_SET_VL 50
#endif

#ifndef PR_SME_SET_VL
#define PR_SME_SET_VL 63
#endif

static void setArmVectorLength(std::string_view helper_name, int option,
                               int bits) {
#if defined(__linux__) && defined(__aarch64__)
  if (bits < 128 || bits % 128 != 0 || bits > 2048) {
    std::cerr << "[error] Invalid aarch64 vector length!" << std::endl;
    abort();
  }
  uint32_t vl = bits / 8;
  if (prctl(option, vl & PR_VL_LEN_MASK) == -1) {
    std::cerr << "[error] prctl failed!" << std::endl;
    abort();
  }
#else
  std::cerr << "[error] " << helper_name << " is unsupported!" << std::endl;
  abort();
#endif
}

void MLIR_ARMRUNNERUTILS_EXPORTED setArmVLBits(uint32_t bits) {
  setArmVectorLength(__func__, PR_SVE_SET_VL, bits);
}

void MLIR_ARMRUNNERUTILS_EXPORTED setArmSVLBits(uint32_t bits) {
  setArmVectorLength(__func__, PR_SME_SET_VL, bits);
}
}
