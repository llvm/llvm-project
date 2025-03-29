//===- ArmSMEStub.cpp - ArmSME ABI routine stubs --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Compiler.h"
#include <cstdint>
#include <iostream>

#if (defined(_WIN32) || defined(__CYGWIN__))
#ifndef MLIR_ARMSMEABISTUBS_EXPORTED
#ifdef mlir_arm_sme_abi_stubs_EXPORTS
// We are building this library
#define MLIR_ARMSMEABISTUBS_EXPORTED __declspec(dllexport)
#else
// We are using this library
#define MLIR_ARMSMEABISTUBS_EXPORTED __declspec(dllimport)
#endif // mlir_arm_sme_abi_stubs_EXPORTS
#endif // MLIR_ARMSMEABISTUBS_EXPORTED
#else
#define MLIR_ARMSMEABISTUBS_EXPORTED                                           \
  __attribute__((visibility("default"))) LLVM_ATTRIBUTE_WEAK
#endif // (defined(_WIN32) || defined(__CYGWIN__))

// The actual implementation of these routines is in:
// compiler-rt/lib/builtins/aarch64/sme-abi.S. These stubs allow the current
// ArmSME tests to run without depending on compiler-rt. This works as we don't
// rely on nested ZA-enabled calls at the moment. The use of these stubs can be
// overridden by setting the ARM_SME_ABI_ROUTINES_SHLIB CMake cache variable to
// a path to an alternate implementation.

extern "C" {

bool MLIR_ARMSMEABISTUBS_EXPORTED __aarch64_sme_accessible() {
  // The ArmSME tests are run within an emulator so we assume SME is available.
  return true;
}

struct sme_state {
  int64_t x0;
  int64_t x1;
};

sme_state MLIR_ARMSMEABISTUBS_EXPORTED __arm_sme_state() {
  std::cerr << "[warning] __arm_sme_state() stubbed!\n";
  return sme_state{};
}

void MLIR_ARMSMEABISTUBS_EXPORTED __arm_tpidr2_restore() {
  std::cerr << "[warning] __arm_tpidr2_restore() stubbed!\n";
}

void MLIR_ARMSMEABISTUBS_EXPORTED __arm_tpidr2_save() {
  std::cerr << "[warning] __arm_tpidr2_save() stubbed!\n";
}

void MLIR_ARMSMEABISTUBS_EXPORTED __arm_za_disable() {
  std::cerr << "[warning] __arm_za_disable() stubbed!\n";
}
}
