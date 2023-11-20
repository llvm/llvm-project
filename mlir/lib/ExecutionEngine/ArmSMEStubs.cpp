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

// The actual implementation of these routines is in:
// compiler-rt/lib/builtins/aarch64/sme-abi.S. These stubs allow the current
// ArmSME tests to run without depending on compiler-rt. This works as we don't
// rely on nested ZA-enabled calls at the moment. The use of these stubs can be
// overridden by setting the ARM_SME_ABI_ROUTINES_SHLIB CMake cache variable to
// a path to an alternate implementation.

extern "C" {

bool LLVM_ATTRIBUTE_WEAK __aarch64_sme_accessible() {
  // The ArmSME tests are run within an emulator so we assume SME is available.
  return true;
}

struct sme_state {
  int64_t x0;
  int64_t x1;
};

sme_state LLVM_ATTRIBUTE_WEAK __arm_sme_state() {
  std::cerr << "[warning] __arm_sme_state() stubbed!\n";
  return sme_state{};
}

void LLVM_ATTRIBUTE_WEAK __arm_tpidr2_restore() {
  std::cerr << "[warning] __arm_tpidr2_restore() stubbed!\n";
}

void LLVM_ATTRIBUTE_WEAK __arm_tpidr2_save() {
  std::cerr << "[warning] __arm_tpidr2_save() stubbed!\n";
}

void LLVM_ATTRIBUTE_WEAK __arm_za_disable() {
  std::cerr << "[warning] __arm_za_disable() stubbed!\n";
}
}
