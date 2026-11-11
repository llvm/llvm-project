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

struct sme_state {
  int64_t x0;
  int64_t x1;
};

LLVM_ALWAYS_EXPORT LLVM_ATTRIBUTE_WEAK sme_state __arm_sme_state() {
  std::cerr << "[warning] __arm_sme_state() stubbed!\n";
  return sme_state{};
}

LLVM_ALWAYS_EXPORT LLVM_ATTRIBUTE_WEAK void __arm_tpidr2_restore() {
  std::cerr << "[warning] __arm_tpidr2_restore() stubbed!\n";
}

LLVM_ALWAYS_EXPORT LLVM_ATTRIBUTE_WEAK void __arm_tpidr2_save() {
  std::cerr << "[warning] __arm_tpidr2_save() stubbed!\n";
}

LLVM_ALWAYS_EXPORT LLVM_ATTRIBUTE_WEAK void __arm_za_disable() {
  std::cerr << "[warning] __arm_za_disable() stubbed!\n";
}
}
