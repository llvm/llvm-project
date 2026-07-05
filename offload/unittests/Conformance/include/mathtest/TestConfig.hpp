//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of the TestConfig struct and declares the
/// functions for retrieving the set of all and default test configurations.
///
//===----------------------------------------------------------------------===//

#ifndef MATHTEST_TESTCONFIG_HPP
#define MATHTEST_TESTCONFIG_HPP

#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mathtest {

struct TestConfig {
  std::string Provider;
  std::string Platform;

  [[nodiscard]] bool operator==(const TestConfig &RHS) const noexcept {
    return Provider == RHS.Provider && Platform == RHS.Platform;
  }
};

[[nodiscard]] const llvm::SmallVector<TestConfig, 4> &getAllTestConfigs();

[[nodiscard]] const llvm::SmallVector<TestConfig, 4> &getDefaultTestConfigs();
} // namespace mathtest

#endif // MATHTEST_TESTCONFIG_HPP
