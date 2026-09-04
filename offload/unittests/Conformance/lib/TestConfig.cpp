//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation for the functions that define the set
/// of all and default test configurations.
///
//===----------------------------------------------------------------------===//

#include "mathtest/TestConfig.hpp"

#include "mathtest/DeviceContext.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"

using namespace mathtest;

[[nodiscard]] const llvm::SmallVector<TestConfig, 4> &
mathtest::getAllTestConfigs() {
  // Thread-safe initialization of a static local variable
  static auto AllTestConfigs = []() -> llvm::SmallVector<TestConfig, 4> {
    return {
        {"llvm-libm", "amdgpu"},
        {"llvm-libm", "cuda"},
        {"cuda-math", "cuda"},
        {"hip-math", "amdgpu"},
    };
  }();

  return AllTestConfigs;
};

[[nodiscard]] const llvm::SmallVector<TestConfig, 4> &
mathtest::getDefaultTestConfigs() {
  // Thread-safe initialization of a static local variable
  static auto DefaultTestConfigs = []() -> llvm::SmallVector<TestConfig, 4> {
    const auto Platforms = getPlatforms();
    const auto AllTestConfigs = getAllTestConfigs();
    llvm::StringRef Provider = "llvm-libm";

    return llvm::filter_to_vector(AllTestConfigs, [&](const auto &Config) {
      return Provider.equals_insensitive(Config.Provider) &&
             llvm::any_of(Platforms, [&](llvm::StringRef Platform) {
               return Platform.equals_insensitive(Config.Platform);
             });
    });
  }();

  return DefaultTestConfigs;
};
