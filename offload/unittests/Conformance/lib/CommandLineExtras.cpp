//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of the command-line options and the
/// implementation of the logic for selecting test configurations.
///
//===----------------------------------------------------------------------===//

#include "mathtest/CommandLineExtras.hpp"

#include "mathtest/CommandLine.hpp"
#include "mathtest/TestConfig.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"

using namespace mathtest;

llvm::cl::opt<bool> mathtest::cl::IsVerbose(
    "verbose",
    llvm::cl::desc("Enable verbose output for failed and unsupported tests"),
    llvm::cl::init(false));

llvm::cl::opt<llvm::cl::TestConfigsArg> mathtest::cl::detail::TestConfigsOpt(
    "test-configs", llvm::cl::Optional,
    llvm::cl::desc("Select test configurations"),
    llvm::cl::value_desc("all|provider:platform[,provider:platform...]"));

const llvm::SmallVector<TestConfig, 4> &mathtest::cl::getTestConfigs() {
  switch (detail::TestConfigsOpt.Mode) {
  case llvm::cl::TestConfigsArg::Mode::Default:
    return getDefaultTestConfigs();
  case llvm::cl::TestConfigsArg::Mode::All:
    return getAllTestConfigs();
  case llvm::cl::TestConfigsArg::Mode::Explicit:
    return detail::TestConfigsOpt.Explicit;
  }
  llvm_unreachable("Unknown TestConfigsArg mode");
}
