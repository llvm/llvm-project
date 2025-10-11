//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the command-line options and the main
/// interface for selecting test configurations.
///
//===----------------------------------------------------------------------===//

#ifndef MATHTEST_COMMANDLINEEXTRAS_HPP
#define MATHTEST_COMMANDLINEEXTRAS_HPP

#include "mathtest/CommandLine.hpp"
#include "mathtest/TestConfig.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"

namespace mathtest {
namespace cl {

extern llvm::cl::opt<bool> IsVerbose;

namespace detail {

extern llvm::cl::opt<llvm::cl::TestConfigsArg> TestConfigsOpt;
} // namespace detail

const llvm::SmallVector<TestConfig, 4> &getTestConfigs();
} // namespace cl
} // namespace mathtest

#endif // MATHTEST_COMMANDLINEEXTRAS_HPP
