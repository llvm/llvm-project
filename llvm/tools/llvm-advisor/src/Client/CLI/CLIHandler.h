//===------------------- CLIHandler.h - LLVM Advisor -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Command-line interface handler. Parses CLI arguments and invokes CoreClient.
// Provides terminal interface for the advisor.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"

namespace llvm::advisor {

class CLIHandler {
public:
  int run(int argc, char **argv);
};

} // namespace llvm::advisor
