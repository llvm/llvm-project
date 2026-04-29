//===------------------- BuildIntegration.h - LLVM Advisor ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Ingests compile_commands.json and normalizes build commands.
// Bridges between build system and Advisor's internal representation.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "AdvisorCommon.h"
#include "Core/AdvisorTypes.h"

namespace llvm::advisor {

class BuildIntegration {
public:
  Expected<SmallVector<CompileCommand, 64>>
  loadCompileCommands(StringRef BuildRoot) const;
};

} // namespace llvm::advisor
