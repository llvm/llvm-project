//===------------------- ExegesisAnalyzer.h - LLVM Advisor ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "Analysis/AnalyzerBase.h"

namespace llvm::advisor {

// llvm-exegesis requires an explicit opcode or snippet specification to run
// benchmarks. It cannot be auto-invoked per compilation unit; it is exposed
// here as a declared capability so callers can request it with explicit args.
class ExegesisAnalyzer final : public SimpleAnalyzer {
public:
  ExegesisAnalyzer();
};

} // namespace llvm::advisor
