//===--- ExegesisAnalyzer.cpp - LLVM Advisor -----------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "Analysis/Inspection/ExegesisAnalyzer.h"

namespace llvm::advisor {

ExegesisAnalyzer::ExegesisAnalyzer()
    : SimpleAnalyzer("llvm.exegesis",
                     "llvm-exegesis requires explicit opcode specification "
                     "(-opcode-name or -snippets-file) and cannot be "
                     "auto-invoked per compilation unit") {}

} // namespace llvm::advisor
