//===- PassInstrumentation.cpp - Pass Instrumentation interface -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file provides the implementation of PassInstrumentation class.
///
//===----------------------------------------------------------------------===//

#include "llvm/IR/PassInstrumentation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

AnalysisKey PassInstrumentationAnalysis::Key;

bool isSpecialPass(StringRef PassID, const std::vector<StringRef> &Specials) {
  size_t Pos = PassID.find('<');
  if (Pos == StringRef::npos)
    return false;
  StringRef Prefix = PassID.substr(0, Pos);
  return any_of(Specials, [Prefix](StringRef S) { return Prefix.endswith(S); });
}

} // namespace llvm
