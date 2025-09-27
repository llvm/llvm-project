//===-- DebuggerView.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Options and functions related to --debugger-view for llvm-debuginfo-analyzer
//
//===----------------------------------------------------------------------===//

#ifndef DEBUGGER_VIEW_H
#define DEBUGGER_VIEW_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/CommandLine.h"
#include <vector>

namespace llvm {
class ScopedPrinter;
class raw_ostream;

namespace logicalview {
class LVReaderHandler;
class LVOptions;
} // namespace logicalview

namespace debuggerview {

extern cl::OptionCategory Category;
extern cl::opt<bool> Enable;
int printDebuggerView(std::vector<std::string> &Objects, raw_ostream &OS);

} // namespace debuggerview
} // namespace llvm
#endif // DEBUGGER_VIEW_H
