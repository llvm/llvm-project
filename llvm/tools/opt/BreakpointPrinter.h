//===- BreakpointPrinter.h - Breakpoint location printer ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Breakpoint location printer.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_OPT_BREAKPOINTPRINTER_H
#define LLVM_TOOLS_OPT_BREAKPOINTPRINTER_H

namespace llvm {

class ModulePass;
class raw_ostream;

ModulePass *createBreakpointPrinter(raw_ostream &out);
}

#endif // LLVM_TOOLS_OPT_BREAKPOINTPRINTER_H
