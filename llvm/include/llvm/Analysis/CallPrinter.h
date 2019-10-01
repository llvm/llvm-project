//===-- CallPrinter.h - Call graph printer external interface ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines external functions that can be called to explicitly
// instantiate the call graph printer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CALLPRINTER_H
#define LLVM_ANALYSIS_CALLPRINTER_H

namespace llvm {

class ModulePass;

ModulePass *createCallGraphViewerPass();
ModulePass *createCallGraphDOTPrinterPass();

} // end namespace llvm

#endif
