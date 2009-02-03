//===-- CFGPrinter.h - CFG printer external interface ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines external functions that can be called to explicitly
// instantiate the CFG printer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CFGPRINTER_H
#define LLVM_ANALYSIS_CFGPRINTER_H

namespace llvm {
  class FunctionPass;
  FunctionPass *createCFGPrinterPass ();
  FunctionPass *createCFGOnlyPrinterPass ();
} // End llvm namespace

#endif
