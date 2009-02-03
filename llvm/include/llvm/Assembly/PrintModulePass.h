//===- llvm/Assembly/PrintModulePass.h - Printing Pass ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines two passes to print out a module.  The PrintModulePass pass
// simply prints out the entire module when it is executed.  The
// PrintFunctionPass class is designed to be pipelined with other
// FunctionPass's, and prints out the functions of the module as they are
// processed.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ASSEMBLY_PRINTMODULEPASS_H
#define LLVM_ASSEMBLY_PRINTMODULEPASS_H

#include <string>

namespace llvm {
  class FunctionPass;
  class ModulePass;
  class raw_ostream;
  
  /// createPrintModulePass - Create and return a pass that writes the
  /// module to the specified raw_ostream.
  ModulePass *createPrintModulePass(raw_ostream *OS, bool DeleteStream=false);
  
  /// createPrintFunctionPass - Create and return a pass that prints
  /// functions to the specified raw_ostream as they are processed.
  FunctionPass *createPrintFunctionPass(const std::string &Banner,
                                        raw_ostream *OS, 
                                        bool DeleteStream=false);  

} // End llvm namespace

#endif
