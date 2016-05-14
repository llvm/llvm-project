//===- llvm-pdbdump.h ----------------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_LLVMPDBDUMP_H
#define LLVM_TOOLS_LLVMPDBDUMP_LLVMPDBDUMP_H

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

namespace opts {
extern llvm::cl::opt<bool> Compilands;
extern llvm::cl::opt<bool> Symbols;
extern llvm::cl::opt<bool> Globals;
extern llvm::cl::opt<bool> Types;
extern llvm::cl::opt<bool> All;

extern llvm::cl::opt<bool> ExcludeCompilerGenerated;

extern llvm::cl::opt<bool> NoClassDefs;
extern llvm::cl::opt<bool> NoEnumDefs;
extern llvm::cl::list<std::string> ExcludeTypes;
extern llvm::cl::list<std::string> ExcludeSymbols;
extern llvm::cl::list<std::string> ExcludeCompilands;
extern llvm::cl::list<std::string> IncludeTypes;
extern llvm::cl::list<std::string> IncludeSymbols;
extern llvm::cl::list<std::string> IncludeCompilands;
}

#endif
