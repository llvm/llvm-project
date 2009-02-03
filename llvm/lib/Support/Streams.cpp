//===-- Streams.cpp - Wrappers for iostreams ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a wrapper for the std::cout and std::cerr I/O streams.
// It prevents the need to include <iostream> to each file just to get I/O.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Streams.h"
#include <iostream>
using namespace llvm;

OStream llvm::cout(std::cout);
OStream llvm::cerr(std::cerr);
IStream llvm::cin(std::cin);

namespace llvm {

/// FlushStream - Function called by BaseStream to flush an ostream.
void FlushStream(std::ostream &S) {
  S << std::flush;
}

} // end anonymous namespace
