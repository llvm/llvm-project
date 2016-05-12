//===- yaml2macho - Convert YAML to a Mach object file --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief The Mach component of yaml2obj.
///
//===----------------------------------------------------------------------===//

#include "yaml2obj.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

int yaml2macho(llvm::yaml::Input &YIn, llvm::raw_ostream &Out) {
  errs() << "yaml2obj: Mach-O not implemented yet!\n";
  return 1;
}
