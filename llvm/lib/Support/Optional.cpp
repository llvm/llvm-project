//===- Optional.cpp - Optional values ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Optional.h"
#include "llvm/Support/raw_ostream.h"

llvm::raw_ostream &llvm::operator<<(raw_ostream &OS, NoneType) {
  return OS << "None";
}
