//===-- Support/DJB.cpp ---DJB Hash -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for the DJ Bernstein hash function.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/DJB.h"

uint32_t llvm::djbHash(StringRef Buffer, uint32_t H) {
  for (char C : Buffer.bytes())
    H = ((H << 5) + H) + C;
  return H;
}
