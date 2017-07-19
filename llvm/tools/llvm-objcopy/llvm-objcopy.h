//===- llvm-objcopy.h -------------------------------------------*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_OBJCOPY_H
#define LLVM_OBJCOPY_H

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

namespace llvm {

LLVM_ATTRIBUTE_NORETURN extern void error(Twine Message);

// This is taken from llvm-readobj.
// [see here](llvm/tools/llvm-readobj/llvm-readobj.h:38)
template <class T> T unwrapOrError(Expected<T> EO) {
  if (EO)
    return *EO;
  std::string Buf;
  raw_string_ostream OS(Buf);
  logAllUnhandledErrors(EO.takeError(), OS, "");
  OS.flush();
  error(Buf);
}
}

#endif
