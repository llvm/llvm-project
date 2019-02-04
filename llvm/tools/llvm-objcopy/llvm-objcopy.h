//===- llvm-objcopy.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_OBJCOPY_OBJCOPY_H
#define LLVM_TOOLS_OBJCOPY_OBJCOPY_H

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace llvm {
namespace objcopy {

LLVM_ATTRIBUTE_NORETURN extern void error(Twine Message);
LLVM_ATTRIBUTE_NORETURN extern void error(Error E);
LLVM_ATTRIBUTE_NORETURN extern void reportError(StringRef File, Error E);
LLVM_ATTRIBUTE_NORETURN extern void reportError(StringRef File,
                                                std::error_code EC);

// This is taken from llvm-readobj.
// [see here](llvm/tools/llvm-readobj/llvm-readobj.h:38)
template <class T> T unwrapOrError(Expected<T> EO) {
  if (EO)
    return *EO;
  std::string Buf;
  raw_string_ostream OS(Buf);
  logAllUnhandledErrors(EO.takeError(), OS);
  OS.flush();
  error(Buf);
}

} // end namespace objcopy
} // end namespace llvm

#endif // LLVM_TOOLS_OBJCOPY_OBJCOPY_H
