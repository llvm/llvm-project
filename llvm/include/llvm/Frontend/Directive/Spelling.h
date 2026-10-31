//===-------------------------------------------------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_FRONTEND_DIRECTIVE_SPELLING_H
#define LLVM_FRONTEND_DIRECTIVE_SPELLING_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Frontend/Directive/Version.h"
#include "llvm/Support/Compiler.h"

#include <limits>
#include <tuple>

namespace llvm::directive {
struct Spelling {
  StringRef Name;
  VersionRange Versions;
};

LLVM_ABI StringRef FindName(llvm::iterator_range<const Spelling *>,
                            Version Version);
} // namespace llvm::directive

#endif // LLVM_FRONTEND_DIRECTIVE_SPELLING_H
