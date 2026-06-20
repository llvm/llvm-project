//===--- BinaryAnalysisUtils.cpp - LLVM Advisor --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared helpers for binary analyzers that open object files.
//
//===----------------------------------------------------------------------===//

#include "Analysis/Binary/BinaryAnalysisUtils.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<object::OwningBinary<object::ObjectFile>>
llvm::advisor::openObjectFile(StringRef Path) {
  return object::ObjectFile::createObjectFile(Path);
}
