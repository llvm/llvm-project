//===- CppGenUtilities.h - MLIR cpp gen utilities ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines common utilities for generating cpp files from tablegen
// structures.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRTBLGEN_CPPGENUTILITIES_H_
#define MLIR_TOOLS_MLIRTBLGEN_CPPGENUTILITIES_H_

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace tblgen {

// Emit the summary and description as a C++ comment. If `terminateComment` is
// true, terminates the comment with a `\n`.
void emitSummaryAndDescComments(llvm::raw_ostream &os, llvm::StringRef summary,
                                llvm::StringRef description,
                                bool terminateComment = true);
} // namespace tblgen
} // namespace mlir

#endif // MLIR_TOOLS_MLIRTBLGEN_CPPGENUTILITIES_H_
