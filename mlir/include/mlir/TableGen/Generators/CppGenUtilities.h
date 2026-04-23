//===- CppGenUtilities.h - MLIR C++ gen utilities ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares common utilities for generating C++ files from TableGen
// structures.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_GENERATORS_CPPGENUTILITIES_H
#define MLIR_TABLEGEN_GENERATORS_CPPGENUTILITIES_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace mlir {
namespace tblgen {

/// Emit the summary and description as a C++ comment. If terminateComment
/// is true, terminates the comment with a newline.
void emitSummaryAndDescComments(llvm::raw_ostream &os, llvm::StringRef summary,
                                llvm::StringRef description,
                                bool terminateComment = true);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_GENERATORS_CPPGENUTILITIES_H
