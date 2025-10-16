//===- CppGenUtilities.cpp - MLIR cpp gen utilities --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines common utilities for generating cpp files from tablegen
// structures.
//
//===----------------------------------------------------------------------===//

#include "CppGenUtilities.h"
#include "mlir/Support/IndentedOstream.h"

void mlir::tblgen::emitSummaryAndDescComments(llvm::raw_ostream &os,
                                              llvm::StringRef summary,
                                              llvm::StringRef description,
                                              bool terminateComment) {
  StringRef trimmedSummary = summary.trim();
  StringRef trimmedDesc = description.trim();
  raw_indented_ostream ros(os);

  bool empty = true;
  if (!trimmedSummary.empty()) {
    ros.printReindented(trimmedSummary, "/// ");
    empty = false;
  }

  if (!trimmedDesc.empty()) {
    if (!empty) {
      // If there is a summary, add a newline after it.
      ros << "\n";
    }
    ros.printReindented(trimmedDesc, "/// ");
    empty = false;
  }

  if (!empty && terminateComment)
    ros << "\n";
}
