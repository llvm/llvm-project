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

std::string
mlir::tblgen::emitSummaryAndDescComments(llvm::StringRef summary,
                                         llvm::StringRef description) {

  std::string comments = "";
  StringRef trimmedSummary = summary.trim();
  StringRef trimmedDesc = description.trim();
  llvm::raw_string_ostream os(comments);
  raw_indented_ostream ros(os);

  if (!trimmedSummary.empty()) {
    ros.printReindented(trimmedSummary, "/// ");
  }

  if (!trimmedDesc.empty()) {
    if (!trimmedSummary.empty()) {
      // If there is a summary, add a newline after it.
      ros << "\n";
    }
    ros.printReindented(trimmedDesc, "/// ");
  }
  return comments;
}
