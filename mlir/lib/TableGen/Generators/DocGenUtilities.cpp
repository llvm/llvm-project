//===- DocGenUtilities.cpp - MLIR doc gen utilities -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines common utilities for generating documentation from TableGen
// structures.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Generators/DocGenUtilities.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/Twine.h"

using namespace mlir;
using namespace mlir::tblgen;
using llvm::StringRef;

void mlir::tblgen::emitSummary(StringRef summary, llvm::raw_ostream &os) {
  if (summary.empty())
    return;
  StringRef trimmed = summary.trim();
  char first = std::toupper(trimmed.front());
  StringRef rest = trimmed.drop_front();
  os << "\n_" << first << rest << "_\n";
}

void mlir::tblgen::emitDescription(StringRef description,
                                   llvm::raw_ostream &os) {
  if (description.empty())
    return;
  os << "\n";
  raw_indented_ostream ros(os);
  StringRef trimmed = description.rtrim(" \t");
  ros.printReindented(trimmed);
  if (!trimmed.ends_with("\n"))
    ros << "\n";
}

void mlir::tblgen::emitDescriptionComment(StringRef description,
                                          llvm::raw_ostream &os,
                                          StringRef prefix) {
  if (description.empty())
    return;
  os << "\n";
  raw_indented_ostream ros(os);
  StringRef trimmed = description.rtrim(" \t");
  ros.printReindented(trimmed, (llvm::Twine(prefix) + "/// ").str());
  if (!trimmed.ends_with("\n"))
    ros << "\n";
}
