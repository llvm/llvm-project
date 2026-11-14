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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/TableGen/Error.h"

void mlir::tblgen::emitSummaryAndDescComments(llvm::raw_ostream &os,
                                              llvm::StringRef summary,
                                              llvm::StringRef description,
                                              bool terminateComment) {
  StringRef trimmedSummary = summary.rtrim();
  StringRef trimmedDesc = description.rtrim();
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

// Returns a description of what makes `header` unsafe to emit verbatim in
// `#include "<header>"`, or an empty string if it is safe. A header is unsafe
// if it could end the header name or the line, or start a comment.
static llvm::StringRef getUnsafeHeaderReason(llvm::StringRef header) {
  if (header.contains('"'))
    return "'\"'";
  if (header.contains('\\'))
    return "'\\'";
  if (header.contains("/*") || header.contains("//"))
    return "a comment";
  if (!llvm::all_of(header, [](char c) { return llvm::isPrint(c); }))
    return "non-printable characters";
  return "";
}

void mlir::tblgen::emitIncludes(raw_ostream &os, ArrayRef<StringRef> headers) {
  SmallVector<StringRef> sorted(headers);
  llvm::sort(sorted);
  sorted.erase(llvm::unique(sorted), sorted.end());
  for (StringRef header : sorted) {
    StringRef reason = getUnsafeHeaderReason(header);
    if (!reason.empty()) {
      std::string escaped;
      llvm::raw_string_ostream(escaped).write_escaped(header);
      llvm::PrintFatalError("cannot emit '#include' for header \"" + escaped +
                            "\": header must not contain " + reason);
    }
    os << "#include \"" << header << "\"\n";
  }
  if (!sorted.empty())
    os << "\n";
}
