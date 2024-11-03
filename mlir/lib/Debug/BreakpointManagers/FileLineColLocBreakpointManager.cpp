//===- FileLineColLocBreakpointManager.cpp - MLIR Optimizer Driver --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Debug/BreakpointManagers/FileLineColLocBreakpointManager.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::tracing;

FailureOr<std::tuple<StringRef, int64_t, int64_t>>
FileLineColLocBreakpoint::parseFromString(StringRef str,
                                          function_ref<void(Twine)> diag) {
  // Watch at debug locations arguments are expected to be in the form:
  // `fileName:line:col`, `fileName:line`, or `fileName`.

  if (str.empty()) {
    if (diag)
      diag("error: initializing FileLineColLocBreakpoint with empty file name");
    return failure();
  }

  // This logic is complex because on Windows `:` is a comment valid path
  // character: `C:\...`.
  auto [fileLine, colStr] = str.rsplit(':');
  auto [file, lineStr] = fileLine.rsplit(':');
  // Extract the line and column value
  int64_t line = -1, col = -1;
  if (lineStr.empty()) {
    // No candidate for line number, try to use the column string as line
    // instead.
    file = fileLine;
    if (!colStr.empty() && colStr.getAsInteger(0, line))
      file = str;
  } else {
    if (lineStr.getAsInteger(0, line)) {
      // Failed to parse a line number, try to use the column string as line
      // instead. If this failed as well, the entire string is the file name.
      file = fileLine;
      if (colStr.getAsInteger(0, line))
        file = str;
    } else {
      // We successfully parsed a line number, try to parse the column number.
      // This shouldn't fail, or the entire string is the file name.
      if (colStr.getAsInteger(0, col)) {
        file = str;
        line = -1;
      }
    }
  }
  return std::tuple<StringRef, int64_t, int64_t>{file, line, col};
}
