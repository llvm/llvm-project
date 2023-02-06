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

  auto [file, lineCol] = str.split(':');
  auto [lineStr, colStr] = lineCol.split(':');
  if (file.empty()) {
    if (diag)
      diag("error: initializing FileLineColLocBreakpoint with empty file name");
    return failure();
  }

  // Extract the line and column value
  int64_t line = -1, col = -1;
  if (!lineStr.empty() && lineStr.getAsInteger(0, line)) {
    if (diag)
      diag("error: initializing FileLineColLocBreakpoint with a non-numeric "
           "line value: `" +
           Twine(lineStr) + "`");
    return failure();
  }
  if (!colStr.empty() && colStr.getAsInteger(0, col)) {
    if (diag)
      diag("error: initializing FileLineColLocBreakpoint with a non-numeric "
           "col value: `" +
           Twine(colStr) + "`");
    return failure();
  }
  return std::tuple<StringRef, int64_t, int64_t>{file, line, col};
}
