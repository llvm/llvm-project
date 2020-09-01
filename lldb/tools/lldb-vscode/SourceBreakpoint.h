//===-- SourceBreakpoint.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_VSCODE_SOURCEBREAKPOINT_H
#define LLDB_TOOLS_LLDB_VSCODE_SOURCEBREAKPOINT_H

#include "BreakpointBase.h"
#include "llvm/ADT/StringRef.h"

namespace lldb_vscode {

struct SourceBreakpoint : public BreakpointBase {

  uint32_t line;   ///< The source line of the breakpoint or logpoint
  uint32_t column; ///< An optional source column of the breakpoint

  SourceBreakpoint() : BreakpointBase(), line(0), column(0) {}
  SourceBreakpoint(const llvm::json::Object &obj);

  // Set this breakpoint in LLDB as a new breakpoint
  void SetBreakpoint(const llvm::StringRef source_path);
};

inline bool operator<(const SourceBreakpoint &lhs,
                      const SourceBreakpoint &rhs) {
  if (lhs.line == rhs.line)
    return lhs.column < rhs.column;
  return lhs.line < rhs.line;
}

} // namespace lldb_vscode

#endif
