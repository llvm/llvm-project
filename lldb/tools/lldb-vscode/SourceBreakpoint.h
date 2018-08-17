//===-- SourceBreakpoint.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDBVSCODE_SOURCEBREAKPOINT_H_
#define LLDBVSCODE_SOURCEBREAKPOINT_H_

#include "llvm/ADT/StringRef.h"
#include "BreakpointBase.h"

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
