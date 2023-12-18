//===-- BreakpointBase.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_BREAKPOINTBASE_H
#define LLDB_TOOLS_LLDB_DAP_BREAKPOINTBASE_H

#include "JSONUtils.h"
#include "lldb/API/SBBreakpoint.h"
#include "llvm/Support/JSON.h"
#include <string>
#include <vector>

namespace lldb_dap {

struct BreakpointBase {
  // logMessage part can be either a raw text or an expression.
  struct LogMessagePart {
    LogMessagePart(llvm::StringRef text, bool is_expr)
        : text(text), is_expr(is_expr) {}
    std::string text;
    bool is_expr;
  };
  // An optional expression for conditional breakpoints.
  std::string condition;
  // An optional expression that controls how many hits of the breakpoint are
  // ignored. The backend is expected to interpret the expression as needed
  std::string hitCondition;
  // If this attribute exists and is non-empty, the backend must not 'break'
  // (stop) but log the message instead. Expressions within {} are
  // interpolated.
  std::string logMessage;
  std::vector<LogMessagePart> logMessageParts;
  // The LLDB breakpoint associated wit this source breakpoint
  lldb::SBBreakpoint bp;

  BreakpointBase() = default;
  BreakpointBase(const llvm::json::Object &obj);

  void SetCondition();
  void SetHitCondition();
  void SetLogMessage();
  void UpdateBreakpoint(const BreakpointBase &request_bp);

  // Format \param text and return formatted text in \param formatted.
  // \return any formatting failures.
  lldb::SBError FormatLogText(llvm::StringRef text, std::string &formatted);
  lldb::SBError AppendLogMessagePart(llvm::StringRef part, bool is_expr);
  void NotifyLogMessageError(llvm::StringRef error);

  static const char *GetBreakpointLabel();
  static bool BreakpointHitCallback(void *baton, lldb::SBProcess &process,
                                    lldb::SBThread &thread,
                                    lldb::SBBreakpointLocation &location);
};

} // namespace lldb_dap

#endif
