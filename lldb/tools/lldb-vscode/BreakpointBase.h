//===-- BreakpointBase.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDBVSCODE_BREAKPOINTBASE_H_
#define LLDBVSCODE_BREAKPOINTBASE_H_

#include "JSONUtils.h"
#include "lldb/API/SBBreakpoint.h"
#include "llvm/Support/JSON.h"
#include <string>

namespace lldb_vscode {
  
struct BreakpointBase {

  // An optional expression for conditional breakpoints.
  std::string condition;
  // An optional expression that controls how many hits of the breakpoint are
  // ignored. The backend is expected to interpret the expression as needed
  std::string hitCondition;
  // If this attribute exists and is non-empty, the backend must not 'break'
  // (stop) but log the message instead. Expressions within {} are
  // interpolated.
  std::string logMessage;
  // The LLDB breakpoint associated wit this source breakpoint
  lldb::SBBreakpoint bp;

  BreakpointBase() = default;
  BreakpointBase(const llvm::json::Object &obj);

  void SetCondition();
  void SetHitCondition();
  void UpdateBreakpoint(const BreakpointBase &request_bp);
};

} // namespace lldb_vscode

#endif
