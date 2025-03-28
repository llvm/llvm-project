//===-- ExceptionBreakpoint.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_EXCEPTIONBREAKPOINT_H
#define LLDB_TOOLS_LLDB_DAP_EXCEPTIONBREAKPOINT_H

#include "DAPForward.h"
#include "lldb/API/SBBreakpoint.h"
#include "lldb/lldb-enumerations.h"
#include <string>
#include <utility>

namespace lldb_dap {

struct ExceptionBreakpoint {
  DAP &dap;
  std::string filter;
  std::string label;
  lldb::LanguageType language;
  bool default_value = false;
  lldb::SBBreakpoint bp;
  ExceptionBreakpoint(DAP &d, std::string f, std::string l,
                      lldb::LanguageType lang)
      : dap(d), filter(std::move(f)), label(std::move(l)), language(lang),
        bp() {}

  void SetBreakpoint();
  void ClearBreakpoint();
};

} // namespace lldb_dap

#endif
