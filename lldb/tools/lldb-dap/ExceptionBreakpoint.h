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
#include "llvm/ADT/StringRef.h"
#include <string>
#include <utility>

namespace lldb_dap {

class ExceptionBreakpoint {
public:
  ExceptionBreakpoint(DAP &d, std::string f, std::string l,
                      lldb::LanguageType lang)
      : m_dap(d), m_filter(std::move(f)), m_label(std::move(l)),
        m_language(lang), m_bp() {}

  void SetBreakpoint();
  void ClearBreakpoint();

  lldb::break_id_t GetID() const { return m_bp.GetID(); }
  llvm::StringRef GetFilter() const { return m_filter; }
  llvm::StringRef GetLabel() const { return m_label; }

  static constexpr bool kDefaultValue = false;

protected:
  DAP &m_dap;
  std::string m_filter;
  std::string m_label;
  lldb::LanguageType m_language;
  lldb::SBBreakpoint m_bp;
};

} // namespace lldb_dap

#endif
