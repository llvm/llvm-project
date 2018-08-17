//===-- ExceptionBreakpoint.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDBVSCODE_EXCEPTIONBREAKPOINT_H_
#define LLDBVSCODE_EXCEPTIONBREAKPOINT_H_

#include <string>

#include "lldb/API/SBBreakpoint.h"

namespace lldb_vscode {

struct ExceptionBreakpoint {
  std::string filter;
  std::string label;
  lldb::LanguageType language;
  bool default_value;
  lldb::SBBreakpoint bp;
  ExceptionBreakpoint(std::string f, std::string l, lldb::LanguageType lang) :
    filter(std::move(f)),
    label(std::move(l)),
    language(lang),
    default_value(false),
    bp() {}

  void SetBreakpoint();
  void ClearBreakpoint();
};

} // namespace lldb_vscode

#endif
