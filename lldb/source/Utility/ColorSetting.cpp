//===-- ColorSetting.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/ColorSetting.h"

#include "lldb/Utility/AnsiTerminal.h"
#include "lldb/Utility/Stream.h"

using namespace lldb_private;

void ColorSetting::render(Stream &s, llvm::StringRef str) const {
  if (!m_prefix.empty())
    s.PutCString(ansi::FormatAnsiTerminalCodes(m_prefix));
  s.PutCString(str);
  if (!m_suffix.empty())
    s.PutCString(ansi::FormatAnsiTerminalCodes(m_suffix));
}
