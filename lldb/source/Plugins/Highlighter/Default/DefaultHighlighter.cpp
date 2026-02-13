//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DefaultHighlighter.h"

LLDB_PLUGIN_DEFINE_ADV(DefaultHighlighter, HighlighterDefault)

using namespace lldb_private;

void DefaultHighlighter::Highlight(const HighlightStyle &options,
                                   llvm::StringRef line,
                                   std::optional<size_t> cursor_pos,
                                   llvm::StringRef previous_lines,
                                   Stream &s) const {
  // If we don't have a valid cursor, then we just print the line as-is.
  if (!cursor_pos || *cursor_pos >= line.size()) {
    s << line;
    return;
  }

  // If we have a valid cursor, we have to apply the 'selected' style around
  // the character below the cursor.

  // Split the line around the character which is below the cursor.
  size_t column = *cursor_pos;
  // Print the characters before the cursor.
  s << line.substr(0, column);
  // Print the selected character with the defined color codes.
  options.selected.Apply(s, line.substr(column, 1));
  // Print the rest of the line.
  s << line.substr(column + 1U);
}

Highlighter *DefaultHighlighter::CreateInstance(lldb::LanguageType language) {
  return new DefaultHighlighter();
}

void DefaultHighlighter::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), GetPluginNameStatic(),
                                CreateInstance);
}

void DefaultHighlighter::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}
