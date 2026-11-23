//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_HIGHLIGHTER_DEFAULT_DEFAULTHIGHLIGHTER_H
#define LLDB_SOURCE_PLUGINS_HIGHLIGHTER_DEFAULT_DEFAULTHIGHLIGHTER_H

#include "lldb/Core/Highlighter.h"

namespace lldb_private {

/// A default highlighter that only highlights the user cursor, but doesn't
/// do any other highlighting.
class DefaultHighlighter : public Highlighter {
public:
  llvm::StringRef GetName() const override { return "none"; }

  void Highlight(const HighlightStyle &options, llvm::StringRef line,
                 std::optional<size_t> cursor_pos,
                 llvm::StringRef previous_lines, Stream &s) const override;

  static Highlighter *CreateInstance(lldb::LanguageType language);

  static void Terminate();
  static void Initialize();

  static llvm::StringRef GetPluginNameStatic() { return "Default Highlighter"; }
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }
};

} // namespace lldb_private

#endif
