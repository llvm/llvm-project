//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_HIGHLIGHTER_TREESITTER_TREESITTERHIGHLIGHTER_H
#define LLDB_SOURCE_PLUGINS_HIGHLIGHTER_TREESITTER_TREESITTERHIGHLIGHTER_H

#include "lldb/Core/Highlighter.h"
#include "lldb/Utility/Stream.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <tree_sitter/api.h>

namespace lldb_private {

struct TSState;

class TreeSitterHighlighter : public Highlighter {
public:
  TreeSitterHighlighter() = default;
  ~TreeSitterHighlighter() override = default;

  /// Highlights a single line of code using tree-sitter parsing.
  void Highlight(const HighlightStyle &options, llvm::StringRef line,
                 std::optional<size_t> cursor_pos,
                 llvm::StringRef previous_lines, Stream &s) const override;

protected:
  /// Returns the tree-sitter language for this highlighter.
  virtual const TSLanguage *GetLanguage() const = 0;

  /// Returns the tree-sitter highlight query for this language.
  virtual llvm::StringRef GetHighlightQuery() const = 0;

private:
  /// Maps a tree-sitter capture name to a HighlightStyle color.
  HighlightStyle::ColorStyle
  GetStyleForCapture(llvm::StringRef capture_name,
                     const HighlightStyle &options) const;

  /// Applies syntax highlighting to a range of text.
  void HighlightRange(const HighlightStyle &options, llvm::StringRef text,
                      uint32_t start_byte, uint32_t end_byte,
                      const HighlightStyle::ColorStyle &style,
                      std::optional<size_t> cursor_pos,
                      bool &highlighted_cursor, Stream &s) const;

  struct HLRange {
    uint32_t start_byte;
    uint32_t end_byte;
    HighlightStyle::ColorStyle style;
  };

  struct TSState {
    TSState() = default;
    TSState &operator=(const TSState &) = delete;
    TSState(const TSState &) = delete;
    ~TSState();

    explicit operator bool() const;
    TSParser *parser = nullptr;
    TSQuery *query = nullptr;
  };

  /// Lazily creates a tree-sitter state (TSState).
  TSState &GetTSState() const;
  mutable std::optional<TSState> m_ts_state;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_LANGUAGE_TREESITTERCOMMON_TREESITTERHIGHLIGHTER_H
