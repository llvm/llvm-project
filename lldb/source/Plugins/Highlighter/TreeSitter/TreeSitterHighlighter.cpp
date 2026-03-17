//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TreeSitterHighlighter.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"

using namespace lldb_private;

TreeSitterHighlighter::TSState::~TSState() {
  if (query)
    ts_query_delete(query);
  if (parser)
    ts_parser_delete(parser);
}

TreeSitterHighlighter::TSState::operator bool() const {
  return parser && query;
}

TreeSitterHighlighter::TSState &TreeSitterHighlighter::GetTSState() const {
  if (m_ts_state)
    return *m_ts_state;

  Log *log = GetLog(LLDBLog::Source);

  m_ts_state.emplace();
  m_ts_state->parser = ts_parser_new();
  if (!m_ts_state->parser) {
    LLDB_LOG(log, "Creating tree-sitter parser failed for {0}", GetName());
    return *m_ts_state;
  }

  const TSLanguage *language = GetLanguage();
  if (!language || !ts_parser_set_language(m_ts_state->parser, language)) {
    LLDB_LOG(log, "Creating tree-sitter language failed for {0}", GetName());
    return *m_ts_state;
  }

  llvm::StringRef query_source = GetHighlightQuery();
  uint32_t error_offset = 0;
  TSQueryError error_type = TSQueryErrorNone;
  m_ts_state->query = ts_query_new(language, query_source.data(),
                                   static_cast<uint32_t>(query_source.size()),
                                   &error_offset, &error_type);
  if (!m_ts_state->query || error_type != TSQueryErrorNone) {
    LLDB_LOG(log,
             "Creating tree-sitter query failed for {0} with error {1}: {2}",
             GetName(), error_type, query_source.substr(error_offset, 64));
    // If we have an error but a valid query, we need to reset the object to
    // (1) avoid it looking valid and (2) release the parser.
    m_ts_state.emplace();
  }

  return *m_ts_state;
}

HighlightStyle::ColorStyle
TreeSitterHighlighter::GetStyleForCapture(llvm::StringRef capture_name,
                                          const HighlightStyle &options) const {
  return llvm::StringSwitch<HighlightStyle::ColorStyle>(capture_name)
      .Case("comment", options.comment)
      .Case("keyword", options.keyword)
      .Case("operator", options.operators)
      .Case("type", options.keyword)
      .Case("punctuation.delimiter.comma", options.comma)
      .Case("punctuation.delimiter.colon", options.colon)
      .Case("punctuation.delimiter.semicolon", options.semicolons)
      .Case("punctuation.bracket.square", options.square_brackets)
      .Cases({"keyword.directive", "preproc"}, options.pp_directive)
      .Cases({"string", "string.literal"}, options.string_literal)
      .Cases({"number", "number.literal", "constant.numeric"},
             options.scalar_literal)
      .Cases({"identifier", "variable", "function"}, options.identifier)
      .Cases({"punctuation.bracket.curly", "punctuation.brace"}, options.braces)
      .Cases({"punctuation.bracket.round", "punctuation.bracket",
              "punctuation.paren"},
             options.parentheses)
      .Default({});
}

void TreeSitterHighlighter::HighlightRange(
    const HighlightStyle &options, llvm::StringRef text, uint32_t start_byte,
    uint32_t end_byte, const HighlightStyle::ColorStyle &style,
    std::optional<size_t> cursor_pos, bool &highlighted_cursor,
    Stream &s) const {

  if (start_byte >= end_byte || start_byte >= text.size())
    return;

  end_byte = std::min(end_byte, static_cast<uint32_t>(text.size()));

  llvm::StringRef range = text.substr(start_byte, end_byte - start_byte);

  auto print = [&](llvm::StringRef str) {
    if (style)
      style.Apply(s, str);
    else
      s << str;
  };

  // Check if cursor is within this range.
  if (cursor_pos && *cursor_pos >= start_byte && *cursor_pos < end_byte &&
      !highlighted_cursor) {
    highlighted_cursor = true;

    // Split range around cursor position.
    const size_t cursor_in_range = *cursor_pos - start_byte;

    // Print everything before the cursor.
    if (cursor_in_range > 0) {
      llvm::StringRef before = range.substr(0, cursor_in_range);
      print(before);
    }

    // Print the cursor itself.
    if (cursor_in_range < range.size()) {
      StreamString cursor_str;
      llvm::StringRef cursor_char = range.substr(cursor_in_range, 1);
      if (style)
        style.Apply(cursor_str, cursor_char);
      else
        cursor_str << cursor_char;
      options.selected.Apply(s, cursor_str.GetString());
    }

    // Print everything after the cursor.
    if (cursor_in_range + 1 < range.size()) {
      llvm::StringRef after = range.substr(cursor_in_range + 1);
      print(after);
    }
  } else {
    // No cursor in this range, apply style directly.
    print(range);
  }
}

void TreeSitterHighlighter::Highlight(const HighlightStyle &options,
                                      llvm::StringRef line,
                                      std::optional<size_t> cursor_pos,
                                      llvm::StringRef previous_lines,
                                      Stream &s) const {
  auto unformatted = [&]() -> void { s << line; };

  TSState &ts_state = GetTSState();
  if (!ts_state)
    return unformatted();

  std::string source = previous_lines.str() + line.str();
  TSTree *tree =
      ts_parser_parse_string(ts_state.parser, nullptr, source.c_str(),
                             static_cast<uint32_t>(source.size()));
  if (!tree)
    return unformatted();

  TSQueryCursor *cursor = ts_query_cursor_new();
  assert(cursor);

  llvm::scope_exit delete_cusor([&] { ts_query_cursor_delete(cursor); });

  TSNode root_node = ts_tree_root_node(tree);
  ts_query_cursor_exec(cursor, ts_state.query, root_node);

  // Collect all matches and their byte ranges.
  std::vector<HLRange> highlights;
  TSQueryMatch match;
  uint32_t capture_index;
  while (ts_query_cursor_next_capture(cursor, &match, &capture_index)) {
    TSQueryCapture capture = match.captures[capture_index];

    uint32_t capture_name_len = 0;
    const char *capture_name = ts_query_capture_name_for_id(
        ts_state.query, capture.index, &capture_name_len);

    HighlightStyle::ColorStyle style = GetStyleForCapture(
        llvm::StringRef(capture_name, capture_name_len), options);

    TSNode node = capture.node;
    uint32_t start = ts_node_start_byte(node);
    uint32_t end = ts_node_end_byte(node);

    if (style && start < end)
      highlights.push_back({start, end, style});
  }

  std::sort(highlights.begin(), highlights.end(),
            [](const HLRange &a, const HLRange &b) {
              if (a.start_byte != b.start_byte)
                return a.start_byte < b.start_byte;
              // Prefer shorter matches. For example, if we have an expression
              // consisting of a variable and a property, we want to highlight
              // them as individual components.
              return (b.end_byte - b.start_byte) > (a.end_byte - a.start_byte);
            });

  uint32_t current_pos = 0;
  bool highlighted_cursor = false;

  for (const auto &h : highlights) {
    // Skip over highlights that start before our current position, which means
    // there's overlap.
    if (h.start_byte < current_pos)
      continue;

    // Output any unhighlighted text before this highlight.
    if (current_pos < h.start_byte) {
      HighlightRange(options, line, current_pos, h.start_byte, {}, cursor_pos,
                     highlighted_cursor, s);
      current_pos = h.start_byte;
    }

    // Output the highlighted range.
    HighlightRange(options, line, h.start_byte, h.end_byte, h.style, cursor_pos,
                   highlighted_cursor, s);
    current_pos = h.end_byte;
  }

  // Output any remaining unhighlighted text.
  if (current_pos < line.size()) {
    HighlightRange(options, line, current_pos,
                   static_cast<uint32_t>(line.size()), {}, cursor_pos,
                   highlighted_cursor, s);
  }
}
