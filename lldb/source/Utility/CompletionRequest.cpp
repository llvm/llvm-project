//===-- CompletionRequest.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/CompletionRequest.h"

using namespace lldb;
using namespace lldb_private;

CompletionRequest::CompletionRequest(llvm::StringRef command,
                                     unsigned raw_cursor_pos, Args &parsed_line,
                                     int cursor_index, int cursor_char_position,
                                     int match_start_point,
                                     int max_return_elements,
                                     bool word_complete, StringList &matches)
    : m_command(command), m_raw_cursor_pos(raw_cursor_pos),
      m_parsed_line(parsed_line), m_cursor_index(cursor_index),
      m_cursor_char_position(cursor_char_position),
      m_match_start_point(match_start_point),
      m_max_return_elements(max_return_elements),
      m_word_complete(word_complete), m_matches(&matches) {}
