//===-- DiagnosticsRendering.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/DiagnosticsRendering.h"

using namespace lldb_private;
using namespace lldb;

namespace lldb_private {

char DiagnosticError::ID;

lldb::ErrorType DiagnosticError::GetErrorType() const {
  return lldb::eErrorTypeExpression;
}

static llvm::raw_ostream &PrintSeverity(Stream &stream,
                                        lldb::Severity severity) {
  llvm::HighlightColor color;
  llvm::StringRef text;
  switch (severity) {
  case lldb::eSeverityError:
    color = llvm::HighlightColor::Error;
    text = "error: ";
    break;
  case lldb::eSeverityWarning:
    color = llvm::HighlightColor::Warning;
    text = "warning: ";
    break;
  case lldb::eSeverityInfo:
    color = llvm::HighlightColor::Remark;
    text = "note: ";
    break;
  }
  return llvm::WithColor(stream.AsRawOstream(), color, llvm::ColorMode::Enable)
         << text;
}

void RenderDiagnosticDetails(Stream &stream,
                             std::optional<uint16_t> offset_in_command,
                             bool show_inline,
                             llvm::ArrayRef<DiagnosticDetail> details) {
  if (details.empty())
    return;

  if (!offset_in_command) {
    for (const DiagnosticDetail &detail : details) {
      PrintSeverity(stream, detail.severity);
      stream << detail.rendered << '\n';
    }
    return;
  }

  // Since there is no other way to find this out, use the color
  // attribute as a proxy for whether the terminal supports Unicode
  // characters.  In the future it might make sense to move this into
  // Host so it can be customized for a specific platform.
  llvm::StringRef cursor, underline, vbar, joint, hbar, spacer;
  if (stream.AsRawOstream().colors_enabled()) {
    cursor = "˄";
    underline = "˜";
    vbar = "│";
    joint = "╰";
    hbar = "─";
    spacer = " ";
  } else {
    cursor = "^";
    underline = "~";
    vbar = "|";
    joint = "";
    hbar = "";
    spacer = "";
  }

  // Print a line with caret indicator(s) below the lldb prompt + command.
  const size_t padding = *offset_in_command;
  stream << std::string(padding, ' ');

  size_t offset = 1;
  std::vector<DiagnosticDetail> remaining_details, other_details,
      hidden_details;
  for (const DiagnosticDetail &detail : details) {
    if (!show_inline || !detail.source_location) {
      other_details.push_back(detail);
      continue;
    }
    if (detail.source_location->hidden) {
      hidden_details.push_back(detail);
      continue;
    }
    if (!detail.source_location->in_user_input) {
      other_details.push_back(detail);
      continue;
    }

    auto &loc = *detail.source_location;
    remaining_details.push_back(detail);
    if (offset > loc.column)
      continue;
    stream << std::string(loc.column - offset, ' ') << cursor;
    for (unsigned i = 0; i + 1 < loc.length; ++i)
      stream << underline;
    offset = loc.column + 1;
  }
  stream << '\n';

  // Work through each detail in reverse order using the vector/stack.
  bool did_print = false;
  for (auto detail = remaining_details.rbegin();
       detail != remaining_details.rend();
       ++detail, remaining_details.pop_back()) {
    // Get the information to print this detail and remove it from the stack.
    // Print all the lines for all the other messages first.
    stream << std::string(padding, ' ');
    size_t offset = 1;
    for (auto &remaining_detail :
         llvm::ArrayRef(remaining_details).drop_back(1)) {
      uint16_t column = remaining_detail.source_location->column;
      stream << std::string(column - offset, ' ') << vbar;
      offset = column + 1;
    }

    // Print the line connecting the ^ with the error message.
    uint16_t column = detail->source_location->column;
    if (offset <= column)
      stream << std::string(column - offset, ' ') << joint << hbar << spacer;

    // Print a colorized string based on the message's severity type.
    PrintSeverity(stream, detail->severity);

    // Finally, print the message and start a new line.
    stream << detail->message << '\n';
    did_print = true;
  }

  // Print the non-located details.
  for (const DiagnosticDetail &detail : other_details) {
    PrintSeverity(stream, detail.severity);
    stream << detail.rendered << '\n';
    did_print = true;
  }

  // Print the hidden details as a last resort.
  if (!did_print)
    for (const DiagnosticDetail &detail : hidden_details) {
      PrintSeverity(stream, detail.severity);
      stream << detail.rendered << '\n';
    }
}

} // namespace lldb_private
