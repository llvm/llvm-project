//===-- DiagnosticsRendering.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/DiagnosticsRendering.h"
#include <cstdint>

using namespace lldb_private;
using namespace lldb;

namespace lldb_private {

char DiagnosticError::ID;

lldb::ErrorType DiagnosticError::GetErrorType() const {
  return lldb::eErrorTypeExpression;
}

StructuredData::ObjectSP Serialize(llvm::ArrayRef<DiagnosticDetail> details) {
  auto make_array = []() { return std::make_unique<StructuredData::Array>(); };
  auto make_dict = []() {
    return std::make_unique<StructuredData::Dictionary>();
  };
  auto dict_up = make_dict();
  dict_up->AddIntegerItem("version", 1u);
  auto array_up = make_array();
  for (const DiagnosticDetail &diag : details) {
    auto detail_up = make_dict();
    if (auto &sloc = diag.source_location) {
      auto sloc_up = make_dict();
      sloc_up->AddStringItem("file", sloc->file.GetPath());
      sloc_up->AddIntegerItem("line", sloc->line);
      sloc_up->AddIntegerItem("length", sloc->length);
      sloc_up->AddBooleanItem("hidden", sloc->hidden);
      sloc_up->AddBooleanItem("in_user_input", sloc->in_user_input);
      detail_up->AddItem("source_location", std::move(sloc_up));
    }
    llvm::StringRef severity = "unknown";
    switch (diag.severity) {
    case lldb::eSeverityError:
      severity = "error";
      break;
    case lldb::eSeverityWarning:
      severity = "warning";
      break;
    case lldb::eSeverityInfo:
      severity = "note";
      break;
    }
    detail_up->AddStringItem("severity", severity);
    detail_up->AddStringItem("message", diag.message);
    detail_up->AddStringItem("rendered", diag.rendered);
    array_up->AddItem(std::move(detail_up));
  }
  dict_up->AddItem("details", std::move(array_up));
  return dict_up;
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

  // Partition the diagnostics.
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

    remaining_details.push_back(detail);
  }

  // Sort the diagnostics.
  auto sort = [](std::vector<DiagnosticDetail> &ds) {
    llvm::stable_sort(ds, [](auto &d1, auto &d2) {
      auto l1 = d1.source_location.value_or(DiagnosticDetail::SourceLocation{});
      auto l2 = d2.source_location.value_or(DiagnosticDetail::SourceLocation{});
      return std::tie(l1.line, l1.column) < std::tie(l2.line, l2.column);
    });
  };
  sort(remaining_details);
  sort(other_details);
  sort(hidden_details);

  // Print a line with caret indicator(s) below the lldb prompt + command.
  const size_t padding = *offset_in_command;
  stream << std::string(padding, ' ');
  {
    size_t x_pos = 1;
    for (const DiagnosticDetail &detail : remaining_details) {
      auto &loc = *detail.source_location;

      if (x_pos > loc.column)
        continue;

      stream << std::string(loc.column - x_pos, ' ') << cursor;
      x_pos = loc.column + 1;
      for (unsigned i = 0; i + 1 < loc.length; ++i) {
        stream << underline;
        x_pos += 1;
      }
    }
  }
  stream << '\n';

  // Reverse the order within groups of diagnostics that are on the same column.
  auto group = [](std::vector<DiagnosticDetail> &details) {
    for (auto it = details.begin(), end = details.end(); it != end;) {
      auto eq_end = std::find_if(it, end, [&](const DiagnosticDetail &d) {
        return d.source_location->column != it->source_location->column;
      });
      std::reverse(it, eq_end);
      it = eq_end;
    }
  };
  group(remaining_details);

  // Work through each detail in reverse order using the vector/stack.
  bool did_print = false;
  for (; !remaining_details.empty(); remaining_details.pop_back()) {
    const auto &detail = remaining_details.back();
    // Get the information to print this detail and remove it from the stack.
    // Print all the lines for all the other messages first.
    stream << std::string(padding, ' ');
    size_t x_pos = 1;
    for (auto &remaining_detail :
         llvm::ArrayRef(remaining_details).drop_back(1)) {
      uint16_t column = remaining_detail.source_location->column;
      // Is this a note with the same column as another diagnostic?
      if (column == detail.source_location->column)
        continue;

      if (column >= x_pos) {
        stream << std::string(column - x_pos, ' ') << vbar;
        x_pos = column + 1;
      }
    }

    uint16_t column = detail.source_location->column;
    // Print the line connecting the ^ with the error message.
    if (column >= x_pos)
      stream << std::string(column - x_pos, ' ') << joint << hbar << spacer;

    // Print a colorized string based on the message's severity type.
    PrintSeverity(stream, detail.severity);

    // Finally, print the message and start a new line.
    stream << detail.message << '\n';
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
