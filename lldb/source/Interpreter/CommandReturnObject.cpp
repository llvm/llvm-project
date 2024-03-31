//===-- CommandReturnObject.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Interpreter/CommandReturnObject.h"

#include "lldb/Utility/Status.h"
#include "lldb/Utility/StreamString.h"

#include <regex>

using namespace lldb;
using namespace lldb_private;

static llvm::raw_ostream &error(Stream &strm) {
  return llvm::WithColor(strm.AsRawOstream(), llvm::HighlightColor::Error,
                         llvm::ColorMode::Enable)
         << "error: ";
}

static llvm::raw_ostream &warning(Stream &strm) {
  return llvm::WithColor(strm.AsRawOstream(), llvm::HighlightColor::Warning,
                         llvm::ColorMode::Enable)
         << "warning: ";
}

static llvm::raw_ostream &remark(Stream &strm) {
  return llvm::WithColor(strm.AsRawOstream(), llvm::HighlightColor::Remark,
                         llvm::ColorMode::Enable)
         << "remark: ";
}

static void DumpStringToStreamWithNewline(Stream &strm, const std::string &s) {
  bool add_newline = false;
  if (!s.empty()) {
    // We already checked for empty above, now make sure there is a newline in
    // the error, and if there isn't one, add one.
    strm.Write(s.c_str(), s.size());

    const char last_char = *s.rbegin();
    add_newline = last_char != '\n' && last_char != '\r';
  }
  if (add_newline)
    strm.EOL();
}

CommandReturnObject::CommandReturnObject(bool colors)
    : m_out_stream(colors), m_err_stream(colors) {}

void CommandReturnObject::AppendErrorWithFormat(const char *format, ...) {
  SetStatus(eReturnStatusFailed);

  if (!format)
    return;
  va_list args;
  va_start(args, format);
  StreamString sstrm;
  sstrm.PrintfVarArg(format, args);
  va_end(args);

  const std::string &s = std::string(sstrm.GetString());
  if (!s.empty()) {
    error(GetErrorStream());
    DumpStringToStreamWithNewline(GetErrorStream(), s);
  }
}

void CommandReturnObject::AppendMessageWithFormat(const char *format, ...) {
  if (!format)
    return;
  va_list args;
  va_start(args, format);
  StreamString sstrm;
  sstrm.PrintfVarArg(format, args);
  va_end(args);

  GetOutputStream() << sstrm.GetString();
}

void CommandReturnObject::AppendWarningWithFormat(const char *format, ...) {
  if (!format)
    return;
  va_list args;
  va_start(args, format);
  StreamString sstrm;
  sstrm.PrintfVarArg(format, args);
  va_end(args);

  warning(GetErrorStream()) << sstrm.GetString();
}

void CommandReturnObject::AppendMessage(llvm::StringRef in_string) {
  if (in_string.empty())
    return;
  GetOutputStream() << in_string.rtrim() << '\n';
}

void CommandReturnObject::AppendWarning(llvm::StringRef in_string) {
  if (in_string.empty())
    return;
  warning(GetErrorStream()) << in_string.rtrim() << '\n';
}

void CommandReturnObject::AppendError(llvm::StringRef in_string) {
  SetStatus(eReturnStatusFailed);
  if (in_string.empty())
    return;
  // Workaround to deal with already fully formatted compiler diagnostics.
  llvm::StringRef msg(in_string.rtrim());
  msg.consume_front("error: ");
  error(GetErrorStream()) << msg << '\n';
}

void CommandReturnObject::SetError(const Status &error,
                                   const char *fallback_error_cstr) {
  m_error_status = error;
  if (m_error_status.Fail()) {
    std::vector<Status::Detail> details = m_error_status.GetDetails();
    if (!details.empty()) {
      for (Status::Detail detail : details)
        AppendError(detail.GetMessage());
    } else
      AppendError(error.AsCString(fallback_error_cstr));
  }
}

void CommandReturnObject::SetError(llvm::Error error) {
  if (error)
    AppendError(llvm::toString(std::move(error)));
}

// Similar to AppendError, but do not prepend 'Status: ' to message, and don't
// append "\n" to the end of it.

std::string
CommandReturnObject::DetailStringForPromptCommand(size_t prompt_size,
                                                  llvm::StringRef input) {

  if (input.empty() || m_error_status.GetDetails().empty())
    return "";

  StreamString stream(true);

  struct ExpressionNote {
    std::string message;
    DiagnosticSeverity type;
    size_t column;
  };
  std::vector<ExpressionNote> notes;
  const size_t not_found = std::string::npos;

  // Start off with a sentinel value;
  size_t expression_position = not_found;

  auto note_builder =
      [&](Status::Detail detail) -> std::optional<ExpressionNote> {
    std::vector<std::string> detail_lines = detail.GetMessageLines();

    // This function only knows how to parse messages with 3 lines.
    if (detail_lines.size() != 3)
      return {};

    const DiagnosticSeverity type = detail.GetType();
    const std::string message = detail_lines[0];
    const std::string expression = detail_lines[1];
    const std::string indicator_line = detail_lines[2];

    // Set the position if this is the first time.
    if (expression_position == not_found) {
      expression_position = input.find(expression);

      // Exit early if the expression isn't in the input string.
      if (expression_position == not_found)
        return {};
    }
    // Ensure this note's expression has the same position as the note before.
    else if (input.find(expression) != expression_position)
      return {};

    // Ensure the 3rd line has an indicator (^) in it.
    if (not_found == indicator_line.find("^"))
      return {};

    // The regular expression pattern that isolates the indicator column.
    // - ^ matches the start of a line
    // - (.*?) matches any character before the angle left angle bracket (<)
    // - <(.*?)> matches any characters inside the left angle brackets (<...>)
    // - :([[:digit:]]+): matches 1+ digits between the 1st and 2nd colons (:)
    // - ([[:digit:]]+): matches 1+ digits between the 2nd and 3rd colons (:)
    const std::regex rx{"^(.*?)<(.*?)>:([[:digit:]]+):([[:digit:]]+):"};
    std::smatch match;

    // Exit if the format doesn't match.
    if (!std::regex_search(message, match, rx))
      return {};

    // std::string preamble(match[1]);
    // std::string message_type(match[2]);
    // std::string line_string(match[3]);
    std::string column_string(match[4]);

    unsigned long long column;
    // Convert the column number string into an integer value.
    if (llvm::StringRef(column_string).consumeInteger(10, column))
      return {};

    // Column values start at 1.
    if (column <= 0)
      return {};

    ExpressionNote note;
    note.message = message;
    note.type = type;
    note.column = column;
    return note;
  };

  // Build a list of notes from the details.
  for (Status::Detail &detail : m_error_status.GetDetails()) {
    if (auto note_optional = note_builder(detail))
      notes.push_back(note_optional.value());
    else
      return "";
  }

  // Print a line with caret indicator(s) below the lldb prompt + command.
  const size_t padding = prompt_size + expression_position;
  stream << std::string(padding, ' ');

  size_t offset = 1;
  for (ExpressionNote note : notes) {
    stream << std::string(note.column - offset, ' ') << '^';
    offset = note.column + 1;
  }
  stream << '\n';

  // Work through each note in reverse order using the vector/stack.
  while (!notes.empty()) {
    // Get the information to print this note and remove it from the stack.
    ExpressionNote this_note = notes.back();
    notes.pop_back();

    // Print all the lines for all the other messages first.
    stream << std::string(padding, ' ');
    size_t offset = 1;
    for (ExpressionNote remaining_note : notes) {
      stream << std::string(remaining_note.column - offset, ' ') << "│";
      offset = remaining_note.column + 1;
    }

    // Print the line connecting the ^ with the error message.
    stream << std::string(this_note.column - offset, ' ') << "╰─ ";

    // Print a colorized string based on the message's severity type.
    switch (this_note.type) {
    case eDiagnosticSeverityError:
      error(stream);
      break;
    case eDiagnosticSeverityWarning:
      warning(stream);
      break;
    case eDiagnosticSeverityRemark:
      remark(stream);
      break;
    }

    // Finally, print the message and start a new line.
    stream << this_note.message << '\n';
  }
  return stream.GetData();
}

void CommandReturnObject::AppendRawError(llvm::StringRef in_string) {
  SetStatus(eReturnStatusFailed);
  assert(!in_string.empty() && "Expected a non-empty error message");
  GetErrorStream() << in_string;
}

void CommandReturnObject::SetStatus(ReturnStatus status) { m_status = status; }

ReturnStatus CommandReturnObject::GetStatus() const { return m_status; }

bool CommandReturnObject::Succeeded() const {
  return m_status <= eReturnStatusSuccessContinuingResult;
}

bool CommandReturnObject::HasResult() const {
  return (m_status == eReturnStatusSuccessFinishResult ||
          m_status == eReturnStatusSuccessContinuingResult);
}

void CommandReturnObject::Clear() {
  lldb::StreamSP stream_sp;
  stream_sp = m_out_stream.GetStreamAtIndex(eStreamStringIndex);
  if (stream_sp)
    static_cast<StreamString *>(stream_sp.get())->Clear();
  stream_sp = m_err_stream.GetStreamAtIndex(eStreamStringIndex);
  if (stream_sp)
    static_cast<StreamString *>(stream_sp.get())->Clear();
  m_status = eReturnStatusStarted;
  m_did_change_process_state = false;
  m_suppress_immediate_output = false;
  m_interactive = true;
}

bool CommandReturnObject::GetDidChangeProcessState() const {
  return m_did_change_process_state;
}

void CommandReturnObject::SetDidChangeProcessState(bool b) {
  m_did_change_process_state = b;
}

bool CommandReturnObject::GetInteractive() const { return m_interactive; }

void CommandReturnObject::SetInteractive(bool b) { m_interactive = b; }

bool CommandReturnObject::GetSuppressImmediateOutput() const {
  return m_suppress_immediate_output;
}

void CommandReturnObject::SetSuppressImmediateOutput(bool b) {
  m_suppress_immediate_output = b;
}
