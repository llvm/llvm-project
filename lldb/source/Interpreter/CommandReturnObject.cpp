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
    : m_out_stream(colors), m_err_stream(colors),
      m_status(eReturnStatusStarted), m_did_change_process_state(false),
      m_interactive(true) {}

CommandReturnObject::~CommandReturnObject() {}

void CommandReturnObject::AppendErrorWithFormat(const char *format, ...) {
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
  GetOutputStream() << in_string << "\n";
}

void CommandReturnObject::AppendWarning(llvm::StringRef in_string) {
  if (in_string.empty())
    return;
  warning(GetErrorStream()) << in_string << '\n';
}

// Similar to AppendWarning, but do not prepend 'warning: ' to message, and
// don't append "\n" to the end of it.

void CommandReturnObject::AppendRawWarning(llvm::StringRef in_string) {
  if (in_string.empty())
    return;
  GetErrorStream() << in_string;
}

void CommandReturnObject::AppendError(llvm::StringRef in_string) {
  if (in_string.empty())
    return;
  error(GetErrorStream()) << in_string << '\n';
}

void CommandReturnObject::SetError(const Status &error,
                                   const char *fallback_error_cstr) {
  const char *error_cstr = error.AsCString();
  if (error_cstr == nullptr)
    error_cstr = fallback_error_cstr;
  SetError(error_cstr);
}

void CommandReturnObject::SetError(llvm::StringRef error_str) {
  if (error_str.empty())
    return;

  AppendError(error_str);
  SetStatus(eReturnStatusFailed);
}

// Similar to AppendError, but do not prepend 'Status: ' to message, and don't
// append "\n" to the end of it.

void CommandReturnObject::AppendRawError(llvm::StringRef in_string) {
  if (in_string.empty())
    return;
  GetErrorStream() << in_string;
}

void CommandReturnObject::SetStatus(ReturnStatus status) { m_status = status; }

ReturnStatus CommandReturnObject::GetStatus() { return m_status; }

bool CommandReturnObject::Succeeded() {
  return m_status <= eReturnStatusSuccessContinuingResult;
}

bool CommandReturnObject::HasResult() {
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
  m_interactive = true;
}

bool CommandReturnObject::GetDidChangeProcessState() {
  return m_did_change_process_state;
}

void CommandReturnObject::SetDidChangeProcessState(bool b) {
  m_did_change_process_state = b;
}

bool CommandReturnObject::GetInteractive() const { return m_interactive; }

void CommandReturnObject::SetInteractive(bool b) { m_interactive = b; }
