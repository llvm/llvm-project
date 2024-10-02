//===-- DiagnosticManager.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/DiagnosticManager.h"

#include "llvm/Support/ErrorHandling.h"

#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb_private;
char ExpressionError::ID;

/// A std::error_code category for eErrorTypeExpression.
class ExpressionCategory : public std::error_category {
  const char *name() const noexcept override {
    return "LLDBExpressionCategory";
  }
  std::string message(int __ev) const override {
    return ExpressionResultAsCString(
        static_cast<lldb::ExpressionResults>(__ev));
  };
};
ExpressionCategory &expression_category() {
  static ExpressionCategory g_expression_category;
  return g_expression_category;
}

ExpressionError::ExpressionError(lldb::ExpressionResults result,
                                 std::string msg,
                                 std::vector<DiagnosticDetail> details)
    : ErrorInfo(std::error_code(result, expression_category())), m_message(msg),
      m_details(details) {}

static llvm::StringRef StringForSeverity(lldb::Severity severity) {
  switch (severity) {
  // this should be exhaustive
  case lldb::eSeverityError:
    return "error: ";
  case lldb::eSeverityWarning:
    return "warning: ";
  case lldb::eSeverityInfo:
    return "";
  }
  llvm_unreachable("switch needs another case for lldb::Severity enum");
}

std::string ExpressionError::message() const {
  std::string str;
  {
    llvm::raw_string_ostream os(str);
    if (!m_message.empty())
      os << m_message << '\n';
    for (const auto &detail : m_details)
      os << StringForSeverity(detail.severity) << detail.rendered << '\n';
  }
  return str;
}

std::error_code ExpressionError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

void ExpressionError::log(llvm::raw_ostream &OS) const { OS << message(); }

std::unique_ptr<CloneableError> ExpressionError::Clone() const {
  return std::make_unique<ExpressionError>(
      (lldb::ExpressionResults)convertToErrorCode().value(), m_message,
      m_details);
}

std::string DiagnosticManager::GetString(char separator) {
  std::string str;
  llvm::raw_string_ostream stream(str);

  for (const auto &diagnostic : Diagnostics()) {
    llvm::StringRef severity = StringForSeverity(diagnostic->GetSeverity());
    stream << severity;

    llvm::StringRef message = diagnostic->GetMessage();
    std::string searchable_message = message.lower();
    auto severity_pos = message.find(severity);
    stream << message.take_front(severity_pos);

    if (severity_pos != llvm::StringRef::npos)
      stream << message.drop_front(severity_pos + severity.size());
    stream << separator;
  }
  return str;
}

void DiagnosticManager::Dump(Log *log) {
  if (!log)
    return;

  std::string str = GetString();

  // We want to remove the last '\n' because log->PutCString will add
  // one for us.

  if (!str.empty() && str.back() == '\n')
    str.pop_back();

  log->PutString(str);
}

llvm::Error DiagnosticManager::GetAsError(lldb::ExpressionResults result,
                                          llvm::Twine message) const {
  std::vector<DiagnosticDetail> details;
  for (const auto &diag : m_diagnostics)
    details.push_back(diag->GetDetail());
  return llvm::make_error<ExpressionError>(result, message.str(), details);
}

void DiagnosticManager::AddDiagnostic(llvm::StringRef message,
                                      lldb::Severity severity,
                                      DiagnosticOrigin origin,
                                      uint32_t compiler_id) {
  m_diagnostics.emplace_back(std::make_unique<Diagnostic>(
      origin, compiler_id,
      DiagnosticDetail{{}, severity, message.str(), message.str()}));
}

size_t DiagnosticManager::Printf(lldb::Severity severity, const char *format,
                                 ...) {
  StreamString ss;

  va_list args;
  va_start(args, format);
  size_t result = ss.PrintfVarArg(format, args);
  va_end(args);

  AddDiagnostic(ss.GetString(), severity, eDiagnosticOriginLLDB);

  return result;
}

void DiagnosticManager::PutString(lldb::Severity severity,
                                  llvm::StringRef str) {
  if (str.empty())
    return;
  AddDiagnostic(str, severity, eDiagnosticOriginLLDB);
}

void Diagnostic::AppendMessage(llvm::StringRef message,
                               bool precede_with_newline) {
  if (precede_with_newline) {
    m_detail.message.push_back('\n');
    m_detail.rendered.push_back('\n');
  }
  m_detail.message += message;
  m_detail.rendered += message;
}
