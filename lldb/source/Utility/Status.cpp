//===-- Status.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/Status.h"

#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/VASPrintf.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/FormatProviders.h"

#include <cerrno>
#include <cstdarg>
#include <string>
#include <system_error>

#ifdef __APPLE__
#include <mach/mach.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif
#include <cstdint>

namespace llvm {
class raw_ostream;
}

using namespace lldb;
using namespace lldb_private;

char CloneableError::ID;
char CloneableECError::ID;
char MachKernelError::ID;
char Win32Error::ID;
char ExpressionError::ID;

namespace {
/// A std::error_code category for eErrorTypeGeneric.
class LLDBGenericCategory : public std::error_category {
  const char *name() const override { return "LLDBGenericCategory"; }
  std::string message(int __ev) const override { return "generic LLDB error"; };
};
LLDBGenericCategory &lldb_generic_category() {
  static LLDBGenericCategory g_generic_category;
  return g_generic_category;
}

/// A std::error_code category for eErrorTypeExpression.
class ExpressionCategory : public std::error_category {
  const char *name() const override { return "LLDBExpressionCategory"; }
  std::string message(int __ev) const override {
    return ExpressionResultAsCString(
        static_cast<lldb::ExpressionResults>(__ev));
  };
};
ExpressionCategory &expression_category() {
  static ExpressionCategory g_expression_category;
  return g_expression_category;
}
} // namespace

Status::Status() : m_error(llvm::Error::success()) {}

static llvm::Error ErrorFromEnums(Status::ValueType err, ErrorType type,
                                  std::string msg) {
  switch (type) {
  case eErrorTypeMachKernel:
    return llvm::make_error<MachKernelError>(
        std::error_code(err, std::system_category()));
  case eErrorTypeWin32:
    return llvm::make_error<Win32Error>(
        std::error_code(err, std::system_category()));
  case eErrorTypePOSIX:
    if (msg.empty())
      return llvm::errorCodeToError(
          std::error_code(err, std::generic_category()));
    return llvm::createStringError(
        std::move(msg), std::error_code(err, std::generic_category()));
  default:
    return llvm::createStringError(
        std::move(msg), std::error_code(err, lldb_generic_category()));
  }
}

Status::Status(ValueType err, ErrorType type, std::string msg)
    : m_error(ErrorFromEnums(err, type, msg)) {}

// This logic is confusing because C++ calls the traditional (posix) errno codes
// "generic errors", while we use the term "generic" to mean completely
// arbitrary (text-based) errors.
Status::Status(std::error_code EC)
    : m_error(!EC ? llvm::Error::success() : llvm::errorCodeToError(EC)) {}

Status::Status(std::string err_str)
    : m_error(
          llvm::createStringError(llvm::inconvertibleErrorCode(), err_str)) {}

const Status &Status::operator=(Status &&other) {
  Clear();
  llvm::consumeError(std::move(m_error));
  m_error = std::move(other.m_error);
  return *this;
}

Status Status::FromErrorStringWithFormat(const char *format, ...) {
  std::string string;
  va_list args;
  va_start(args, format);
  if (format != nullptr && format[0]) {
    llvm::SmallString<1024> buf;
    VASprintf(buf, format, args);
    string = std::string(buf.str());
  }
  va_end(args);
  return Status(string);
}

Status Status::FromExpressionError(lldb::ExpressionResults result,
                                   std::string msg) {
  return Status(llvm::make_error<ExpressionError>(
      std::error_code(result, expression_category()), msg));
}

/// Creates a deep copy of all known errors and converts all other
/// errors to a new llvm::StringError.
static llvm::Error CloneError(const llvm::Error &error) {
  llvm::Error result = llvm::Error::success();
  auto clone = [](const llvm::ErrorInfoBase &e) {
    if (e.isA<CloneableError>())
      return llvm::Error(static_cast<const CloneableError &>(e).Clone());
    return llvm::make_error<llvm::StringError>(e.message(),
                                               e.convertToErrorCode(), true);
  };
  visitErrors(error, [&](const llvm::ErrorInfoBase &e) {
    result = joinErrors(std::move(result), clone(e));
  });
  return result;
}

Status Status::FromError(llvm::Error error) { return Status(std::move(error)); }

llvm::Error Status::ToError() const { return CloneError(m_error); }

Status::~Status() { llvm::consumeError(std::move(m_error)); }

#ifdef _WIN32
static std::string RetrieveWin32ErrorString(uint32_t error_code) {
  char *buffer = nullptr;
  std::string message;
  // Retrieve win32 system error.
  // First, attempt to load a en-US message
  if (::FormatMessageA(
          FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
              FORMAT_MESSAGE_MAX_WIDTH_MASK,
          NULL, error_code, MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US),
          (LPSTR)&buffer, 0, NULL)) {
    message.assign(buffer);
    ::LocalFree(buffer);
  }
  // If the previous didn't work, use the default OS language
  else if (::FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                                FORMAT_MESSAGE_FROM_SYSTEM |
                                FORMAT_MESSAGE_MAX_WIDTH_MASK,
                            NULL, error_code, 0, (LPSTR)&buffer, 0, NULL)) {
    message.assign(buffer);
    ::LocalFree(buffer);
  }
  return message;
}
#endif

std::string MachKernelError::message() const {
#if defined(__APPLE__)
  if (const char *s = ::mach_error_string(convertToErrorCode().value()))
    return s;
#endif
  return "MachKernelError";
}

std::string Win32Error::message() const {
#if defined(_WIN32)
  return RetrieveWin32ErrorString(convertToErrorCode().value());
#endif
  return "Win32Error";
}

std::unique_ptr<CloneableError> CloneableECError::Clone() const {
  return std::make_unique<CloneableECError>(convertToErrorCode());
}

std::unique_ptr<CloneableError> MachKernelError::Clone() const {
  return std::make_unique<MachKernelError>(convertToErrorCode());
}

std::unique_ptr<CloneableError> Win32Error::Clone() const {
  return std::make_unique<Win32Error>(convertToErrorCode());
}

std::unique_ptr<CloneableError> ExpressionError::Clone() const {
  return std::make_unique<ExpressionError>(convertToErrorCode(), message());
}

// Get the error value as a NULL C string. The error string will be fetched and
// cached on demand. The cached error string value will remain until the error
// value is changed or cleared.
const char *Status::AsCString(const char *default_error_str) const {
  if (Success())
    return nullptr;

  m_string = llvm::toStringWithoutConsuming(m_error);
  // Backwards compatibility with older implementations of Status.
  if (m_error.isA<llvm::ECError>())
    if (!m_string.empty() && m_string[m_string.size() - 1] == '\n')
      m_string.pop_back();

  if (m_string.empty()) {
    if (default_error_str)
      m_string.assign(default_error_str);
    else
      return nullptr; // User wanted a nullptr string back...
  }
  return m_string.c_str();
}

// Clear the error and any cached error string that it might contain.
void Status::Clear() {
  if (m_error)
    LLDB_LOG_ERRORV(GetLog(LLDBLog::API), std::move(m_error),
                    "dropping error {0}");
  m_error = llvm::Error::success();
}

Status::ValueType Status::GetError() const {
  Status::ValueType result = 0;
  llvm::visitErrors(m_error, [&](const llvm::ErrorInfoBase &error) {
    // Return the first only.
    if (result)
      return;
    std::error_code ec = error.convertToErrorCode();
    result = ec.value();
  });
  return result;
}

// Access the error type.
ErrorType Status::GetType() const {
  ErrorType result = eErrorTypeInvalid;
  llvm::visitErrors(m_error, [&](const llvm::ErrorInfoBase &error) {
    // Return the first only.
    if (result != eErrorTypeInvalid)
      return;
    if (error.isA<MachKernelError>())
      result = eErrorTypeMachKernel;
    else if (error.isA<Win32Error>())
      result = eErrorTypeWin32;
    else if (error.isA<ExpressionError>())
      result = eErrorTypeExpression;
    else if (error.convertToErrorCode().category() == std::generic_category())
      result = eErrorTypePOSIX;
    else if (error.convertToErrorCode().category() == lldb_generic_category() ||
             error.convertToErrorCode() == llvm::inconvertibleErrorCode())
      result = eErrorTypeGeneric;
    else
      result = eErrorTypeInvalid;
  });
  return result;
}

bool Status::Fail() const {
  // Note that this does not clear the checked flag in
  // m_error. Otherwise we'd need to make this thread-safe.
  return m_error.isA<llvm::ErrorInfoBase>();
}

Status Status::FromErrno() {
  std::error_code ec = llvm::errnoAsErrorCode();
  if (ec)
    return Status::FromError(llvm::make_error<CloneableECError>(ec));
  return Status();
}

// Returns true if the error code in this object is considered a successful
// return value.
bool Status::Success() const { return !Fail(); }

void llvm::format_provider<lldb_private::Status>::format(
    const lldb_private::Status &error, llvm::raw_ostream &OS,
    llvm::StringRef Options) {
  llvm::format_provider<llvm::StringRef>::format(error.AsCString(), OS,
                                                 Options);
}

const char *lldb_private::ExpressionResultAsCString(ExpressionResults result) {
  switch (result) {
  case eExpressionCompleted:
    return "eExpressionCompleted";
  case eExpressionDiscarded:
    return "eExpressionDiscarded";
  case eExpressionInterrupted:
    return "eExpressionInterrupted";
  case eExpressionHitBreakpoint:
    return "eExpressionHitBreakpoint";
  case eExpressionSetupError:
    return "eExpressionSetupError";
  case eExpressionParseError:
    return "eExpressionParseError";
  case eExpressionResultUnavailable:
    return "eExpressionResultUnavailable";
  case eExpressionTimedOut:
    return "eExpressionTimedOut";
  case eExpressionStoppedForDebug:
    return "eExpressionStoppedForDebug";
  case eExpressionThreadVanished:
    return "eExpressionThreadVanished";
  }
  return "<unknown>";
}
