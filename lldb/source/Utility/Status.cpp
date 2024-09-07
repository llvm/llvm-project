//===-- Status.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/Status.h"

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

Status::Status() {}

Status::Status(ValueType err, ErrorType type, std::string msg)
    : m_code(err), m_type(type), m_string(std::move(msg)) {}

// This logic is confusing because c++ calls the traditional (posix) errno codes
// "generic errors", while we use the term "generic" to mean completely
// arbitrary (text-based) errors.
Status::Status(std::error_code EC)
    : m_code(EC.value()),
      m_type(EC.category() == std::generic_category() ? eErrorTypePOSIX
                                                      : eErrorTypeGeneric),
      m_string(EC.message()) {}

Status::Status(std::string err_str)
    : m_code(LLDB_GENERIC_ERROR), m_type(eErrorTypeGeneric),
      m_string(std::move(err_str)) {}

Status::Status(llvm::Error error) {
  if (!error) {
    Clear();
    return;
  }

  // if the error happens to be a errno error, preserve the error code
  error = llvm::handleErrors(
      std::move(error), [&](std::unique_ptr<llvm::ECError> e) -> llvm::Error {
        std::error_code ec = e->convertToErrorCode();
        if (ec.category() == std::generic_category()) {
          m_code = ec.value();
          m_type = ErrorType::eErrorTypePOSIX;
          return llvm::Error::success();
        }
        return llvm::Error(std::move(e));
      });

  // Otherwise, just preserve the message
  if (error) {
    m_code = LLDB_GENERIC_ERROR;
    m_type = eErrorTypeGeneric;
    m_string = llvm::toString(std::move(error));
  }
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

Status Status::FromError(llvm::Error error) { return Status(std::move(error)); }

llvm::Error Status::ToError() const {
  if (Success())
    return llvm::Error::success();
  if (m_type == ErrorType::eErrorTypePOSIX)
    return llvm::errorCodeToError(
        std::error_code(m_code, std::generic_category()));
  return llvm::createStringError(AsCString());
}

Status::~Status() = default;

const Status &Status::operator=(Status &&other) {
  m_code = other.m_code;
  m_type = other.m_type;
  m_string = std::move(other.m_string);
  return *this;
}

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

// Get the error value as a NULL C string. The error string will be fetched and
// cached on demand. The cached error string value will remain until the error
// value is changed or cleared.
const char *Status::AsCString(const char *default_error_str) const {
  if (Success())
    return nullptr;

  if (m_string.empty()) {
    switch (m_type) {
    case eErrorTypeMachKernel:
#if defined(__APPLE__)
      if (const char *s = ::mach_error_string(m_code))
        m_string.assign(s);
#endif
      break;

    case eErrorTypePOSIX:
      m_string = llvm::sys::StrError(m_code);
      break;

    case eErrorTypeWin32:
#if defined(_WIN32)
      m_string = RetrieveWin32ErrorString(m_code);
#endif
      break;

    default:
      break;
    }
  }
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
  m_code = 0;
  m_type = eErrorTypeInvalid;
  m_string.clear();
}

// Access the error value.
Status::ValueType Status::GetError() const { return m_code; }

// Access the error type.
ErrorType Status::GetType() const { return m_type; }

// Returns true if this object contains a value that describes an error or
// otherwise non-success result.
bool Status::Fail() const { return m_code != 0; }

Status Status::FromErrno() {
  // Update the error value to be "errno" and update the type to be "POSIX".
  return Status(errno, eErrorTypePOSIX);
}

// Returns true if the error code in this object is considered a successful
// return value.
bool Status::Success() const { return m_code == 0; }

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
