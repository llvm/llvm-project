//===-- DAPError.h --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_DAPERROR_H
#define LLDB_TOOLS_LLDB_DAP_DAPERROR_H

#include "llvm/Support/Error.h"
#include <optional>
#include <string>
#include <system_error>

namespace lldb_dap {

/// An error that is reported as a DAP Error Message, which may be presented to
/// the user.
class DAPError : public llvm::ErrorInfo<DAPError> {
public:
  static char ID;

  explicit DAPError(std::string message);

  DAPError(std::string message, std::error_code EC);

  DAPError(std::string message, bool show_user);

  DAPError(std::string message, std::error_code EC, bool show_user);

  DAPError(std::string message, std::error_code EC, bool show_user,
           std::string url, std::string url_label);

  void log(llvm::raw_ostream &OS) const override;
  std::error_code convertToErrorCode() const override;

  const std::string &getMessage() const { return m_message; }
  bool getShowUser() const { return m_show_user; }
  const std::string &getURL() const { return m_url; }
  const std::string &getURLLabel() const { return m_url_label; }

private:
  std::string m_message;
  std::error_code m_ec;
  bool m_show_user;
  std::string m_url;
  std::string m_url_label;
};

/// An error that indicates the current request handler cannot execute because
/// the process is not stopped.
class NotStoppedError : public llvm::ErrorInfo<NotStoppedError> {
public:
  static char ID;
  void log(llvm::raw_ostream &OS) const override;
  std::error_code convertToErrorCode() const override;
};

} // namespace lldb_dap

#endif // LLDB_TOOLS_LLDB_DAP_DAPERROR_H
