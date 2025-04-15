//===-- DAPError.h --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Error.h"
#include <optional>
#include <string>
#include <system_error>

namespace lldb_dap {

/// An Error that is reported as a DAP Error Message, which may be presented to
/// the user.
class DAPError : public llvm::ErrorInfo<DAPError> {
public:
  static char ID;

  DAPError(std::string message,
           std::error_code EC = llvm::inconvertibleErrorCode(),
           bool show_user = true, std::optional<std::string> url = std::nullopt,
           std::optional<std::string> url_label = std::nullopt);

  void log(llvm::raw_ostream &OS) const override;
  std::error_code convertToErrorCode() const override;

  const std::string &getMessage() const { return m_message; }
  bool getShowUser() const { return m_show_user; }
  const std::optional<std::string> &getURL() const { return m_url; }
  const std::optional<std::string> &getURLLabel() const { return m_url; }

private:
  std::string m_message;
  std::error_code m_ec;
  bool m_show_user;
  std::optional<std::string> m_url;
  std::optional<std::string> m_url_label;
};

} // namespace lldb_dap
