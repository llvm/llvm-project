//===-- DAPError.h --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Error.h"
#include <string>

namespace lldb_dap {

/// An Error that is reported as a DAP Error Message, which may be presented to
/// the user.
class DAPError : public llvm::ErrorInfo<DAPError> {
public:
  static char ID;

  DAPError(std::string message, bool show_user = false);

  void log(llvm::raw_ostream &OS) const override;
  std::error_code convertToErrorCode() const override;

  const std::string &getMessage() const { return m_message; }
  bool getShowUser() const { return m_show_user; }

private:
  std::string m_message;
  bool m_show_user;
};

} // namespace lldb_dap
