//===-- MCPError.h --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol.h"
#include "llvm/Support/Error.h"
#include <string>

namespace lldb_private::mcp {

class MCPError : public llvm::ErrorInfo<MCPError> {
public:
  static char ID;

  MCPError(std::string message, int64_t error_code);

  void log(llvm::raw_ostream &OS) const override;
  std::error_code convertToErrorCode() const override;

  const std::string &getMessage() const { return m_message; }

  protocol::Error toProtcolError() const;

private:
  std::string m_message;
  int64_t m_error_code;
};

} // namespace lldb_private::mcp
