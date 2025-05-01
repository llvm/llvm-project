//===-- ResponseHandler.h -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_HANDLER_RESPONSEHANDLER_H
#define LLDB_TOOLS_LLDB_DAP_HANDLER_RESPONSEHANDLER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include <cstdint>

namespace lldb_dap {
struct DAP;

/// Handler for responses to reverse requests.
class ResponseHandler {
public:
  ResponseHandler(llvm::StringRef command, int64_t id)
      : m_command(command), m_id(id) {}

  /// ResponseHandlers are not copyable.
  /// @{
  ResponseHandler(const ResponseHandler &) = delete;
  ResponseHandler &operator=(const ResponseHandler &) = delete;
  /// @}

  virtual ~ResponseHandler() = default;

  virtual void operator()(llvm::Expected<llvm::json::Value> value) const = 0;

protected:
  llvm::StringRef m_command;
  int64_t m_id;
};

/// Response handler used for unknown responses.
class UnknownResponseHandler : public ResponseHandler {
public:
  using ResponseHandler::ResponseHandler;
  void operator()(llvm::Expected<llvm::json::Value> value) const override;
};

/// Response handler which logs to stderr in case of a failure.
class LogFailureResponseHandler : public ResponseHandler {
public:
  using ResponseHandler::ResponseHandler;
  void operator()(llvm::Expected<llvm::json::Value> value) const override;
};

} // namespace lldb_dap

#endif
