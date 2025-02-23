//===-- Request.h ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_HANDLER_HANDLER_H
#define LLDB_TOOLS_LLDB_DAP_HANDLER_HANDLER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

namespace lldb_dap {
struct DAP;

class RequestHandler {
public:
  RequestHandler(DAP &dap) : dap(dap) {}

  /// RequestHandler are not copyable.
  /// @{
  RequestHandler(const RequestHandler &) = delete;
  RequestHandler &operator=(const RequestHandler &) = delete;
  /// @}

  virtual ~RequestHandler() = default;

  virtual void operator()(const llvm::json::Object &request) = 0;

  /// Helpers used by multiple request handlers.
  /// FIXME: Move these into the DAP class?
  /// @{
  void SetSourceMapFromArguments(const llvm::json::Object &arguments);
  /// @}

protected:
  DAP &dap;
};

class AttachRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "attach"; }
  void operator()(const llvm::json::Object &request) override;
};

class BreakpointLocationsRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "breakpointLocations"; }
  void operator()(const llvm::json::Object &request) override;
};

class CompletionsRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "completions"; }
  void operator()(const llvm::json::Object &request) override;
};

class ContinueRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "continue"; }
  void operator()(const llvm::json::Object &request) override;
};

class ConfigurationDoneRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "configurationDone"; }
  void operator()(const llvm::json::Object &request) override;
};

class DisconnectRequestHandler : public RequestHandler {
public:
  using RequestHandler::RequestHandler;
  static llvm::StringLiteral getCommand() { return "disconnect"; }
  void operator()(const llvm::json::Object &request) override;
};

} // namespace lldb_dap

#endif
