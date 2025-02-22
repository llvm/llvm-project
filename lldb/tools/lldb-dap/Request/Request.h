//===-- Request.h ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_REQUEST_REQUEST_H
#define LLDB_TOOLS_LLDB_DAP_REQUEST_REQUEST_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

namespace lldb_dap {
struct DAP;

class Request {
public:
  Request(DAP &dap) : dap(dap) {}
  virtual ~Request() = default;

  virtual void operator()(const llvm::json::Object &request) = 0;

  static llvm::StringLiteral getName() { return "invalid"; };

  enum LaunchMethod { Launch, Attach, AttachForSuspendedLaunch };

  void SendProcessEvent(LaunchMethod launch_method);
  void SetSourceMapFromArguments(const llvm::json::Object &arguments);

protected:
  DAP &dap;
};

class AttachRequest : public Request {
public:
  using Request::Request;
  static llvm::StringLiteral getName() { return "attach"; }
  void operator()(const llvm::json::Object &request) override;
};

} // namespace lldb_dap

#endif
