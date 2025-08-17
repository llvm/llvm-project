//===-- TestGetTargetBreakpointsRequestHandler.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "EventHelper.h"
#include "JSONUtils.h"
#include "RequestHandler.h"

namespace lldb_dap {

void TestGetTargetBreakpointsRequestHandler::operator()(
    const llvm::json::Object &request) const {
  llvm::json::Object response;
  FillResponse(request, response);
  llvm::json::Array response_breakpoints;
  for (uint32_t i = 0; dap.target.GetBreakpointAtIndex(i).IsValid(); ++i) {
    auto bp = Breakpoint(dap, dap.target.GetBreakpointAtIndex(i));
    response_breakpoints.push_back(bp.ToProtocolBreakpoint());
  }
  llvm::json::Object body;
  body.try_emplace("breakpoints", std::move(response_breakpoints));
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

} // namespace lldb_dap
