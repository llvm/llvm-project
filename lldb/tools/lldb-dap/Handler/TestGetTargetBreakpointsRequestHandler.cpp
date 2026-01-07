//===-- TestGetTargetBreakpointsRequestHandler.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "EventHelper.h"
#include "Protocol/ProtocolRequests.h"
#include "RequestHandler.h"

using namespace lldb_dap;
using namespace lldb_dap::protocol;

/// A request used in testing to get the details on all breakpoints that are
/// currently set in the target. This helps us to test "setBreakpoints" and
/// "setFunctionBreakpoints" requests to verify we have the correct set of
/// breakpoints currently set in LLDB.
llvm::Expected<TestGetTargetBreakpointsResponseBody>
TestGetTargetBreakpointsRequestHandler::Run(
    const TestGetTargetBreakpointsArguments &args) const {
  std::vector<protocol::Breakpoint> breakpoints;
  for (uint32_t i = 0; dap.target.GetBreakpointAtIndex(i).IsValid(); ++i) {
    auto bp = Breakpoint(dap, dap.target.GetBreakpointAtIndex(i));
    breakpoints.push_back(bp.ToProtocolBreakpoint());
  }
  return TestGetTargetBreakpointsResponseBody{std::move(breakpoints)};
}
