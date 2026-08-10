//===-- SetDataBreakpointsRequestHandler.cpp ------------------------------===//
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
#include "Watchpoint.h"
#include <set>

namespace lldb_dap {

/// Replaces all existing data breakpoints with new data breakpoints.
/// To clear all data breakpoints, specify an empty array.
/// When a data breakpoint is hit, a stopped event (with reason data breakpoint)
/// is generated. Clients should only call this request if the corresponding
/// capability supportsDataBreakpoints is true.
llvm::Expected<protocol::SetDataBreakpointsResponseBody>
SetDataBreakpointsRequestHandler::Run(
    const protocol::SetDataBreakpointsArguments &args) const {
  std::vector<protocol::Breakpoint> response_breakpoints;

  for (lldb::watch_id_t watch_id : dap.data_breakpoints)
    dap.target.DeleteWatchpoint(watch_id);
  dap.data_breakpoints.clear();

  std::vector<Watchpoint> watchpoints;
  for (const auto &bp : args.breakpoints)
    watchpoints.emplace_back(dap, bp);

  // If two watchpoints start at the same address, the latter overwrite the
  // former. So, we only enable those at first-seen addresses when iterating
  // backward.
  std::set<lldb::addr_t> addresses;
  for (auto iter = watchpoints.rbegin(); iter != watchpoints.rend(); ++iter) {
    if (addresses.count(iter->GetAddress()) == 0) {
      iter->SetWatchpoint();
      addresses.insert(iter->GetAddress());
      if (lldb::watch_id_t watch_id = iter->GetID();
          watch_id != LLDB_INVALID_WATCH_ID)
        dap.data_breakpoints.push_back(watch_id);
    }
  }
  for (auto wp : watchpoints)
    response_breakpoints.push_back(wp.ToProtocolBreakpoint());

  return protocol::SetDataBreakpointsResponseBody{
      std::move(response_breakpoints)};
}

} // namespace lldb_dap
