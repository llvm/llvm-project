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
#include "llvm/ADT/DenseSet.h"
#include <algorithm>

namespace lldb_dap {

/// Replaces all existing data breakpoints with new data breakpoints.
/// To clear all data breakpoints, specify an empty array.
/// When a data breakpoint is hit, a stopped event (with reason data breakpoint)
/// is generated. Clients should only call this request if the corresponding
/// capability supportsDataBreakpoints is true.
llvm::Expected<protocol::SetDataBreakpointsResponseBody>
SetDataBreakpointsRequestHandler::Run(
    const protocol::SetDataBreakpointsArguments &args) const {
  std::vector<Watchpoint> watchpoints;
  watchpoints.reserve(args.breakpoints.size());
  for (const auto &bp : args.breakpoints)
    watchpoints.emplace_back(dap, bp);

  llvm::DenseSet<lldb::addr_t> outdated(
      llvm::from_range, llvm::make_first_range(dap.data_breakpoints));

  std::vector<protocol::Breakpoint> response_breakpoints;
  response_breakpoints.reserve(watchpoints.size());
  // If two watchpoints start at the same address, the latter overwrite the
  // former. So, we only enable those at first-seen addresses when iterating
  // backward.
  llvm::DenseSet<lldb::addr_t> addresses;
  for (auto it = watchpoints.rbegin(); it != watchpoints.rend(); ++it) {
    const lldb::addr_t addr = it->GetAddress();
    if (addresses.contains(addr)) {
      response_breakpoints.push_back(it->ToProtocolBreakpoint());
      continue;
    }
    addresses.insert(addr);
    outdated.erase(addr);

    auto existing = dap.data_breakpoints.find(addr);
    if (existing == dap.data_breakpoints.end()) {
      // Set the new one.
      it->SetWatchpoint();
      dap.data_breakpoints.try_emplace(addr, *it);
      response_breakpoints.push_back(it->ToProtocolBreakpoint());
    } else if (existing->second.HasSameSizeAndType(*it)) {
      // Update existing.
      existing->second.UpdateBreakpoint(*it);
      response_breakpoints.push_back(existing->second.ToProtocolBreakpoint());
    } else {
      // Delete existing and set the new one.
      if (lldb::watch_id_t watch_id = existing->second.GetID();
          watch_id != LLDB_INVALID_WATCH_ID)
        dap.target.DeleteWatchpoint(watch_id);
      dap.data_breakpoints.erase(existing);
      it->SetWatchpoint();
      dap.data_breakpoints.try_emplace(addr, *it);
      response_breakpoints.push_back(it->ToProtocolBreakpoint());
    }
  }

  for (lldb::addr_t addr : outdated) {
    auto it = dap.data_breakpoints.find(addr);
    if (it == dap.data_breakpoints.end())
      continue;
    if (lldb::watch_id_t watch_id = it->second.GetID();
        watch_id != LLDB_INVALID_WATCH_ID)
      dap.target.DeleteWatchpoint(watch_id);
    dap.data_breakpoints.erase(it);
  }

  std::reverse(response_breakpoints.begin(), response_breakpoints.end());
  return protocol::SetDataBreakpointsResponseBody{
      std::move(response_breakpoints)};
}

} // namespace lldb_dap
