//===-- SetBreakpointsRequestHandler.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "EventHelper.h"
#include "JSONUtils.h"
#include "Protocol/ProtocolRequests.h"
#include "RequestHandler.h"
#include <vector>

namespace lldb_dap {

/// Sets multiple breakpoints for a single source and clears all previous
/// breakpoints in that source. To clear all breakpoint for a source, specify an
/// empty array. When a breakpoint is hit, a `stopped` event (with reason
/// `breakpoint`) is generated.
llvm::Expected<protocol::SetBreakpointsResponseBody>
SetBreakpointsRequestHandler::Run(
    const protocol::SetBreakpointsArguments &args) const {
  const auto &source = args.source;
  const auto path = source.path.value_or("");
  std::vector<protocol::Breakpoint> response_breakpoints;

  // Decode the source breakpoint infos for this "setBreakpoints" request
  SourceBreakpointMap request_bps;
  // "breakpoints" may be unset, in which case we treat it the same as being set
  // to an empty array.
  if (args.breakpoints) {
    for (const auto &bp : *args.breakpoints) {
      SourceBreakpoint src_bp(dap, bp);
      std::pair<uint32_t, uint32_t> bp_pos(src_bp.GetLine(),
                                           src_bp.GetColumn());
      request_bps.try_emplace(bp_pos, src_bp);
      const auto [iv, inserted] =
          dap.source_breakpoints[path].try_emplace(bp_pos, src_bp);
      // We check if this breakpoint already exists to update it
      if (inserted)
        iv->getSecond().SetBreakpoint(path.data());
      else
        iv->getSecond().UpdateBreakpoint(src_bp);

      protocol::Breakpoint response_bp = iv->getSecond().ToProtocolBreakpoint();

      // Use the path from the request if it is set
      if (!path.empty())
        response_bp.source = CreateSource(path);

      if (!response_bp.line)
        response_bp.line = src_bp.GetLine();
      if (!response_bp.column)
        response_bp.column = src_bp.GetColumn();
      response_breakpoints.push_back(response_bp);
    }
  }

  // Delete any breakpoints in this source file that aren't in the
  // request_bps set. There is no call to remove breakpoints other than
  // calling this function with a smaller or empty "breakpoints" list.
  auto old_src_bp_pos = dap.source_breakpoints.find(path);
  if (old_src_bp_pos != dap.source_breakpoints.end()) {
    for (auto &old_bp : old_src_bp_pos->second) {
      auto request_pos = request_bps.find(old_bp.first);
      if (request_pos == request_bps.end()) {
        // This breakpoint no longer exists in this source file, delete it
        dap.target.BreakpointDelete(old_bp.second.GetID());
        old_src_bp_pos->second.erase(old_bp.first);
      }
    }
  }

  return protocol::SetBreakpointsResponseBody{std::move(response_breakpoints)};
}

} // namespace lldb_dap
