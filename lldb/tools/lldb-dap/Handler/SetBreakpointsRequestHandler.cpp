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
#include "LLDBUtils.h"
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
  std::vector<protocol::Breakpoint> response_breakpoints;
  if (source.sourceReference)
    response_breakpoints = SetAssemblyBreakpoints(source, args.breakpoints);
  else if (source.path)
    response_breakpoints = SetSourceBreakpoints(source, args.breakpoints);

  return protocol::SetBreakpointsResponseBody{std::move(response_breakpoints)};
}

std::vector<protocol::Breakpoint>
SetBreakpointsRequestHandler::SetSourceBreakpoints(
    const protocol::Source &source,
    const std::optional<std::vector<protocol::SourceBreakpoint>> &breakpoints)
    const {
  std::vector<protocol::Breakpoint> response_breakpoints;
  std::string path = source.path.value_or("");

  // Decode the source breakpoint infos for this "setBreakpoints" request
  SourceBreakpointMap request_bps;
  // "breakpoints" may be unset, in which case we treat it the same as being set
  // to an empty array.
  if (breakpoints) {
    for (const auto &bp : *breakpoints) {
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

  return response_breakpoints;
}

std::vector<protocol::Breakpoint>
SetBreakpointsRequestHandler::SetAssemblyBreakpoints(
    const protocol::Source &source,
    const std::optional<std::vector<protocol::SourceBreakpoint>> &breakpoints)
    const {
  std::vector<protocol::Breakpoint> response_breakpoints;
  int64_t sourceReference = source.sourceReference.value_or(0);

  lldb::SBProcess process = dap.target.GetProcess();
  lldb::SBThread thread =
      process.GetThreadByIndexID(GetLLDBThreadIndexID(sourceReference));
  lldb::SBFrame frame = thread.GetFrameAtIndex(GetLLDBFrameID(sourceReference));

  if (!frame.IsValid())
    return response_breakpoints;

  lldb::SBSymbol symbol = frame.GetSymbol();
  if (!symbol.IsValid())
    return response_breakpoints; // Not yet supporting breakpoints in assembly
                                 // without a valid symbol

  llvm::DenseMap<uint32_t, SourceBreakpoint> request_bps;
  if (breakpoints) {
    for (const auto &bp : *breakpoints) {
      SourceBreakpoint src_bp(dap, bp);
      request_bps.try_emplace(src_bp.GetLine(), src_bp);
      const auto [iv, inserted] =
          dap.assembly_breakpoints[sourceReference].try_emplace(
              src_bp.GetLine(), src_bp);
      // We check if this breakpoint already exists to update it
      if (inserted)
        iv->getSecond().SetBreakpoint(symbol);
      else
        iv->getSecond().UpdateBreakpoint(src_bp);

      protocol::Breakpoint response_bp = iv->getSecond().ToProtocolBreakpoint();
      response_bp.source = source;
      if (!response_bp.line)
        response_bp.line = src_bp.GetLine();
      if (bp.column)
        response_bp.column = *bp.column;
      response_breakpoints.push_back(response_bp);
    }
  }

  // Delete existing breakpoints for this sourceReference that are not in the
  // request_bps set.
  auto old_src_bp_pos = dap.assembly_breakpoints.find(sourceReference);
  if (old_src_bp_pos != dap.assembly_breakpoints.end()) {
    for (auto &old_bp : old_src_bp_pos->second) {
      auto request_pos = request_bps.find(old_bp.first);
      if (request_pos == request_bps.end()) {
        dap.target.BreakpointDelete(old_bp.second.GetID());
        old_src_bp_pos->second.erase(old_bp.first);
      }
    }
  }

  return response_breakpoints;
}

} // namespace lldb_dap
