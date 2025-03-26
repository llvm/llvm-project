//===-- GoToTargetsRequestHandler.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"

#include "JSONUtils.h"
#include "Protocol/ProtocolRequests.h"
#include "RequestHandler.h"

#include <lldb/API/SBBreakpointLocation.h>
#include <lldb/API/SBListener.h>
#include <lldb/API/SBStream.h>

namespace lldb_dap {

static llvm::SmallVector<lldb::SBLineEntry>
GetLineValidEntry(DAP &dap, const lldb::SBFileSpec &file_spec, uint32_t line) {

  // Create a breakpoint to resolve the line if it is on an empty line.
  lldb::SBBreakpoint goto_bp =
      dap.target.BreakpointCreateByLocation(file_spec, line);
  if (!goto_bp.IsValid())
    return {};

  llvm::SmallVector<lldb::SBLineEntry> entry_locations{};
  const size_t resolved_count = goto_bp.GetNumResolvedLocations();
  for (size_t idx = 0; idx < resolved_count; ++idx) {
    lldb::SBBreakpointLocation location = goto_bp.GetLocationAtIndex(idx);
    if (!location.IsValid())
      continue;

    lldb::SBAddress addr = location.GetAddress();
    if (!addr.IsValid())
      continue;

    lldb::SBLineEntry line_entry = addr.GetLineEntry();
    if (!line_entry.IsValid())
      continue;

    entry_locations.push_back(line_entry);
  }

  // clean up;
  dap.target.BreakpointDelete(goto_bp.GetID());

  return entry_locations;
}

/// GotoTargets request; value of command field is 'gotoTargets'.
llvm::Expected<protocol::GotoTargetsResponseBody>
GotoTargetsRequestHandler::Run(
    const protocol::GotoTargetsArguments &args) const {
  const lldb::SBFileSpec file_spec(args.source.path.value_or("").c_str(), true);
  const uint64_t goto_line = args.line;

  llvm::SmallVector<lldb::SBLineEntry> goto_locations =
      GetLineValidEntry(dap, file_spec, goto_line);

  if (goto_locations.empty())
    return llvm::make_error<DAPError>("Invalid jump location");

  protocol::GotoTargetsResponseBody body{};

  for (lldb::SBLineEntry &line_entry : goto_locations) {
    const uint64_t target_id = dap.gotos.InsertLineEntry(line_entry);
    const uint32_t target_line = line_entry.GetLine();
    protocol::GotoTarget target{};
    target.id = target_id;

    lldb::SBStream stream;
    line_entry.GetDescription(stream);
    target.label = std::string(stream.GetData(), stream.GetSize());
    target.line = target_line;
    body.targets.emplace_back(target);
  }

  return body;
}

} // namespace lldb_dap
