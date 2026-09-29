//===-- SetExceptionBreakpointsRequestHandler.cpp -------------------------===//
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
#include <set>

using namespace llvm;
using namespace lldb_dap::protocol;

namespace lldb_dap {

/// The request configures the debugger’s response to thrown exceptions. Each of
/// the `filters`, `filterOptions`, and `exceptionOptions` in the request are
/// independent configurations to a debug adapter indicating a kind of exception
/// to catch. An exception thrown in a program should result in a `stopped`
/// event from the debug adapter (with reason `exception`) if any of the
/// configured filters match.
///
/// Clients should only call this request if the corresponding capability
/// `exceptionBreakpointFilters` returns one or more filters.
Expected<SetExceptionBreakpointsResponseBody>
SetExceptionBreakpointsRequestHandler::Run(
    const SetExceptionBreakpointsArguments &arguments) const {
  // Keep a list of any exception breakpoint filter names that weren't set
  // so we can clear any exception breakpoints if needed.
  std::set<StringRef> unset_filters;
  for (const auto &bp : dap.exception_breakpoints)
    unset_filters.insert(bp.GetFilter());

  SetExceptionBreakpointsResponseBody body;
  for (const auto &filter : arguments.filters) {
    auto *exc_bp = dap.GetExceptionBreakpoint(filter);
    if (!exc_bp)
      continue;

    body.breakpoints.push_back(exc_bp->SetBreakpoint());
    unset_filters.erase(filter);
  }
  for (const auto &filterOptions : arguments.filterOptions) {
    auto *exc_bp = dap.GetExceptionBreakpoint(filterOptions.filterId);
    if (!exc_bp)
      continue;

    body.breakpoints.push_back(exc_bp->SetBreakpoint(filterOptions.condition));
    unset_filters.erase(filterOptions.filterId);
  }

  // Clear any unset filters.
  for (const auto &filter : unset_filters) {
    auto *exc_bp = dap.GetExceptionBreakpoint(filter);
    if (!exc_bp)
      continue;

    exc_bp->ClearBreakpoint();
  }

  return body;
}

} // namespace lldb_dap
