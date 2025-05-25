//===-- Breakpoint.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Breakpoint.h"
#include "DAP.h"
#include "JSONUtils.h"
#include "LLDBUtils.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBBreakpointLocation.h"
#include "lldb/API/SBLineEntry.h"
#include "lldb/API/SBMutex.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/ADT/StringExtras.h"
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>

using namespace lldb_dap;

void Breakpoint::SetCondition() { m_bp.SetCondition(m_condition.c_str()); }

void Breakpoint::SetHitCondition() {
  uint64_t hitCount = 0;
  if (llvm::to_integer(m_hit_condition, hitCount))
    m_bp.SetIgnoreCount(hitCount - 1);
}

protocol::Breakpoint Breakpoint::ToProtocolBreakpoint() {
  protocol::Breakpoint breakpoint;

  // Each breakpoint location is treated as a separate breakpoint for VS code.
  // They don't have the notion of a single breakpoint with multiple locations.
  if (!m_bp.IsValid())
    return breakpoint;

  breakpoint.verified = m_bp.GetNumResolvedLocations() > 0;
  breakpoint.id = m_bp.GetID();
  // VS Code DAP doesn't currently allow one breakpoint to have multiple
  // locations so we just report the first one. If we report all locations
  // then the IDE starts showing the wrong line numbers and locations for
  // other source file and line breakpoints in the same file.

  // Below we search for the first resolved location in a breakpoint and report
  // this as the breakpoint location since it will have a complete location
  // that is at least loaded in the current process.
  lldb::SBBreakpointLocation bp_loc;
  const auto num_locs = m_bp.GetNumLocations();
  for (size_t i = 0; i < num_locs; ++i) {
    bp_loc = m_bp.GetLocationAtIndex(i);
    if (bp_loc.IsResolved())
      break;
  }
  // If not locations are resolved, use the first location.
  if (!bp_loc.IsResolved())
    bp_loc = m_bp.GetLocationAtIndex(0);
  auto bp_addr = bp_loc.GetAddress();

  if (bp_addr.IsValid()) {
    std::string formatted_addr =
        "0x" + llvm::utohexstr(bp_addr.GetLoadAddress(m_bp.GetTarget()));
    breakpoint.instructionReference = formatted_addr;

    lldb::StopDisassemblyType stop_disassembly_display =
        GetStopDisassemblyDisplay(m_dap.debugger);
    auto line_entry = bp_addr.GetLineEntry();
    if (!ShouldDisplayAssemblySource(line_entry, stop_disassembly_display)) {
      const auto line = line_entry.GetLine();
      if (line != LLDB_INVALID_LINE_NUMBER)
        breakpoint.line = line;
      const auto column = line_entry.GetColumn();
      if (column != LLDB_INVALID_COLUMN_NUMBER)
        breakpoint.column = column;
      breakpoint.source = CreateSource(line_entry);
    } else {
      // Assembly breakpoint.
      auto symbol = bp_addr.GetSymbol();
      if (symbol.IsValid()) {
        breakpoint.line =
            m_bp.GetTarget()
                .ReadInstructions(symbol.GetStartAddress(), bp_addr, nullptr)
                .GetSize() +
            1;

        breakpoint.source = CreateAssemblySource(m_dap.target, bp_addr);
      }
    }
  }

  return breakpoint;
}

bool Breakpoint::MatchesName(const char *name) {
  return m_bp.MatchesName(name);
}

void Breakpoint::SetBreakpoint() {
  lldb::SBMutex lock = m_dap.GetAPIMutex();
  std::lock_guard<lldb::SBMutex> guard(lock);

  m_bp.AddName(kDAPBreakpointLabel);
  if (!m_condition.empty())
    SetCondition();
  if (!m_hit_condition.empty())
    SetHitCondition();
}
