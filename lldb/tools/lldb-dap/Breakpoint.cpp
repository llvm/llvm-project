//===-- Breakpoint.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Breakpoint.h"
#include "DAP.h"
#include "JSONUtils.h"
#include "llvm/ADT/StringExtras.h"

using namespace lldb_dap;

void Breakpoint::SetCondition() { bp.SetCondition(condition.c_str()); }

void Breakpoint::SetHitCondition() {
  uint64_t hitCount = 0;
  if (llvm::to_integer(hitCondition, hitCount))
    bp.SetIgnoreCount(hitCount - 1);
}

// logMessage will be divided into array of LogMessagePart as two kinds:
// 1. raw print text message, and
// 2. interpolated expression for evaluation which is inside matching curly
//    braces.
//
// The function tries to parse logMessage into a list of LogMessageParts
// for easy later access in BreakpointHitCallback.
void Breakpoint::SetLogMessage() {
  logMessageParts.clear();

  // Contains unmatched open curly braces indices.
  std::vector<int> unmatched_curly_braces;

  // Contains all matched curly braces in logMessage.
  // Loop invariant: matched_curly_braces_ranges are sorted by start index in
  // ascending order without any overlap between them.
  std::vector<std::pair<int, int>> matched_curly_braces_ranges;

  lldb::SBError error;
  // Part1 - parse matched_curly_braces_ranges.
  // locating all curly braced expression ranges in logMessage.
  // The algorithm takes care of nested and imbalanced curly braces.
  for (size_t i = 0; i < logMessage.size(); ++i) {
    if (logMessage[i] == '{') {
      unmatched_curly_braces.push_back(i);
    } else if (logMessage[i] == '}') {
      if (unmatched_curly_braces.empty())
        // Nothing to match.
        continue;

      int last_unmatched_index = unmatched_curly_braces.back();
      unmatched_curly_braces.pop_back();

      // Erase any matched ranges included in the new match.
      while (!matched_curly_braces_ranges.empty()) {
        assert(matched_curly_braces_ranges.back().first !=
                   last_unmatched_index &&
               "How can a curley brace be matched twice?");
        if (matched_curly_braces_ranges.back().first < last_unmatched_index)
          break;

        // This is a nested range let's earse it.
        assert((size_t)matched_curly_braces_ranges.back().second < i);
        matched_curly_braces_ranges.pop_back();
      }

      // Assert invariant.
      assert(matched_curly_braces_ranges.empty() ||
             matched_curly_braces_ranges.back().first < last_unmatched_index);
      matched_curly_braces_ranges.emplace_back(last_unmatched_index, i);
    }
  }

  // Part2 - parse raw text and expresions parts.
  // All expression ranges have been parsed in matched_curly_braces_ranges.
  // The code below uses matched_curly_braces_ranges to divide logMessage
  // into raw text parts and expression parts.
  int last_raw_text_start = 0;
  for (const std::pair<int, int> &curly_braces_range :
       matched_curly_braces_ranges) {
    // Raw text before open curly brace.
    assert(curly_braces_range.first >= last_raw_text_start);
    size_t raw_text_len = curly_braces_range.first - last_raw_text_start;
    if (raw_text_len > 0) {
      error = AppendLogMessagePart(
          llvm::StringRef(logMessage.c_str() + last_raw_text_start,
                          raw_text_len),
          /*is_expr=*/false);
      if (error.Fail()) {
        NotifyLogMessageError(error.GetCString());
        return;
      }
    }

    // Expression between curly braces.
    assert(curly_braces_range.second > curly_braces_range.first);
    size_t expr_len = curly_braces_range.second - curly_braces_range.first - 1;
    error = AppendLogMessagePart(
        llvm::StringRef(logMessage.c_str() + curly_braces_range.first + 1,
                        expr_len),
        /*is_expr=*/true);
    if (error.Fail()) {
      NotifyLogMessageError(error.GetCString());
      return;
    }

    last_raw_text_start = curly_braces_range.second + 1;
  }
  // Trailing raw text after close curly brace.
  assert(last_raw_text_start >= 0);
  if (logMessage.size() > (size_t)last_raw_text_start) {
    error = AppendLogMessagePart(
        llvm::StringRef(logMessage.c_str() + last_raw_text_start,
                        logMessage.size() - last_raw_text_start),
        /*is_expr=*/false);
    if (error.Fail()) {
      NotifyLogMessageError(error.GetCString());
      return;
    }
  }

  bp.SetCallback(BreakpointBase::BreakpointHitCallback, this);
}

void Breakpoint::CreateJsonObject(llvm::json::Object &object) {
  // Each breakpoint location is treated as a separate breakpoint for VS code.
  // They don't have the notion of a single breakpoint with multiple locations.
  if (!bp.IsValid())
    return;
  object.try_emplace("verified", bp.GetNumResolvedLocations() > 0);
  object.try_emplace("id", bp.GetID());
  // VS Code DAP doesn't currently allow one breakpoint to have multiple
  // locations so we just report the first one. If we report all locations
  // then the IDE starts showing the wrong line numbers and locations for
  // other source file and line breakpoints in the same file.

  // Below we search for the first resolved location in a breakpoint and report
  // this as the breakpoint location since it will have a complete location
  // that is at least loaded in the current process.
  lldb::SBBreakpointLocation bp_loc;
  const auto num_locs = bp.GetNumLocations();
  for (size_t i = 0; i < num_locs; ++i) {
    bp_loc = bp.GetLocationAtIndex(i);
    if (bp_loc.IsResolved())
      break;
  }
  // If not locations are resolved, use the first location.
  if (!bp_loc.IsResolved())
    bp_loc = bp.GetLocationAtIndex(0);
  auto bp_addr = bp_loc.GetAddress();

  if (bp_addr.IsValid()) {
    std::string formatted_addr =
        "0x" + llvm::utohexstr(bp_addr.GetLoadAddress(g_dap.target));
    object.try_emplace("instructionReference", formatted_addr);
    auto line_entry = bp_addr.GetLineEntry();
    const auto line = line_entry.GetLine();
    if (line != UINT32_MAX)
      object.try_emplace("line", line);
    const auto column = line_entry.GetColumn();
    if (column != 0)
      object.try_emplace("column", column);
    object.try_emplace("source", CreateSource(line_entry));
  }
}

bool Breakpoint::MatchesName(const char *name) { return bp.MatchesName(name); }

void Breakpoint::SetBreakpoint() {
  // See comments in BreakpointBase::GetBreakpointLabel() for details of why
  // we add a label to our breakpoints.
  bp.AddName(GetBreakpointLabel());
  if (!condition.empty())
    SetCondition();
  if (!hitCondition.empty())
    SetHitCondition();
  if (!logMessage.empty())
    SetLogMessage();
}
