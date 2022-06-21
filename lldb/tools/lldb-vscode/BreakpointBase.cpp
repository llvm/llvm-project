//===-- BreakpointBase.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BreakpointBase.h"
#include "VSCode.h"
#include "llvm/ADT/StringExtras.h"

using namespace lldb_vscode;

BreakpointBase::BreakpointBase(const llvm::json::Object &obj)
    : condition(std::string(GetString(obj, "condition"))),
      hitCondition(std::string(GetString(obj, "hitCondition"))),
      logMessage(std::string(GetString(obj, "logMessage"))) {}

void BreakpointBase::SetCondition() { bp.SetCondition(condition.c_str()); }

void BreakpointBase::SetHitCondition() {
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
void BreakpointBase::SetLogMessage() {
  logMessageParts.clear();

  // Contains unmatched open curly braces indices.
  std::vector<int> unmatched_curly_braces;

  // Contains all matched curly braces in logMessage.
  // Loop invariant: matched_curly_braces_ranges are sorted by start index in
  // ascending order without any overlap between them.
  std::vector<std::pair<int, int>> matched_curly_braces_ranges;

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
    if (raw_text_len > 0)
      logMessageParts.emplace_back(
          llvm::StringRef(logMessage.c_str() + last_raw_text_start,
                          raw_text_len),
          /*is_expr=*/false);

    // Expression between curly braces.
    assert(curly_braces_range.second > curly_braces_range.first);
    size_t expr_len = curly_braces_range.second - curly_braces_range.first - 1;
    logMessageParts.emplace_back(
        llvm::StringRef(logMessage.c_str() + curly_braces_range.first + 1,
                        expr_len),
        /*is_expr=*/true);
    last_raw_text_start = curly_braces_range.second + 1;
  }
  // Trailing raw text after close curly brace.
  assert(last_raw_text_start >= 0);
  if (logMessage.size() > (size_t)last_raw_text_start)
    logMessageParts.emplace_back(
        llvm::StringRef(logMessage.c_str() + last_raw_text_start,
                        logMessage.size() - last_raw_text_start),
        /*is_expr=*/false);
  bp.SetCallback(BreakpointBase::BreakpointHitCallback, this);
}

/*static*/
bool BreakpointBase::BreakpointHitCallback(
    void *baton, lldb::SBProcess &process, lldb::SBThread &thread,
    lldb::SBBreakpointLocation &location) {
  if (!baton)
    return true;

  BreakpointBase *bp = (BreakpointBase *)baton;
  lldb::SBFrame frame = thread.GetSelectedFrame();

  std::string output;
  for (const BreakpointBase::LogMessagePart &messagePart :
       bp->logMessageParts) {
    if (messagePart.is_expr) {
      // Try local frame variables first before fall back to expression
      // evaluation
      std::string expr_str = messagePart.text.str();
      const char *expr = expr_str.c_str();
      lldb::SBValue value =
          frame.GetValueForVariablePath(expr, lldb::eDynamicDontRunTarget);
      if (value.GetError().Fail())
        value = frame.EvaluateExpression(expr);
      const char *expr_val = value.GetValue();
      if (expr_val)
        output += expr_val;
    } else {
      output += messagePart.text.str();
    }
  }
  g_vsc.SendOutput(OutputType::Console, output.c_str());

  // Do not stop.
  return false;
}

void BreakpointBase::UpdateBreakpoint(const BreakpointBase &request_bp) {
  if (condition != request_bp.condition) {
    condition = request_bp.condition;
    SetCondition();
  }
  if (hitCondition != request_bp.hitCondition) {
    hitCondition = request_bp.hitCondition;
    SetHitCondition();
  }
  if (logMessage != request_bp.logMessage) {
    logMessage = request_bp.logMessage;
    SetLogMessage();
  }
}

const char *BreakpointBase::GetBreakpointLabel() {
  // Breakpoints in LLDB can have names added to them which are kind of like
  // labels or categories. All breakpoints that are set through the IDE UI get
  // sent through the various VS code DAP set*Breakpoint packets, and these
  // breakpoints will be labeled with this name so if breakpoint update events
  // come in for breakpoints that the IDE doesn't know about, like if a
  // breakpoint is set manually using the debugger console, we won't report any
  // updates on them and confused the IDE. This function gets called by all of
  // the breakpoint classes after they set breakpoints to mark a breakpoint as
  // a UI breakpoint. We can later check a lldb::SBBreakpoint object that comes
  // in via LLDB breakpoint changed events and check the breakpoint by calling
  // "bool lldb::SBBreakpoint::MatchesName(const char *)" to check if a
  // breakpoint in one of the UI breakpoints that we should report changes for.
  return "vscode";
}
