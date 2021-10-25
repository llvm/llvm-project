//===-- Statistics.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Statistics.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

json::Value StatsSuccessFail::ToJSON() const {
  return json::Object{{"successes", successes}, {"failures", failures}};
}

static double elapsed(const StatsTimepoint &start, const StatsTimepoint &end) {
  StatsDuration elapsed = end.time_since_epoch() - start.time_since_epoch();
  return elapsed.count();
}

json::Value TargetStats::ToJSON() {
  json::Object target_metrics_json{{m_expr_eval.name, m_expr_eval.ToJSON()},
                                   {m_frame_var.name, m_frame_var.ToJSON()}};
  if (m_launch_or_attach_time && m_first_private_stop_time) {
    double elapsed_time =
        elapsed(*m_launch_or_attach_time, *m_first_private_stop_time);
    target_metrics_json.try_emplace("launchOrAttachTime", elapsed_time);
  }
  if (m_launch_or_attach_time && m_first_public_stop_time) {
    double elapsed_time =
        elapsed(*m_launch_or_attach_time, *m_first_public_stop_time);
    target_metrics_json.try_emplace("firstStopTime", elapsed_time);
  }
  target_metrics_json.try_emplace("targetCreateTime", m_create_time.count());

  return target_metrics_json;
}

void TargetStats::SetLaunchOrAttachTime() {
  m_launch_or_attach_time = StatsClock::now();
  m_first_private_stop_time = llvm::None;
}

void TargetStats::SetFirstPrivateStopTime() {
  // Launching and attaching has many paths depending on if synchronous mode
  // was used or if we are stopping at the entry point or not. Only set the
  // first stop time if it hasn't already been set.
  if (!m_first_private_stop_time)
    m_first_private_stop_time = StatsClock::now();
}

void TargetStats::SetFirstPublicStopTime() {
  // Launching and attaching has many paths depending on if synchronous mode
  // was used or if we are stopping at the entry point or not. Only set the
  // first stop time if it hasn't already been set.
  if (!m_first_public_stop_time)
    m_first_public_stop_time = StatsClock::now();
}

bool DebuggerStats::g_collecting_stats = false;

llvm::json::Value DebuggerStats::ReportStatistics(Debugger &debugger) {
  json::Array targets;
  for (const auto &target : debugger.GetTargetList().Targets()) {
    targets.emplace_back(target->ReportStatistics());
  }
  json::Object global_stats{
      {"targets", std::move(targets)},
  };
  return std::move(global_stats);
}
