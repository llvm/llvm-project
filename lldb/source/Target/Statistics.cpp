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

static void EmplaceSafeString(llvm::json::Object &obj, llvm::StringRef key,
                              const std::string &str) {
  if (str.empty())
    return;
  if (LLVM_LIKELY(llvm::json::isUTF8(str)))
    obj.try_emplace(key, str);
  else
    obj.try_emplace(key, llvm::json::fixUTF8(str));
}

json::Value StatsSuccessFail::ToJSON() const {
  return json::Object{{"successes", successes}, {"failures", failures}};
}

static double elapsed(const StatsTimepoint &start, const StatsTimepoint &end) {
  StatsDuration elapsed = end.time_since_epoch() - start.time_since_epoch();
  return elapsed.count();
}

void TargetStats::CollectStats(Target &target) {
  m_module_identifiers.clear();
  for (ModuleSP module_sp : target.GetImages().Modules())
    m_module_identifiers.emplace_back((intptr_t)module_sp.get());
}

json::Value ModuleStats::ToJSON() const {
  json::Object module;
  EmplaceSafeString(module, "path", path);
  EmplaceSafeString(module, "uuid", uuid);
  EmplaceSafeString(module, "triple", triple);
  module.try_emplace("identifier", identifier);
  module.try_emplace("symbolTableParseTime", symtab_parse_time);
  module.try_emplace("symbolTableIndexTime", symtab_index_time);
  return module;
}

json::Value TargetStats::ToJSON(Target &target) {
  CollectStats(target);

  json::Array json_module_uuid_array;
  for (auto module_identifier : m_module_identifiers)
    json_module_uuid_array.emplace_back(module_identifier);

  json::Object target_metrics_json{
      {m_expr_eval.name, m_expr_eval.ToJSON()},
      {m_frame_var.name, m_frame_var.ToJSON()},
      {"moduleIdentifiers", std::move(json_module_uuid_array)}};

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

llvm::json::Value DebuggerStats::ReportStatistics(Debugger &debugger,
                                                  Target *target) {
  json::Array json_targets;
  json::Array json_modules;
  double symtab_parse_time = 0.0;
  double symtab_index_time = 0.0;
  if (target) {
    json_targets.emplace_back(target->ReportStatistics());
  } else {
    for (const auto &target : debugger.GetTargetList().Targets())
      json_targets.emplace_back(target->ReportStatistics());
  }
  std::vector<ModuleStats> modules;
  std::lock_guard<std::recursive_mutex> guard(
      Module::GetAllocationModuleCollectionMutex());
  const size_t num_modules = Module::GetNumberAllocatedModules();
  ModuleSP module_sp;
  for (size_t image_idx = 0; image_idx < num_modules; ++image_idx) {
    Module *module = Module::GetAllocatedModuleAtIndex(image_idx);
    ModuleStats module_stat;
    module_stat.identifier = (intptr_t)module;
    module_stat.path = module->GetFileSpec().GetPath();
    module_stat.uuid = module->GetUUID().GetAsString();
    module_stat.triple = module->GetArchitecture().GetTriple().str();
    module_stat.symtab_parse_time = module->GetSymtabParseTime().count();
    module_stat.symtab_index_time = module->GetSymtabIndexTime().count();
    symtab_parse_time += module_stat.symtab_parse_time;
    symtab_index_time += module_stat.symtab_index_time;
    json_modules.emplace_back(module_stat.ToJSON());
  }

  json::Object global_stats{
      {"targets", std::move(json_targets)},
      {"modules", std::move(json_modules)},
      {"totalSymbolTableParseTime", symtab_parse_time},
      {"totalSymbolTableIndexTime", symtab_index_time},
  };
  return std::move(global_stats);
}
