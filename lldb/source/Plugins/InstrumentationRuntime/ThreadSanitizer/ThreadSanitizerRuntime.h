//===-- ThreadSanitizerRuntime.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadSanitizerRuntime_h_
#define liblldb_ThreadSanitizerRuntime_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/StructuredData.h"
#include "lldb/Target/ABI.h"
#include "lldb/Target/InstrumentationRuntime.h"
#include "lldb/Target/Process.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class ThreadSanitizerRuntime : public lldb_private::InstrumentationRuntime {
public:
  ~ThreadSanitizerRuntime() override;

  static lldb::InstrumentationRuntimeSP
  CreateInstance(const lldb::ProcessSP &process_sp);

  static void Initialize();

  static void Terminate();

  static lldb_private::ConstString GetPluginNameStatic();

  static lldb::InstrumentationRuntimeType GetTypeStatic();

  lldb_private::ConstString GetPluginName() override {
    return GetPluginNameStatic();
  }

  virtual lldb::InstrumentationRuntimeType GetType() { return GetTypeStatic(); }

  uint32_t GetPluginVersion() override { return 1; }

  void ModulesDidLoad(lldb_private::ModuleList &module_list) override;

  bool IsActive() override;

  lldb::ThreadCollectionSP
  GetBacktracesFromExtendedStopInfo(StructuredData::ObjectSP info) override;

private:
  ThreadSanitizerRuntime(const lldb::ProcessSP &process_sp);

  lldb::ProcessSP GetProcessSP() { return m_process_wp.lock(); }

  lldb::ModuleSP GetRuntimeModuleSP() { return m_runtime_module_wp.lock(); }

  void Activate();

  void Deactivate();

  static bool NotifyBreakpointHit(void *baton,
                                  StoppointCallbackContext *context,
                                  lldb::user_id_t break_id,
                                  lldb::user_id_t break_loc_id);

  StructuredData::ObjectSP RetrieveReportData(ExecutionContextRef exe_ctx_ref);

  std::string FormatDescription(StructuredData::ObjectSP report);

  std::string GenerateSummary(StructuredData::ObjectSP report);

  lldb::addr_t GetMainRacyAddress(StructuredData::ObjectSP report);

  std::string GetLocationDescription(StructuredData::ObjectSP report,
                                     lldb::addr_t &global_addr,
                                     std::string &global_name,
                                     std::string &filename, uint32_t &line);

  lldb::addr_t GetFirstNonInternalFramePc(StructuredData::ObjectSP trace);

  bool m_is_active;
  lldb::ModuleWP m_runtime_module_wp;
  lldb::ProcessWP m_process_wp;
  lldb::user_id_t m_breakpoint_id;
};

} // namespace lldb_private

#endif // liblldb_ThreadSanitizerRuntime_h_
