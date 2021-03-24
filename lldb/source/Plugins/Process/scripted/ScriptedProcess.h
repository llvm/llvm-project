//===-- ScriptedProcess.h ------------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SCRIPTED_PROCESS_H
#define LLDB_SOURCE_PLUGINS_SCRIPTED_PROCESS_H

#include "lldb/Target/Process.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Status.h"

namespace lldb_private {

class ScriptedProcess : public Process {
protected:
  class LaunchInfo {
  public:
    LaunchInfo(const ProcessLaunchInfo &launch_info) {
      m_class_name = launch_info.GetScriptedProcessClassName();
      m_dictionary_sp = launch_info.GetScriptedProcessDictionarySP();
    }

    std::string GetClassName() const { return m_class_name; }
    StructuredData::DictionarySP GetDictionarySP() const {
      return m_dictionary_sp;
    }

  private:
    std::string m_class_name;
    StructuredData::DictionarySP m_dictionary_sp;
  };

public:
  static lldb::ProcessSP CreateInstance(lldb::TargetSP target_sp,
                                        lldb::ListenerSP listener_sp,
                                        const FileSpec *crash_file_path,
                                        bool can_connect);

  static void Initialize();

  static void Terminate();

  static ConstString GetPluginNameStatic();

  static const char *GetPluginDescriptionStatic();

  ScriptedProcess(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp,
                  const ScriptedProcess::LaunchInfo &launch_info);

  ~ScriptedProcess() override;

  bool CanDebug(lldb::TargetSP target_sp,
                bool plugin_specified_by_name) override;

  DynamicLoader *GetDynamicLoader() override { return nullptr; }

  ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override;

  SystemRuntime *GetSystemRuntime() override { return nullptr; }

  Status DoLoadCore() override;

  Status DoLaunch(Module *exe_module, ProcessLaunchInfo &launch_info) override;

  void DidLaunch() override;

  Status DoResume() override;

  Status DoDestroy() override;

  void RefreshStateAfterStop() override{};

  bool IsAlive() override;

  size_t ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                    Status &error) override;

  size_t DoReadMemory(lldb::addr_t addr, void *buf, size_t size,
                      Status &error) override;

  ArchSpec GetArchitecture();

  Status GetMemoryRegionInfo(lldb::addr_t load_addr,
                             MemoryRegionInfo &range_info) override;

  Status
  GetMemoryRegions(lldb_private::MemoryRegionInfos &region_list) override;

  bool GetProcessInfo(ProcessInstanceInfo &info) override;

protected:
  void Clear();

  bool DoUpdateThreadList(ThreadList &old_thread_list,
                          ThreadList &new_thread_list) override;

private:
  ScriptedProcessInterface &GetInterface() const;

  const LaunchInfo m_launch_info;
  lldb_private::ScriptInterpreter *m_interpreter = nullptr;
  lldb_private::StructuredData::ObjectSP m_script_object_sp = nullptr;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SCRIPTED_PROCESS_H
