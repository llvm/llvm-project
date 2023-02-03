//===-- ScriptedProcess.h ------------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SCRIPTED_PROCESS_H
#define LLDB_SOURCE_PLUGINS_SCRIPTED_PROCESS_H

#include "lldb/Interpreter/ScriptedMetadata.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Status.h"

#include "ScriptedThread.h"

#include <mutex>

namespace lldb_private {
class ScriptedProcess : public Process {
public:
  static lldb::ProcessSP CreateInstance(lldb::TargetSP target_sp,
                                        lldb::ListenerSP listener_sp,
                                        const FileSpec *crash_file_path,
                                        bool can_connect);

  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "ScriptedProcess"; }

  static llvm::StringRef GetPluginDescriptionStatic();

  ~ScriptedProcess() override;

  bool CanDebug(lldb::TargetSP target_sp,
                bool plugin_specified_by_name) override;

  DynamicLoader *GetDynamicLoader() override { return nullptr; }

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  Status DoLoadCore() override;

  Status DoLaunch(Module *exe_module, ProcessLaunchInfo &launch_info) override;

  void DidLaunch() override;

  Status DoResume() override;

  Status DoDestroy() override;

  void RefreshStateAfterStop() override;

  bool IsAlive() override;

  size_t DoReadMemory(lldb::addr_t addr, void *buf, size_t size,
                      Status &error) override;

  ArchSpec GetArchitecture();

  Status
  GetMemoryRegions(lldb_private::MemoryRegionInfos &region_list) override;

  bool GetProcessInfo(ProcessInstanceInfo &info) override;

  lldb_private::StructuredData::ObjectSP
  GetLoadedDynamicLibrariesInfos() override;

  lldb_private::StructuredData::DictionarySP GetMetadata() override;

  void UpdateQueueListIfNeeded() override;

  void *GetImplementation() override;

protected:
  ScriptedProcess(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp,
                  const ScriptedMetadata &scripted_metadata, Status &error);

  Status DoStop();

  void Clear();

  bool DoUpdateThreadList(ThreadList &old_thread_list,
                          ThreadList &new_thread_list) override;

  Status DoGetMemoryRegionInfo(lldb::addr_t load_addr,
                               MemoryRegionInfo &range_info) override;

private:
  friend class ScriptedThread;

  void CheckInterpreterAndScriptObject() const;
  ScriptedProcessInterface &GetInterface() const;
  static bool IsScriptLanguageSupported(lldb::ScriptLanguage language);

  // Member variables.
  const ScriptedMetadata m_scripted_metadata;
  lldb_private::ScriptInterpreter *m_interpreter = nullptr;
  lldb_private::StructuredData::ObjectSP m_script_object_sp = nullptr;
  //@}
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SCRIPTED_PROCESS_H
