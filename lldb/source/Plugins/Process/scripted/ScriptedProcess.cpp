//===-- ScriptedProcess.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ScriptedProcess.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"

#include "lldb/Host/OptionParser.h"

#include "lldb/Interpreter/OptionArgParser.h"
#include "lldb/Interpreter/OptionGroupBoolean.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Target/MemoryRegionInfo.h"

LLDB_PLUGIN_DEFINE(ScriptedProcess)

using namespace lldb;
using namespace lldb_private;

ConstString ScriptedProcess::GetPluginNameStatic() {
  static ConstString g_name("ScriptedProcess");
  return g_name;
}

const char *ScriptedProcess::GetPluginDescriptionStatic() {
  return "Scripted Process plug-in.";
}

lldb::ProcessSP ScriptedProcess::CreateInstance(lldb::TargetSP target_sp,
                                                lldb::ListenerSP listener_sp,
                                                const FileSpec *file,
                                                bool can_connect) {
  ScriptedProcess::LaunchInfo launch_info(target_sp->GetProcessLaunchInfo());

  auto process_sp =
      std::make_shared<ScriptedProcess>(target_sp, listener_sp, launch_info);

  if (!process_sp || !process_sp->m_script_object_sp ||
      !process_sp->m_script_object_sp->IsValid())
    return nullptr;

  return process_sp;
}

bool ScriptedProcess::CanDebug(lldb::TargetSP target_sp,
                               bool plugin_specified_by_name) {
  return true;
}

ScriptedProcess::ScriptedProcess(lldb::TargetSP target_sp,
                                 lldb::ListenerSP listener_sp,
                                 const ScriptedProcess::LaunchInfo &launch_info)
    : Process(target_sp, listener_sp), m_launch_info(launch_info) {
  if (!target_sp)
    return;

  m_interpreter = target_sp->GetDebugger().GetScriptInterpreter();

  if (!m_interpreter)
    return;

  StructuredData::ObjectSP object_sp = GetInterface().CreatePluginObject(
      m_launch_info.GetClassName().c_str(), target_sp,
      m_launch_info.GetDictionarySP());

  if (object_sp && object_sp->IsValid())
    m_script_object_sp = object_sp;
}

ScriptedProcess::~ScriptedProcess() {
  Clear();
  // We need to call finalize on the process before destroying ourselves to
  // make sure all of the broadcaster cleanup goes as planned. If we destruct
  // this class, then Process::~Process() might have problems trying to fully
  // destroy the broadcaster.
  Finalize();
}

void ScriptedProcess::Initialize() {
  static llvm::once_flag g_once_flag;

  llvm::call_once(g_once_flag, []() {
    PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                  GetPluginDescriptionStatic(), CreateInstance);
  });
}

void ScriptedProcess::Terminate() {
  PluginManager::UnregisterPlugin(ScriptedProcess::CreateInstance);
}

ConstString ScriptedProcess::GetPluginName() { return GetPluginNameStatic(); }

uint32_t ScriptedProcess::GetPluginVersion() { return 1; }

Status ScriptedProcess::DoLoadCore() {
  ProcessLaunchInfo launch_info = GetTarget().GetProcessLaunchInfo();

  return DoLaunch(nullptr, launch_info);
}

Status ScriptedProcess::DoLaunch(Module *exe_module,
                                 ProcessLaunchInfo &launch_info) {
  if (!m_interpreter)
    return Status("No interpreter.");

  if (!m_script_object_sp)
    return Status("No python object.");

  Status status = GetInterface().Launch();

  if (status.Success()) {
    SetPrivateState(eStateRunning);
    SetPrivateState(eStateStopped);
  }

  return status;
};

void ScriptedProcess::DidLaunch() {
  if (m_interpreter)
    m_pid = GetInterface().GetProcessID();
}

Status ScriptedProcess::DoResume() {
  if (!m_interpreter)
    return Status("No interpreter.");

  if (!m_script_object_sp)
    return Status("No python object.");

  Status status = GetInterface().Resume();

  if (status.Success()) {
    SetPrivateState(eStateRunning);
    SetPrivateState(eStateStopped);
  }

  return status;
}

Status ScriptedProcess::DoDestroy() { return Status(); }

bool ScriptedProcess::IsAlive() {
  if (!m_interpreter)
    return false;

  return GetInterface().IsAlive();
}

size_t ScriptedProcess::ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                                   Status &error) {
  return DoReadMemory(addr, buf, size, error);
}

size_t ScriptedProcess::DoReadMemory(lldb::addr_t addr, void *buf, size_t size,
                                     Status &error) {

  auto error_with_message = [&error](llvm::StringRef message) {
    error.SetErrorString(message);
    return LLDB_INVALID_ADDRESS;
  };

  if (!m_interpreter)
    return error_with_message("No interpreter.");

  lldb::DataExtractorSP data_extractor_sp =
      GetInterface().ReadMemoryAtAddress(addr, size, error);

  if (!data_extractor_sp || error.Fail())
    return LLDB_INVALID_ADDRESS;

  if (data_extractor_sp->GetByteSize() != size)
    return error_with_message("Failed to read requested memory size.");

  offset_t bytes_copied = data_extractor_sp->CopyByteOrderedData(
      0, size, buf, size, GetByteOrder());

  if (!bytes_copied || bytes_copied == LLDB_INVALID_OFFSET)
    return error_with_message("Failed to copy read memory to buffer.");

  return size;
}

ArchSpec ScriptedProcess::GetArchitecture() {
  return GetTarget().GetArchitecture();
}

Status ScriptedProcess::GetMemoryRegionInfo(lldb::addr_t load_addr,
                                            MemoryRegionInfo &region) {
  return Status();
}

Status ScriptedProcess::GetMemoryRegions(MemoryRegionInfos &region_list) {
  Status error;

  if (!m_interpreter) {
    error.SetErrorString("No interpreter.");
    return error;
  }

  lldb::addr_t address = 0;
  lldb::MemoryRegionInfoSP mem_region_sp = nullptr;

  while ((mem_region_sp =
              GetInterface().GetMemoryRegionContainingAddress(address))) {
    auto range = mem_region_sp->GetRange();
    address += range.GetRangeBase() + range.GetByteSize();
    region_list.push_back(*mem_region_sp.get());
  }

  return error;
}

void ScriptedProcess::Clear() { Process::m_thread_list.Clear(); }

bool ScriptedProcess::DoUpdateThreadList(ThreadList &old_thread_list,
                                         ThreadList &new_thread_list) {
  return new_thread_list.GetSize(false) > 0;
}

bool ScriptedProcess::GetProcessInfo(ProcessInstanceInfo &info) {
  info.Clear();
  info.SetProcessID(GetID());
  info.SetArchitecture(GetArchitecture());
  lldb::ModuleSP module_sp = GetTarget().GetExecutableModule();
  if (module_sp) {
    const bool add_exe_file_as_first_arg = false;
    info.SetExecutableFile(GetTarget().GetExecutableModule()->GetFileSpec(),
                           add_exe_file_as_first_arg);
  }
  return true;
}

ScriptedProcessInterface &ScriptedProcess::GetInterface() const {
  return m_interpreter->GetScriptedProcessInterface();
}
