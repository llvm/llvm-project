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
#include "lldb/Host/ThreadLauncher.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/OptionArgParser.h"
#include "lldb/Interpreter/OptionGroupBoolean.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Target/RegisterContext.h"

#include "lldb/Utility/Log.h"
#include "lldb/Utility/Logging.h"
#include "lldb/Utility/State.h"

#include <mutex>

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

static constexpr lldb::ScriptLanguage g_supported_script_languages[] = {
    ScriptLanguage::eScriptLanguagePython,
};

bool ScriptedProcess::IsScriptLanguageSupported(lldb::ScriptLanguage language) {
  llvm::ArrayRef<lldb::ScriptLanguage> supported_languages =
      llvm::makeArrayRef(g_supported_script_languages);

  return llvm::is_contained(supported_languages, language);
}

void ScriptedProcess::CheckInterpreterAndScriptObject() const {
  lldbassert(m_interpreter && "Invalid Script Interpreter.");
  lldbassert(m_script_object_sp && "Invalid Script Object.");
}

lldb::ProcessSP ScriptedProcess::CreateInstance(lldb::TargetSP target_sp,
                                                lldb::ListenerSP listener_sp,
                                                const FileSpec *file,
                                                bool can_connect) {
  if (!target_sp ||
      !IsScriptLanguageSupported(target_sp->GetDebugger().GetScriptLanguage()))
    return nullptr;

  Status error;
  ScriptedProcess::ScriptedProcessInfo scripted_process_info(
      target_sp->GetProcessLaunchInfo());

  auto process_sp = std::make_shared<ScriptedProcess>(
      target_sp, listener_sp, scripted_process_info, error);

  if (error.Fail() || !process_sp || !process_sp->m_script_object_sp ||
      !process_sp->m_script_object_sp->IsValid()) {
    LLDB_LOGF(GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS), "%s",
              error.AsCString());
    return nullptr;
  }

  return process_sp;
}

bool ScriptedProcess::CanDebug(lldb::TargetSP target_sp,
                               bool plugin_specified_by_name) {
  return true;
}

ScriptedProcess::ScriptedProcess(
    lldb::TargetSP target_sp, lldb::ListenerSP listener_sp,
    const ScriptedProcess::ScriptedProcessInfo &scripted_process_info,
    Status &error)
    : Process(target_sp, listener_sp),
      m_scripted_process_info(scripted_process_info) {

  if (!target_sp) {
    error.SetErrorStringWithFormat("ScriptedProcess::%s () - ERROR: %s",
                                   __FUNCTION__, "Invalid target");
    return;
  }

  m_interpreter = target_sp->GetDebugger().GetScriptInterpreter();

  if (!m_interpreter) {
    error.SetErrorStringWithFormat("ScriptedProcess::%s () - ERROR: %s",
                                   __FUNCTION__,
                                   "Debugger has no Script Interpreter");
    return;
  }

  StructuredData::ObjectSP object_sp = GetInterface().CreatePluginObject(
      m_scripted_process_info.GetClassName().c_str(), target_sp,
      m_scripted_process_info.GetDictionarySP());

  if (!object_sp || !object_sp->IsValid()) {
    error.SetErrorStringWithFormat("ScriptedProcess::%s () - ERROR: %s",
                                   __FUNCTION__,
                                   "Failed to create valid script object");
    return;
  }

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
  CheckInterpreterAndScriptObject();

  /* FIXME: This doesn't reflect how lldb actually launches a process.
           In reality, it attaches to debugserver, then resume the process. */
  Status error = GetInterface().Launch();
  SetPrivateState(eStateRunning);

  if (error.Fail())
    return error;

  // TODO: Fetch next state from stopped event queue then send stop event
  //  const StateType state = SetThreadStopInfo(response);
  //  if (state != eStateInvalid) {
  //    SetPrivateState(state);

  SetPrivateState(eStateStopped);

  UpdateThreadListIfNeeded();
  GetThreadList();

  return {};
}

void ScriptedProcess::DidLaunch() {
  CheckInterpreterAndScriptObject();
  m_pid = GetInterface().GetProcessID();
}

Status ScriptedProcess::DoResume() {
  CheckInterpreterAndScriptObject();

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));
  // FIXME: Fetch data from thread.
  const StateType thread_resume_state = eStateRunning;
  LLDB_LOGF(log, "ScriptedProcess::%s thread_resume_state = %s", __FUNCTION__,
            StateAsCString(thread_resume_state));

  bool resume = (thread_resume_state == eStateRunning);
  assert(thread_resume_state == eStateRunning && "invalid thread resume state");

  Status error;
  if (resume) {
    LLDB_LOGF(log, "ScriptedProcess::%s sending resume", __FUNCTION__);

    SetPrivateState(eStateRunning);
    SetPrivateState(eStateStopped);
    error = GetInterface().Resume();
  }

  return error;
}

Status ScriptedProcess::DoStop() {
  CheckInterpreterAndScriptObject();

  Log *log(lldb_private::GetLogIfAllCategoriesSet(LIBLLDB_LOG_PROCESS));

  if (GetInterface().ShouldStop()) {
    SetPrivateState(eStateStopped);
    LLDB_LOGF(log, "ScriptedProcess::%s Immediate stop", __FUNCTION__);
    return {};
  }

  LLDB_LOGF(log, "ScriptedProcess::%s Delayed stop", __FUNCTION__);
  return GetInterface().Stop();
}

Status ScriptedProcess::DoDestroy() { return Status(); }

bool ScriptedProcess::IsAlive() {
  if (m_interpreter && m_script_object_sp)
    return GetInterface().IsAlive();
  return false;
}

size_t ScriptedProcess::DoReadMemory(lldb::addr_t addr, void *buf, size_t size,
                                     Status &error) {

  auto error_with_message = [&error](llvm::StringRef message) {
    error.SetErrorString(message);
    return 0;
  };

  if (!m_interpreter)
    return error_with_message("No interpreter.");

  lldb::DataExtractorSP data_extractor_sp =
      GetInterface().ReadMemoryAtAddress(addr, size, error);

  if (!data_extractor_sp || error.Fail())
    return 0;

  offset_t bytes_copied = data_extractor_sp->CopyByteOrderedData(
      0, data_extractor_sp->GetByteSize(), buf, size, GetByteOrder());

  if (!bytes_copied || bytes_copied == LLDB_INVALID_OFFSET)
    return error_with_message("Failed to copy read memory to buffer.");

  return size;
}

ArchSpec ScriptedProcess::GetArchitecture() {
  return GetTarget().GetArchitecture();
}

Status ScriptedProcess::GetMemoryRegionInfo(lldb::addr_t load_addr,
                                            MemoryRegionInfo &region) {
  // TODO: Implement
  return Status();
}

Status ScriptedProcess::GetMemoryRegions(MemoryRegionInfos &region_list) {
  CheckInterpreterAndScriptObject();

  lldb::addr_t address = 0;
  lldb::MemoryRegionInfoSP mem_region_sp = nullptr;

  while ((mem_region_sp =
              GetInterface().GetMemoryRegionContainingAddress(address))) {
    auto range = mem_region_sp->GetRange();
    address += range.GetRangeBase() + range.GetByteSize();
    region_list.push_back(*mem_region_sp.get());
  }

  return {};
}

void ScriptedProcess::Clear() { Process::m_thread_list.Clear(); }

bool ScriptedProcess::DoUpdateThreadList(ThreadList &old_thread_list,
                                         ThreadList &new_thread_list) {
  // TODO: Implement
  // This is supposed to get the current set of threads, if any of them are in
  // old_thread_list then they get copied to new_thread_list, and then any
  // actually new threads will get added to new_thread_list.
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
