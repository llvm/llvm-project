//===-- ScriptedPlatform.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ScriptedPlatform.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Utility/LLDBLog.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(ScriptedPlatform)

static uint32_t g_initialize_count = 0;

static constexpr lldb::ScriptLanguage g_supported_script_languages[] = {
    ScriptLanguage::eScriptLanguagePython,
};

bool ScriptedPlatform::IsScriptLanguageSupported(
    lldb::ScriptLanguage language) {
  return llvm::is_contained(g_supported_script_languages, language);
}

ScriptedPlatformInterface &ScriptedPlatform::GetInterface() const {
  return *m_interface_up;
}

lldb::PlatformSP ScriptedPlatform::CreateInstance(bool force,
                                                  const ArchSpec *arch) {
  Log *log = GetLog(LLDBLog::Platform);
  if (log) {
    const char *arch_name;
    if (arch && arch->GetArchitectureName())
      arch_name = arch->GetArchitectureName();
    else
      arch_name = "<null>";

    const char *triple_cstr =
        arch ? arch->GetTriple().getTriple().c_str() : "<null>";

    LLDB_LOGF(log, "ScriptedPlatform::%s(force=%s, arch={%s,%s})",
              __PRETTY_FUNCTION__, force ? "true" : "false", arch_name,
              triple_cstr);
  }

  return std::make_shared<ScriptedPlatform>();
}

ScriptedPlatform::ScriptedPlatform() : Platform(false) {}

llvm::Error ScriptedPlatform::SetupScriptedObject() {

  auto error_with_message = [](llvm::StringRef message) -> llvm::Error {
    Status error;
    ScriptedInterface::ErrorWithMessage<bool>(LLVM_PRETTY_FUNCTION, message,
                                              error, LLDBLog::Platform);
    return error.ToError();
  };

  Debugger &debugger = m_metadata->GetDebugger();

  if (!IsScriptLanguageSupported(debugger.GetScriptLanguage()))
    return error_with_message("Debugger language not supported");

  ScriptInterpreter *interpreter = debugger.GetScriptInterpreter();
  if (!interpreter)
    return error_with_message("Debugger has no Script Interpreter");

  // Create platform instance interface
  m_interface_up = interpreter->CreateScriptedPlatformInterface();
  if (!m_interface_up)
    return error_with_message(
        "Script interpreter couldn't create Scripted Process Interface");

  const ScriptedMetadata scripted_metadata = m_metadata->GetScriptedMetadata();

  ExecutionContext e(&debugger.GetSelectedOrDummyTarget());
  auto obj_or_err = GetInterface().CreatePluginObject(
      scripted_metadata.GetClassName(), e, scripted_metadata.GetArgsSP());

  if (!obj_or_err) {
    llvm::consumeError(obj_or_err.takeError());
    return error_with_message("Failed to create script object.");
  }

  StructuredData::GenericSP object_sp = *obj_or_err;
  if (!object_sp || !object_sp->IsValid())
    return error_with_message("Failed to create valid script object");

  m_hostname = GetHostPlatform()->GetHostname();
  return llvm::Error::success();
}

ScriptedPlatform::~ScriptedPlatform() {}

llvm::Error ScriptedPlatform::ReloadMetadata() {
  if (!m_metadata)
    return llvm::createStringError(
        "Scripted Platform has no platform metadata.");
  return SetupScriptedObject();
}

void ScriptedPlatform::Initialize() {
  if (g_initialize_count++ == 0) {
    // NOTE: This should probably be using the driving process platform
    PluginManager::RegisterPlugin(ScriptedPlatform::GetPluginNameStatic(),
                                  ScriptedPlatform::GetDescriptionStatic(),
                                  ScriptedPlatform::CreateInstance);
  }
}

void ScriptedPlatform::Terminate() {
  if (g_initialize_count > 0) {
    if (--g_initialize_count == 0) {
      PluginManager::UnregisterPlugin(ScriptedPlatform::CreateInstance);
    }
  }
}

std::vector<ArchSpec>
ScriptedPlatform::GetSupportedArchitectures(const ArchSpec &process_host_arch) {
  CheckInterpreterAndScriptObject();
  StructuredData::ArraySP archs_sp = GetInterface().GetSupportedArchitectures();
  if (!archs_sp)
    return {};

  // If the scripted platform didn't provide any supported architecture, we
  // should use the host platform support architecture instead.
  if (!archs_sp->GetSize())
    return GetHostPlatform()->GetSupportedArchitectures(process_host_arch);

  std::vector<ArchSpec> result;
  auto extract_arch_specs = [&result](StructuredData::Object *obj) {
    if (!obj)
      return false;

    StructuredData::String *arch_str = obj->GetAsString();
    if (!arch_str)
      return false;

    ArchSpec arch_spec(arch_str->GetValue());
    if (!arch_spec.IsValid())
      return false;

    result.push_back(arch_spec);
    return true;
  };

  archs_sp->ForEach(extract_arch_specs);

  return result;
}

lldb::ProcessSP
ScriptedPlatform::Attach(lldb_private::ProcessAttachInfo &attach_info,
                         lldb_private::Debugger &debugger,
                         lldb_private::Target *target, // Can be nullptr, if
                                                       // nullptr create a new
                                                       // target, else use
                                                       // existing one
                         lldb_private::Status &error) {
  lldb::ProcessAttachInfoSP attach_info_sp =
      std::make_shared<ProcessAttachInfo>(attach_info);

  if (!target) {
    target = &debugger.GetSelectedOrDummyTarget();
  }

  ProcessSP process_sp =
      GetInterface().AttachToProcess(attach_info_sp, target->shared_from_this(),
                                     debugger.shared_from_this(), error);
  if (!process_sp || error.Fail())
    return {};
  return process_sp;
}

llvm::Expected<ProcessInstanceInfo>
ScriptedPlatform::ParseProcessInfo(StructuredData::Dictionary &dict,
                                   lldb::pid_t pid) const {
  if (!dict.HasKey("name"))
    return llvm::createStringError("No 'name' key in process info dictionary.");
  if (!dict.HasKey("arch"))
    return llvm::createStringError("No 'arch' key in process info dictionary.");

  llvm::StringRef result;
  if (!dict.GetValueForKeyAsString("name", result))
    return llvm::createStringError(
        "Couldn't extract 'name' key from process info dictionary.");
  std::string name = result.str();

  if (!dict.GetValueForKeyAsString("arch", result))
    return llvm::createStringError(
        "Couldn't extract 'arch' key from process info dictionary.");
  const ArchSpec arch(result.data());
  if (!result.empty() && !arch.IsValid())
    return llvm::createStringError(
        "Invalid 'arch' key in process info dictionary.");

  ProcessInstanceInfo proc_info = ProcessInstanceInfo(name.c_str(), arch, pid);

  lldb::pid_t parent = LLDB_INVALID_PROCESS_ID;
  if (dict.GetValueForKeyAsInteger("parent", parent))
    proc_info.SetParentProcessID(parent);

  uint32_t uid = UINT32_MAX;
  if (dict.GetValueForKeyAsInteger("uid", uid))
    proc_info.SetUserID(uid);

  uint32_t gid = UINT32_MAX;
  if (dict.GetValueForKeyAsInteger("gid", gid))
    proc_info.SetGroupID(gid);

  return proc_info;
}

uint32_t
ScriptedPlatform::FindProcesses(const ProcessInstanceInfoMatch &match_info,
                                ProcessInstanceInfoList &proc_infos) {
  CheckInterpreterAndScriptObject();
  StructuredData::DictionarySP processes_sp = GetInterface().ListProcesses();

  Status error;
  if (!processes_sp)
    return ScriptedInterface::ErrorWithMessage<uint32_t>(
        LLVM_PRETTY_FUNCTION, "Failed to get scripted platform processes.",
        error, LLDBLog::Platform);

  // Because `StructuredData::Dictionary` uses a `std::map<ConstString,
  // ObjectSP>` for storage, each item is sorted based on the key alphabetical
  // order. Since `ListProcesses` provides process ids as the key element,
  // process info comes ordered alphabetically, instead of numerically, so we
  // need to sort the process ids before listing them.

  StructuredData::ArraySP keys = processes_sp->GetKeys();

  std::map<lldb::pid_t, StructuredData::ObjectSP> sorted_processes;
  auto sort_keys = [&sorted_processes,
                    &processes_sp](StructuredData::Object *item) -> bool {
    if (!item)
      return false;

    llvm::StringRef key = item->GetStringValue();
    size_t idx = 0;

    // Make sure the provided id is actually an integer
    if (!llvm::to_integer(key, idx))
      return false;

    sorted_processes[idx] = processes_sp->GetValueForKey(key);
    return true;
  };

  size_t process_count = processes_sp->GetSize();

  if (!keys->ForEach(sort_keys) || sorted_processes.size() != process_count)
    // Might be worth showing the unsorted platform process list instead of
    // return early.
    return ScriptedInterface::ErrorWithMessage<bool>(
        LLVM_PRETTY_FUNCTION, "Couldn't sort platform process list.", error);

  auto parse_process_info =
      [this, &proc_infos](
          const std::pair<lldb::pid_t, StructuredData::ObjectSP> pair) -> bool {
    const lldb::pid_t pid = pair.first;
    const StructuredData::ObjectSP val = pair.second;
    if (!val)
      return false;

    StructuredData::Dictionary *dict = val->GetAsDictionary();

    if (!dict || !dict->IsValid())
      return false;

    auto proc_info_or_error = ParseProcessInfo(*dict, pid);

    if (llvm::Error e = proc_info_or_error.takeError()) {
      LLDB_LOG_ERROR(GetLog(LLDBLog::Platform), std::move(e), "{1} ERROR = {0}",
                     LLVM_PRETTY_FUNCTION);
      return false;
    }

    proc_infos.push_back(*proc_info_or_error);
    return true;
  };

  llvm::for_each(sorted_processes, parse_process_info);

  // TODO: Use match_info to filter through processes
  return proc_infos.size();
}

bool ScriptedPlatform::GetProcessInfo(lldb::pid_t pid,
                                      ProcessInstanceInfo &proc_info) {
  if (pid == LLDB_INVALID_PROCESS_ID)
    return false;

  StructuredData::DictionarySP dict_sp = GetInterface().GetProcessInfo(pid);

  if (!dict_sp || !dict_sp->IsValid())
    return false;

  auto proc_info_or_error = ParseProcessInfo(*dict_sp.get(), pid);

  if (!proc_info_or_error) {
    llvm::consumeError(proc_info_or_error.takeError());
    return false;
  }

  proc_info = *proc_info_or_error;
  return true;
}

Status ScriptedPlatform::LaunchProcess(ProcessLaunchInfo &launch_info) {
  lldb::ProcessLaunchInfoSP launch_info_sp =
      std::make_shared<ProcessLaunchInfo>(launch_info);
  return GetInterface().LaunchProcess(launch_info_sp);
}

Status ScriptedPlatform::KillProcess(const lldb::pid_t pid) {
  return GetInterface().KillProcess(pid);
}
