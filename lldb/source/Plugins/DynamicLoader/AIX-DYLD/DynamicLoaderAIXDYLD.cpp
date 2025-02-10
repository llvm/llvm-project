//===-- DynamicLoaderAIXDYLD.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DynamicLoaderAIXDYLD.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/ThreadPlanStepInstruction.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/Support/FileSystem.h"
#if defined(_AIX)
#include <sys/ldr.h>
#include <procinfo.h>
#include <sys/procfs.h>
#include <iostream>
#include <fstream>
#endif
#include "/LLDB/hemang/lldb-for-aix/lldb/source/Plugins/DynamicLoader/POSIX-DYLD/DYLDRendezvous.h"

/*#include "llvm/ADT/Triple.h"
*/

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(DynamicLoaderAIXDYLD)

DynamicLoaderAIXDYLD::DynamicLoaderAIXDYLD(Process *process)
    : DynamicLoader(process) {}

DynamicLoaderAIXDYLD::~DynamicLoaderAIXDYLD() = default;

void DynamicLoaderAIXDYLD::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance);
}

void DynamicLoaderAIXDYLD::Terminate() {}

llvm::StringRef DynamicLoaderAIXDYLD::GetPluginDescriptionStatic() {
  return "Dynamic loader plug-in that watches for shared library "
         "loads/unloads in AIX processes.";
}

DynamicLoader *DynamicLoaderAIXDYLD::CreateInstance(Process *process,
                                                        bool force) {
  bool should_create = force;
  if (!should_create) {
    const llvm::Triple &triple_ref =
        process->GetTarget().GetArchitecture().GetTriple();
    if (triple_ref.getOS() == llvm::Triple::AIX)
      should_create = true;
  }

  if (should_create)
    return new DynamicLoaderAIXDYLD(process);

  return nullptr;
}

void DynamicLoaderAIXDYLD::OnLoadModule(lldb::ModuleSP module_sp,
                                            const ModuleSpec module_spec,
                                            lldb::addr_t module_addr) {

  // Resolve the module unless we already have one.
  if (!module_sp) {
    Status error;
    module_sp = m_process->GetTarget().GetOrCreateModule(module_spec, 
                                             true /* notify */, &error);
    if (error.Fail())
      return;
  }

  m_loaded_modules[module_sp] = module_addr;
  UpdateLoadedSectionsCommon(module_sp, module_addr, false);
  ModuleList module_list;
  module_list.Append(module_sp);
  m_process->GetTarget().ModulesDidLoad(module_list);
}

void DynamicLoaderAIXDYLD::OnUnloadModule(lldb::addr_t module_addr) {
    Log *log = GetLog(LLDBLog::DynamicLoader);                                    
  LLDB_LOGF(log, "DynamicLoaderAIXDYLD::%s()", __FUNCTION__); 

    Address resolved_addr;
  if (!m_process->GetTarget().ResolveLoadAddress(module_addr, resolved_addr))
    return;

  ModuleSP module_sp = resolved_addr.GetModule();
  if (module_sp) {
    m_loaded_modules.erase(module_sp);
    UnloadSectionsCommon(module_sp);
    ModuleList module_list;
    module_list.Append(module_sp);
    m_process->GetTarget().ModulesDidUnload(module_list, false);
  }
}

lldb::addr_t DynamicLoaderAIXDYLD::GetLoadAddress(ModuleSP executable) {
  // First, see if the load address is already cached.
  auto it = m_loaded_modules.find(executable);
  if (it != m_loaded_modules.end() && it->second != LLDB_INVALID_ADDRESS)
    return it->second;

  lldb::addr_t load_addr = LLDB_INVALID_ADDRESS;

  // Second, try to get it through the process plugins.  For a remote process,
  // the remote platform will be responsible for providing it.
  FileSpec file_spec(executable->GetPlatformFileSpec());
  bool is_loaded = false;
  Status status =
      m_process->GetFileLoadAddress(file_spec, is_loaded, load_addr);
  // Servers other than lldb server could respond with a bogus address.
  if (status.Success() && is_loaded && load_addr != LLDB_INVALID_ADDRESS) {
    m_loaded_modules[executable] = load_addr;
    return load_addr;
  }

  //// Hack to try set breakpoint
  //Breakpoint *dyld_break = m_process->GetTarget().CreateBreakpoint(0x100000638, true, false).get();
  //dyld_break->SetCallback(DynamicLoaderAIXDYLD::NotifyBreakpointHit, this, true);
  //dyld_break->SetBreakpointKind("hack-debug");

  return LLDB_INVALID_ADDRESS;
}

bool DynamicLoaderAIXDYLD::NotifyBreakpointHit(
    void *baton, StoppointCallbackContext *context, lldb::user_id_t break_id,
    lldb::user_id_t break_loc_id) {
}


void DynamicLoaderAIXDYLD::ResolveExecutableModule(
    lldb::ModuleSP &module_sp) {
  Log *log = GetLog(LLDBLog::DynamicLoader);

  if (m_process == nullptr)
    return;

  auto &target = m_process->GetTarget();
  const auto platform_sp = target.GetPlatform();

  ProcessInstanceInfo process_info;
  if (!m_process->GetProcessInfo(process_info)) {
    LLDB_LOGF(log,
              "DynamicLoaderAIXDYLD::%s - failed to get process info for "
              "pid %" PRIu64,
              __FUNCTION__, m_process->GetID());
    return;
  }

  int32long64_t pid = m_process->GetID();
  char cwd[PATH_MAX], resolved_path[PATH_MAX];
  std::string executable_name;
  bool path_resolved = false;
  psinfo_t psinfo;

  std::string proc_file = "/proc/" + std::to_string(pid) + "/psinfo";
  std::string cwd_link = "/proc/" + std::to_string(pid) + "/cwd";
  std::ifstream file(proc_file, std::ios::binary);
  if(!file.is_open())
      LLDB_LOGF(log, "Error: Unable to access process info ");

  file.read(reinterpret_cast<char*>(&psinfo), sizeof(psinfo_t));
  if(!file)
      LLDB_LOGF(log, "Process info error: Failed to read ");

  std::string relative_path(psinfo.pr_fname);
  LLDB_LOGF(log, "Relative path %s",relative_path.c_str());

  if(readlink(cwd_link.c_str(), cwd, sizeof(cwd)) != -1){
      std::filesystem::path full_path = std::filesystem::path(cwd)/relative_path; 
  if(realpath(full_path.c_str(), resolved_path)) {
      LLDB_LOGF(log, "Resolved Path using process info : %s", resolved_path);
      path_resolved = true;
  }
  else
      LLDB_LOGF(log, "Realpath error: Unable to resolve. ");
  }
  
  executable_name = resolved_path;
  if(path_resolved == false) {
      std::string command_line(psinfo.pr_psargs);
      LLDB_LOGF(log, "Command line: %s",command_line.c_str());
      if (!command_line.empty()) {
          size_t space1 = command_line.find(' ');
          executable_name = command_line.substr(0, space1);
          LLDB_LOGF(log, "Resolved path using command line arg %s",executable_name.c_str());
      } 
  }

  LLDB_LOGF(log, "Executable Name %s",executable_name.c_str());
  process_info.SetExecutableFile(lldb_private::FileSpec(executable_name),
          true);
 
  LLDB_LOGF(
      log, "DynamicLoaderAIXDYLD::%s - got executable by pid %" PRIu64 ": %s",
      __FUNCTION__, m_process->GetID(),
      process_info.GetExecutableFile().GetPath().c_str());

  ModuleSpec module_spec(process_info.GetExecutableFile(),
                         process_info.GetArchitecture());

  if (module_sp && module_sp->MatchesModuleSpec(module_spec))
    return;
 LLDB_LOGF(log,"DynamicLoaderAIXDYLD::%s",__FUNCTION__);
  const auto executable_search_paths(Target::GetDefaultExecutableSearchPaths());
  auto error = platform_sp->ResolveExecutable(
      module_spec, module_sp,
      !executable_search_paths.IsEmpty() ? &executable_search_paths : nullptr);
  if (error.Fail()) {
    StreamString stream;
    module_spec.Dump(stream);

    LLDB_LOGF(log,
              "DynamicLoaderPOSIXDYLD::%s - failed to resolve executable "
              "with module spec \"%s\": %s",
              __FUNCTION__, stream.GetData(), error.AsCString());
    
    return;
  }

  target.SetExecutableModule(module_sp, eLoadDependentsNo);
}

void DynamicLoaderAIXDYLD::DidAttach() {
    Log *log = GetLog(LLDBLog::DynamicLoader);
  LLDB_LOGF(log, "DynamicLoaderAIXDYLD::%s()", __FUNCTION__);

  ModuleSP executable = GetTargetExecutable();
  ResolveExecutableModule(executable);

  Module *exe_mod = m_process->GetTarget().GetExecutableModulePointer();
  if (exe_mod) {
      exe_mod->GetPlatformFileSpec();
      LLDB_LOGF(log, "DynamicLoaderAIXDYLD::%s exe module executable path set",
              __FUNCTION__);
  } else {
      LLDB_LOGF(log,
              "DynamicLoaderAIXDYLD::%s cannot cache exe module path: null "
              "executable module pointer",
              __FUNCTION__);
  }

  if (!executable.get())
      return;
  LLDB_LOGF(log, "DynamicLoaderAIXDYLD::%s()", __FUNCTION__);

  // Try to fetch the load address of the file from the process, since there
  // could be randomization of the load address.
  lldb::addr_t load_addr = GetLoadAddress(executable);
  if (load_addr == LLDB_INVALID_ADDRESS)
    return;

  LLDB_LOGF(log,"DidAttach load_addr 0x%p",load_addr);
  // Request the process base address.
  lldb::addr_t image_base = m_process->GetImageInfoAddress();
  if (image_base == load_addr)
    return;

  // Rebase the process's modules if there is a mismatch.
  UpdateLoadedSections(executable, LLDB_INVALID_ADDRESS, load_addr, false);

/*
  DYLDRendezvous m_rendezvous(m_process);
  if (!m_rendezvous.Resolve()) {
    LLDB_LOGF(log,
              "DynamicLoaderPOSIXDYLD::%s unable to resolve POSIX DYLD "
              "rendezvous address",
              __FUNCTION__);
    return;
  } */


  ModuleList module_list;
  module_list.Append(executable);
  m_process->GetTarget().ModulesDidLoad(module_list);
  auto error = m_process->LoadModules();
  LLDB_LOG_ERROR(log, std::move(error), "failed to load modules: {0}");

#if defined(_AIX)
   LLDB_LOGF(log, "DynamicLoaderAIXDYLD::%s() 289", __FUNCTION__);
  // Get struct ld_xinfo (FIXME)
  struct ld_xinfo ldinfo[64];
  Status status = m_process->GetLDXINFO(&(ldinfo[0]));
  if (status.Fail()) {
    Log *log = GetLog(LLDBLog::DynamicLoader);
    LLDB_LOG(log, "LDXINFO failed: {0}", status);
    return;
  }
  struct ld_xinfo *ptr = &(ldinfo[0]);
  bool skip_current = true;
  while (ptr != nullptr) {
    char *pathName = (char *)ptr + ptr->ldinfo_filename;
    char *memberName = pathName + (strlen(pathName) + 1);
    if (!skip_current) {
      // FIXME: buffer size
      char pathWithMember[128] = {0};
      if (strlen(memberName) > 0) {
        sprintf(pathWithMember, "%s(%s)", pathName, memberName);
      } else {
        sprintf(pathWithMember, "%s", pathName);
      }
       LLDB_LOGF(log, "DynamicLoaderAIXDYLD1::%s(), %s", __FUNCTION__,pathWithMember);
      FileSpec file(pathWithMember);
      ModuleSpec module_spec(file, m_process->GetTarget().GetArchitecture());
      LLDB_LOGF(log,"After dis %s",module_spec.GetFileSpec().GetPath().c_str());
      if (ModuleSP module_sp = m_process->GetTarget().GetOrCreateModule(module_spec, true /* notify */)) {
        UpdateLoadedSectionsByType(module_sp, LLDB_INVALID_ADDRESS, (lldb::addr_t)ptr->ldinfo_textorg, false, 1);
        UpdateLoadedSectionsByType(module_sp, LLDB_INVALID_ADDRESS, (lldb::addr_t)ptr->ldinfo_dataorg, false, 2);
        // FIXME: .tdata, .bss
      }
    } else {
      skip_current = false;
    }
    if (ptr->ldinfo_next == 0) {
      ptr = nullptr;
    } else {
      ptr = (struct ld_xinfo *)((char *)ptr + ptr->ldinfo_next);
    }
  }
#endif
}

void DynamicLoaderAIXDYLD::DidLaunch() {
  Log *log = GetLog(LLDBLog::DynamicLoader);
  LLDB_LOGF(log, "DynamicLoaderAIXDYLD::%s()", __FUNCTION__);

  ModuleSP executable = GetTargetExecutable();
  if (!executable.get())
    return;

  lldb::addr_t load_addr = GetLoadAddress(executable);
  LLDB_LOGF(log,"DidLaunch load_addr 0x%p",load_addr);
  if (load_addr != LLDB_INVALID_ADDRESS) {
    // Update the loaded sections so that the breakpoints can be resolved.
    UpdateLoadedSections(executable, LLDB_INVALID_ADDRESS, load_addr, false);

    ModuleList module_list;
    module_list.Append(executable);
    m_process->GetTarget().ModulesDidLoad(module_list);
    auto error = m_process->LoadModules();
    LLDB_LOG_ERROR(log, std::move(error), "failed to load modules: {0}");
  }

#if defined(_AIX)
  // Get struct ld_xinfo (FIXME)
  struct ld_xinfo ldinfo[64];
  Status status = m_process->GetLDXINFO(&(ldinfo[0]));
  if (status.Fail()) {
    Log *log = GetLog(LLDBLog::DynamicLoader);
    LLDB_LOG(log, "LDXINFO failed: {0}", status);
    return;
  }
  struct ld_xinfo *ptr = &(ldinfo[0]);
  bool skip_current = true;
  while (ptr != nullptr) {
    char *pathName = (char *)ptr + ptr->ldinfo_filename;
    char *memberName = pathName + (strlen(pathName) + 1);
    if (!skip_current) {
      // FIXME: buffer size
      char pathWithMember[128] = {0};
      if (strlen(memberName) > 0) {
        sprintf(pathWithMember, "%s(%s)", pathName, memberName);
      } else {
        sprintf(pathWithMember, "%s", pathName);
      }
      FileSpec file(pathWithMember);
      ModuleSpec module_spec(file, m_process->GetTarget().GetArchitecture());
      if (ModuleSP module_sp = m_process->GetTarget().GetOrCreateModule(module_spec, true /* notify */)) {
        UpdateLoadedSectionsByType(module_sp, LLDB_INVALID_ADDRESS, (lldb::addr_t)ptr->ldinfo_textorg, false, 1);
        UpdateLoadedSectionsByType(module_sp, LLDB_INVALID_ADDRESS, (lldb::addr_t)ptr->ldinfo_dataorg, false, 2);
        // FIXME: .tdata, .bss
      }
    } else {
      skip_current = false;
    }
    if (ptr->ldinfo_next == 0) {
      ptr = nullptr;
    } else {
      ptr = (struct ld_xinfo *)((char *)ptr + ptr->ldinfo_next);
    }
  }
#endif
}

Status DynamicLoaderAIXDYLD::CanLoadImage() { return Status(); }

ThreadPlanSP
DynamicLoaderAIXDYLD::GetStepThroughTrampolinePlan(Thread &thread,
                                                       bool stop) {
  //FIXME
  return ThreadPlanSP();
}
