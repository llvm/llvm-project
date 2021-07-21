//===-- DynamicLoaderDumpWithModuleList.h --------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_DYNAMICLOADER_MODULELIST_DYLD_DYNAMICLOADERDUMPWITHMODULELIST_H
#define LLDB_SOURCE_PLUGINS_DYNAMICLOADER_MODULELIST_DYLD_DYNAMICLOADERDUMPWITHMODULELIST_H

#include "../POSIX-DYLD/DYLDRendezvous.h"
#include "Plugins/Process/Utility/AuxVector.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Target/DynamicLoader.h"
#include "llvm/Support/RWMutex.h"

/**
 * Dynamic loader for dump process with module list available.
 * For example, some coredump files have NT_FILE note section available
 * so can directly provide the module list without main executable's dynamic
 * section.
 */
class DynamicLoaderDumpWithModuleList : public lldb_private::DynamicLoader {
public:
  DynamicLoaderDumpWithModuleList(lldb_private::Process *process);

  ~DynamicLoaderDumpWithModuleList() override;

  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() {
    return "dump-modulelist-dyld";
  }

  static llvm::StringRef GetPluginDescriptionStatic();

  static lldb_private::DynamicLoader *
  CreateInstance(lldb_private::Process *process, bool force);

  // DynamicLoader protocol

  void DidAttach() override;

  void DidLaunch() override {
    llvm_unreachable(
        "DynamicLoaderDumpWithModuleList::DidLaunch shouldn't be called");
  }

  lldb::ThreadPlanSP GetStepThroughTrampolinePlan(lldb_private::Thread &thread,
                                                  bool stop_others) override {
    llvm_unreachable("DynamicLoaderDumpWithModuleList::"
                     "GetStepThroughTrampolinePlan shouldn't be called");
  }

  lldb_private::Status CanLoadImage() override;

  lldb::addr_t GetThreadLocalData(const lldb::ModuleSP module,
                                  const lldb::ThreadSP thread,
                                  lldb::addr_t tls_file_addr) override;

  // PluginInterface protocol
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

private:
  DynamicLoaderDumpWithModuleList(const DynamicLoaderDumpWithModuleList &) =
      delete;
  const DynamicLoaderDumpWithModuleList &
  operator=(const DynamicLoaderDumpWithModuleList &) = delete;

  // Structure to hold module information
  struct ModuleInfo {
    std::string name;
    lldb::addr_t base_addr;
    lldb::addr_t module_size;
    lldb::addr_t link_map_addr;
  };

  typedef std::function<void(const std::string &, lldb::addr_t, lldb::addr_t,
                             lldb::addr_t)>
      LoadModuleCallback;
  void LoadAllModules(LoadModuleCallback callback);

  bool ShouldLoadModule(const std::string &module_name);

  std::string SanitizeName(const std::string &input);

  std::optional<const LoadedModuleInfoList::LoadedModuleInfo>
  GetModuleInfo(lldb::addr_t module_base_addr);

  void DetectModuleListMismatch();

  /// Evaluate if Aux vectors contain vDSO and LD information
  /// in case they do, read and assign the address to m_vdso_base
  /// and m_interpreter_base.
  void EvalSpecialModulesStatus();

  void LoadVDSO();

  /// Runtime linker rendezvous structure.
  DYLDRendezvous m_rendezvous;

  /// Auxiliary vector of the inferior process.
  std::unique_ptr<AuxVector> m_auxv;

  /// Contains AT_SYSINFO_EHDR, which means a vDSO has been
  /// mapped to the address space
  lldb::addr_t m_vdso_base;

  /// Cache of module base addr => LoadedModuleInfo map.
  /// It can be used to cross check with posix r_debug link map.
  std::unordered_map<lldb::addr_t, const LoadedModuleInfoList::LoadedModuleInfo>
      m_module_addr_to_info_map;

  // TODO: merge with DynamicLoaderPOSIXDYLD::m_loaded_modules
  // The same as DynamicLoaderPOSIXDYLD::m_loaded_modules to track all loaded
  // module's link map addresses. It is used by TLS to get DTV data structure.
  /// This may be accessed in a multi-threaded context. Use the accessor methods
  /// to access `m_loaded_modules` safely.
  std::map<lldb::ModuleWP, lldb::addr_t, std::owner_less<lldb::ModuleWP>>
      m_loaded_modules;
  mutable llvm::sys::RWMutex m_loaded_modules_rw_mutex;

  void SetLoadedModule(const lldb::ModuleSP &module_sp,
                       lldb::addr_t link_map_addr);
  void UnloadModule(const lldb::ModuleSP &module_sp);
  std::optional<lldb::addr_t>
  GetLoadedModuleLinkAddr(const lldb::ModuleSP &module_sp);
};

#endif // LLDB_SOURCE_PLUGINS_DYNAMICLOADER_MODULELIST_DYLD_DYNAMICLOADERDUMPWITHMODULELIST_H
