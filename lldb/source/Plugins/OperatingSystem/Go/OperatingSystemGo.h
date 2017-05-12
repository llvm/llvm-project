//===-- OperatingSystemGo.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _liblldb_OperatingSystemGo_h_
#define _liblldb_OperatingSystemGo_h_

// C Includes
// C++ Includes
#include <memory>

// Other libraries and framework includes
// Project includes
#include "lldb/Target/OperatingSystem.h"

class DynamicRegisterInfo;

class OperatingSystemGo : public lldb_private::OperatingSystem {
public:
  OperatingSystemGo(lldb_private::Process *process);

  ~OperatingSystemGo() override;

  //------------------------------------------------------------------
  // Static Functions
  //------------------------------------------------------------------
  static lldb_private::OperatingSystem *
  CreateInstance(lldb_private::Process *process, bool force);

  static void Initialize();

  static void DebuggerInitialize(lldb_private::Debugger &debugger);

  static void Terminate();

  static lldb_private::ConstString GetPluginNameStatic();

  static const char *GetPluginDescriptionStatic();

  //------------------------------------------------------------------
  // lldb_private::PluginInterface Methods
  //------------------------------------------------------------------
  lldb_private::ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override;

  //------------------------------------------------------------------
  // lldb_private::OperatingSystem Methods
  //------------------------------------------------------------------
  bool UpdateThreadList(lldb_private::ThreadList &old_thread_list,
                        lldb_private::ThreadList &real_thread_list,
                        lldb_private::ThreadList &new_thread_list) override;

  void ThreadWasSelected(lldb_private::Thread *thread) override;

  lldb::RegisterContextSP
  CreateRegisterContextForThread(lldb_private::Thread *thread,
                                 lldb::addr_t reg_data_addr) override;

  lldb::StopInfoSP
  CreateThreadStopReason(lldb_private::Thread *thread) override;

  //------------------------------------------------------------------
  // Method for lazy creation of threads on demand
  //------------------------------------------------------------------
  lldb::ThreadSP CreateThread(lldb::tid_t tid, lldb::addr_t context) override;

private:
  struct Goroutine;

  static lldb::ValueObjectSP FindGlobal(lldb::TargetSP target,
                                        const char *name);

  static lldb::TypeSP FindType(lldb::TargetSP target_sp, const char *name);

  bool Init(lldb_private::ThreadList &threads);

  Goroutine CreateGoroutineAtIndex(uint64_t idx, lldb_private::Status &err);

  std::unique_ptr<DynamicRegisterInfo> m_reginfo;
  lldb::ValueObjectSP m_allg_sp;
  lldb::ValueObjectSP m_allglen_sp;
};

#endif // liblldb_OperatingSystemGo_h_
