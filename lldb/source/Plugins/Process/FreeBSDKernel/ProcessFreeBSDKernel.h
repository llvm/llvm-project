//===-- ProcessFreeBSDKernel.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_FREEBSDKERNEL_PROCESSFREEBSDKERNEL_H
#define LLDB_SOURCE_PLUGINS_PROCESS_FREEBSDKERNEL_PROCESSFREEBSDKERNEL_H

#include "lldb/Target/PostMortemProcess.h"

#include <kvm.h>

class ProcessFreeBSDKernel : public lldb_private::PostMortemProcess {
public:
  ProcessFreeBSDKernel(lldb::TargetSP target_sp, lldb::ListenerSP listener,
                       kvm_t *kvm, const lldb_private::FileSpec &core_file);

  ~ProcessFreeBSDKernel();

  static lldb::ProcessSP
  CreateInstance(lldb::TargetSP target_sp, lldb::ListenerSP listener,
                 const lldb_private::FileSpec *crash_file_path,
                 bool can_connect);

  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "freebsd-kernel"; }

  static llvm::StringRef GetPluginDescriptionStatic() {
    return "FreeBSD kernel vmcore debugging plug-in.";
  }

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  lldb_private::Status DoDestroy() override;

  bool CanDebug(lldb::TargetSP target_sp,
                bool plugin_specified_by_name) override;

  void RefreshStateAfterStop() override;

  lldb_private::Status DoLoadCore() override;

  lldb_private::DynamicLoader *GetDynamicLoader() override;

  size_t DoReadMemory(lldb::addr_t addr, void *buf, size_t size,
                      lldb_private::Status &error) override;

protected:
  bool DoUpdateThreadList(lldb_private::ThreadList &old_thread_list,
                          lldb_private::ThreadList &new_thread_list) override;

  lldb::addr_t FindSymbol(const char *name);

private:
  void PrintUnreadMessage();

  const char *GetError();

  bool m_printed_unread_message = false;

  kvm_t *m_kvm;
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_FREEBSDKERNEL_PROCESSFREEBSDKERNEL_H
