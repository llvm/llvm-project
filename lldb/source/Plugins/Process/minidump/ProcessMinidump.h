//===-- ProcessMinidump.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ProcessMinidump_h_
#define liblldb_ProcessMinidump_h_

// Project includes
#include "MinidumpParser.h"
#include "MinidumpTypes.h"

// Other libraries and framework includes
#include "lldb/Target/Process.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/Status.h"

#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

// C Includes
// C++ Includes

namespace lldb_private {

namespace minidump {

class ProcessMinidump : public Process {
public:
  static lldb::ProcessSP CreateInstance(lldb::TargetSP target_sp,
                                        lldb::ListenerSP listener_sp,
                                        const FileSpec *crash_file_path);

  static void Initialize();

  static void Terminate();

  static ConstString GetPluginNameStatic();

  static const char *GetPluginDescriptionStatic();

  ProcessMinidump(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp,
                  const FileSpec &core_file, MinidumpParser minidump_parser);

  ~ProcessMinidump() override;

  bool CanDebug(lldb::TargetSP target_sp,
                bool plugin_specified_by_name) override;

  Status DoLoadCore() override;

  DynamicLoader *GetDynamicLoader() override;

  ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override;

  Status DoDestroy() override;

  void RefreshStateAfterStop() override;

  bool IsAlive() override;

  bool WarnBeforeDetach() const override;

  size_t ReadMemory(lldb::addr_t addr, void *buf, size_t size,
                    Status &error) override;

  size_t DoReadMemory(lldb::addr_t addr, void *buf, size_t size,
                      Status &error) override;

  ArchSpec GetArchitecture();

  Status GetMemoryRegionInfo(lldb::addr_t load_addr,
                             MemoryRegionInfo &range_info) override;

  bool GetProcessInfo(ProcessInstanceInfo &info) override;

  MinidumpParser m_minidump_parser;

protected:
  void Clear();

  bool UpdateThreadList(ThreadList &old_thread_list,
                        ThreadList &new_thread_list) override;

  void ReadModuleList();

private:
  FileSpec m_core_file;
  llvm::ArrayRef<MinidumpThread> m_thread_list;
  const MinidumpExceptionStream *m_active_exception;
  bool m_is_wow64;
};

} // namespace minidump
} // namespace lldb_private

#endif // liblldb_ProcessMinidump_h_
