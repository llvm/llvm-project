//===-- ThreadAIXCore.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_AIX_CORE_THREADAIXCORE_H
#define LLDB_SOURCE_PLUGINS_PROCESS_AIX_CORE_THREADAIXCORE_H

#include "lldb/Target/Thread.h"
#include "lldb/Utility/DataExtractor.h"
#include "llvm/ADT/DenseMap.h"
#include <optional>
#include <string>
#include "ProcessAIXCore.h"
#include "AIXCore.h"
#include "ThreadAIXCore.h"

namespace lldb_private {
class ProcessInstanceInfo;
}

struct AIXSigInfo {
 
  //COPY siginfo_t correctly for AIX version
  int32_t si_signo; // Order matters for the first 3.
  int32_t si_errno;
  int32_t si_code;
  struct alignas(8) {
    lldb::addr_t si_addr; 
    int16_t si_addr_lsb;
    union {
      struct {
        lldb::addr_t _lower;
        lldb::addr_t _upper;
      } _addr_bnd;
      uint32_t _pkey;
    } bounds;
  } sigfault;

  enum SigInfoNoteType : uint8_t { eUnspecified, eNT_SIGINFO };
  SigInfoNoteType note_type;

  AIXSigInfo();

  void Parse(const AIXCORE::AIXCore64Header data,
                             const lldb_private::ArchSpec &arch,
                             const lldb_private::UnixSignals &unix_signals);

  std::string
  GetDescription(const lldb_private::UnixSignals &unix_signals) const;

  static size_t GetSize(const lldb_private::ArchSpec &arch);
};

struct ThreadData {
  lldb_private::DataExtractor gpregset;
  std::vector<lldb_private::DataExtractor> notes;
  lldb::tid_t tid;
  std::string name;
  AIXSigInfo siginfo;
  int prstatus_sig = 0;
};

class ThreadAIXCore : public lldb_private::Thread {
public:
  ThreadAIXCore(lldb_private::Process &process, const ThreadData &td);

  ~ThreadAIXCore() override;

  void RefreshStateAfterStop() override;

  lldb::RegisterContextSP GetRegisterContext() override;

  lldb::RegisterContextSP
  CreateRegisterContextForFrame(lldb_private::StackFrame *frame) override;

  static bool ThreadIDIsValid(lldb::tid_t thread) { return thread != 0; }

  const char *GetName() override {
    if (m_thread_name.empty())
      return nullptr;
    return m_thread_name.c_str();
  }

  void SetName(const char *name) override {
    if (name && name[0])
      m_thread_name.assign(name);
    else
      m_thread_name.clear();
  }

  void CreateStopFromSigInfo(const AIXSigInfo &siginfo,
                             const lldb_private::UnixSignals &unix_signals);

protected:
  // Member variables.
  std::string m_thread_name;
  lldb::RegisterContextSP m_thread_reg_ctx_sp;

  lldb_private::DataExtractor m_gpregset_data;
  std::vector<lldb_private::DataExtractor> m_notes;
  AIXSigInfo m_siginfo;

  bool CalculateStopInfo() override;
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_AIX_CORE_THREADAIXCORE_H
