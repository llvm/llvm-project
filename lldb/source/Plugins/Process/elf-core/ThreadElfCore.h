//===-- ThreadElfCore.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_THREADELFCORE_H
#define LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_THREADELFCORE_H

#include "Plugins/Process/elf-core/RegisterUtilities.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/DataExtractor.h"
#include "llvm/ADT/DenseMap.h"
#include <optional>
#include <string>

struct compat_timeval {
  alignas(8) uint64_t tv_sec;
  alignas(8) uint64_t tv_usec;
};

namespace lldb_private {
class ProcessInstanceInfo;
}

// PRSTATUS structure's size differs based on architecture.
// This is the layout in the x86-64 arch.
// In the i386 case we parse it manually and fill it again
// in the same structure
// The gp registers are also a part of this struct, but they are handled
// separately

#undef si_signo
#undef si_code
#undef si_errno
#undef si_addr
#undef si_addr_lsb

struct ELFLinuxPrStatus {
  int32_t si_signo;
  int32_t si_code;
  int32_t si_errno;

  int16_t pr_cursig;

  alignas(8) uint64_t pr_sigpend;
  alignas(8) uint64_t pr_sighold;

  uint32_t pr_pid;
  uint32_t pr_ppid;
  uint32_t pr_pgrp;
  uint32_t pr_sid;

  compat_timeval pr_utime;
  compat_timeval pr_stime;
  compat_timeval pr_cutime;
  compat_timeval pr_cstime;

  ELFLinuxPrStatus();

  lldb_private::Status Parse(const lldb_private::DataExtractor &data,
                             const lldb_private::ArchSpec &arch);

  static std::optional<ELFLinuxPrStatus>
  Populate(const lldb::ThreadSP &thread_sp);

  // Return the bytesize of the structure
  // 64 bit - just sizeof
  // 32 bit - hardcoded because we are reusing the struct, but some of the
  // members are smaller -
  // so the layout is not the same
  static size_t GetSize(const lldb_private::ArchSpec &arch);
};

static_assert(sizeof(ELFLinuxPrStatus) == 112,
              "sizeof ELFLinuxPrStatus is not correct!");

struct ELFLinuxSigInfo {

  int32_t si_signo; // Order matters for the first 3.
  int32_t si_errno;
  int32_t si_code;
  // Copied from siginfo_t so we don't have to include signal.h on non 'Nix
  // builds. Slight modifications to ensure no 32b vs 64b differences.
  struct alignas(8) {
    lldb::addr_t si_addr; /* faulting insn/memory ref. */
    int16_t si_addr_lsb;  /* Valid LSB of the reported address.  */
    union {
      /* used when si_code=SEGV_BNDERR */
      struct {
        lldb::addr_t _lower;
        lldb::addr_t _upper;
      } _addr_bnd;
      /* used when si_code=SEGV_PKUERR */
      uint32_t _pkey;
    } bounds;
  } sigfault;

  enum SigInfoNoteType : uint8_t { eUnspecified, eNT_SIGINFO };
  SigInfoNoteType note_type;

  ELFLinuxSigInfo();

  lldb_private::Status Parse(const lldb_private::DataExtractor &data,
                             const lldb_private::ArchSpec &arch,
                             const lldb_private::UnixSignals &unix_signals);

  std::string
  GetDescription(const lldb_private::UnixSignals &unix_signals) const;

  // Return the bytesize of the structure
  // 64 bit - just sizeof
  // 32 bit - hardcoded because we are reusing the struct, but some of the
  // members are smaller -
  // so the layout is not the same
  static size_t GetSize(const lldb_private::ArchSpec &arch);
};

static_assert(sizeof(ELFLinuxSigInfo) == 56,
              "sizeof ELFLinuxSigInfo is not correct!");

// PRPSINFO structure's size differs based on architecture.
// This is the layout in the x86-64 arch case.
// In the i386 case we parse it manually and fill it again
// in the same structure
struct ELFLinuxPrPsInfo {
  char pr_state;
  char pr_sname;
  char pr_zomb;
  char pr_nice;
  alignas(8) uint64_t pr_flag;
  uint32_t pr_uid;
  uint32_t pr_gid;
  int32_t pr_pid;
  int32_t pr_ppid;
  int32_t pr_pgrp;
  int32_t pr_sid;
  char pr_fname[16];
  char pr_psargs[80];

  ELFLinuxPrPsInfo();

  lldb_private::Status Parse(const lldb_private::DataExtractor &data,
                             const lldb_private::ArchSpec &arch);

  static std::optional<ELFLinuxPrPsInfo>
  Populate(const lldb::ProcessSP &process_sp);

  static std::optional<ELFLinuxPrPsInfo>
  Populate(const lldb_private::ProcessInstanceInfo &info,
           lldb::StateType state);

  // Return the bytesize of the structure
  // 64 bit - just sizeof
  // 32 bit - hardcoded because we are reusing the struct, but some of the
  // members are smaller -
  // so the layout is not the same
  static size_t GetSize(const lldb_private::ArchSpec &arch);
};

static_assert(sizeof(ELFLinuxPrPsInfo) == 136,
              "sizeof ELFLinuxPrPsInfo is not correct!");

struct ThreadData {
  lldb_private::DataExtractor gpregset;
  std::vector<lldb_private::CoreNote> notes;
  lldb::tid_t tid;
  std::string name;
  ELFLinuxSigInfo siginfo;
  int prstatus_sig = 0;
};

class ThreadElfCore : public lldb_private::Thread {
public:
  ThreadElfCore(lldb_private::Process &process, const ThreadData &td);

  ~ThreadElfCore() override;

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

  void CreateStopFromSigInfo(const ELFLinuxSigInfo &siginfo,
                             const lldb_private::UnixSignals &unix_signals);

protected:
  // Member variables.
  std::string m_thread_name;
  lldb::RegisterContextSP m_thread_reg_ctx_sp;

  lldb_private::DataExtractor m_gpregset_data;
  std::vector<lldb_private::CoreNote> m_notes;
  ELFLinuxSigInfo m_siginfo;

  bool CalculateStopInfo() override;
};

#endif // LLDB_SOURCE_PLUGINS_PROCESS_ELF_CORE_THREADELFCORE_H
