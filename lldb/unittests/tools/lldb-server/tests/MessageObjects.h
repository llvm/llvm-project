//===-- MessageObjects.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SERVER_TESTS_MESSAGEOBJECTS_H
#define LLDB_SERVER_TESTS_MESSAGEOBJECTS_H

#include "lldb/Core/RegisterValue.h"
#include "lldb/Host/Host.h"
#include "lldb/Utility/StructuredData.h"
#include "lldb/lldb-types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include <string>

namespace llgs_tests {
class ThreadInfo;
typedef llvm::DenseMap<uint64_t, ThreadInfo> ThreadInfoMap;
typedef llvm::DenseMap<unsigned int, lldb_private::RegisterValue> RegisterMap;

template <typename T> struct Parser { using result_type = T; };

class ProcessInfo : public Parser<ProcessInfo> {
public:
  static llvm::Expected<ProcessInfo> create(llvm::StringRef response);
  lldb::pid_t GetPid() const;
  llvm::support::endianness GetEndian() const;

private:
  ProcessInfo() = default;
  lldb::pid_t m_pid;
  lldb::pid_t m_parent_pid;
  uint32_t m_real_uid;
  uint32_t m_real_gid;
  uint32_t m_effective_uid;
  uint32_t m_effective_gid;
  std::string m_triple;
  llvm::SmallString<16> m_ostype;
  llvm::support::endianness m_endian;
  unsigned int m_ptrsize;
};

class ThreadInfo {
public:
  ThreadInfo() = default;
  ThreadInfo(llvm::StringRef name, llvm::StringRef reason,
             RegisterMap registers, unsigned int signal);

  const lldb_private::RegisterValue *ReadRegister(unsigned int Id) const;

private:
  std::string m_name;
  std::string m_reason;
  RegisterMap m_registers;
  unsigned int m_signal;
};

class JThreadsInfo : public Parser<JThreadsInfo> {
public:
  static llvm::Expected<JThreadsInfo>
  create(llvm::StringRef Response,
         llvm::ArrayRef<lldb_private::RegisterInfo> RegInfos);

  const ThreadInfoMap &GetThreadInfos() const;

private:
  static llvm::Expected<RegisterMap>
  parseRegisters(const lldb_private::StructuredData::Dictionary &Dict,
                 llvm::ArrayRef<lldb_private::RegisterInfo> RegInfos);

  JThreadsInfo() = default;
  ThreadInfoMap m_thread_infos;
};

struct RegisterInfoParser : public Parser<lldb_private::RegisterInfo> {
  static llvm::Expected<lldb_private::RegisterInfo>
  create(llvm::StringRef Response);
};

llvm::Expected<lldb_private::RegisterValue>
parseRegisterValue(const lldb_private::RegisterInfo &Info,
                   llvm::StringRef HexValue, llvm::support::endianness Endian,
                   bool ZeroPad = false);

class StopReply : public Parser<std::unique_ptr<StopReply>> {
public:
  StopReply() = default;
  virtual ~StopReply() = default;

  static llvm::Expected<std::unique_ptr<StopReply>>
  create(llvm::StringRef Response, llvm::support::endianness Endian,
         llvm::ArrayRef<lldb_private::RegisterInfo> RegInfos);

  // for llvm::cast<>
  virtual lldb_private::WaitStatus getKind() const = 0;

  StopReply(const StopReply &) = delete;
  void operator=(const StopReply &) = delete;
};

class StopReplyStop : public StopReply {
public:
  StopReplyStop(uint8_t Signal, lldb::tid_t ThreadId, llvm::StringRef Name,
                RegisterMap ThreadPcs, RegisterMap Registers,
                llvm::StringRef Reason)
      : Signal(Signal), ThreadId(ThreadId), Name(Name),
        ThreadPcs(std::move(ThreadPcs)), Registers(std::move(Registers)),
        Reason(Reason) {}

  static llvm::Expected<std::unique_ptr<StopReplyStop>>
  create(llvm::StringRef Response, llvm::support::endianness Endian,
         llvm::ArrayRef<lldb_private::RegisterInfo> RegInfos);

  const RegisterMap &getThreadPcs() const { return ThreadPcs; }
  lldb::tid_t getThreadId() const { return ThreadId; }

  // for llvm::cast<>
  lldb_private::WaitStatus getKind() const override {
    return lldb_private::WaitStatus{lldb_private::WaitStatus::Stop, Signal};
  }
  static bool classof(const StopReply *R) {
    return R->getKind().type == lldb_private::WaitStatus::Stop;
  }

private:
  static llvm::Expected<RegisterMap> parseRegisters(
      const llvm::StringMap<llvm::SmallVector<llvm::StringRef, 2>> &Elements,
      llvm::support::endianness Endian,
      llvm::ArrayRef<lldb_private::RegisterInfo> RegInfos);

  uint8_t Signal;
  lldb::tid_t ThreadId;
  std::string Name;
  RegisterMap ThreadPcs;
  RegisterMap Registers;
  std::string Reason;
};

class StopReplyExit : public StopReply {
public:
  explicit StopReplyExit(uint8_t Status) : Status(Status) {}

  static llvm::Expected<std::unique_ptr<StopReplyExit>>
  create(llvm::StringRef response);

  // for llvm::cast<>
  lldb_private::WaitStatus getKind() const override {
    return lldb_private::WaitStatus{lldb_private::WaitStatus::Exit, Status};
  }
  static bool classof(const StopReply *R) {
    return R->getKind().type == lldb_private::WaitStatus::Exit;
  }

private:
  uint8_t Status;
};

// Common functions for parsing packet data.
llvm::Expected<llvm::StringMap<llvm::StringRef>>
SplitUniquePairList(llvm::StringRef caller, llvm::StringRef s);

llvm::StringMap<llvm::SmallVector<llvm::StringRef, 2>>
SplitPairList(llvm::StringRef s);

template <typename... Args>
llvm::Error make_parsing_error(llvm::StringRef format, Args &&... args) {
  std::string error =
      "Unable to parse " +
      llvm::formatv(format.data(), std::forward<Args>(args)...).str();
  return llvm::make_error<llvm::StringError>(error,
                                             llvm::inconvertibleErrorCode());
}

} // namespace llgs_tests

namespace lldb_private {
std::ostream &operator<<(std::ostream &OS, const RegisterValue &RegVal);
}

#endif // LLDB_SERVER_TESTS_MESSAGEOBJECTS_H
