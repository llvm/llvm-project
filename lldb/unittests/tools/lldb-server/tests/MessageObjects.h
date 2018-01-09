//===-- MessageObjects.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

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
typedef llvm::DenseMap<uint64_t, uint64_t> U64Map;
typedef llvm::DenseMap<unsigned int, std::string> RegisterMap;

class ProcessInfo {
public:
  static llvm::Expected<ProcessInfo> Create(llvm::StringRef response);
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
             const RegisterMap &registers, unsigned int signal);

  llvm::StringRef ReadRegister(unsigned int register_id) const;
  bool ReadRegisterAsUint64(unsigned int register_id, uint64_t &value) const;

private:
  std::string m_name;
  std::string m_reason;
  RegisterMap m_registers;
  unsigned int m_signal;
};

class JThreadsInfo {
public:
  static llvm::Expected<JThreadsInfo> Create(llvm::StringRef response,
                                             llvm::support::endianness endian);

  const ThreadInfoMap &GetThreadInfos() const;

private:
  JThreadsInfo() = default;
  ThreadInfoMap m_thread_infos;
};

class StopReply {
public:
  static llvm::Expected<StopReply> Create(llvm::StringRef response,
                                          llvm::support::endianness endian);
  const U64Map &GetThreadPcs() const;

private:
  StopReply() = default;
  void ParseResponse(llvm::StringRef response,
                     llvm::support::endianness endian);
  unsigned int m_signal;
  lldb::tid_t m_thread;
  std::string m_name;
  U64Map m_thread_pcs;
  RegisterMap m_registers;
  std::string m_reason;
};

// Common functions for parsing packet data.
llvm::Expected<llvm::StringMap<llvm::StringRef>>
SplitPairList(llvm::StringRef caller, llvm::StringRef s);

template <typename... Args>
llvm::Error make_parsing_error(llvm::StringRef format, Args &&... args) {
  std::string error =
      "Unable to parse " +
      llvm::formatv(format.data(), std::forward<Args>(args)...).str();
  return llvm::make_error<llvm::StringError>(error,
                                             llvm::inconvertibleErrorCode());
}
} // namespace llgs_tests
