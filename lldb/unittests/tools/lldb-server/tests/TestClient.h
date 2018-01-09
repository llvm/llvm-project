//===-- TestClient.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MessageObjects.h"
#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationClient.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Target/ProcessLaunchInfo.h"
#include "llvm/ADT/Optional.h"
#include <memory>
#include <string>

namespace llgs_tests {
// TODO: Make the test client an abstract base class, with different children
// for different types of connections: llgs v. debugserver
class TestClient
    : public lldb_private::process_gdb_remote::GDBRemoteCommunicationClient {
public:
  static void Initialize();
  TestClient(const std::string &test_name, const std::string &test_case_name);
  virtual ~TestClient();
  LLVM_NODISCARD bool StartDebugger();
  LLVM_NODISCARD bool StopDebugger();
  LLVM_NODISCARD bool SetInferior(llvm::ArrayRef<std::string> inferior_args);
  LLVM_NODISCARD bool ListThreadsInStopReply();
  LLVM_NODISCARD bool SetBreakpoint(unsigned long address);
  LLVM_NODISCARD bool ContinueAll();
  LLVM_NODISCARD bool ContinueThread(unsigned long thread_id);
  const ProcessInfo &GetProcessInfo();
  llvm::Optional<JThreadsInfo> GetJThreadsInfo();
  const StopReply &GetLatestStopReply();
  LLVM_NODISCARD bool SendMessage(llvm::StringRef message);
  LLVM_NODISCARD bool SendMessage(llvm::StringRef message,
                                  std::string &response_string);
  LLVM_NODISCARD bool SendMessage(llvm::StringRef message,
                                  std::string &response_string,
                                  PacketResult expected_result);
  unsigned int GetPcRegisterId();

private:
  LLVM_NODISCARD bool Continue(llvm::StringRef message);
  LLVM_NODISCARD bool GenerateConnectionAddress(std::string &address);
  std::string GenerateLogFileName(const lldb_private::ArchSpec &arch) const;
  std::string FormatFailedResult(
      const std::string &message,
      lldb_private::process_gdb_remote::GDBRemoteCommunication::PacketResult
          result);

  llvm::Optional<ProcessInfo> m_process_info;
  llvm::Optional<StopReply> m_stop_reply;
  lldb_private::ProcessLaunchInfo m_server_process_info;
  std::string m_test_name;
  std::string m_test_case_name;
  unsigned int m_pc_register;
};
} // namespace llgs_tests
