//===-- TestClient.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SERVER_TESTS_TESTCLIENT_H
#define LLDB_SERVER_TESTS_TESTCLIENT_H

#include "MessageObjects.h"
#include "Plugins/Process/gdb-remote/GDBRemoteCommunicationClient.h"
#include "lldb/Target/ProcessLaunchInfo.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/Connection.h"
#include "llvm/ADT/Optional.h"
#include <memory>
#include <string>

namespace llgs_tests {
class TestClient
    : public lldb_private::process_gdb_remote::GDBRemoteCommunicationClient {
public:
  static bool IsDebugServer();
  static bool IsLldbServer();

  /// Launches the server, connects it to the client and returns the client. If
  /// Log is non-empty, the server will write it's log to this file.
  static llvm::Expected<std::unique_ptr<TestClient>> launch(llvm::StringRef Log);

  /// Launches the server, while specifying the inferior on its command line.
  /// When the client connects, it already has a process ready.
  static llvm::Expected<std::unique_ptr<TestClient>>
  launch(llvm::StringRef Log, llvm::ArrayRef<llvm::StringRef> InferiorArgs);

  ~TestClient() override;
  llvm::Error SetInferior(llvm::ArrayRef<std::string> inferior_args);
  llvm::Error ListThreadsInStopReply();
  llvm::Error SetBreakpoint(unsigned long address);
  llvm::Error ContinueAll();
  llvm::Error ContinueThread(unsigned long thread_id);
  const ProcessInfo &GetProcessInfo();
  llvm::Optional<JThreadsInfo> GetJThreadsInfo();
  const StopReply &GetLatestStopReply();
  llvm::Error SendMessage(llvm::StringRef message);
  llvm::Error SendMessage(llvm::StringRef message,
                          std::string &response_string);
  llvm::Error SendMessage(llvm::StringRef message, std::string &response_string,
                          PacketResult expected_result);
  unsigned int GetPcRegisterId();

private:
  TestClient(std::unique_ptr<lldb_private::Connection> Conn);

  llvm::Error QueryProcessInfo();
  llvm::Error Continue(llvm::StringRef message);
  std::string FormatFailedResult(
      const std::string &message,
      lldb_private::process_gdb_remote::GDBRemoteCommunication::PacketResult
          result);

  llvm::Optional<ProcessInfo> m_process_info;
  llvm::Optional<StopReply> m_stop_reply;
  unsigned int m_pc_register = UINT_MAX;
};

} // namespace llgs_tests

#endif // LLDB_SERVER_TESTS_TESTCLIENT_H
