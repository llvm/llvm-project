//===-- TestClient.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TestClient.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/common/TCPSocket.h"
#include "lldb/Host/posix/ConnectionFileDescriptorPosix.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Target/ProcessLaunchInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Path.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <future>
#include <sstream>
#include <string>

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

namespace llgs_tests {
void TestClient::Initialize() { HostInfo::Initialize(); }

bool TestClient::IsDebugServer() {
  return sys::path::filename(LLDB_SERVER).contains("debugserver");
}

bool TestClient::IsLldbServer() { return !IsDebugServer(); }

TestClient::TestClient(const std::string &test_name,
                       const std::string &test_case_name)
    : m_test_name(test_name), m_test_case_name(test_case_name),
      m_pc_register(UINT_MAX) {}

TestClient::~TestClient() {}

bool TestClient::StartDebugger() {
  const ArchSpec &arch_spec = HostInfo::GetArchitecture();
  Args args;
  args.AppendArgument(LLDB_SERVER);
  if (IsLldbServer()) {
    args.AppendArgument("gdbserver");
    args.AppendArgument("--log-channels=gdb-remote packets");
  } else {
    args.AppendArgument("--log-flags=0x800000");
  }
  args.AppendArgument("--reverse-connect");
  std::string log_file_name = GenerateLogFileName(arch_spec);
  if (log_file_name.size())
    args.AppendArgument("--log-file=" + log_file_name);

  Status error;
  TCPSocket listen_socket(true, false);
  error = listen_socket.Listen("127.0.0.1:0", 5);
  if (error.Fail()) {
    GTEST_LOG_(ERROR) << "Unable to open listen socket.";
    return false;
  }

  char connect_remote_address[64];
  snprintf(connect_remote_address, sizeof(connect_remote_address),
           "localhost:%u", listen_socket.GetLocalPortNumber());

  args.AppendArgument(connect_remote_address);

  m_server_process_info.SetArchitecture(arch_spec);
  m_server_process_info.SetArguments(args, true);
  Status status = Host::LaunchProcess(m_server_process_info);
  if (status.Fail()) {
    GTEST_LOG_(ERROR)
        << formatv("Failure to launch lldb server: {0}.", status).str();
    return false;
  }

  char connect_remote_uri[64];
  snprintf(connect_remote_uri, sizeof(connect_remote_uri), "connect://%s",
           connect_remote_address);
  Socket *accept_socket;
  listen_socket.Accept(accept_socket);
  SetConnection(new ConnectionFileDescriptor(accept_socket));

  SendAck(); // Send this as a handshake.
  return true;
}

bool TestClient::StopDebugger() {
  std::string response;
  // Debugserver (non-conformingly?) sends a reply to the k packet instead of
  // simply closing the connection.
  PacketResult result =
      IsDebugServer() ? PacketResult::Success : PacketResult::ErrorDisconnected;
  return SendMessage("k", response, result);
}

bool TestClient::SetInferior(llvm::ArrayRef<std::string> inferior_args) {
  StringList env;
  Host::GetEnvironment(env);
  for (size_t i = 0; i < env.GetSize(); ++i) {
    if (SendEnvironmentPacket(env[i].c_str()) != 0) {
      GTEST_LOG_(ERROR) << "failed to set environment variable `" << env[i] << "`";
      return false;
    }
  }
  std::stringstream command;
  command << "A";
  for (size_t i = 0; i < inferior_args.size(); i++) {
    if (i > 0)
      command << ',';
    std::string hex_encoded = toHex(inferior_args[i]);
    command << hex_encoded.size() << ',' << i << ',' << hex_encoded;
  }

  if (!SendMessage(command.str()))
    return false;
  if (!SendMessage("qLaunchSuccess"))
    return false;
  std::string response;
  if (!SendMessage("qProcessInfo", response))
    return false;
  auto create_or_error = ProcessInfo::Create(response);
  if (auto create_error = create_or_error.takeError()) {
    GTEST_LOG_(ERROR) << toString(std::move(create_error));
    return false;
  }

  m_process_info = *create_or_error;
  return true;
}

bool TestClient::ListThreadsInStopReply() {
  return SendMessage("QListThreadsInStopReply");
}

bool TestClient::SetBreakpoint(unsigned long address) {
  std::stringstream command;
  command << "Z0," << std::hex << address << ",1";
  return SendMessage(command.str());
}

bool TestClient::ContinueAll() { return Continue("vCont;c"); }

bool TestClient::ContinueThread(unsigned long thread_id) {
  return Continue(formatv("vCont;c:{0:x-}", thread_id).str());
}

const ProcessInfo &TestClient::GetProcessInfo() { return *m_process_info; }

Optional<JThreadsInfo> TestClient::GetJThreadsInfo() {
  std::string response;
  if (!SendMessage("jThreadsInfo", response))
    return llvm::None;
  auto creation = JThreadsInfo::Create(response, m_process_info->GetEndian());
  if (auto create_error = creation.takeError()) {
    GTEST_LOG_(ERROR) << toString(std::move(create_error));
    return llvm::None;
  }

  return std::move(*creation);
}

const StopReply &TestClient::GetLatestStopReply() {
  return m_stop_reply.getValue();
}

bool TestClient::SendMessage(StringRef message) {
  std::string dummy_string;
  return SendMessage(message, dummy_string);
}

bool TestClient::SendMessage(StringRef message, std::string &response_string) {
  if (!SendMessage(message, response_string, PacketResult::Success))
    return false;
  else if (response_string[0] == 'E') {
    GTEST_LOG_(ERROR) << "Error " << response_string
                      << " while sending message: " << message.str();
    return false;
  }

  return true;
}

bool TestClient::SendMessage(StringRef message, std::string &response_string,
                             PacketResult expected_result) {
  StringExtractorGDBRemote response;
  GTEST_LOG_(INFO) << "Send Packet: " << message.str();
  PacketResult result = SendPacketAndWaitForResponse(message, response, false);
  response.GetEscapedBinaryData(response_string);
  GTEST_LOG_(INFO) << "Read Packet: " << response_string;
  if (result != expected_result) {
    GTEST_LOG_(ERROR) << FormatFailedResult(message, result);
    return false;
  }

  return true;
}

unsigned int TestClient::GetPcRegisterId() {
  if (m_pc_register != UINT_MAX)
    return m_pc_register;

  for (unsigned int register_id = 0;; register_id++) {
    std::string message = formatv("qRegisterInfo{0:x-}", register_id).str();
    std::string response;
    if (!SendMessage(message, response)) {
      GTEST_LOG_(ERROR) << "Unable to query register ID for PC register.";
      return UINT_MAX;
    }

    auto elements_or_error = SplitUniquePairList("GetPcRegisterId", response);
    if (auto split_error = elements_or_error.takeError()) {
      GTEST_LOG_(ERROR) << "GetPcRegisterId: Error splitting response: "
                        << response;
      return UINT_MAX;
    }

    auto elements = *elements_or_error;
    if (elements["alt-name"] == "pc" || elements["generic"] == "pc") {
      m_pc_register = register_id;
      break;
    }
  }

  return m_pc_register;
}

bool TestClient::Continue(StringRef message) {
  if (!m_process_info.hasValue()) {
    GTEST_LOG_(ERROR) << "Continue() called before m_process_info initialized.";
    return false;
  }

  std::string response;
  if (!SendMessage(message, response))
    return false;
  auto creation = StopReply::Create(response, m_process_info->GetEndian());
  if (auto create_error = creation.takeError()) {
    GTEST_LOG_(ERROR) << toString(std::move(create_error));
    return false;
  }

  m_stop_reply = std::move(*creation);
  return true;
}

std::string TestClient::GenerateLogFileName(const ArchSpec &arch) const {
  char *log_directory = getenv("LOG_FILE_DIRECTORY");
  if (!log_directory)
    return "";

  if (!llvm::sys::fs::is_directory(log_directory)) {
    GTEST_LOG_(WARNING) << "Cannot access log directory: " << log_directory;
    return "";
  }

  std::string log_file_name;
  raw_string_ostream log_file(log_file_name);
  log_file << log_directory << "/lldb-" << m_test_case_name << '-'
           << m_test_name << '-' << arch.GetArchitectureName() << ".log";
  return log_file.str();
}

std::string TestClient::FormatFailedResult(const std::string &message,
                                           PacketResult result) {
  std::string formatted_error;
  raw_string_ostream error_stream(formatted_error);
  error_stream << "Failure sending message: " << message << " Result: ";

  switch (result) {
  case PacketResult::ErrorSendFailed:
    error_stream << "ErrorSendFailed";
    break;
  case PacketResult::ErrorSendAck:
    error_stream << "ErrorSendAck";
    break;
  case PacketResult::ErrorReplyFailed:
    error_stream << "ErrorReplyFailed";
    break;
  case PacketResult::ErrorReplyTimeout:
    error_stream << "ErrorReplyTimeout";
    break;
  case PacketResult::ErrorReplyInvalid:
    error_stream << "ErrorReplyInvalid";
    break;
  case PacketResult::ErrorReplyAck:
    error_stream << "ErrorReplyAck";
    break;
  case PacketResult::ErrorDisconnected:
    error_stream << "ErrorDisconnected";
    break;
  case PacketResult::ErrorNoSequenceLock:
    error_stream << "ErrorNoSequenceLock";
    break;
  default:
    error_stream << "Unknown Error";
  }

  error_stream.str();
  return formatted_error;
}
} // namespace llgs_tests
