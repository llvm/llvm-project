//===-- TestClient.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TestClient.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/common/TCPSocket.h"
#include "lldb/Host/posix/ConnectionFileDescriptorPosix.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Target/ProcessLaunchInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Path.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <cstdlib>
#include <future>
#include <sstream>
#include <string>

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

namespace llgs_tests {

TestClient::TestClient(std::unique_ptr<Connection> Conn) {
  SetConnection(Conn.release());

  SendAck(); // Send this as a handshake.
}

TestClient::~TestClient() {
  if (!IsConnected())
    return;

  std::string response;
  // Debugserver (non-conformingly?) sends a reply to the k packet instead of
  // simply closing the connection.
  PacketResult result =
      IsDebugServer() ? PacketResult::Success : PacketResult::ErrorDisconnected;
  EXPECT_THAT_ERROR(SendMessage("k", response, result), Succeeded());
}

Expected<std::unique_ptr<TestClient>> TestClient::launch(StringRef Log) {
  return launch(Log, {});
}

Expected<std::unique_ptr<TestClient>> TestClient::launch(StringRef Log, ArrayRef<StringRef> InferiorArgs) {
  return launchCustom(Log, {}, InferiorArgs);
}

Expected<std::unique_ptr<TestClient>> TestClient::launchCustom(StringRef Log, ArrayRef<StringRef> ServerArgs, ArrayRef<StringRef> InferiorArgs) {
  const ArchSpec &arch_spec = HostInfo::GetArchitecture();
  Args args;
  args.AppendArgument(LLDB_SERVER);
  if (IsLldbServer())
    args.AppendArgument("gdbserver");
  args.AppendArgument("--reverse-connect");

  if (!Log.empty()) {
    args.AppendArgument(("--log-file=" + Log).str());
    if (IsLldbServer())
      args.AppendArgument("--log-channels=gdb-remote packets");
    else
      args.AppendArgument("--log-flags=0x800000");
  }

  Status status;
  TCPSocket listen_socket(true, false);
  status = listen_socket.Listen("127.0.0.1:0", 5);
  if (status.Fail())
    return status.ToError();

  args.AppendArgument(
      ("localhost:" + Twine(listen_socket.GetLocalPortNumber())).str());

  for (StringRef arg : ServerArgs)
    args.AppendArgument(arg);

  if (!InferiorArgs.empty()) {
    args.AppendArgument("--");
    for (StringRef arg : InferiorArgs)
      args.AppendArgument(arg);
  }

  ProcessLaunchInfo Info;
  Info.SetArchitecture(arch_spec);
  Info.SetArguments(args, true);

  StringList Env;
  Host::GetEnvironment(Env);
  Info.GetEnvironmentEntries() = Args(Env);

  status = Host::LaunchProcess(Info);
  if (status.Fail())
    return status.ToError();

  Socket *accept_socket;
  listen_socket.Accept(accept_socket);
  auto Conn = llvm::make_unique<ConnectionFileDescriptor>(accept_socket);
  auto Client = std::unique_ptr<TestClient>(new TestClient(std::move(Conn)));

  if (!InferiorArgs.empty()) {
    if (Error E = Client->QueryProcessInfo())
      return std::move(E);
  }

  return std::move(Client);
}

Error TestClient::SetInferior(llvm::ArrayRef<std::string> inferior_args) {
  StringList env;
  Host::GetEnvironment(env);
  for (size_t i = 0; i < env.GetSize(); ++i) {
    if (SendEnvironmentPacket(env[i].c_str()) != 0) {
      return make_error<StringError>(
          formatv("Failed to set environment variable: {0}", env[i]).str(),
          inconvertibleErrorCode());
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

  if (Error E = SendMessage(command.str()))
    return E;
  if (Error E = SendMessage("qLaunchSuccess"))
    return E;
  if (Error E = QueryProcessInfo())
    return E;
  return Error::success();
}

Error TestClient::ListThreadsInStopReply() {
  return SendMessage("QListThreadsInStopReply");
}

Error TestClient::SetBreakpoint(unsigned long address) {
  return SendMessage(formatv("Z0,{0:x-},1", address).str());
}

Error TestClient::ContinueAll() { return Continue("vCont;c"); }

Error TestClient::ContinueThread(unsigned long thread_id) {
  return Continue(formatv("vCont;c:{0:x-}", thread_id).str());
}

const ProcessInfo &TestClient::GetProcessInfo() { return *m_process_info; }

Optional<JThreadsInfo> TestClient::GetJThreadsInfo() {
  std::string response;
  if (SendMessage("jThreadsInfo", response))
    return llvm::None;
  auto creation = JThreadsInfo::Create(response, m_process_info->GetEndian());
  if (auto create_error = creation.takeError()) {
    GTEST_LOG_(ERROR) << toString(std::move(create_error));
    return llvm::None;
  }

  return std::move(*creation);
}

const StopReply &TestClient::GetLatestStopReply() {
  assert(m_stop_reply);
  return *m_stop_reply;
}

Error TestClient::SendMessage(StringRef message) {
  std::string dummy_string;
  return SendMessage(message, dummy_string);
}

Error TestClient::SendMessage(StringRef message, std::string &response_string) {
  if (Error E = SendMessage(message, response_string, PacketResult::Success))
    return E;
  if (response_string[0] == 'E') {
    return make_error<StringError>(
        formatv("Error `{0}` while sending message: {1}", response_string,
                message)
            .str(),
        inconvertibleErrorCode());
  }
  return Error::success();
}

Error TestClient::SendMessage(StringRef message, std::string &response_string,
                              PacketResult expected_result) {
  StringExtractorGDBRemote response;
  GTEST_LOG_(INFO) << "Send Packet: " << message.str();
  PacketResult result = SendPacketAndWaitForResponse(message, response, false);
  response.GetEscapedBinaryData(response_string);
  GTEST_LOG_(INFO) << "Read Packet: " << response_string;
  if (result != expected_result)
    return make_error<StringError>(
        formatv("Error sending message `{0}`: {1}", message, result).str(),
        inconvertibleErrorCode());

  return Error::success();
}

unsigned int TestClient::GetPcRegisterId() {
  if (m_pc_register != UINT_MAX)
    return m_pc_register;

  for (unsigned int register_id = 0;; register_id++) {
    std::string message = formatv("qRegisterInfo{0:x-}", register_id).str();
    std::string response;
    if (SendMessage(message, response)) {
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

llvm::Error TestClient::QueryProcessInfo() {
  std::string response;
  if (Error E = SendMessage("qProcessInfo", response))
    return E;
  auto create_or_error = ProcessInfo::Create(response);
  if (!create_or_error)
    return create_or_error.takeError();
  m_process_info = *create_or_error;
  return Error::success();
}

Error TestClient::Continue(StringRef message) {
  assert(m_process_info.hasValue());

  std::string response;
  if (Error E = SendMessage(message, response))
    return E;
  auto creation = StopReply::create(response, m_process_info->GetEndian());
  if (Error E = creation.takeError())
    return E;

  m_stop_reply = std::move(*creation);
  if (!isa<StopReplyStop>(m_stop_reply)) {
    StringExtractorGDBRemote R;
    PacketResult result = ReadPacket(R, GetPacketTimeout(), false);
    if (result != PacketResult::ErrorDisconnected) {
      return make_error<StringError>(
          formatv("Expected connection close after receiving {0}. Got {1}/{2} "
                  "instead.",
                  response, result, R.GetStringRef())
              .str(),
          inconvertibleErrorCode());
    }
  }
  return Error::success();
}

} // namespace llgs_tests
