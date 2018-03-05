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
using namespace llgs_tests;

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
  Info.GetEnvironment() = Host::GetEnvironment();

  status = Host::LaunchProcess(Info);
  if (status.Fail())
    return status.ToError();

  Socket *accept_socket;
  listen_socket.Accept(accept_socket);
  auto Conn = llvm::make_unique<ConnectionFileDescriptor>(accept_socket);
  auto Client = std::unique_ptr<TestClient>(new TestClient(std::move(Conn)));

  if (!InferiorArgs.empty()) {
    if (Error E = Client->queryProcess())
      return std::move(E);
  }

  return std::move(Client);
}

Error TestClient::SetInferior(llvm::ArrayRef<std::string> inferior_args) {
  if (SendEnvironment(Host::GetEnvironment()) != 0) {
    return make_error<StringError>("Failed to set launch environment",
                                   inconvertibleErrorCode());
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
  if (Error E = queryProcess())
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

const llgs_tests::ProcessInfo &TestClient::GetProcessInfo() {
  return *m_process_info;
}

Expected<JThreadsInfo> TestClient::GetJThreadsInfo() {
  return SendMessage<JThreadsInfo>("jThreadsInfo", m_register_infos);
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
  assert(m_pc_register != LLDB_INVALID_REGNUM);
  return m_pc_register;
}

Error TestClient::qProcessInfo() {
  m_process_info = None;
  auto InfoOr = SendMessage<ProcessInfo>("qProcessInfo");
  if (!InfoOr)
    return InfoOr.takeError();
  m_process_info = std::move(*InfoOr);
  return Error::success();
}

Error TestClient::qRegisterInfos() {
  for (unsigned int Reg = 0;; ++Reg) {
    std::string Message = formatv("qRegisterInfo{0:x-}", Reg).str();
    Expected<RegisterInfo> InfoOr = SendMessage<RegisterInfoParser>(Message);
    if (!InfoOr) {
      consumeError(InfoOr.takeError());
      break;
    }
    m_register_infos.emplace_back(std::move(*InfoOr));
    if (m_register_infos[Reg].kinds[eRegisterKindGeneric] ==
        LLDB_REGNUM_GENERIC_PC)
      m_pc_register = Reg;
  }
  if (m_pc_register == LLDB_INVALID_REGNUM)
    return make_parsing_error("qRegisterInfo: generic");
  return Error::success();
}

Error TestClient::queryProcess() {
  if (Error E = qProcessInfo())
    return E;
  if (Error E = qRegisterInfos())
    return E;
  return Error::success();
}

Error TestClient::Continue(StringRef message) {
  assert(m_process_info.hasValue());

  auto StopReplyOr = SendMessage<StopReply>(
      message, m_process_info->GetEndian(), m_register_infos);
  if (!StopReplyOr)
    return StopReplyOr.takeError();

  m_stop_reply = std::move(*StopReplyOr);
  if (!isa<StopReplyStop>(m_stop_reply)) {
    StringExtractorGDBRemote R;
    PacketResult result = ReadPacket(R, GetPacketTimeout(), false);
    if (result != PacketResult::ErrorDisconnected) {
      return make_error<StringError>(
          formatv("Expected connection close after sending {0}. Got {1}/{2} "
                  "instead.",
                  message, result, R.GetStringRef())
              .str(),
          inconvertibleErrorCode());
    }
  }
  return Error::success();
}
