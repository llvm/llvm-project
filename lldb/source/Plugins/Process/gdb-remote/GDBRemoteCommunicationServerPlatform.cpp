//===-- GDBRemoteCommunicationServerPlatform.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GDBRemoteCommunicationServerPlatform.h"

#include <cerrno>

#include <chrono>
#include <csignal>
#include <cstring>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Threading.h"

#include "lldb/Host/Config.h"
#include "lldb/Host/ConnectionFileDescriptor.h"
#include "lldb/Host/FileAction.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Interpreter/CommandCompletions.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/UnixSignals.h"
#include "lldb/Utility/GDBRemote.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/StructuredData.h"
#include "lldb/Utility/TildeExpressionResolver.h"
#include "lldb/Utility/UriParser.h"

#include "lldb/Utility/StringExtractorGDBRemote.h"

using namespace lldb;
using namespace lldb_private::process_gdb_remote;
using namespace lldb_private;

// GDBRemoteCommunicationServerPlatform constructor
GDBRemoteCommunicationServerPlatform::GDBRemoteCommunicationServerPlatform(
    const Socket::SocketProtocol socket_protocol, uint16_t gdbserver_port)
    : GDBRemoteCommunicationServerCommon(), m_socket_protocol(socket_protocol),
      m_gdbserver_port(gdbserver_port) {

  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qC,
      &GDBRemoteCommunicationServerPlatform::Handle_qC);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qGetWorkingDir,
      &GDBRemoteCommunicationServerPlatform::Handle_qGetWorkingDir);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qLaunchGDBServer,
      &GDBRemoteCommunicationServerPlatform::Handle_qLaunchGDBServer);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qQueryGDBServer,
      &GDBRemoteCommunicationServerPlatform::Handle_qQueryGDBServer);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qKillSpawnedProcess,
      &GDBRemoteCommunicationServerPlatform::Handle_qKillSpawnedProcess);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qProcessInfo,
      &GDBRemoteCommunicationServerPlatform::Handle_qProcessInfo);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_qPathComplete,
      &GDBRemoteCommunicationServerPlatform::Handle_qPathComplete);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_QSetWorkingDir,
      &GDBRemoteCommunicationServerPlatform::Handle_QSetWorkingDir);
  RegisterMemberFunctionHandler(
      StringExtractorGDBRemote::eServerPacketType_jSignalsInfo,
      &GDBRemoteCommunicationServerPlatform::Handle_jSignalsInfo);

  RegisterPacketHandler(StringExtractorGDBRemote::eServerPacketType_interrupt,
                        [](StringExtractorGDBRemote packet, Status &error,
                           bool &interrupt, bool &quit) {
                          error = Status::FromErrorString("interrupt received");
                          interrupt = true;
                          return PacketResult::Success;
                        });
}

// Destructor
GDBRemoteCommunicationServerPlatform::~GDBRemoteCommunicationServerPlatform() =
    default;

Status GDBRemoteCommunicationServerPlatform::LaunchGDBServer(
    const lldb_private::Args &args, lldb::pid_t &pid, std::string &socket_name,
    shared_fd_t fd) {
  std::ostringstream url;
  if (fd == SharedSocket::kInvalidFD) {
    if (m_socket_protocol == Socket::ProtocolTcp) {
      // Just check that GDBServer exists. GDBServer must be launched after
      // accepting the connection.
      if (!GetDebugserverPath(nullptr))
        return Status::FromErrorString("unable to locate debugserver");
      return Status();
    }

    // debugserver does not accept the URL scheme prefix.
#if !defined(__APPLE__)
    url << Socket::FindSchemeByProtocol(m_socket_protocol) << "://";
#endif
    socket_name = GetDomainSocketPath("gdbserver").GetPath();
    url << socket_name;
  } else {
    if (m_socket_protocol != Socket::ProtocolTcp)
      return Status::FromErrorString("protocol must be tcp");
  }

  // Spawn a debugserver and try to get the port it listens to.
  ProcessLaunchInfo debugserver_launch_info;
  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOG(log, "Launching debugserver url='{0}', fd={1}...", url.str(), fd);

  // Do not run in a new session so that it can not linger after the platform
  // closes.
  debugserver_launch_info.SetLaunchInSeparateProcessGroup(false);
  debugserver_launch_info.SetMonitorProcessCallback(
      [](lldb::pid_t, int, int) {});

  Status error = StartDebugserverProcess(
      url.str().c_str(), nullptr, debugserver_launch_info, nullptr, &args, fd);

  if (error.Success()) {
    pid = debugserver_launch_info.GetProcessID();
    AddSpawnedProcess(pid);
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerPlatform::%s() "
              "debugserver launched successfully as pid %" PRIu64,
              __FUNCTION__, pid);
  } else {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerPlatform::%s() "
              "debugserver launch failed: %s",
              __FUNCTION__, error.AsCString());
  }
  return error;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerPlatform::Handle_qLaunchGDBServer(
    StringExtractorGDBRemote &packet) {
  // Spawn a local debugserver as a platform so we can then attach or launch a
  // process...

  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOGF(log, "GDBRemoteCommunicationServerPlatform::%s() called",
            __FUNCTION__);

  ConnectionFileDescriptor file_conn;
  std::string hostname;
  packet.SetFilePos(::strlen("qLaunchGDBServer;"));
  llvm::StringRef name;
  llvm::StringRef value;
  std::optional<uint16_t> port;
  while (packet.GetNameColonValue(name, value)) {
    if (name == "host")
      hostname = std::string(value);
    else if (name == "port") {
      // Make the Optional valid so we can use its value
      port = 0;
      value.getAsInteger(0, *port);
    }
  }

  // Ignore client's hostname and the port.

  lldb::pid_t debugserver_pid = LLDB_INVALID_PROCESS_ID;
  std::string socket_name;
  Status error = LaunchGDBServer(Args(), debugserver_pid, socket_name,
                                 SharedSocket::kInvalidFD);
  if (error.Fail())
    return SendErrorResponse(9); // EBADF

  StreamGDBRemote response;
  uint16_t gdbserver_port = socket_name.empty() ? m_gdbserver_port : 0;
  response.Printf("pid:%" PRIu64 ";port:%u;", debugserver_pid, gdbserver_port);
  if (!socket_name.empty()) {
    response.PutCString("socket_name:");
    response.PutStringAsRawHex8(socket_name);
    response.PutChar(';');
  }

  PacketResult packet_result = SendPacketNoLock(response.GetString());
  if (packet_result != PacketResult::Success) {
    if (debugserver_pid != LLDB_INVALID_PROCESS_ID)
      Host::Kill(debugserver_pid, SIGINT);
  }
  return packet_result;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerPlatform::Handle_qQueryGDBServer(
    StringExtractorGDBRemote &packet) {
  namespace json = llvm::json;

  if (!m_pending_gdb_server_socket_name)
    return SendErrorResponse(4);

  json::Object server{{"port", m_pending_gdb_server_socket_name->empty()
                                   ? m_gdbserver_port
                                   : 0}};

  if (!m_pending_gdb_server_socket_name->empty())
    server.try_emplace("socket_name", *m_pending_gdb_server_socket_name);

  json::Array server_list;
  server_list.push_back(std::move(server));

  StreamGDBRemote response;
  response.AsRawOstream() << std::move(server_list);

  StreamGDBRemote escaped_response;
  escaped_response.PutEscapedBytes(response.GetString().data(),
                                   response.GetSize());
  return SendPacketNoLock(escaped_response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerPlatform::Handle_qKillSpawnedProcess(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("qKillSpawnedProcess:"));

  lldb::pid_t pid = packet.GetU64(LLDB_INVALID_PROCESS_ID);

  // verify that we know anything about this pid.
  if (!SpawnedProcessIsRunning(pid)) {
    // not a pid we know about
    return SendErrorResponse(10);
  }

  // go ahead and attempt to kill the spawned process
  if (KillSpawnedProcess(pid))
    return SendOKResponse();
  else
    return SendErrorResponse(11);
}

void GDBRemoteCommunicationServerPlatform::AddSpawnedProcess(lldb::pid_t pid) {
  assert(pid != LLDB_INVALID_PROCESS_ID);
  std::lock_guard<std::recursive_mutex> guard(m_spawned_pids_mutex);
  m_spawned_pids.insert(pid);
}

bool GDBRemoteCommunicationServerPlatform::SpawnedProcessIsRunning(
    lldb::pid_t pid) {
  std::lock_guard<std::recursive_mutex> guard(m_spawned_pids_mutex);
  return (m_spawned_pids.find(pid) != m_spawned_pids.end());
}

bool GDBRemoteCommunicationServerPlatform::KillSpawnedProcess(lldb::pid_t pid) {
  // make sure we know about this process
  if (!SpawnedProcessIsRunning(pid)) {
    // it seems the process has been finished recently
    return true;
  }

  // first try a SIGTERM (standard kill)
  Host::Kill(pid, SIGTERM);

  // check if that worked
  for (size_t i = 0; i < 10; ++i) {
    if (!SpawnedProcessIsRunning(pid)) {
      // it is now killed
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  if (!SpawnedProcessIsRunning(pid))
    return true;

  // the launched process still lives.  Now try killing it again, this time
  // with an unblockable signal.
  Host::Kill(pid, SIGKILL);

  for (size_t i = 0; i < 10; ++i) {
    if (!SpawnedProcessIsRunning(pid)) {
      // it is now killed
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // check one more time after the final sleep
  return !SpawnedProcessIsRunning(pid);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerPlatform::Handle_qProcessInfo(
    StringExtractorGDBRemote &packet) {
  lldb::pid_t pid = m_process_launch_info.GetProcessID();
  m_process_launch_info.Clear();

  if (pid == LLDB_INVALID_PROCESS_ID)
    return SendErrorResponse(1);

  ProcessInstanceInfo proc_info;
  if (!Host::GetProcessInfo(pid, proc_info))
    return SendErrorResponse(1);

  StreamString response;
  CreateProcessInfoResponse_DebugServerStyle(proc_info, response);
  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerPlatform::Handle_qPathComplete(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("qPathComplete:"));
  const bool only_dir = (packet.GetHexMaxU32(false, 0) == 1);
  if (packet.GetChar() != ',')
    return SendErrorResponse(85);
  std::string path;
  packet.GetHexByteString(path);

  StringList matches;
  StandardTildeExpressionResolver resolver;
  if (only_dir)
    CommandCompletions::DiskDirectories(path, matches, resolver);
  else
    CommandCompletions::DiskFiles(path, matches, resolver);

  StreamString response;
  response.PutChar('M');
  llvm::StringRef separator;
  std::sort(matches.begin(), matches.end());
  for (const auto &match : matches) {
    response << separator;
    separator = ",";
    // encode result strings into hex bytes to avoid unexpected error caused by
    // special characters like '$'.
    response.PutStringAsRawHex8(match.c_str());
  }

  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerPlatform::Handle_qGetWorkingDir(
    StringExtractorGDBRemote &packet) {

  llvm::SmallString<64> cwd;
  if (std::error_code ec = llvm::sys::fs::current_path(cwd))
    return SendErrorResponse(ec.value());

  StreamString response;
  response.PutBytesAsRawHex8(cwd.data(), cwd.size());
  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerPlatform::Handle_QSetWorkingDir(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("QSetWorkingDir:"));
  std::string path;
  packet.GetHexByteString(path);

  if (std::error_code ec = llvm::sys::fs::set_current_path(path))
    return SendErrorResponse(ec.value());
  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerPlatform::Handle_qC(
    StringExtractorGDBRemote &packet) {
  // NOTE: lldb should now be using qProcessInfo for process IDs.  This path
  // here
  // should not be used.  It is reporting process id instead of thread id.  The
  // correct answer doesn't seem to make much sense for lldb-platform.
  // CONSIDER: flip to "unsupported".
  lldb::pid_t pid = m_process_launch_info.GetProcessID();

  StreamString response;
  response.Printf("QC%" PRIx64, pid);

  // If we launch a process and this GDB server is acting as a platform, then
  // we need to clear the process launch state so we can start launching
  // another process. In order to launch a process a bunch or packets need to
  // be sent: environment packets, working directory, disable ASLR, and many
  // more settings. When we launch a process we then need to know when to clear
  // this information. Currently we are selecting the 'qC' packet as that
  // packet which seems to make the most sense.
  if (pid != LLDB_INVALID_PROCESS_ID) {
    m_process_launch_info.Clear();
  }

  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerPlatform::Handle_jSignalsInfo(
    StringExtractorGDBRemote &packet) {
  StructuredData::Array signal_array;

  lldb::UnixSignalsSP signals = UnixSignals::CreateForHost();
  for (auto signo = signals->GetFirstSignalNumber();
       signo != LLDB_INVALID_SIGNAL_NUMBER;
       signo = signals->GetNextSignalNumber(signo)) {
    auto dictionary = std::make_shared<StructuredData::Dictionary>();

    dictionary->AddIntegerItem("signo", signo);
    dictionary->AddStringItem("name", signals->GetSignalAsStringRef(signo));

    bool suppress, stop, notify;
    signals->GetSignalInfo(signo, suppress, stop, notify);
    dictionary->AddBooleanItem("suppress", suppress);
    dictionary->AddBooleanItem("stop", stop);
    dictionary->AddBooleanItem("notify", notify);

    signal_array.Push(dictionary);
  }

  StreamString response;
  signal_array.Dump(response);
  return SendPacketNoLock(response.GetString());
}

void GDBRemoteCommunicationServerPlatform::DebugserverProcessReaped(
    lldb::pid_t pid) {
  std::lock_guard<std::recursive_mutex> guard(m_spawned_pids_mutex);
  m_spawned_pids.erase(pid);
}

Status GDBRemoteCommunicationServerPlatform::LaunchProcess() {
  if (!m_process_launch_info.GetArguments().GetArgumentCount())
    return Status::FromErrorStringWithFormat(
        "%s: no process command line specified to launch", __FUNCTION__);

  // specify the process monitor if not already set.  This should generally be
  // what happens since we need to reap started processes.
  if (!m_process_launch_info.GetMonitorProcessCallback())
    m_process_launch_info.SetMonitorProcessCallback(std::bind(
        &GDBRemoteCommunicationServerPlatform::DebugserverProcessReaped, this,
        std::placeholders::_1));

  Status error = Host::LaunchProcess(m_process_launch_info);
  if (!error.Success()) {
    fprintf(stderr, "%s: failed to launch executable %s", __FUNCTION__,
            m_process_launch_info.GetArguments().GetArgumentAtIndex(0));
    return error;
  }

  printf("Launched '%s' as process %" PRIu64 "...\n",
         m_process_launch_info.GetArguments().GetArgumentAtIndex(0),
         m_process_launch_info.GetProcessID());

  // add to list of spawned processes.  On an lldb-gdbserver, we would expect
  // there to be only one.
  const auto pid = m_process_launch_info.GetProcessID();
  AddSpawnedProcess(pid);

  return error;
}

const FileSpec &GDBRemoteCommunicationServerPlatform::GetDomainSocketDir() {
  static FileSpec g_domainsocket_dir;
  static llvm::once_flag g_once_flag;

  llvm::call_once(g_once_flag, []() {
    const char *domainsocket_dir_env =
        ::getenv("LLDB_DEBUGSERVER_DOMAINSOCKET_DIR");
    if (domainsocket_dir_env != nullptr)
      g_domainsocket_dir = FileSpec(domainsocket_dir_env);
    else
      g_domainsocket_dir = HostInfo::GetProcessTempDir();
  });

  return g_domainsocket_dir;
}

FileSpec
GDBRemoteCommunicationServerPlatform::GetDomainSocketPath(const char *prefix) {
  llvm::SmallString<128> socket_path;
  llvm::SmallString<128> socket_name(
      (llvm::StringRef(prefix) + ".%%%%%%").str());

  FileSpec socket_path_spec(GetDomainSocketDir());
  socket_path_spec.AppendPathComponent(socket_name.c_str());

  llvm::sys::fs::createUniqueFile(socket_path_spec.GetPath().c_str(),
                                  socket_path);
  return FileSpec(socket_path.c_str());
}

void GDBRemoteCommunicationServerPlatform::SetPendingGdbServer(
    const std::string &socket_name) {
  m_pending_gdb_server_socket_name = socket_name;
}
