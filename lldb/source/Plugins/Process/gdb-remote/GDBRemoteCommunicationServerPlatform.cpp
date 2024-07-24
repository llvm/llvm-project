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

#include "llvm/Support/JSON.h"
#include "llvm/Support/Threading.h"

#include "lldb/Host/Config.h"
#include "lldb/Host/ConnectionFileDescriptor.h"
#include "lldb/Host/FileAction.h"
#include "lldb/Host/FileSystem.h"
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

// Copy assignment operator to avoid copying m_mutex
GDBRemoteCommunicationServerPlatform::PortMap &
GDBRemoteCommunicationServerPlatform::PortMap::operator=(
    const GDBRemoteCommunicationServerPlatform::PortMap &o) {
  m_port_map = std::move(o.m_port_map);
  return *this;
}

GDBRemoteCommunicationServerPlatform::PortMap::PortMap(uint16_t min_port,
                                                       uint16_t max_port)
    : m_mutex() {
  assert(min_port);
  for (; min_port < max_port; ++min_port)
    m_port_map[min_port] = LLDB_INVALID_PROCESS_ID;
}

void GDBRemoteCommunicationServerPlatform::PortMap::AllowPort(uint16_t port) {
  assert(port);
  // Do not modify existing mappings
  std::lock_guard<std::mutex> guard(m_mutex);
  m_port_map.insert({port, LLDB_INVALID_PROCESS_ID});
}

llvm::Expected<uint16_t>
GDBRemoteCommunicationServerPlatform::PortMap::GetNextAvailablePort() {
  std::lock_guard<std::mutex> guard(m_mutex);
  if (m_port_map.empty())
    return 0; // Bind to port zero and get a port, we didn't have any
              // limitations

  for (auto &pair : m_port_map) {
    if (pair.second == LLDB_INVALID_PROCESS_ID) {
      pair.second = ~(lldb::pid_t)LLDB_INVALID_PROCESS_ID;
      return pair.first;
    }
  }
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "No free port found in port map");
}

bool GDBRemoteCommunicationServerPlatform::PortMap::AssociatePortWithProcess(
    uint16_t port, lldb::pid_t pid) {
  std::lock_guard<std::mutex> guard(m_mutex);
  auto pos = m_port_map.find(port);
  if (pos != m_port_map.end()) {
    pos->second = pid;
    return true;
  }
  return false;
}

bool GDBRemoteCommunicationServerPlatform::PortMap::FreePort(uint16_t port) {
  std::lock_guard<std::mutex> guard(m_mutex);
  std::map<uint16_t, lldb::pid_t>::iterator pos = m_port_map.find(port);
  if (pos != m_port_map.end()) {
    pos->second = LLDB_INVALID_PROCESS_ID;
    return true;
  }
  return false;
}

bool GDBRemoteCommunicationServerPlatform::PortMap::FreePortForProcess(
    lldb::pid_t pid) {
  std::lock_guard<std::mutex> guard(m_mutex);
  if (!m_port_map.empty()) {
    for (auto &pair : m_port_map) {
      if (pair.second == pid) {
        pair.second = LLDB_INVALID_PROCESS_ID;
        return true;
      }
    }
  }
  return false;
}

bool GDBRemoteCommunicationServerPlatform::PortMap::empty() const {
  std::lock_guard<std::mutex> guard(m_mutex);
  return m_port_map.empty();
}

GDBRemoteCommunicationServerPlatform::PortMap
    GDBRemoteCommunicationServerPlatform::g_port_map;
std::set<lldb::pid_t> GDBRemoteCommunicationServerPlatform::g_spawned_pids;
std::mutex GDBRemoteCommunicationServerPlatform::g_spawned_pids_mutex;

// GDBRemoteCommunicationServerPlatform constructor
GDBRemoteCommunicationServerPlatform::GDBRemoteCommunicationServerPlatform(
    const Socket::SocketProtocol socket_protocol, const char *socket_scheme,
    const lldb_private::Args &args, uint16_t port_offset)
    : GDBRemoteCommunicationServerCommon(), m_socket_protocol(socket_protocol),
      m_socket_scheme(socket_scheme), m_inferior_arguments(args),
      m_port_offset(port_offset) {
  m_pending_gdb_server.pid = LLDB_INVALID_PROCESS_ID;
  m_pending_gdb_server.port = 0;

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
                          error.SetErrorString("interrupt received");
                          interrupt = true;
                          return PacketResult::Success;
                        });
}

// Destructor
GDBRemoteCommunicationServerPlatform::~GDBRemoteCommunicationServerPlatform() =
    default;

lldb::thread_result_t GDBRemoteCommunicationServerPlatform::ThreadProc() {
  // We need a virtual working directory per thread.
  FileSystem::InitializePerThread();

  Log *log = GetLog(LLDBLog::Platform);

  if (IsConnected()) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerPlatform::%s() "
              "Thread started...",
              __FUNCTION__);

    if (m_inferior_arguments.GetArgumentCount() > 0) {
      lldb::pid_t pid = LLDB_INVALID_PROCESS_ID;
      std::optional<uint16_t> port;
      std::string socket_name;
      Status error = LaunchGDBServer(m_inferior_arguments,
                                     "", // hostname
                                     pid, port, socket_name);
      if (error.Success())
        SetPendingGdbServer(pid, *port, socket_name);
    }

    bool interrupt = false;
    bool done = false;
    Status error;
    while (!interrupt && !done) {
      if (GetPacketAndSendResponse(std::nullopt, error, interrupt, done) !=
          GDBRemoteCommunication::PacketResult::Success)
        break;
    }

    if (error.Fail()) {
      LLDB_LOGF(log,
                "GDBRemoteCommunicationServerPlatform::%s() "
                "GetPacketAndSendResponse: %s",
                __FUNCTION__, error.AsCString());
    }
  }

  LLDB_LOGF(log,
            "GDBRemoteCommunicationServerPlatform::%s() "
            "Disconnected. Killing child processes...",
            __FUNCTION__);
  for (lldb::pid_t pid : m_spawned_pids)
    KillSpawnedProcess(pid);

  // Do do not wait for child processes. See comments in
  // DebugserverProcessReaped() for details.

  FileSystem::Terminate();

  LLDB_LOGF(log,
            "GDBRemoteCommunicationServerPlatform::%s() "
            "Thread exited.",
            __FUNCTION__);

  delete this;
  return {};
}

Status GDBRemoteCommunicationServerPlatform::LaunchGDBServer(
    const lldb_private::Args &args, std::string hostname, lldb::pid_t &pid,
    std::optional<uint16_t> &port, std::string &socket_name) {
  if (!port) {
    llvm::Expected<uint16_t> available_port = g_port_map.GetNextAvailablePort();
    if (available_port)
      port = *available_port;
    else
      return Status(available_port.takeError());
  }

  // Spawn a new thread to accept the port that gets bound after binding to
  // port 0 (zero).

  // ignore the hostname send from the remote end, just use the ip address that
  // we're currently communicating with as the hostname

  // Spawn a debugserver and try to get the port it listens to.
  ProcessLaunchInfo debugserver_launch_info;
  if (hostname.empty())
    hostname = "127.0.0.1";

  auto cwd = FileSystem::Instance()
                 .GetVirtualFileSystem()
                 ->getCurrentWorkingDirectory();
  if (cwd)
    debugserver_launch_info.SetWorkingDirectory(FileSpec(*cwd));

  // Do not run in a new session so that it can not linger after the platform
  // closes.
  debugserver_launch_info.SetLaunchInSeparateProcessGroup(false);
  debugserver_launch_info.SetMonitorProcessCallback(
      &GDBRemoteCommunicationServerPlatform::DebugserverProcessReaped);

  std::ostringstream url;
// debugserver does not accept the URL scheme prefix.
#if !defined(__APPLE__)
  url << m_socket_scheme << "://";
#endif
  uint16_t child_port = *port;
  uint16_t *port_ptr = &child_port;
  if (m_socket_protocol == Socket::ProtocolTcp) {
    std::string platform_uri = GetConnection()->GetURI();
    std::optional<URI> parsed_uri = URI::Parse(platform_uri);
    url << '[' << parsed_uri->hostname.str() << "]:" << *port;
  } else {
    socket_name = GetDomainSocketPath("gdbserver").GetPath();
    url << socket_name;
    port_ptr = nullptr;
  }

  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOGF(log,
            "GDBRemoteCommunicationServerPlatform::%s() "
            "Host %s launching debugserver with: %s...",
            __FUNCTION__, hostname.c_str(), url.str().c_str());

  Status error = StartDebugserverProcess(
      url.str().c_str(), nullptr, debugserver_launch_info, port_ptr, &args, -1);

  pid = debugserver_launch_info.GetProcessID();

  if (error.Success()) {
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

  // TODO: Be sure gdbserver uses the requested port.
  // assert(!port_ptr || *port == 0 || *port == child_port)
  // Use only the original *port returned by GetNextAvailablePort()
  // for AssociatePortWithProcess() or FreePort() below.

  if (pid != LLDB_INVALID_PROCESS_ID) {
    AddSpawnedProcess(pid);
    if (*port > 0)
      g_port_map.AssociatePortWithProcess(*port, pid);
  } else {
    if (*port > 0)
      g_port_map.FreePort(*port);
  }
  if (port_ptr)
    *port = child_port;
  return error;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerPlatform::Handle_qLaunchGDBServer(
    StringExtractorGDBRemote &packet) {
  // Spawn a local debugserver as a platform so we can then attach or launch a
  // process...

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

  lldb::pid_t debugserver_pid = LLDB_INVALID_PROCESS_ID;
  std::string socket_name;
  Status error =
      LaunchGDBServer(Args(), hostname, debugserver_pid, port, socket_name);
  if (error.Fail()) {
    return SendErrorResponse(9);
  }

  StreamGDBRemote response;
  assert(port);
  response.Printf("pid:%" PRIu64 ";port:%u;", debugserver_pid,
                  *port + m_port_offset);
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

  if (m_pending_gdb_server.pid == LLDB_INVALID_PROCESS_ID)
    return SendErrorResponse(4);

  json::Object server{{"port", m_pending_gdb_server.port}};

  if (!m_pending_gdb_server.socket_name.empty())
    server.try_emplace("socket_name", m_pending_gdb_server.socket_name);

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

  if (SpawnedProcessFinished(pid))
    m_spawned_pids.erase(pid);

  // verify that we know anything about this pid. Scope for locker
  if ((m_spawned_pids.find(pid) == m_spawned_pids.end())) {
    // not a pid we know about
    return SendErrorResponse(10); // ECHILD
  }

  // go ahead and attempt to kill the spawned process
  if (KillSpawnedProcess(pid)) {
    m_spawned_pids.erase(pid);
    return SendOKResponse();
  } else
    return SendErrorResponse(11); // EDEADLK
}

void GDBRemoteCommunicationServerPlatform::AddSpawnedProcess(lldb::pid_t pid) {
  std::lock_guard<std::mutex> guard(g_spawned_pids_mutex);

  // If MonitorChildProcessThreadFunction() failed hope the system will not
  // reuse pid of zombie processes.
  // assert(g_spawned_pids.find(pid) == g_spawned_pids.end());

  g_spawned_pids.insert(pid);
  m_spawned_pids.insert(pid);
}

bool GDBRemoteCommunicationServerPlatform::SpawnedProcessFinished(
    lldb::pid_t pid) {
  std::lock_guard<std::mutex> guard(g_spawned_pids_mutex);
  return (g_spawned_pids.find(pid) == g_spawned_pids.end());
}

bool GDBRemoteCommunicationServerPlatform::KillSpawnedProcess(lldb::pid_t pid) {
  // make sure we know about this process
  if (SpawnedProcessFinished(pid)) {
    // it seems the process has been finished recently
    return true;
  }

  // first try a SIGTERM (standard kill)
  Host::Kill(pid, SIGTERM);

  // check if that worked
  for (size_t i = 0; i < 10; ++i) {
    if (SpawnedProcessFinished(pid)) {
      // it is now killed
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  if (SpawnedProcessFinished(pid))
    return true;

  // the launched process still lives.  Now try killing it again, this time
  // with an unblockable signal.
  Host::Kill(pid, SIGKILL);

  for (size_t i = 0; i < 10; ++i) {
    if (SpawnedProcessFinished(pid)) {
      // it is now killed
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // check one more time after the final sleep
  return SpawnedProcessFinished(pid);
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

  auto cwd = FileSystem::Instance()
                 .GetVirtualFileSystem()
                 ->getCurrentWorkingDirectory();
  if (!cwd)
    return SendErrorResponse(cwd.getError());

  StreamString response;
  response.PutBytesAsRawHex8(cwd->data(), cwd->size());
  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerPlatform::Handle_QSetWorkingDir(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("QSetWorkingDir:"));
  std::string path;
  packet.GetHexByteString(path);

  if (std::error_code ec = FileSystem::Instance()
                               .GetVirtualFileSystem()
                               ->setCurrentWorkingDirectory(path))
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
    lldb::pid_t pid, int signal, int status) {

  // Note MonitoringProcessLauncher::LaunchProcess() does not store the monitor
  // thread and we cannot control it. The child process monitor thread will call
  // static DebugserverProcessReaped() callback. It may happen after destroying
  // the GDBRemoteCommunicationServerPlatform instance.
  // HostProcessWindows::MonitorThread() calls the callback anyway when the
  // process is finished. But MonitorChildProcessThreadFunction() in
  // common/Host.cpp can fail and exit w/o calling the callback. So g_port_map
  // and g_spawned_pids may leak in this case. The system must not reuse pid
  // of zombie processes, so leaking g_spawned_pids shouldn't be a problem.
  // But we can do nothing with g_port_map in this case.

  g_port_map.FreePortForProcess(pid);

  {
    std::lock_guard<std::mutex> guard(g_spawned_pids_mutex);
    g_spawned_pids.erase(pid);
  }
}

Status GDBRemoteCommunicationServerPlatform::LaunchProcess() {
  if (!m_process_launch_info.GetArguments().GetArgumentCount())
    return Status("%s: no process command line specified to launch",
                  __FUNCTION__);

  auto cwd = FileSystem::Instance()
                 .GetVirtualFileSystem()
                 ->getCurrentWorkingDirectory();
  if (cwd)
    m_process_launch_info.SetWorkingDirectory(FileSpec(*cwd));

  // specify the process monitor if not already set.  This should generally be
  // what happens since we need to reap started processes.
  if (!m_process_launch_info.GetMonitorProcessCallback())
    m_process_launch_info.SetMonitorProcessCallback(
        &GDBRemoteCommunicationServerPlatform::DebugserverProcessReaped);

  Log *log = GetLog(LLDBLog::Platform);

  Status error = Host::LaunchProcess(m_process_launch_info);
  if (!error.Success()) {
    LLDB_LOGF(log,
              "GDBRemoteCommunicationServerPlatform::%s() "
              "Failed to launch executable %s: %s",
              __FUNCTION__,
              m_process_launch_info.GetArguments().GetArgumentAtIndex(0),
              error.AsCString());
    return error;
  }

  LLDB_LOGF(log,
            "GDBRemoteCommunicationServerPlatform::%s() "
            "Launched '%s' as process %" PRIu64,
            __FUNCTION__,
            m_process_launch_info.GetArguments().GetArgumentAtIndex(0),
            m_process_launch_info.GetProcessID());

  // add to list of spawned processes.  On an lldb-gdbserver, we would expect
  // there to be only one.
  const auto pid = m_process_launch_info.GetProcessID();
  if (pid != LLDB_INVALID_PROCESS_ID) {
    // add to spawned pids
    AddSpawnedProcess(pid);
  }

  return error;
}

void GDBRemoteCommunicationServerPlatform::SetPortMap(PortMap &&port_map) {
  g_port_map = std::move(port_map);
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
    lldb::pid_t pid, uint16_t port, const std::string &socket_name) {
  m_pending_gdb_server.pid = pid;
  m_pending_gdb_server.port = port;
  m_pending_gdb_server.socket_name = socket_name;
}
