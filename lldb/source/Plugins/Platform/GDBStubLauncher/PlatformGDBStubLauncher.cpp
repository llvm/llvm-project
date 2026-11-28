//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Platform/GDBStubLauncher/PlatformGDBStubLauncher.h"
#include "Plugins/Process/gdb-remote/ProcessGDBRemote.h"
#include "lldb/Breakpoint/Breakpoint.h"
#include "lldb/Breakpoint/BreakpointList.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/PosixApi.h"
#include "lldb/Host/ProcessLaunchInfo.h"
#include "lldb/Host/common/TCPSocket.h"
#include "lldb/Interpreter/OptionValueProperties.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Args.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Listener.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/ProcessInfo.h"
#include "lldb/Utility/State.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"

#include <chrono>
#include <condition_variable>
#include <csignal>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(PlatformGDBStubLauncher)

namespace {
enum ConnectEnum { eConnectGDBStub, eConnectRemotePlatform };

static constexpr OptionEnumValueElement g_connect_enums[] = {
    {eConnectGDBStub, "gdb-stub",
     "Connect to the program's gdb stub on the chosen port."},
    {eConnectRemotePlatform, "remote-platform",
     "Connect to the gdb-remote platform server at connect-url and attach to "
     "one of its processes."},
};

#define LLDB_PROPERTIES_platformgdbstublauncher
#include "PlatformGDBStubLauncherProperties.inc"

enum {
#define LLDB_PROPERTIES_platformgdbstublauncher
#include "PlatformGDBStubLauncherPropertiesEnum.inc"
};

class PluginProperties : public Properties {
public:
  PluginProperties() {
    m_collection_sp = std::make_shared<OptionValueProperties>(
        PlatformGDBStubLauncher::GetPluginNameStatic());
    m_collection_sp->Initialize(g_platformgdbstublauncher_properties_def);
  }

  llvm::StringRef GetString(uint32_t idx) const {
    return GetPropertyAtIndexAs<llvm::StringRef>(idx, "");
  }
  FileSpec GetFile(uint32_t idx) const {
    return GetPropertyAtIndexAs<FileSpec>(idx, {});
  }
  ConnectEnum GetConnect() const {
    const uint32_t idx = ePropertyConnect;
    return GetPropertyAtIndexAs<ConnectEnum>(
        idx, static_cast<ConnectEnum>(
                 g_platformgdbstublauncher_properties[idx].default_uint_value));
  }
  bool GetRequireHardwareBreakpoints() const {
    const uint32_t idx = ePropertyRequireHardwareBreakpoints;
    return GetPropertyAtIndexAs<bool>(
        idx, g_platformgdbstublauncher_properties[idx].default_uint_value != 0);
  }
  uint64_t GetTimeout() const {
    const uint32_t idx = ePropertyTimeout;
    return GetPropertyAtIndexAs<uint64_t>(
        idx, g_platformgdbstublauncher_properties[idx].default_uint_value);
  }
};

/// How the program exited, as reported by its monitor thread.
class LauncherExit {
public:
  void Set(int signal, int status) {
    std::lock_guard<std::mutex> guard(m_mutex);
    m_description =
        signal != 0
            ? llvm::formatv("launcher was terminated by signal {0}", signal)
            : llvm::formatv("launcher exited with status {0}", status);
  }

  std::optional<std::string> Get() {
    std::lock_guard<std::mutex> guard(m_mutex);
    return m_description;
  }

private:
  std::mutex m_mutex;
  std::optional<std::string> m_description;
};

} // namespace

static PluginProperties &GetGlobalProperties() {
  static PluginProperties g_settings;
  return g_settings;
}

llvm::StringRef PlatformGDBStubLauncher::GetPluginDescriptionStatic() {
  return "Launches a local program that serves a gdb-remote stub and attaches "
         "to it";
}

void PlatformGDBStubLauncher::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(),
                                PlatformGDBStubLauncher::CreateInstance,
                                PlatformGDBStubLauncher::DebuggerInitialize);
}

void PlatformGDBStubLauncher::Terminate() {
  PluginManager::UnregisterPlugin(PlatformGDBStubLauncher::CreateInstance);
}

void PlatformGDBStubLauncher::DebuggerInitialize(Debugger &debugger) {
  if (!PluginManager::GetSettingForPlatformPlugin(debugger,
                                                  GetPluginNameStatic())) {
    PluginManager::CreateSettingForPlatformPlugin(
        debugger, GetGlobalProperties().GetValueProperties(),
        "Properties for the gdb-stub-launcher platform plugin.",
        /*is_global_property=*/true);
  }
}

PlatformSP PlatformGDBStubLauncher::CreateInstance(bool force,
                                                   const ArchSpec *arch) {
  Log *log = GetLog(LLDBLog::Platform);
  LLDB_LOG(log, "force = {0}, arch = ({1}, {2})", force,
           arch ? arch->GetArchitectureName() : "<null>",
           arch ? arch->GetTriple().getTriple() : "<null>");

  // Only selected explicitly, with `platform select gdb-stub-launcher`.
  if (force)
    return PlatformSP(new PlatformGDBStubLauncher());
  return nullptr;
}

std::vector<ArchSpec> PlatformGDBStubLauncher::GetSupportedArchitectures(
    const ArchSpec &process_host_arch) {
  llvm::StringRef arch = GetGlobalProperties().GetString(ePropertyArchitecture);
  if (arch.empty())
    return {HostInfo::GetArchitecture()};

  llvm::Triple triple = HostInfo::GetArchitecture().GetTriple();
  triple.setArchName(arch);
  if (triple.getArch() != llvm::Triple::UnknownArch)
    return {ArchSpec(triple)};
  return {};
}

llvm::Error PlatformGDBStubLauncher::CheckConfig(const LaunchConfig &config) {
  llvm::SmallVector<llvm::StringRef, 4> missing;
  if (config.command.empty())
    missing.push_back("launch-command");
  if (config.connect == ConnectMode::RemotePlatform) {
    if (config.connect_url.empty())
      missing.push_back("connect-url");
    if (config.attach.empty())
      missing.push_back("attach");
  }
  // Either alone has nothing to wait for.
  if (config.ready_log.empty() != config.ready_pattern.empty())
    missing.push_back(config.ready_log.empty() ? "ready-log" : "ready-pattern");
  if (missing.empty())
    return llvm::Error::success();
  return llvm::createStringErrorV("gdb-stub-launcher is missing settings: {0} "
                                  "(platform.plugin.gdb-stub-launcher.*)",
                                  llvm::join(missing, ", "));
}

std::string PlatformGDBStubLauncher::ExpandCommand(llvm::StringRef command,
                                                   uint16_t port,
                                                   llvm::StringRef executable) {
  // Single quotes make every character literal but ', which is closed, escaped
  // and reopened.
  std::string quoted_executable = "'";
  for (char c : executable) {
    if (c == '\'')
      quoted_executable += "'\\''";
    else
      quoted_executable += c;
  }
  quoted_executable += '\'';
  std::string result;
  while (!command.empty()) {
    const size_t brace = command.find('{');
    result += command.take_front(brace);
    if (brace == llvm::StringRef::npos)
      break;
    command = command.drop_front(brace);
    if (command.consume_front("{port}"))
      result += std::to_string(port);
    else if (command.consume_front("{executable}"))
      result += quoted_executable;
    else {
      result += '{';
      command = command.drop_front();
    }
  }
  return result;
}

PlatformGDBStubLauncher::LaunchConfig PlatformGDBStubLauncher::ReadConfig() {
  const PluginProperties &properties = GetGlobalProperties();
  LaunchConfig config;
  config.command = properties.GetString(ePropertyLaunchCommand).str();
  config.working_dir = properties.GetFile(ePropertyWorkingDir).GetPath();
  config.port_env_var = properties.GetString(ePropertyPortEnvVar).str();
  config.connect = properties.GetConnect() == eConnectRemotePlatform
                       ? ConnectMode::RemotePlatform
                       : ConnectMode::GDBStub;
  config.connect_url = properties.GetString(ePropertyConnectURL).str();
  config.attach = properties.GetString(ePropertyAttach).str();
  config.ready_log = properties.GetFile(ePropertyReadyLog).GetPath();
  config.ready_pattern = properties.GetString(ePropertyReadyPattern).str();
  if (!config.ready_log.empty() &&
      llvm::sys::path::is_relative(config.ready_log) &&
      !config.working_dir.empty()) {
    llvm::SmallString<256> path(config.working_dir);
    llvm::sys::path::append(path, config.ready_log);
    config.ready_log = path.str().str();
  }
  return config;
}

lldb::ProcessSP
PlatformGDBStubLauncher::DebugProcess(ProcessLaunchInfo &launch_info,
                                      Debugger &debugger, Target &target,
                                      Status &error) {
  Log *log = GetLog(LLDBLog::Platform);
  const PluginProperties &properties = GetGlobalProperties();

  const LaunchConfig config = ReadConfig();
  if (llvm::Error err = CheckConfig(config)) {
    error = Status::FromError(std::move(err));
    return nullptr;
  }

  // The platform tracks one program, so don't take it from another target's
  // live process.
  if (LauncherServesOtherTarget(target)) {
    error = Status::FromErrorString(
        "gdb-stub-launcher is already serving another target; kill or detach "
        "its process first");
    return nullptr;
  }

  // Target::Launch has already deleted the previous process; make sure its
  // program is gone before starting another.
  ReleaseLauncher(std::chrono::seconds(30));
  m_remote_platform_sp.reset();

  llvm::Expected<uint16_t> expected_port = TCPSocket::FindFreePort();
  if (!expected_port) {
    error = Status::FromError(expected_port.takeError());
    return nullptr;
  }
  const uint16_t port = *expected_port;

  std::string executable;
  if (ModuleSP exe_module_sp = target.GetExecutableModule())
    executable = exe_module_sp->GetFileSpec().GetPath();

  if (properties.GetRequireHardwareBreakpoints()) {
    // The setting covers breakpoints created from now on; convert the existing
    // ones too.
    target.SetRequireHardwareBreakpoints(true);
    BreakpointList &breakpoints = target.GetBreakpointList();
    std::unique_lock<std::recursive_mutex> lock;
    breakpoints.GetListMutex(lock);
    for (size_t i = 0; i < breakpoints.GetSize(); ++i)
      if (BreakpointSP bp_sp = breakpoints.GetBreakpointAtIndex(i))
        LLDB_LOG_ERROR(log, bp_sp->SetIsHardware(true),
                       "failed to make breakpoint {1} hardware: {0}",
                       bp_sp->GetID());
  }

  // We launch the program serving the stub, not the target, so clear
  // inferior-oriented launch flags.
  launch_info.SetArguments(
      Args({"/bin/sh", "-c", ExpandCommand(config.command, port, executable)}),
      /*first_arg_is_executable_name=*/true);
  if (!config.working_dir.empty())
    launch_info.SetWorkingDirectory(FileSpec(config.working_dir));
  launch_info.SetLaunchInSeparateProcessGroup(true);
  launch_info.GetFlags().Clear(eLaunchFlagDebug | eLaunchFlagDisableASLR);

  // The launch environment already has target.env-vars and `process launch -E`
  // applied.
  if (!config.port_env_var.empty())
    launch_info.GetEnvironment()[config.port_env_var] = std::to_string(port);

  auto launcher_exit = std::make_shared<LauncherExit>();
  launch_info.SetMonitorProcessCallback(
      [=](lldb::pid_t pid, int signal, int status) {
        LLDB_LOG(log, "launcher exited: pid = {0}, signal = {1}, status = {2}",
                 pid, signal, status);
        launcher_exit->Set(signal, status);
      });

  // A program doing terminal I/O from a separate process group gets EIO or is
  // stopped, so keep it off the tty: stdin from /dev/null, output to a log file
  // you can tail. The default log is per debugger process, so concurrent
  // sessions don't truncate each other's.
  launch_info.AppendSuppressFileAction(STDIN_FILENO, /*read=*/true,
                                       /*write=*/false);
  FileSpec log_file = properties.GetFile(ePropertyLog);
  if (!log_file) {
    llvm::SmallString<128> path;
    llvm::sys::path::system_temp_directory(/*ErasedOnReboot=*/true, path);
    llvm::sys::path::append(path,
                            llvm::formatv("gdb-stub-launcher-{0}.log",
                                          llvm::sys::Process::getProcessId())
                                .str());
    log_file = FileSpec(path);
  }
  launch_info.AppendOpenFileAction(STDOUT_FILENO, log_file, /*read=*/false,
                                   /*write=*/true);
  launch_info.AppendDuplicateFileAction(STDOUT_FILENO, STDERR_FILENO);
  LLDB_LOG(log, "launcher stdout/stderr -> {0}", log_file.GetPath());

  // Only text written by this launch counts as ready.
  uint64_t ready_log_offset = 0;
  if (!config.ready_log.empty())
    llvm::sys::fs::file_size(config.ready_log, ready_log_offset);

  llvm::ArrayRef<const char *> argv =
      launch_info.GetArguments().GetArgumentArrayRef();
  LLDB_LOG(log, "launching {0}", llvm::make_range(argv.begin(), argv.end()));
  error = Host::LaunchProcess(launch_info);
  if (error.Fail())
    return nullptr;
  const lldb::pid_t launcher_pid = launch_info.GetProcessID();
  // The timeout bounds the whole wait, however many steps it takes.
  const Deadline deadline = std::chrono::steady_clock::now() +
                            std::chrono::seconds(properties.GetTimeout());

  if (!config.ready_log.empty())
    error = WaitForLogLine(config.ready_log, ready_log_offset,
                           config.ready_pattern, launcher_pid, deadline);
  ProcessSP process_sp;
  if (error.Success()) {
    if (config.connect == ConnectMode::RemotePlatform) {
      process_sp = AttachRemotePlatform(config, launch_info, debugger, target,
                                        deadline, error);
    } else {
      error = WaitForPortOpen(port, launcher_pid, deadline);
      if (error.Success())
        process_sp = ConnectGDBStub(launch_info, target, port, error);
    }
  }

  if (!process_sp) {
    std::optional<std::string> exit_description = launcher_exit->Get();
    if (exit_description)
      error = Status::FromError(llvm::joinErrors(
          llvm::createStringError(*exit_description), error.takeError()));
    error = Status::FromErrorStringWithFormatv("{0} (launcher output: {1})",
                                               error.AsCString("launch failed"),
                                               log_file.GetPath());
    // We launched it but never got a process, so nothing else will reap it.
    TerminateLaunched(launcher_pid);
    return nullptr;
  }

  WatchLauncher(launcher_pid, process_sp);
  return process_sp;
}

lldb::ProcessSP
PlatformGDBStubLauncher::ConnectGDBStub(ProcessLaunchInfo &launch_info,
                                        Target &target, uint16_t port,
                                        Status &error) {
  ProcessSP process_sp = target.CreateProcess(
      launch_info.GetListener(),
      process_gdb_remote::ProcessGDBRemote::GetPluginNameStatic(), nullptr,
      true);
  if (!process_sp) {
    error = Status::FromErrorString("failed to create GDB remote process");
    return nullptr;
  }

  process_sp->HijackProcessEvents(launch_info.GetHijackListener());
  error = process_sp->ConnectRemote(
      llvm::formatv("connect://localhost:{0}", port).str());
  if (error.Fail())
    return nullptr;

  // ConnectRemote treats the connection as an attach, which would make quitting
  // detach and leave the program running. We launched it, so quit kills it.
  process_sp->SetShouldDetach(false);
  return process_sp;
}

lldb::ProcessSP PlatformGDBStubLauncher::AttachRemotePlatform(
    const LaunchConfig &config, ProcessLaunchInfo &launch_info,
    Debugger &debugger, Target &target, Deadline deadline, Status &error) {
  Log *log = GetLog(LLDBLog::Platform);
  const lldb::pid_t launcher_pid = launch_info.GetProcessID();

  // The same platform `platform connect` uses behind remote-macosx.
  m_remote_platform_sp = Platform::Create("remote-gdb-server");
  if (!m_remote_platform_sp) {
    error = Status::FromErrorString(
        "the remote-gdb-server platform is not available");
    return nullptr;
  }
  Platform &remote = *m_remote_platform_sp;

  // The server may accept before it can list processes, so retry until the
  // process is listed.
  lldb::pid_t pid = LLDB_INVALID_PROCESS_ID;
  if (!llvm::to_integer(config.attach, pid))
    pid = LLDB_INVALID_PROCESS_ID;
  Status last_error;
  while (std::chrono::steady_clock::now() < deadline) {
    if (!remote.IsConnected()) {
      Args url;
      url.AppendArgument(config.connect_url);
      last_error = remote.ConnectRemote(url);
    }
    if (remote.IsConnected() && pid == LLDB_INVALID_PROCESS_ID) {
      ProcessInstanceInfoMatch match(config.attach.c_str(), NameMatch::Equals);
      ProcessInstanceInfoList infos;
      remote.FindProcesses(match, infos);
      if (infos.size() > 1) {
        error = Status::FromErrorStringWithFormatv(
            "{0} lists {1} processes named '{2}'; use a pid",
            config.connect_url, infos.size(), config.attach);
        m_remote_platform_sp.reset();
        return nullptr;
      }
      if (infos.size() == 1)
        pid = infos[0].GetProcessID();
    }
    if (remote.IsConnected() && pid != LLDB_INVALID_PROCESS_ID)
      break;
    ProcessInstanceInfo proc_info;
    if (!Host::GetProcessInfo(launcher_pid, proc_info)) {
      error = Status::FromErrorStringWithFormatv(
          "launcher (pid {0}) exited before {1} served", launcher_pid,
          config.connect_url);
      m_remote_platform_sp.reset();
      return nullptr;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  if (!remote.IsConnected()) {
    error = Status::FromErrorStringWithFormatv(
        "could not connect to {0}: {1}", config.connect_url,
        last_error.AsCString(
            "timed out (platform.plugin.gdb-stub-launcher.timeout)"));
    m_remote_platform_sp.reset();
    return nullptr;
  }
  if (pid == LLDB_INVALID_PROCESS_ID) {
    ProcessInstanceInfoMatch all;
    ProcessInstanceInfoList infos;
    remote.FindProcesses(all, infos);
    llvm::SmallVector<std::string, 16> names;
    for (const ProcessInstanceInfo &info : infos)
      names.push_back(info.GetName().empty() ? "?" : info.GetName().str());
    error = Status::FromErrorStringWithFormatv(
        "{0} does not list '{1}' (available: {2})", config.connect_url,
        config.attach, llvm::join(names, ", "));
    m_remote_platform_sp.reset();
    return nullptr;
  }

  ProcessAttachInfo attach_info;
  attach_info.SetProcessID(pid);
  attach_info.SetListener(launch_info.GetListener());
  attach_info.SetHijackListener(launch_info.GetHijackListener());
  attach_info.SetShadowListener(launch_info.GetShadowListener());
  // The remote platform can return a process whose connect or attach failed.
  ProcessSP process_sp = remote.Attach(attach_info, debugger, &target, error);
  if (!process_sp || error.Fail()) {
    if (error.Success())
      error = Status::FromErrorStringWithFormatv("attaching through {0} failed",
                                                 config.connect_url);
    m_remote_platform_sp.reset();
    return nullptr;
  }
  LLDB_LOG(log, "attached to {0} (pid {1}) through {2}", config.attach, pid,
           config.connect_url);

  // As with ConnectGDBStub: we launched the program, so quitting kills it
  // rather than detaching and leaving it running.
  process_sp->SetShouldDetach(false);
  return process_sp;
}

Status PlatformGDBStubLauncher::WaitForPortOpen(uint16_t port, lldb::pid_t pid,
                                                Deadline deadline) {
  Log *log = GetLog(LLDBLog::Platform);

  // Many stubs serve a single client, so a probe connection would use up the
  // one slot. Detect the listening socket passively with lsof instead.
  // ProcessGDBRemote then makes the only connection.
  const std::string cmd =
      llvm::formatv("lsof -nP -iTCP:{0} -sTCP:LISTEN", port).str();

  while (std::chrono::steady_clock::now() < deadline) {
    int exit_status = -1;
    Host::RunShellCommand(cmd, /*working_dir=*/FileSpec(), &exit_status,
                          /*signo_ptr=*/nullptr, /*command_output=*/nullptr,
                          /*error_output=*/nullptr,
                          /*timeout=*/std::chrono::seconds(10));
    if (exit_status == 0) {
      LLDB_LOG(log, "gdb stub listening on localhost:{0}", port);
      return Status();
    }

    ProcessInstanceInfo proc_info;
    if (pid != LLDB_INVALID_PROCESS_ID && !Host::GetProcessInfo(pid, proc_info))
      return Status::FromErrorStringWithFormatv(
          "launcher (pid {0}) exited before its gdb stub listened", pid);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  return Status::FromErrorStringWithFormatv(
      "timed out waiting for a gdb stub on localhost:{0} "
      "(platform.plugin.gdb-stub-launcher.timeout)",
      port);
}

Status PlatformGDBStubLauncher::WaitForLogLine(llvm::StringRef path,
                                               uint64_t offset,
                                               llvm::StringRef pattern,
                                               lldb::pid_t pid,
                                               Deadline deadline) {
  while (std::chrono::steady_clock::now() < deadline) {
    if (auto buffer = llvm::MemoryBuffer::getFile(path, /*IsText=*/true)) {
      llvm::StringRef text = (*buffer)->getBuffer();
      // A program that truncates the log starts it over.
      if (text.size() < offset)
        offset = 0;
      if (text.drop_front(offset).contains(pattern))
        return Status();
    }
    ProcessInstanceInfo proc_info;
    if (pid != LLDB_INVALID_PROCESS_ID && !Host::GetProcessInfo(pid, proc_info))
      return Status::FromErrorStringWithFormatv(
          "launcher (pid {0}) exited before '{1}' appeared in {2}", pid,
          pattern, path);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  return Status::FromErrorStringWithFormatv(
      "timed out waiting for '{0}' in {1} "
      "(platform.plugin.gdb-stub-launcher.timeout)",
      pattern, path);
}

void PlatformGDBStubLauncher::TerminateLaunched(lldb::pid_t pid) {
  if (pid == LLDB_INVALID_PROCESS_ID)
    return;
  Log *log = GetLog(LLDBLog::Platform);
#ifndef _WIN32
  // A negative pid signals the whole process group.
  if (::kill(-pid, SIGTERM) == 0)
    LLDB_LOG(log, "sent SIGTERM to launcher process group {0}", pid);
#else
  Host::Kill(pid, SIGTERM);
  LLDB_LOG(log, "sent SIGTERM to launcher pid {0}", pid);
#endif
}

// Shared by the platform and the watcher thread of one program. Only the
// watcher reads the process's state events, so the detach-or-terminate
// decision is made in one place, in event order.
class PlatformGDBStubLauncher::LauncherWatch {
public:
  LauncherWatch(lldb::pid_t pid, ListenerSP listener_sp, ProcessWP process_wp)
      : m_pid(pid), m_listener_sp(std::move(listener_sp)),
        m_process_wp(std::move(process_wp)) {}

  /// The watcher thread: decide once the process detaches, exits or goes away.
  void Run() {
    while (true) {
      EventSP event_sp;
      const bool got_event =
          m_listener_sp->GetEvent(event_sp, std::chrono::milliseconds(100));
      std::lock_guard<std::mutex> guard(m_mutex);
      if (got_event) {
        const StateType state =
            Process::ProcessEventData::GetStateFromEvent(event_sp.get());
        if (state == eStateDetached || state == eStateExited) {
          Decide(/*detached=*/state == eStateDetached);
          return;
        }
        continue;
      }
      // With no detach event queued, a process that is gone, or that the
      // platform is replacing, went away without detaching.
      if (m_release_requested || m_process_wp.expired()) {
        Decide(/*detached=*/false);
        return;
      }
    }
  }

  /// Have the watcher decide now, waiting up to \p wait for it. Returns false
  /// if the program was disowned.
  bool Release(std::chrono::milliseconds wait) {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_release_requested = true;
    if (!m_cv.wait_for(lock, wait, [&] { return m_decided; }))
      Decide(/*detached=*/false);
    return !m_disowned;
  }

  /// Whether the program still serves a process of a target other than
  /// \p target.
  bool ServesOtherTarget(const Target &target) {
    std::lock_guard<std::mutex> guard(m_mutex);
    if (m_decided)
      return false;
    ProcessSP process_sp = m_process_wp.lock();
    if (!process_sp)
      return false;
    TargetSP process_target_sp = process_sp->CalculateTarget();
    return process_target_sp && process_target_sp.get() != &target;
  }

  lldb::pid_t GetPID() const { return m_pid; }

private:
  /// Disown the program after a detach, otherwise SIGTERM it. Requires mutex.
  void Decide(bool detached) {
    if (m_decided)
      return;
    m_decided = true;
    m_disowned = detached;
    if (!detached)
      TerminateLaunched(m_pid);
    m_cv.notify_all();
  }

  const lldb::pid_t m_pid;
  const ListenerSP m_listener_sp;
  const ProcessWP m_process_wp;
  std::mutex m_mutex;
  std::condition_variable m_cv;
  bool m_release_requested = false;
  bool m_decided = false;
  bool m_disowned = false;
};

bool PlatformGDBStubLauncher::LauncherServesOtherTarget(const Target &target) {
  return m_launcher && m_launcher->ServesOtherTarget(target);
}

PlatformGDBStubLauncher::~PlatformGDBStubLauncher() {
  ReleaseLauncher(std::chrono::seconds(0));
}

void PlatformGDBStubLauncher::WatchLauncher(lldb::pid_t pid,
                                            const ProcessSP &process_sp) {
  // A platform gets no callback when its Process detaches or goes away, and by
  // the time the Process is deleted its state reads eStateExited even after a
  // detach. Listen for the state events instead: a secondary listener gets
  // each one after the primary listener has handled it, so a detach is known
  // by the time `process detach` reports it.
  ListenerSP listener_sp =
      Listener::MakeListener("lldb.platform.gdb-stub-launcher.watch");
  listener_sp->StartListeningForEvents(process_sp.get(),
                                       Process::eBroadcastBitStateChanged);
  auto watch =
      std::make_shared<LauncherWatch>(pid, std::move(listener_sp), process_sp);
  m_launcher = watch;
  std::thread([watch] { watch->Run(); }).detach();
}

void PlatformGDBStubLauncher::ReleaseLauncher(std::chrono::seconds wait) {
  std::shared_ptr<LauncherWatch> watch = std::move(m_launcher);
  if (!watch || !watch->Release(std::chrono::seconds(1)))
    return;
#ifndef _WIN32
  // The program tears down asynchronously, and one may refuse to start while
  // the previous instance is still up, so give the group time to exit.
  const auto deadline = std::chrono::steady_clock::now() + wait;
  while (::kill(-watch->GetPID(), 0) == 0 &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
#endif
}
