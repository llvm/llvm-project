//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_PLATFORM_GDBSTUBLAUNCHER_PLATFORMGDBSTUBLAUNCHER_H
#define LLDB_SOURCE_PLUGINS_PLATFORM_GDBSTUBLAUNCHER_PLATFORMGDBSTUBLAUNCHER_H

#include "lldb/Host/HostInfo.h"
#include "lldb/Target/Platform.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

namespace lldb_private {

/// Platform that, as one `process launch`, starts a local program serving a
/// gdb-remote stub and attaches to what it serves: a simulator, an emulator in
/// system mode, or a hypervisor with a gdb stub. The program is a shell command
/// with the chosen port substituted in, or passed in an environment variable.
///
/// Once the program is up, the platform connects ProcessGDBRemote to its stub,
/// or, for a program that serves a gdb-remote platform, attaches to one process
/// it lists. The program's process group is terminated when the debugged
/// process is killed, exits or is replaced by another launch. `process detach`
/// leaves it running.
///
/// One target at a time can have a program running: launching from another
/// target fails until the first target's process is killed, exits or detaches.
class PlatformGDBStubLauncher : public Platform {
public:
  enum class ConnectMode { GDBStub, RemotePlatform };

  /// The settings for one launch.
  struct LaunchConfig {
    std::string command;
    std::string working_dir;
    std::string port_env_var;
    ConnectMode connect = ConnectMode::GDBStub;
    std::string connect_url;
    std::string attach;
    std::string ready_log;
    std::string ready_pattern;
  };

  static void Initialize();
  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "gdb-stub-launcher"; }
  static llvm::StringRef GetPluginDescriptionStatic();

  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }
  llvm::StringRef GetDescription() override {
    return GetPluginDescriptionStatic();
  }

  UserIDResolver &GetUserIDResolver() override {
    return HostInfo::GetUserIDResolver();
  }

  std::vector<ArchSpec>
  GetSupportedArchitectures(const ArchSpec &process_host_arch) override;

  lldb::ProcessSP DebugProcess(ProcessLaunchInfo &launch_info,
                               Debugger &debugger, Target &target,
                               Status &error) override;

  /// Check that \p config names everything its connect mode needs, naming every
  /// missing setting in one error.
  static llvm::Error CheckConfig(const LaunchConfig &config);

  /// \p command with every {port} replaced by \p port and every {executable} by
  /// \p executable, shell quoted.
  static std::string ExpandCommand(llvm::StringRef command, uint16_t port,
                                   llvm::StringRef executable);

  lldb::ProcessSP Attach(ProcessAttachInfo &attach_info, Debugger &debugger,
                         Target *target, Status &status) override {
    status = Status::FromErrorString("Not supported");
    return nullptr;
  }

  /// After a remote-platform launch, these list the processes it serves, so
  /// `platform process list` works.
  uint32_t FindProcesses(const ProcessInstanceInfoMatch &match_info,
                         ProcessInstanceInfoList &proc_infos) override {
    if (m_remote_platform_sp)
      return m_remote_platform_sp->FindProcesses(match_info, proc_infos);
    return 0;
  }

  bool GetProcessInfo(lldb::pid_t pid,
                      ProcessInstanceInfo &proc_info) override {
    if (m_remote_platform_sp)
      return m_remote_platform_sp->GetProcessInfo(pid, proc_info);
    return false;
  }

  bool IsConnected() const override { return true; }

  void CalculateTrapHandlerSymbolNames() override {}

  MmapArgList GetMmapArgumentList(const ArchSpec &arch, lldb::addr_t addr,
                                  lldb::addr_t length, unsigned prot,
                                  unsigned flags, lldb::addr_t fd,
                                  lldb::addr_t offset) override {
    return Platform::GetHostPlatform()->GetMmapArgumentList(
        arch, addr, length, prot, flags, fd, offset);
  }

  /// dSYMs of what the program serves can carry LLDB helpers under
  /// Contents/Resources/Python, which only the host's Darwin platform loads.
  llvm::SmallDenseMap<FileSpec, LoadScriptFromSymFile>
  LocateExecutableScriptingResourcesForPlatform(
      Target *target, Module &module, Stream &feedback_stream) override {
    return Platform::GetHostPlatform()
        ->LocateExecutableScriptingResourcesForPlatform(target, module,
                                                        feedback_stream);
  }

  ~PlatformGDBStubLauncher() override;

protected:
  PlatformGDBStubLauncher() : Platform(/*is_host=*/true) {}

private:
  class LauncherWatch;
  using Deadline = std::chrono::steady_clock::time_point;

  /// Read the settings into a LaunchConfig.
  static LaunchConfig ReadConfig();

  /// Connect ProcessGDBRemote to the gdb stub on localhost:\p port.
  lldb::ProcessSP ConnectGDBStub(ProcessLaunchInfo &launch_info, Target &target,
                                 uint16_t port, Status &error);

  /// Connect a remote platform to the program and attach to the process in
  /// \p config, once the program lists it, or fail at \p deadline.
  lldb::ProcessSP AttachRemotePlatform(const LaunchConfig &config,
                                       ProcessLaunchInfo &launch_info,
                                       Debugger &debugger, Target &target,
                                       Deadline deadline, Status &error);

  /// Poll until localhost:\p port is listening, the launched \p pid exits, or
  /// \p deadline passes. A boot can take far longer than ProcessGDBRemote's own
  /// connect retry (~5s), so we wait here first.
  static Status WaitForPortOpen(uint16_t port, lldb::pid_t pid,
                                Deadline deadline);

  /// Poll until text written to \p path after byte \p offset contains
  /// \p pattern, the launched \p pid exits, or \p deadline passes.
  static Status WaitForLogLine(llvm::StringRef path, uint64_t offset,
                               llvm::StringRef pattern, lldb::pid_t pid,
                               Deadline deadline);

  /// SIGTERM the program's process group. It is launched as a group leader, so
  /// this reaps everything it started, not just the `sh` wrapper.
  static void TerminateLaunched(lldb::pid_t pid);

  /// Take ownership of the program \p pid, terminating it once \p process_sp
  /// exits or goes away. A detach leaves it running.
  void WatchLauncher(lldb::pid_t pid, const lldb::ProcessSP &process_sp);

  /// Terminate the current program, unless its process detached, and wait up
  /// to \p wait for its process group to exit.
  void ReleaseLauncher(std::chrono::seconds wait);

  /// Whether the current program still serves a live process of a target
  /// other than \p target.
  bool LauncherServesOtherTarget(const Target &target);

  static lldb::PlatformSP CreateInstance(bool force, const ArchSpec *arch);
  static void DebuggerInitialize(Debugger &debugger);

  std::shared_ptr<LauncherWatch> m_launcher;
  /// The connection behind the last remote-platform launch.
  lldb::PlatformSP m_remote_platform_sp;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_PLATFORM_GDBSTUBLAUNCHER_PLATFORMGDBSTUBLAUNCHER_H
