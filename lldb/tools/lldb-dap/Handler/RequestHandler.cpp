//===-- RequestHandler.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Handler/RequestHandler.h"
#include "DAP.h"
#include "EventHelper.h"
#include "Handler/ResponseHandler.h"
#include "JSONUtils.h"
#include "LLDBUtils.h"
#include "Protocol/ProtocolBase.h"
#include "Protocol/ProtocolRequests.h"
#include "lldb/API/SBDefines.h"
#include "lldb/API/SBEnvironment.h"
#include "llvm/Support/Error.h"
#include <cerrno>
#include <csignal>
#include <limits>
#include <mutex>

#if !defined(_WIN32)
#include <unistd.h>
#endif

using namespace lldb_dap::protocol;

namespace lldb_dap {

static std::vector<const char *>
MakeArgv(const llvm::ArrayRef<std::string> &strs) {
  // Create and return an array of "const char *", one for each C string in
  // "strs" and terminate the list with a NULL. This can be used for argument
  // vectors (argv) or environment vectors (envp) like those passed to the
  // "main" function in C programs.
  std::vector<const char *> argv;
  for (const auto &s : strs)
    argv.push_back(s.c_str());
  argv.push_back(nullptr);
  return argv;
}

static uint32_t SetLaunchFlag(uint32_t flags, bool flag,
                              lldb::LaunchFlags mask) {
  if (flag)
    flags |= mask;
  else
    flags &= ~mask;

  return flags;
}

// Assume single thread
class PidReceiver {
private:
  inline static volatile std::sig_atomic_t pid;
  sigset_t oldmask;
  struct sigaction oldaction;

  static_assert(std::numeric_limits<volatile sig_atomic_t>::max() >=
                    std::numeric_limits<pid_t>::max(),
                "sig_atomic_t must be able to hold a pid_t value");

  static void sig_handler(int, siginfo_t *info, void *) {
    // Store the PID from the signal info
    pid = info->si_pid;
  }

  PidReceiver(const PidReceiver &) = delete;
  PidReceiver &operator=(const PidReceiver &) = delete;
  PidReceiver(PidReceiver &&) = delete;

public:
  PidReceiver() {
    pid = LLDB_INVALID_PROCESS_ID;
    // sigprocmask and sigaction can only fail by programmer error
    // Defer SIGUSR1, otherwise we might receive it out of pselect and hang
    sigset_t mask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGUSR1);
    assert(!sigprocmask(SIG_BLOCK, &mask, &oldmask));

    // Set up sigaction for SIGUSR1 with SA_SIGINFO
    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = sig_handler;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    assert(!sigaction(SIGUSR1, &sa, &oldaction));
  }

  ~PidReceiver() {
    // Restore the old signal mask
    assert(!sigprocmask(SIG_SETMASK, &oldmask, nullptr));
    assert(!sigaction(SIGUSR1, &oldaction, nullptr));
  }

  llvm::Expected<lldb::pid_t> WaitForPid() {
    // Wait for the signal to be received
    while (pid == LLDB_INVALID_PROCESS_ID) {
      struct timespec timeout;
      timeout.tv_sec = 5;
      timeout.tv_nsec = 0;
      auto ret = pselect(0, nullptr, nullptr, nullptr, &timeout, &oldmask);
      if (ret == -1 && errno != EINTR) {
        return llvm::createStringError(
            std::error_code(errno, std::generic_category()), "pselect failed");
      } else if (ret == 0) {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "Timed out waiting for signal");
      }
    }
    return pid;
  }
};

static llvm::Error
RunInTerminal(DAP &dap, const protocol::LaunchRequestArguments &arguments) {
  if (!dap.clientFeatures.contains(
          protocol::eClientFeatureRunInTerminalRequest))
    return llvm::make_error<DAPError>("Cannot use runInTerminal, feature is "
                                      "not supported by the connected client");

  if (arguments.configuration.program.empty())
    return llvm::make_error<DAPError>(
        "program must be set to when using runInTerminal");

  dap.is_attach = true;
  lldb::SBAttachInfo attach_info;

  lldb::pid_t debugger_pid = LLDB_INVALID_PROCESS_ID;
#if !defined(_WIN32)
  debugger_pid = getpid();
#endif

  auto pid_receiver = PidReceiver();
  llvm::json::Object reverse_request = CreateRunInTerminalReverseRequest(
      arguments.configuration.program, arguments.args, arguments.env,
      arguments.cwd, debugger_pid);
  dap.SendReverseRequest<LogFailureResponseHandler>("runInTerminal",
                                                    std::move(reverse_request));

  if (llvm::Expected<lldb::pid_t> pid = pid_receiver.WaitForPid())
    attach_info.SetProcessID(*pid);
  else
    return pid.takeError();

  std::optional<ScopeSyncMode> scope_sync_mode(dap.debugger);
  lldb::SBError error;
  dap.target.Attach(attach_info, error);

  if (error.Fail())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to attach to the target process.");

  // We just attached to the runInTerminal launcher, which was waiting to be
  // attached. We now resume it, so it can receive the didAttach notification
  // and then perform the exec. Upon continuing, the debugger will stop the
  // process right in the middle of the exec. To the user, what we are doing is
  // transparent, as they will only be able to see the process since the exec,
  // completely unaware of the preparatory work.
  dap.target.GetProcess().Continue();

  // Now that the actual target is just starting (i.e. exec was just invoked),
  // we return the debugger to its sync state.
  scope_sync_mode.reset();

  return llvm::Error::success();
}

void BaseRequestHandler::Run(const Request &request) {
  // If this request was cancelled, send a cancelled response.
  if (dap.IsCancelled(request)) {
    Response cancelled{/*request_seq=*/request.seq,
                       /*command=*/request.command,
                       /*success=*/false,
                       /*message=*/eResponseMessageCancelled,
                       /*body=*/std::nullopt};
    dap.Send(cancelled);
    return;
  }

  lldb::SBMutex lock = dap.GetAPIMutex();
  std::lock_guard<lldb::SBMutex> guard(lock);

  // FIXME: After all the requests have migrated from LegacyRequestHandler >
  // RequestHandler<> we should be able to move this into
  // RequestHandler<>::operator().
  operator()(request);

  // FIXME: After all the requests have migrated from LegacyRequestHandler >
  // RequestHandler<> we should be able to check `debugger.InterruptRequest` and
  // mark the response as cancelled.
}

llvm::Error BaseRequestHandler::LaunchProcess(
    const protocol::LaunchRequestArguments &arguments) const {
  auto launchCommands = arguments.launchCommands;

  // Instantiate a launch info instance for the target.
  auto launch_info = dap.target.GetLaunchInfo();

  // Grab the current working directory if there is one and set it in the
  // launch info.
  if (!arguments.cwd.empty())
    launch_info.SetWorkingDirectory(arguments.cwd.data());

  // Extract any extra arguments and append them to our program arguments for
  // when we launch
  if (!arguments.args.empty())
    launch_info.SetArguments(MakeArgv(arguments.args).data(), true);

  // Pass any environment variables along that the user specified.
  if (!arguments.env.empty()) {
    lldb::SBEnvironment env;
    for (const auto &kv : arguments.env)
      env.Set(kv.first().data(), kv.second.c_str(), true);
    launch_info.SetEnvironment(env, true);
  }

  launch_info.SetDetachOnError(arguments.detachOnError);
  launch_info.SetShellExpandArguments(arguments.shellExpandArguments);

  auto flags = launch_info.GetLaunchFlags();
  flags =
      SetLaunchFlag(flags, arguments.disableASLR, lldb::eLaunchFlagDisableASLR);
  flags = SetLaunchFlag(flags, arguments.disableSTDIO,
                        lldb::eLaunchFlagDisableSTDIO);
  launch_info.SetLaunchFlags(flags | lldb::eLaunchFlagDebug |
                             lldb::eLaunchFlagStopAtEntry);

  {
    // Perform the launch in synchronous mode so that we don't have to worry
    // about process state changes during the launch.
    ScopeSyncMode scope_sync_mode(dap.debugger);

    if (arguments.runInTerminal) {
      if (llvm::Error err = RunInTerminal(dap, arguments))
        return err;
    } else if (launchCommands.empty()) {
      lldb::SBError error;
      dap.target.Launch(launch_info, error);
      if (error.Fail())
        return ToError(error);
    } else {
      // Set the launch info so that run commands can access the configured
      // launch details.
      dap.target.SetLaunchInfo(launch_info);
      if (llvm::Error err = dap.RunLaunchCommands(launchCommands))
        return err;

      // The custom commands might have created a new target so we should use
      // the selected target after these commands are run.
      dap.target = dap.debugger.GetSelectedTarget();
    }
  }

  // Make sure the process is launched and stopped at the entry point before
  // proceeding.
  lldb::SBError error =
      dap.WaitForProcessToStop(arguments.configuration.timeout);
  if (error.Fail())
    return ToError(error);

  return llvm::Error::success();
}

void BaseRequestHandler::PrintWelcomeMessage() const {
#ifdef LLDB_DAP_WELCOME_MESSAGE
  dap.SendOutput(OutputType::Console, LLDB_DAP_WELCOME_MESSAGE);
#endif
}

bool BaseRequestHandler::HasInstructionGranularity(
    const llvm::json::Object &arguments) const {
  if (std::optional<llvm::StringRef> value = arguments.getString("granularity"))
    return value == "instruction";
  return false;
}

} // namespace lldb_dap
