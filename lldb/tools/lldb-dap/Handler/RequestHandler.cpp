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
#include "RunInTerminal.h"
#include "lldb/API/SBDefines.h"
#include "lldb/API/SBEnvironment.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <mutex>

#ifdef _WIN32
#include "lldb/Host/windows/PosixApi.h"
#else
#include <unistd.h>
#endif

#ifndef LLDB_DAP_README_URL
#define LLDB_DAP_README_URL                                                    \
  "https://lldb.llvm.org/use/lldbdap.html#debug-console"
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

static void
SetupIORedirection(const std::vector<std::optional<std::string>> &stdio,
                   lldb::SBLaunchInfo &launch_info) {
  for (const auto &[idx, value_opt] : llvm::enumerate(stdio)) {
    if (!value_opt)
      continue;
    const std::string &path = value_opt.value();
    assert(!path.empty() && "paths should not be empty");

    const int fd = static_cast<int>(idx);
    switch (fd) {
    case 0:
      launch_info.AddOpenFileAction(STDIN_FILENO, path.c_str(), true, false);
      break;
    case 1:
      launch_info.AddOpenFileAction(STDOUT_FILENO, path.c_str(), false, true);
      break;
    case 2:
      launch_info.AddOpenFileAction(STDERR_FILENO, path.c_str(), false, true);
      break;
    default:
      launch_info.AddOpenFileAction(fd, path.c_str(), true, true);
      break;
    }
  }
}

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

  llvm::Expected<std::shared_ptr<FifoFile>> comm_file_or_err =
      CreateRunInTerminalCommFile();
  if (!comm_file_or_err)
    return comm_file_or_err.takeError();
  FifoFile &comm_file = *comm_file_or_err.get();

  RunInTerminalDebugAdapterCommChannel comm_channel(comm_file.m_path);

  lldb::pid_t debugger_pid = LLDB_INVALID_PROCESS_ID;
#if !defined(_WIN32)
  debugger_pid = getpid();
#endif

  llvm::json::Object reverse_request = CreateRunInTerminalReverseRequest(
      arguments.configuration.program, arguments.args, arguments.env,
      arguments.cwd, comm_file.m_path, debugger_pid, arguments.stdio,
      arguments.console == protocol::eConsoleExternalTerminal);
  dap.SendReverseRequest<LogFailureResponseHandler>("runInTerminal",
                                                    std::move(reverse_request));

  if (llvm::Expected<lldb::pid_t> pid = comm_channel.GetLauncherPid())
    attach_info.SetProcessID(*pid);
  else
    return pid.takeError();

  std::optional<ScopeSyncMode> scope_sync_mode(dap.debugger);
  lldb::SBError error;
  dap.target.Attach(attach_info, error);

  if (error.Fail())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to attach to the target process. %s",
                                   comm_channel.GetLauncherError().c_str());
  // This will notify the runInTerminal launcher that we attached.
  // We have to make this async, as the function won't return until the launcher
  // resumes and reads the data.
  std::future<lldb::SBError> did_attach_message_success =
      comm_channel.NotifyDidAttach();

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

  // If sending the notification failed, the launcher should be dead by now and
  // the async didAttach notification should have an error message, so we
  // return it. Otherwise, everything was a success.
  did_attach_message_success.wait();
  error = did_attach_message_success.get();
  if (error.Success())
    return llvm::Error::success();
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 error.GetCString());
}

void BaseRequestHandler::Run(const Request &request) {
  // If this request was cancelled, send a cancelled response.
  if (dap.IsCancelled(request)) {
    Response cancelled{
        /*request_seq=*/request.seq,
        /*command=*/request.command,
        /*success=*/false,
        /*message=*/eResponseMessageCancelled,
    };
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
  const std::vector<std::string> &launchCommands = arguments.launchCommands;

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

  if (!arguments.stdio.empty() && !arguments.disableSTDIO)
    SetupIORedirection(arguments.stdio, launch_info);

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

    if (arguments.console != protocol::eConsoleInternal) {
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
  std::string message;
  llvm::raw_string_ostream OS(message);

#ifdef LLDB_DAP_WELCOME_MESSAGE
  dap.SendOutput(eOutputCategoryConsole, LLDB_DAP_WELCOME_MESSAGE);
#endif

  // Trying to provide a brief but helpful welcome message for users to better
  // understand how the debug console repl works.
  OS << "To get started with the debug console try ";
  switch (dap.repl_mode) {
  case ReplMode::Auto:
    OS << "\"<variable>\", \"<lldb-cmd>\" or \"help [<lldb-cmd>]\"\r\n";
    break;
  case ReplMode::Command:
    OS << "\"<lldb-cmd>\" or \"help [<lldb-cmd>]\".\r\n";
    break;
  case ReplMode::Variable:
    OS << "\"<variable>\" or \"" << dap.configuration.commandEscapePrefix
       << "help [<lldb-cmd>]\".\r\n";
    break;
  }

  OS << "For more information visit " LLDB_DAP_README_URL ".\r\n";

  dap.SendOutput(OutputType::Console, message);
}

void BaseRequestHandler::PrintIntroductionMessage() const {
  std::string msg;
  llvm::raw_string_ostream os(msg);
  if (dap.target && dap.target.GetExecutable()) {
    std::string path = GetSBFileSpecPath(dap.target.GetExecutable());
    os << llvm::formatv("Executable binary set to '{0}' ({1}).\r\n", path,
                        dap.target.GetTriple());
  }
  if (dap.target.GetProcess()) {
    os << llvm::formatv("Attached to process {0}.\r\n",
                        dap.target.GetProcess().GetProcessID());
  }
  dap.SendOutput(OutputType::Console, msg);
}

bool BaseRequestHandler::HasInstructionGranularity(
    const llvm::json::Object &arguments) const {
  if (std::optional<llvm::StringRef> value = arguments.getString("granularity"))
    return value == "instruction";
  return false;
}

void BaseRequestHandler::BuildErrorResponse(
    llvm::Error err, protocol::Response &response) const {
  // Handle the ErrorSuccess case.
  if (!err) {
    response.success = true;
    return;
  }

  response.success = false;

  llvm::handleAllErrors(
      std::move(err),
      [&](const NotStoppedError &err) {
        response.message = lldb_dap::protocol::eResponseMessageNotStopped;
      },
      [&](const DAPError &err) {
        protocol::ErrorMessage error_message;
        error_message.sendTelemetry = false;
        error_message.format = err.getMessage();
        error_message.showUser = err.getShowUser();
        error_message.id = err.convertToErrorCode().value();
        error_message.url = err.getURL();
        error_message.urlLabel = err.getURLLabel();
        protocol::ErrorResponseBody body;
        body.error = error_message;

        response.body = body;
      },
      [&](const llvm::ErrorInfoBase &err) {
        protocol::ErrorMessage error_message;
        error_message.showUser = true;
        error_message.sendTelemetry = false;
        error_message.format = err.message();
        error_message.id = err.convertToErrorCode().value();
        protocol::ErrorResponseBody body;
        body.error = error_message;

        response.body = body;
      });
}

void BaseRequestHandler::SendError(llvm::Error err,
                                   protocol::Response &response) const {
  BuildErrorResponse(std::move(err), response);
  Send(response);
}

void BaseRequestHandler::SendSuccess(
    protocol::Response &response, std::optional<llvm::json::Value> body) const {
  response.success = true;
  if (body)
    response.body = std::move(*body);

  Send(response);
}

void BaseRequestHandler::Send(protocol::Response &response) const {
  // Mark the request as 'cancelled' if the debugger was interrupted while
  // evaluating this handler.
  if (dap.debugger.InterruptRequested()) {
    response.success = false;
    response.message = protocol::eResponseMessageCancelled;
    response.body = std::nullopt;
  }

  dap.Send(response);
}

} // namespace lldb_dap
