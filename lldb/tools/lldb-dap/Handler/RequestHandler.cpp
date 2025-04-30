//===-- RequestHandler.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Handler/RequestHandler.h"
#include "DAP.h"
#include "Handler/ResponseHandler.h"
#include "JSONUtils.h"
#include "LLDBUtils.h"
#include "Protocol/ProtocolBase.h"
#include "Protocol/ProtocolRequests.h"
#include "RunInTerminal.h"
#include "lldb/API/SBDefines.h"
#include "lldb/API/SBEnvironment.h"
#include "llvm/Support/Error.h"
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

// Both attach and launch take either a sourcePath or a sourceMap
// argument (or neither), from which we need to set the target.source-map.
void BaseRequestHandler::SetSourceMapFromArguments(
    const llvm::json::Object &arguments) const {
  const char *sourceMapHelp =
      "source must be be an array of two-element arrays, "
      "each containing a source and replacement path string.\n";

  std::string sourceMapCommand;
  llvm::raw_string_ostream strm(sourceMapCommand);
  strm << "settings set target.source-map ";
  const auto sourcePath = GetString(arguments, "sourcePath").value_or("");

  // sourceMap is the new, more general form of sourcePath and overrides it.
  constexpr llvm::StringRef sourceMapKey = "sourceMap";

  if (const auto *sourceMapArray = arguments.getArray(sourceMapKey)) {
    for (const auto &value : *sourceMapArray) {
      const auto *mapping = value.getAsArray();
      if (mapping == nullptr || mapping->size() != 2 ||
          (*mapping)[0].kind() != llvm::json::Value::String ||
          (*mapping)[1].kind() != llvm::json::Value::String) {
        dap.SendOutput(OutputType::Console, llvm::StringRef(sourceMapHelp));
        return;
      }
      const auto mapFrom = GetAsString((*mapping)[0]);
      const auto mapTo = GetAsString((*mapping)[1]);
      strm << "\"" << mapFrom << "\" \"" << mapTo << "\" ";
    }
  } else if (const auto *sourceMapObj = arguments.getObject(sourceMapKey)) {
    for (const auto &[key, value] : *sourceMapObj) {
      if (value.kind() == llvm::json::Value::String) {
        strm << "\"" << key.str() << "\" \"" << GetAsString(value) << "\" ";
      }
    }
  } else {
    if (ObjectContainsKey(arguments, sourceMapKey)) {
      dap.SendOutput(OutputType::Console, llvm::StringRef(sourceMapHelp));
      return;
    }
    if (sourcePath.empty())
      return;
    // Do any source remapping needed before we create our targets
    strm << "\".\" \"" << sourcePath << "\"";
  }
  if (!sourceMapCommand.empty()) {
    dap.RunLLDBCommands("Setting source map:", {sourceMapCommand});
  }
}

static llvm::Error
RunInTerminal(DAP &dap, const protocol::LaunchRequestArguments &arguments) {
  if (!dap.clientFeatures.contains(
          protocol::eClientFeatureRunInTerminalRequest))
    return llvm::make_error<DAPError>("Cannot use runInTerminal, feature is "
                                      "not supported by the connected client");

  if (!arguments.configuration.program ||
      arguments.configuration.program->empty())
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
      *arguments.configuration.program, arguments.args, arguments.env,
      arguments.cwd.value_or(""), comm_file.m_path, debugger_pid);
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
  // we return the debugger to its async state.
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
  const auto cwd = arguments.cwd.value_or("");
  if (!cwd.empty())
    launch_info.SetWorkingDirectory(cwd.data());

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

  if (arguments.runInTerminal) {
    if (llvm::Error err = RunInTerminal(dap, arguments))
      return err;
  } else if (launchCommands.empty()) {
    lldb::SBError error;
    // Disable async events so the launch will be successful when we return from
    // the launch call and the launch will happen synchronously
    ScopeSyncMode scope_sync_mode(dap.debugger);
    dap.target.Launch(launch_info, error);
    if (error.Fail())
      return llvm::make_error<DAPError>(error.GetCString());
  } else {
    // Set the launch info so that run commands can access the configured
    // launch details.
    dap.target.SetLaunchInfo(launch_info);
    if (llvm::Error err = dap.RunLaunchCommands(launchCommands))
      return err;

    // The custom commands might have created a new target so we should use the
    // selected target after these commands are run.
    dap.target = dap.debugger.GetSelectedTarget();
    // Make sure the process is launched and stopped at the entry point before
    // proceeding as the launch commands are not run using the synchronous
    // mode.
    lldb::SBError error = dap.WaitForProcessToStop(arguments.timeout);
    if (error.Fail())
      return llvm::make_error<DAPError>(error.GetCString());
  }

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
