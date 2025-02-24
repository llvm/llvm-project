//===-- RunInTerminal.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RunInTerminal.h"
#include "JSONUtils.h"

#include <chrono>
#include <future>

#include "llvm/Support/FileSystem.h"

using namespace llvm;

namespace lldb_dap {

const RunInTerminalMessagePid *RunInTerminalMessage::GetAsPidMessage() const {
  return static_cast<const RunInTerminalMessagePid *>(this);
}

const RunInTerminalMessageError *
RunInTerminalMessage::GetAsErrorMessage() const {
  return static_cast<const RunInTerminalMessageError *>(this);
}

RunInTerminalMessage::RunInTerminalMessage(RunInTerminalMessageKind kind)
    : kind(kind) {}

RunInTerminalMessagePid::RunInTerminalMessagePid(lldb::pid_t pid)
    : RunInTerminalMessage(eRunInTerminalMessageKindPID), pid(pid) {}

json::Value RunInTerminalMessagePid::ToJSON() const {
  return json::Object{{"kind", "pid"}, {"pid", static_cast<int64_t>(pid)}};
}

RunInTerminalMessageError::RunInTerminalMessageError(StringRef error)
    : RunInTerminalMessage(eRunInTerminalMessageKindError), error(error) {}

json::Value RunInTerminalMessageError::ToJSON() const {
  return json::Object{{"kind", "error"}, {"value", error}};
}

RunInTerminalMessageDidAttach::RunInTerminalMessageDidAttach()
    : RunInTerminalMessage(eRunInTerminalMessageKindDidAttach) {}

json::Value RunInTerminalMessageDidAttach::ToJSON() const {
  return json::Object{{"kind", "didAttach"}};
}

static Expected<RunInTerminalMessageUP>
ParseJSONMessage(const json::Value &json) {
  if (const json::Object *obj = json.getAsObject()) {
    if (std::optional<StringRef> kind = obj->getString("kind")) {
      if (*kind == "pid") {
        if (std::optional<int64_t> pid = obj->getInteger("pid"))
          return std::make_unique<RunInTerminalMessagePid>(
              static_cast<lldb::pid_t>(*pid));
      } else if (*kind == "error") {
        if (std::optional<StringRef> error = obj->getString("error"))
          return std::make_unique<RunInTerminalMessageError>(*error);
      } else if (*kind == "didAttach") {
        return std::make_unique<RunInTerminalMessageDidAttach>();
      }
    }
  }

  return createStringError(inconvertibleErrorCode(),
                           "Incorrect JSON message: " + JSONToString(json));
}

static Expected<RunInTerminalMessageUP>
GetNextMessage(FifoFileIO &io, std::chrono::milliseconds timeout) {
  if (Expected<json::Value> json = io.ReadJSON(timeout))
    return ParseJSONMessage(*json);
  else
    return json.takeError();
}

static Error ToError(const RunInTerminalMessage &message) {
  if (message.kind == eRunInTerminalMessageKindError)
    return createStringError(inconvertibleErrorCode(),
                             message.GetAsErrorMessage()->error);
  return createStringError(inconvertibleErrorCode(),
                           "Unexpected JSON message: " +
                               JSONToString(message.ToJSON()));
}

RunInTerminalLauncherCommChannel::RunInTerminalLauncherCommChannel(
    FifoFile &comm_file)
    : m_io(std::move(comm_file), "debug adapter") {}

Error RunInTerminalLauncherCommChannel::WaitUntilDebugAdapterAttaches(
    std::chrono::milliseconds timeout) {
  if (Expected<RunInTerminalMessageUP> message =
          GetNextMessage(m_io, timeout)) {
    if (message.get()->kind == eRunInTerminalMessageKindDidAttach)
      return Error::success();
    else
      return ToError(*message.get());
  } else
    return message.takeError();
}

Error RunInTerminalLauncherCommChannel::NotifyPid(lldb::pid_t pid) {
  return m_io.SendJSON(RunInTerminalMessagePid(pid).ToJSON());
}

void RunInTerminalLauncherCommChannel::NotifyError(StringRef error) {
  if (Error err = m_io.SendJSON(RunInTerminalMessageError(error).ToJSON(),
                                std::chrono::seconds(2)))
    llvm::errs() << llvm::toString(std::move(err)) << "\n";
}

RunInTerminalDebugAdapterCommChannel::RunInTerminalDebugAdapterCommChannel(
    FifoFile &comm_file)
    : m_io(std::move(comm_file), "runInTerminal launcher") {}

Error RunInTerminalDebugAdapterCommChannel::WaitForLauncher() {
  return m_io.WaitForPeer();
}

// Can't use \a std::future<llvm::Error> because it doesn't compile on Windows
std::future<lldb::SBError>
RunInTerminalDebugAdapterCommChannel::NotifyDidAttach() {
  return std::async(std::launch::async, [&]() {
    lldb::SBError error;
    if (llvm::Error err =
            m_io.SendJSON(RunInTerminalMessageDidAttach().ToJSON()))
      error.SetErrorString(llvm::toString(std::move(err)).c_str());
    return error;
  });
}

Expected<lldb::pid_t> RunInTerminalDebugAdapterCommChannel::GetLauncherPid() {
  if (Expected<RunInTerminalMessageUP> message =
          GetNextMessage(m_io, std::chrono::seconds(20))) {
    if (message.get()->kind == eRunInTerminalMessageKindPID)
      return message.get()->GetAsPidMessage()->pid;
    return ToError(*message.get());
  } else {
    return message.takeError();
  }
}

std::string RunInTerminalDebugAdapterCommChannel::GetLauncherError() {
  // We know there's been an error, so a small timeout is enough.
  if (Expected<RunInTerminalMessageUP> message =
          GetNextMessage(m_io, std::chrono::seconds(1)))
    return toString(ToError(*message.get()));
  else
    return toString(message.takeError());
}

Expected<std::shared_ptr<FifoFile>> CreateRunInTerminalCommFile() {
  int comm_fd;
  SmallString<256> comm_file;
  if (auto error = createUniqueNamedPipe("lldb-dap-run-in-terminal-comm", "",
                                         comm_fd, comm_file))
    return error;
  FILE *cf = fdopen(comm_fd, "r+");
  // There is no portable way to conjure an ofstream from HANDLE, so use FILE *
  // llvm::raw_fd_stream does not support getline() and there is no
  // llvm::buffer_istream

  if (cf == NULL)
    return createStringError(std::error_code(errno, std::generic_category()),
                             "Error converting file descriptor to C FILE for "
                             "runInTerminal comm-file");
  if (setvbuf(cf, NULL, _IONBF, 0))
    return createStringError(std::error_code(errno, std::generic_category()),
                             "Error setting unbuffered mode on C FILE");
  return std::make_shared<FifoFile>(comm_file, cf);
}

} // namespace lldb_dap
