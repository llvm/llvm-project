//===-- GoToRequestHandler.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "EventHelper.h"
#include "JSONUtils.h"
#include "Protocol/ProtocolRequests.h"
#include "RequestHandler.h"

namespace lldb_dap {

/// Creates an \p StoppedEvent with the reason \a goto.
static void SendThreadGotoEvent(DAP &dap, lldb::tid_t thread_id) {
  llvm::json::Object event(CreateEventObject("stopped"));
  llvm::json::Object body;
  body.try_emplace("reason", "goto");
  body.try_emplace("description", "Paused on Jump To Cursor");
  body.try_emplace("threadId", thread_id);
  body.try_emplace("preserveFocusHint", false);
  body.try_emplace("allThreadsStopped", true);

  event.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(event)));
}

llvm::Expected<protocol::GotoResponseBody>
GotoRequestHandler::Run(const protocol::GotoArguments &args) const {
  const lldb::tid_t thread_id = args.threadId;
  lldb::SBThread current_thread =
      dap.target.GetProcess().GetThreadByID(thread_id);

  if (!current_thread.IsValid()) {
    return llvm::createStringError(llvm::formatv("Thread id `{0}` is not valid",
                                                 current_thread.GetThreadID()));
  }

  const uint64_t target_id = args.targetId;
  const std::optional<lldb::SBLineEntry> line_entry =
      dap.gotos.GetLineEntry(target_id);
  if (!line_entry) {
    return llvm::createStringError(
        llvm::formatv("Target id `{0}` is not valid", thread_id));
  }

  lldb::SBFileSpec file_spec = line_entry->GetFileSpec();
  const lldb::SBError error =
      current_thread.JumpToLine(file_spec, line_entry->GetLine());

  if (error.Fail()) {
    return llvm::make_error<DAPError>(error.GetCString());
  }

  SendThreadGotoEvent(dap, thread_id);
  return protocol::GotoResponseBody();
}

} // namespace lldb_dap
