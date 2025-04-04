//===-- ThreadsRequestHandler.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "EventHelper.h"
#include "JSONUtils.h"
#include "RequestHandler.h"
#include "lldb/API/SBError.h"

namespace lldb_dap {

// "ThreadsRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Thread request; value of command field is 'threads'. The
//     request retrieves a list of all threads.", "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "threads" ]
//       }
//     },
//     "required": [ "command" ]
//   }]
// },
// "ThreadsResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'threads' request.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "threads": {
//             "type": "array",
//             "items": {
//               "$ref": "#/definitions/Thread"
//             },
//             "description": "All threads."
//           }
//         },
//         "required": [ "threads" ]
//       }
//     },
//     "required": [ "body" ]
//   }]
// }
void ThreadsRequestHandler::operator()(
    const llvm::json::Object &request) const {
  lldb::SBProcess process = dap.target.GetProcess();
  llvm::json::Object response;
  FillResponse(request, response);

  const auto state = process.GetState();
  const bool stop_and_resume =
      state != lldb::eStateCrashed && state != lldb::eStateStopped;

  if (stop_and_resume) {
    lldb::SBError error = dap.WaitForProcessToStop(1);
    if (error.Fail()) {
      SetError(response, error);
      dap.SendJSON(llvm::json::Value(std::move(response)));
    }
  }

  const uint32_t num_threads = process.GetNumThreads();
  llvm::json::Array threads;
  for (uint32_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
    lldb::SBThread thread = process.GetThreadAtIndex(thread_idx);
    threads.emplace_back(CreateThread(thread, dap.thread_format));
  }

  if (stop_and_resume) {
    lldb::SBError error = process.Continue();
    if (error.Fail()) {
      SetError(response, error);
      dap.SendJSON(llvm::json::Value(std::move(response)));
    }
  }

  response["success"] = llvm::json::Value(threads.size() != 0);
  llvm::json::Object body;
  body.try_emplace("threads", std::move(threads));
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

} // namespace lldb_dap
