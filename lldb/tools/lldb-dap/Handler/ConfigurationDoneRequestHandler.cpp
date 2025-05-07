//===-- ConfigurationDoneRequestHandler..cpp ------------------------------===//
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

namespace lldb_dap {

// "ConfigurationDoneRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//             "type": "object",
//             "description": "ConfigurationDone request; value of command field
//             is 'configurationDone'.\nThe client of the debug protocol must
//             send this request at the end of the sequence of configuration
//             requests (which was started by the InitializedEvent).",
//             "properties": {
//             "command": {
//             "type": "string",
//             "enum": [ "configurationDone" ]
//             },
//             "arguments": {
//             "$ref": "#/definitions/ConfigurationDoneArguments"
//             }
//             },
//             "required": [ "command" ]
//             }]
// },
// "ConfigurationDoneArguments": {
//   "type": "object",
//   "description": "Arguments for 'configurationDone' request.\nThe
//   configurationDone request has no standardized attributes."
// },
// "ConfigurationDoneResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//             "type": "object",
//             "description": "Response to 'configurationDone' request. This is
//             just an acknowledgement, so no body field is required."
//             }]
// },

void ConfigurationDoneRequestHandler::operator()(
    const llvm::json::Object &request) const {
  llvm::json::Object response;
  FillResponse(request, response);
  dap.SendJSON(llvm::json::Value(std::move(response)));
  dap.configuration_done_sent = true;
  if (dap.stop_at_entry)
    SendThreadStoppedEvent(dap);
  else {
    // Client requests the baseline of currently existing threads after
    // a successful launch or attach by sending a 'threads' request
    // right after receiving the configurationDone response.
    // Obtain the list of threads before we resume the process
    dap.initial_thread_list =
        GetThreads(dap.target.GetProcess(), dap.thread_format);
    dap.target.GetProcess().Continue();
  }
}

} // namespace lldb_dap
