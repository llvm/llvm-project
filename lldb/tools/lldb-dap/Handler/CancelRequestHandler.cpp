//===-- SourceRequestHandler.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Protocol.h"
#include "RequestHandler.h"
#include "llvm/Support/Error.h"

using namespace lldb_dap;
using namespace lldb_dap::protocol;

namespace lldb_dap {

// "CancelRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "The `cancel` request is used by the client in two
//     situations:\n- to indicate that it is no longer interested in the result
//     produced by a specific request issued earlier\n- to cancel a progress
//     sequence.\nClients should only call this request if the corresponding
//     capability `supportsCancelRequest` is true.\nThis request has a hint
//     characteristic: a debug adapter can only be expected to make a 'best
//     effort' in honoring this request but there are no guarantees.\nThe
//     `cancel` request may return an error if it could not cancel an operation
//     but a client should refrain from presenting this error to end users.\nThe
//     request that got cancelled still needs to send a response back. This can
//     either be a normal result (`success` attribute true) or an error response
//     (`success` attribute false and the `message` set to
//     `cancelled`).\nReturning partial results from a cancelled request is
//     possible but please note that a client has no generic way for detecting
//     that a response is partial or not.\nThe progress that got cancelled still
//     needs to send a `progressEnd` event back.\n A client should not assume
//     that progress just got cancelled after sending the `cancel` request.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "cancel" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/CancelArguments"
//       }
//     },
//     "required": [ "command" ]
//   }]
// },
llvm::Expected<CancelResponseBody>
CancelRequestHandler::Run(const CancelArguments &arguments) const {
  /* no-op, simple ack of the request. */
  return nullptr;
}

} // namespace lldb_dap
