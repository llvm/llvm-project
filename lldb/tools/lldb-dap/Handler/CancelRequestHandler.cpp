//===-- CancelRequestHandler.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Handler/RequestHandler.h"
#include "Protocol/ProtocolRequests.h"
#include "llvm/Support/Error.h"

using namespace llvm;
using namespace lldb_dap::protocol;

namespace lldb_dap {

/// The `cancel` request is used by the client in two situations:
///
/// - to indicate that it is no longer interested in the result produced by a
/// specific request issued earlier
/// - to cancel a progress sequence.
///
/// Clients should only call this request if the corresponding capability
/// `supportsCancelRequest` is true.
///
/// This request has a hint characteristic: a debug adapter can only be
/// expected to make a 'best effort' in honoring this request but there are no
/// guarantees.
///
/// The `cancel` request may return an error if it could not cancel
/// an operation but a client should refrain from presenting this error to end
/// users.
///
/// The request that got cancelled still needs to send a response back.
/// This can either be a normal result (`success` attribute true) or an error
/// response (`success` attribute false and the `message` set to `cancelled`).
///
/// Returning partial results from a cancelled request is possible but please
/// note that a client has no generic way for detecting that a response is
/// partial or not.
///
/// The progress that got cancelled still needs to send a `progressEnd` event
/// back.
///
/// A client cannot assume that progress just got cancelled after sending
/// the `cancel` request.
Error CancelRequestHandler::Run(const CancelArguments &arguments) const {
  // Cancel support is built into the DAP::Loop handler for detecting
  // cancellations of pending or inflight requests.
  dap.ClearCancelRequest(arguments);
  return Error::success();
}

} // namespace lldb_dap
