//===-- ReadMemoryRequestHandler.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "JSONUtils.h"
#include "RequestHandler.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Base64.h"

namespace lldb_dap {

// Reads bytes from memory at the provided location.
//
// Clients should only call this request if the corresponding capability
// `supportsReadMemoryRequest` is true
llvm::Expected<protocol::ReadMemoryResponse>
ReadMemoryRequestHandler::Run(const protocol::ReadMemoryArguments &args) const {

  const std::optional<lldb::addr_t> addr_opt =
      DecodeMemoryReference(args.memoryReference);
  if (!addr_opt)
    return llvm::make_error<DAPError>(
        llvm::formatv("Malformed memory reference: {}", args.memoryReference));

  const lldb::addr_t raw_address = *addr_opt + args.offset.value_or(0);

  lldb::SBProcess process = dap.target.GetProcess();
  if (!lldb::SBDebugger::StateIsStoppedState(process.GetState()))
    return llvm::make_error<NotStoppedError>();

  const uint64_t count_read = std::max<uint64_t>(args.count, 1);
  // We also need support reading 0 bytes
  // VS Code sends those requests to check if a `memoryReference`
  // can be dereferenced.
  auto buffer = std::vector<uint8_t>(count_read);

  lldb::SBError error;
  const size_t memory_count = dap.target.GetProcess().ReadMemory(
      raw_address, buffer.data(), buffer.size(), error);

  protocol::ReadMemoryResponse response;
  response.address = "0x" + llvm::utohexstr(raw_address);

  // reading memory may fail for multiple reasons. memory not readable,
  // reading out of memory range and gaps in memory. return from
  // the last readable byte.
  if (error.Fail() && (memory_count < count_read)) {
    response.unreadableBytes = count_read - memory_count;
  }

  buffer.resize(std::min<size_t>(memory_count, args.count));
  if (!buffer.empty())
    response.data = llvm::encodeBase64(buffer);

  return response;
}

} // namespace lldb_dap
