//===-- WriteMemoryRequestHandler.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "EventHelper.h"
#include "JSONUtils.h"
#include "Protocol/ProtocolEvents.h"
#include "RequestHandler.h"
#include "lldb/API/SBMemoryRegionInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Base64.h"

using namespace lldb_dap::protocol;

namespace lldb_dap {

// Writes bytes to memory at the provided location.
//
// Clients should only call this request if the corresponding capability
//  supportsWriteMemoryRequest is true.
llvm::Expected<WriteMemoryResponseBody>
WriteMemoryRequestHandler::Run(const WriteMemoryArguments &args) const {
  const lldb::addr_t address = args.memoryReference + args.offset;

  lldb::SBProcess process = dap.target.GetProcess();
  if (!lldb::SBDebugger::StateIsStoppedState(process.GetState()))
    return llvm::make_error<NotStoppedError>();

  if (args.data.empty()) {
    return llvm::make_error<DAPError>(
        "Data cannot be empty value. Provide valid data");
  }

  // The VSCode IDE or other DAP clients send memory data as a Base64 string.
  // This function decodes it into raw binary before writing it to the target
  // process memory.
  std::vector<char> output;
  auto decode_error = llvm::decodeBase64(args.data, output);

  if (decode_error) {
    return llvm::make_error<DAPError>(
        llvm::toString(std::move(decode_error)).c_str());
  }

  lldb::SBError write_error;
  uint64_t bytes_written = 0;

  // Write the memory.
  if (!output.empty()) {
    lldb::SBProcess process = dap.target.GetProcess();
    // If 'allowPartial' is false or missing, a debug adapter should attempt to
    // verify the region is writable before writing, and fail the response if it
    // is not.
    if (!args.allowPartial) {
      // Start checking from the initial write address.
      lldb::addr_t start_address = address;
      // Compute the end of the write range.
      lldb::addr_t end_address = start_address + output.size() - 1;

      while (start_address <= end_address) {
        // Get memory region info for the given address.
        // This provides the region's base, end, and permissions
        // (read/write/executable).
        lldb::SBMemoryRegionInfo region_info;
        lldb::SBError error =
            process.GetMemoryRegionInfo(start_address, region_info);
        // Fail if the region info retrieval fails, is not writable, or the
        // range exceeds the region.
        if (!error.Success() || !region_info.IsWritable()) {
          return llvm::make_error<DAPError>(
              "Memory 0x" + llvm::utohexstr(args.memoryReference) +
              " region is not writable");
        }
        // If the current region covers the full requested range, stop further
        // iterations.
        if (end_address <= region_info.GetRegionEnd()) {
          break;
        }
        // Move to the start of the next memory region.
        start_address = region_info.GetRegionEnd() + 1;
      }
    }

    bytes_written =
        process.WriteMemory(address, static_cast<void *>(output.data()),
                            output.size(), write_error);
  }

  if (bytes_written == 0) {
    return llvm::make_error<DAPError>(write_error.GetCString());
  }
  WriteMemoryResponseBody response;
  response.bytesWritten = bytes_written;

  // Also send invalidated event to signal client that some things
  // (e.g. variables) can be changed.
  SendInvalidatedEvent(dap, {InvalidatedEventBody::eAreaAll});

  return response;
}

} // namespace lldb_dap
