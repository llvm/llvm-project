//===-- WriteMemoryRequestHandler.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "JSONUtils.h"
#include "RequestHandler.h"
#include "lldb/API/SBMemoryRegionInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Base64.h"

namespace lldb_dap {

// "WriteMemoryRequest": {
//       "allOf": [ { "$ref": "#/definitions/Request" }, {
//         "type": "object",
//         "description": "Writes bytes to memory at the provided location.\n
//         Clients should only call this request if the corresponding
//         capability `supportsWriteMemoryRequest` is true.",
//         "properties": {
//           "command": {
//             "type": "string",
//             "enum": [ "writeMemory" ]
//           },
//           "arguments": {
//             "$ref": "#/definitions/WriteMemoryArguments"
//           }
//         },
//         "required": [ "command", "arguments" ]
//       }]
//     },
//     "WriteMemoryArguments": {
//       "type": "object",
//       "description": "Arguments for `writeMemory` request.",
//       "properties": {
//         "memoryReference": {
//           "type": "string",
//           "description": "Memory reference to the base location to which
//           data should be written."
//         },
//         "offset": {
//           "type": "integer",
//           "description": "Offset (in bytes) to be applied to the reference
//           location before writing data. Can be negative."
//         },
//         "allowPartial": {
//           "type": "boolean",
//           "description": "Property to control partial writes. If true, the
//           debug adapter should attempt to write memory even if the entire
//           memory region is not writable. In such a case the debug adapter
//           should stop after hitting the first byte of memory that cannot be
//           written and return the number of bytes written in the response
//           via the `offset` and `bytesWritten` properties.\nIf false or
//           missing, a debug adapter should attempt to verify the region is
//           writable before writing, and fail the response if it is not."
//         },
//         "data": {
//           "type": "string",
//           "description": "Bytes to write, encoded using base64."
//         }
//       },
//       "required": [ "memoryReference", "data" ]
//     },
//     "WriteMemoryResponse": {
//       "allOf": [ { "$ref": "#/definitions/Response" }, {
//         "type": "object",
//         "description": "Response to `writeMemory` request.",
//         "properties": {
//           "body": {
//             "type": "object",
//             "properties": {
//               "offset": {
//                 "type": "integer",
//                 "description": "Property that should be returned when
//                 `allowPartial` is true to indicate the offset of the first
//                 byte of data successfully written. Can be negative."
//               },
//               "bytesWritten": {
//                 "type": "integer",
//                 "description": "Property that should be returned when
//                 `allowPartial` is true to indicate the number of bytes
//                 starting from address that were successfully written."
//               }
//             }
//           }
//         }
//       }]
//     },
void WriteMemoryRequestHandler::operator()(
    const llvm::json::Object &request) const {
  llvm::json::Object response;
  FillResponse(request, response);

  auto arguments = request.getObject("arguments");
  llvm::StringRef memoryReference =
      GetString(arguments, "memoryReference").value_or("");

  auto addr_opt = DecodeMemoryReference(memoryReference);
  if (!addr_opt.has_value()) {
    response["success"] = false;
    response["message"] =
        "Malformed memory reference: " + memoryReference.str();
    dap.SendJSON(llvm::json::Value(std::move(response)));
    return;
  }

  lldb::addr_t address = *addr_opt;
  lldb::addr_t address_offset =
      address + GetInteger<uint64_t>(arguments, "offset").value_or(0);

  llvm::StringRef data64 = GetString(arguments, "data").value_or("");
  if (data64.empty()) {
    response["success"] = false;
    EmplaceSafeString(response, "message","Data cannot be empty value. Provide valid data");
    dap.SendJSON(llvm::json::Value(std::move(response)));
    return;
  }

  // The VSCode IDE or other DAP clients send memory data as a Base64 string.
  // This function decodes it into raw binary before writing it to the target
  // process memory.
  std::vector<char> output;
  auto decode_error = llvm::decodeBase64(data64, output);

  if (decode_error) {
    response["success"] = false;
    EmpleceSafeErrorMessage(dap, response, "message",
                      llvm::toString(std::move(decode_error)).c_str());
    dap.SendJSON(llvm::json::Value(std::move(response)));
    return;
  }

  bool allowPartial = GetBoolean(arguments, "allowPartial").value_or(true);
  lldb::SBError write_error;
  uint64_t bytes_written = 0;

  // Write the memory
  if (!output.empty()) {
    lldb::SBProcess process = dap.target.GetProcess();
    // If 'allowPartial' is false or missing, a debug adapter should attempt to
    // verify the region is writable before writing, and fail the response if it
    // is not.
    if (allowPartial == false) {

      lldb::SBMemoryRegionInfo region_info;
      lldb::SBError error =
          process.GetMemoryRegionInfo(address_offset, region_info);
      if (!error.Success() || !region_info.IsWritable()) {
        response["success"] = false;
        EmplaceSafeString(response, "message",
                          "Memory 0x" + llvm::utohexstr(address_offset) +
                              " region is not writable");
        dap.SendJSON(llvm::json::Value(std::move(response)));
        return;
      }
    }

    bytes_written =
        process.WriteMemory(address_offset, static_cast<void *>(output.data()),
                            output.size(), write_error);
  }

  if (bytes_written == 0) {
    response["success"] = false;
    EmplaceSafeString(response, "message", write_error.GetCString());
    dap.SendJSON(llvm::json::Value(std::move(response)));
    return;
  }

  llvm::json::Object body;
  body.try_emplace("bytesWritten", std::move(bytes_written));

  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

} // namespace lldb_dap
