//===-- ReadMemoryRequestHandler.cpp --------------------------------------===//
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
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Base64.h"

namespace lldb_dap {

// "ReadMemoryRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Reads bytes from memory at the provided location. Clients
//                     should only call this request if the corresponding
//                     capability `supportsReadMemoryRequest` is true.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "readMemory" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/ReadMemoryArguments"
//       }
//     },
//     "required": [ "command", "arguments" ]
//   }]
// },
// "ReadMemoryArguments": {
//   "type": "object",
//   "description": "Arguments for `readMemory` request.",
//   "properties": {
//     "memoryReference": {
//       "type": "string",
//       "description": "Memory reference to the base location from which data
//                       should be read."
//     },
//     "offset": {
//       "type": "integer",
//       "description": "Offset (in bytes) to be applied to the reference
//                       location before reading data. Can be negative."
//     },
//     "count": {
//       "type": "integer",
//       "description": "Number of bytes to read at the specified location and
//                       offset."
//     }
//   },
//   "required": [ "memoryReference", "count" ]
// },
// "ReadMemoryResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to `readMemory` request.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "address": {
//             "type": "string",
//             "description": "The address of the first byte of data returned.
//                             Treated as a hex value if prefixed with `0x`, or
//                             as a decimal value otherwise."
//           },
//           "unreadableBytes": {
//             "type": "integer",
//             "description": "The number of unreadable bytes encountered after
//                             the last successfully read byte.\nThis can be
//                             used to determine the number of bytes that should
//                             be skipped before a subsequent
//             `readMemory` request succeeds."
//           },
//           "data": {
//             "type": "string",
//             "description": "The bytes read from memory, encoded using base64.
//                             If the decoded length of `data` is less than the
//                             requested `count` in the original `readMemory`
//                             request, and `unreadableBytes` is zero or
//                             omitted, then the client should assume it's
//                             reached the end of readable memory."
//           }
//         },
//         "required": [ "address" ]
//       }
//     }
//   }]
// },
void ReadMemoryRequestHandler::operator()(
    const llvm::json::Object &request) const {
  llvm::json::Object response;
  FillResponse(request, response);
  auto *arguments = request.getObject("arguments");

  llvm::StringRef memoryReference = GetString(arguments, "memoryReference");
  auto addr_opt = DecodeMemoryReference(memoryReference);
  if (!addr_opt.has_value()) {
    response["success"] = false;
    response["message"] =
        "Malformed memory reference: " + memoryReference.str();
    dap.SendJSON(llvm::json::Value(std::move(response)));
    return;
  }
  lldb::addr_t addr_int = *addr_opt;
  addr_int += GetInteger<uint64_t>(arguments, "offset").value_or(0);
  const uint64_t count_requested =
      GetInteger<uint64_t>(arguments, "count").value_or(0);

  // We also need support reading 0 bytes
  // VS Code sends those requests to check if a `memoryReference`
  // can be dereferenced.
  const uint64_t count_read = std::max<uint64_t>(count_requested, 1);
  std::vector<uint8_t> buf;
  buf.resize(count_read);
  lldb::SBError error;
  lldb::SBAddress addr{addr_int, dap.target};
  size_t count_result =
      dap.target.ReadMemory(addr, buf.data(), count_read, error);
  if (count_result == 0) {
    response["success"] = false;
    EmplaceSafeString(response, "message", error.GetCString());
    dap.SendJSON(llvm::json::Value(std::move(response)));
    return;
  }
  buf.resize(std::min<size_t>(count_result, count_requested));

  llvm::json::Object body;
  std::string formatted_addr = "0x" + llvm::utohexstr(addr_int);
  body.try_emplace("address", formatted_addr);
  body.try_emplace("data", llvm::encodeBase64(buf));
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

} // namespace lldb_dap
