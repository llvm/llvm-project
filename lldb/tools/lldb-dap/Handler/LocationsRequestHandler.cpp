//===-- LocationsRequestHandler.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "EventHelper.h"
#include "JSONUtils.h"
#include "LLDBUtils.h"
#include "ProtocolUtils.h"
#include "RequestHandler.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBDeclaration.h"
#include "lldb/API/SBLineEntry.h"

namespace lldb_dap {

// "LocationsRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Looks up information about a location reference
//                     previously returned by the debug adapter.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "locations" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/LocationsArguments"
//       }
//     },
//     "required": [ "command", "arguments" ]
//   }]
// },
// "LocationsArguments": {
//   "type": "object",
//   "description": "Arguments for `locations` request.",
//   "properties": {
//     "locationReference": {
//       "type": "integer",
//       "description": "Location reference to resolve."
//     }
//   },
//   "required": [ "locationReference" ]
// },
// "LocationsResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to `locations` request.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "source": {
//             "$ref": "#/definitions/Source",
//             "description": "The source containing the location; either
//                             `source.path` or `source.sourceReference` must be
//                             specified."
//           },
//           "line": {
//             "type": "integer",
//             "description": "The line number of the location. The client
//                             capability `linesStartAt1` determines whether it
//                             is 0- or 1-based."
//           },
//           "column": {
//             "type": "integer",
//             "description": "Position of the location within the `line`. It is
//                             measured in UTF-16 code units and the client
//                             capability `columnsStartAt1` determines whether
//                             it is 0- or 1-based. If no column is given, the
//                             first position in the start line is assumed."
//           },
//           "endLine": {
//             "type": "integer",
//             "description": "End line of the location, present if the location
//                             refers to a range.  The client capability
//                             `linesStartAt1` determines whether it is 0- or
//                             1-based."
//           },
//           "endColumn": {
//             "type": "integer",
//             "description": "End position of the location within `endLine`,
//                             present if the location refers to a range. It is
//                             measured in UTF-16 code units and the client
//                             capability `columnsStartAt1` determines whether
//                             it is 0- or 1-based."
//           }
//         },
//         "required": [ "source", "line" ]
//       }
//     }
//   }]
// },
void LocationsRequestHandler::operator()(
    const llvm::json::Object &request) const {
  llvm::json::Object response;
  FillResponse(request, response);
  auto *arguments = request.getObject("arguments");

  const auto location_id =
      GetInteger<uint64_t>(arguments, "locationReference").value_or(0);
  // We use the lowest bit to distinguish between value location and declaration
  // location
  auto [var_ref, is_value_location] = UnpackLocation(location_id);
  lldb::SBValue variable = dap.variables.GetVariable(var_ref);
  if (!variable.IsValid()) {
    response["success"] = false;
    response["message"] = "Invalid variable reference";
    dap.SendJSON(llvm::json::Value(std::move(response)));
    return;
  }

  llvm::json::Object body;
  if (is_value_location) {
    // Get the value location
    if (!variable.GetType().IsPointerType() &&
        !variable.GetType().IsReferenceType()) {
      response["success"] = false;
      response["message"] =
          "Value locations are only available for pointers and references";
      dap.SendJSON(llvm::json::Value(std::move(response)));
      return;
    }

    lldb::addr_t raw_addr = variable.GetValueAsAddress();
    lldb::SBAddress addr = dap.target.ResolveLoadAddress(raw_addr);
    lldb::SBLineEntry line_entry = GetLineEntryForAddress(dap.target, addr);

    if (!line_entry.IsValid()) {
      response["success"] = false;
      response["message"] = "Failed to resolve line entry for location";
      dap.SendJSON(llvm::json::Value(std::move(response)));
      return;
    }

    const std::optional<protocol::Source> source =
        CreateSource(line_entry.GetFileSpec());
    if (!source) {
      response["success"] = false;
      response["message"] = "Failed to resolve file path for location";
      dap.SendJSON(llvm::json::Value(std::move(response)));
      return;
    }

    body.try_emplace("source", *source);
    if (int line = line_entry.GetLine())
      body.try_emplace("line", line);
    if (int column = line_entry.GetColumn())
      body.try_emplace("column", column);
  } else {
    // Get the declaration location
    lldb::SBDeclaration decl = variable.GetDeclaration();
    if (!decl.IsValid()) {
      response["success"] = false;
      response["message"] = "No declaration location available";
      dap.SendJSON(llvm::json::Value(std::move(response)));
      return;
    }

    const std::optional<protocol::Source> source =
        CreateSource(decl.GetFileSpec());
    if (!source) {
      response["success"] = false;
      response["message"] = "Failed to resolve file path for location";
      dap.SendJSON(llvm::json::Value(std::move(response)));
      return;
    }

    body.try_emplace("source", *source);
    if (int line = decl.GetLine())
      body.try_emplace("line", line);
    if (int column = decl.GetColumn())
      body.try_emplace("column", column);
  }

  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

} // namespace lldb_dap
