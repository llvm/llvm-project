//===-- GoToTargetsRequestHandler.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"

#include "JSONUtils.h"

#include <lldb/API/SBBreakpointLocation.h>
#include <lldb/API/SBListener.h>
#include <lldb/API/SBStream.h>

namespace lldb_dap {

static llvm::SmallVector<lldb::SBLineEntry>
GetLineValidEntry(DAP &dap, const lldb::SBFileSpec &file_spec, uint32_t line) {
  // disable breakpoint listeners so they do not send events to the DAP client.
  lldb::SBListener listener = dap.debugger.GetListener();
  lldb::SBBroadcaster broadcaster = dap.target.GetBroadcaster();
  constexpr auto event_mask = lldb::SBTarget::eBroadcastBitBreakpointChanged;
  listener.StopListeningForEvents(broadcaster, event_mask);

  // create a breakpoint to resolve the line if it is on an empty line.
  lldb::SBBreakpoint goto_bp =
      dap.target.BreakpointCreateByLocation(file_spec, line);
  if (!goto_bp.IsValid())
    return {};

  llvm::SmallVector<lldb::SBLineEntry> entry_locations{};
  const size_t resolved_count = goto_bp.GetNumResolvedLocations();
  for (size_t idx = 0; idx < resolved_count; ++idx) {
    lldb::SBBreakpointLocation location = goto_bp.GetLocationAtIndex(idx);
    if (!location.IsValid())
      continue;

    lldb::SBAddress addr = location.GetAddress();
    if (!addr.IsValid())
      continue;

    lldb::SBLineEntry line_entry = addr.GetLineEntry();
    if (!line_entry.IsValid())
      continue;

    entry_locations.push_back(line_entry);
  }

  // clean up;
  dap.target.BreakpointDelete(goto_bp.GetID());
  listener.StartListeningForEvents(broadcaster, event_mask);

  return entry_locations;
}

//  "GotoTargetsRequest": {
//    "allOf": [ { "$ref": "#/definitions/Request" }, {
//      "type": "object",
//      "description": "This request retrieves the possible goto targets for the
//      specified source location.\nThese targets can be used in the `goto`
//      request.\nClients should only call this request if the corresponding
//      capability `supportsGotoTargetsRequest` is true.",
//.     "properties": {
//        "command": {
//          "type": "string",
//          "enum": [ "gotoTargets" ]
//        },
//        "arguments": {
//          "$ref": "#/definitions/GotoTargetsArguments"
//        }
//      },
//      "required": [ "command", "arguments"  ]
//    }]
//  },
//  "GotoTargetsArguments": {
//    "type": "object",
//    "description": "Arguments for `gotoTargets` request.",
//    "properties": {
//      "source": {
//        "$ref": "#/definitions/Source",
//        "description": "The source location for which the goto targets are
//        determined."
//      },
//      "line": {
//        "type": "integer",
//        "description": "The line location for which the goto targets are
//        determined."
//      },
//      "column": {
//        "type": "integer",
//        "description": "The position within `line` for which the goto targets
//        are determined. It is measured in UTF-16 code units and the client
//        capability `columnsStartAt1` determines whether it is 0- or 1-based."
//      }
//    },
//    "required": [ "source", "line" ]
//  },
//  "GotoTargetsResponse": {
//    "allOf": [ { "$ref": "#/definitions/Response" }, {
//      "type": "object",
//      "description": "Response to `gotoTargets` request.",
//      "properties": {
//        "body": {
//          "type": "object",
//          "properties": {
//            "targets": {
//              "type": "array",
//              "items": {
//                "$ref": "#/definitions/GotoTarget"
//              },
//              "description": "The possible goto targets of the specified
//              location."
//            }
//          },
//          "required": [ "targets" ]
//        }
//      },
//      "required": [ "body" ]
//    }]
//  },
void GoToTargetsRequestHandler::operator()(
    const llvm::json::Object &request) const {
  llvm::json::Object response;
  FillResponse(request, response);

  const llvm::json::Object *arguments = request.getObject("arguments");
  const llvm::json::Object *source = arguments->getObject("source");
  const std::string path = GetString(source, "path").str();
  const lldb::SBFileSpec file_spec(path.c_str(), true);
  const uint64_t goto_line =
      GetInteger<uint64_t>(arguments, "line").value_or(1U);

  llvm::json::Object body;

  llvm::SmallVector<lldb::SBLineEntry> goto_locations =
      GetLineValidEntry(dap, file_spec, goto_line);
  if (goto_locations.empty()) {
    response["success"] = false;
    response["message"] = "Invalid jump location";
  } else {
    llvm::json::Array response_targets;
    for (lldb::SBLineEntry &line_entry : goto_locations) {
      const uint64_t target_id = dap.goto_id_map.InsertLineEntry(line_entry);
      const uint32_t target_line = line_entry.GetLine();
      auto target = llvm::json::Object();
      target.try_emplace("id", target_id);

      lldb::SBStream stream;
      line_entry.GetDescription(stream);
      target.try_emplace("label",
                         llvm::StringRef(stream.GetData(), stream.GetSize()));
      target.try_emplace("line", target_line);
      response_targets.push_back(std::move(target));
    }

    body.try_emplace("targets", std::move(response_targets));
  }

  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

} // namespace lldb_dap
