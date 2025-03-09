//===-- GoToTargetsRequestHandler.cpp
//--------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"

#include "JSONUtils.h"

#include <lldb/API/SBStream.h>

namespace lldb_dap {

//  "GotoTargetsRequest": {
//    "allOf": [ { "$ref": "#/definitions/Request" }, {
//      "type": "object",
//      "description": "This request retrieves the possible goto targets for the
//      specified source location.\nThese targets can be used in the `goto`
//      request.\nClients should only call this request if the corresponding
//      capability `supportsGotoTargetsRequest` is true.", "properties": {
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
  const auto *arguments = request.getObject("arguments");
  const auto *source = arguments->getObject("source");
  const std::string path = GetString(source, "path").str();

  const auto goto_line = GetInteger<uint64_t>(arguments, "line").value_or(0u);
  const auto goto_column =
      GetInteger<uint64_t>(arguments, "column").value_or(0u);

  lldb::SBLineEntry line_entry{};
  const lldb::SBFileSpec file_spec(path.c_str(), true);
  line_entry.SetFileSpec(file_spec);
  line_entry.SetLine(goto_line);
  line_entry.SetColumn(goto_column);

  const auto target_id = dap.goto_id_map.InsertLineEntry(line_entry);
  llvm::json::Array response_targets;
  const auto target_line = line_entry.GetLine();
  const auto target_column = line_entry.GetColumn();
  auto target = llvm::json::Object();
  target.try_emplace("id", target_id);

  lldb::SBStream stream;
  line_entry.GetDescription(stream);
  target.try_emplace("label",
                     llvm::StringRef(stream.GetData(), stream.GetSize()));
  target.try_emplace("column", target_column);
  target.try_emplace("line", target_line);

  response_targets.push_back(std::move(target));
  llvm::json::Object body;
  body.try_emplace("targets", std::move(response_targets));
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}
} // namespace lldb_dap
