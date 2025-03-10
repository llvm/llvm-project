//===-- DataBreakpointInfoRequestHandler.cpp ------------------------------===//
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
#include "lldb/API/SBMemoryRegionInfo.h"
#include "llvm/ADT/StringExtras.h"

namespace lldb_dap {

// "DataBreakpointInfoRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Obtains information on a possible data breakpoint that
//     could be set on an expression or variable.\nClients should only call this
//     request if the corresponding capability `supportsDataBreakpoints` is
//     true.", "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "dataBreakpointInfo" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/DataBreakpointInfoArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "DataBreakpointInfoArguments": {
//   "type": "object",
//   "description": "Arguments for `dataBreakpointInfo` request.",
//   "properties": {
//     "variablesReference": {
//       "type": "integer",
//       "description": "Reference to the variable container if the data
//       breakpoint is requested for a child of the container. The
//       `variablesReference` must have been obtained in the current suspended
//       state. See 'Lifetime of Object References' in the Overview section for
//       details."
//     },
//     "name": {
//       "type": "string",
//       "description": "The name of the variable's child to obtain data
//       breakpoint information for.\nIf `variablesReference` isn't specified,
//       this can be an expression."
//     },
//     "frameId": {
//       "type": "integer",
//       "description": "When `name` is an expression, evaluate it in the scope
//       of this stack frame. If not specified, the expression is evaluated in
//       the global scope. When `variablesReference` is specified, this property
//       has no effect."
//     }
//   },
//   "required": [ "name" ]
// },
// "DataBreakpointInfoResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to `dataBreakpointInfo` request.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "dataId": {
//             "type": [ "string", "null" ],
//             "description": "An identifier for the data on which a data
//             breakpoint can be registered with the `setDataBreakpoints`
//             request or null if no data breakpoint is available. If a
//             `variablesReference` or `frameId` is passed, the `dataId` is
//             valid in the current suspended state, otherwise it's valid
//             indefinitely. See 'Lifetime of Object References' in the Overview
//             section for details. Breakpoints set using the `dataId` in the
//             `setDataBreakpoints` request may outlive the lifetime of the
//             associated `dataId`."
//           },
//           "description": {
//             "type": "string",
//             "description": "UI string that describes on what data the
//             breakpoint is set on or why a data breakpoint is not available."
//           },
//           "accessTypes": {
//             "type": "array",
//             "items": {
//               "$ref": "#/definitions/DataBreakpointAccessType"
//             },
//             "description": "Attribute lists the available access types for a
//             potential data breakpoint. A UI client could surface this
//             information."
//           },
//           "canPersist": {
//             "type": "boolean",
//             "description": "Attribute indicates that a potential data
//             breakpoint could be persisted across sessions."
//           }
//         },
//         "required": [ "dataId", "description" ]
//       }
//     },
//     "required": [ "body" ]
//   }]
// }
void DataBreakpointInfoRequestHandler::operator()(
    const llvm::json::Object &request) const {
  llvm::json::Object response;
  FillResponse(request, response);
  llvm::json::Object body;
  lldb::SBError error;
  llvm::json::Array accessTypes{"read", "write", "readWrite"};
  const auto *arguments = request.getObject("arguments");
  const auto variablesReference =
      GetInteger<uint64_t>(arguments, "variablesReference").value_or(0);
  llvm::StringRef name = GetString(arguments, "name");
  lldb::SBFrame frame = dap.GetLLDBFrame(*arguments);
  lldb::SBValue variable = dap.variables.FindVariable(variablesReference, name);
  std::string addr, size;

  if (variable.IsValid()) {
    lldb::addr_t load_addr = variable.GetLoadAddress();
    size_t byte_size = variable.GetByteSize();
    if (load_addr == LLDB_INVALID_ADDRESS) {
      body.try_emplace("dataId", nullptr);
      body.try_emplace("description",
                       "does not exist in memory, its location is " +
                           std::string(variable.GetLocation()));
    } else if (byte_size == 0) {
      body.try_emplace("dataId", nullptr);
      body.try_emplace("description", "variable size is 0");
    } else {
      addr = llvm::utohexstr(load_addr);
      size = llvm::utostr(byte_size);
    }
  } else if (variablesReference == 0 && frame.IsValid()) {
    lldb::SBValue value = frame.EvaluateExpression(name.data());
    if (value.GetError().Fail()) {
      lldb::SBError error = value.GetError();
      const char *error_cstr = error.GetCString();
      body.try_emplace("dataId", nullptr);
      body.try_emplace("description", error_cstr && error_cstr[0]
                                          ? std::string(error_cstr)
                                          : "evaluation failed");
    } else {
      uint64_t load_addr = value.GetValueAsUnsigned();
      lldb::SBData data = value.GetPointeeData();
      if (data.IsValid()) {
        size = llvm::utostr(data.GetByteSize());
        addr = llvm::utohexstr(load_addr);
        lldb::SBMemoryRegionInfo region;
        lldb::SBError err =
            dap.target.GetProcess().GetMemoryRegionInfo(load_addr, region);
        // Only lldb-server supports "qMemoryRegionInfo". So, don't fail this
        // request if SBProcess::GetMemoryRegionInfo returns error.
        if (err.Success()) {
          if (!(region.IsReadable() || region.IsWritable())) {
            body.try_emplace("dataId", nullptr);
            body.try_emplace("description",
                             "memory region for address " + addr +
                                 " has no read or write permissions");
          }
        }
      } else {
        body.try_emplace("dataId", nullptr);
        body.try_emplace("description",
                         "unable to get byte size for expression: " +
                             name.str());
      }
    }
  } else {
    body.try_emplace("dataId", nullptr);
    body.try_emplace("description", "variable not found: " + name.str());
  }

  if (!body.getObject("dataId")) {
    body.try_emplace("dataId", addr + "/" + size);
    body.try_emplace("accessTypes", std::move(accessTypes));
    body.try_emplace("description",
                     size + " bytes at " + addr + " " + name.str());
  }
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

} // namespace lldb_dap
