//===-- SetVariableRequestHandler.cpp -------------------------------------===//
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

namespace lldb_dap {

// "SetVariableRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "setVariable request; value of command field is
//     'setVariable'. Set the variable with the given name in the variable
//     container to a new value.", "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "setVariable" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/SetVariableArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "SetVariableArguments": {
//   "type": "object",
//   "description": "Arguments for 'setVariable' request.",
//   "properties": {
//     "variablesReference": {
//       "type": "integer",
//       "description": "The reference of the variable container."
//     },
//     "name": {
//       "type": "string",
//       "description": "The name of the variable."
//     },
//     "value": {
//       "type": "string",
//       "description": "The value of the variable."
//     },
//     "format": {
//       "$ref": "#/definitions/ValueFormat",
//       "description": "Specifies details on how to format the response value."
//     }
//   },
//   "required": [ "variablesReference", "name", "value" ]
// },
// "SetVariableResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'setVariable' request.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "value": {
//             "type": "string",
//             "description": "The new value of the variable."
//           },
//           "type": {
//             "type": "string",
//             "description": "The type of the new value. Typically shown in the
//             UI when hovering over the value."
//           },
//           "variablesReference": {
//             "type": "number",
//             "description": "If variablesReference is > 0, the new value is
//             structured and its children can be retrieved by passing
//             variablesReference to the VariablesRequest."
//           },
//           "namedVariables": {
//             "type": "number",
//             "description": "The number of named child variables. The client
//             can use this optional information to present the variables in a
//             paged UI and fetch them in chunks."
//           },
//           "indexedVariables": {
//             "type": "number",
//             "description": "The number of indexed child variables. The client
//             can use this optional information to present the variables in a
//             paged UI and fetch them in chunks."
//           },
//           "valueLocationReference": {
//             "type": "integer",
//             "description": "A reference that allows the client to request the
//             location where the new value is declared. For example, if the new
//             value is function pointer, the adapter may be able to look up the
//             function's location. This should be present only if the adapter
//             is likely to be able to resolve the location.\n\nThis reference
//             shares the same lifetime as the `variablesReference`. See
//             'Lifetime of Object References' in the Overview section for
//             details."
//           }
//         },
//         "required": [ "value" ]
//       }
//     },
//     "required": [ "body" ]
//   }]
// }
void SetVariableRequestHandler::operator()(
    const llvm::json::Object &request) const {
  llvm::json::Object response;
  FillResponse(request, response);
  llvm::json::Array variables;
  llvm::json::Object body;
  const auto *arguments = request.getObject("arguments");
  // This is a reference to the containing variable/scope
  const auto variablesReference =
      GetUnsigned(arguments, "variablesReference", 0);
  llvm::StringRef name = GetString(arguments, "name");

  const auto value = GetString(arguments, "value");
  // Set success to false just in case we don't find the variable by name
  response.try_emplace("success", false);

  lldb::SBValue variable;

  // The "id" is the unique integer ID that is unique within the enclosing
  // variablesReference. It is optionally added to any "interface Variable"
  // objects to uniquely identify a variable within an enclosing
  // variablesReference. It helps to disambiguate between two variables that
  // have the same name within the same scope since the "setVariables" request
  // only specifies the variable reference of the enclosing scope/variable, and
  // the name of the variable. We could have two shadowed variables with the
  // same name in "Locals" or "Globals". In our case the "id" absolute index
  // of the variable within the dap.variables list.
  const auto id_value = GetUnsigned(arguments, "id", UINT64_MAX);
  if (id_value != UINT64_MAX) {
    variable = dap.variables.GetVariable(id_value);
  } else {
    variable = dap.variables.FindVariable(variablesReference, name);
  }

  if (variable.IsValid()) {
    lldb::SBError error;
    bool success = variable.SetValueFromCString(value.data(), error);
    if (success) {
      VariableDescription desc(variable, dap.enable_auto_variable_summaries);
      EmplaceSafeString(body, "result", desc.display_value);
      EmplaceSafeString(body, "type", desc.display_type_name);

      // We don't know the index of the variable in our dap.variables
      // so always insert a new one to get its variablesReference.
      // is_permanent is false because debug console does not support
      // setVariable request.
      int64_t new_var_ref =
          dap.variables.InsertVariable(variable, /*is_permanent=*/false);
      if (variable.MightHaveChildren())
        body.try_emplace("variablesReference", new_var_ref);
      else
        body.try_emplace("variablesReference", 0);
      if (lldb::addr_t addr = variable.GetLoadAddress();
          addr != LLDB_INVALID_ADDRESS)
        body.try_emplace("memoryReference", EncodeMemoryReference(addr));
      if (ValuePointsToCode(variable))
        body.try_emplace("valueLocationReference", new_var_ref);
    } else {
      EmplaceSafeString(body, "message", std::string(error.GetCString()));
    }
    response["success"] = llvm::json::Value(success);
  } else {
    response["success"] = llvm::json::Value(false);
  }

  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

} // namespace lldb_dap
