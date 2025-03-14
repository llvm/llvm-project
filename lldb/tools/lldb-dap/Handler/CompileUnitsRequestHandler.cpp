//===-- CompileUnitsRequestHandler.cpp ------------------------------------===//
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

// "compileUnitsRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Compile Unit request; value of command field is
//                     'compileUnits'.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "compileUnits" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/compileUnitRequestArguments"
//       }
//     },
//     "required": [ "command", "arguments" ]
//   }]
// },
// "compileUnitsRequestArguments": {
//   "type": "object",
//   "description": "Arguments for 'compileUnits' request.",
//   "properties": {
//     "moduleId": {
//       "type": "string",
//       "description": "The ID of the module."
//     }
//   },
//   "required": [ "moduleId" ]
// },
// "compileUnitsResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'compileUnits' request.",
//     "properties": {
//       "body": {
//         "description": "Response to 'compileUnits' request. Array of
//                         paths of compile units."
//       }
//     }
//   }]
// }
void CompileUnitsRequestHandler::operator()(
    const llvm::json::Object &request) const {
  llvm::json::Object response;
  FillResponse(request, response);
  llvm::json::Object body;
  llvm::json::Array units;
  const auto *arguments = request.getObject("arguments");
  std::string module_id = std::string(GetString(arguments, "moduleId"));
  int num_modules = dap.target.GetNumModules();
  for (int i = 0; i < num_modules; i++) {
    auto curr_module = dap.target.GetModuleAtIndex(i);
    if (module_id == curr_module.GetUUIDString()) {
      int num_units = curr_module.GetNumCompileUnits();
      for (int j = 0; j < num_units; j++) {
        auto curr_unit = curr_module.GetCompileUnitAtIndex(j);
        units.emplace_back(CreateCompileUnit(curr_unit));
      }
      body.try_emplace("compileUnits", std::move(units));
      break;
    }
  }
  response.try_emplace("body", std::move(body));
  dap.SendJSON(llvm::json::Value(std::move(response)));
}

} // namespace lldb_dap
