//===-- ScopesRequestHandler.cpp ------------------------------------------===//
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

// "ScopesRequest": {
//   "allOf": [ { "$ref": "#/definitions/Request" }, {
//     "type": "object",
//     "description": "Scopes request; value of command field is 'scopes'. The
//     request returns the variable scopes for a given stackframe ID.",
//     "properties": {
//       "command": {
//         "type": "string",
//         "enum": [ "scopes" ]
//       },
//       "arguments": {
//         "$ref": "#/definitions/ScopesArguments"
//       }
//     },
//     "required": [ "command", "arguments"  ]
//   }]
// },
// "ScopesArguments": {
//   "type": "object",
//   "description": "Arguments for 'scopes' request.",
//   "properties": {
//     "frameId": {
//       "type": "integer",
//       "description": "Retrieve the scopes for this stackframe."
//     }
//   },
//   "required": [ "frameId" ]
// },
// "ScopesResponse": {
//   "allOf": [ { "$ref": "#/definitions/Response" }, {
//     "type": "object",
//     "description": "Response to 'scopes' request.",
//     "properties": {
//       "body": {
//         "type": "object",
//         "properties": {
//           "scopes": {
//             "type": "array",
//             "items": {
//               "$ref": "#/definitions/Scope"
//             },
//             "description": "The scopes of the stackframe. If the array has
//             length zero, there are no scopes available."
//           }
//         },
//         "required": [ "scopes" ]
//       }
//     },
//     "required": [ "body" ]
//   }]
// }

/// Create a "Scope" JSON object as described in the debug adapter definition.
///
/// \param[in] name
///     The value to place into the "name" key
//
/// \param[in] variablesReference
///     The value to place into the "variablesReference" key
//
/// \param[in] namedVariables
///     The value to place into the "namedVariables" key
//
/// \param[in] expensive
///     The value to place into the "expensive" key
///
/// \return
///     A "Scope" JSON object with that follows the formal JSON
///     definition outlined by Microsoft.
protocol::Scope CreateScope2(const llvm::StringRef name,
                             int64_t variablesReference, int64_t namedVariables,
                             bool expensive) {
  protocol::Scope scope;

  scope.name = name;

  // TODO: Support "arguments" scope. At the moment lldb-dap includes the
  // arguments into the "locals" scope.
  // add presentation hint;
  if (variablesReference == VARREF_LOCALS)
    scope.presentationHint = protocol::Scope::ePresentationHintLocals;
  else if (variablesReference == VARREF_REGS)
    scope.presentationHint = protocol::Scope::ePresentationHintRegisters;

  scope.variablesReference = variablesReference;
  scope.namedVariables = namedVariables;
  scope.expensive = expensive;

  return scope;
}

static std::vector<protocol::Scope> CreateTopLevelScopes(DAP &dap) {
  std::vector<protocol::Scope> scopes;
  scopes.reserve(3);
  scopes.emplace_back(CreateScope2("Locals", VARREF_LOCALS,
                                   dap.variables.locals.GetSize(), false));
  scopes.emplace_back(CreateScope2("Globals", VARREF_GLOBALS,
                                   dap.variables.globals.GetSize(), false));
  scopes.emplace_back(CreateScope2("Registers", VARREF_REGS,
                                   dap.variables.registers.GetSize(), false));

  return scopes;
}

llvm::Expected<protocol::ScopesResponseBody>
ScopesRequestHandler::Run(const protocol::ScopesArguments &args) const {
  lldb::SBFrame frame = dap.GetLLDBFrame(args.frameId);

  // As the user selects different stack frames in the GUI, a "scopes" request
  // will be sent to the DAP. This is the only way we know that the user has
  // selected a frame in a thread. There are no other notifications that are
  // sent and VS code doesn't allow multiple frames to show variables
  // concurrently. If we select the thread and frame as the "scopes" requests
  // are sent, this allows users to type commands in the debugger console
  // with a backtick character to run lldb commands and these lldb commands
  // will now have the right context selected as they are run. If the user
  // types "`bt" into the debugger console and we had another thread selected
  // in the LLDB library, we would show the wrong thing to the user. If the
  // users switches threads with a lldb command like "`thread select 14", the
  // GUI will not update as there are no "event" notification packets that
  // allow us to change the currently selected thread or frame in the GUI that
  // I am aware of.
  if (frame.IsValid()) {
    frame.GetThread().GetProcess().SetSelectedThread(frame.GetThread());
    frame.GetThread().SetSelectedFrame(frame.GetFrameID());
  }
  dap.variables.locals = frame.GetVariables(/*arguments=*/true,
                                            /*locals=*/true,
                                            /*statics=*/false,
                                            /*in_scope_only=*/true);
  dap.variables.globals = frame.GetVariables(/*arguments=*/false,
                                             /*locals=*/false,
                                             /*statics=*/true,
                                             /*in_scope_only=*/true);
  dap.variables.registers = frame.GetRegisters();

  return protocol::ScopesResponseBody{CreateTopLevelScopes(dap)};
}

} // namespace lldb_dap
