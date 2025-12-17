//===-- ScopesRequestHandler.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAP.h"
#include "RequestHandler.h"

using namespace lldb_dap::protocol;
namespace lldb_dap {

/// Creates a `protocol::Scope` struct.
///
///
/// \param[in] name
///     The value to place into the "name" key
///
/// \param[in] variablesReference
///     The value to place into the "variablesReference" key
///
/// \param[in] namedVariables
///     The value to place into the "namedVariables" key
///
/// \param[in] expensive
///     The value to place into the "expensive" key
///
/// \return
///     A `protocol::Scope`
static Scope CreateScope(const llvm::StringRef name, int64_t variablesReference,
                         int64_t namedVariables, bool expensive) {
  Scope scope;
  scope.name = name;

  // TODO: Support "arguments" and "return value" scope.
  // At the moment lldb-dap includes the arguments and return_value  into the
  // "locals" scope.
  // vscode only expands the first non-expensive scope, this causes friction
  // if we add the arguments above the local scope as the locals scope will not
  // be expanded if we enter a function with arguments. It becomes more
  // annoying when the scope has arguments, return_value and locals.
  if (variablesReference == VARREF_LOCALS)
    scope.presentationHint = Scope::eScopePresentationHintLocals;
  else if (variablesReference == VARREF_REGS)
    scope.presentationHint = Scope::eScopePresentationHintRegisters;

  scope.variablesReference = variablesReference;
  scope.namedVariables = namedVariables;
  scope.expensive = expensive;

  return scope;
}

llvm::Expected<ScopesResponseBody>
ScopesRequestHandler::Run(const ScopesArguments &args) const {
  lldb::SBFrame frame = dap.GetLLDBFrame(args.frameId);

  // As the user selects different stack frames in the GUI, a "scopes" request
  // will be sent to the DAP. This is the only way we know that the user has
  // selected a frame in a thread. There are no other notifications that are
  // sent and VS code doesn't allow multiple frames to show variables
  // concurrently. If we select the thread and frame as the "scopes" requests
  // are sent, this allows users to type commands in the debugger console
  // with a backtick character to run lldb commands and these lldb commands
  // will now have the right context selected as they are run. If the user
  // types "`bt" into the debugger console, and we had another thread selected
  // in the LLDB library, we would show the wrong thing to the user. If the
  // users switch threads with a lldb command like "`thread select 14", the
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

  std::vector scopes = {CreateScope("Locals", VARREF_LOCALS,
                                    dap.variables.locals.GetSize(), false),
                        CreateScope("Globals", VARREF_GLOBALS,
                                    dap.variables.globals.GetSize(), false),
                        CreateScope("Registers", VARREF_REGS,
                                    dap.variables.registers.GetSize(), false)};

  return ScopesResponseBody{std::move(scopes)};
}

} // namespace lldb_dap
