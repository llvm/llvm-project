//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLDBServerMockAcceleratorPlugin.h"

using namespace lldb_private;
using namespace lldb_private::lldb_server;

LLDBServerMockAcceleratorPlugin::LLDBServerMockAcceleratorPlugin(
    GDBServer &gdb_server, MainLoop &main_loop)
    : LLDBServerAcceleratorPlugin(gdb_server, main_loop) {}

llvm::StringRef LLDBServerMockAcceleratorPlugin::GetPluginName() {
  return "mock";
}

std::optional<AcceleratorActions>
LLDBServerMockAcceleratorPlugin::GetInitializeActions() {
  AcceleratorActions actions(GetPluginName(), 1);
  actions.session_name = "Mock Accelerator Session";

  AcceleratorBreakpointInfo bp;
  bp.identifier = kBreakpointIDInitialize;
  bp.by_name = AcceleratorBreakpointByName{std::nullopt, "main"};
  actions.breakpoints.push_back(std::move(bp));

  AcceleratorConnectionInfo conn;
  conn.connect_url = "connect://localhost:0";
  conn.triple = "x86_64-unknown-linux-gnu";
  actions.connect_info = std::move(conn);

  return actions;
}

llvm::Expected<AcceleratorBreakpointHitResponse>
LLDBServerMockAcceleratorPlugin::BreakpointWasHit(
    AcceleratorBreakpointHitArgs &args) {
  AcceleratorBreakpointHitResponse response;
  if (args.breakpoint.identifier == kBreakpointIDInitialize) {
    response.disable_bp = true;
    response.auto_resume_native = false;

    AcceleratorActions actions(GetPluginName(), kBreakpointIDExit);
    AcceleratorBreakpointInfo bp;
    bp.identifier = kBreakpointIDExit;
    bp.by_name = AcceleratorBreakpointByName{std::nullopt, "exit"};
    actions.breakpoints.push_back(std::move(bp));
    response.actions = std::move(actions);
  }
  return response;
}
