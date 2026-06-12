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

  // Set a breakpoint by function name (no shared library scope) on the
  // dedicated "mock_gpu_accelerator_initialize" hook and ask for the load
  // address of "mock_gpu_accelerator_compute" to be delivered when it is hit.
  // Using a dedicated, uniquely named function (rather than "main") keeps this
  // mock from affecting other inferiors that lldb-server launches when the
  // plugin is compiled in.
  AcceleratorBreakpointInfo bp;
  bp.identifier = kBreakpointIDInitialize;
  bp.by_name = AcceleratorBreakpointByName{std::nullopt,
                                           "mock_gpu_accelerator_initialize"};
  bp.symbol_names.push_back("mock_gpu_accelerator_compute");
  actions.breakpoints.push_back(std::move(bp));

  return actions;
}

llvm::Expected<AcceleratorBreakpointHitResponse>
LLDBServerMockAcceleratorPlugin::BreakpointWasHit(
    AcceleratorBreakpointHitArgs &args) {
  AcceleratorBreakpointHitResponse response;

  switch (args.breakpoint.identifier) {
  case kBreakpointIDInitialize: {
    // The initialize breakpoint was hit. Disable it, stop the native process,
    // and request two more breakpoints to exercise the remaining breakpoint
    // types.
    response.disable_bp = true;
    response.auto_resume_native = false;

    AcceleratorActions actions(GetPluginName(), 2);

    // Breakpoint by function name scoped to a shared library. Tests build to
    // "a.out", so use that as the shared library name.
    AcceleratorBreakpointInfo by_name_shlib;
    by_name_shlib.identifier = kBreakpointIDByNameShlib;
    by_name_shlib.by_name =
        AcceleratorBreakpointByName{"a.out", "mock_gpu_accelerator_finish"};
    actions.breakpoints.push_back(std::move(by_name_shlib));

    // Breakpoint by address, using the "mock_gpu_accelerator_compute" symbol
    // value that was delivered with this breakpoint hit.
    if (std::optional<uint64_t> compute_addr =
            args.GetSymbolValue("mock_gpu_accelerator_compute")) {
      AcceleratorBreakpointInfo by_address;
      by_address.identifier = kBreakpointIDByAddress;
      by_address.by_address = AcceleratorBreakpointByAddress{*compute_addr};
      actions.breakpoints.push_back(std::move(by_address));
    }

    response.actions = std::move(actions);
    break;
  }
  case kBreakpointIDByAddress:
  case kBreakpointIDByNameShlib:
    // Disable and stop the native process so the hit is observable.
    response.disable_bp = true;
    response.auto_resume_native = false;
    break;
  }

  return response;
}
