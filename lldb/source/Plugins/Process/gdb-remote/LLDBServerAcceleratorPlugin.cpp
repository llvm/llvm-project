//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLDBServerAcceleratorPlugin.h"

#include "GDBRemoteCommunicationServerLLGS.h"

using namespace lldb_private::lldb_server;

LLDBServerAcceleratorPlugin::LLDBServerAcceleratorPlugin(
    GDBServer &native_gdb_server, MainLoop &native_main_loop)
    : m_native_gdb_server(native_gdb_server),
      m_native_main_loop(native_main_loop) {}

LLDBServerAcceleratorPlugin::~LLDBServerAcceleratorPlugin() = default;
