//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLDBServerAcceleratorPlugin.h"

using namespace lldb_private::lldb_server;

LLDBServerAcceleratorPlugin::LLDBServerAcceleratorPlugin(GDBServer &gdb_server,
                                                         MainLoop &main_loop)
    : m_gdb_server(gdb_server), m_main_loop(main_loop) {}

LLDBServerAcceleratorPlugin::~LLDBServerAcceleratorPlugin() = default;
