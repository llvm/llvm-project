//===-- LLDBServerPlugin.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLDBServerPlugin.h"
#include "GDBRemoteCommunicationServerLLGS.h"

using namespace lldb_private;
using namespace lldb_server;


LLDBServerPlugin::LLDBServerPlugin(GDBServer &native_process) :
  m_native_process(native_process) {}

LLDBServerPlugin::~LLDBServerPlugin() {}

const GPUPluginInfo &LLDBServerPlugin::GetPluginInfo() {
  if (m_info.name.empty())
    InitializePluginInfo();
  return m_info;
}
