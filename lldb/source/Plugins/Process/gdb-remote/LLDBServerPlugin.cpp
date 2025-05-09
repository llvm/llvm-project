//===-- LLDBServerPlugin.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLDBServerPlugin.h"
#include "GDBRemoteCommunicationServerLLGS.h"
#include <chrono>
#include <thread>

using namespace lldb_private;
using namespace lldb_server;


LLDBServerPlugin::LLDBServerPlugin(GDBServer &native_process) :
  m_native_process(native_process) {}

LLDBServerPlugin::~LLDBServerPlugin() {}


lldb::StateType 
LLDBServerPlugin::HaltNativeProcessIfNeeded(bool &was_halted, 
                                            uint32_t timeout_sec) {
  using namespace std::chrono;
  NativeProcessProtocol *process = m_native_process.GetCurrentProcess();
  if (process->IsRunning()) {
    was_halted = true;
    process->Halt();
 
    auto end_time = steady_clock::now() + seconds(timeout_sec);
    while (std::chrono::steady_clock::now() < end_time) {
      std::this_thread::sleep_for(milliseconds(250));
      if (process->IsStopped())
        break;
    }
  }
  return process->GetState();
}
