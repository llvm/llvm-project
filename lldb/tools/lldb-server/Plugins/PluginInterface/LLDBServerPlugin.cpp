//===-- LLDBServerPlugin.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLDBServerPlugin.h"
#include <vector>

using namespace lldb_private;
using namespace lldb_server;

static std::vector<LLDBServerPlugin::CreateCallback> g_create_callbacks;

void LLDBServerPlugin::RegisterCreatePlugin(CreateCallback callback) {
  g_create_callbacks.push_back(callback);
}

size_t LLDBServerPlugin::GetNumCreateCallbacks() {
  return g_create_callbacks.size();
}

LLDBServerPlugin::CreateCallback
LLDBServerPlugin::GetCreateCallbackAtIndex(size_t i) {
  if (i < g_create_callbacks.size())
    return g_create_callbacks[i];
  return nullptr;
}

LLDBServerPlugin::~LLDBServerPlugin() {}
