//===-- LLDBServerNativeProcess.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLDBServerNativeProcess.h"

using namespace lldb_private;
using namespace lldb_server;

static LLDBServerNativeProcess *g_native_process = nullptr;

LLDBServerNativeProcess *LLDBServerNativeProcess::GetNativeProcess() {
  return g_native_process;
}

void LLDBServerNativeProcess::SetNativeProcess(
    LLDBServerNativeProcess *process) {
  g_native_process = process;
}
