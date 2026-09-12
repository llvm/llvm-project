//===-- HostInfoEmscripten.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/emscripten/HostInfoEmscripten.h"

using namespace lldb_private;

void HostInfoEmscripten::Initialize() { HostInfoPosix::Initialize(); }

void HostInfoEmscripten::Terminate() { HostInfoBase::Terminate(); }

FileSpec HostInfoEmscripten::GetProgramFileSpec() { return {}; }
