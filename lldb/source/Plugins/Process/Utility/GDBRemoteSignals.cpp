//===-- GDBRemoteSignals.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GDBRemoteSignals.h"

using namespace lldb_private;

GDBRemoteSignals::GDBRemoteSignals() : UnixSignals() { Reset(); }

GDBRemoteSignals::GDBRemoteSignals(const lldb::UnixSignalsSP &rhs)
    : UnixSignals(*rhs) {}

void GDBRemoteSignals::Reset() { m_signals.clear(); }
