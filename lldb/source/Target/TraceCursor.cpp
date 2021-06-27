//===-- TraceCursor.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/TraceCursor.h"

#include "lldb/Target/Trace.h"

using namespace lldb_private;

bool TraceCursor::IsStale() { return m_stop_id != m_trace_sp->GetStopID(); }
