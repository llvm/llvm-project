//===-- DAPLog.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DAPLog.h"

using namespace lldb_private;
using namespace lldb_dap;

static constexpr Log::Category g_categories[] = {
    {{"transport"}, {"log DAP transport"}, DAPLog::Transport},
    {{"protocol"}, {"log protocol handling"}, DAPLog::Protocol},
    {{"connection"}, {"log connection handling"}, DAPLog::Connection},
};

static Log::Channel g_log_channel(g_categories, DAPLog::Transport |
                                                    DAPLog::Protocol |
                                                    DAPLog::Connection);

template <> Log::Channel &lldb_private::LogChannelFor<DAPLog>() {
  return g_log_channel;
}

void lldb_dap::InitializeDAPChannel() {
  Log::Register("lldb-dap", g_log_channel);
}
