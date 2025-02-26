#include "DAPLog.h"

using namespace lldb_private;
using namespace lldb_dap;

static constexpr Log::Category g_categories[] = {
    {{"transport"}, {"log DAP transport"}, DAPLog::Transport},
    {{"protocol"}, {"log protocol handling"}, DAPLog::Protocol},
};

static Log::Channel g_log_channel(g_categories,
                                  DAPLog::Transport | DAPLog::Protocol);

template <> Log::Channel &lldb_private::LogChannelFor<DAPLog>() {
  return g_log_channel;
}

void lldb_dap::InitializeDAPChannel() {
  Log::Register("lldb-dap", g_log_channel);
}
