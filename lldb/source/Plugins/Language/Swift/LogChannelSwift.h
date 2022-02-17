//===-- LogChannelSwift.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_LogChannelSwift_h_
#define liblldb_LogChannelSwift_h_

#include "lldb/Utility/Log.h"
#include "llvm/ADT/StringRef.h"

namespace lldb_private {

#define LIBLLDB_SWIFT_LOG_HEALTH (1u << 1)
#define SWIFT_LOG_DEFAULT (LIBLLDB_SWIFT_LOG_HEALTH)

class LogChannelSwift {
  static Log::Channel g_channel;

public:
  static void Initialize();
  static void Terminate();

  static Log *GetLogIfAll(uint32_t mask) { return g_channel.GetLogIfAll(mask); }
  static Log *GetLogIfAny(uint32_t mask) { return g_channel.GetLogIfAny(mask); }
};

Log *GetSwiftHealthLog();
llvm::StringRef GetSwiftHealthLogData();
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_LOGCHANNELDWARF_H
