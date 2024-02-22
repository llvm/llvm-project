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
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/StringRef.h"

namespace lldb_private {

enum class SwiftLog : Log::MaskType {
  Health = Log::ChannelFlag<0>,
  LLVM_MARK_AS_BITMASK_ENUM(Health)
};

struct LogChannelSwift {
  static void Initialize();
  static void Terminate();

  static Log *GetLogIfAll(SwiftLog mask) { return GetLog(mask); }
  static Log *GetLogIfAny(SwiftLog mask) { return GetLog(mask); }
};

template <> Log::Channel &LogChannelFor<SwiftLog>();

Log *GetSwiftHealthLog();
llvm::StringRef GetSwiftHealthLogData();
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_LOGCHANNELDWARF_H
