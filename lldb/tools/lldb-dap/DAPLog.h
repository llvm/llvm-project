//===-- DAPLog.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_LLDBLOG_H
#define LLDB_UTILITY_LLDBLOG_H

#include "lldb/Utility/Log.h"
#include "llvm/ADT/BitmaskEnum.h"

namespace lldb_dap {

enum class DAPLog : lldb_private::Log::MaskType {
  Transport = lldb_private::Log::ChannelFlag<0>,
  Protocol = lldb_private::Log::ChannelFlag<1>,
  Connection = lldb_private::Log::ChannelFlag<2>,
  LLVM_MARK_AS_BITMASK_ENUM(Connection),
};

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

void InitializeDAPChannel();

} // end namespace lldb_dap

namespace lldb_private {
template <> lldb_private::Log::Channel &LogChannelFor<lldb_dap::DAPLog>();
} // namespace lldb_private

#endif
