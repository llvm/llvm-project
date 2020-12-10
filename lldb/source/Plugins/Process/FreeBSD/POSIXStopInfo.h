//===-- POSIXStopInfo.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_POSIXStopInfo_H_
#define liblldb_POSIXStopInfo_H_

#include "FreeBSDThread.h"
#include "Plugins/Process/POSIX/CrashReason.h"
#include "lldb/Target/StopInfo.h"
#include <string>

//===----------------------------------------------------------------------===//
/// \class POSIXStopInfo
/// Simple base class for all POSIX-specific StopInfo objects.
///
class POSIXStopInfo : public lldb_private::StopInfo {
public:
  POSIXStopInfo(lldb_private::Thread &thread, uint32_t status)
      : StopInfo(thread, status) {}
};

//===----------------------------------------------------------------------===//
/// \class POSIXLimboStopInfo
/// Represents the stop state of a process ready to exit.
///
class POSIXLimboStopInfo : public POSIXStopInfo {
public:
  POSIXLimboStopInfo(FreeBSDThread &thread) : POSIXStopInfo(thread, 0) {}

  ~POSIXLimboStopInfo();

  lldb::StopReason GetStopReason() const override;

  const char *GetDescription() override;

  bool ShouldStop(lldb_private::Event *event_ptr) override;

  bool ShouldNotify(lldb_private::Event *event_ptr) override;
};

//===----------------------------------------------------------------------===//
/// \class POSIXNewThreadStopInfo
/// Represents the stop state of process when a new thread is spawned.
///

class POSIXNewThreadStopInfo : public POSIXStopInfo {
public:
  POSIXNewThreadStopInfo(FreeBSDThread &thread) : POSIXStopInfo(thread, 0) {}

  ~POSIXNewThreadStopInfo();

  lldb::StopReason GetStopReason() const override;

  const char *GetDescription() override;

  bool ShouldStop(lldb_private::Event *event_ptr) override;

  bool ShouldNotify(lldb_private::Event *event_ptr) override;
};

#endif
