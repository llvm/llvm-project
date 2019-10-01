//===-- LogMessageOsLog.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LogMessageOsLog_h
#define LogMessageOsLog_h

#include "DarwinLogInterfaces.h"

#include "ActivityStreamSPI.h"
#include "LogMessage.h"

using ActivityStreamEntry = struct os_activity_stream_entry_s;

/// Provides a unified wrapper around os_log()-style log messages.
///
/// The lifetime of this class is intended to be very short.  The caller
/// must ensure that the passed in ActivityStore and ActivityStreamEntry
/// outlive this LogMessageOsLog entry.

class LogMessageOsLog : public LogMessage {
public:
  static void SetFormatterFunction(os_log_copy_formatted_message_t format_func);

  LogMessageOsLog(const ActivityStore &activity_store,
                  ActivityStreamEntry &entry);

  // API methods

  bool HasActivity() const override;

  const char *GetActivity() const override;

  std::string GetActivityChain() const override;

  bool HasCategory() const override;

  const char *GetCategory() const override;

  bool HasSubsystem() const override;

  const char *GetSubsystem() const override;

  const char *GetMessage() const override;

private:
  const ActivityStore &m_activity_store;
  ActivityStreamEntry &m_entry;
  mutable std::string m_message;
};

#endif /* LogMessageOsLog_h */
