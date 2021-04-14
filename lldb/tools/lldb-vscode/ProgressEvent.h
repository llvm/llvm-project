//===-- ProgressEvent.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VSCodeForward.h"

#include "llvm/Support/JSON.h"

namespace lldb_vscode {

enum ProgressEventType {
  progressInvalid,
  progressStart,
  progressUpdate,
  progressEnd
};

class ProgressEvent {
public:
  ProgressEvent() {}

  ProgressEvent(uint64_t progress_id, const char *message, uint64_t completed,
                uint64_t total);

  llvm::json::Value ToJSON() const;

  /// This operator returns \b true if two event messages
  /// would result in the same event for the IDE, e.g.
  /// same rounded percentage.
  bool operator==(const ProgressEvent &other) const;

  const char *GetEventName() const;

  bool IsValid() const;

  uint64_t GetID() const;

private:
  uint64_t m_progress_id;
  const char *m_message;
  ProgressEventType m_event_type;
  llvm::Optional<uint32_t> m_percentage;
};

/// Class that filters out progress event messages that shouldn't be reported
/// to the IDE, either because they are invalid or because they are too chatty.
class ProgressEventFilterQueue {
public:
  ProgressEventFilterQueue(std::function<void(ProgressEvent)> callback);

  void Push(const ProgressEvent &event);

private:
  std::function<void(ProgressEvent)> m_callback;
  std::map<uint64_t, ProgressEvent> m_last_events;
};

} // namespace lldb_vscode
