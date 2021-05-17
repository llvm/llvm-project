//===-- ProgressEvent.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProgressEvent.h"

#include "JSONUtils.h"

using namespace lldb_vscode;
using namespace llvm;

ProgressEvent::ProgressEvent(uint64_t progress_id, const char *message,
                             uint64_t completed, uint64_t total)
    : m_progress_id(progress_id), m_message(message) {
  if (completed == total)
    m_event_type = progressEnd;
  else if (completed == 0)
    m_event_type = progressStart;
  else if (completed < total)
    m_event_type = progressUpdate;
  else
    m_event_type = progressInvalid;

  if (0 < total && total < UINT64_MAX)
    m_percentage = (uint32_t)(((float)completed / (float)total) * 100.0);
}

bool ProgressEvent::operator==(const ProgressEvent &other) const {
  return m_progress_id == other.m_progress_id &&
         m_event_type == other.m_event_type &&
         m_percentage == other.m_percentage;
}

const char *ProgressEvent::GetEventName() const {
  if (m_event_type == progressStart)
    return "progressStart";
  else if (m_event_type == progressEnd)
    return "progressEnd";
  else if (m_event_type == progressUpdate)
    return "progressUpdate";
  else
    return "progressInvalid";
}

bool ProgressEvent::IsValid() const { return m_event_type != progressInvalid; }

uint64_t ProgressEvent::GetID() const { return m_progress_id; }

json::Value ProgressEvent::ToJSON() const {
  llvm::json::Object event(CreateEventObject(GetEventName()));
  llvm::json::Object body;

  std::string progress_id_str;
  llvm::raw_string_ostream progress_id_strm(progress_id_str);
  progress_id_strm << m_progress_id;
  progress_id_strm.flush();
  body.try_emplace("progressId", progress_id_str);

  if (m_event_type == progressStart) {
    EmplaceSafeString(body, "title", m_message);
    body.try_emplace("cancellable", false);
  }

  auto now = std::chrono::duration<double>(
      std::chrono::system_clock::now().time_since_epoch());
  std::string timestamp(llvm::formatv("{0:f9}", now.count()));
  EmplaceSafeString(body, "timestamp", timestamp);

  if (m_percentage)
    body.try_emplace("percentage", *m_percentage);

  event.try_emplace("body", std::move(body));
  return json::Value(std::move(event));
}

ProgressEventFilterQueue::ProgressEventFilterQueue(
    std::function<void(ProgressEvent)> callback)
    : m_callback(callback) {}

void ProgressEventFilterQueue::Push(const ProgressEvent &event) {
  if (!event.IsValid())
    return;

  auto it = m_last_events.find(event.GetID());
  if (it == m_last_events.end() || !(it->second == event)) {
    m_last_events[event.GetID()] = event;
    m_callback(event);
  }
}
