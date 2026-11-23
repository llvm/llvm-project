//===-- ProgressEvent.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProgressEvent.h"
#include "Protocol/ProtocolEvents.h"

#include <optional>
#include <string>
#include <utility>

using namespace lldb_dap;
using namespace lldb_dap::protocol;

ProgressEventReporter::ProgressEventReporter(SendEventFn send)
    : m_send(std::move(send)) {}

void ProgressEventReporter::Report(uint64_t progress_id,
                                   std::optional<std::string> title,
                                   std::string details, uint64_t completed,
                                   uint64_t total, TimePoint now) {

  auto [it, is_new_progress] = m_pending.try_emplace(progress_id);
  PendingProgress &pending = it->second;

  if (is_new_progress) {
    // Title only updates when it is a new progress.
    if (!title.has_value()) {
      m_pending.erase(it);
      return;
    }
    pending.title = std::move(*title);
    pending.state.start_time = now;
  }

  pending.latest_details = std::move(details);
  pending.latest_completed = completed;
  pending.latest_total = total;
  pending.state.has_pending_update = true;

  const bool progress_ended = completed == total;
  Flush(progress_id, pending, progress_ended, now);
  if (progress_ended)
    m_pending.erase(it);
}

void ProgressEventReporter::Flush(uint64_t progress_id,
                                  PendingProgress &pending, bool has_finished,
                                  TimePoint now) {

  auto percentage = [&]() -> std::optional<uint32_t> {
    const uint64_t completed = pending.latest_completed;
    const uint64_t total = pending.latest_total;

    if (total == UINT64_MAX || total == 0 || completed > total)
      return std::nullopt;
    return static_cast<uint32_t>(100.0 * completed / total);
  };
  auto send_start = [&] {
    ProgressStartEventBody body;
    body.progressId = std::to_string(progress_id);
    body.title = pending.title;
    body.message = pending.latest_details;
    body.percentage = percentage();
    m_send(Event{"progressStart", toJSON(body)});
  };
  auto send_update = [&] {
    ProgressUpdateEventBody body;
    body.progressId = std::to_string(progress_id);
    body.message = pending.latest_details;
    body.percentage = percentage();
    m_send(Event{"progressUpdate", toJSON(body)});
  };
  auto send_end = [&] {
    ProgressEndEventBody body;
    body.progressId = std::to_string(progress_id);
    body.message = pending.latest_details;
    m_send(Event{"progressEnd", toJSON(body)});
  };

  State &state = pending.state;
  switch (GetAction(state, now, has_finished)) {
  case Action::SendStart:
    send_start();
    state.start_sent = true;
    state.has_pending_update = false;
    state.last_send_time = now;
    break;
  case Action::SendStartAndEnd:
    send_start();
    send_end();
    break;
  case Action::SendEnd:
    send_end();
    break;
  case Action::SendUpdate:
    send_update();
    state.has_pending_update = false;
    state.last_send_time = now;
    break;
  case Action::None:
    break;
  }
}

ProgressEventReporter::Action
ProgressEventReporter::GetAction(const State &state, TimePoint now,
                                 bool has_finished) {
  if (!state.start_sent) {
    if (state.start_time + k_start_delay > now)
      return Action::None;

    return has_finished ? Action::SendStartAndEnd : Action::SendStart;
  }

  if (has_finished)
    return Action::SendEnd;

  if (!state.has_pending_update)
    return Action::None;

  if (state.last_send_time + k_update_interval > now)
    return Action::None; // throttled
  return Action::SendUpdate;
}
