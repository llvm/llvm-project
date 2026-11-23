//===-- ProgressEvent.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_PROGRESS_EVENT_H
#define LLDB_TOOLS_LLDB_DAP_PROGRESS_EVENT_H

#include "Protocol/ProtocolBase.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FunctionExtras.h"

#include <chrono>
#include <cstdint>
#include <string>

namespace lldb_dap {

/// Translates lldb `Progress` events into DAP progress events.
///
/// Filters out progress events that shouldn't be reported either because they
/// spam the client's UI or they don't last long enough.
class ProgressEventReporter {
public:
  /// The Callback used to send DAP progress events.
  using SendEventFn = llvm::unique_function<void(protocol::Event)>;

  using TimePoint = std::chrono::steady_clock::time_point;

  explicit ProgressEventReporter(SendEventFn send);

  ProgressEventReporter(const ProgressEventReporter &) = delete;
  ProgressEventReporter(ProgressEventReporter &&) = delete;
  ProgressEventReporter &operator=(const ProgressEventReporter &) = delete;
  ProgressEventReporter &operator=(ProgressEventReporter &&) = delete;
  ~ProgressEventReporter() = default;

  // The minimum duration of an event for it to be reported
  static constexpr auto k_start_delay = std::chrono::milliseconds(1000);
  // The minimum time interval between update events for reporting. If multiple
  // updates fall within the same time interval, only the latest is reported.
  static constexpr auto k_update_interval = std::chrono::milliseconds(250);

  /// Reports a new lldb progress event and determines if we need to send
  /// it to the client. `now` is the wall time observed by the caller.
  void Report(uint64_t progress_id, std::optional<std::string> title,
              std::string details, uint64_t completed, uint64_t total,
              TimePoint now);

  /// Flush any pending events whose deadlines have passed.
  ///
  /// \param now
  ///   The current wall timepoint of the caller.
  void Drain(TimePoint now) {
    for (auto &[progress_id, pending] : m_pending)
      Flush(progress_id, pending, /*has_finished=*/false, now);
  }

  /// Whether the reporter is currently tracking any progress. Used to decide
  /// if we need to wait indefinitely for the next event.
  bool HasPending() const { return !m_pending.empty(); }

private:
  /// The action the reporter should take for a given state and time.
  enum class Action {
    /// Do nothing.
    None,
    /// Send `progressStart`
    SendStart,
    /// Send `progressStart` and `progressEnd`.
    SendStartAndEnd,
    /// Send `progressEnd`.
    SendEnd,
    /// Send `progressUpdate`.
    SendUpdate,
  };

  /// The current state of a pending progress.
  struct State {
    TimePoint start_time;
    TimePoint last_send_time;
    bool start_sent = false;
    bool has_pending_update = false;
  };

  struct PendingProgress {
    /// Fields that end up in DAP progressEvent bodies.
    std::string title;
    std::string latest_details;
    uint64_t latest_completed = 0;
    uint64_t latest_total = 0;

    State state;
  };

  /// Determines the if the reporter needs to do nothing or what
  /// progressEvent(s) it needs to send.
  static Action GetAction(const State &state, TimePoint now, bool has_finished);

  /// Send progressEvent for the progress id if it is needed.
  /// \param progress_id
  ///   The id for the progressEvent to send.
  /// \param pending
  ///   The data and state of the pending progress.
  /// \param has_finished.
  ///   If the progress for the id has ended.
  /// \param now.
  ///   Is the wall time observed by the caller to determine if we need to send
  ///   a new progress event see `GetAction` for the heuristic.
  void Flush(uint64_t progress_id, PendingProgress &pending, bool has_finished,
             TimePoint now);

  SendEventFn m_send;
  llvm::DenseMap<uint64_t, PendingProgress> m_pending;
};

} // namespace lldb_dap

#endif // LLDB_TOOLS_LLDB_DAP_PROGRESS_EVENT_H
