//===-- DecodedThread.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_DECODEDTHREAD_H
#define LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_DECODEDTHREAD_H

#include <utility>
#include <vector>

#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"

#include "lldb/Target/Trace.h"
#include "lldb/Utility/TraceIntelPTGDBRemotePackets.h"

#include "intel-pt.h"

namespace lldb_private {
namespace trace_intel_pt {

/// libipt status utils
/// \{
bool IsLibiptError(int libipt_status);

bool IsEndOfStream(int libipt_status);

bool IsTscUnavailable(int libipt_status);
/// \}

/// Class for representing a libipt decoding error.
class IntelPTError : public llvm::ErrorInfo<IntelPTError> {
public:
  static char ID;

  /// \param[in] libipt_error_code
  ///     Negative number returned by libipt when decoding the trace and
  ///     signaling errors.
  ///
  /// \param[in] address
  ///     Optional instruction address. When decoding an individual instruction,
  ///     its address might be available in the \a pt_insn object, and should be
  ///     passed to this constructor. Other errors don't have an associated
  ///     address.
  IntelPTError(int libipt_error_code,
               lldb::addr_t address = LLDB_INVALID_ADDRESS);

  std::error_code convertToErrorCode() const override {
    return llvm::errc::not_supported;
  }

  int GetLibiptErrorCode() const { return m_libipt_error_code; }

  void log(llvm::raw_ostream &OS) const override;

private:
  int m_libipt_error_code;
  lldb::addr_t m_address;
};

/// \class DecodedThread
/// Class holding the instructions and function call hierarchy obtained from
/// decoding a trace, as well as a position cursor used when reverse debugging
/// the trace.
///
/// Each decoded thread contains a cursor to the current position the user is
/// stopped at. See \a Trace::GetCursorPosition for more information.
class DecodedThread : public std::enable_shared_from_this<DecodedThread> {
public:
  /// \class TscRange
  /// Class that represents the trace range associated with a given TSC.
  /// It provides efficient iteration to the previous or next TSC range in the
  /// decoded trace.
  ///
  /// TSC timestamps are emitted by the decoder infrequently, which means
  /// that each TSC covers a range of instruction indices, which can be used to
  /// speed up TSC lookups.
  class TscRange {
  public:
    /// Check if this TSC range includes the given instruction index.
    bool InRange(size_t insn_index) const;

    /// Get the next range chronologically.
    llvm::Optional<TscRange> Next() const;

    /// Get the previous range chronologically.
    llvm::Optional<TscRange> Prev() const;

    /// Get the TSC value.
    size_t GetTsc() const;
    /// Get the smallest instruction index that has this TSC.
    size_t GetStartInstructionIndex() const;
    /// Get the largest instruction index that has this TSC.
    size_t GetEndInstructionIndex() const;

  private:
    friend class DecodedThread;

    TscRange(std::map<size_t, uint64_t>::const_iterator it,
             const DecodedThread &decoded_thread);

    /// The iterator pointing to the beginning of the range.
    std::map<size_t, uint64_t>::const_iterator m_it;
    /// The largest instruction index that has this TSC.
    size_t m_end_index;

    const DecodedThread *m_decoded_thread;
  };

  // Struct holding counts for libipts errors;
  struct LibiptErrorsStats {
    // libipt error -> count
    llvm::DenseMap<const char *, int> libipt_errors_counts;
    size_t total_count = 0;

    void RecordError(int libipt_error_code);
  };

  // Struct holding counts for events;
  struct EventsStats {
    /// A count for each individual event kind. We use an unordered map instead
    /// of a DenseMap because DenseMap can't understand enums.
    std::unordered_map<lldb::TraceEvent, size_t> events_counts;
    size_t total_count = 0;

    void RecordEvent(lldb::TraceEvent event);
  };

  DecodedThread(lldb::ThreadSP thread_sp);

  /// Utility constructor that initializes the trace with a provided error.
  DecodedThread(lldb::ThreadSP thread_sp, llvm::Error &&err);

  /// Get the total number of instruction, errors and events from the decoded
  /// trace.
  int64_t GetItemsCount() const;

  /// Construct the TSC range that covers the given instruction index.
  /// This operation is O(logn) and should be used sparingly.
  /// If the trace was collected with TSC support, all the instructions of
  /// the trace will have associated TSCs. This means that this method will
  /// only return \b llvm::None if there are no TSCs whatsoever in the trace.
  ///
  /// \param[in] insn_index
  ///   The instruction index in question.
  ///
  /// \param[in] hint_range
  ///   An optional range that might include the given index or might be a
  ///   neighbor of it. It might help speed it traversals of the trace with
  ///   short jumps.
  llvm::Optional<TscRange> CalculateTscRange(
      size_t insn_index,
      const llvm::Optional<DecodedThread::TscRange> &hint_range) const;

  /// \return
  ///   The error associated with a given trace item.
  const char *GetErrorByIndex(size_t item_index) const;

  /// \return
  ///   The trace item kind given an item index.
  lldb::TraceItemKind GetItemKindByIndex(size_t item_index) const;

  /// \return
  ///   The underlying event type for the given trace item index.
  lldb::TraceEvent GetEventByIndex(int item_index) const;

  /// Get the most recent CPU id before or at the given trace item index.
  ///
  /// \param[in] item_index
  ///   The trace item index to compare with.
  ///
  /// \return
  ///   The requested cpu id, or \a llvm::None if not available.
  llvm::Optional<lldb::cpu_id_t> GetCPUByIndex(uint64_t item_index) const;

  /// \return
  ///     The load address of the instruction at the given index.
  lldb::addr_t GetInstructionLoadAddress(size_t item_index) const;

  /// Get a new cursor for the decoded thread.
  lldb::TraceCursorUP CreateNewCursor();

  /// Return an object with statistics of the TSC decoding errors that happened.
  /// A TSC error is not a fatal error and doesn't create gaps in the trace.
  /// Instead we only keep track of them as statistics.
  ///
  /// \return
  ///   An object with the statistics of TSC decoding errors.
  const LibiptErrorsStats &GetTscErrorsStats() const;

  /// Return an object with statistics of the trace events that happened.
  ///
  /// \return
  ///   The stats object of all the events.
  const EventsStats &GetEventsStats() const;

  /// Record an error decoding a TSC timestamp.
  ///
  /// See \a GetTscErrors() for more documentation.
  ///
  /// \param[in] libipt_error_code
  ///   An error returned by the libipt library.
  void RecordTscError(int libipt_error_code);

  /// The approximate size in bytes used by this instance,
  /// including all the already decoded instructions.
  size_t CalculateApproximateMemoryUsage() const;

  lldb::ThreadSP GetThread();

  /// Notify this object that a new tsc has been seen.
  /// If this a new TSC, an event will be created.
  void NotifyTsc(uint64_t tsc);

  /// Notify this object that a CPU has been seen.
  /// If this a new CPU, an event will be created.
  void NotifyCPU(lldb::cpu_id_t cpu_id);

  /// Append a decoding error.
  void AppendError(const IntelPTError &error);

  /// Append a custom decoding.
  void AppendCustomError(llvm::StringRef error);

  /// Append an event.
  void AppendEvent(lldb::TraceEvent);

  /// Append an instruction.
  void AppendInstruction(const pt_insn &insn);

private:
  /// When adding new members to this class, make sure
  /// to update \a CalculateApproximateMemoryUsage() accordingly.
  lldb::ThreadSP m_thread_sp;

  /// We use a union to optimize the memory usage for the different kinds of
  /// trace items.
  union TraceItemStorage {
    /// The load addresses of this item if it's an instruction.
    uint64_t load_address;

    /// The event kind of this item if it's an event
    lldb::TraceEvent event;

    /// The string message of this item if it's an error
    const char *error;
  };

  /// Create a new trace item.
  ///
  /// \return
  ///   The index of the new item.
  DecodedThread::TraceItemStorage &CreateNewTraceItem(lldb::TraceItemKind kind);

  /// Most of the trace data is stored here.
  std::vector<TraceItemStorage> m_item_data;
  /// The TraceItemKind for each trace item encoded as uint8_t. We don't include
  /// it in TraceItemStorage to avoid padding.
  std::vector<uint8_t> m_item_kinds;

  /// This map contains the TSCs of the decoded instructions. It maps
  /// `instruction index -> TSC`, where `instruction index` is the first index
  /// at which the mapped TSC appears. We use this representation because TSCs
  /// are sporadic and we can think of them as ranges. If TSCs are present in
  /// the trace, all instructions will have an associated TSC, including the
  /// first one. Otherwise, this map will be empty.
  std::map<uint64_t, uint64_t> m_timestamps;
  /// This is the chronologically last TSC that has been added.
  llvm::Optional<uint64_t> m_last_tsc = llvm::None;

  // The cpu information is stored as a map. It maps `instruction index -> CPU`
  // A CPU is associated with the next instructions that follow until the next
  // cpu is seen.
  std::map<uint64_t, lldb::cpu_id_t> m_cpus;
  /// This is the chronologically last CPU ID.
  llvm::Optional<uint64_t> m_last_cpu = llvm::None;

  /// Statistics of all tracing events.
  EventsStats m_events_stats;
  /// Statistics of libipt errors when decoding TSCs.
  LibiptErrorsStats m_tsc_errors_stats;
  /// Total amount of time spent decoding.
  std::chrono::milliseconds m_total_decoding_time{0};
};

using DecodedThreadSP = std::shared_ptr<DecodedThread>;

} // namespace trace_intel_pt
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_DECODEDTHREAD_H
