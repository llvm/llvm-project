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

  void log(llvm::raw_ostream &OS) const override;

private:
  int m_libipt_error_code;
  lldb::addr_t m_address;
};

/// Helper struct for building an instruction or error from the decoder.
/// It holds associated events and timing information.
struct DecodedInstruction {
  DecodedInstruction() {
    pt_insn.ip = LLDB_INVALID_ADDRESS;
    libipt_error = pte_ok;
  }

  DecodedInstruction(int libipt_error_code) : DecodedInstruction() {
    libipt_error = libipt_error_code;
  }

  /// \return \b true if and only if this struct holds a libipt error.
  explicit operator bool() const;

  int libipt_error;
  lldb::TraceEvents events = (lldb::TraceEvents)0;
  llvm::Optional<uint64_t> tsc = llvm::None;
  pt_insn pt_insn;
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
    std::unordered_map<lldb::TraceEvents, size_t> events_counts;
    size_t total_count = 0;
    size_t total_instructions_with_events = 0;

    void RecordEventsForInstruction(lldb::TraceEvents events);
  };

  DecodedThread(lldb::ThreadSP thread_sp);

  /// Utility constructor that initializes the trace with a provided error.
  DecodedThread(lldb::ThreadSP thread_sp, llvm::Error &&err);

  /// Append an instruction or a libipt error.
  void Append(const DecodedInstruction &insn);

  /// Append an error signaling that decoding completely failed.
  void SetAsFailed(llvm::Error &&error);

  /// Get a bitmask with the events that happened chronologically right before
  /// the instruction pointed by the given instruction index, but after the
  /// previous instruction.
  lldb::TraceEvents GetEvents(int insn_index);

  /// Get the total number of instruction pointers from the decoded trace.
  /// This will include instructions that indicate errors (or gaps) in the
  /// trace. For an instruction error, you can access its underlying error
  /// message with the \a GetErrorByInstructionIndex() method.
  size_t GetInstructionsCount() const;

  /// \return
  ///     The load address of the instruction at the given index, or \a
  ///     LLDB_INVALID_ADDRESS if it is an error.
  lldb::addr_t GetInstructionLoadAddress(size_t insn_index) const;

  /// Get the \a lldb::TraceInstructionControlFlowType categories of the
  /// instruction.
  ///
  /// \return
  ///     The control flow categories, or \b 0 if the instruction is an error.
  lldb::TraceInstructionControlFlowType
  GetInstructionControlFlowType(size_t insn_index) const;

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

  /// Check if an instruction given by its index is an error.
  bool IsInstructionAnError(size_t insn_idx) const;

  /// Get the error associated with a given instruction index.
  ///
  /// \return
  ///   The error message of \b nullptr if the given index
  ///   points to a valid instruction.
  const char *GetErrorByInstructionIndex(size_t ins_idx);

  /// Get a new cursor for the decoded thread.
  lldb::TraceCursorUP GetCursor();

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

  /// Append a decoding error given an llvm::Error.
  void AppendError(llvm::Error &&error);

private:
  /// Notify this class that the last added instruction or error has
  /// an associated TSC.
  void RecordTscForLastInstruction(uint64_t tsc);

  /// When adding new members to this class, make sure
  /// to update \a CalculateApproximateMemoryUsage() accordingly.
  lldb::ThreadSP m_thread_sp;
  /// The low level storage of all instruction addresses. Each instruction has
  /// an index in this vector and it will be used in other parts of the code.
  std::vector<lldb::addr_t> m_instruction_ips;
  /// The size in bytes of each instruction.
  std::vector<uint8_t> m_instruction_sizes;
  /// The libipt instruction class for each instruction.
  std::vector<pt_insn_class> m_instruction_classes;

  /// This map contains the TSCs of the decoded instructions. It maps
  /// `instruction index -> TSC`, where `instruction index` is the first index
  /// at which the mapped TSC appears. We use this representation because TSCs
  /// are sporadic and we can think of them as ranges. If TSCs are present in
  /// the trace, all instructions will have an associated TSC, including the
  /// first one. Otherwise, this map will be empty.
  std::map<uint64_t, uint64_t> m_instruction_timestamps;
  /// This is the chronologically last TSC that has been added.
  llvm::Optional<uint64_t> m_last_tsc = llvm::None;
  // This variables stores the messages of all the error instructions in the
  // trace. It maps `instruction index -> error message`.
  llvm::DenseMap<uint64_t, std::string> m_errors;
  /// This variable stores the bitmask of events that happened right before
  /// the instruction given as a key. It maps `instruction index -> events`.
  llvm::DenseMap<uint64_t, lldb::TraceEvents> m_events;

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
