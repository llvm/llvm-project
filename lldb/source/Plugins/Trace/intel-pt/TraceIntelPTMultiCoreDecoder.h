//===-- TraceIntelPTMultiCoreDecoder.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPTMULTICOREDECODER_H
#define LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPTMULTICOREDECODER_H

#include "ThreadDecoder.h"

namespace lldb_private {
namespace trace_intel_pt {

/// This class indicates the time interval in which a thread was running
/// continuously on a cpu core.
///
/// In most cases both endpoints of the intervals can be accurately recovered
/// from a context switch trace, but in some cases one of these endpoints might
/// be guessed or not known at all, due to contention problems in the trace or
/// because tracing was interrupted.
///
/// Note: we use the terms CPU and cores interchangeably.
struct ThreadContinuousExecution {
  enum class Variant {
    /// Both endpoints are known
    Complete,
    /// The end is known and we have a guess for the start
    HintedStart,
    /// The start is known and we have a guess for the end
    HintedEnd,
    /// We only know the start. This might be the last entry of a core trace.
    OnlyStart,
    /// We only know the end. This might be the first entry or a core trace.
    OnlyEnd,
  } variant;

  union {
    struct {
      uint64_t start;
      uint64_t end;
    } complete;
    struct {
      uint64_t start;
    } only_start;
    struct {
      uint64_t end;
    } only_end;
    /// The following 'hinted' structures are useful when there are contention
    /// problems in the trace
    struct {
      uint64_t hinted_start;
      uint64_t end;
    } hinted_start;
    struct {
      uint64_t start;
      uint64_t hinted_end;
    } hinted_end;
  } tscs;

  lldb::core_id_t core_id;
  lldb::tid_t tid;

  /// \return
  ///   A tsc that we are certain of, either the start or the end.
  uint64_t GetErrorFreeTSC() const;

  /// Constructors for the different variants of this object
  ///
  /// \{
  static ThreadContinuousExecution
  CreateCompleteExecution(lldb::core_id_t core_id, lldb::tid_t tid,
                          uint64_t start, uint64_t end);

  static ThreadContinuousExecution
  CreateHintedStartExecution(lldb::core_id_t core_id, lldb::tid_t tid,
                             uint64_t hinted_start, uint64_t end);

  static ThreadContinuousExecution
  CreateHintedEndExecution(lldb::core_id_t core_id, lldb::tid_t tid,
                           uint64_t start, uint64_t hinted_end);

  static ThreadContinuousExecution
  CreateOnlyEndExecution(lldb::core_id_t core_id, lldb::tid_t tid,
                         uint64_t end);

  static ThreadContinuousExecution
  CreateOnlyStartExecution(lldb::core_id_t core_id, lldb::tid_t tid,
                           uint64_t start);
  /// \}

  /// Comparator by TSCs
  bool operator<(const ThreadContinuousExecution &o) const;

private:
  ThreadContinuousExecution(lldb::core_id_t core_id, lldb::tid_t tid)
      : core_id(core_id), tid(tid) {}
};

/// Class used to decode a multi-core Intel PT trace. It assumes that each
/// thread could have potentially been executed on different cores. It uses a
/// context switch trace per CPU with timestamps to identify which thread owns
/// each Intel PT decoded instruction and in which order. It also assumes that
/// the Intel PT data and context switches might have gaps in their traces due
/// to contention or race conditions.
class TraceIntelPTMultiCoreDecoder {
public:
  /// \param[in] core_ids
  ///   The list of cores where the traced programs were running on.
  ///
  /// \param[in] tid
  ///   The full list of tids that were traced.
  ///
  /// \param[in] tsc_conversion
  ///   The conversion values for converting between nanoseconds and TSCs.
  TraceIntelPTMultiCoreDecoder(
      TraceIntelPT &trace, llvm::ArrayRef<lldb::core_id_t> core_ids,
      llvm::ArrayRef<lldb::tid_t> tids,
      const LinuxPerfZeroTscConversion &tsc_conversion);

  /// \return
  ///   A \a DecodedThread for the \p thread by decoding its instructions on all
  ///   CPUs, sorted by TSCs.
  DecodedThreadSP Decode(Thread &thread);

  /// \return
  ///   \b true if the given \p tid is managed by this decoder, regardless of
  ///   whether there's tracing data associated to it or not.
  bool TracesThread(lldb::tid_t tid) const;

  /// \return
  ///   The number of continuous executions found for the given \p tid.
  size_t GetNumContinuousExecutionsForThread(lldb::tid_t tid) const;

  /// \return
  ///   The total number of continuous executions found across CPUs.
  size_t GetTotalContinuousExecutionsCount() const;

private:
  /// Traverse the context switch traces and recover the continuous executions
  /// by thread.
  llvm::Error DecodeContextSwitchTraces();

  TraceIntelPT &m_trace;
  std::set<lldb::core_id_t> m_cores;
  std::set<lldb::tid_t> m_tids;
  llvm::Optional<
      llvm::DenseMap<lldb::tid_t, std::vector<ThreadContinuousExecution>>>
      m_continuous_executions_per_thread;
  LinuxPerfZeroTscConversion m_tsc_conversion;
  /// This variable will be non-None if a severe error happened during the setup
  /// of the decoder.
  llvm::Optional<std::string> m_setup_error;
};

} // namespace trace_intel_pt
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_TRACEINTELPTMULTICOREDECODER_H
