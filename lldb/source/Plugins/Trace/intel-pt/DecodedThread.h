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

/// \class IntelPTInstruction
/// An instruction obtained from decoding a trace. It is either an actual
/// instruction or an error indicating a gap in the trace.
///
/// Gaps in the trace can come in a few flavors:
///   - tracing gaps (e.g. tracing was paused and then resumed)
///   - tracing errors (e.g. buffer overflow)
///   - decoding errors (e.g. some memory region couldn't be decoded)
/// As mentioned, any gap is represented as an error in this class.
class IntelPTInstruction {
public:
  IntelPTInstruction(const pt_insn &pt_insn, uint64_t timestamp)
      : m_pt_insn(pt_insn), m_timestamp(timestamp), m_is_error(false) {}

  IntelPTInstruction(const pt_insn &pt_insn)
      : m_pt_insn(pt_insn), m_is_error(false) {}

  /// Error constructor
  IntelPTInstruction();

  /// Check if this object represents an error (i.e. a gap).
  ///
  /// \return
  ///     Whether this object represents an error.
  bool IsError() const;

  /// \return
  ///     The instruction pointer address, or \a LLDB_INVALID_ADDRESS if it is
  ///     an error.
  lldb::addr_t GetLoadAddress() const;

  /// Get the size in bytes of an instance of this class
  static size_t GetMemoryUsage();

  /// Get the timestamp associated with the current instruction. The timestamp
  /// is similar to what a rdtsc instruction would return.
  ///
  /// \return
  ///     The timestamp or \b llvm::None if not available.
  llvm::Optional<uint64_t> GetTimestampCounter() const;

  /// Get the \a lldb::TraceInstructionControlFlowType categories of the
  /// instruction.
  ///
  /// \param[in] next_load_address
  ///     The address of the next instruction in the trace or \b
  ///     LLDB_INVALID_ADDRESS if not available.
  ///
  /// \return
  ///     The control flow categories, or \b 0 if the instruction is an error.
  lldb::TraceInstructionControlFlowType
  GetControlFlowType(lldb::addr_t next_load_address) const;

  IntelPTInstruction(IntelPTInstruction &&other) = default;

private:
  IntelPTInstruction(const IntelPTInstruction &other) = delete;
  const IntelPTInstruction &operator=(const IntelPTInstruction &other) = delete;

  // When adding new members to this class, make sure to update
  // IntelPTInstruction::GetNonErrorMemoryUsage() if needed.
  pt_insn m_pt_insn;
  llvm::Optional<uint64_t> m_timestamp;
  bool m_is_error;
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
  DecodedThread(lldb::ThreadSP thread_sp);

  /// Utility constructor that initializes the trace with a provided error.
  DecodedThread(lldb::ThreadSP thread_sp, llvm::Error &&err);

  /// Get the instructions from the decoded trace. Some of them might indicate
  /// errors (i.e. gaps) in the trace. For an instruction error, you can access
  /// its underlying error message with the \a GetErrorByInstructionIndex()
  /// method.
  ///
  /// \return
  ///   The instructions of the trace.
  llvm::ArrayRef<IntelPTInstruction> GetInstructions() const;

  /// Get the error associated with a given instruction index.
  ///
  /// \return
  ///   The error message of \b nullptr if the given index
  ///   points to a valid instruction.
  const char *GetErrorByInstructionIndex(uint64_t ins_idx);

  /// Append a successfully decoded instruction.
  template <typename... Ts> void AppendInstruction(Ts... instruction_args) {
    m_instructions.emplace_back(instruction_args...);
  }

  /// Append a decoding error (i.e. an instruction that failed to be decoded).
  void AppendError(llvm::Error &&error);

  /// Get a new cursor for the decoded thread.
  lldb::TraceCursorUP GetCursor();

  /// Set the size in bytes of the corresponding Intel PT raw trace.
  void SetRawTraceSize(size_t size);

  /// Get the size in bytes of the corresponding Intel PT raw trace.
  ///
  /// \return
  ///   The size of the trace, or \b llvm::None if not available.
  llvm::Optional<size_t> GetRawTraceSize() const;

  /// The approximate size in bytes used by this instance,
  /// including all the already decoded instructions.
  size_t CalculateApproximateMemoryUsage() const;

  lldb::ThreadSP GetThread();

private:
  /// When adding new members to this class, make sure
  /// to update \a CalculateApproximateMemoryUsage() accordingly.
  lldb::ThreadSP m_thread_sp;
  std::vector<IntelPTInstruction> m_instructions;
  llvm::DenseMap<uint64_t, std::string> m_errors;
  llvm::Optional<size_t> m_raw_trace_size;
};

using DecodedThreadSP = std::shared_ptr<DecodedThread>;

} // namespace trace_intel_pt
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TRACE_INTEL_PT_DECODEDTHREAD_H
