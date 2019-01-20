//===-- Decoder.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef Decoder_h_
#define Decoder_h_

// C/C++ Includes
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBStructuredData.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBTrace.h"
#include "lldb/API/SBTraceOptions.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-types.h"

#include "intel-pt.h"

namespace ptdecoder_private {
//----------------------------------------------------------------------
/// @class Instruction
/// Represents an assembly instruction containing raw
///     instruction bytes, instruction address along with information
///     regarding execution flow context and Intel(R) Processor Trace
///     context.
//----------------------------------------------------------------------
class Instruction {
public:
  Instruction() : ip(0), data(), error(), iclass(ptic_error), speculative(0) {}

  Instruction(const Instruction &insn) = default;

  Instruction(const struct pt_insn &insn)
      : ip(insn.ip), data(), error(insn.size == 0 ? "invalid instruction" : ""),
        iclass(insn.iclass), speculative(insn.speculative) {
    if (insn.size != 0)
      data.assign(insn.raw, insn.raw + insn.size);
  }

  Instruction(const char *err)
      : ip(0), data(), error(err ? err : "unknown error"), iclass(ptic_error),
        speculative(0) {}

  ~Instruction() {}

  uint64_t GetInsnAddress() const { return ip; }

  size_t GetRawBytes(void *buf, size_t size) const {
    if ((buf == nullptr) || (size == 0))
      return data.size();

    size_t bytes_to_read = ((size <= data.size()) ? size : data.size());
    ::memcpy(buf, data.data(), bytes_to_read);
    return bytes_to_read;
  }

  const std::string &GetError() const { return error; }

  bool GetSpeculative() const { return speculative; }

private:
  uint64_t ip;               // instruction address in inferior's memory image
  std::vector<uint8_t> data; // raw bytes
  std::string error;         // Error string if instruction is invalid
  enum pt_insn_class iclass; // classification of the instruction
  // A collection of flags giving additional information about instruction
  uint32_t speculative : 1; // Instruction was executed speculatively or not
};

//---------------------------------------------------------------------------
/// @class InstructionList
/// Represents a list of assembly instructions. Each instruction is of
///     type Instruction.
//---------------------------------------------------------------------------
class InstructionList {
public:
  InstructionList() : m_insn_vec() {}

  InstructionList(const InstructionList &insn_list)
      : m_insn_vec(insn_list.m_insn_vec) {}

  ~InstructionList() {}

  // Get number of instructions in the list
  size_t GetSize() const { return m_insn_vec.size(); }

  // Get instruction at index
  Instruction GetInstructionAtIndex(uint32_t idx) {
    return (idx < m_insn_vec.size() ? m_insn_vec[idx]
                                    : Instruction("invalid instruction"));
  }

  // Append intruction at the end of the list
  void AppendInstruction(Instruction inst) { m_insn_vec.push_back(inst); }

private:
  std::vector<Instruction> m_insn_vec;
};

//----------------------------------------------------------------------
/// @class TraceOptions
/// Provides Intel(R) Processor Trace specific configuration options and
///     other information obtained by decoding and post-processing the trace
///     data. Currently, this information comprises of the total number of
///     assembly instructions executed for an inferior.
//----------------------------------------------------------------------
class TraceOptions : public lldb::SBTraceOptions {
public:
  TraceOptions() : lldb::SBTraceOptions(), m_insn_log_size(0) {}

  ~TraceOptions() {}

  //------------------------------------------------------------------
  /// Get total number of assembly instructions obtained after decoding the
  /// complete Intel(R) Processor Trace data obtained from LLDB.
  ///
  /// @return
  ///     Total number of instructions.
  //------------------------------------------------------------------
  uint32_t getInstructionLogSize() const { return m_insn_log_size; }

  //------------------------------------------------------------------
  /// Set total number of assembly instructions.
  ///
  /// @param[in] size
  ///     Value to be set.
  //------------------------------------------------------------------
  void setInstructionLogSize(uint32_t size) { m_insn_log_size = size; }

private:
  uint32_t m_insn_log_size;
};

//----------------------------------------------------------------------
/// @class Decoder
/// This class makes use of Intel(R) Processor Trace hardware feature
///     (implememted inside LLDB) to gather trace data for an inferior (being
///     debugged with LLDB) to provide meaningful information out of it.
///
///     Currently the meaningful information comprises of the execution flow
///     of the inferior (in terms of assembly instructions executed). The class
///     enables user to:
///     - start the trace with configuration options for a thread/process,
///     - stop the trace for a thread/process,
///     - get the execution flow (assembly instructions) for a thread and
///     - get trace specific information for a thread
//----------------------------------------------------------------------
class Decoder {
public:
  typedef std::vector<Instruction> Instructions;

  Decoder(lldb::SBDebugger &sbdebugger)
      : m_mapProcessUID_mapThreadID_TraceInfo_mutex(),
        m_mapProcessUID_mapThreadID_TraceInfo(),
        m_debugger_user_id(sbdebugger.GetID()) {}

  ~Decoder() {}

  void StartProcessorTrace(lldb::SBProcess &sbprocess,
                           lldb::SBTraceOptions &sbtraceoptions,
                           lldb::SBError &sberror);

  void StopProcessorTrace(lldb::SBProcess &sbprocess, lldb::SBError &sberror,
                          lldb::tid_t tid = LLDB_INVALID_THREAD_ID);

  void GetInstructionLogAtOffset(lldb::SBProcess &sbprocess, lldb::tid_t tid,
                                 uint32_t offset, uint32_t count,
                                 InstructionList &result_list,
                                 lldb::SBError &sberror);

  void GetProcessorTraceInfo(lldb::SBProcess &sbprocess, lldb::tid_t tid,
                             TraceOptions &traceinfo, lldb::SBError &sberror);

private:
  class ThreadTraceInfo;
  typedef std::vector<uint8_t> Buffer;

  // internal class to manage inferior's read-execute section information
  class ReadExecuteSectionInfo {
  public:
    uint64_t load_address;
    uint64_t file_offset;
    uint64_t size;
    std::string image_path;

    ReadExecuteSectionInfo(const uint64_t addr, const uint64_t offset,
                           const uint64_t sz, const std::string &path)
        : load_address(addr), file_offset(offset), size(sz), image_path(path) {}

    ReadExecuteSectionInfo(const ReadExecuteSectionInfo &rxsection) = default;
  };

  typedef struct pt_cpu CPUInfo;
  typedef std::vector<ReadExecuteSectionInfo> ReadExecuteSectionInfos;

  // Check whether the provided SBProcess belongs to the same SBDebugger with
  // which Decoder class instance was constructed.
  void CheckDebuggerID(lldb::SBProcess &sbprocess, lldb::SBError &sberror);

  // Function to remove entries of finished processes/threads in the class
  void RemoveDeadProcessesAndThreads(lldb::SBProcess &sbprocess);

  // Parse cpu information from trace configuration received from LLDB
  void ParseCPUInfo(CPUInfo &pt_cpu, lldb::SBStructuredData &s,
                    lldb::SBError &sberror);

  ///------------------------------------------------------------------------
  /// Function performs following tasks for a given process and thread:
  ///  - Checks if the given thread is registered in the class or not. If not
  ///  then tries to register it if trace was ever started on the entire
  ///  process. Else returns error.
  ///  - fetches trace and other necessary information from LLDB (using
  ///  ReadTraceDataAndImageInfo()) and decodes the trace (using
  ///  DecodeProcessorTrace())
  ///------------------------------------------------------------------------
  void FetchAndDecode(lldb::SBProcess &sbprocess, lldb::tid_t tid,
                      lldb::SBError &sberror,
                      ThreadTraceInfo **threadTraceInfo);

  // Helper function of FetchAndDecode() to get raw trace data and memory image
  // info of inferior from LLDB
  void ReadTraceDataAndImageInfo(lldb::SBProcess &sbprocess, lldb::tid_t tid,
                                 lldb::SBError &sberror,
                                 ThreadTraceInfo &threadTraceInfo);

  // Helper function of FetchAndDecode() to initialize raw trace decoder and
  // start trace decoding
  void DecodeProcessorTrace(lldb::SBProcess &sbprocess, lldb::tid_t tid,
                            lldb::SBError &sberror,
                            ThreadTraceInfo &threadTraceInfo);

  // Helper function of ReadTraceDataAndImageInfo() function for gathering
  // inferior's memory image info along with all dynamic libraries linked with
  // it
  void GetTargetModulesInfo(lldb::SBTarget &sbtarget,
                            ReadExecuteSectionInfos &readExecuteSectionInfos,
                            lldb::SBError &sberror);

  ///------------------------------------------------------------------------
  /// Helper functions of DecodeProcessorTrace() function for:
  ///  - initializing raw trace decoder (provided by Intel(R) Processor Trace
  ///    Decoding library)
  ///  - start trace decoding
  ///------------------------------------------------------------------------
  void InitializePTInstDecoder(
      struct pt_insn_decoder **decoder, struct pt_config *config,
      const CPUInfo &pt_cpu, Buffer &pt_buffer,
      const ReadExecuteSectionInfos &readExecuteSectionInfos,
      lldb::SBError &sberror) const;
  void DecodeTrace(struct pt_insn_decoder *decoder,
                   Instructions &instruction_list, lldb::SBError &sberror);

  // Function to diagnose and indicate errors during raw trace decoding
  void Diagnose(struct pt_insn_decoder *decoder, int errcode,
                lldb::SBError &sberror, const struct pt_insn *insn = nullptr);

  class ThreadTraceInfo {
  public:
    ThreadTraceInfo()
        : m_pt_buffer(), m_readExecuteSectionInfos(), m_thread_stop_id(0),
          m_trace(), m_pt_cpu(), m_instruction_log() {}

    ThreadTraceInfo(const ThreadTraceInfo &trace_info) = default;

    ~ThreadTraceInfo() {}

    Buffer &GetPTBuffer() { return m_pt_buffer; }

    void AllocatePTBuffer(uint64_t size) { m_pt_buffer.assign(size, 0); }

    ReadExecuteSectionInfos &GetReadExecuteSectionInfos() {
      return m_readExecuteSectionInfos;
    }

    CPUInfo &GetCPUInfo() { return m_pt_cpu; }

    Instructions &GetInstructionLog() { return m_instruction_log; }

    uint32_t GetStopID() const { return m_thread_stop_id; }

    void SetStopID(uint32_t stop_id) { m_thread_stop_id = stop_id; }

    lldb::SBTrace &GetUniqueTraceInstance() { return m_trace; }

    void SetUniqueTraceInstance(lldb::SBTrace &trace) { m_trace = trace; }

    friend class Decoder;

  private:
    Buffer m_pt_buffer; // raw trace buffer
    ReadExecuteSectionInfos
        m_readExecuteSectionInfos; // inferior's memory image info
    uint32_t m_thread_stop_id;     // stop id for thread
    lldb::SBTrace m_trace; // unique tracing instance of a thread/process
    CPUInfo m_pt_cpu; // cpu info of the target on which inferior is running
    Instructions m_instruction_log; // complete instruction log
  };

  typedef std::map<lldb::user_id_t, ThreadTraceInfo> MapThreadID_TraceInfo;
  typedef std::map<uint32_t, MapThreadID_TraceInfo>
      MapProcessUID_MapThreadID_TraceInfo;

  std::mutex m_mapProcessUID_mapThreadID_TraceInfo_mutex;
  MapProcessUID_MapThreadID_TraceInfo
      m_mapProcessUID_mapThreadID_TraceInfo; // to store trace information for
                                             // each process and its associated
                                             // threads
  lldb::user_id_t m_debugger_user_id; // SBDebugger instance which is associated
                                      // to this Decoder instance
};

} // namespace ptdecoder_private
#endif // Decoder_h_
