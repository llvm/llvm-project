//===-- Trace.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Trace.h"

#include "llvm/Support/Format.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPostMortemTrace.h"
#include "lldb/Utility/Stream.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

// Helper structs used to extract the type of a trace session json without
// having to parse the entire object.

struct JSONSimplePluginSettings {
  std::string type;
};

struct JSONSimpleTraceSession {
  JSONSimplePluginSettings trace;
};

namespace llvm {
namespace json {

bool fromJSON(const Value &value, JSONSimplePluginSettings &plugin_settings,
              Path path) {
  json::ObjectMapper o(value, path);
  return o && o.map("type", plugin_settings.type);
}

bool fromJSON(const Value &value, JSONSimpleTraceSession &session, Path path) {
  json::ObjectMapper o(value, path);
  return o && o.map("trace", session.trace);
}

} // namespace json
} // namespace llvm

static Error createInvalidPlugInError(StringRef plugin_name) {
  return createStringError(
      std::errc::invalid_argument,
      "no trace plug-in matches the specified type: \"%s\"",
      plugin_name.data());
}

Expected<lldb::TraceSP>
Trace::FindPluginForPostMortemProcess(Debugger &debugger,
                                      const json::Value &trace_session_file,
                                      StringRef session_file_dir) {
  JSONSimpleTraceSession json_session;
  json::Path::Root root("traceSession");
  if (!json::fromJSON(trace_session_file, json_session, root))
    return root.getError();

  ConstString plugin_name(json_session.trace.type);
  if (auto create_callback = PluginManager::GetTraceCreateCallback(plugin_name))
    return create_callback(trace_session_file, session_file_dir, debugger);

  return createInvalidPlugInError(json_session.trace.type);
}

Expected<lldb::TraceSP>
Trace::FindPluginForLiveProcess(llvm::StringRef plugin_name, Process &process) {
  if (!process.IsLiveDebugSession())
    return createStringError(inconvertibleErrorCode(),
                             "Can't trace non-live processes");

  ConstString name(plugin_name);
  if (auto create_callback =
          PluginManager::GetTraceCreateCallbackForLiveProcess(name))
    return create_callback(process);

  return createInvalidPlugInError(plugin_name);
}

Expected<StringRef> Trace::FindPluginSchema(StringRef name) {
  ConstString plugin_name(name);
  StringRef schema = PluginManager::GetTraceSchema(plugin_name);
  if (!schema.empty())
    return schema;

  return createInvalidPlugInError(name);
}

static int GetNumberOfDigits(size_t num) {
  return num == 0 ? 1 : static_cast<int>(log10(num)) + 1;
}

/// \return
///     \b true if the provided line entries match line, column and source file.
///     This function assumes that the line entries are valid.
static bool FileLineAndColumnMatches(const LineEntry &a, const LineEntry &b) {
  if (a.line != b.line)
    return false;
  if (a.column != b.column)
    return false;
  return a.file == b.file;
}

// This custom LineEntry validator is neded because some line_entries have
// 0 as line, which is meaningless. Notice that LineEntry::IsValid only
// checks that line is not LLDB_INVALID_LINE_NUMBER, i.e. UINT32_MAX.
static bool IsLineEntryValid(const LineEntry &line_entry) {
  return line_entry.IsValid() && line_entry.line > 0;
}

/// Helper structure for \a TraverseInstructionsWithSymbolInfo.
struct InstructionSymbolInfo {
  SymbolContext sc;
  Address address;
  lldb::addr_t load_address;
  lldb::DisassemblerSP disassembler;
  lldb::InstructionSP instruction;
  lldb_private::ExecutionContext exe_ctx;
};

/// InstructionSymbolInfo object with symbol information for the given
/// instruction, calculated efficiently.
///
/// \param[in] symbol_scope
///     If not \b 0, then the \a InstructionSymbolInfo will have its
///     SymbolContext calculated up to that level.
///
/// \param[in] include_disassembler
///     If \b true, then the \a InstructionSymbolInfo will have the
///     \a disassembler and \a instruction objects calculated.
static void TraverseInstructionsWithSymbolInfo(
    Trace &trace, Thread &thread, size_t position,
    Trace::TraceDirection direction, SymbolContextItem symbol_scope,
    bool include_disassembler,
    std::function<bool(size_t index, Expected<InstructionSymbolInfo> insn)>
        callback) {
  InstructionSymbolInfo prev_insn;

  Target &target = thread.GetProcess()->GetTarget();
  ExecutionContext exe_ctx;
  target.CalculateExecutionContext(exe_ctx);
  const ArchSpec &arch = target.GetArchitecture();

  // Find the symbol context for the given address reusing the previous
  // instruction's symbol context when possible.
  auto calculate_symbol_context = [&](const Address &address) {
    AddressRange range;
    if (prev_insn.sc.GetAddressRange(symbol_scope, 0,
                                     /*inline_block_range*/ false, range) &&
        range.Contains(address))
      return prev_insn.sc;

    SymbolContext sc;
    address.CalculateSymbolContext(&sc, symbol_scope);
    return sc;
  };

  // Find the disassembler for the given address reusing the previous
  // instruction's disassembler when possible.
  auto calculate_disass = [&](const Address &address, const SymbolContext &sc) {
    if (prev_insn.disassembler) {
      if (InstructionSP instruction =
              prev_insn.disassembler->GetInstructionList()
                  .GetInstructionAtAddress(address))
        return std::make_tuple(prev_insn.disassembler, instruction);
    }

    if (sc.function) {
      if (DisassemblerSP disassembler =
              sc.function->GetInstructions(exe_ctx, nullptr)) {
        if (InstructionSP instruction =
                disassembler->GetInstructionList().GetInstructionAtAddress(
                    address))
          return std::make_tuple(disassembler, instruction);
      }
    }
    // We fallback to a single instruction disassembler
    AddressRange range(address, arch.GetMaximumOpcodeByteSize());
    DisassemblerSP disassembler =
        Disassembler::DisassembleRange(arch, /*plugin_name*/ nullptr,
                                       /*flavor*/ nullptr, target, range);
    return std::make_tuple(disassembler,
                           disassembler ? disassembler->GetInstructionList()
                                              .GetInstructionAtAddress(address)
                                        : InstructionSP());
  };

  trace.TraverseInstructions(
      thread, position, direction,
      [&](size_t index, Expected<lldb::addr_t> load_address) -> bool {
        if (!load_address)
          return callback(index, load_address.takeError());

        InstructionSymbolInfo insn;
        insn.load_address = *load_address;
        insn.exe_ctx = exe_ctx;
        insn.address.SetLoadAddress(*load_address, &target);
        if (symbol_scope != 0)
          insn.sc = calculate_symbol_context(insn.address);
        if (include_disassembler)
          std::tie(insn.disassembler, insn.instruction) =
              calculate_disass(insn.address, insn.sc);
        prev_insn = insn;
        return callback(index, insn);
      });
}

/// Compare the symbol contexts of the provided \a InstructionSymbolInfo
/// objects.
///
/// \return
///     \a true if both instructions belong to the same scope level analized
///     in the following order:
///       - module
///       - symbol
///       - function
///       - line
static bool
IsSameInstructionSymbolContext(const InstructionSymbolInfo &prev_insn,
                               const InstructionSymbolInfo &insn) {
  // module checks
  if (insn.sc.module_sp != prev_insn.sc.module_sp)
    return false;

  // symbol checks
  if (insn.sc.symbol != prev_insn.sc.symbol)
    return false;

  // function checks
  if (!insn.sc.function && !prev_insn.sc.function)
    return true;
  else if (insn.sc.function != prev_insn.sc.function)
    return false;

  // line entry checks
  const bool curr_line_valid = IsLineEntryValid(insn.sc.line_entry);
  const bool prev_line_valid = IsLineEntryValid(prev_insn.sc.line_entry);
  if (curr_line_valid && prev_line_valid)
    return FileLineAndColumnMatches(insn.sc.line_entry,
                                    prev_insn.sc.line_entry);
  return curr_line_valid == prev_line_valid;
}

/// Dump the symbol context of the given instruction address if it's different
/// from the symbol context of the previous instruction in the trace.
///
/// \param[in] prev_sc
///     The symbol context of the previous instruction in the trace.
///
/// \param[in] address
///     The address whose symbol information will be dumped.
///
/// \return
///     The symbol context of the current address, which might differ from the
///     previous one.
static void
DumpInstructionSymbolContext(Stream &s,
                             Optional<InstructionSymbolInfo> prev_insn,
                             InstructionSymbolInfo &insn) {
  if (prev_insn && IsSameInstructionSymbolContext(*prev_insn, insn))
    return;

  s.Printf("  ");

  if (!insn.sc.module_sp)
    s.Printf("(none)");
  else if (!insn.sc.function && !insn.sc.symbol)
    s.Printf("%s`(none)",
             insn.sc.module_sp->GetFileSpec().GetFilename().AsCString());
  else
    insn.sc.DumpStopContext(&s, insn.exe_ctx.GetTargetPtr(), insn.address,
                            /*show_fullpath*/ false,
                            /*show_module*/ true, /*show_inlined_frames*/ false,
                            /*show_function_arguments*/ true,
                            /*show_function_name*/ true);
  s.Printf("\n");
}

static void DumpInstructionDisassembly(Stream &s, InstructionSymbolInfo &insn) {
  if (!insn.instruction)
    return;
  s.Printf("    ");
  insn.instruction->Dump(&s, /*show_address*/ false, /*show_bytes*/ false,
                         /*max_opcode_byte_size*/ 0, &insn.exe_ctx, &insn.sc,
                         /*prev_sym_ctx*/ nullptr,
                         /*disassembly_addr_format*/ nullptr,
                         /*max_address_text_size*/ 0);
}

void Trace::DumpTraceInstructions(Thread &thread, Stream &s, size_t count,
                                  size_t end_position, bool raw) {
  Optional<size_t> instructions_count = GetInstructionCount(thread);
  if (!instructions_count) {
    s.Printf("thread #%u: tid = %" PRIu64 ", not traced\n", thread.GetIndexID(),
             thread.GetID());
    return;
  }

  s.Printf("thread #%u: tid = %" PRIu64 ", total instructions = %zu\n",
           thread.GetIndexID(), thread.GetID(), *instructions_count);

  if (count == 0 || end_position >= *instructions_count)
    return;

  int digits_count = GetNumberOfDigits(end_position);
  size_t start_position =
      end_position + 1 < count ? 0 : end_position + 1 - count;
  auto printInstructionIndex = [&](size_t index) {
    s.Printf("    [%*zu] ", digits_count, index);
  };

  bool was_prev_instruction_an_error = false;
  Optional<InstructionSymbolInfo> prev_insn;

  TraverseInstructionsWithSymbolInfo(
      *this, thread, start_position, TraceDirection::Forwards,
      eSymbolContextEverything, /*disassembler*/ true,
      [&](size_t index, Expected<InstructionSymbolInfo> insn) -> bool {
        if (!insn) {
          printInstructionIndex(index);
          s << toString(insn.takeError());

          prev_insn = None;
          was_prev_instruction_an_error = true;
        } else {
          if (was_prev_instruction_an_error)
            s.Printf("    ...missing instructions\n");

          if (!raw)
            DumpInstructionSymbolContext(s, prev_insn, *insn);

          printInstructionIndex(index);
          s.Printf("0x%016" PRIx64, insn->load_address);

          if (!raw)
            DumpInstructionDisassembly(s, *insn);

          prev_insn = *insn;
          was_prev_instruction_an_error = false;
        }

        s.Printf("\n");
        return index < end_position;
      });
}

Error Trace::Start(const llvm::json::Value &request) {
  if (!m_live_process)
    return createStringError(inconvertibleErrorCode(),
                             "Tracing requires a live process.");
  return m_live_process->TraceStart(request);
}

Error Trace::Stop() {
  if (!m_live_process)
    return createStringError(inconvertibleErrorCode(),
                             "Tracing requires a live process.");
  return m_live_process->TraceStop(
      TraceStopRequest(GetPluginName().AsCString()));
}

Error Trace::Stop(llvm::ArrayRef<lldb::tid_t> tids) {
  if (!m_live_process)
    return createStringError(inconvertibleErrorCode(),
                             "Tracing requires a live process.");
  return m_live_process->TraceStop(
      TraceStopRequest(GetPluginName().AsCString(), tids));
}

Expected<std::string> Trace::GetLiveProcessState() {
  if (!m_live_process)
    return createStringError(inconvertibleErrorCode(),
                             "Tracing requires a live process.");
  return m_live_process->TraceGetState(GetPluginName().AsCString());
}

Optional<size_t> Trace::GetLiveThreadBinaryDataSize(lldb::tid_t tid,
                                                    llvm::StringRef kind) {
  auto it = m_live_thread_data.find(tid);
  if (it == m_live_thread_data.end())
    return None;
  std::unordered_map<std::string, size_t> &single_thread_data = it->second;
  auto single_thread_data_it = single_thread_data.find(kind.str());
  if (single_thread_data_it == single_thread_data.end())
    return None;
  return single_thread_data_it->second;
}

Optional<size_t> Trace::GetLiveProcessBinaryDataSize(llvm::StringRef kind) {
  auto data_it = m_live_process_data.find(kind.str());
  if (data_it == m_live_process_data.end())
    return None;
  return data_it->second;
}

Expected<ArrayRef<uint8_t>>
Trace::GetLiveThreadBinaryData(lldb::tid_t tid, llvm::StringRef kind) {
  if (!m_live_process)
    return createStringError(inconvertibleErrorCode(),
                             "Tracing requires a live process.");
  llvm::Optional<size_t> size = GetLiveThreadBinaryDataSize(tid, kind);
  if (!size)
    return createStringError(
        inconvertibleErrorCode(),
        "Tracing data \"%s\" is not available for thread %" PRIu64 ".",
        kind.data(), tid);

  TraceGetBinaryDataRequest request{GetPluginName().AsCString(), kind.str(),
                                    static_cast<int64_t>(tid), 0,
                                    static_cast<int64_t>(*size)};
  return m_live_process->TraceGetBinaryData(request);
}

Expected<ArrayRef<uint8_t>>
Trace::GetLiveProcessBinaryData(llvm::StringRef kind) {
  if (!m_live_process)
    return createStringError(inconvertibleErrorCode(),
                             "Tracing requires a live process.");
  llvm::Optional<size_t> size = GetLiveProcessBinaryDataSize(kind);
  if (!size)
    return createStringError(
        inconvertibleErrorCode(),
        "Tracing data \"%s\" is not available for the process.", kind.data());

  TraceGetBinaryDataRequest request{GetPluginName().AsCString(), kind.str(),
                                    None, 0, static_cast<int64_t>(*size)};
  return m_live_process->TraceGetBinaryData(request);
}

void Trace::RefreshLiveProcessState() {
  if (!m_live_process)
    return;

  uint32_t new_stop_id = m_live_process->GetStopID();
  if (new_stop_id == m_stop_id)
    return;

  m_stop_id = new_stop_id;
  m_live_thread_data.clear();

  Expected<std::string> json_string = GetLiveProcessState();
  if (!json_string) {
    DoRefreshLiveProcessState(json_string.takeError());
    return;
  }
  Expected<TraceGetStateResponse> live_process_state =
      json::parse<TraceGetStateResponse>(*json_string, "TraceGetStateResponse");
  if (!live_process_state) {
    DoRefreshLiveProcessState(live_process_state.takeError());
    return;
  }

  for (const TraceThreadState &thread_state :
       live_process_state->tracedThreads) {
    for (const TraceBinaryData &item : thread_state.binaryData)
      m_live_thread_data[thread_state.tid][item.kind] = item.size;
  }

  for (const TraceBinaryData &item : live_process_state->processBinaryData)
    m_live_process_data[item.kind] = item.size;

  DoRefreshLiveProcessState(std::move(live_process_state));
}

uint32_t Trace::GetStopID() {
  RefreshLiveProcessState();
  return m_stop_id;
}
