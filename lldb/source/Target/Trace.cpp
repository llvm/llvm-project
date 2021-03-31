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
static SymbolContext DumpSymbolContext(Stream &s, const SymbolContext &prev_sc,
                                       Target &target, const Address &address) {
  AddressRange range;
  if (prev_sc.GetAddressRange(eSymbolContextEverything, 0,
                              /*inline_block_range*/ false, range) &&
      range.ContainsFileAddress(address))
    return prev_sc;

  SymbolContext sc;
  address.CalculateSymbolContext(&sc, eSymbolContextEverything);

  if (!prev_sc.module_sp && !sc.module_sp)
    return sc;
  if (prev_sc.module_sp == sc.module_sp && !sc.function && !sc.symbol &&
      !prev_sc.function && !prev_sc.symbol)
    return sc;

  s.Printf("  ");

  if (!sc.module_sp)
    s.Printf("(none)");
  else if (!sc.function && !sc.symbol)
    s.Printf("%s`(none)",
             sc.module_sp->GetFileSpec().GetFilename().AsCString());
  else
    sc.DumpStopContext(&s, &target, address, /*show_fullpath*/ false,
                       /*show_module*/ true, /*show_inlined_frames*/ false,
                       /*show_function_arguments*/ true,
                       /*show_function_name*/ true,
                       /*show_inline_callsite_line_info*/ false);
  s.Printf("\n");
  return sc;
}

/// Dump an instruction given by its address using a given disassembler, unless
/// the instruction is not present in the disassembler.
///
/// \param[in] disassembler
///     A disassembler containing a certain instruction list.
///
/// \param[in] address
///     The address of the instruction to dump.
///
/// \return
///     \b true if the information could be dumped, \b false otherwise.
static bool TryDumpInstructionInfo(Stream &s,
                                   const DisassemblerSP &disassembler,
                                   const ExecutionContext &exe_ctx,
                                   const Address &address) {
  if (!disassembler)
    return false;

  if (InstructionSP instruction =
          disassembler->GetInstructionList().GetInstructionAtAddress(address)) {
    instruction->Dump(&s, /*show_address*/ false, /*show_bytes*/ false,
                      /*max_opcode_byte_size*/ 0, &exe_ctx,
                      /*sym_ctx*/ nullptr, /*prev_sym_ctx*/ nullptr,
                      /*disassembly_addr_format*/ nullptr,
                      /*max_address_text_size*/ 0);
    return true;
  }

  return false;
}

/// Dump an instruction instruction given by its address.
///
/// \param[in] prev_disassembler
///     The disassembler that was used to dump the previous instruction in the
///     trace. It is useful to avoid recomputations.
///
/// \param[in] address
///     The address of the instruction to dump.
///
/// \return
///     A disassembler that contains the given instruction, which might differ
///     from the previous disassembler.
static DisassemblerSP
DumpInstructionInfo(Stream &s, const SymbolContext &sc,
                    const DisassemblerSP &prev_disassembler,
                    ExecutionContext &exe_ctx, const Address &address) {
  // We first try to use the previous disassembler
  if (TryDumpInstructionInfo(s, prev_disassembler, exe_ctx, address))
    return prev_disassembler;

  // Now we try using the current function's disassembler
  if (sc.function) {
    DisassemblerSP disassembler =
        sc.function->GetInstructions(exe_ctx, nullptr, true);
    if (TryDumpInstructionInfo(s, disassembler, exe_ctx, address))
      return disassembler;
  }

  // We fallback to disassembly one instruction
  Target &target = exe_ctx.GetTargetRef();
  const ArchSpec &arch = target.GetArchitecture();
  AddressRange range(address, arch.GetMaximumOpcodeByteSize() * 1);
  DisassemblerSP disassembler = Disassembler::DisassembleRange(
      arch, /*plugin_name*/ nullptr,
      /*flavor*/ nullptr, target, range, /*prefer_file_cache*/ true);
  if (TryDumpInstructionInfo(s, disassembler, exe_ctx, address))
    return disassembler;
  return nullptr;
}

void Trace::DumpTraceInstructions(Thread &thread, Stream &s, size_t count,
                                  size_t end_position, bool raw) {
  if (!IsTraced(thread)) {
    s.Printf("thread #%u: tid = %" PRIu64 ", not traced\n", thread.GetIndexID(),
             thread.GetID());
    return;
  }

  size_t instructions_count = GetInstructionCount(thread);
  s.Printf("thread #%u: tid = %" PRIu64 ", total instructions = %zu\n",
           thread.GetIndexID(), thread.GetID(), instructions_count);

  if (count == 0 || end_position >= instructions_count)
    return;

  size_t start_position =
      end_position + 1 < count ? 0 : end_position + 1 - count;

  int digits_count = GetNumberOfDigits(end_position);
  auto printInstructionIndex = [&](size_t index) {
    s.Printf("    [%*zu] ", digits_count, index);
  };

  bool was_prev_instruction_an_error = false;
  Target &target = thread.GetProcess()->GetTarget();

  SymbolContext sc;
  DisassemblerSP disassembler;
  ExecutionContext exe_ctx;
  target.CalculateExecutionContext(exe_ctx);

  TraverseInstructions(
      thread, start_position, TraceDirection::Forwards,
      [&](size_t index, Expected<lldb::addr_t> load_address) -> bool {
        if (load_address) {
          // We print an empty line after a sequence of errors to show more
          // clearly that there's a gap in the trace
          if (was_prev_instruction_an_error)
            s.Printf("    ...missing instructions\n");

          Address address;
          if (!raw) {
            target.GetSectionLoadList().ResolveLoadAddress(*load_address,
                                                           address);

            sc = DumpSymbolContext(s, sc, target, address);
          }

          printInstructionIndex(index);
          s.Printf("0x%016" PRIx64 "    ", *load_address);

          if (!raw) {
            disassembler =
                DumpInstructionInfo(s, sc, disassembler, exe_ctx, address);
          }

          was_prev_instruction_an_error = false;
        } else {
          printInstructionIndex(index);
          s << toString(load_address.takeError());
          was_prev_instruction_an_error = true;
          if (!raw)
            sc = SymbolContext();
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

Error Trace::StopProcess() {
  if (!m_live_process)
    return createStringError(inconvertibleErrorCode(),
                             "Tracing requires a live process.");
  return m_live_process->TraceStop(
      TraceStopRequest(GetPluginName().AsCString()));
}

Error Trace::StopThreads(const std::vector<lldb::tid_t> &tids) {
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

Expected<std::vector<uint8_t>>
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

Expected<std::vector<uint8_t>>
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
