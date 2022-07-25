//===-- TraceDumper.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/TraceDumper.h"

#include "lldb/Core/Module.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/SectionLoadList.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

/// \return
///   The given string or \b None if it's empty.
static Optional<const char *> ToOptionalString(const char *s) {
  if (!s)
    return None;
  return s;
}
/// \return
///   The module name (basename if the module is a file, or the actual name if
///   it's a virtual module), or \b nullptr if no name nor module was found.
static const char *GetModuleName(const TraceDumper::TraceItem &item) {
  if (!item.symbol_info || !item.symbol_info->sc.module_sp)
    return nullptr;
  return item.symbol_info->sc.module_sp->GetFileSpec()
      .GetFilename()
      .AsCString();
}

// This custom LineEntry validator is neded because some line_entries have
// 0 as line, which is meaningless. Notice that LineEntry::IsValid only
// checks that line is not LLDB_INVALID_LINE_NUMBER, i.e. UINT32_MAX.
static bool IsLineEntryValid(const LineEntry &line_entry) {
  return line_entry.IsValid() && line_entry.line > 0;
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

/// Compare the symbol contexts of the provided \a SymbolInfo
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
IsSameInstructionSymbolContext(const TraceDumper::SymbolInfo &prev_insn,
                               const TraceDumper::SymbolInfo &insn) {
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

class OutputWriterCLI : public TraceDumper::OutputWriter {
public:
  OutputWriterCLI(Stream &s, const TraceDumperOptions &options, Thread &thread)
      : m_s(s), m_options(options) {
    m_s.Format("thread #{0}: tid = {1}\n", thread.GetIndexID(), thread.GetID());
  };

  void NoMoreData() override { m_s << "    no more data\n"; }

  void TraceItem(const TraceDumper::TraceItem &item) override {
    if (item.symbol_info) {
      if (!item.prev_symbol_info ||
          !IsSameInstructionSymbolContext(*item.prev_symbol_info,
                                          *item.symbol_info)) {
        m_s << "  ";
        const char *module_name = GetModuleName(item);
        if (!module_name)
          m_s << "(none)";
        else if (!item.symbol_info->sc.function && !item.symbol_info->sc.symbol)
          m_s.Format("{0}`(none)", module_name);
        else
          item.symbol_info->sc.DumpStopContext(
              &m_s, item.symbol_info->exe_ctx.GetTargetPtr(),
              item.symbol_info->address,
              /*show_fullpaths=*/false,
              /*show_module=*/true, /*show_inlined_frames=*/false,
              /*show_function_arguments=*/true,
              /*show_function_name=*/true);
        m_s << "\n";
      }
    }

    if (item.error && !m_was_prev_instruction_an_error)
      m_s << "    ...missing instructions\n";

    m_s.Format("    {0}: ", item.id);

    if (m_options.show_tsc) {
      m_s.Format("[tsc={0}] ",
                 item.tsc ? std::to_string(*item.tsc) : "unavailable");
    }

    if (item.event) {
      m_s << "(event) " << TraceCursor::EventKindToString(*item.event);
      if (*item.event == eTraceEventCPUChanged) {
        m_s.Format(" [new CPU={0}]",
                   item.cpu_id ? std::to_string(*item.cpu_id) : "unavailable");
      }
    } else if (item.error) {
      m_s << "(error) " << *item.error;
    } else {
      m_s.Format("{0:x+16}", item.load_address);
      if (item.symbol_info && item.symbol_info->instruction) {
        m_s << "    ";
        item.symbol_info->instruction->Dump(
            &m_s, /*max_opcode_byte_size=*/0,
            /*show_address=*/false,
            /*show_bytes=*/false, m_options.show_control_flow_kind,
            &item.symbol_info->exe_ctx, &item.symbol_info->sc,
            /*prev_sym_ctx=*/nullptr,
            /*disassembly_addr_format=*/nullptr,
            /*max_address_text_size=*/0);
      }
    }

    m_was_prev_instruction_an_error = (bool)item.error;
    m_s << "\n";
  }

private:
  Stream &m_s;
  TraceDumperOptions m_options;
  bool m_was_prev_instruction_an_error = false;
};

class OutputWriterJSON : public TraceDumper::OutputWriter {
  /* schema:
    error_message: string
    | {
      "event": string,
      "id": decimal,
      "tsc"?: string decimal,
      "cpuId"? decimal,
    } | {
      "error": string,
      "id": decimal,
      "tsc"?: string decimal,
    | {
      "loadAddress": string decimal,
      "id": decimal,
      "tsc"?: string decimal,
      "module"?: string,
      "symbol"?: string,
      "line"?: decimal,
      "column"?: decimal,
      "source"?: string,
      "mnemonic"?: string,
    }
  */
public:
  OutputWriterJSON(Stream &s, const TraceDumperOptions &options)
      : m_s(s), m_options(options),
        m_j(m_s.AsRawOstream(),
            /*IndentSize=*/options.pretty_print_json ? 2 : 0) {
    m_j.arrayBegin();
  };

  ~OutputWriterJSON() { m_j.arrayEnd(); }

  void DumpEvent(const TraceDumper::TraceItem &item) {
    m_j.attribute("event", TraceCursor::EventKindToString(*item.event));
    if (item.event == eTraceEventCPUChanged)
      m_j.attribute("cpuId", item.cpu_id);
  }

  void DumpInstruction(const TraceDumper::TraceItem &item) {
    m_j.attribute("loadAddress", formatv("{0:x}", item.load_address));
    if (item.symbol_info) {
      m_j.attribute("module", ToOptionalString(GetModuleName(item)));
      m_j.attribute(
          "symbol",
          ToOptionalString(item.symbol_info->sc.GetFunctionName().AsCString()));

      if (item.symbol_info->instruction) {
        m_j.attribute("mnemonic",
                      ToOptionalString(item.symbol_info->instruction->GetMnemonic(
                          &item.symbol_info->exe_ctx)));
      }

      if (IsLineEntryValid(item.symbol_info->sc.line_entry)) {
        m_j.attribute(
            "source",
            ToOptionalString(
                item.symbol_info->sc.line_entry.file.GetPath().c_str()));
        m_j.attribute("line", item.symbol_info->sc.line_entry.line);
        m_j.attribute("column", item.symbol_info->sc.line_entry.column);
      }
    }
  }

  void TraceItem(const TraceDumper::TraceItem &item) override {
    m_j.object([&] {
      m_j.attribute("id", item.id);
      if (m_options.show_tsc)
        m_j.attribute(
            "tsc",
            item.tsc ? Optional<std::string>(std::to_string(*item.tsc)) : None);

      if (item.event) {
        DumpEvent(item);
      } else if (item.error) {
        m_j.attribute("error", *item.error);
      } else {
        DumpInstruction(item);
      }
    });
  }

private:
  Stream &m_s;
  TraceDumperOptions m_options;
  json::OStream m_j;
};

static std::unique_ptr<TraceDumper::OutputWriter>
CreateWriter(Stream &s, const TraceDumperOptions &options, Thread &thread) {
  if (options.json)
    return std::unique_ptr<TraceDumper::OutputWriter>(
        new OutputWriterJSON(s, options));
  else
    return std::unique_ptr<TraceDumper::OutputWriter>(
        new OutputWriterCLI(s, options, thread));
}

TraceDumper::TraceDumper(lldb::TraceCursorUP &&cursor_up, Stream &s,
                         const TraceDumperOptions &options)
    : m_cursor_up(std::move(cursor_up)), m_options(options),
      m_writer_up(CreateWriter(
          s, m_options, *m_cursor_up->GetExecutionContextRef().GetThreadSP())) {

  if (m_options.id)
    m_cursor_up->GoToId(*m_options.id);
  else if (m_options.forwards)
    m_cursor_up->Seek(0, TraceCursor::SeekType::Beginning);
  else
    m_cursor_up->Seek(0, TraceCursor::SeekType::End);

  m_cursor_up->SetForwards(m_options.forwards);
  if (m_options.skip) {
    m_cursor_up->Seek((m_options.forwards ? 1 : -1) * *m_options.skip,
                      TraceCursor::SeekType::Current);
  }
}

TraceDumper::TraceItem TraceDumper::CreatRawTraceItem() {
  TraceItem item = {};
  item.id = m_cursor_up->GetId();

  if (m_options.show_tsc)
    item.tsc = m_cursor_up->GetCounter(lldb::eTraceCounterTSC);
  return item;
}

/// Find the symbol context for the given address reusing the previous
/// instruction's symbol context when possible.
static SymbolContext
CalculateSymbolContext(const Address &address,
                       const TraceDumper::SymbolInfo &prev_symbol_info) {
  AddressRange range;
  if (prev_symbol_info.sc.GetAddressRange(eSymbolContextEverything, 0,
                                          /*inline_block_range*/ false,
                                          range) &&
      range.Contains(address))
    return prev_symbol_info.sc;

  SymbolContext sc;
  address.CalculateSymbolContext(&sc, eSymbolContextEverything);
  return sc;
}

/// Find the disassembler for the given address reusing the previous
/// instruction's disassembler when possible.
static std::tuple<DisassemblerSP, InstructionSP>
CalculateDisass(const TraceDumper::SymbolInfo &symbol_info,
                const TraceDumper::SymbolInfo &prev_symbol_info,
                const ExecutionContext &exe_ctx) {
  if (prev_symbol_info.disassembler) {
    if (InstructionSP instruction =
            prev_symbol_info.disassembler->GetInstructionList()
                .GetInstructionAtAddress(symbol_info.address))
      return std::make_tuple(prev_symbol_info.disassembler, instruction);
  }

  if (symbol_info.sc.function) {
    if (DisassemblerSP disassembler =
            symbol_info.sc.function->GetInstructions(exe_ctx, nullptr)) {
      if (InstructionSP instruction =
              disassembler->GetInstructionList().GetInstructionAtAddress(
                  symbol_info.address))
        return std::make_tuple(disassembler, instruction);
    }
  }
  // We fallback to a single instruction disassembler
  Target &target = exe_ctx.GetTargetRef();
  const ArchSpec arch = target.GetArchitecture();
  AddressRange range(symbol_info.address, arch.GetMaximumOpcodeByteSize());
  DisassemblerSP disassembler =
      Disassembler::DisassembleRange(arch, /*plugin_name*/ nullptr,
                                     /*flavor*/ nullptr, target, range);
  return std::make_tuple(
      disassembler,
      disassembler ? disassembler->GetInstructionList().GetInstructionAtAddress(
                         symbol_info.address)
                   : InstructionSP());
}

Optional<lldb::user_id_t> TraceDumper::DumpInstructions(size_t count) {
  ThreadSP thread_sp = m_cursor_up->GetExecutionContextRef().GetThreadSP();

  SymbolInfo prev_symbol_info;
  Optional<lldb::user_id_t> last_id;

  ExecutionContext exe_ctx;
  thread_sp->GetProcess()->GetTarget().CalculateExecutionContext(exe_ctx);

  for (size_t insn_seen = 0; insn_seen < count && m_cursor_up->HasValue();
       m_cursor_up->Next()) {

    last_id = m_cursor_up->GetId();
    TraceItem item = CreatRawTraceItem();

    if (m_cursor_up->IsEvent()) {
      if (!m_options.show_events)
        continue;
      item.event = m_cursor_up->GetEventType();
      if (*item.event == eTraceEventCPUChanged)
        item.cpu_id = m_cursor_up->GetCPU();
    } else if (m_cursor_up->IsError()) {
      item.error = m_cursor_up->GetError();
    } else {
      insn_seen++;
      item.load_address = m_cursor_up->GetLoadAddress();

      if (!m_options.raw) {
        SymbolInfo symbol_info;
        symbol_info.exe_ctx = exe_ctx;
        symbol_info.address.SetLoadAddress(item.load_address,
                                           exe_ctx.GetTargetPtr());
        symbol_info.sc =
            CalculateSymbolContext(symbol_info.address, prev_symbol_info);
        std::tie(symbol_info.disassembler, symbol_info.instruction) =
            CalculateDisass(symbol_info, prev_symbol_info, exe_ctx);
        item.prev_symbol_info = prev_symbol_info;
        item.symbol_info = symbol_info;
        prev_symbol_info = symbol_info;
      }
    }
    m_writer_up->TraceItem(item);
  }
  if (!m_cursor_up->HasValue())
    m_writer_up->NoMoreData();
  return last_id;
}
