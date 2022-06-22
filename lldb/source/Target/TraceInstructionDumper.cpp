//===-- TraceInstructionDumper.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/TraceInstructionDumper.h"

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
static const char *
GetModuleName(const TraceInstructionDumper::InstructionEntry &insn) {
  if (!insn.symbol_info || !insn.symbol_info->sc.module_sp)
    return nullptr;
  return insn.symbol_info->sc.module_sp->GetFileSpec()
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
static bool IsSameInstructionSymbolContext(
    const TraceInstructionDumper::SymbolInfo &prev_insn,
    const TraceInstructionDumper::SymbolInfo &insn) {
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

class OutputWriterCLI : public TraceInstructionDumper::OutputWriter {
public:
  OutputWriterCLI(Stream &s, const TraceInstructionDumperOptions &options)
      : m_s(s), m_options(options){};

  void InfoMessage(StringRef text) override { m_s << "    " << text << "\n"; }

  void Event(StringRef text) override { m_s.Format("  [{0}]\n", text); }

  void
  Instruction(const TraceInstructionDumper::InstructionEntry &insn) override {
    if (insn.symbol_info) {
      if (!insn.prev_symbol_info ||
          !IsSameInstructionSymbolContext(*insn.prev_symbol_info,
                                          *insn.symbol_info)) {
        m_s << "  ";
        const char *module_name = GetModuleName(insn);
        if (!module_name)
          m_s << "(none)";
        else if (!insn.symbol_info->sc.function && !insn.symbol_info->sc.symbol)
          m_s.Format("{0}`(none)", module_name);
        else
          insn.symbol_info->sc.DumpStopContext(
              &m_s, insn.symbol_info->exe_ctx.GetTargetPtr(),
              insn.symbol_info->address,
              /*show_fullpaths=*/false,
              /*show_module=*/true, /*show_inlined_frames=*/false,
              /*show_function_arguments=*/true,
              /*show_function_name=*/true);
        m_s << "\n";
      }
    }

    if (insn.error && !m_was_prev_instruction_an_error)
      InfoMessage("...missing instructions");

    m_s.Format("    {0}: ", insn.id);

    if (m_options.show_tsc) {
      m_s << "[tsc=";

      if (insn.tsc)
        m_s.Format("{0}", *insn.tsc);
      else
        m_s << "unavailable";

      m_s << "] ";
    }

    if (insn.error) {
      m_s << *insn.error;
      m_was_prev_instruction_an_error = true;
    } else {
      m_s.Format("{0:x+16}", insn.load_address);
      if (insn.symbol_info) {
        m_s << "    ";
        insn.symbol_info->instruction->Dump(&m_s, /*max_opcode_byte_size=*/0,
                                            /*show_address=*/false,
                                            /*show_bytes=*/false,
                                            &insn.symbol_info->exe_ctx,
                                            &insn.symbol_info->sc,
                                            /*prev_sym_ctx=*/nullptr,
                                            /*disassembly_addr_format=*/nullptr,
                                            /*max_address_text_size=*/0);
      }
      m_was_prev_instruction_an_error = false;
    }
    m_s << "\n";
  }

private:
  Stream &m_s;
  TraceInstructionDumperOptions m_options;
  bool m_was_prev_instruction_an_error = false;
};

class OutputWriterJSON : public TraceInstructionDumper::OutputWriter {
  /* schema:
    error_message: string
    | {
      "event": string
    } | {
      "id": decimal,
      "tsc"?: string decimal,
      "error": string,
    | {
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
  OutputWriterJSON(Stream &s, const TraceInstructionDumperOptions &options)
      : m_s(s), m_options(options),
        m_j(m_s.AsRawOstream(),
            /*IndentSize=*/options.pretty_print_json ? 2 : 0) {
    m_j.arrayBegin();
  };

  ~OutputWriterJSON() { m_j.arrayEnd(); }

  void Event(StringRef text) override {
    m_j.object([&] { m_j.attribute("event", text); });
  }

  void
  Instruction(const TraceInstructionDumper::InstructionEntry &insn) override {
    m_j.object([&] {
      m_j.attribute("id", insn.id);
      if (m_options.show_tsc)
        m_j.attribute(
            "tsc",
            insn.tsc ? Optional<std::string>(std::to_string(*insn.tsc)) : None);

      if (insn.error) {
        m_j.attribute("error", *insn.error);
        return;
      }

      m_j.attribute("loadAddress", formatv("{0:x}", insn.load_address));
      if (insn.symbol_info) {
        m_j.attribute("module", ToOptionalString(GetModuleName(insn)));
        m_j.attribute("symbol",
                      ToOptionalString(
                          insn.symbol_info->sc.GetFunctionName().AsCString()));
        m_j.attribute(
            "mnemonic",
            ToOptionalString(insn.symbol_info->instruction->GetMnemonic(
                &insn.symbol_info->exe_ctx)));

        if (IsLineEntryValid(insn.symbol_info->sc.line_entry)) {
          m_j.attribute(
              "source",
              ToOptionalString(
                  insn.symbol_info->sc.line_entry.file.GetPath().c_str()));
          m_j.attribute("line", insn.symbol_info->sc.line_entry.line);
          m_j.attribute("column", insn.symbol_info->sc.line_entry.column);
        }
      }
    });
  }

private:
  Stream &m_s;
  TraceInstructionDumperOptions m_options;
  json::OStream m_j;
};

static std::unique_ptr<TraceInstructionDumper::OutputWriter>
CreateWriter(Stream &s, const TraceInstructionDumperOptions &options) {
  if (options.json)
    return std::unique_ptr<TraceInstructionDumper::OutputWriter>(
        new OutputWriterJSON(s, options));
  else
    return std::unique_ptr<TraceInstructionDumper::OutputWriter>(
        new OutputWriterCLI(s, options));
}

TraceInstructionDumper::TraceInstructionDumper(
    lldb::TraceCursorUP &&cursor_up, Stream &s,
    const TraceInstructionDumperOptions &options)
    : m_cursor_up(std::move(cursor_up)), m_options(options),
      m_writer_up(CreateWriter(s, m_options)) {

  if (m_options.id) {
    if (!m_cursor_up->GoToId(*m_options.id)) {
      m_writer_up->InfoMessage("invalid instruction id");
      SetNoMoreData();
    }
  } else if (m_options.forwards) {
    m_cursor_up->Seek(0, TraceCursor::SeekType::Beginning);
  } else {
    m_cursor_up->Seek(0, TraceCursor::SeekType::End);
  }

  m_cursor_up->SetForwards(m_options.forwards);
  if (m_options.skip) {
    uint64_t to_skip = *m_options.skip;
    if (m_cursor_up->Seek((m_options.forwards ? 1 : -1) * to_skip,
                          TraceCursor::SeekType::Current) < to_skip) {
      // This happens when the skip value was more than the number of
      // available instructions.
      SetNoMoreData();
    }
  }
}

void TraceInstructionDumper::SetNoMoreData() { m_no_more_data = true; }

bool TraceInstructionDumper::HasMoreData() { return !m_no_more_data; }

TraceInstructionDumper::InstructionEntry
TraceInstructionDumper::CreatRawInstructionEntry() {
  InstructionEntry insn;
  insn.id = m_cursor_up->GetId();

  if (m_options.show_tsc)
    insn.tsc = m_cursor_up->GetCounter(lldb::eTraceCounterTSC);
  return insn;
}

void TraceInstructionDumper::PrintEvents() {
  if (!m_options.show_events)
    return;

  trace_event_utils::ForEachEvent(
      m_cursor_up->GetEvents(), [&](TraceEvents event) {
        m_writer_up->Event(trace_event_utils::EventToDisplayString(event));
      });
}

/// Find the symbol context for the given address reusing the previous
/// instruction's symbol context when possible.
static SymbolContext CalculateSymbolContext(
    const Address &address,
    const TraceInstructionDumper::SymbolInfo &prev_symbol_info) {
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
CalculateDisass(const TraceInstructionDumper::SymbolInfo &symbol_info,
                const TraceInstructionDumper::SymbolInfo &prev_symbol_info,
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

Optional<lldb::user_id_t>
TraceInstructionDumper::DumpInstructions(size_t count) {
  ThreadSP thread_sp = m_cursor_up->GetExecutionContextRef().GetThreadSP();

  m_writer_up->InfoMessage(formatv("thread #{0}: tid = {1}",
                                   thread_sp->GetIndexID(), thread_sp->GetID())
                               .str());

  SymbolInfo prev_symbol_info;
  Optional<lldb::user_id_t> last_id;

  ExecutionContext exe_ctx;
  thread_sp->GetProcess()->GetTarget().CalculateExecutionContext(exe_ctx);

  for (size_t i = 0; i < count; i++) {
    if (!HasMoreData()) {
      m_writer_up->InfoMessage("no more data");
      break;
    }
    last_id = m_cursor_up->GetId();

    if (m_options.forwards) {
      // When moving forwards, we first print the event before printing
      // the actual instruction.
      PrintEvents();
    }

    InstructionEntry insn = CreatRawInstructionEntry();

    if (const char *err = m_cursor_up->GetError()) {
      insn.error = err;
      m_writer_up->Instruction(insn);
    } else {
      insn.load_address = m_cursor_up->GetLoadAddress();

      if (!m_options.raw) {
        SymbolInfo symbol_info;
        symbol_info.exe_ctx = exe_ctx;
        symbol_info.address.SetLoadAddress(insn.load_address,
                                           exe_ctx.GetTargetPtr());
        symbol_info.sc =
            CalculateSymbolContext(symbol_info.address, prev_symbol_info);
        std::tie(symbol_info.disassembler, symbol_info.instruction) =
            CalculateDisass(symbol_info, prev_symbol_info, exe_ctx);
        insn.prev_symbol_info = prev_symbol_info;
        insn.symbol_info = symbol_info;
        prev_symbol_info = symbol_info;
      }
      m_writer_up->Instruction(insn);
    }

    if (!m_options.forwards) {
      // If we move backwards, we print the events after printing
      // the actual instruction so that reading chronologically
      // makes sense.
      PrintEvents();
    }

    if (!m_cursor_up->Next())
      SetNoMoreData();
  }
  return last_id;
}
