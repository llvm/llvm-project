//===-- TraceInstructionDumper.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/TraceInstructionDumper.h"

#include "lldb/Core/Module.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/SectionLoadList.h"

using namespace lldb;
using namespace lldb_private;
using namespace llvm;

TraceInstructionDumper::TraceInstructionDumper(
    lldb::TraceCursorUP &&cursor_up, Stream &s,
    const TraceInstructionDumperOptions &options)
    : m_cursor_up(std::move(cursor_up)), m_options(options), m_s(s) {
  // We first set the cursor in its initial position
  if (m_options.id) {
    if (!m_cursor_up->GoToId(*m_options.id)) {
      s.PutCString("    invalid instruction id\n");
      SetNoMoreData();
      return;
    }
  } else if (m_options.forwards) {
    m_cursor_up->Seek(0, TraceCursor::SeekType::Beginning);
  } else {
    m_cursor_up->Seek(0, TraceCursor::SeekType::End);
  }

  m_cursor_up->SetForwards(m_options.forwards);
  if (m_options.skip) {
    uint64_t to_skip = m_options.skip.getValue();
    if (m_cursor_up->Seek((m_options.forwards ? 1 : -1) * to_skip,
                          TraceCursor::SeekType::Current) < to_skip) {
      // This happens when the skip value was more than the number of
      // available instructions.
      SetNoMoreData();
    }
  }
}

bool TraceInstructionDumper::TryMoveOneStep() {
  if (!m_cursor_up->Next()) {
    SetNoMoreData();
    return false;
  }
  return true;
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

void TraceInstructionDumper::DumpInstructionSymbolContext(
    const Optional<InstructionSymbolInfo> &prev_insn,
    const InstructionSymbolInfo &insn) {
  if (prev_insn && IsSameInstructionSymbolContext(*prev_insn, insn))
    return;

  m_s << "  ";

  if (!insn.sc.module_sp)
    m_s << "(none)";
  else if (!insn.sc.function && !insn.sc.symbol)
    m_s.Format("{0}`(none)",
               insn.sc.module_sp->GetFileSpec().GetFilename().AsCString());
  else
    insn.sc.DumpStopContext(&m_s, insn.exe_ctx.GetTargetPtr(), insn.address,
                            /*show_fullpaths=*/false,
                            /*show_module=*/true, /*show_inlined_frames=*/false,
                            /*show_function_arguments=*/true,
                            /*show_function_name=*/true);
  m_s << "\n";
}

void TraceInstructionDumper::DumpInstructionDisassembly(
    const InstructionSymbolInfo &insn) {
  if (!insn.instruction)
    return;
  m_s << "    ";
  insn.instruction->Dump(&m_s, /*max_opcode_byte_size=*/0,
                         /*show_address=*/false,
                         /*show_bytes=*/false, &insn.exe_ctx, &insn.sc,
                         /*prev_sym_ctx=*/nullptr,
                         /*disassembly_addr_format=*/nullptr,
                         /*max_address_text_size=*/0);
}

void TraceInstructionDumper::SetNoMoreData() { m_no_more_data = true; }

bool TraceInstructionDumper::HasMoreData() { return !m_no_more_data; }

void TraceInstructionDumper::PrintMissingInstructionsMessage() {
  m_s << "    ...missing instructions\n";
}

void TraceInstructionDumper::PrintInstructionHeader() {
  m_s.Format("    {0}: ", m_cursor_up->GetId());

  if (m_options.show_tsc) {
    m_s << "[tsc=";

    if (Optional<uint64_t> timestamp =
            m_cursor_up->GetCounter(lldb::eTraceCounterTSC))
      m_s.Format("{0:x+16}", *timestamp);
    else
      m_s << "unavailable";

    m_s << "] ";
  }
}

void TraceInstructionDumper::PrintEvents() {
  if (!m_options.show_events)
    return;

  trace_event_utils::ForEachEvent(
      m_cursor_up->GetEvents(), [&](TraceEvents event) {
        m_s.Format("  [{0}]\n", trace_event_utils::EventToDisplayString(event));
      });
}

/// Find the symbol context for the given address reusing the previous
/// instruction's symbol context when possible.
static SymbolContext
CalculateSymbolContext(const Address &address,
                       const InstructionSymbolInfo &prev_insn_info) {
  AddressRange range;
  if (prev_insn_info.sc.GetAddressRange(eSymbolContextEverything, 0,
                                        /*inline_block_range*/ false, range) &&
      range.Contains(address))
    return prev_insn_info.sc;

  SymbolContext sc;
  address.CalculateSymbolContext(&sc, eSymbolContextEverything);
  return sc;
}

/// Find the disassembler for the given address reusing the previous
/// instruction's disassembler when possible.
static std::tuple<DisassemblerSP, InstructionSP>
CalculateDisass(const InstructionSymbolInfo &insn_info,
                const InstructionSymbolInfo &prev_insn_info,
                const ExecutionContext &exe_ctx) {
  if (prev_insn_info.disassembler) {
    if (InstructionSP instruction =
            prev_insn_info.disassembler->GetInstructionList()
                .GetInstructionAtAddress(insn_info.address))
      return std::make_tuple(prev_insn_info.disassembler, instruction);
  }

  if (insn_info.sc.function) {
    if (DisassemblerSP disassembler =
            insn_info.sc.function->GetInstructions(exe_ctx, nullptr)) {
      if (InstructionSP instruction =
              disassembler->GetInstructionList().GetInstructionAtAddress(
                  insn_info.address))
        return std::make_tuple(disassembler, instruction);
    }
  }
  // We fallback to a single instruction disassembler
  Target &target = exe_ctx.GetTargetRef();
  const ArchSpec arch = target.GetArchitecture();
  AddressRange range(insn_info.address, arch.GetMaximumOpcodeByteSize());
  DisassemblerSP disassembler =
      Disassembler::DisassembleRange(arch, /*plugin_name*/ nullptr,
                                     /*flavor*/ nullptr, target, range);
  return std::make_tuple(
      disassembler,
      disassembler ? disassembler->GetInstructionList().GetInstructionAtAddress(
                         insn_info.address)
                   : InstructionSP());
}

Optional<lldb::user_id_t>
TraceInstructionDumper::DumpInstructions(size_t count) {
  ThreadSP thread_sp = m_cursor_up->GetExecutionContextRef().GetThreadSP();
  if (!thread_sp) {
    m_s << "invalid thread";
    return None;
  }

  bool was_prev_instruction_an_error = false;
  InstructionSymbolInfo prev_insn_info;
  Optional<lldb::user_id_t> last_id;

  ExecutionContext exe_ctx;
  thread_sp->GetProcess()->GetTarget().CalculateExecutionContext(exe_ctx);

  for (size_t i = 0; i < count; i++) {
    if (!HasMoreData()) {
      m_s << "    no more data\n";
      break;
    }
    last_id = m_cursor_up->GetId();
    if (m_options.forwards) {
      // When moving forwards, we first print the event before printing
      // the actual instruction.
      PrintEvents();
    }

    if (const char *err = m_cursor_up->GetError()) {
      if (!m_cursor_up->IsForwards() && !was_prev_instruction_an_error)
        PrintMissingInstructionsMessage();

      was_prev_instruction_an_error = true;

      PrintInstructionHeader();
      m_s << err;
    } else {
      if (m_cursor_up->IsForwards() && was_prev_instruction_an_error)
        PrintMissingInstructionsMessage();

      was_prev_instruction_an_error = false;

      InstructionSymbolInfo insn_info;

      if (!m_options.raw) {
        insn_info.load_address = m_cursor_up->GetLoadAddress();
        insn_info.exe_ctx = exe_ctx;
        insn_info.address.SetLoadAddress(insn_info.load_address,
                                         exe_ctx.GetTargetPtr());
        insn_info.sc =
            CalculateSymbolContext(insn_info.address, prev_insn_info);
        std::tie(insn_info.disassembler, insn_info.instruction) =
            CalculateDisass(insn_info, prev_insn_info, exe_ctx);

        DumpInstructionSymbolContext(prev_insn_info, insn_info);
      }

      PrintInstructionHeader();
      m_s.Format("{0:x+16}", m_cursor_up->GetLoadAddress());

      if (!m_options.raw)
        DumpInstructionDisassembly(insn_info);

      prev_insn_info = insn_info;
    }

    m_s << "\n";

    if (!m_options.forwards) {
      // If we move backwards, we print the events after printing
      // the actual instruction so that reading chronologically
      // makes sense.
      PrintEvents();
    }
    TryMoveOneStep();
  }
  return last_id;
}
