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

TraceInstructionDumper::TraceInstructionDumper(lldb::TraceCursorUP &&cursor_up,
                                               int initial_index, bool raw,
                                               bool show_tsc)
    : m_cursor_up(std::move(cursor_up)), m_index(initial_index), m_raw(raw),
      m_show_tsc(show_tsc) {}

/// \return
///     Return \b true if the cursor could move one step.
bool TraceInstructionDumper::TryMoveOneStep() {
  if (!m_cursor_up->Next()) {
    SetNoMoreData();
    return false;
  }
  m_index += m_cursor_up->IsForwards() ? 1 : -1;
  return true;
}

/// \return
///     The number of characters that would be needed to print the given
///     integer.
static int GetNumberOfChars(int num) {
  if (num == 0)
    return 1;
  return (num < 0 ? 1 : 0) + static_cast<int>(log10(abs(num))) + 1;
}

/// Helper struct that holds symbol, disassembly and address information of an
/// instruction.
struct InstructionSymbolInfo {
  SymbolContext sc;
  Address address;
  lldb::addr_t load_address;
  lldb::DisassemblerSP disassembler;
  lldb::InstructionSP instruction;
  lldb_private::ExecutionContext exe_ctx;
};

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
                            /*show_fullpaths=*/false,
                            /*show_module=*/true, /*show_inlined_frames=*/false,
                            /*show_function_arguments=*/true,
                            /*show_function_name=*/true);
  s.Printf("\n");
}

static void DumpInstructionDisassembly(Stream &s, InstructionSymbolInfo &insn) {
  if (!insn.instruction)
    return;
  s.Printf("    ");
  insn.instruction->Dump(&s, /*max_opcode_byte_size=*/0, /*show_address=*/false,
                         /*show_bytes=*/false, &insn.exe_ctx, &insn.sc,
                         /*prev_sym_ctx=*/nullptr,
                         /*disassembly_addr_format=*/nullptr,
                         /*max_address_text_size=*/0);
}

void TraceInstructionDumper::SetNoMoreData() { m_no_more_data = true; }

bool TraceInstructionDumper::HasMoreData() { return !m_no_more_data; }

void TraceInstructionDumper::DumpInstructions(Stream &s, size_t count) {
  ThreadSP thread_sp = m_cursor_up->GetExecutionContextRef().GetThreadSP();
  if (!thread_sp) {
    s.Printf("invalid thread");
    return;
  }

  s.Printf("thread #%u: tid = %" PRIu64 "\n", thread_sp->GetIndexID(),
           thread_sp->GetID());

  int digits_count = GetNumberOfChars(
      m_cursor_up->IsForwards() ? m_index + count - 1 : m_index - count + 1);
  bool was_prev_instruction_an_error = false;

  auto printMissingInstructionsMessage = [&]() {
    s.Printf("    ...missing instructions\n");
  };

  auto printInstructionIndex = [&]() {
    s.Printf("    [%*d] ", digits_count, m_index);

    if (m_show_tsc) {
      s.Printf("[tsc=");

      if (Optional<uint64_t> timestamp = m_cursor_up->GetCounter(lldb::eTraceCounterTSC))
        s.Printf("0x%016" PRIx64, *timestamp);
      else
        s.Printf("unavailable");

      s.Printf("] ");
    }
  };

  InstructionSymbolInfo prev_insn_info;

  Target &target = thread_sp->GetProcess()->GetTarget();
  ExecutionContext exe_ctx;
  target.CalculateExecutionContext(exe_ctx);
  const ArchSpec &arch = target.GetArchitecture();

  // Find the symbol context for the given address reusing the previous
  // instruction's symbol context when possible.
  auto calculateSymbolContext = [&](const Address &address) {
    AddressRange range;
    if (prev_insn_info.sc.GetAddressRange(eSymbolContextEverything, 0,
                                          /*inline_block_range*/ false,
                                          range) &&
        range.Contains(address))
      return prev_insn_info.sc;

    SymbolContext sc;
    address.CalculateSymbolContext(&sc, eSymbolContextEverything);
    return sc;
  };

  // Find the disassembler for the given address reusing the previous
  // instruction's disassembler when possible.
  auto calculateDisass = [&](const Address &address, const SymbolContext &sc) {
    if (prev_insn_info.disassembler) {
      if (InstructionSP instruction =
              prev_insn_info.disassembler->GetInstructionList()
                  .GetInstructionAtAddress(address))
        return std::make_tuple(prev_insn_info.disassembler, instruction);
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

  for (size_t i = 0; i < count; i++) {
    if (!HasMoreData()) {
      s.Printf("    no more data\n");
      break;
    }

    if (const char *err = m_cursor_up->GetError()) {
      if (!m_cursor_up->IsForwards() && !was_prev_instruction_an_error)
        printMissingInstructionsMessage();

      was_prev_instruction_an_error = true;

      printInstructionIndex();
      s << err;
    } else {
      if (m_cursor_up->IsForwards() && was_prev_instruction_an_error)
        printMissingInstructionsMessage();

      was_prev_instruction_an_error = false;

      InstructionSymbolInfo insn_info;

      if (!m_raw) {
        insn_info.load_address = m_cursor_up->GetLoadAddress();
        insn_info.exe_ctx = exe_ctx;
        insn_info.address.SetLoadAddress(insn_info.load_address, &target);
        insn_info.sc = calculateSymbolContext(insn_info.address);
        std::tie(insn_info.disassembler, insn_info.instruction) =
            calculateDisass(insn_info.address, insn_info.sc);

        DumpInstructionSymbolContext(s, prev_insn_info, insn_info);
      }

      printInstructionIndex();
      s.Printf("0x%016" PRIx64, m_cursor_up->GetLoadAddress());

      if (!m_raw)
        DumpInstructionDisassembly(s, insn_info);

      prev_insn_info = insn_info;
    }

    s.Printf("\n");
    TryMoveOneStep();
  }
}
