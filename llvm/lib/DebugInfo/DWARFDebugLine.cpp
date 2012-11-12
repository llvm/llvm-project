//===-- DWARFDebugLine.cpp ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFDebugLine.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
using namespace llvm;
using namespace dwarf;

void DWARFDebugLine::Prologue::dump(raw_ostream &OS) const {
  OS << "Line table prologue:\n"
     << format("   total_length: 0x%8.8x\n", TotalLength)
     << format("        version: %u\n", Version)
     << format("prologue_length: 0x%8.8x\n", PrologueLength)
     << format("min_inst_length: %u\n", MinInstLength)
     << format("default_is_stmt: %u\n", DefaultIsStmt)
     << format("      line_base: %i\n", LineBase)
     << format("     line_range: %u\n", LineRange)
     << format("    opcode_base: %u\n", OpcodeBase);

  for (uint32_t i = 0; i < StandardOpcodeLengths.size(); ++i)
    OS << format("standard_opcode_lengths[%s] = %u\n", LNStandardString(i+1),
                 StandardOpcodeLengths[i]);

  if (!IncludeDirectories.empty())
    for (uint32_t i = 0; i < IncludeDirectories.size(); ++i)
      OS << format("include_directories[%3u] = '", i+1)
         << IncludeDirectories[i] << "'\n";

  if (!FileNames.empty()) {
    OS << "                Dir  Mod Time   File Len   File Name\n"
       << "                ---- ---------- ---------- -----------"
          "----------------\n";
    for (uint32_t i = 0; i < FileNames.size(); ++i) {
      const FileNameEntry& fileEntry = FileNames[i];
      OS << format("file_names[%3u] %4" PRIu64 " ", i+1, fileEntry.DirIdx)
         << format("0x%8.8" PRIx64 " 0x%8.8" PRIx64 " ",
                   fileEntry.ModTime, fileEntry.Length)
         << fileEntry.Name << '\n';
    }
  }
}

void DWARFDebugLine::Row::postAppend() {
  BasicBlock = false;
  PrologueEnd = false;
  EpilogueBegin = false;
}

void DWARFDebugLine::Row::reset(bool default_is_stmt) {
  Address = 0;
  Line = 1;
  Column = 0;
  File = 1;
  Isa = 0;
  IsStmt = default_is_stmt;
  BasicBlock = false;
  EndSequence = false;
  PrologueEnd = false;
  EpilogueBegin = false;
}

void DWARFDebugLine::Row::dump(raw_ostream &OS) const {
  OS << format("0x%16.16" PRIx64 " %6u %6u", Address, Line, Column)
     << format(" %6u %3u ", File, Isa)
     << (IsStmt ? " is_stmt" : "")
     << (BasicBlock ? " basic_block" : "")
     << (PrologueEnd ? " prologue_end" : "")
     << (EpilogueBegin ? " epilogue_begin" : "")
     << (EndSequence ? " end_sequence" : "")
     << '\n';
}

void DWARFDebugLine::LineTable::dump(raw_ostream &OS) const {
  Prologue.dump(OS);
  OS << '\n';

  if (!Rows.empty()) {
    OS << "Address            Line   Column File   ISA Flags\n"
       << "------------------ ------ ------ ------ --- -------------\n";
    for (std::vector<Row>::const_iterator pos = Rows.begin(),
         end = Rows.end(); pos != end; ++pos)
      pos->dump(OS);
  }
}

DWARFDebugLine::State::~State() {}

void DWARFDebugLine::State::appendRowToMatrix(uint32_t offset) {
  if (Sequence::Empty) {
    // Record the beginning of instruction sequence.
    Sequence::Empty = false;
    Sequence::LowPC = Address;
    Sequence::FirstRowIndex = row;
  }
  ++row;  // Increase the row number.
  LineTable::appendRow(*this);
  if (EndSequence) {
    // Record the end of instruction sequence.
    Sequence::HighPC = Address;
    Sequence::LastRowIndex = row;
    if (Sequence::isValid())
      LineTable::appendSequence(*this);
    Sequence::reset();
  }
  Row::postAppend();
}

void DWARFDebugLine::State::finalize() {
  row = DoneParsingLineTable;
  if (!Sequence::Empty) {
    fprintf(stderr, "warning: last sequence in debug line table is not"
                    "terminated!\n");
  }
  // Sort all sequences so that address lookup will work faster.
  if (!Sequences.empty()) {
    std::sort(Sequences.begin(), Sequences.end(), Sequence::orderByLowPC);
    // Note: actually, instruction address ranges of sequences should not
    // overlap (in shared objects and executables). If they do, the address
    // lookup would still work, though, but result would be ambiguous.
    // We don't report warning in this case. For example,
    // sometimes .so compiled from multiple object files contains a few
    // rudimentary sequences for address ranges [0x0, 0xsomething).
  }
}

DWARFDebugLine::DumpingState::~DumpingState() {}

void DWARFDebugLine::DumpingState::finalize() {
  LineTable::dump(OS);
}

const DWARFDebugLine::LineTable *
DWARFDebugLine::getLineTable(uint32_t offset) const {
  LineTableConstIter pos = LineTableMap.find(offset);
  if (pos != LineTableMap.end())
    return &pos->second;
  return 0;
}

const DWARFDebugLine::LineTable *
DWARFDebugLine::getOrParseLineTable(DataExtractor debug_line_data,
                                    uint32_t offset) {
  std::pair<LineTableIter, bool> pos =
    LineTableMap.insert(LineTableMapTy::value_type(offset, LineTable()));
  if (pos.second) {
    // Parse and cache the line table for at this offset.
    State state;
    if (!parseStatementTable(debug_line_data, &offset, state))
      return 0;
    pos.first->second = state;
  }
  return &pos.first->second;
}

bool
DWARFDebugLine::parsePrologue(DataExtractor debug_line_data,
                              uint32_t *offset_ptr, Prologue *prologue) {
  const uint32_t prologue_offset = *offset_ptr;

  prologue->clear();
  prologue->TotalLength = debug_line_data.getU32(offset_ptr);
  prologue->Version = debug_line_data.getU16(offset_ptr);
  if (prologue->Version != 2)
    return false;

  prologue->PrologueLength = debug_line_data.getU32(offset_ptr);
  const uint32_t end_prologue_offset = prologue->PrologueLength + *offset_ptr;
  prologue->MinInstLength = debug_line_data.getU8(offset_ptr);
  prologue->DefaultIsStmt = debug_line_data.getU8(offset_ptr);
  prologue->LineBase = debug_line_data.getU8(offset_ptr);
  prologue->LineRange = debug_line_data.getU8(offset_ptr);
  prologue->OpcodeBase = debug_line_data.getU8(offset_ptr);

  prologue->StandardOpcodeLengths.reserve(prologue->OpcodeBase-1);
  for (uint32_t i = 1; i < prologue->OpcodeBase; ++i) {
    uint8_t op_len = debug_line_data.getU8(offset_ptr);
    prologue->StandardOpcodeLengths.push_back(op_len);
  }

  while (*offset_ptr < end_prologue_offset) {
    const char *s = debug_line_data.getCStr(offset_ptr);
    if (s && s[0])
      prologue->IncludeDirectories.push_back(s);
    else
      break;
  }

  while (*offset_ptr < end_prologue_offset) {
    const char *name = debug_line_data.getCStr(offset_ptr);
    if (name && name[0]) {
      FileNameEntry fileEntry;
      fileEntry.Name = name;
      fileEntry.DirIdx = debug_line_data.getULEB128(offset_ptr);
      fileEntry.ModTime = debug_line_data.getULEB128(offset_ptr);
      fileEntry.Length = debug_line_data.getULEB128(offset_ptr);
      prologue->FileNames.push_back(fileEntry);
    } else {
      break;
    }
  }

  if (*offset_ptr != end_prologue_offset) {
    fprintf(stderr, "warning: parsing line table prologue at 0x%8.8x should"
                    " have ended at 0x%8.8x but it ended ad 0x%8.8x\n",
            prologue_offset, end_prologue_offset, *offset_ptr);
    return false;
  }
  return true;
}

bool
DWARFDebugLine::parseStatementTable(DataExtractor debug_line_data,
                                    uint32_t *offset_ptr, State &state) {
  const uint32_t debug_line_offset = *offset_ptr;

  Prologue *prologue = &state.Prologue;

  if (!parsePrologue(debug_line_data, offset_ptr, prologue)) {
    // Restore our offset and return false to indicate failure!
    *offset_ptr = debug_line_offset;
    return false;
  }

  const uint32_t end_offset = debug_line_offset + prologue->TotalLength +
                              sizeof(prologue->TotalLength);

  state.reset();

  while (*offset_ptr < end_offset) {
    uint8_t opcode = debug_line_data.getU8(offset_ptr);

    if (opcode == 0) {
      // Extended Opcodes always start with a zero opcode followed by
      // a uleb128 length so you can skip ones you don't know about
      uint32_t ext_offset = *offset_ptr;
      uint64_t len = debug_line_data.getULEB128(offset_ptr);
      uint32_t arg_size = len - (*offset_ptr - ext_offset);

      uint8_t sub_opcode = debug_line_data.getU8(offset_ptr);
      switch (sub_opcode) {
      case DW_LNE_end_sequence:
        // Set the end_sequence register of the state machine to true and
        // append a row to the matrix using the current values of the
        // state-machine registers. Then reset the registers to the initial
        // values specified above. Every statement program sequence must end
        // with a DW_LNE_end_sequence instruction which creates a row whose
        // address is that of the byte after the last target machine instruction
        // of the sequence.
        state.EndSequence = true;
        state.appendRowToMatrix(*offset_ptr);
        state.reset();
        break;

      case DW_LNE_set_address:
        // Takes a single relocatable address as an operand. The size of the
        // operand is the size appropriate to hold an address on the target
        // machine. Set the address register to the value given by the
        // relocatable address. All of the other statement program opcodes
        // that affect the address register add a delta to it. This instruction
        // stores a relocatable value into it instead.
        state.Address = debug_line_data.getAddress(offset_ptr);
        break;

      case DW_LNE_define_file:
        // Takes 4 arguments. The first is a null terminated string containing
        // a source file name. The second is an unsigned LEB128 number
        // representing the directory index of the directory in which the file
        // was found. The third is an unsigned LEB128 number representing the
        // time of last modification of the file. The fourth is an unsigned
        // LEB128 number representing the length in bytes of the file. The time
        // and length fields may contain LEB128(0) if the information is not
        // available.
        //
        // The directory index represents an entry in the include_directories
        // section of the statement program prologue. The index is LEB128(0)
        // if the file was found in the current directory of the compilation,
        // LEB128(1) if it was found in the first directory in the
        // include_directories section, and so on. The directory index is
        // ignored for file names that represent full path names.
        //
        // The files are numbered, starting at 1, in the order in which they
        // appear; the names in the prologue come before names defined by
        // the DW_LNE_define_file instruction. These numbers are used in the
        // the file register of the state machine.
        {
          FileNameEntry fileEntry;
          fileEntry.Name = debug_line_data.getCStr(offset_ptr);
          fileEntry.DirIdx = debug_line_data.getULEB128(offset_ptr);
          fileEntry.ModTime = debug_line_data.getULEB128(offset_ptr);
          fileEntry.Length = debug_line_data.getULEB128(offset_ptr);
          prologue->FileNames.push_back(fileEntry);
        }
        break;

      default:
        // Length doesn't include the zero opcode byte or the length itself, but
        // it does include the sub_opcode, so we have to adjust for that below
        (*offset_ptr) += arg_size;
        break;
      }
    } else if (opcode < prologue->OpcodeBase) {
      switch (opcode) {
      // Standard Opcodes
      case DW_LNS_copy:
        // Takes no arguments. Append a row to the matrix using the
        // current values of the state-machine registers. Then set
        // the basic_block register to false.
        state.appendRowToMatrix(*offset_ptr);
        break;

      case DW_LNS_advance_pc:
        // Takes a single unsigned LEB128 operand, multiplies it by the
        // min_inst_length field of the prologue, and adds the
        // result to the address register of the state machine.
        state.Address += debug_line_data.getULEB128(offset_ptr) *
                         prologue->MinInstLength;
        break;

      case DW_LNS_advance_line:
        // Takes a single signed LEB128 operand and adds that value to
        // the line register of the state machine.
        state.Line += debug_line_data.getSLEB128(offset_ptr);
        break;

      case DW_LNS_set_file:
        // Takes a single unsigned LEB128 operand and stores it in the file
        // register of the state machine.
        state.File = debug_line_data.getULEB128(offset_ptr);
        break;

      case DW_LNS_set_column:
        // Takes a single unsigned LEB128 operand and stores it in the
        // column register of the state machine.
        state.Column = debug_line_data.getULEB128(offset_ptr);
        break;

      case DW_LNS_negate_stmt:
        // Takes no arguments. Set the is_stmt register of the state
        // machine to the logical negation of its current value.
        state.IsStmt = !state.IsStmt;
        break;

      case DW_LNS_set_basic_block:
        // Takes no arguments. Set the basic_block register of the
        // state machine to true
        state.BasicBlock = true;
        break;

      case DW_LNS_const_add_pc:
        // Takes no arguments. Add to the address register of the state
        // machine the address increment value corresponding to special
        // opcode 255. The motivation for DW_LNS_const_add_pc is this:
        // when the statement program needs to advance the address by a
        // small amount, it can use a single special opcode, which occupies
        // a single byte. When it needs to advance the address by up to
        // twice the range of the last special opcode, it can use
        // DW_LNS_const_add_pc followed by a special opcode, for a total
        // of two bytes. Only if it needs to advance the address by more
        // than twice that range will it need to use both DW_LNS_advance_pc
        // and a special opcode, requiring three or more bytes.
        {
          uint8_t adjust_opcode = 255 - prologue->OpcodeBase;
          uint64_t addr_offset = (adjust_opcode / prologue->LineRange) *
                                 prologue->MinInstLength;
          state.Address += addr_offset;
        }
        break;

      case DW_LNS_fixed_advance_pc:
        // Takes a single uhalf operand. Add to the address register of
        // the state machine the value of the (unencoded) operand. This
        // is the only extended opcode that takes an argument that is not
        // a variable length number. The motivation for DW_LNS_fixed_advance_pc
        // is this: existing assemblers cannot emit DW_LNS_advance_pc or
        // special opcodes because they cannot encode LEB128 numbers or
        // judge when the computation of a special opcode overflows and
        // requires the use of DW_LNS_advance_pc. Such assemblers, however,
        // can use DW_LNS_fixed_advance_pc instead, sacrificing compression.
        state.Address += debug_line_data.getU16(offset_ptr);
        break;

      case DW_LNS_set_prologue_end:
        // Takes no arguments. Set the prologue_end register of the
        // state machine to true
        state.PrologueEnd = true;
        break;

      case DW_LNS_set_epilogue_begin:
        // Takes no arguments. Set the basic_block register of the
        // state machine to true
        state.EpilogueBegin = true;
        break;

      case DW_LNS_set_isa:
        // Takes a single unsigned LEB128 operand and stores it in the
        // column register of the state machine.
        state.Isa = debug_line_data.getULEB128(offset_ptr);
        break;

      default:
        // Handle any unknown standard opcodes here. We know the lengths
        // of such opcodes because they are specified in the prologue
        // as a multiple of LEB128 operands for each opcode.
        {
          assert(opcode - 1U < prologue->StandardOpcodeLengths.size());
          uint8_t opcode_length = prologue->StandardOpcodeLengths[opcode - 1];
          for (uint8_t i=0; i<opcode_length; ++i)
            debug_line_data.getULEB128(offset_ptr);
        }
        break;
      }
    } else {
      // Special Opcodes

      // A special opcode value is chosen based on the amount that needs
      // to be added to the line and address registers. The maximum line
      // increment for a special opcode is the value of the line_base
      // field in the header, plus the value of the line_range field,
      // minus 1 (line base + line range - 1). If the desired line
      // increment is greater than the maximum line increment, a standard
      // opcode must be used instead of a special opcode. The "address
      // advance" is calculated by dividing the desired address increment
      // by the minimum_instruction_length field from the header. The
      // special opcode is then calculated using the following formula:
      //
      //  opcode = (desired line increment - line_base) +
      //           (line_range * address advance) + opcode_base
      //
      // If the resulting opcode is greater than 255, a standard opcode
      // must be used instead.
      //
      // To decode a special opcode, subtract the opcode_base from the
      // opcode itself to give the adjusted opcode. The amount to
      // increment the address register is the result of the adjusted
      // opcode divided by the line_range multiplied by the
      // minimum_instruction_length field from the header. That is:
      //
      //  address increment = (adjusted opcode / line_range) *
      //                      minimum_instruction_length
      //
      // The amount to increment the line register is the line_base plus
      // the result of the adjusted opcode modulo the line_range. That is:
      //
      // line increment = line_base + (adjusted opcode % line_range)

      uint8_t adjust_opcode = opcode - prologue->OpcodeBase;
      uint64_t addr_offset = (adjust_opcode / prologue->LineRange) *
                             prologue->MinInstLength;
      int32_t line_offset = prologue->LineBase +
                            (adjust_opcode % prologue->LineRange);
      state.Line += line_offset;
      state.Address += addr_offset;
      state.appendRowToMatrix(*offset_ptr);
    }
  }

  state.finalize();

  return end_offset;
}

uint32_t
DWARFDebugLine::LineTable::lookupAddress(uint64_t address) const {
  uint32_t unknown_index = UINT32_MAX;
  if (Sequences.empty())
    return unknown_index;
  // First, find an instruction sequence containing the given address.
  DWARFDebugLine::Sequence sequence;
  sequence.LowPC = address;
  SequenceIter first_seq = Sequences.begin();
  SequenceIter last_seq = Sequences.end();
  SequenceIter seq_pos = std::lower_bound(first_seq, last_seq, sequence,
      DWARFDebugLine::Sequence::orderByLowPC);
  DWARFDebugLine::Sequence found_seq;
  if (seq_pos == last_seq) {
    found_seq = Sequences.back();
  } else if (seq_pos->LowPC == address) {
    found_seq = *seq_pos;
  } else {
    if (seq_pos == first_seq)
      return unknown_index;
    found_seq = *(seq_pos - 1);
  }
  if (!found_seq.containsPC(address))
    return unknown_index;
  // Search for instruction address in the rows describing the sequence.
  // Rows are stored in a vector, so we may use arithmetical operations with
  // iterators.
  DWARFDebugLine::Row row;
  row.Address = address;
  RowIter first_row = Rows.begin() + found_seq.FirstRowIndex;
  RowIter last_row = Rows.begin() + found_seq.LastRowIndex;
  RowIter row_pos = std::lower_bound(first_row, last_row, row,
      DWARFDebugLine::Row::orderByAddress);
  if (row_pos == last_row) {
    return found_seq.LastRowIndex - 1;
  }
  uint32_t index = found_seq.FirstRowIndex + (row_pos - first_row);
  if (row_pos->Address > address) {
    if (row_pos == first_row)
      return unknown_index;
    else
      index--;
  }
  return index;
}

bool
DWARFDebugLine::LineTable::getFileNameByIndex(uint64_t FileIndex,
                                              bool NeedsAbsoluteFilePath,
                                              std::string &Result) const {
  if (FileIndex == 0 || FileIndex > Prologue.FileNames.size())
    return false;
  const FileNameEntry &Entry = Prologue.FileNames[FileIndex - 1];
  const char *FileName = Entry.Name;
  if (!NeedsAbsoluteFilePath ||
      sys::path::is_absolute(FileName)) {
    Result = FileName;
    return true;
  }
  SmallString<16> FilePath;
  uint64_t IncludeDirIndex = Entry.DirIdx;
  // Be defensive about the contents of Entry.
  if (IncludeDirIndex > 0 &&
      IncludeDirIndex <= Prologue.IncludeDirectories.size()) {
    const char *IncludeDir = Prologue.IncludeDirectories[IncludeDirIndex - 1];
    sys::path::append(FilePath, IncludeDir);
  }
  sys::path::append(FilePath, FileName);
  Result = FilePath.str();
  return true;
}
