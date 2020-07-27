//===- DWARFEmitter - Convert YAML to DWARF binary data -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// The DWARF component of yaml2obj. Provided as library code for tests.
///
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/DWARFEmitter.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/ObjectYAML/DWARFYAML.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/SwapByteOrder.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace llvm;

template <typename T>
static void writeInteger(T Integer, raw_ostream &OS, bool IsLittleEndian) {
  if (IsLittleEndian != sys::IsLittleEndianHost)
    sys::swapByteOrder(Integer);
  OS.write(reinterpret_cast<char *>(&Integer), sizeof(T));
}

static Error writeVariableSizedInteger(uint64_t Integer, size_t Size,
                                       raw_ostream &OS, bool IsLittleEndian) {
  if (8 == Size)
    writeInteger((uint64_t)Integer, OS, IsLittleEndian);
  else if (4 == Size)
    writeInteger((uint32_t)Integer, OS, IsLittleEndian);
  else if (2 == Size)
    writeInteger((uint16_t)Integer, OS, IsLittleEndian);
  else if (1 == Size)
    writeInteger((uint8_t)Integer, OS, IsLittleEndian);
  else
    return createStringError(errc::not_supported,
                             "invalid integer write size: %zu", Size);

  return Error::success();
}

static void ZeroFillBytes(raw_ostream &OS, size_t Size) {
  std::vector<uint8_t> FillData;
  FillData.insert(FillData.begin(), Size, 0);
  OS.write(reinterpret_cast<char *>(FillData.data()), Size);
}

static void writeInitialLength(const DWARFYAML::InitialLength &Length,
                               raw_ostream &OS, bool IsLittleEndian) {
  writeInteger((uint32_t)Length.TotalLength, OS, IsLittleEndian);
  if (Length.isDWARF64())
    writeInteger((uint64_t)Length.TotalLength64, OS, IsLittleEndian);
}

static void writeInitialLength(const dwarf::DwarfFormat Format,
                               const uint64_t Length, raw_ostream &OS,
                               bool IsLittleEndian) {
  bool IsDWARF64 = Format == dwarf::DWARF64;
  if (IsDWARF64)
    cantFail(writeVariableSizedInteger(dwarf::DW_LENGTH_DWARF64, 4, OS,
                                       IsLittleEndian));
  cantFail(
      writeVariableSizedInteger(Length, IsDWARF64 ? 8 : 4, OS, IsLittleEndian));
}

static void writeDWARFOffset(uint64_t Offset, dwarf::DwarfFormat Format,
                             raw_ostream &OS, bool IsLittleEndian) {
  cantFail(writeVariableSizedInteger(Offset, Format == dwarf::DWARF64 ? 8 : 4,
                                     OS, IsLittleEndian));
}

Error DWARFYAML::emitDebugStr(raw_ostream &OS, const DWARFYAML::Data &DI) {
  for (auto Str : DI.DebugStrings) {
    OS.write(Str.data(), Str.size());
    OS.write('\0');
  }

  return Error::success();
}

Error DWARFYAML::emitDebugAbbrev(raw_ostream &OS, const DWARFYAML::Data &DI) {
  uint64_t AbbrevCode = 0;
  for (auto AbbrevDecl : DI.AbbrevDecls) {
    AbbrevCode = AbbrevDecl.Code ? (uint64_t)*AbbrevDecl.Code : AbbrevCode + 1;
    encodeULEB128(AbbrevCode, OS);
    encodeULEB128(AbbrevDecl.Tag, OS);
    OS.write(AbbrevDecl.Children);
    for (auto Attr : AbbrevDecl.Attributes) {
      encodeULEB128(Attr.Attribute, OS);
      encodeULEB128(Attr.Form, OS);
      if (Attr.Form == dwarf::DW_FORM_implicit_const)
        encodeSLEB128(Attr.Value, OS);
    }
    encodeULEB128(0, OS);
    encodeULEB128(0, OS);
  }

  // The abbreviations for a given compilation unit end with an entry consisting
  // of a 0 byte for the abbreviation code.
  OS.write_zeros(1);

  return Error::success();
}

Error DWARFYAML::emitDebugAranges(raw_ostream &OS, const DWARFYAML::Data &DI) {
  for (auto Range : DI.ARanges) {
    auto HeaderStart = OS.tell();
    writeInitialLength(Range.Format, Range.Length, OS, DI.IsLittleEndian);
    writeInteger((uint16_t)Range.Version, OS, DI.IsLittleEndian);
    writeDWARFOffset(Range.CuOffset, Range.Format, OS, DI.IsLittleEndian);
    writeInteger((uint8_t)Range.AddrSize, OS, DI.IsLittleEndian);
    writeInteger((uint8_t)Range.SegSize, OS, DI.IsLittleEndian);

    auto HeaderSize = OS.tell() - HeaderStart;
    auto FirstDescriptor = alignTo(HeaderSize, Range.AddrSize * 2);
    ZeroFillBytes(OS, FirstDescriptor - HeaderSize);

    for (auto Descriptor : Range.Descriptors) {
      if (Error Err = writeVariableSizedInteger(
              Descriptor.Address, Range.AddrSize, OS, DI.IsLittleEndian))
        return createStringError(errc::not_supported,
                                 "unable to write debug_aranges address: %s",
                                 toString(std::move(Err)).c_str());
      cantFail(writeVariableSizedInteger(Descriptor.Length, Range.AddrSize, OS,
                                         DI.IsLittleEndian));
    }
    ZeroFillBytes(OS, Range.AddrSize * 2);
  }

  return Error::success();
}

Error DWARFYAML::emitDebugRanges(raw_ostream &OS, const DWARFYAML::Data &DI) {
  const size_t RangesOffset = OS.tell();
  uint64_t EntryIndex = 0;
  for (auto DebugRanges : DI.DebugRanges) {
    const size_t CurrOffset = OS.tell() - RangesOffset;
    if (DebugRanges.Offset && (uint64_t)*DebugRanges.Offset < CurrOffset)
      return createStringError(errc::invalid_argument,
                               "'Offset' for 'debug_ranges' with index " +
                                   Twine(EntryIndex) +
                                   " must be greater than or equal to the "
                                   "number of bytes written already (0x" +
                                   Twine::utohexstr(CurrOffset) + ")");
    if (DebugRanges.Offset)
      ZeroFillBytes(OS, *DebugRanges.Offset - CurrOffset);

    uint8_t AddrSize;
    if (DebugRanges.AddrSize)
      AddrSize = *DebugRanges.AddrSize;
    else
      AddrSize = DI.Is64BitAddrSize ? 8 : 4;
    for (auto Entry : DebugRanges.Entries) {
      if (Error Err = writeVariableSizedInteger(Entry.LowOffset, AddrSize, OS,
                                                DI.IsLittleEndian))
        return createStringError(
            errc::not_supported,
            "unable to write debug_ranges address offset: %s",
            toString(std::move(Err)).c_str());
      cantFail(writeVariableSizedInteger(Entry.HighOffset, AddrSize, OS,
                                         DI.IsLittleEndian));
    }
    ZeroFillBytes(OS, AddrSize * 2);
    ++EntryIndex;
  }

  return Error::success();
}

Error DWARFYAML::emitPubSection(raw_ostream &OS,
                                const DWARFYAML::PubSection &Sect,
                                bool IsLittleEndian, bool IsGNUPubSec) {
  writeInitialLength(Sect.Length, OS, IsLittleEndian);
  writeInteger((uint16_t)Sect.Version, OS, IsLittleEndian);
  writeInteger((uint32_t)Sect.UnitOffset, OS, IsLittleEndian);
  writeInteger((uint32_t)Sect.UnitSize, OS, IsLittleEndian);
  for (auto Entry : Sect.Entries) {
    writeInteger((uint32_t)Entry.DieOffset, OS, IsLittleEndian);
    if (IsGNUPubSec)
      writeInteger((uint8_t)Entry.Descriptor, OS, IsLittleEndian);
    OS.write(Entry.Name.data(), Entry.Name.size());
    OS.write('\0');
  }

  return Error::success();
}

static Expected<uint64_t> writeDIE(ArrayRef<DWARFYAML::Abbrev> AbbrevDecls,
                                   const DWARFYAML::Unit &Unit,
                                   const DWARFYAML::Entry &Entry,
                                   raw_ostream &OS, bool IsLittleEndian) {
  uint64_t EntryBegin = OS.tell();
  encodeULEB128(Entry.AbbrCode, OS);
  uint32_t AbbrCode = Entry.AbbrCode;
  if (AbbrCode == 0 || Entry.Values.empty())
    return OS.tell() - EntryBegin;

  if (AbbrCode > AbbrevDecls.size())
    return createStringError(
        errc::invalid_argument,
        "abbrev code must be less than or equal to the number of "
        "entries in abbreviation table");
  const DWARFYAML::Abbrev &Abbrev = AbbrevDecls[AbbrCode - 1];
  auto FormVal = Entry.Values.begin();
  auto AbbrForm = Abbrev.Attributes.begin();
  for (; FormVal != Entry.Values.end() && AbbrForm != Abbrev.Attributes.end();
       ++FormVal, ++AbbrForm) {
    dwarf::Form Form = AbbrForm->Form;
    bool Indirect;
    do {
      Indirect = false;
      switch (Form) {
      case dwarf::DW_FORM_addr:
        // TODO: Test this error.
        if (Error Err = writeVariableSizedInteger(
                FormVal->Value, Unit.FormParams.AddrSize, OS, IsLittleEndian))
          return std::move(Err);
        break;
      case dwarf::DW_FORM_ref_addr:
        // TODO: Test this error.
        if (Error Err = writeVariableSizedInteger(
                FormVal->Value, Unit.FormParams.getRefAddrByteSize(), OS,
                IsLittleEndian))
          return std::move(Err);
        break;
      case dwarf::DW_FORM_exprloc:
      case dwarf::DW_FORM_block:
        encodeULEB128(FormVal->BlockData.size(), OS);
        OS.write((const char *)FormVal->BlockData.data(),
                 FormVal->BlockData.size());
        break;
      case dwarf::DW_FORM_block1: {
        writeInteger((uint8_t)FormVal->BlockData.size(), OS, IsLittleEndian);
        OS.write((const char *)FormVal->BlockData.data(),
                 FormVal->BlockData.size());
        break;
      }
      case dwarf::DW_FORM_block2: {
        writeInteger((uint16_t)FormVal->BlockData.size(), OS, IsLittleEndian);
        OS.write((const char *)FormVal->BlockData.data(),
                 FormVal->BlockData.size());
        break;
      }
      case dwarf::DW_FORM_block4: {
        writeInteger((uint32_t)FormVal->BlockData.size(), OS, IsLittleEndian);
        OS.write((const char *)FormVal->BlockData.data(),
                 FormVal->BlockData.size());
        break;
      }
      case dwarf::DW_FORM_strx:
      case dwarf::DW_FORM_addrx:
      case dwarf::DW_FORM_rnglistx:
      case dwarf::DW_FORM_loclistx:
      case dwarf::DW_FORM_udata:
      case dwarf::DW_FORM_ref_udata:
      case dwarf::DW_FORM_GNU_addr_index:
      case dwarf::DW_FORM_GNU_str_index:
        encodeULEB128(FormVal->Value, OS);
        break;
      case dwarf::DW_FORM_data1:
      case dwarf::DW_FORM_ref1:
      case dwarf::DW_FORM_flag:
      case dwarf::DW_FORM_strx1:
      case dwarf::DW_FORM_addrx1:
        writeInteger((uint8_t)FormVal->Value, OS, IsLittleEndian);
        break;
      case dwarf::DW_FORM_data2:
      case dwarf::DW_FORM_ref2:
      case dwarf::DW_FORM_strx2:
      case dwarf::DW_FORM_addrx2:
        writeInteger((uint16_t)FormVal->Value, OS, IsLittleEndian);
        break;
      case dwarf::DW_FORM_data4:
      case dwarf::DW_FORM_ref4:
      case dwarf::DW_FORM_ref_sup4:
      case dwarf::DW_FORM_strx4:
      case dwarf::DW_FORM_addrx4:
        writeInteger((uint32_t)FormVal->Value, OS, IsLittleEndian);
        break;
      case dwarf::DW_FORM_data8:
      case dwarf::DW_FORM_ref8:
      case dwarf::DW_FORM_ref_sup8:
      case dwarf::DW_FORM_ref_sig8:
        writeInteger((uint64_t)FormVal->Value, OS, IsLittleEndian);
        break;
      case dwarf::DW_FORM_sdata:
        encodeSLEB128(FormVal->Value, OS);
        break;
      case dwarf::DW_FORM_string:
        OS.write(FormVal->CStr.data(), FormVal->CStr.size());
        OS.write('\0');
        break;
      case dwarf::DW_FORM_indirect:
        encodeULEB128(FormVal->Value, OS);
        Indirect = true;
        Form = static_cast<dwarf::Form>((uint64_t)FormVal->Value);
        ++FormVal;
        break;
      case dwarf::DW_FORM_strp:
      case dwarf::DW_FORM_sec_offset:
      case dwarf::DW_FORM_GNU_ref_alt:
      case dwarf::DW_FORM_GNU_strp_alt:
      case dwarf::DW_FORM_line_strp:
      case dwarf::DW_FORM_strp_sup:
        cantFail(writeVariableSizedInteger(
            FormVal->Value, Unit.FormParams.getDwarfOffsetByteSize(), OS,
            IsLittleEndian));
        break;
      default:
        break;
      }
    } while (Indirect);
  }

  return OS.tell() - EntryBegin;
}

Error DWARFYAML::emitDebugInfo(raw_ostream &OS, const DWARFYAML::Data &DI) {
  for (const DWARFYAML::Unit &Unit : DI.CompileUnits) {
    uint64_t Length = 3; // sizeof(version) + sizeof(address_size)
    Length += Unit.FormParams.Version >= 5 ? 1 : 0; // sizeof(unit_type)
    Length +=
        Unit.FormParams.getDwarfOffsetByteSize(); // sizeof(debug_abbrev_offset)

    // Since the length of the current compilation unit is undetermined yet, we
    // firstly write the content of the compilation unit to a buffer to
    // calculate it and then serialize the buffer content to the actual output
    // stream.
    std::string EntryBuffer;
    raw_string_ostream EntryBufferOS(EntryBuffer);

    for (const DWARFYAML::Entry &Entry : Unit.Entries) {
      if (Expected<uint64_t> EntryLength = writeDIE(
              DI.AbbrevDecls, Unit, Entry, EntryBufferOS, DI.IsLittleEndian))
        Length += *EntryLength;
      else
        return EntryLength.takeError();
    }

    // If the length is specified in the YAML description, we use it instead of
    // the actual length.
    if (Unit.Length)
      Length = *Unit.Length;

    writeInitialLength(Unit.FormParams.Format, Length, OS, DI.IsLittleEndian);
    writeInteger((uint16_t)Unit.FormParams.Version, OS, DI.IsLittleEndian);
    if (Unit.FormParams.Version >= 5) {
      writeInteger((uint8_t)Unit.Type, OS, DI.IsLittleEndian);
      writeInteger((uint8_t)Unit.FormParams.AddrSize, OS, DI.IsLittleEndian);
      writeDWARFOffset(Unit.AbbrOffset, Unit.FormParams.Format, OS,
                       DI.IsLittleEndian);
    } else {
      writeDWARFOffset(Unit.AbbrOffset, Unit.FormParams.Format, OS,
                       DI.IsLittleEndian);
      writeInteger((uint8_t)Unit.FormParams.AddrSize, OS, DI.IsLittleEndian);
    }

    OS.write(EntryBuffer.data(), EntryBuffer.size());
  }

  return Error::success();
}

static void emitFileEntry(raw_ostream &OS, const DWARFYAML::File &File) {
  OS.write(File.Name.data(), File.Name.size());
  OS.write('\0');
  encodeULEB128(File.DirIdx, OS);
  encodeULEB128(File.ModTime, OS);
  encodeULEB128(File.Length, OS);
}

Error DWARFYAML::emitDebugLine(raw_ostream &OS, const DWARFYAML::Data &DI) {
  for (const auto &LineTable : DI.DebugLines) {
    writeInitialLength(LineTable.Format, LineTable.Length, OS,
                       DI.IsLittleEndian);
    uint64_t SizeOfPrologueLength = LineTable.Format == dwarf::DWARF64 ? 8 : 4;
    writeInteger((uint16_t)LineTable.Version, OS, DI.IsLittleEndian);
    cantFail(writeVariableSizedInteger(
        LineTable.PrologueLength, SizeOfPrologueLength, OS, DI.IsLittleEndian));
    writeInteger((uint8_t)LineTable.MinInstLength, OS, DI.IsLittleEndian);
    if (LineTable.Version >= 4)
      writeInteger((uint8_t)LineTable.MaxOpsPerInst, OS, DI.IsLittleEndian);
    writeInteger((uint8_t)LineTable.DefaultIsStmt, OS, DI.IsLittleEndian);
    writeInteger((uint8_t)LineTable.LineBase, OS, DI.IsLittleEndian);
    writeInteger((uint8_t)LineTable.LineRange, OS, DI.IsLittleEndian);
    writeInteger((uint8_t)LineTable.OpcodeBase, OS, DI.IsLittleEndian);

    for (auto OpcodeLength : LineTable.StandardOpcodeLengths)
      writeInteger((uint8_t)OpcodeLength, OS, DI.IsLittleEndian);

    for (auto IncludeDir : LineTable.IncludeDirs) {
      OS.write(IncludeDir.data(), IncludeDir.size());
      OS.write('\0');
    }
    OS.write('\0');

    for (auto File : LineTable.Files)
      emitFileEntry(OS, File);
    OS.write('\0');

    for (auto Op : LineTable.Opcodes) {
      writeInteger((uint8_t)Op.Opcode, OS, DI.IsLittleEndian);
      if (Op.Opcode == 0) {
        encodeULEB128(Op.ExtLen, OS);
        writeInteger((uint8_t)Op.SubOpcode, OS, DI.IsLittleEndian);
        switch (Op.SubOpcode) {
        case dwarf::DW_LNE_set_address:
        case dwarf::DW_LNE_set_discriminator:
          // TODO: Test this error.
          if (Error Err = writeVariableSizedInteger(
                  Op.Data, DI.CompileUnits[0].FormParams.AddrSize, OS,
                  DI.IsLittleEndian))
            return Err;
          break;
        case dwarf::DW_LNE_define_file:
          emitFileEntry(OS, Op.FileEntry);
          break;
        case dwarf::DW_LNE_end_sequence:
          break;
        default:
          for (auto OpByte : Op.UnknownOpcodeData)
            writeInteger((uint8_t)OpByte, OS, DI.IsLittleEndian);
        }
      } else if (Op.Opcode < LineTable.OpcodeBase) {
        switch (Op.Opcode) {
        case dwarf::DW_LNS_copy:
        case dwarf::DW_LNS_negate_stmt:
        case dwarf::DW_LNS_set_basic_block:
        case dwarf::DW_LNS_const_add_pc:
        case dwarf::DW_LNS_set_prologue_end:
        case dwarf::DW_LNS_set_epilogue_begin:
          break;

        case dwarf::DW_LNS_advance_pc:
        case dwarf::DW_LNS_set_file:
        case dwarf::DW_LNS_set_column:
        case dwarf::DW_LNS_set_isa:
          encodeULEB128(Op.Data, OS);
          break;

        case dwarf::DW_LNS_advance_line:
          encodeSLEB128(Op.SData, OS);
          break;

        case dwarf::DW_LNS_fixed_advance_pc:
          writeInteger((uint16_t)Op.Data, OS, DI.IsLittleEndian);
          break;

        default:
          for (auto OpData : Op.StandardOpcodeData) {
            encodeULEB128(OpData, OS);
          }
        }
      }
    }
  }

  return Error::success();
}

Error DWARFYAML::emitDebugAddr(raw_ostream &OS, const Data &DI) {
  for (const AddrTableEntry &TableEntry : DI.DebugAddr) {
    uint8_t AddrSize;
    if (TableEntry.AddrSize)
      AddrSize = *TableEntry.AddrSize;
    else
      AddrSize = DI.Is64BitAddrSize ? 8 : 4;

    uint64_t Length;
    if (TableEntry.Length)
      Length = (uint64_t)*TableEntry.Length;
    else
      // 2 (version) + 1 (address_size) + 1 (segment_selector_size) = 4
      Length = 4 + (AddrSize + TableEntry.SegSelectorSize) *
                       TableEntry.SegAddrPairs.size();

    writeInitialLength(TableEntry.Format, Length, OS, DI.IsLittleEndian);
    writeInteger((uint16_t)TableEntry.Version, OS, DI.IsLittleEndian);
    writeInteger((uint8_t)AddrSize, OS, DI.IsLittleEndian);
    writeInteger((uint8_t)TableEntry.SegSelectorSize, OS, DI.IsLittleEndian);

    for (const SegAddrPair &Pair : TableEntry.SegAddrPairs) {
      if (TableEntry.SegSelectorSize != 0)
        if (Error Err = writeVariableSizedInteger(Pair.Segment,
                                                  TableEntry.SegSelectorSize,
                                                  OS, DI.IsLittleEndian))
          return createStringError(errc::not_supported,
                                   "unable to write debug_addr segment: %s",
                                   toString(std::move(Err)).c_str());
      if (AddrSize != 0)
        if (Error Err = writeVariableSizedInteger(Pair.Address, AddrSize, OS,
                                                  DI.IsLittleEndian))
          return createStringError(errc::not_supported,
                                   "unable to write debug_addr address: %s",
                                   toString(std::move(Err)).c_str());
    }
  }

  return Error::success();
}

Error DWARFYAML::emitDebugStrOffsets(raw_ostream &OS, const Data &DI) {
  assert(DI.DebugStrOffsets && "unexpected emitDebugStrOffsets() call");
  for (const DWARFYAML::StringOffsetsTable &Table : *DI.DebugStrOffsets) {
    uint64_t Length;
    if (Table.Length)
      Length = *Table.Length;
    else
      // sizeof(version) + sizeof(padding) = 4
      Length =
          4 + Table.Offsets.size() * (Table.Format == dwarf::DWARF64 ? 8 : 4);

    writeInitialLength(Table.Format, Length, OS, DI.IsLittleEndian);
    writeInteger((uint16_t)Table.Version, OS, DI.IsLittleEndian);
    writeInteger((uint16_t)Table.Padding, OS, DI.IsLittleEndian);

    for (uint64_t Offset : Table.Offsets)
      writeDWARFOffset(Offset, Table.Format, OS, DI.IsLittleEndian);
  }

  return Error::success();
}

static Error checkListEntryOperands(StringRef EncodingString,
                                    ArrayRef<yaml::Hex64> Values,
                                    uint64_t ExpectedOperands) {
  if (Values.size() != ExpectedOperands)
    return createStringError(
        errc::invalid_argument,
        "invalid number (%zu) of operands for the operator: %s, %" PRIu64
        " expected",
        Values.size(), EncodingString.str().c_str(), ExpectedOperands);

  return Error::success();
}

static Error writeListEntryAddress(StringRef EncodingName, raw_ostream &OS,
                                   uint64_t Addr, uint8_t AddrSize,
                                   bool IsLittleEndian) {
  if (Error Err = writeVariableSizedInteger(Addr, AddrSize, OS, IsLittleEndian))
    return createStringError(errc::invalid_argument,
                             "unable to write address for the operator %s: %s",
                             EncodingName.str().c_str(),
                             toString(std::move(Err)).c_str());

  return Error::success();
}

static Expected<uint64_t> writeListEntry(raw_ostream &OS,
                                         const DWARFYAML::RnglistEntry &Entry,
                                         uint8_t AddrSize,
                                         bool IsLittleEndian) {
  uint64_t BeginOffset = OS.tell();
  writeInteger((uint8_t)Entry.Operator, OS, IsLittleEndian);

  StringRef EncodingName = dwarf::RangeListEncodingString(Entry.Operator);

  auto CheckOperands = [&](uint64_t ExpectedOperands) -> Error {
    return checkListEntryOperands(EncodingName, Entry.Values, ExpectedOperands);
  };

  auto WriteAddress = [&](uint64_t Addr) -> Error {
    return writeListEntryAddress(EncodingName, OS, Addr, AddrSize,
                                 IsLittleEndian);
  };

  switch (Entry.Operator) {
  case dwarf::DW_RLE_end_of_list:
    if (Error Err = CheckOperands(0))
      return std::move(Err);
    break;
  case dwarf::DW_RLE_base_addressx:
    if (Error Err = CheckOperands(1))
      return std::move(Err);
    encodeULEB128(Entry.Values[0], OS);
    break;
  case dwarf::DW_RLE_startx_endx:
  case dwarf::DW_RLE_startx_length:
  case dwarf::DW_RLE_offset_pair:
    if (Error Err = CheckOperands(2))
      return std::move(Err);
    encodeULEB128(Entry.Values[0], OS);
    encodeULEB128(Entry.Values[1], OS);
    break;
  case dwarf::DW_RLE_base_address:
    if (Error Err = CheckOperands(1))
      return std::move(Err);
    if (Error Err = WriteAddress(Entry.Values[0]))
      return std::move(Err);
    break;
  case dwarf::DW_RLE_start_end:
    if (Error Err = CheckOperands(2))
      return std::move(Err);
    if (Error Err = WriteAddress(Entry.Values[0]))
      return std::move(Err);
    cantFail(WriteAddress(Entry.Values[1]));
    break;
  case dwarf::DW_RLE_start_length:
    if (Error Err = CheckOperands(2))
      return std::move(Err);
    if (Error Err = WriteAddress(Entry.Values[0]))
      return std::move(Err);
    encodeULEB128(Entry.Values[1], OS);
    break;
  }

  return OS.tell() - BeginOffset;
}

template <typename EntryType>
Error writeDWARFLists(raw_ostream &OS,
                      ArrayRef<DWARFYAML::ListTable<EntryType>> Tables,
                      bool IsLittleEndian, bool Is64BitAddrSize) {
  for (const DWARFYAML::ListTable<EntryType> &Table : Tables) {
    // sizeof(version) + sizeof(address_size) + sizeof(segment_selector_size) +
    // sizeof(offset_entry_count) = 8
    uint64_t Length = 8;

    uint8_t AddrSize;
    if (Table.AddrSize)
      AddrSize = *Table.AddrSize;
    else
      AddrSize = Is64BitAddrSize ? 8 : 4;

    // Since the length of the current range/location lists entry is
    // undetermined yet, we firstly write the content of the range/location
    // lists to a buffer to calculate the length and then serialize the buffer
    // content to the actual output stream.
    std::string ListBuffer;
    raw_string_ostream ListBufferOS(ListBuffer);

    // Offsets holds offsets for each range/location list. The i-th element is
    // the offset from the beginning of the first range/location list to the
    // location of the i-th range list.
    std::vector<uint64_t> Offsets;

    for (const DWARFYAML::ListEntries<EntryType> &List : Table.Lists) {
      Offsets.push_back(ListBufferOS.tell());
      for (const EntryType &Entry : List.Entries) {
        Expected<uint64_t> EntrySize =
            writeListEntry(ListBufferOS, Entry, AddrSize, IsLittleEndian);
        if (!EntrySize)
          return EntrySize.takeError();
        Length += *EntrySize;
      }
    }

    // If the offset_entry_count field isn't specified, yaml2obj will infer it
    // from the 'Offsets' field in the YAML description. If the 'Offsets' field
    // isn't specified either, yaml2obj will infer it from the auto-generated
    // offsets.
    uint32_t OffsetEntryCount;
    if (Table.OffsetEntryCount)
      OffsetEntryCount = *Table.OffsetEntryCount;
    else
      OffsetEntryCount = Table.Offsets ? Table.Offsets->size() : Offsets.size();
    uint64_t OffsetsSize =
        OffsetEntryCount * (Table.Format == dwarf::DWARF64 ? 8 : 4);
    Length += OffsetsSize;

    // If the length is specified in the YAML description, we use it instead of
    // the actual length.
    if (Table.Length)
      Length = *Table.Length;

    writeInitialLength(Table.Format, Length, OS, IsLittleEndian);
    writeInteger((uint16_t)Table.Version, OS, IsLittleEndian);
    writeInteger((uint8_t)AddrSize, OS, IsLittleEndian);
    writeInteger((uint8_t)Table.SegSelectorSize, OS, IsLittleEndian);
    writeInteger((uint32_t)OffsetEntryCount, OS, IsLittleEndian);

    auto EmitOffsets = [&](ArrayRef<uint64_t> Offsets, uint64_t OffsetsSize) {
      for (uint64_t Offset : Offsets)
        writeDWARFOffset(OffsetsSize + Offset, Table.Format, OS,
                         IsLittleEndian);
    };

    if (Table.Offsets)
      EmitOffsets(ArrayRef<uint64_t>((const uint64_t *)Table.Offsets->data(),
                                     Table.Offsets->size()),
                  0);
    else
      EmitOffsets(Offsets, OffsetsSize);

    OS.write(ListBuffer.data(), ListBuffer.size());
  }

  return Error::success();
}

Error DWARFYAML::emitDebugRnglists(raw_ostream &OS, const Data &DI) {
  assert(DI.DebugRnglists && "unexpected emitDebugRnglists() call");
  return writeDWARFLists<DWARFYAML::RnglistEntry>(
      OS, *DI.DebugRnglists, DI.IsLittleEndian, DI.Is64BitAddrSize);
}

using EmitFuncType = Error (*)(raw_ostream &, const DWARFYAML::Data &);

static Error
emitDebugSectionImpl(const DWARFYAML::Data &DI, EmitFuncType EmitFunc,
                     StringRef Sec,
                     StringMap<std::unique_ptr<MemoryBuffer>> &OutputBuffers) {
  std::string Data;
  raw_string_ostream DebugInfoStream(Data);
  if (Error Err = EmitFunc(DebugInfoStream, DI))
    return Err;
  DebugInfoStream.flush();
  if (!Data.empty())
    OutputBuffers[Sec] = MemoryBuffer::getMemBufferCopy(Data);

  return Error::success();
}

Expected<StringMap<std::unique_ptr<MemoryBuffer>>>
DWARFYAML::emitDebugSections(StringRef YAMLString, bool IsLittleEndian) {
  auto CollectDiagnostic = [](const SMDiagnostic &Diag, void *DiagContext) {
    *static_cast<SMDiagnostic *>(DiagContext) = Diag;
  };

  SMDiagnostic GeneratedDiag;
  yaml::Input YIn(YAMLString, /*Ctxt=*/nullptr, CollectDiagnostic,
                  &GeneratedDiag);

  DWARFYAML::Data DI;
  DI.IsLittleEndian = IsLittleEndian;
  YIn >> DI;
  if (YIn.error())
    return createStringError(YIn.error(), GeneratedDiag.getMessage());

  StringMap<std::unique_ptr<MemoryBuffer>> DebugSections;
  Error Err = emitDebugSectionImpl(DI, &DWARFYAML::emitDebugInfo, "debug_info",
                                   DebugSections);
  Err = joinErrors(std::move(Err),
                   emitDebugSectionImpl(DI, &DWARFYAML::emitDebugLine,
                                        "debug_line", DebugSections));
  Err = joinErrors(std::move(Err),
                   emitDebugSectionImpl(DI, &DWARFYAML::emitDebugStr,
                                        "debug_str", DebugSections));
  Err = joinErrors(std::move(Err),
                   emitDebugSectionImpl(DI, &DWARFYAML::emitDebugAbbrev,
                                        "debug_abbrev", DebugSections));
  Err = joinErrors(std::move(Err),
                   emitDebugSectionImpl(DI, &DWARFYAML::emitDebugAranges,
                                        "debug_aranges", DebugSections));
  Err = joinErrors(std::move(Err),
                   emitDebugSectionImpl(DI, &DWARFYAML::emitDebugRanges,
                                        "debug_ranges", DebugSections));

  if (Err)
    return std::move(Err);
  return std::move(DebugSections);
}
