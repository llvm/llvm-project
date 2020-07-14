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
#include "DWARFVisitor.h"
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
    if (Range.Format == dwarf::DWARF64)
      writeInteger((uint64_t)Range.CuOffset, OS, DI.IsLittleEndian);
    else
      writeInteger((uint32_t)Range.CuOffset, OS, DI.IsLittleEndian);
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

namespace {
/// An extension of the DWARFYAML::ConstVisitor which writes compile
/// units and DIEs to a stream.
class DumpVisitor : public DWARFYAML::ConstVisitor {
  raw_ostream &OS;

protected:
  void onStartCompileUnit(const DWARFYAML::Unit &CU) override {
    writeInitialLength(CU.Format, CU.Length, OS, DebugInfo.IsLittleEndian);
    writeInteger((uint16_t)CU.Version, OS, DebugInfo.IsLittleEndian);
    if (CU.Version >= 5) {
      writeInteger((uint8_t)CU.Type, OS, DebugInfo.IsLittleEndian);
      writeInteger((uint8_t)CU.AddrSize, OS, DebugInfo.IsLittleEndian);
      cantFail(writeVariableSizedInteger(CU.AbbrOffset,
                                         CU.Format == dwarf::DWARF64 ? 8 : 4,
                                         OS, DebugInfo.IsLittleEndian));
    } else {
      cantFail(writeVariableSizedInteger(CU.AbbrOffset,
                                         CU.Format == dwarf::DWARF64 ? 8 : 4,
                                         OS, DebugInfo.IsLittleEndian));
      writeInteger((uint8_t)CU.AddrSize, OS, DebugInfo.IsLittleEndian);
    }
  }

  void onStartDIE(const DWARFYAML::Unit &CU,
                  const DWARFYAML::Entry &DIE) override {
    encodeULEB128(DIE.AbbrCode, OS);
  }

  void onValue(const uint8_t U) override {
    writeInteger(U, OS, DebugInfo.IsLittleEndian);
  }

  void onValue(const uint16_t U) override {
    writeInteger(U, OS, DebugInfo.IsLittleEndian);
  }

  void onValue(const uint32_t U) override {
    writeInteger(U, OS, DebugInfo.IsLittleEndian);
  }

  void onValue(const uint64_t U, const bool LEB = false) override {
    if (LEB)
      encodeULEB128(U, OS);
    else
      writeInteger(U, OS, DebugInfo.IsLittleEndian);
  }

  void onValue(const int64_t S, const bool LEB = false) override {
    if (LEB)
      encodeSLEB128(S, OS);
    else
      writeInteger(S, OS, DebugInfo.IsLittleEndian);
  }

  void onValue(const StringRef String) override {
    OS.write(String.data(), String.size());
    OS.write('\0');
  }

  void onValue(const MemoryBufferRef MBR) override {
    OS.write(MBR.getBufferStart(), MBR.getBufferSize());
  }

public:
  DumpVisitor(const DWARFYAML::Data &DI, raw_ostream &Out)
      : DWARFYAML::ConstVisitor(DI), OS(Out) {}
};
} // namespace

Error DWARFYAML::emitDebugInfo(raw_ostream &OS, const DWARFYAML::Data &DI) {
  DumpVisitor Visitor(DI, OS);
  return Visitor.traverseDebugInfo();
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
                  Op.Data, DI.CompileUnits[0].AddrSize, OS, DI.IsLittleEndian))
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

namespace {
class DIEFixupVisitor : public DWARFYAML::Visitor {
  uint64_t Length;

public:
  DIEFixupVisitor(DWARFYAML::Data &DI) : DWARFYAML::Visitor(DI){};

protected:
  void onStartCompileUnit(DWARFYAML::Unit &CU) override {
    // Size of the unit header, excluding the length field itself.
    Length = CU.Version >= 5 ? 8 : 7;
  }

  void onEndCompileUnit(DWARFYAML::Unit &CU) override { CU.Length = Length; }

  void onStartDIE(DWARFYAML::Unit &CU, DWARFYAML::Entry &DIE) override {
    Length += getULEB128Size(DIE.AbbrCode);
  }

  void onValue(const uint8_t U) override { Length += 1; }
  void onValue(const uint16_t U) override { Length += 2; }
  void onValue(const uint32_t U) override { Length += 4; }
  void onValue(const uint64_t U, const bool LEB = false) override {
    if (LEB)
      Length += getULEB128Size(U);
    else
      Length += 8;
  }
  void onValue(const int64_t S, const bool LEB = false) override {
    if (LEB)
      Length += getSLEB128Size(S);
    else
      Length += 8;
  }
  void onValue(const StringRef String) override { Length += String.size() + 1; }

  void onValue(const MemoryBufferRef MBR) override {
    Length += MBR.getBufferSize();
  }
};
} // namespace

Expected<StringMap<std::unique_ptr<MemoryBuffer>>>
DWARFYAML::emitDebugSections(StringRef YAMLString, bool ApplyFixups,
                             bool IsLittleEndian) {
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

  if (ApplyFixups) {
    DIEFixupVisitor DIFixer(DI);
    if (Error Err = DIFixer.traverseDebugInfo())
      return std::move(Err);
  }

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
