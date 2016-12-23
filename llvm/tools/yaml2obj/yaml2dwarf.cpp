//===- yaml2dwarf - Convert YAML to DWARF binary data ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief The DWARF component of yaml2obj.
///
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/DWARFYAML.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SwapByteOrder.h"

#include <algorithm>

using namespace llvm;

template <typename T>
void writeInteger(T Integer, raw_ostream &OS, bool IsLittleEndian) {
  if (IsLittleEndian != sys::IsLittleEndianHost)
    sys::swapByteOrder(Integer);
  OS.write(reinterpret_cast<char *>(&Integer), sizeof(T));
}

void writeVariableSizedInteger(uint64_t Integer, size_t Size, raw_ostream &OS,
                               bool IsLittleEndian) {
  if (8 == Size)
    writeInteger((uint64_t)Integer, OS, IsLittleEndian);
  else if (4 == Size)
    writeInteger((uint32_t)Integer, OS, IsLittleEndian);
  else if (2 == Size)
    writeInteger((uint16_t)Integer, OS, IsLittleEndian);
  else if (1 == Size)
    writeInteger((uint8_t)Integer, OS, IsLittleEndian);
  else
    assert(false && "Invalid integer write size.");
}

void ZeroFillBytes(raw_ostream &OS, size_t Size) {
  std::vector<uint8_t> FillData;
  FillData.insert(FillData.begin(), Size, 0);
  OS.write(reinterpret_cast<char *>(FillData.data()), Size);
}

void yaml2debug_str(raw_ostream &OS, const DWARFYAML::Data &DI) {
  for (auto Str : DI.DebugStrings) {
    OS.write(Str.data(), Str.size());
    OS.write('\0');
  }
}

void yaml2debug_abbrev(raw_ostream &OS, const DWARFYAML::Data &DI) {
  for (auto AbbrevDecl : DI.AbbrevDecls) {
    encodeULEB128(AbbrevDecl.Code, OS);
    encodeULEB128(AbbrevDecl.Tag, OS);
    OS.write(AbbrevDecl.Children);
    for (auto Attr : AbbrevDecl.Attributes) {
      encodeULEB128(Attr.Attribute, OS);
      encodeULEB128(Attr.Form, OS);
    }
    encodeULEB128(0, OS);
    encodeULEB128(0, OS);
  }
}

void yaml2debug_aranges(raw_ostream &OS, const DWARFYAML::Data &DI) {
  for (auto Range : DI.ARanges) {
    auto HeaderStart = OS.tell();
    writeInteger((uint32_t)Range.Length, OS, DI.IsLittleEndian);
    writeInteger((uint16_t)Range.Version, OS, DI.IsLittleEndian);
    writeInteger((uint32_t)Range.CuOffset, OS, DI.IsLittleEndian);
    writeInteger((uint8_t)Range.AddrSize, OS, DI.IsLittleEndian);
    writeInteger((uint8_t)Range.SegSize, OS, DI.IsLittleEndian);

    auto HeaderSize = OS.tell() - HeaderStart;
    auto FirstDescriptor = alignTo(HeaderSize, Range.AddrSize * 2);
    ZeroFillBytes(OS, FirstDescriptor - HeaderSize);

    for (auto Descriptor : Range.Descriptors) {
      writeVariableSizedInteger(Descriptor.Address, Range.AddrSize, OS,
                                DI.IsLittleEndian);
      writeVariableSizedInteger(Descriptor.Length, Range.AddrSize, OS,
                                DI.IsLittleEndian);
    }
    ZeroFillBytes(OS, Range.AddrSize * 2);
  }
}

void yaml2pubsection(raw_ostream &OS, const DWARFYAML::PubSection &Sect,
                     bool IsLittleEndian) {
  writeInteger((uint32_t)Sect.Length, OS, IsLittleEndian);
  writeInteger((uint16_t)Sect.Version, OS, IsLittleEndian);
  writeInteger((uint32_t)Sect.UnitOffset, OS, IsLittleEndian);
  writeInteger((uint32_t)Sect.UnitSize, OS, IsLittleEndian);
  for (auto Entry : Sect.Entries) {
    writeInteger((uint32_t)Entry.DieOffset, OS, IsLittleEndian);
    if (Sect.IsGNUStyle)
      writeInteger((uint32_t)Entry.Descriptor, OS, IsLittleEndian);
    OS.write(Entry.Name.data(), Entry.Name.size());
    OS.write('\0');
  }
}

void yaml2debug_info(raw_ostream &OS, const DWARFYAML::Data &DI) {

  for (auto CU : DI.CompileUnits) {
    writeInteger((uint32_t)CU.Length, OS, DI.IsLittleEndian);
    writeInteger((uint16_t)CU.Version, OS, DI.IsLittleEndian);
    writeInteger((uint32_t)CU.AbbrOffset, OS, DI.IsLittleEndian);
    writeInteger((uint8_t)CU.AddrSize, OS, DI.IsLittleEndian);

    auto FirstAbbrevCode = CU.Entries[0].AbbrCode;

    for (auto Entry : CU.Entries) {
      encodeULEB128(Entry.AbbrCode, OS);
      if (Entry.AbbrCode == 0u)
        continue;
      bool Indirect = false;
      assert(Entry.AbbrCode - FirstAbbrevCode < DI.AbbrevDecls.size() &&
             "Out of range AbbCode");
      auto &Abbrev = DI.AbbrevDecls[Entry.AbbrCode - FirstAbbrevCode];

      auto FormVal = Entry.Values.begin();
      auto AbbrForm = Abbrev.Attributes.begin();
      for (;
           FormVal != Entry.Values.end() && AbbrForm != Abbrev.Attributes.end();
           ++FormVal, ++AbbrForm) {
        dwarf::Form Form = AbbrForm->Form;
        do {
          Indirect = false;
          switch (Form) {
          case dwarf::DW_FORM_addr:
            writeVariableSizedInteger(FormVal->Value, CU.AddrSize, OS,
                                      DI.IsLittleEndian);
            break;
          case dwarf::DW_FORM_ref_addr: {
            // TODO: Handle DWARF32/DWARF64 after Line Table data is done
            auto writeSize = CU.Version == 2 ? CU.AddrSize : 4;
            writeVariableSizedInteger(FormVal->Value, writeSize, OS,
                                      DI.IsLittleEndian);
            break;
          }
          case dwarf::DW_FORM_exprloc:
          case dwarf::DW_FORM_block:
            encodeULEB128(FormVal->BlockData.size(), OS);
            OS.write(reinterpret_cast<char *>(&FormVal->BlockData[0]),
                     FormVal->BlockData.size());
            break;
          case dwarf::DW_FORM_block1: {
            auto writeSize = FormVal->BlockData.size();
            writeInteger((uint8_t)writeSize, OS, DI.IsLittleEndian);
            OS.write(reinterpret_cast<char *>(&FormVal->BlockData[0]),
                     FormVal->BlockData.size());
            break;
          }
          case dwarf::DW_FORM_block2: {
            auto writeSize = FormVal->BlockData.size();
            writeInteger((uint16_t)writeSize, OS, DI.IsLittleEndian);
            OS.write(reinterpret_cast<char *>(&FormVal->BlockData[0]),
                     FormVal->BlockData.size());
            break;
          }
          case dwarf::DW_FORM_block4: {
            auto writeSize = FormVal->BlockData.size();
            writeInteger((uint32_t)writeSize, OS, DI.IsLittleEndian);
            OS.write(reinterpret_cast<char *>(&FormVal->BlockData[0]),
                     FormVal->BlockData.size());
            break;
          }
          case dwarf::DW_FORM_data1:
          case dwarf::DW_FORM_ref1:
          case dwarf::DW_FORM_flag:
            writeInteger((uint8_t)FormVal->Value, OS, DI.IsLittleEndian);
            break;
          case dwarf::DW_FORM_data2:
          case dwarf::DW_FORM_ref2:
            writeInteger((uint16_t)FormVal->Value, OS, DI.IsLittleEndian);
            break;
          case dwarf::DW_FORM_data4:
          case dwarf::DW_FORM_ref4:
            writeInteger((uint32_t)FormVal->Value, OS, DI.IsLittleEndian);
            break;
          case dwarf::DW_FORM_data8:
          case dwarf::DW_FORM_ref8:
            writeInteger((uint64_t)FormVal->Value, OS, DI.IsLittleEndian);
            break;
          case dwarf::DW_FORM_sdata:
            encodeSLEB128(FormVal->Value, OS);
            break;
          case dwarf::DW_FORM_udata:
          case dwarf::DW_FORM_ref_udata:
            encodeULEB128(FormVal->Value, OS);
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
          case dwarf::DW_FORM_ref_sup:
            // TODO: Handle DWARF32/64
            writeInteger((uint32_t)FormVal->Value, OS, DI.IsLittleEndian);
            break;
          case dwarf::DW_FORM_ref_sig8:
            writeInteger((uint64_t)FormVal->Value, OS, DI.IsLittleEndian);
            break;
          case dwarf::DW_FORM_GNU_addr_index:
          case dwarf::DW_FORM_GNU_str_index:
            encodeULEB128(FormVal->Value, OS);
            break;
          default:
            break;
          }
        } while (Indirect);
      }
    }
  }
}
