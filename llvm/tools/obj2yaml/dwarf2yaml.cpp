//===------ dwarf2yaml.cpp - obj2yaml conversion tool -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Error.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugArangeSet.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/ObjectYAML/DWARFYAML.h"

#include <algorithm>

using namespace llvm;

void dumpDebugAbbrev(DWARFContextInMemory &DCtx, DWARFYAML::Data &Y) {
  auto AbbrevSetPtr = DCtx.getDebugAbbrev();
  if (AbbrevSetPtr) {
    for (auto AbbrvDeclSet : *AbbrevSetPtr) {
      for (auto AbbrvDecl : AbbrvDeclSet.second) {
        DWARFYAML::Abbrev Abbrv;
        Abbrv.Code = AbbrvDecl.getCode();
        Abbrv.Tag = AbbrvDecl.getTag();
        Abbrv.Children = AbbrvDecl.hasChildren() ? dwarf::DW_CHILDREN_yes
                                                 : dwarf::DW_CHILDREN_no;
        for (auto Attribute : AbbrvDecl.attributes()) {
          DWARFYAML::AttributeAbbrev AttAbrv;
          AttAbrv.Attribute = Attribute.Attr;
          AttAbrv.Form = Attribute.Form;
          Abbrv.Attributes.push_back(AttAbrv);
        }
        Y.AbbrevDecls.push_back(Abbrv);
      }
    }
  }
}

void dumpDebugStrings(DWARFContextInMemory &DCtx, DWARFYAML::Data &Y) {
  StringRef RemainingTable = DCtx.getStringSection();
  while (RemainingTable.size() > 0) {
    auto SymbolPair = RemainingTable.split('\0');
    RemainingTable = SymbolPair.second;
    Y.DebugStrings.push_back(SymbolPair.first);
  }
}

void dumpDebugARanges(DWARFContextInMemory &DCtx, DWARFYAML::Data &Y) {
  DataExtractor ArangesData(DCtx.getARangeSection(), DCtx.isLittleEndian(), 0);
  uint32_t Offset = 0;
  DWARFDebugArangeSet Set;

  while (Set.extract(ArangesData, &Offset)) {
    DWARFYAML::ARange Range;
    Range.Length = Set.getHeader().Length;
    Range.Version = Set.getHeader().Version;
    Range.CuOffset = Set.getHeader().CuOffset;
    Range.AddrSize = Set.getHeader().AddrSize;
    Range.SegSize = Set.getHeader().SegSize;
    for (auto Descriptor : Set.descriptors()) {
      DWARFYAML::ARangeDescriptor Desc;
      Desc.Address = Descriptor.Address;
      Desc.Length = Descriptor.Length;
      Range.Descriptors.push_back(Desc);
    }
    Y.ARanges.push_back(Range);
  }
}

void dumpPubSection(DWARFContextInMemory &DCtx, DWARFYAML::PubSection &Y,
                    StringRef Section) {
  DataExtractor PubSectionData(Section, DCtx.isLittleEndian(), 0);
  uint32_t Offset = 0;
  Y.Length = PubSectionData.getU32(&Offset);
  Y.Version = PubSectionData.getU16(&Offset);
  Y.UnitOffset = PubSectionData.getU32(&Offset);
  Y.UnitSize = PubSectionData.getU32(&Offset);
  while (Offset < Y.Length) {
    DWARFYAML::PubEntry NewEntry;
    NewEntry.DieOffset = PubSectionData.getU32(&Offset);
    if (Y.IsGNUStyle)
      NewEntry.Descriptor = PubSectionData.getU8(&Offset);
    NewEntry.Name = PubSectionData.getCStr(&Offset);
    Y.Entries.push_back(NewEntry);
  }
}

void dumpDebugPubSections(DWARFContextInMemory &DCtx, DWARFYAML::Data &Y) {
  Y.PubNames.IsGNUStyle = false;
  dumpPubSection(DCtx, Y.PubNames, DCtx.getPubNamesSection());

  Y.PubTypes.IsGNUStyle = false;
  dumpPubSection(DCtx, Y.PubTypes, DCtx.getPubTypesSection());

  Y.GNUPubNames.IsGNUStyle = true;
  dumpPubSection(DCtx, Y.GNUPubNames, DCtx.getGnuPubNamesSection());

  Y.GNUPubTypes.IsGNUStyle = true;
  dumpPubSection(DCtx, Y.GNUPubTypes, DCtx.getGnuPubTypesSection());
}

void dumpDebugInfo(DWARFContextInMemory &DCtx, DWARFYAML::Data &Y) {
  for (const auto &CU : DCtx.compile_units()) {
    DWARFYAML::Unit NewUnit;
    NewUnit.Length = CU->getLength();
    NewUnit.Version = CU->getVersion();
    NewUnit.AbbrOffset = CU->getAbbreviations()->getOffset();
    NewUnit.AddrSize = CU->getAddressByteSize();
    for (auto DIE : CU->dies()) {
      DWARFYAML::Entry NewEntry;
      DataExtractor EntryData = CU->getDebugInfoExtractor();
      uint32_t offset = DIE.getOffset();

      assert(EntryData.isValidOffset(offset) && "Invalid DIE Offset");
      if (!EntryData.isValidOffset(offset))
        continue;

      NewEntry.AbbrCode = EntryData.getULEB128(&offset);

      auto AbbrevDecl = DIE.getAbbreviationDeclarationPtr();
      if (AbbrevDecl) {
        for (const auto &AttrSpec : AbbrevDecl->attributes()) {
          DWARFYAML::FormValue NewValue;
          NewValue.Value = 0xDEADBEEFDEADBEEF;
          DWARFDie DIEWrapper(CU.get(), &DIE);
          auto FormValue = DIEWrapper.find(AttrSpec.Attr);
          if (!FormValue)
            return;
          auto Form = FormValue.getValue().getForm();
          bool indirect = false;
          do {
            indirect = false;
            switch (Form) {
            case dwarf::DW_FORM_addr:
            case dwarf::DW_FORM_GNU_addr_index:
              if (auto Val = FormValue.getValue().getAsAddress())
                NewValue.Value = Val.getValue();
              break;
            case dwarf::DW_FORM_ref_addr:
            case dwarf::DW_FORM_ref1:
            case dwarf::DW_FORM_ref2:
            case dwarf::DW_FORM_ref4:
            case dwarf::DW_FORM_ref8:
            case dwarf::DW_FORM_ref_udata:
            case dwarf::DW_FORM_ref_sig8:
              if (auto Val = FormValue.getValue().getAsReferenceUVal())
                NewValue.Value = Val.getValue();
              break;
            case dwarf::DW_FORM_exprloc:
            case dwarf::DW_FORM_block:
            case dwarf::DW_FORM_block1:
            case dwarf::DW_FORM_block2:
            case dwarf::DW_FORM_block4:
              if (auto Val = FormValue.getValue().getAsBlock()) {
                auto BlockData = Val.getValue();
                std::copy(BlockData.begin(), BlockData.end(),
                          std::back_inserter(NewValue.BlockData));
              }
              NewValue.Value = NewValue.BlockData.size();
              break;
            case dwarf::DW_FORM_data1:
            case dwarf::DW_FORM_flag:
            case dwarf::DW_FORM_data2:
            case dwarf::DW_FORM_data4:
            case dwarf::DW_FORM_data8:
            case dwarf::DW_FORM_sdata:
            case dwarf::DW_FORM_udata:
              if (auto Val = FormValue.getValue().getAsUnsignedConstant())
                NewValue.Value = Val.getValue();
              break;
            case dwarf::DW_FORM_string:
              if (auto Val = FormValue.getValue().getAsCString())
                NewValue.CStr = Val.getValue();
              break;
            case dwarf::DW_FORM_indirect:
              indirect = true;
              if (auto Val = FormValue.getValue().getAsUnsignedConstant()) {
                NewValue.Value = Val.getValue();
                NewEntry.Values.push_back(NewValue);
                Form = static_cast<dwarf::Form>(Val.getValue());
              }
              break;
            case dwarf::DW_FORM_strp:
            case dwarf::DW_FORM_sec_offset:
            case dwarf::DW_FORM_GNU_ref_alt:
            case dwarf::DW_FORM_GNU_strp_alt:
            case dwarf::DW_FORM_line_strp:
            case dwarf::DW_FORM_strp_sup:
            case dwarf::DW_FORM_ref_sup:
            case dwarf::DW_FORM_GNU_str_index:
              if (auto Val = FormValue.getValue().getAsCStringOffset())
                NewValue.Value = Val.getValue();
              break;
            case dwarf::DW_FORM_flag_present:
              NewValue.Value = 1;
              break;
            default:
              break;
            }
          } while (indirect);
          NewEntry.Values.push_back(NewValue);
        }
      }

      NewUnit.Entries.push_back(NewEntry);
    }
    Y.CompileUnits.push_back(NewUnit);
  }
}

bool dumpFileEntry(DataExtractor &Data, uint32_t &Offset,
                   DWARFYAML::File &File) {
  File.Name = Data.getCStr(&Offset);
  if (File.Name.empty())
    return false;
  File.DirIdx = Data.getULEB128(&Offset);
  File.ModTime = Data.getULEB128(&Offset);
  File.Length = Data.getULEB128(&Offset);
  return true;
}

void dumpDebugLines(DWARFContextInMemory &DCtx, DWARFYAML::Data &Y) {
  for (const auto &CU : DCtx.compile_units()) {
    auto CUDIE = CU->getUnitDIE();
    if (!CUDIE)
      continue;
    if (auto StmtOffset =
            dwarf::toSectionOffset(CUDIE.find(dwarf::DW_AT_stmt_list))) {
      DWARFYAML::LineTable DebugLines;
      DataExtractor LineData(DCtx.getLineSection().Data, DCtx.isLittleEndian(),
                             CU->getAddressByteSize());
      uint32_t Offset = *StmtOffset;
      uint64_t SizeOfPrologueLength = 4;
      DebugLines.TotalLength = LineData.getU32(&Offset);
      uint64_t LineTableLength = DebugLines.TotalLength;
      if (DebugLines.TotalLength == UINT32_MAX) {
        DebugLines.TotalLength64 = LineData.getU64(&Offset);
        LineTableLength = DebugLines.TotalLength64;
        SizeOfPrologueLength = 8;
      }
      DebugLines.Version = LineData.getU16(&Offset);
      DebugLines.PrologueLength =
          LineData.getUnsigned(&Offset, SizeOfPrologueLength);
      const uint64_t EndPrologue = DebugLines.PrologueLength + Offset;

      DebugLines.MinInstLength = LineData.getU8(&Offset);
      if (DebugLines.Version >= 4)
        DebugLines.MaxOpsPerInst = LineData.getU8(&Offset);
      DebugLines.DefaultIsStmt = LineData.getU8(&Offset);
      DebugLines.LineBase = LineData.getU8(&Offset);
      DebugLines.LineRange = LineData.getU8(&Offset);
      DebugLines.OpcodeBase = LineData.getU8(&Offset);

      DebugLines.StandardOpcodeLengths.reserve(DebugLines.OpcodeBase - 1);
      for (uint8_t i = 1; i < DebugLines.OpcodeBase; ++i)
        DebugLines.StandardOpcodeLengths.push_back(LineData.getU8(&Offset));

      while (Offset < EndPrologue) {
        StringRef Dir = LineData.getCStr(&Offset);
        if (!Dir.empty())
          DebugLines.IncludeDirs.push_back(Dir);
        else
          break;
      }

      while (Offset < EndPrologue) {
        DWARFYAML::File TmpFile;
        if (dumpFileEntry(LineData, Offset, TmpFile))
          DebugLines.Files.push_back(TmpFile);
        else
          break;
      }

      const uint64_t LineEnd =
          LineTableLength + *StmtOffset + SizeOfPrologueLength;
      while (Offset < LineEnd) {
        DWARFYAML::LineTableOpcode NewOp;
        NewOp.Opcode = (dwarf::LineNumberOps)LineData.getU8(&Offset);
        if (NewOp.Opcode == 0) {
          auto StartExt = Offset;
          NewOp.ExtLen = LineData.getULEB128(&Offset);
          NewOp.SubOpcode =
              (dwarf::LineNumberExtendedOps)LineData.getU8(&Offset);
          switch (NewOp.SubOpcode) {
          case dwarf::DW_LNE_set_address:
          case dwarf::DW_LNE_set_discriminator:
            NewOp.Data = LineData.getAddress(&Offset);
            break;
          case dwarf::DW_LNE_define_file:
            dumpFileEntry(LineData, Offset, NewOp.FileEntry);
            break;
          case dwarf::DW_LNE_end_sequence:
            break;
          default:
            while (Offset < StartExt + NewOp.ExtLen)
              NewOp.UnknownOpcodeData.push_back(LineData.getU8(&Offset));
          }
        } else if (NewOp.Opcode < DebugLines.OpcodeBase) {
          switch (NewOp.Opcode) {
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
            NewOp.Data = LineData.getULEB128(&Offset);
            break;

          case dwarf::DW_LNS_advance_line:
            NewOp.SData = LineData.getSLEB128(&Offset);
            break;

          case dwarf::DW_LNS_fixed_advance_pc:
            NewOp.Data = LineData.getU16(&Offset);
            break;

          default:
            for (uint8_t i = 0;
                 i < DebugLines.StandardOpcodeLengths[NewOp.Opcode - 1]; ++i)
              NewOp.StandardOpcodeData.push_back(LineData.getULEB128(&Offset));
          }
        }
        DebugLines.Opcodes.push_back(NewOp);
      }
      Y.DebugLines.push_back(DebugLines);
    }
  }
}

std::error_code dwarf2yaml(DWARFContextInMemory &DCtx, DWARFYAML::Data &Y) {
  dumpDebugAbbrev(DCtx, Y);
  dumpDebugStrings(DCtx, Y);
  dumpDebugARanges(DCtx, Y);
  dumpDebugPubSections(DCtx, Y);
  dumpDebugInfo(DCtx, Y);
  dumpDebugLines(DCtx, Y);
  return obj2yaml_error::success;
}
