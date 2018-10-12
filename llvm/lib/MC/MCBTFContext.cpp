//===- lib/MC/MCBTFContext.cpp - Machine Code BTF Context -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCBTFContext.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <tuple>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "btf"

void MCBTFContext::addTypeEntry(std::unique_ptr<BTFTypeEntry> Entry) {
   TypeEntries.push_back(std::move(Entry));
}

void MCBTFContext::dump(raw_ostream &OS) {
  OS << "Type Table:\n";
  for (size_t i = 0; i < TypeEntries.size(); i++) {
    auto TypeEntry = TypeEntries[i].get();
    TypeEntry->print(OS, *this);
  }

  OS << "\nString Table:\n";
  StringTable.showTable(OS);

  OS << "\nFuncInfo Table:\n";
  for (auto &FuncSec : FuncInfoTable) {
    OS << "sec_name_off=" << FuncSec.first << "\n";
    for (auto &FuncInfo : FuncSec.second) {
      OS << "\tinsn_offset=<Omitted> type_id="
         << FuncInfo.TypeId << "\n";
    }
  }

  OS << "\nLineInfo Table:\n";
  for (auto &LineSec : LineInfoTable) {
    OS << "sec_name_off=" << LineSec.first << "\n";
    for (auto &LineInfo : LineSec.second) {
      OS << "\tinsn_offset=<Omitted> file_name_off="
         << LineInfo.FileNameOff
         << " line_off=" << LineInfo.LineOff
         << " line_num=" << LineInfo.LineNum
         << " column_num=" << LineInfo.ColumnNum
         << "\n";
    }
  }
}

void MCBTFContext::emitCommonHeader(MCObjectStreamer *MCOS) {
  MCOS->EmitIntValue(BTF_MAGIC, 2);
  MCOS->EmitIntValue(BTF_VERSION, 1);
  MCOS->EmitIntValue(0, 1);
}

void MCBTFContext::emitBTFSection(MCObjectStreamer *MCOS) {
  MCContext &context = MCOS->getContext();
  MCOS->SwitchSection(context.getObjectFileInfo()->getBTFSection());

  // emit header
  emitCommonHeader(MCOS);
  MCOS->EmitIntValue(sizeof(struct btf_header), 4);

  uint32_t type_len = 0, str_len;
  for (auto &TypeEntry : TypeEntries)
    type_len += TypeEntry->getSize();
  str_len = StringTable.getSize();

  MCOS->EmitIntValue(0, 4);
  MCOS->EmitIntValue(type_len, 4);
  MCOS->EmitIntValue(type_len, 4);
  MCOS->EmitIntValue(str_len, 4);

  // emit type table
  for (auto &TypeEntry: TypeEntries)
    TypeEntry->emitData(MCOS);

  // emit string table
  for (auto &S : StringTable.getTable()) {
    for (auto C : S)
      MCOS->EmitIntValue(C, 1);
    MCOS->EmitIntValue('\0', 1);
  }
}

void MCBTFContext::emitBTFExtSection(MCObjectStreamer *MCOS) {
  MCContext &context = MCOS->getContext();
  MCOS->SwitchSection(context.getObjectFileInfo()->getBTFExtSection());

  // emit header
  emitCommonHeader(MCOS);
  MCOS->EmitIntValue(sizeof(struct btf_ext_header), 4);

  uint32_t func_len = 0, line_len = 0;
  for (auto &FuncSec : FuncInfoTable) {
    func_len += sizeof(struct btf_sec_func_info);
    func_len += FuncSec.second.size() * sizeof(struct bpf_func_info);
  }
  for (auto &LineSec : LineInfoTable) {
    line_len += sizeof(struct btf_sec_line_info);
    line_len += LineSec.second.size() * sizeof(struct bpf_line_info);
  }

  MCOS->EmitIntValue(0, 4);
  MCOS->EmitIntValue(func_len, 4);
  MCOS->EmitIntValue(func_len, 4);
  MCOS->EmitIntValue(line_len, 4);

  // emit func_info table
  for (const auto &FuncSec : FuncInfoTable) {
    MCOS->EmitIntValue(FuncSec.first, 4);
    MCOS->EmitIntValue(FuncSec.second.size(), 4);
    for (const auto &FuncInfo : FuncSec.second) {
      MCOS->EmitBTFAdvanceLineAddr(FuncInfo.Label, 4);
      MCOS->EmitIntValue(FuncInfo.TypeId, 4);
    }
  }

  // emit line_info table
  for (const auto &LineSec : LineInfoTable) {
    MCOS->EmitIntValue(LineSec.first, 4);
    MCOS->EmitIntValue(LineSec.second.size(), 4);
    for (const auto &LineInfo : LineSec.second) {
      MCOS->EmitBTFAdvanceLineAddr(LineInfo.Label, 4);
      MCOS->EmitIntValue(LineInfo.FileNameOff, 4);
      MCOS->EmitIntValue(LineInfo.LineOff, 4);
      MCOS->EmitIntValue(LineInfo.LineNum << 10 | LineInfo.ColumnNum, 4);
    }
  }
}

void MCBTFContext::emitAll(MCObjectStreamer *MCOS) {
  LLVM_DEBUG(dump(dbgs()));
  emitBTFSection(MCOS);
  emitBTFExtSection(MCOS);
}

void BTFTypeEntry::print(raw_ostream &OS, MCBTFContext& MCBTFContext) {
  OS << "[" << Id << "] "
     << btf_kind_str[BTF_INFO_KIND(BTFType.info)]
     << " name_off=" << BTFType.name_off
     << " info=" << format("0x%08lx", BTFType.info)
     << " size/type=" << BTFType.size << "\n";
}

void BTFTypeEntry::emitData(MCObjectStreamer *MCOS) {
  MCOS->EmitIntValue(BTFType.name_off, 4);
  MCOS->EmitIntValue(BTFType.info, 4);
  MCOS->EmitIntValue(BTFType.size, 4);
}

void BTFTypeEntryInt::print(raw_ostream &OS, MCBTFContext& MCBTFContext) {
  BTFTypeEntry::print(OS, MCBTFContext);
  OS << "\tdesc=" << format("0x%08lx", IntVal) << "\n";
}

void BTFTypeEntryInt::emitData(MCObjectStreamer *MCOS) {
  BTFTypeEntry::emitData(MCOS);
  MCOS->EmitIntValue(IntVal, 4);
}

void BTFTypeEntryEnum::print(raw_ostream &OS, MCBTFContext& MCBTFContext) {
 BTFTypeEntry::print(OS, MCBTFContext);
  for (size_t i = 0; i < BTF_INFO_VLEN(BTFType.info); i++) {
    auto &EnumValue = EnumValues[i];
    OS << "\tname_off=" << EnumValue.name_off
       << " value=" << EnumValue.val << "\n";
  }
}

void BTFTypeEntryEnum::emitData(MCObjectStreamer *MCOS) {
  BTFTypeEntry::emitData(MCOS);
  for (auto &EnumValue : EnumValues) {
    MCOS->EmitIntValue(EnumValue.name_off, 4);
    MCOS->EmitIntValue(EnumValue.val, 4);
  }
}

void BTFTypeEntryArray::print(raw_ostream &OS, MCBTFContext& MCBTFContext) {
  BTFTypeEntry::print(OS, MCBTFContext);
  OS << "\telem_type=" << format("0x%08lx", ArrayInfo.type)
     << " index_type=" << format("0x%08lx", ArrayInfo.index_type)
     << " num_element=" << ArrayInfo.nelems << "\n";
}

void BTFTypeEntryArray::emitData(MCObjectStreamer *MCOS) {
  BTFTypeEntry::emitData(MCOS);
  MCOS->EmitIntValue(ArrayInfo.type, 4);
  MCOS->EmitIntValue(ArrayInfo.index_type, 4);
  MCOS->EmitIntValue(ArrayInfo.nelems, 4);
}

void BTFTypeEntryStruct::print(raw_ostream &OS, MCBTFContext& MCBTFContext) {
  BTFTypeEntry::print(OS, MCBTFContext);
   for (size_t i = 0; i < BTF_INFO_VLEN(BTFType.info); i++) {
    auto &Member = Members[i];
    OS << "\tname_off=" << Member.name_off
       << " type=" << Member.type
       << " bit_offset=" << Member.offset << "\n";
  }
}

void BTFTypeEntryStruct::emitData(MCObjectStreamer *MCOS) {
  BTFTypeEntry::emitData(MCOS);
  for (auto &Member : Members) {
    MCOS->EmitIntValue(Member.name_off, 4);
    MCOS->EmitIntValue(Member.type, 4);
    MCOS->EmitIntValue(Member.offset, 4);
  }
}

void BTFTypeEntryFunc::print(raw_ostream &OS, MCBTFContext& MCBTFContext) {
  BTFTypeEntry::print(OS, MCBTFContext);
   for (size_t i = 0; i < BTF_INFO_VLEN(BTFType.info); i++) {
    auto Parameter = Parameters[i];
    OS << "\tparam_type=" << Parameter << "\n";
  }
}

void BTFTypeEntryFunc::emitData(MCObjectStreamer *MCOS) {
  BTFTypeEntry::emitData(MCOS);
  for (auto &Parameter: Parameters)
    MCOS->EmitIntValue(Parameter, 4);
}
