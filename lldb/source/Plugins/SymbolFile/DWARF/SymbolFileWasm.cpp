//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolFileWasm.h"
#include "Plugins/SymbolFile/DWARF/DWARFDebugInfo.h"
#include "Plugins/SymbolFile/DWARF/DWARFUnit.h"
#include "Plugins/SymbolFile/DWARF/LogChannelDWARF.h"
#include "Utility/WasmVirtualRegisters.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Symbol/Symtab.h"
#include "lldb/Utility/LLDBLog.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::plugin::dwarf;
using namespace llvm::dwarf;

SymbolFileWasm::SymbolFileWasm(ObjectFileSP objfile_sp,
                               SectionList *dwo_section_list)
    : SymbolFileDWARF(objfile_sp, dwo_section_list) {}

SymbolFileWasm::~SymbolFileWasm() = default;

void SymbolFileWasm::AddSymbols(Symtab &symtab) {
  SymbolFileDWARF::AddSymbols(symtab);

  // Copy each function's mangled name from its DWARF DW_AT_linkage_name onto
  // the matching symbol. The match is by file address: a subprogram's
  // DW_AT_low_pc resolves to the same code-section file address as its symbol.
  DWARFDebugInfo &debug_info = DebugInfo();
  const size_t num_units = debug_info.GetNumUnits();
  for (size_t i = 0; i < num_units; ++i) {
    DWARFUnit *unit = debug_info.GetUnitAtIndex(i);
    if (!unit)
      continue;

    for (const DWARFDebugInfoEntry &entry : unit->dies()) {
      if (entry.Tag() != DW_TAG_subprogram)
        continue;

      DWARFDIE die(unit, &entry);
      const char *mangled =
          die.GetMangledName(/*substitute_name_allowed=*/false);
      if (!mangled)
        continue;

      const addr_t file_addr =
          die.GetAttributeValueAsAddress(DW_AT_low_pc, LLDB_INVALID_ADDRESS);
      if (file_addr == LLDB_INVALID_ADDRESS)
        continue;

      Symbol *symbol = symtab.FindSymbolAtFileAddress(file_addr);
      if (symbol && !symbol->GetMangled().GetMangledName())
        symbol->GetMangled().SetMangledName(ConstString(mangled));
    }
  }
}

lldb::offset_t
SymbolFileWasm::GetVendorDWARFOpcodeSize(const DataExtractor &data,
                                         const lldb::offset_t data_offset,
                                         const uint8_t op) const {
  if (op != llvm::dwarf::DW_OP_WASM_location)
    return LLDB_INVALID_OFFSET;

  lldb::offset_t offset = data_offset;
  const uint8_t wasm_op = data.GetU8(&offset);
  switch (wasm_op) {
  case 0: // LOCAL
  case 1: // GLOBAL_FIXED
  case 2: // OPERAND_STACK
    data.GetULEB128(&offset);
    break;
  case 3: // GLOBAL_RELOC
    data.GetU32(&offset);
    break;
  default:
    return LLDB_INVALID_OFFSET;
  }

  return offset - data_offset;
}

bool SymbolFileWasm::ParseVendorDWARFOpcode(uint8_t op,
                                            const llvm::DataExtractor &opcodes,
                                            lldb::offset_t &offset,
                                            RegisterContext *reg_ctx,
                                            lldb::RegisterKind reg_kind,
                                            std::vector<Value> &stack) const {
  if (op != llvm::dwarf::DW_OP_WASM_location)
    return false;

  uint32_t index = 0;
  uint8_t tag = eWasmTagNotAWasmLocation;

  /// |DWARF Location Index | WebAssembly Construct |
  /// |---------------------|-----------------------|
  /// |0                    | Local                 |
  /// |1 or 3               | Global                |
  /// |2                    | Operand Stack         |
  const uint8_t wasm_op = opcodes.getU8(&offset);
  switch (wasm_op) {
  case 0: // LOCAL
    index = opcodes.getULEB128(&offset);
    tag = eWasmTagLocal;
    break;
  case 1: // GLOBAL_FIXED
    index = opcodes.getULEB128(&offset);
    tag = eWasmTagGlobal;
    break;
  case 2: // OPERAND_STACK
    index = opcodes.getULEB128(&offset);
    tag = eWasmTagOperandStack;
    break;
  case 3: // GLOBAL_RELOC
    index = opcodes.getU32(&offset);
    tag = eWasmTagGlobal;
    break;
  default:
    return false;
  }

  const uint32_t reg_num = GetWasmRegister(tag, index);

  Value tmp;
  llvm::Error error = DWARFExpression::ReadRegisterValueAsScalar(
      reg_ctx, reg_kind, reg_num, tmp);
  if (error) {
    LLDB_LOG_ERROR(GetLog(DWARFLog::DebugInfo), std::move(error), "{0}");
    return false;
  }

  stack.push_back(tmp);
  return true;
}
