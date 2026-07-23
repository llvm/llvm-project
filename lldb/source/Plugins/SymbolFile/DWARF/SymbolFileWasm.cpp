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
#include "lldb/Core/Address.h"
#include "lldb/Core/AddressRange.h"
#include "lldb/Core/Module.h"
#include "lldb/Expression/DWARFExpression.h"
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

  // The Wasm "name" section names functions but not data, so data symbols are
  // absent from the symbol table. Recover them from the DWARF. A subprogram's
  // mangled name goes onto its existing code symbol. A global variable at a
  // static DW_OP_addr gets a synthesized data symbol so that its address
  // resolves back to the variable's name. This is how the Itanium C++ runtime
  // recovers a dynamic type from a vtable ("vtable for X"), and it also lets a
  // plain C global resolve back to its name.
  ModuleSP module_sp = GetObjectFile()->GetModule();
  SectionList *section_list = module_sp ? module_sp->GetSectionList() : nullptr;

  DWARFDebugInfo &debug_info = DebugInfo();
  const size_t num_units = debug_info.GetNumUnits();
  for (size_t i = 0; i < num_units; ++i) {
    DWARFUnit *unit = debug_info.GetUnitAtIndex(i);
    if (!unit)
      continue;

    for (const DWARFDebugInfoEntry &entry : unit->dies()) {
      const dw_tag_t tag = entry.Tag();
      if (tag != DW_TAG_subprogram && tag != DW_TAG_variable)
        continue;

      DWARFDIE die(unit, &entry);

      if (tag == DW_TAG_subprogram) {
        // Only a mangled (linkage) name is worth putting onto a code symbol.
        // The source name is already carried by the Wasm name section.
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
        continue;
      }

      // A global variable has no symbol at all. Use its linkage name, or the
      // source name when there is none (plain C globals have only a
      // DW_AT_name).
      const char *name = die.GetMangledName(/*substitute_name_allowed=*/true);
      if (!name)
        continue;

      // A global's location is a plain DW_OP_addr.
      if (!section_list)
        continue;
      DWARFAttributes attributes = die.GetAttributes();
      DWARFFormValue location;
      bool has_location = false;
      for (size_t attr_idx = 0; attr_idx < attributes.Size(); ++attr_idx) {
        if (attributes.AttributeAtIndex(attr_idx) == DW_AT_location) {
          has_location = attributes.ExtractFormValueAtIndex(attr_idx, location);
          break;
        }
      }
      if (!has_location || !DWARFFormValue::IsBlockForm(location.Form()))
        continue;

      const DWARFDataExtractor &data = die.GetData();
      DWARFExpression expr(
          DataExtractor(data, location.BlockData() - data.GetDataStart(),
                        location.Unsigned()));
      llvm::Expected<addr_t> file_addr = expr.GetLocation_DW_OP_addr(unit);
      if (!file_addr) {
        llvm::consumeError(file_addr.takeError());
        continue;
      }

      Address addr(*file_addr, section_list);
      if (!addr.IsSectionOffset())
        continue;

      // Leave the size unset for Symtab to compute from the gap to the next
      // symbol. Don't look the address up in the symtab here. That would build
      // the address index before the remaining vtables are added, mis-sizing
      // them.
      symtab.AddSymbol(Symbol(
          /*symID=*/0, Mangled(ConstString(name)), eSymbolTypeData,
          /*external=*/true, /*is_debug=*/false, /*is_trampoline=*/false,
          /*is_artificial=*/false, AddressRange(addr, 0),
          /*size_is_valid=*/false, /*contains_linker_annotations=*/false,
          /*flags=*/0));
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
