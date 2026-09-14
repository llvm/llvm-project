//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the DWARF AST parser for the Fortran language.
///
//===----------------------------------------------------------------------===//

#include "DWARFASTParserFortran.h"
#include "DWARFDIE.h"
#include "DWARFDebugInfo.h"
#include "DWARFDeclContext.h"
#include "DWARFDefines.h"
#include "LogChannelDWARF.h"
#include "SymbolFileDWARF.h"
#include "SymbolFileDWARFDebugMap.h"
#include "UniqueDWARFASTType.h"

#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Utility/Log.h"
#include "lldb/ValueObject/ValueObject.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::plugin::dwarf;
using namespace llvm::dwarf;

DWARFASTParserFortran::DWARFASTParserFortran(
    lldb_private::TypeSystemFortran &m_ast)
    : lldb_private::plugin::dwarf::DWARFASTParser(Kind::DWARFASTParserFortran),
      m_ast(m_ast) {}

DWARFASTParserFortran::~DWARFASTParserFortran() {}

TypeSP DWARFASTParserFortran::UpdateSymbolContextScopeForType(
    const SymbolContext &sc, const DWARFDIE &die, TypeSP type_sp) {
  if (!type_sp)
    return type_sp;

  DWARFDIE sc_parent_die = SymbolFileDWARF::GetParentSymbolContextDIE(die);
  dw_tag_t sc_parent_tag = sc_parent_die.Tag();

  SymbolContextScope *symbol_context_scope = nullptr;
  if (sc_parent_tag == DW_TAG_compile_unit ||
      sc_parent_tag == DW_TAG_partial_unit) {
    symbol_context_scope = sc.comp_unit;
  } else if (sc.function != nullptr && sc_parent_die) {
    symbol_context_scope =
        sc.function->GetBlock(true).FindBlockByID(sc_parent_die.GetID());
    if (symbol_context_scope == nullptr)
      symbol_context_scope = sc.function;
  } else {
    symbol_context_scope = sc.module_sp.get();
  }

  if (symbol_context_scope != nullptr)
    type_sp->SetSymbolContextScope(symbol_context_scope);
  return type_sp;
}

lldb::TypeSP DWARFASTParserFortran::ParseTypeFromDWARF(
    const lldb_private::SymbolContext &sc,
    const lldb_private::plugin::dwarf::DWARFDIE &die, bool *type_is_new_ptr) {
  TypeSP type_sp;
  if (type_is_new_ptr)
    *type_is_new_ptr = false;
  Log *log = GetLog(DWARFLog::TypeCompletion | DWARFLog::Lookups);

  if (die) {
    SymbolFileDWARF *dwarf = die.GetDWARF();
    if (log) {
      dwarf->GetObjectFile()->GetModule()->LogMessage(
          log,
          "DWARFASTParserFortran::ParseTypeFromDWARF (die = 0x%8.8x) %s name"
          "= "
          "'%s')",
          die.GetOffset(), plugin::dwarf::DW_TAG_value_to_name(die.Tag()),
          die.GetName());
    }
    Type *type_ptr = dwarf->GetDIEToType().lookup(die.GetDIE());
    if (!type_ptr) {
      if (type_is_new_ptr)
        *type_is_new_ptr = true;

      const dw_tag_t tag = die.Tag();
      ConstString type_name;
      const char *type_name_cstr = nullptr;
      CompilerType compiler_type;
      DWARFAttributes attributes;
      DWARFFormValue form_value;
      Declaration decl;
      uint32_t encoding = 0;
      switch (tag) {
      case DW_TAG_base_type: {
        dwarf->GetDIEToType()[die.GetDIE()] = DIE_IS_BEING_PARSED;
        attributes = die.GetAttributes();
        uint64_t bit_size = 0;
        for (size_t idx = 0; idx < attributes.Size(); idx++) {
          if (attributes.ExtractFormValueAtIndex(idx, form_value)) {
            switch (attributes.AttributeAtIndex(idx)) {
            case DW_AT_name:
              type_name_cstr = form_value.AsCString();
              if (type_name_cstr &&
                  type_name_cstr[0]) { // Check for null AND empty string
                type_name.SetString(llvm::StringRef(type_name_cstr).upper());
              } else {
                type_name.SetCString("UNKNOWN_FORTRAN_TYPE");
              }
              break;
            case DW_AT_encoding:
              encoding = form_value.Unsigned();
              break;
            case DW_AT_byte_size:
              bit_size = form_value.Unsigned() * 8;
              break;
            case DW_AT_bit_size:
              bit_size = form_value.Unsigned();
              break;
            default:
              break;
            }
          }
        }
        compiler_type = m_ast.CreateBaseType(encoding, bit_size, type_name);
        type_sp =
            dwarf->MakeType(die.GetID(), type_name, (bit_size + 7) / 8, nullptr,
                            LLDB_INVALID_UID, Type::eEncodingIsUID, decl,
                            compiler_type, Type::ResolveState::Full);
      } break;
      default:
        if (log) {
          dwarf->GetObjectFile()->GetModule()->LogMessage(
              log, "[{0:x16}]: unhandled type tag {1:x4} ({2})",
              die.GetOffset(), tag, DW_TAG_value_to_name(tag));
        }
        break;
      }
      UpdateSymbolContextScopeForType(sc, die, type_sp);
      if (type_sp.get())
        dwarf->GetDIEToType()[die.GetDIE()] = type_sp.get();

    } else if (type_ptr != DIE_IS_BEING_PARSED) {
      type_sp = type_ptr->shared_from_this();
    }
  }
  return type_sp;
}

lldb_private::Function *DWARFASTParserFortran::ParseFunctionFromDWARF(
    lldb_private::CompileUnit &comp_unit,
    const lldb_private::plugin::dwarf::DWARFDIE &die,
    lldb_private::AddressRanges ranges) {
  return nullptr;
}

bool DWARFASTParserFortran::CompleteTypeFromDWARF(
    const lldb_private::plugin::dwarf::DWARFDIE &die, lldb_private::Type *type,
    const lldb_private::CompilerType &compiler_type) {
  return false;
}
