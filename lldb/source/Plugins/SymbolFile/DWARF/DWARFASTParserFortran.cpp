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
  } else if (!sc.function && sc_parent_die) {
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

void DWARFASTParserFortran::ParseChildParameters(
    const DWARFDIE &parent_die,
    llvm::SmallVectorImpl<CompilerType> &function_param_types,
    llvm::SmallVectorImpl<llvm::StringRef> &function_param_names) {
  if (!parent_die)
    return;

  for (DWARFDIE die : parent_die.children()) {
    const dw_tag_t tag = die.Tag();
    switch (tag) {
    case DW_TAG_formal_parameter: {
      if (die.GetAttributeValueAsUnsigned(DW_AT_artificial, 0))
        continue;

      DWARFDIE param_type_die = die.GetAttributeValueAsReferenceDIE(DW_AT_type);

      Type *type = die.ResolveTypeUID(param_type_die);
      if (!type)
        break;

      function_param_names.emplace_back(die.GetName());
      function_param_types.push_back(type->GetForwardCompilerType());
    } break;
    default:
      break;
    }
  }

  assert(function_param_names.size() == function_param_types.size());
}

lldb::TypeSP DWARFASTParserFortran::ParseTypeFromDWARF(
    const lldb_private::SymbolContext &sc,
    const lldb_private::plugin::dwarf::DWARFDIE &die, bool *type_is_new_ptr) {
  TypeSP type_sp;
  if (type_is_new_ptr)
    *type_is_new_ptr = false;

  Log *log = GetLog(DWARFLog::TypeCompletion | DWARFLog::Lookups);

  if (!die)
    return type_sp;
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

  if (type_ptr) {
    if (type_ptr != DIE_IS_BEING_PARSED)
      return type_ptr->shared_from_this();
    else
      return TypeSP();
  }

  if (type_is_new_ptr)
    *type_is_new_ptr = true;

  ConstString type_name;
  const char *type_name_cstr = nullptr;
  CompilerType compiler_type;
  DWARFAttributes attributes;
  DWARFFormValue form_value;
  Declaration decl;
  // We use 0x00 as a sentinel value, it doesn't map to any DW_ATE attribute
  llvm::dwarf::TypeKind encoding = static_cast<llvm::dwarf::TypeKind>(0x00);

  switch (const dw_tag_t tag = die.Tag()) {
  case DW_TAG_base_type: {
    dwarf->GetDIEToType()[die.GetDIE()] = DIE_IS_BEING_PARSED;
    attributes = die.GetAttributes();
    uint64_t bit_size = 0;
    for (size_t idx = 0; idx < attributes.Size(); idx++) {
      if (attributes.ExtractFormValueAtIndex(idx, form_value)) {

        switch (attributes.AttributeAtIndex(idx)) {
        case DW_AT_name:
          type_name_cstr = form_value.AsCString();

          if (type_name_cstr && type_name_cstr[0] != '\0')
            type_name.SetString(llvm::StringRef(type_name_cstr).upper());
          else
            type_name.SetCString("UNKNOWN_FORTRAN_TYPE");

          break;
        case DW_AT_encoding:
          encoding = static_cast<llvm::dwarf::TypeKind>(form_value.Unsigned());
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
        dwarf->MakeType(die.GetID(), type_name, llvm::divideCeil(bit_size, 8),
                        nullptr, LLDB_INVALID_UID, Type::eEncodingIsUID, decl,
                        compiler_type, Type::ResolveState::Full);
    break;
  }
  case DW_TAG_subprogram:
  case DW_TAG_subroutine_type: {
    CompilerType return_type;
    dwarf->GetDIEToType()[die.GetDIE()] = DIE_IS_BEING_PARSED;
    attributes = die.GetAttributes();
    size_t num_attr = attributes.Size();

    for (size_t i = 0; i < num_attr; ++i) {
      if (!attributes.ExtractFormValueAtIndex(i, form_value))
        continue;
      switch (attributes.AttributeAtIndex(i)) {
      case DW_AT_name:
        type_name_cstr = form_value.AsCString();
        if (type_name_cstr &&
            type_name_cstr[0]) { // Check for null AND empty string
          type_name.SetString(type_name_cstr);
        } else {
          type_name.SetCString("UNKNOWN_FORTRAN_FUNCTION");
        }
        break;

      case DW_AT_type: {
        if (DWARFDIE type_die = form_value.Reference())
          if (Type *resolved_type = die.GetDWARF()->ResolveType(type_die))
            return_type = resolved_type->GetForwardCompilerType();

        break;
      }

      default:
        break;
      }
    }

    llvm::SmallVector<CompilerType, 4> function_params_types;
    llvm::SmallVector<llvm::StringRef, 4> function_params_names;
    ParseChildParameters(die, function_params_types, function_params_names);

    compiler_type = m_ast.CreateFortranFunction(
        type_name, function_params_types, function_params_names, return_type);

    type_sp = dwarf->MakeType(die.GetID(), type_name, 0, nullptr,
                              LLDB_INVALID_UID, Type::eEncodingIsUID, decl,
                              compiler_type, Type::ResolveState::Full);
    break;
  }

  default:
    if (log)
      dwarf->GetObjectFile()->GetModule()->LogMessage(
          log, "[{0:x16}]: unhandled type tag {1:x4} ({2})", die.GetOffset(),
          tag, DW_TAG_value_to_name(tag));

    break;
  }

  UpdateSymbolContextScopeForType(sc, die, type_sp);
  if (type_sp.get())
    dwarf->GetDIEToType()[die.GetDIE()] = type_sp.get();

  return type_sp;
}

lldb_private::Function *DWARFASTParserFortran::ParseFunctionFromDWARF(
    lldb_private::CompileUnit &comp_unit,
    const lldb_private::plugin::dwarf::DWARFDIE &die,
    lldb_private::AddressRanges ranges) {
  if (die.Tag() != DW_TAG_subprogram && die.Tag() != DW_TAG_subroutine_type)
    return nullptr;

  llvm::DWARFAddressRangesVector unused_func_ranges;
  const char *name = nullptr;
  const char *mangled = nullptr;
  std::optional<int> decl_file = 0;
  std::optional<int> decl_line = 0;
  std::optional<int> decl_column = 0;
  std::optional<int> call_file = 0;
  std::optional<int> call_line = 0;
  std::optional<int> call_column = 0;
  DWARFExpressionList frame_base;
  if (!die.GetDIENamesAndRanges(name, mangled, unused_func_ranges, decl_file,
                                decl_line, decl_column, call_file, call_line,
                                call_column, &frame_base))
    return nullptr;

  Mangled func_name;
  // Mangled doesn't know how to demangle fortran names
  if (mangled)
    func_name.SetMangledName(ConstString(mangled));

  if (name)
    func_name.SetDemangledName(ConstString(name));

  FunctionSP func_sp;
  SymbolFileDWARF *dwarf = die.GetDWARF();
  // Supply the type only if it has already been parsed
  Type *func_type = dwarf->GetDIEToType().lookup(die.GetDIE());
  assert(func_type == nullptr || func_type != DIE_IS_BEING_PARSED);
  const user_id_t func_user_id = die.GetID();
  Address func_addr = ranges[0].GetBaseAddress();

  func_sp = std::make_shared<Function>(&comp_unit,
                                       func_user_id, // UserID is the DIE offset
                                       func_user_id, func_name, func_type,
                                       std::move(func_addr), std::move(ranges));

  if (func_sp.get() == nullptr)
    return nullptr;

  if (frame_base.IsValid())
    func_sp->GetFrameBaseExpression() = frame_base;

  comp_unit.AddFunction(func_sp);
  return func_sp.get();
}

bool DWARFASTParserFortran::CompleteTypeFromDWARF(
    const lldb_private::plugin::dwarf::DWARFDIE &die, lldb_private::Type *type,
    const lldb_private::CompilerType &compiler_type) {
  return false;
}
