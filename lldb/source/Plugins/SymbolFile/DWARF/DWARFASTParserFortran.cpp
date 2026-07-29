//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the DWARF AST Parser for the Fortran language.
///
//===----------------------------------------------------------------------===//

#include "DWARFASTParserFortran.h"

#include "DWARFDIE.h"
#include "DWARFDebugInfo.h"
#include "DWARFDeclContext.h"
#include "DWARFDefines.h"
#include "LogChannelDWARF.h"
#include "Plugins/TypeSystem/Fortran/FortranTypes.h"
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
using namespace lldb_private::plugin::fortran;

DWARFASTParserFortran::DWARFASTParserFortran(TypeSystemFortran &ast)
    : DWARFASTParser(Kind::DWARFASTParserFortran), m_ast(ast) {}

DWARFASTParserFortran::~DWARFASTParserFortran() {}

// TODO: Add more logging here there is not enough
lldb::TypeSP DWARFASTParserFortran::ParseTypeFromDWARF(const SymbolContext &sc,
                                                       const DWARFDIE &die,
                                                       bool *type_is_new_ptr) {
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
        compiler_type = m_ast.CreateType(encoding, bit_size, type_name);
        type_sp =
            dwarf->MakeType(die.GetID(), type_name, (bit_size + 7) / 8, nullptr,
                            LLDB_INVALID_UID, Type::eEncodingIsUID, decl,
                            compiler_type, Type::ResolveState::Full);
      } break;
      case DW_TAG_subprogram:
      case DW_TAG_subroutine_type: {
        dwarf->GetDIEToType()[die.GetDIE()] = DIE_IS_BEING_PARSED;
        attributes = die.GetAttributes();
        size_t num_attr = attributes.Size();
        for (size_t i = 0; i < num_attr; ++i) {
          if (attributes.ExtractFormValueAtIndex(i, form_value)) {
            switch (attributes.AttributeAtIndex(i)) {
            case DW_AT_name:
              type_name_cstr = form_value.AsCString();
              if (type_name_cstr &&
                  type_name_cstr[0]) { // Check for null AND empty string
                type_name.SetString(llvm::StringRef(type_name_cstr).upper());
              } else {
                type_name.SetCString("UNKNOWN_FORTRAN_FUNCTION");
              }
              break;
            default:
              break;
            }
          }
        }
        llvm::SmallVector<CompilerType, 4> function_params_types;
        // TODO: Parse Parameters here, for now this is not supported
        compiler_type =
            m_ast.GetOrCreateFortranFunction(type_name, function_params_types);
        type_sp = dwarf->MakeType(die.GetID(), type_name, 0, nullptr,
                                  LLDB_INVALID_UID, Type::eEncodingIsUID, decl,
                                  compiler_type, Type::ResolveState::Full);
      } break;
      default:
        break;
      }
      if (type_sp.get()) {
        // TODO: Here calculate the variable scope
        dwarf->GetDIEToType()[die.GetDIE()] = type_sp.get();
      }
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
  if (die.Tag() != DW_TAG_subprogram)
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
  if (die.GetDIENamesAndRanges(name, mangled, unused_func_ranges, decl_file,
                               decl_line, decl_column, call_file, call_line,
                               call_column, &frame_base)) {
    Mangled func_name;
    // Mangled doesn't know how to demangle fortran names
    if (mangled)
      func_name.SetMangledName(ConstString(mangled));
    if (name)
      func_name.SetDemangledName(ConstString(name));

    FunctionSP func_sp;

    SymbolFileDWARF *dwarf = die.GetDWARF();
    // Supply the type _only_ if it has already been parsed
    Type *func_type = dwarf->GetDIEToType().lookup(die.GetDIE());

    assert(func_type == nullptr || func_type != DIE_IS_BEING_PARSED);

    const user_id_t func_user_id = die.GetID();

    Address func_addr = ranges[0].GetBaseAddress();

    func_sp =
        std::make_shared<Function>(&comp_unit,
                                   func_user_id, // UserID is the DIE offset
                                   func_user_id, func_name, func_type,
                                   std::move(func_addr), std::move(ranges));

    if (func_sp.get() != nullptr) {
      if (frame_base.IsValid())
        func_sp->GetFrameBaseExpression() = frame_base;
      comp_unit.AddFunction(func_sp);
      return func_sp.get();
    }
  }
  return nullptr;
}