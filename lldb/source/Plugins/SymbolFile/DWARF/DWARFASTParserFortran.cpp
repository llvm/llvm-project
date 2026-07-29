//===-- DWARFASTParserFortran.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
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

DWARFExpressionList GetDWARFExpression(const DWARFDIE &die,
                                       const DWARFFormValue &form_value,
                                       ModuleSP module) {
  auto data = die.GetData();
  uint32_t offset = form_value.BlockData() - data.GetDataStart();
  uint32_t length = form_value.Unsigned();
  return DWARFExpressionList(module, DataExtractor(data, offset, length),
                             die.GetCU());
}

FortranArrayMetadata ParseArray(const DWARFDIE &parent_die,
                                const ExecutionContext *exe_ctx) {
  // We first process the array type attributes and then each individual
  // subrange
  FortranArrayMetadata array_info;
  DWARFAttributes parent_attributes = parent_die.GetAttributes();
  ModuleSP parent_module(parent_die.GetModule());
  for (size_t idx = 0; idx < parent_attributes.Size(); idx++) {
    const dw_attr_t attr = parent_attributes.AttributeAtIndex(idx);
    DWARFFormValue form_value;
    if (!parent_attributes.ExtractFormValueAtIndex(idx, form_value))
      continue;
    switch (attr) {
    case DW_AT_data_location:
      array_info.is_dynamic = true;
      array_info.data_location_exp =
          GetDWARFExpression(parent_die, form_value, parent_module);
      break;
    case DW_AT_allocated:
      array_info.is_allocatable = true;
      array_info.is_dynamic = true;
      array_info.allocated_exp =
          GetDWARFExpression(parent_die, form_value, parent_module);
      break;
    default:
      break;
    }
  }
  for (DWARFDIE die : parent_die.children()) {
    const dw_tag_t tag = die.Tag();
    ModuleSP module(die.GetModule());

    if (tag != DW_TAG_subrange_type)
      continue;
    // If a subrange of an array is a star meaning we can't infer how many
    // elements it has it is always the last dimension and is identified by not
    // having a DW_AT_count attribute, if this at the end is true then it is a
    // star and needs to be treated as such
    array_info.is_star = true;
    DWARFAttributes attributes = die.GetAttributes();
    if (attributes.Size() == 0)
      continue;

    DWARFValue num_elements;
    DWARFValue byte_stride;
    DWARFValue lower_bound;
    DWARFValue upper_bound;
    for (size_t i = 0; i < attributes.Size(); ++i) {
      const dw_attr_t attr = attributes.AttributeAtIndex(i);
      DWARFFormValue form_value;
      if (!attributes.ExtractFormValueAtIndex(i, form_value))
        continue;
      switch (attr) {
      case DW_AT_name:
        break;

      case DW_AT_count:
        array_info.is_star = false;
        if (DWARFDIE var_die = die.GetReferencedDIE(DW_AT_count)) {
          if (var_die.Tag() == DW_TAG_variable)
            if (exe_ctx) {
              if (auto frame = exe_ctx->GetFrameSP()) {
                Status error;
                lldb::VariableSP var_sp;
                auto valobj_sp = frame->GetValueForVariableExpressionPath(
                    var_die.GetName(), eNoDynamicValues, 0, var_sp, error);
                if (valobj_sp) {
                  num_elements = valobj_sp->GetValueAsUnsigned(0);
                  break;
                }
              }
            }
        } else if (DWARFFormValue::IsBlockForm(form_value.Form())) {
          num_elements = GetDWARFExpression(die, form_value, module);
          array_info.is_dynamic = true;
        } else
          num_elements = form_value.Unsigned();
        break;

      case DW_AT_byte_stride:
        if (DWARFFormValue::IsBlockForm(form_value.Form())) {
          byte_stride = GetDWARFExpression(die, form_value, module);
          array_info.is_dynamic = true;
        } else
          byte_stride = form_value.Unsigned();
        break;

      case DW_AT_lower_bound:
        if (DWARFFormValue::IsBlockForm(form_value.Form())) {
          lower_bound = GetDWARFExpression(die, form_value, module);
          array_info.is_dynamic = true;
        } else
          lower_bound = form_value.Signed();
        break;

      case DW_AT_upper_bound:
        if (DWARFFormValue::IsBlockForm(form_value.Form())) {
          array_info.is_dynamic = true;
          upper_bound = GetDWARFExpression(die, form_value, module);
        } else
          upper_bound = form_value.Signed();
        break;

      default:
        break;
      }
    }

    if (std::holds_alternative<std::monostate>(num_elements)) {
      if (std::holds_alternative<std::int64_t>(upper_bound) &&
          std::holds_alternative<std::int64_t>(lower_bound))
        num_elements =
            static_cast<uint64_t>(std::get<int64_t>(upper_bound) -
                                  std::get<int64_t>(lower_bound) + 1);
    }
    FortranDimension dimension;
    dimension.element_count = num_elements;
    dimension.lower_bound = lower_bound;
    dimension.upper_bound = upper_bound;
    dimension.byte_stride = byte_stride;
    array_info.dimensions.push_back(dimension);
  }
  return array_info;
}

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
      case DW_TAG_array_type: {
        dwarf->GetDIEToType()[die.GetDIE()] = DIE_IS_BEING_PARSED;

        DWARFDIE element_die = die.GetAttributeValueAsReferenceDIE(DW_AT_type);

        Type *element_type = dwarf->ResolveTypeUID(element_die, true);
        if (element_type) {

          CompilerType array_element_type =
              element_type->GetForwardCompilerType();
          uint64_t total_array_size = 0;
          uint64_t total_elements = 1;
          if (array_element_type.GetCompleteType()) {
            FortranArrayMetadata array_info = ParseArray(die, nullptr);

            array_info.element_type = array_element_type;
            // We need to calculate the total array size, if it is known
            // at compile time
            if (!array_info.is_dynamic) {
              for (size_t idx = 0; idx < array_info.dimensions.size(); idx++) {
                total_elements *= std::get<uint64_t>(
                    array_info.dimensions[idx].element_count);
              }

              // Total size is just total elements * the size of one element
              auto byte_size_or_err = array_element_type.GetByteSize(nullptr);
              if (byte_size_or_err)
                total_array_size = total_elements * (*byte_size_or_err);
              else
                total_array_size = 0;
            }

            compiler_type = m_ast.CreateArrayType(array_info, total_array_size,
                                                  total_elements);

            type_sp = dwarf->MakeType(
                die.GetID(), compiler_type.GetTypeName(), total_array_size,
                nullptr, LLDB_INVALID_UID, Type::eEncodingIsUID, decl,
                compiler_type, Type::ResolveState::Full);
            type_sp->SetEncodingType(element_type);
          } else {
            dwarf->GetObjectFile()->GetModule()->LogMessage(
                log,
                "DWARFASTParserFortran::ParseTypeFromDWARF (die = 0x%8.8x) %s "
                "name "
                "= '%s'), incomplete type array element not supported, yet!.",
                die.GetOffset(), plugin::dwarf::DW_TAG_value_to_name(die.Tag()),
                die.GetName());
          }
        }
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