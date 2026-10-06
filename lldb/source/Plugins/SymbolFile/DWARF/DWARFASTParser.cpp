//===-- DWARFASTParser.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFASTParser.h"
#include "DWARFAttribute.h"
#include "DWARFDIE.h"
#include "DWARFFormValue.h"
#include "DWARFUnit.h"
#include "LogChannelDWARF.h"
#include "SymbolFileDWARF.h"

#include "lldb/Core/Value.h"
#include "lldb/Expression/DWARFExpression.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Target/StackFrame.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/ValueObject/ValueObject.h"

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Error.h"

#include <optional>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::plugin::dwarf;
using namespace llvm::dwarf;

/// Array properties can be encoded directly as constants, or as DWARF
/// expression blocks whose values need to be evaluated in an execution context.
static llvm::Expected<uint64_t>
EvaluateArrayPropertyAsUnsigned(const DWARFFormValue &form_value,
                                const DWARFDIE &parent_die,
                                const ExecutionContext *exe_ctx) {
  if (DWARFFormValue::IsBlockForm(form_value.Form())) {
    // Block-form values carry their byte length as the stored unsigned value.
    // Evaluate the block instead of treating that length as the property value.
    if (!exe_ctx)
      return llvm::createStringError(
          "cannot evaluate array property expression without an execution "
          "context");

    DWARFUnit *unit = parent_die.GetCU();
    SymbolFileDWARF *dwarf = parent_die.GetDWARF();
    if (!unit || !dwarf || !dwarf->GetObjectFile())
      return llvm::createStringError(
          "cannot evaluate array property expression without DWARF context");

    lldb::RegisterContextSP reg_ctx_sp;
    if (lldb::StackFrameSP frame_sp = exe_ctx->GetFrameSP())
      reg_ctx_sp = frame_sp->GetRegisterContext();

    DataExtractor data(form_value.BlockData(), form_value.Unsigned(),
                       unit->GetByteOrder(), unit->GetAddressByteSize());
    ExecutionContext exe_ctx_copy(*exe_ctx);

    // Evaluate the DWARF expression to obtain the dynamic property value.
    llvm::Expected<Value> result = DWARFExpression::Evaluate(
        &exe_ctx_copy, reg_ctx_sp.get(), dwarf->GetObjectFile()->GetModule(),
        data, unit, eRegisterKindDWARF,
        /*initial_value_ptr=*/nullptr, /*object_address_ptr=*/nullptr);
    if (!result)
      return result.takeError();

    return result->GetScalar().ULongLong();
  }

  if (std::optional<uint64_t> value = form_value.getAsUnsignedConstant())
    return *value;
  if (std::optional<int64_t> value = form_value.getAsSignedConstant()) {
    if (*value >= 0)
      return *value;
    return llvm::createStringError("array property value is negative: %lld",
                                   static_cast<long long>(*value));
  }
  return llvm::createStringError("unsupported array property form");
}

static std::optional<uint64_t>
GetDefaultArrayLowerBound(const DWARFDIE &parent_die) {
  DWARFUnit *unit = parent_die.GetCU();
  if (!unit)
    return std::nullopt;

  std::optional<unsigned> lower_bound = LanguageLowerBound(
      static_cast<llvm::dwarf::SourceLanguage>(unit->GetDWARFLanguageType()));
  if (!lower_bound)
    return std::nullopt;

  return *lower_bound;
}

std::optional<SymbolFile::ArrayInfo>
DWARFASTParser::ParseChildArrayInfo(const DWARFDIE &parent_die,
                                    const ExecutionContext *exe_ctx) {
  SymbolFile::ArrayInfo array_info;
  if (!parent_die)
    return std::nullopt;

  for (DWARFDIE die : parent_die.children()) {
    const dw_tag_t tag = die.Tag();
    if (tag != DW_TAG_subrange_type)
      continue;

    DWARFAttributes attributes = die.GetAttributes();
    if (attributes.Size() == 0)
      continue;

    std::optional<uint64_t> num_elements;
    std::optional<uint64_t> lower_bound = GetDefaultArrayLowerBound(parent_die);
    std::optional<uint64_t> upper_bound;

    auto evaluate_array_property =
        [&](const DWARFFormValue &form_value) -> std::optional<uint64_t> {
      llvm::Expected<uint64_t> value =
          EvaluateArrayPropertyAsUnsigned(form_value, parent_die, exe_ctx);
      if (!value) {
        LLDB_LOG_ERROR(GetLog(DWARFLog::DebugInfo), value.takeError(),
                       "failed to evaluate array property: {0}");
        return std::nullopt;
      }
      return value.get();
    };

    for (size_t i = 0; i < attributes.Size(); ++i) {
      const dw_attr_t attr = attributes.AttributeAtIndex(i);
      DWARFFormValue form_value;
      if (attributes.ExtractFormValueAtIndex(i, form_value)) {
        switch (attr) {
        case DW_AT_name:
          break;

        case DW_AT_count:
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
          } else
            num_elements = evaluate_array_property(form_value);
          break;

        case DW_AT_bit_stride:
          if (std::optional<uint64_t> bit_stride =
                  evaluate_array_property(form_value))
            array_info.bit_stride = *bit_stride;
          break;

        case DW_AT_byte_stride:
          if (std::optional<uint64_t> byte_stride =
                  evaluate_array_property(form_value))
            array_info.byte_stride = *byte_stride;
          break;

        case DW_AT_lower_bound:
          lower_bound = evaluate_array_property(form_value);
          break;

        case DW_AT_upper_bound:
          upper_bound = evaluate_array_property(form_value);
          break;

        default:
          break;
        }
      }
    }

    if (!num_elements || *num_elements == 0) {
      if (lower_bound && upper_bound && *upper_bound >= *lower_bound)
        num_elements = *upper_bound - *lower_bound + 1;
    }

    array_info.element_orders.push_back(num_elements);
  }
  return array_info;
}

Type *DWARFASTParser::GetTypeForDIE(const DWARFDIE &die) {
  if (!die)
    return nullptr;

  SymbolFileDWARF *dwarf = die.GetDWARF();
  if (!dwarf)
    return nullptr;

  DWARFAttributes attributes = die.GetAttributes();
  if (attributes.Size() == 0)
    return nullptr;

  DWARFFormValue type_die_form;
  for (size_t i = 0; i < attributes.Size(); ++i) {
    dw_attr_t attr = attributes.AttributeAtIndex(i);
    DWARFFormValue form_value;

    if (attr == DW_AT_type && attributes.ExtractFormValueAtIndex(i, form_value))
      return dwarf->ResolveTypeUID(form_value.Reference(), true);
  }

  return nullptr;
}

AccessType
DWARFASTParser::GetAccessTypeFromDWARF(uint32_t dwarf_accessibility) {
  switch (dwarf_accessibility) {
  case DW_ACCESS_public:
    return eAccessPublic;
  case DW_ACCESS_private:
    return eAccessPrivate;
  case DW_ACCESS_protected:
    return eAccessProtected;
  default:
    break;
  }
  return eAccessNone;
}
