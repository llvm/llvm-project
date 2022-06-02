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

#include "lldb/Core/ValueObject.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Target/StackFrame.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::dwarf;

llvm::Optional<SymbolFile::ArrayInfo>
DWARFASTParser::ParseChildArrayInfo(const DWARFDIE &parent_die,
                                    const ExecutionContext *exe_ctx) {
  SymbolFile::ArrayInfo array_info;
  if (!parent_die)
    return llvm::None;

  for (DWARFDIE die : parent_die.children()) {
    const dw_tag_t tag = die.Tag();
    if (tag != DW_TAG_subrange_type)
      continue;

    DWARFAttributes attributes;
    const size_t num_child_attributes = die.GetAttributes(attributes);
    if (num_child_attributes > 0) {
      uint64_t num_elements = 0;
      uint64_t lower_bound = 0;
      uint64_t upper_bound = 0;
      bool upper_bound_valid = false;
      uint32_t i;
      for (i = 0; i < num_child_attributes; ++i) {
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
              num_elements = form_value.Unsigned();
            break;

          case DW_AT_bit_stride:
            array_info.bit_stride = form_value.Unsigned();
            break;

          case DW_AT_byte_stride:
            array_info.byte_stride = form_value.Unsigned();
            break;

          case DW_AT_lower_bound:
            lower_bound = form_value.Unsigned();
            break;

          case DW_AT_upper_bound:
            upper_bound_valid = true;
            upper_bound = form_value.Unsigned();
            break;

          default:
            break;
          }
        }
      }

      if (num_elements == 0) {
        if (upper_bound_valid && upper_bound >= lower_bound)
          num_elements = upper_bound - lower_bound + 1;
      }

      array_info.element_orders.push_back(num_elements);
    }
  }
  return array_info;
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
