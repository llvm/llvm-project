//===-- MsvcStlVariant.cpp-------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MsvcStl.h"
#include "lldb/DataFormatters/FormattersHelpers.h"
#include "lldb/Symbol/CompilerType.h"
#include <optional>

using namespace lldb;
using namespace lldb_private;

namespace {

// A variant when using DWARF looks as follows:
// (lldb) fr v -R v1
// (std::variant<int, double, char>) v1 = {
//   std::_SMF_control<std::_Variant_base<int, double, char>, int, double, char>
//   = {
//     std::_Variant_storage<int, double, char> = {
//        = {
//         _Head = 0
//         _Tail = {
//            = {
//             _Head = 2
//             _Tail = {
//                = {
//                 _Head = '\0'
//                 _Tail = {}
//               }
//             }
//           }
//         }
//       }
//     }
//     _Which = '\x01'
//   }
// }
//
// ... when using PDB, it looks like this:
// (lldb) fr v -R v1
// (std::variant<int,double,char>) v1 = {
//   std::_Variant_base<int,double,char> = {
//     std::_Variant_storage_<1,int,double,char> = {
//       _Head = 0
//       _Tail = {
//         _Head = 2
//         _Tail = {
//           _Head = '\0'
//           _Tail = {}
//         }
//       }
//     }
//     _Which = '\x01'
//   }
// }

ValueObjectSP GetStorageAtIndex(ValueObject &valobj, size_t index) {
  // PDB flattens the members on unions to the parent
  if (valobj.GetCompilerType().GetNumFields() == 2)
    return valobj.GetChildAtIndex(index);

  // DWARF keeps the union
  ValueObjectSP union_sp = valobj.GetChildAtIndex(0);
  if (!union_sp)
    return nullptr;
  return union_sp->GetChildAtIndex(index);
}

ValueObjectSP GetHead(ValueObject &valobj) {
  return GetStorageAtIndex(valobj, 0);
}
ValueObjectSP GetTail(ValueObject &valobj) {
  return GetStorageAtIndex(valobj, 1);
}

std::optional<int64_t> GetIndexValue(ValueObject &valobj) {
  ValueObjectSP index_sp = valobj.GetChildMemberWithName("_Which");
  if (!index_sp)
    return std::nullopt;

  return {index_sp->GetValueAsSigned(-1)};
}

ValueObjectSP GetNthStorage(ValueObject &outer, int64_t index) {
  ValueObjectSP container_sp = outer.GetSP();

  // When using DWARF, we need to find the std::_Variant_storage base class.
  // There, the top level type doesn't have any fields.
  if (container_sp->GetCompilerType().GetNumFields() == 0) {
    // -> std::_SMF_control (typedef to std::_Variant_base)
    container_sp = container_sp->GetChildAtIndex(0);
    if (!container_sp)
      return nullptr;
    // -> std::_Variant_storage
    container_sp = container_sp->GetChildAtIndex(0);
    if (!container_sp)
      return nullptr;
  }

  for (int64_t i = 0; i < index; i++) {
    container_sp = GetTail(*container_sp);
    if (!container_sp)
      return nullptr;
  }
  return container_sp;
}

} // namespace

bool formatters::IsMsvcStlVariant(ValueObject &valobj) {
  if (auto valobj_sp = valobj.GetNonSyntheticValue()) {
    return valobj_sp->GetChildMemberWithName("_Which") != nullptr;
  }
  return false;
}

bool formatters::MsvcStlVariantSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  ValueObjectSP valobj_sp(valobj.GetNonSyntheticValue());
  if (!valobj_sp)
    return false;

  auto index = GetIndexValue(*valobj_sp);
  if (!index)
    return false;

  if (*index < 0) {
    stream.Printf(" No Value");
    return true;
  }

  ValueObjectSP storage = GetNthStorage(*valobj_sp, *index);
  if (!storage)
    return false;
  CompilerType storage_type = storage->GetCompilerType();
  if (!storage_type)
    return false;
  // With DWARF, it's a typedef, with PDB, it's not
  if (storage_type.IsTypedefType())
    storage_type = storage_type.GetTypedefedType();

  CompilerType active_type = storage_type.GetTypeTemplateArgument(1, true);
  if (!active_type) {
    // not enough debug info, try the type of _Head
    ValueObjectSP head_sp = GetHead(*storage);
    if (!head_sp)
      return false;
    active_type = head_sp->GetCompilerType();
    if (!active_type)
      return false;
  }

  stream << " Active Type = " << active_type.GetDisplayTypeName() << " ";
  return true;
}

namespace {
class VariantFrontEnd : public SyntheticChildrenFrontEnd {
public:
  VariantFrontEnd(ValueObject &valobj) : SyntheticChildrenFrontEnd(valobj) {
    Update();
  }

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) override {
    auto optional_idx = formatters::ExtractIndexFromString(name.GetCString());
    if (!optional_idx) {
      return llvm::createStringError("Type has no child named '%s'",
                                     name.AsCString());
    }
    return *optional_idx;
  }

  lldb::ChildCacheState Update() override;
  llvm::Expected<uint32_t> CalculateNumChildren() override { return m_size; }
  ValueObjectSP GetChildAtIndex(uint32_t idx) override;

private:
  size_t m_size = 0;
};
} // namespace

lldb::ChildCacheState VariantFrontEnd::Update() {
  m_size = 0;

  auto index = GetIndexValue(m_backend);
  if (index && *index >= 0)
    m_size = 1;

  return lldb::ChildCacheState::eRefetch;
}

ValueObjectSP VariantFrontEnd::GetChildAtIndex(uint32_t idx) {
  if (idx >= m_size)
    return nullptr;

  auto index = GetIndexValue(m_backend);
  if (!index)
    return nullptr;

  ValueObjectSP storage_sp = GetNthStorage(m_backend, *index);
  if (!storage_sp)
    return nullptr;

  ValueObjectSP head_sp = GetHead(*storage_sp);
  if (!head_sp)
    return nullptr;

  return head_sp->Clone(ConstString("Value"));
}

SyntheticChildrenFrontEnd *formatters::MsvcStlVariantSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (valobj_sp)
    return new VariantFrontEnd(*valobj_sp);
  return nullptr;
}
