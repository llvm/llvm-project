//===-- MsvcStlAtomic.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MsvcStl.h"

#include "lldb/DataFormatters/TypeSynthetic.h"
#include "llvm/Support/ErrorExtras.h"

using namespace lldb;
using namespace lldb_private;

namespace lldb_private {
namespace formatters {

class MsvcStlAtomicSyntheticFrontEnd : public SyntheticChildrenFrontEnd {
public:
  MsvcStlAtomicSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp,
                                 bool is_smart_ptr);

  llvm::Expected<uint32_t> CalculateNumChildren() override;

  lldb::ValueObjectSP GetChildAtIndex(uint32_t idx) override;

  lldb::ChildCacheState Update() override;

  llvm::Expected<size_t> GetIndexOfChildWithName(llvm::StringRef name) override;

private:
  ValueObjectSP m_storage;
  bool m_is_smart_ptr = false;
};

} // namespace formatters
} // namespace lldb_private

static ValueObjectSP ValueOfAtomic(ValueObject &valobj) {
  ValueObjectSP storage_sp = valobj.GetChildMemberWithName("_Storage");
  if (!storage_sp)
    return nullptr;

  CompilerType backend_type = valobj.GetCompilerType();
  if (!backend_type)
    return nullptr;

  CompilerType element_type = backend_type.GetTypeTemplateArgument(0);
  if (!element_type) {
    // PDB doesn't have info about templates, so use value_type which equals T.
    element_type = backend_type.GetDirectNestedTypeWithName("value_type");

    if (!element_type)
      return nullptr;
  }

  return valobj.Cast(element_type);
}

lldb_private::formatters::MsvcStlAtomicSyntheticFrontEnd::
    MsvcStlAtomicSyntheticFrontEnd(lldb::ValueObjectSP valobj_sp,
                                   bool is_smart_ptr)
    : SyntheticChildrenFrontEnd(*valobj_sp), m_is_smart_ptr(is_smart_ptr) {}

llvm::Expected<uint32_t> lldb_private::formatters::
    MsvcStlAtomicSyntheticFrontEnd::CalculateNumChildren() {
  return m_storage ? 1 : 0;
}

lldb::ValueObjectSP
lldb_private::formatters::MsvcStlAtomicSyntheticFrontEnd::GetChildAtIndex(
    uint32_t idx) {
  if (idx == 0 && m_storage)
    return m_storage;
  return nullptr;
}

lldb::ChildCacheState
lldb_private::formatters::MsvcStlAtomicSyntheticFrontEnd::Update() {
  m_storage.reset();

  if (m_is_smart_ptr) {
    ValueObjectSP ptr = m_backend.GetChildMemberWithName("_Ptr");
    if (!ptr)
      return lldb::eRefetch;
    m_storage = ValueOfAtomic(*ptr);
  } else {
    m_storage = ValueOfAtomic(m_backend);
  }

  if (m_storage)
    m_storage = m_storage->Clone("Value");

  return lldb::eRefetch;
}

llvm::Expected<size_t>
lldb_private::formatters::MsvcStlAtomicSyntheticFrontEnd::
    GetIndexOfChildWithName(llvm::StringRef name) {
  if (name == "Value" || (m_is_smart_ptr && name == "$$dereference$$"))
    return 0;
  return llvm::createStringErrorV("type has no child named '{0}'", name);
}

lldb_private::SyntheticChildrenFrontEnd *
lldb_private::formatters::MsvcStlAtomicSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  bool is_smart_ptr = false;
  if (valobj_sp && IsMsvcStlAtomic(*valobj_sp, &is_smart_ptr))
    return new MsvcStlAtomicSyntheticFrontEnd(valobj_sp, is_smart_ptr);
  return nullptr;
}

static CompilerType FindLockedPointeeType(Target &target,
                                          const CompilerType &locked_ptr_type) {
  ConstString locked_name = locked_ptr_type.GetTypeName();
  llvm::StringRef pointee_name = locked_name;
  if (!pointee_name.consume_front("std::_Locked_pointer<"))
    return {};
  if (!pointee_name.consume_back(">"))
    return {};

  TypeQuery query(pointee_name, TypeQueryOptions::e_find_one);
  TypeResults results;
  target.GetImages().FindTypes(/*search_first=*/nullptr, query, results);
  if (TypeSP type_sp = results.GetFirstType())
    return type_sp->GetFullCompilerType();
  return {};
}

static bool AtomicSharedPtrSummaryProvider(ValueObject &ptr,
                                           ValueObject &rep_wrap,
                                           Stream &stream,
                                           const TypeSummaryOptions &options) {
  // rep_wrap: _Locked_pointer<_Ref_count_base> {
  //   static constexpr uintptr_t _Lock_mask = 3;
  //   static constexpr uintptr_t _Ptr_value_mask = ~_Lock_mask;
  //
  //   atomic<uintptr_t> Storage_;
  // }
  ValueObjectSP rep_storage_sp = rep_wrap.GetChildAtIndex(0);
  if (!rep_storage_sp)
    return false;
  ValueObjectSP rep_ptr_sp = ValueOfAtomic(*rep_storage_sp);
  if (!rep_ptr_sp)
    return false;

  bool ok = false;
  uint64_t rep_ptr_val = rep_ptr_sp->GetValueAsUnsigned(0, &ok);
  if (!ok)
    return false;

  CompilerType rep_type = rep_wrap.GetCompilerType();
  if (!rep_type)
    return false;
  CompilerDecl ptr_value_decl =
      rep_type.GetStaticFieldWithName("_Ptr_value_mask");
  if (!ptr_value_decl)
    return false;
  uint64_t rep_ptr_mask = ptr_value_decl.GetConstantValue().ULongLong(~(3ULL));
  rep_ptr_val &= rep_ptr_mask;

  CompilerType ref_count_base = rep_type.GetTypeTemplateArgument(0);
  if (!ref_count_base) {
    // PDB: Resolve search for the type by name, as it doesn't know about
    // templates. Unfortunately, there's no typedef inside the type.
    TargetSP target = rep_wrap.GetTargetSP();
    if (!target)
      return false;
    ref_count_base = FindLockedPointeeType(*target, rep_type);
    if (!ref_count_base)
      return false;
  }

  ValueObjectSP rep = ValueObject::CreateValueObjectFromAddress(
      "", rep_ptr_val, rep_wrap.GetExecutionContextRef(), ref_count_base);
  if (!rep)
    return false;

  return formatters::DumpMsvcStlSmartPointerSummary(ptr, *rep, rep_ptr_val,
                                                    stream, options);
}

bool lldb_private::formatters::MsvcStlAtomicSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  ValueObjectSP non_synth = valobj.GetNonSyntheticValue();
  if (!non_synth)
    return false;
  ValueObjectSP value_sp = valobj.GetChildAtIndex(0);
  if (!value_sp)
    return false;

  ValueObjectSP smart_ptr_rep = non_synth->GetChildMemberWithName("_Repptr");
  if (smart_ptr_rep)
    return AtomicSharedPtrSummaryProvider(*value_sp, *smart_ptr_rep, stream,
                                          options);

  std::string summary;
  if (value_sp->GetSummaryAsCString(summary, options) && !summary.empty()) {
    stream << summary;
    return true;
  }
  return false;
}

bool lldb_private::formatters::IsMsvcStlAtomic(ValueObject &valobj,
                                               bool *is_smart_ptr) {
  if (auto valobj_sp = valobj.GetNonSyntheticValue()) {
    bool has_storage = valobj_sp->GetChildMemberWithName("_Storage") != nullptr;
    bool has_smart =
        !has_storage && valobj_sp->GetChildMemberWithName("_Ptr") != nullptr;
    if (is_smart_ptr)
      *is_smart_ptr = has_smart;

    return has_storage || has_smart;
  }
  return false;
}
