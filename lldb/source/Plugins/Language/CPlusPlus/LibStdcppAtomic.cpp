//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LibStdcpp.h"

#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/ValueObject/ValueObject.h"
#include "lldb/lldb-enumerations.h"
#include "llvm/Support/Error.h"

using namespace lldb;

namespace lldb_private::formatters {

class LibStdcppAtomicSyntheticFrontEnd final
    : public SyntheticChildrenFrontEnd {
public:
  explicit LibStdcppAtomicSyntheticFrontEnd(ValueObject &valobj_sp)
      : SyntheticChildrenFrontEnd(valobj_sp),
        m_inner_name(ConstString("Value")) {}

  llvm::Expected<uint32_t> CalculateNumChildren() final {
    if (!m_inner)
      return llvm::createStringError("invalide atomic ValueObject");
    return 1;
  }

  ValueObjectSP GetChildAtIndex(uint32_t idx) final {
    if (idx == 0 && m_inner)
      return m_inner->GetSP()->Clone(m_inner_name);

    return {};
  }

  lldb::ChildCacheState Update() final {
    if (ValueObjectSP value = ContainerFieldName(m_backend)) {
      // show the Type, instead of std::__atomic_base<Type>::__Type_type.
      value = value->Cast(value->GetCompilerType().GetCanonicalType());
      m_inner = value.get();
    }

    return lldb::ChildCacheState::eRefetch;
  }

  llvm::Expected<size_t> GetIndexOfChildWithName(ConstString name) final {
    if (name == m_inner_name)
      return 0;

    return llvm::createStringError("Type has no child named '%s'",
                                   name.AsCString());
  }

  static ValueObjectSP ContainerFieldName(ValueObject &backend_syn) {
    const ValueObjectSP non_synthetic = backend_syn.GetNonSyntheticValue();
    if (!non_synthetic)
      return {};
    ValueObject &backend = *non_synthetic;

    const CompilerType type = backend.GetCompilerType();
    if (!type || type.GetNumTemplateArguments() < 1)
      return {};

    const CompilerType first_type = type.GetTypeTemplateArgument(0);
    if (!first_type)
      return {};

    const lldb::BasicType basic_type = first_type.GetBasicTypeEnumeration();
    if (basic_type == eBasicTypeBool)
      return backend.GetChildAtNamePath({"_M_base", "_M_i"});

    const uint32_t float_mask = lldb::eTypeIsFloat | lldb::eTypeIsBuiltIn;
    if (first_type.GetTypeInfo() & float_mask) {
      // added float types specialization in c++17
      if (const auto child = backend.GetChildMemberWithName("_M_fp"))
        return child;

      return backend.GetChildMemberWithName("_M_i");
    }

    if (first_type.IsPointerType())
      return backend.GetChildAtNamePath({"_M_b", "_M_p"});

    const auto first_typename = first_type.GetDisplayTypeName().GetStringRef();
    if (first_typename.starts_with("std::shared_ptr<") ||
        first_typename.starts_with("std::weak_ptr<"))
      return backend.GetChildAtNamePath({"_M_impl", "_M_ptr"});

    return backend.GetChildMemberWithName("_M_i");
  }

private:
  ConstString m_inner_name;
  ValueObject *m_inner = nullptr;
};

SyntheticChildrenFrontEnd *
LibStdcppAtomicSyntheticFrontEndCreator(CXXSyntheticChildren * /*unused*/,
                                        const lldb::ValueObjectSP &valobj_sp) {
  if (!valobj_sp)
    return nullptr;

  const lldb::ValueObjectSP member =
      LibStdcppAtomicSyntheticFrontEnd::ContainerFieldName(*valobj_sp);
  if (!member)
    return nullptr;

  return new LibStdcppAtomicSyntheticFrontEnd(*valobj_sp);
}

bool LibStdcppAtomicSummaryProvider(ValueObject &valobj, Stream &stream,
                                    const TypeSummaryOptions &options) {

  if (const ValueObjectSP atomic_value =
          LibStdcppAtomicSyntheticFrontEnd::ContainerFieldName(valobj)) {
    std::string summary;
    if (atomic_value->GetSummaryAsCString(summary, options) &&
        !summary.empty()) {
      stream << summary;
      return true;
    }

    auto aparent = atomic_value->GetParent();
    if (aparent && aparent->GetName().GetStringRef() == "_M_impl") {
      return LibStdcppSmartPointerSummaryProvider(*aparent, stream, options,
                                                  /*is_atomic_child=*/true);
    }
  }

  return false;
}

} // namespace lldb_private::formatters
