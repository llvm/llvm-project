//===-- Generic.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "Generic.h"
#include "LibStdcpp.h"
#include "MsvcStl.h"

lldb::ValueObjectSP lldb_private::formatters::GetDesugaredSmartPointerValue(
    ValueObject &ptr, ValueObject &container) {
  auto container_type = container.GetCompilerType().GetNonReferenceType();
  if (!container_type)
    return nullptr;

  auto arg = container_type.GetTypeTemplateArgument(0);
  if (!arg)
    // If there isn't enough debug info, use the pointer type as is
    return ptr.GetSP();

  return ptr.Cast(arg.GetPointerType());
}

lldb_private::SyntheticChildrenFrontEnd *
lldb_private::formatters::GenericSmartPointerSyntheticFrontEndCreator(
    CXXSyntheticChildren *children, lldb::ValueObjectSP valobj_sp) {
  if (auto *msvc = MsvcStlSmartPointerSyntheticFrontEndCreator(valobj_sp))
    return msvc;

  return LibStdcppSharedPtrSyntheticFrontEndCreator(children, valobj_sp);
}

bool lldb_private::formatters::GenericSmartPointerSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  return MsvcStlSmartPointerSummaryProvider(valobj, stream, options) ||
         LibStdcppSmartPointerSummaryProvider(valobj, stream, options);
}
