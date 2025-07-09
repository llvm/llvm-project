//===-- Generic.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#include "Generic.h"

lldb::ValueObjectSP lldb_private::formatters::GetDesugaredSmartPointerValue(
    ValueObject &ptr, ValueObject &container) {
  auto container_type = container.GetCompilerType().GetNonReferenceType();
  if (!container_type)
    return nullptr;

  auto arg = container_type.GetTypeTemplateArgument(0);
  if (!arg)
    return nullptr;

  return ptr.Cast(arg.GetPointerType());
}
