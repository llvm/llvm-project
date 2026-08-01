//===-- DynamicArray.h ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGE_FORTRAN_DYNAMICARRAY_H
#define LLDB_SOURCE_PLUGINS_LANGUAGE_FORTRAN_DYNAMICARRAY_H

#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/Utility/Stream.h"
#include "lldb/ValueObject/ValueObject.h"

namespace lldb_private {
namespace formatters {

SyntheticChildrenFrontEnd *
FortranDynamicArraySyntheticFrontEndCreator(CXXSyntheticChildren *,
                                            lldb::ValueObjectSP);

} // namespace formatters
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_LANGUAGE_FORTRAN_DYNAMICARRAY_H