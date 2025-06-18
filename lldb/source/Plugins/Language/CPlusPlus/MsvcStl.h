//===-- MsvcStl.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGE_CPLUSPLUS_MSVCSTL_H
#define LLDB_SOURCE_PLUGINS_LANGUAGE_CPLUSPLUS_MSVCSTL_H

#include "lldb/DataFormatters/StringPrinter.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/Utility/Stream.h"
#include "lldb/ValueObject/ValueObject.h"

namespace lldb_private {
namespace formatters {

bool IsMsvcStlStringType(ValueObject &valobj);

template <StringPrinter::StringElementType element_type>
bool MsvcStlStringSummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions
        &summary_options); // VC 2015+ std::string,u8string,u16string,u32string

bool MsvcStlWStringSummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &options); // VC 2015+ std::wstring

} // namespace formatters
} // namespace lldb_private

#endif
