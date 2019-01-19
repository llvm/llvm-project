//===-- CxxStringTypes.h ----------------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CxxStringTypes_h_
#define liblldb_CxxStringTypes_h_

#include "lldb/Core/ValueObject.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/Utility/Stream.h"

namespace lldb_private {
namespace formatters {
bool Char16StringSummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &options); // char16_t* and unichar*

bool Char32StringSummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &options); // char32_t*

bool WCharStringSummaryProvider(ValueObject &valobj, Stream &stream,
                                const TypeSummaryOptions &options); // wchar_t*

bool Char16SummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &options); // char16_t and unichar

bool Char32SummaryProvider(ValueObject &valobj, Stream &stream,
                           const TypeSummaryOptions &options); // char32_t

bool WCharSummaryProvider(ValueObject &valobj, Stream &stream,
                          const TypeSummaryOptions &options); // wchar_t

} // namespace formatters
} // namespace lldb_private

#endif // liblldb_CxxStringTypes_h_
