//===-- CxxStringTypes.h ----------------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGE_CPLUSPLUS_CXXSTRINGTYPES_H
#define LLDB_SOURCE_PLUGINS_LANGUAGE_CPLUSPLUS_CXXSTRINGTYPES_H

#include "lldb/DataFormatters/StringPrinter.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/Utility/Stream.h"
#include "lldb/ValueObject/ValueObject.h"

namespace lldb_private {
namespace formatters {

template <StringPrinter::StringElementType element_type>
bool CharTStringSummaryProvider(ValueObject &valobj, Stream &stream);

bool Char8StringSummaryProvider(ValueObject &valobj, Stream &stream,
                                const TypeSummaryOptions &options); // char8_t*

bool Char16StringSummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &options); // char16_t* and unichar*

bool Char32StringSummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &options); // char32_t*

bool WCharStringSummaryProvider(ValueObject &valobj, Stream &stream,
                                const TypeSummaryOptions &options); // wchar_t*

bool Char8SummaryProvider(ValueObject &valobj, Stream &stream,
                          const TypeSummaryOptions &options); // char8_t

bool Char16SummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &options); // char16_t and unichar

bool Char32SummaryProvider(ValueObject &valobj, Stream &stream,
                           const TypeSummaryOptions &options); // char32_t

bool WCharSummaryProvider(ValueObject &valobj, Stream &stream,
                          const TypeSummaryOptions &options); // wchar_t

template <StringPrinter::StringElementType element_type>
bool StdStringSummaryProviderImpl(ValueObject &valobj, Stream &stream,
                                  const TypeSummaryOptions &summary_options,
                                  std::string prefix_token,
                                  lldb::ValueObjectSP location_sp,
                                  uint64_t size);

bool StdStringSummaryProvider(ValueObject &valobj, Stream &stream,
                              const TypeSummaryOptions &summary_options,
                              std::string prefix_token,
                              lldb::ValueObjectSP location_sp, uint64_t size);
bool StdU8StringSummaryProvider(ValueObject &valobj, Stream &stream,
                                const TypeSummaryOptions &summary_options,
                                std::string prefix_token,
                                lldb::ValueObjectSP location_sp, uint64_t size);
bool StdU16StringSummaryProvider(ValueObject &valobj, Stream &stream,
                                 const TypeSummaryOptions &summary_options,
                                 std::string prefix_token,
                                 lldb::ValueObjectSP location_sp,
                                 uint64_t size);
bool StdU32StringSummaryProvider(ValueObject &valobj, Stream &stream,
                                 const TypeSummaryOptions &summary_options,
                                 std::string prefix_token,
                                 lldb::ValueObjectSP location_sp,
                                 uint64_t size);
bool StdWStringSummaryProvider(ValueObject &valobj, Stream &stream,
                               const TypeSummaryOptions &summary_options,
                               std::string prefix_token,
                               lldb::ValueObjectSP location_sp, uint64_t size);

} // namespace formatters
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_LANGUAGE_CPLUSPLUS_CXXSTRINGTYPES_H
