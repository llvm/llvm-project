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

std::optional<uint64_t> GetWCharByteSize(ValueObject &valobj);

/// Print a summary for a string buffer to \a stream.
///
/// \param[in] stream
///     The output stream to print the summary to.
///
/// \param[in] summary_options
///     Options for printing the string contents. This function respects the
///     capping.
///
/// \param[in] location_sp
///     ValueObject of a pointer to the string being printed.
///
/// \param[in] size
///     The size of the buffer pointed to by \a location_sp.
///
/// \param[in] prefix_token
///     A prefix before the double quotes (e.g. 'u' results in u"...").
///
/// \return
///     Returns whether the string buffer was successfully printed.
template <StringPrinter::StringElementType element_type>
bool StringBufferSummaryProvider(Stream &stream,
                                 const TypeSummaryOptions &summary_options,
                                 lldb::ValueObjectSP location_sp, uint64_t size,
                                 std::string prefix_token);

} // namespace formatters
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_LANGUAGE_CPLUSPLUS_CXXSTRINGTYPES_H
