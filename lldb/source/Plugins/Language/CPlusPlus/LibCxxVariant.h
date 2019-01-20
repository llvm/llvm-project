//===-- LibCxxVariant.h -------------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_LibCxxVariant_h_
#define liblldb_LibCxxVariant_h_

#include "lldb/Core/ValueObject.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/Utility/Stream.h"

namespace lldb_private {
namespace formatters {
bool LibcxxVariantSummaryProvider(
    ValueObject &valobj, Stream &stream,
    const TypeSummaryOptions &options); // libc++ std::variant<>

SyntheticChildrenFrontEnd *LibcxxVariantFrontEndCreator(CXXSyntheticChildren *,
                                                        lldb::ValueObjectSP);

} // namespace formatters
} // namespace lldb_private

#endif // liblldb_LibCxxVariant_h_
