//===-- JavaFormatterFunctions.h---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_JavaFormatterFunctions_h_
#define liblldb_JavaFormatterFunctions_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-forward.h"

namespace lldb_private {
namespace formatters {

bool JavaStringSummaryProvider(ValueObject &valobj, Stream &stream,
                               const TypeSummaryOptions &options);

bool JavaArraySummaryProvider(ValueObject &valobj, Stream &stream,
                              const TypeSummaryOptions &options);

SyntheticChildrenFrontEnd *
JavaArraySyntheticFrontEndCreator(CXXSyntheticChildren *,
                                  lldb::ValueObjectSP valobj_sp);

} // namespace formatters
} // namespace lldb_private

#endif // liblldb_JavaFormatterFunctions_h_
