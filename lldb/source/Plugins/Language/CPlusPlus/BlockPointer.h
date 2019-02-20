//===-- BlockPointer.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_BlockPointer_h_
#define liblldb_BlockPointer_h_

#include "lldb/lldb-forward.h"

namespace lldb_private {
namespace formatters {
bool BlockPointerSummaryProvider(ValueObject &, Stream &,
                                 const TypeSummaryOptions &);

SyntheticChildrenFrontEnd *
BlockPointerSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                     lldb::ValueObjectSP);
} // namespace formatters
} // namespace lldb_private

#endif // liblldb_BlockPointer_h_
