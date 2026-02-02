//===-- SystemValueTypes.h --------------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2026 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SystemValueTypes_h_
#define liblldb_SystemValueTypes_h_

#include "lldb/lldb-forward.h"

#include "lldb/DataFormatters/TypeSummary.h"

namespace lldb_private {
namespace formatters {
namespace swift {
bool FilePath_SummaryProvider(ValueObject &valobj, Stream &stream,
                              const TypeSummaryOptions &options);

bool SystemChar_SummaryProvider(ValueObject &valobj, Stream &stream,
                                const TypeSummaryOptions &options);
}
}
}

#endif // liblldb_SystemValueTypes_h_
