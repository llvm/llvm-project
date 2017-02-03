//===-- SwiftMetatype.h -----------------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SwiftMetatype_h_
#define liblldb_SwiftMetatype_h_

#include "lldb/lldb-forward.h"
#include "lldb/lldb-private.h"

namespace lldb_private {
namespace formatters {
namespace swift {
bool SwiftMetatype_SummaryProvider(ValueObject &valobj, Stream &stream,
                                   const TypeSummaryOptions &options);
}
}
}

#endif // liblldb_SwiftMetatype_h_
