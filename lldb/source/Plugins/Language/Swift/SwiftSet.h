//===-- SwiftSet.h ----------------------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2015 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See http://swift.org/LICENSE.txt for license information
// See http://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SwiftSet_h_
#define liblldb_SwiftSet_h_

#include "lldb/lldb-forward.h"

#include "SwiftHashedContainer.h"
#include "lldb/Core/ConstString.h"
#include "lldb/DataFormatters/FormatClasses.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Target/Target.h"

namespace lldb_private {
namespace formatters {
namespace swift {
bool Set_SummaryProvider(ValueObject &valobj, Stream &stream,
                         const TypeSummaryOptions &options);

SyntheticChildrenFrontEnd *SetSyntheticFrontEndCreator(CXXSyntheticChildren *,
                                                       lldb::ValueObjectSP);
}
}
}

#endif // liblldb_SwiftSet_h_
