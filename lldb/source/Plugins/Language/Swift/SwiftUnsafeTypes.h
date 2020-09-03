//===-- SwiftUnsafeTypes.h --------------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2020 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SwiftUnsafeTypes_h_
#define liblldb_SwiftUnsafeTypes_h_

#include "lldb/Core/ValueObject.h"
#include "lldb/Utility/Stream.h"

namespace lldb_private {
namespace formatters {
namespace swift {

bool UnsafeTypeSummaryProvider(ValueObject &valobj, Stream &stream,
                               const TypeSummaryOptions &);

SyntheticChildrenFrontEnd *
UnsafeTypeSyntheticFrontEndCreator(CXXSyntheticChildren *, lldb::ValueObjectSP);

}; // namespace swift
}; // namespace formatters
}; // namespace lldb_private

#endif
