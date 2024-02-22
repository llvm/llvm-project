//===-- SwiftBasicTypes.cpp -------------------------------------*- C++ -*-===//
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

#include "SwiftBasicTypes.h"

#include "lldb/Core/ValueObject.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace lldb_private::formatters::swift;

lldb::ValueObjectSP lldb_private::formatters::swift::
    SwiftBasicTypeSyntheticFrontEnd::GetSyntheticValue() {
  return m_backend.GetChildAtIndex(0, true);
}

SyntheticChildrenFrontEnd *
lldb_private::formatters::swift::SwiftBasicTypeSyntheticFrontEndCreator(
    CXXSyntheticChildren *, lldb::ValueObjectSP valobj_sp) {
  if (valobj_sp)
    return new SwiftBasicTypeSyntheticFrontEnd(*valobj_sp);
  return nullptr;
}
