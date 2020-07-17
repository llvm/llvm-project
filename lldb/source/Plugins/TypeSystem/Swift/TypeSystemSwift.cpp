//===-- TypeSystemSwift.cpp ==---------------------------------------------===//
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

#include "Plugins/TypeSystem/Swift/TypeSystemSwift.h"
#include <lldb/lldb-enumerations.h>

using namespace lldb;
using namespace lldb_private;

bool TypeSystemSwift::IsFloatingPointType(opaque_compiler_type_t type,
                                          uint32_t &count, bool &is_complex) {
  count = 0;
  is_complex = false;
  if (GetTypeInfo(type, nullptr) & eTypeIsFloat) {
    count = 1;
    return true;
  }
  return false;
}

bool TypeSystemSwift::IsIntegerType(opaque_compiler_type_t type,
                                    bool &is_signed) {
  return (GetTypeInfo(type, nullptr) & eTypeIsInteger);
}

bool TypeSystemSwift::IsScalarType(opaque_compiler_type_t type) {
  if (!type)
    return false;

  return (GetTypeInfo(type, nullptr) & eTypeIsScalar) != 0;
}

bool TypeSystemSwift::ShouldTreatScalarValueAsAddress(
    opaque_compiler_type_t type) {
  return Flags(GetTypeInfo(type, nullptr))
      .AnySet(eTypeInstanceIsPointer | eTypeIsReference);
}
