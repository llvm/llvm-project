//===-- SwiftMetatype.cpp -------------------------------------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2023 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "SwiftMetatype.h"
#include "Plugins/LanguageRuntime/Swift/SwiftLanguageRuntime.h"
#include "lldb/Symbol/CompilerType.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace lldb_private::formatters::swift;

bool lldb_private::formatters::swift::SwiftMetatype_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  ConstString name;
  lldb::addr_t metadata_ptr = valobj.GetPointerValue();
  if (metadata_ptr == LLDB_INVALID_ADDRESS || metadata_ptr == 0) {
    CompilerType compiler_metatype_type(valobj.GetCompilerType());
    CompilerType instancetype =
      TypeSystemSwift::GetInstanceType(compiler_metatype_type);
    name = instancetype.GetDisplayTypeName();
  } else if (CompilerType meta_type = valobj.GetCompilerType()) {
    name = meta_type.GetDisplayTypeName();
  }
  if (!name)
    return false;
  stream << name;
  return true;
}
