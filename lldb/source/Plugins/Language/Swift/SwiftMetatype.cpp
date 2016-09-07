//===-- SwiftMetatype.cpp ---------------------------------------*- C++ -*-===//
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

#include "SwiftMetatype.h"
#include "lldb/Core/Mangled.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/SwiftLanguageRuntime.h"

#include "swift/AST/Type.h"
#include "swift/AST/Types.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace lldb_private::formatters::swift;

bool lldb_private::formatters::swift::SwiftMetatype_SummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &options) {
  lldb::addr_t metadata_ptr = valobj.GetValueAsUnsigned(LLDB_INVALID_ADDRESS);
  if (metadata_ptr == LLDB_INVALID_ADDRESS || metadata_ptr == 0) {
    CompilerType compiler_metatype_type(valobj.GetCompilerType());
    CompilerType instancetype(compiler_metatype_type.GetInstanceType());
    const char *ptr = instancetype.GetDisplayTypeName().AsCString(nullptr);
    if (ptr && *ptr) {
      stream.Printf("%s", ptr);
      return true;
    }
  } else {
    auto swift_runtime = valobj.GetProcessSP()->GetSwiftLanguageRuntime();
    if (!swift_runtime)
      return false;
    SwiftLanguageRuntime::MetadataPromiseSP metadata_promise_sp =
        swift_runtime->GetMetadataPromise(metadata_ptr);
    if (CompilerType resolved_type =
            metadata_promise_sp->FulfillTypePromise()) {
      stream.Printf("%s", resolved_type.GetDisplayTypeName().AsCString());
      return true;
    }
  }
  return false;
}
