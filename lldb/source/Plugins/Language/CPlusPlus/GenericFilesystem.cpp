//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Generic.h"

using namespace lldb;
using namespace lldb_private;

bool lldb_private::formatters::GenericFilesystemPathSummaryProvider(
    ValueObject &valobj, Stream &stream, const TypeSummaryOptions &) {
  ValueObjectSP path_sp = valobj.GetChildMemberWithName("_Text");
  if (!path_sp)
    path_sp = valobj.GetChildMemberWithName("_M_pathname");
  if (!path_sp)
    return false;

  if (const char *summary = path_sp->GetSummaryAsCString()) {
    stream << summary;
    return true;
  }
  return false;
}
