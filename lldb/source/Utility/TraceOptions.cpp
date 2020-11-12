//===-- TraceOptions.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/TraceOptions.h"

using namespace lldb_private;

namespace llvm {
namespace json {

bool fromJSON(const Value &value, TraceTypeInfo &info, Path path) {
  ObjectMapper o(value, path);
  if (!o)
    return false;
  o.map("description", info.description);
  return o.map("name", info.name);
}

} // namespace json
} // namespace llvm
