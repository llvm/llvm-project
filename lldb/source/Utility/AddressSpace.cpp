//===-- AddressSpace.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/AddressSpace.h"

using namespace llvm;
using namespace llvm::json;

namespace lldb_private {

bool fromJSON(const json::Value &value, AddressSpaceInfo &data, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("name", data.name) && o.map("value", data.value) &&
         o.map("is_thread_specific", data.is_thread_specific);
}

json::Value toJSON(const AddressSpaceInfo &data) {
  return json::Value(Object{{"name", data.name},
                            {"value", data.value},
                            {"is_thread_specific", data.is_thread_specific}});
}

} // namespace lldb_private
