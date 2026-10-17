//===----------------------------------------------------------------------===//
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
  return o && o.map("name", data.name) && o.map("space_id", data.space_id) &&
         o.map("is_thread_specific", data.is_thread_specific) &&
         o.mapOptional("dwarf_address_class", data.dwarf_address_class) &&
         o.mapOptional("flat_base", data.flat_base) &&
         o.mapOptional("flat_size", data.flat_size) &&
         o.mapOptional("pointer_unit", data.pointer_unit);
}

json::Value toJSON(const AddressSpaceInfo &data) {
  Object object{{"name", data.name},
                {"space_id", data.space_id},
                {"is_thread_specific", data.is_thread_specific}};
  if (data.dwarf_address_class)
    object["dwarf_address_class"] = *data.dwarf_address_class;
  if (data.flat_base) {
    object["flat_base"] = *data.flat_base;
    object["flat_size"] = data.flat_size;
    object["pointer_unit"] = data.pointer_unit;
  }
  return json::Value(std::move(object));
}

} // namespace lldb_private
