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

bool fromJSON(const json::Value &value, FlatAperture &data, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("base", data.base) && o.mapOptional("size", data.size) &&
         o.mapOptional("pointer_unit", data.pointer_unit);
}

json::Value toJSON(const FlatAperture &data) {
  return json::Value(Object{{"base", data.base},
                            {"size", data.size},
                            {"pointer_unit", data.pointer_unit}});
}

bool fromJSON(const json::Value &value, AddressSpaceInfo &data, Path path) {
  ObjectMapper o(value, path);
  return o && o.map("name", data.name) && o.map("space_id", data.space_id) &&
         o.map("is_thread_specific", data.is_thread_specific) &&
         o.mapOptional("flat_aperture", data.flat_aperture);
}

json::Value toJSON(const AddressSpaceInfo &data) {
  Object object{{"name", data.name},
                {"space_id", data.space_id},
                {"is_thread_specific", data.is_thread_specific}};
  if (data.flat_aperture)
    object["flat_aperture"] = toJSON(*data.flat_aperture);
  return json::Value(std::move(object));
}

} // namespace lldb_private
