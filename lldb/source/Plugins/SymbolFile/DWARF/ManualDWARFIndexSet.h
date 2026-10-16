//===-- ManualDWARFIndexSet.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_MANUALDWARFINDEXSET_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_MANUALDWARFINDEXSET_H

#include "Plugins/SymbolFile/DWARF/NameToDIE.h"
#include "lldb/Utility/DataEncoder.h"
#include "lldb/Utility/DataExtractor.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>

namespace lldb_private::plugin::dwarf {

template <typename T> struct IndexSet {
  T function_basenames;
  T function_fullnames;
  T function_methods;
  T function_selectors;
  T objc_class_selectors;
  T globals;
  T types;
  T namespaces;

  static std::array<T(IndexSet::*), 8> Indices() {
    return {&IndexSet::function_basenames,
            &IndexSet::function_fullnames,
            &IndexSet::function_methods,
            &IndexSet::function_selectors,
            &IndexSet::objc_class_selectors,
            &IndexSet::globals,
            &IndexSet::types,
            &IndexSet::namespaces};
  }

  friend bool operator==(const IndexSet &lhs, const IndexSet &rhs) {
    return llvm::all_of(Indices(), [&lhs, &rhs](T(IndexSet::*index)) {
      return lhs.*index == rhs.*index;
    });
  }
};

std::optional<IndexSet<NameToDIE>> DecodeIndexSet(const DataExtractor &data,
                                                  lldb::offset_t *offset_ptr);
void EncodeIndexSet(const IndexSet<NameToDIE> &set, DataEncoder &encoder);

} // namespace lldb_private::plugin::dwarf

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_MANUALDWARFINDEXSET_H
