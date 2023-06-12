//===- StringTable.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DWARFLINKERPARALLEL_STRINGTABLE_H
#define LLVM_DWARFLINKERPARALLEL_STRINGTABLE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/DWARFLinkerParallel/StringPool.h"

namespace llvm {
namespace dwarflinker_parallel {

using StringsVector = SmallVector<StringEntry *>;

/// This class prepares strings for emission into .debug_str table:
/// translates string if necessary, assigns index and offset, keeps in order.
class StringTable {
public:
  StringTable(StringPool &Strings,
              std::function<StringRef(StringRef)> StringsTranslator)
      : Strings(Strings), StringsTranslator(StringsTranslator) {}
  ~StringTable() {}

  /// Add string to the vector of strings which should be emitted.
  /// Translate input string if neccessary, assign index and offset.
  /// \returns updated string entry.
  StringEntry *add(StringEntry *String) {
    // Translate string if necessary.
    if (StringsTranslator)
      String = Strings.insert(StringsTranslator(String->first())).first;

    // Store String for emission and assign index and offset.
    if (String->getValue() == nullptr) {
      DwarfStringPoolEntry *NewEntry =
          Strings.getAllocatorRef().Allocate<DwarfStringPoolEntry>();

      NewEntry->Symbol = nullptr;
      NewEntry->Index = StringEntriesForEmission.size();

      if (StringEntriesForEmission.empty())
        NewEntry->Offset = 0;
      else {
        StringEntry *PrevString = StringEntriesForEmission.back();
        NewEntry->Offset =
            PrevString->getValue()->Offset + PrevString->getKeyLength() + 1;
      }

      String->getValue() = NewEntry;
      StringEntriesForEmission.push_back(String);
    }

    return String;
  }

  /// Erase contents of StringsForEmission.
  void clear() { StringEntriesForEmission.clear(); }

  /// Enumerate all strings in sequential order and call \p Handler for each
  /// string.
  void forEach(function_ref<void(DwarfStringPoolEntryRef)> Handler) const {
    for (const StringEntry *Entry : StringEntriesForEmission)
      Handler(*Entry);
  }

  std::function<StringRef(StringRef)> getTranslator() {
    return StringsTranslator;
  }

protected:
  /// List of strings for emission.
  StringsVector StringEntriesForEmission;

  /// String pool for the translated strings.
  StringPool &Strings;

  /// Translator for the strings.
  std::function<StringRef(StringRef)> StringsTranslator;
};

} // end of namespace dwarflinker_parallel
} // end namespace llvm

#endif // LLVM_DWARFLINKERPARALLEL_STRINGTABLE_H
