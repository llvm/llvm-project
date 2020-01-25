//===-- RegisterUtilities.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/Process/elf-core/RegisterUtilities.h"
#include "llvm/ADT/STLExtras.h"

using namespace lldb_private;

static llvm::Optional<uint32_t>
getNoteType(const llvm::Triple &Triple,
            llvm::ArrayRef<RegsetDesc> RegsetDescs) {
  for (const auto &Entry : RegsetDescs) {
    if (Entry.OS != Triple.getOS())
      continue;
    if (Entry.Arch != llvm::Triple::UnknownArch &&
        Entry.Arch != Triple.getArch())
      continue;
    return Entry.Note;
  }
  return llvm::None;
}

DataExtractor lldb_private::getRegset(llvm::ArrayRef<CoreNote> Notes,
                                      const llvm::Triple &Triple,
                                      llvm::ArrayRef<RegsetDesc> RegsetDescs) {
  auto TypeOr = getNoteType(Triple, RegsetDescs);
  if (!TypeOr)
    return DataExtractor();
  uint32_t Type = *TypeOr;
  auto Iter = llvm::find_if(
      Notes, [Type](const CoreNote &Note) { return Note.info.n_type == Type; });
  return Iter == Notes.end() ? DataExtractor() : Iter->data;
}
