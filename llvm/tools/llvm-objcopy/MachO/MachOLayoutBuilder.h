//===- MachOLayoutBuilder.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJCOPY_MACHO_MACHOLAYOUTBUILDER_H
#define LLVM_OBJCOPY_MACHO_MACHOLAYOUTBUILDER_H

#include "MachOObjcopy.h"
#include "Object.h"

namespace llvm {
namespace objcopy {
namespace macho {

class MachOLayoutBuilder {
  Object &O;
  bool Is64Bit;
  uint64_t PageSize;

  // Points to the __LINKEDIT segment if it exists.
  MachO::macho_load_command *LinkEditLoadCommand = nullptr;
  StringTableBuilder StrTableBuilder{StringTableBuilder::MachO};

  uint32_t computeSizeOfCmds() const;
  void constructStringTable();
  void updateSymbolIndexes();
  void updateDySymTab(MachO::macho_load_command &MLC);
  uint64_t layoutSegments();
  uint64_t layoutRelocations(uint64_t Offset);
  Error layoutTail(uint64_t Offset);

public:
  MachOLayoutBuilder(Object &O, bool Is64Bit, uint64_t PageSize)
      : O(O), Is64Bit(Is64Bit), PageSize(PageSize) {}

  // Recomputes and updates fields in the given object such as file offsets.
  Error layout();

  StringTableBuilder &getStringTableBuilder() { return StrTableBuilder; }
};

} // end namespace macho
} // end namespace objcopy
} // end namespace llvm

#endif // LLVM_OBJCOPY_MACHO_MACHOLAYOUTBUILDER_H
