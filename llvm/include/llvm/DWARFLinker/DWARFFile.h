//===- DWARFFile.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DWARFLINKER_DWARFFILE_H
#define LLVM_DWARFLINKER_DWARFFILE_H

#include "AddressesMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include <functional>
#include <memory>

namespace llvm {
namespace dwarf_linker {

/// This class represents DWARF information for source file
/// and it's address map.
///
/// May be used asynchroniously for reading.
class DWARFFile {
public:
  using UnloadCallbackTy = std::function<void(StringRef FileName)>;

  using UnitOffsetsCallbackTy =
      std::function<void(uint64_t OriginalUnitOffset, uint64_t DebugInfoOffset,
                         uint64_t DebugAddrOffset)>;

  DWARFFile(StringRef Name, std::unique_ptr<DWARFContext> Dwarf,
            std::unique_ptr<AddressesMap> Addresses,
            UnloadCallbackTy UnloadFunc = nullptr)
      : FileName(Name), Dwarf(std::move(Dwarf)),
        Addresses(std::move(Addresses)), UnloadFunc(UnloadFunc) {}

  /// Object file name.
  StringRef FileName;

  /// Source DWARF information.
  std::unique_ptr<DWARFContext> Dwarf;

  /// Helpful address information(list of valid address ranges, relocations).
  std::unique_ptr<AddressesMap> Addresses;

  /// Callback to the module keeping object file to unload.
  UnloadCallbackTy UnloadFunc;

  /// Reports where a unit landed in the output .debug_info and .debug_addr, for
  /// the relocations saved relative to it. Unlike Addresses, it outlives
  /// unload(), as a linker may only place the units after unloading them.
  UnitOffsetsCallbackTy UnitOffsetsFunc;

  /// Unloads object file and corresponding AddressesMap and Dwarf Context.
  void unload() {
    Addresses.reset();
    Dwarf.reset();

    if (UnloadFunc)
      UnloadFunc(FileName);
  }
};

} // namespace dwarf_linker
} // end namespace llvm

#endif // LLVM_DWARFLINKER_DWARFFILE_H
