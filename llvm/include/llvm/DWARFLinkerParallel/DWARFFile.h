//===- DWARFFile.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DWARFLINKERPARALLEL_DWARFFILE_H
#define LLVM_DWARFLINKERPARALLEL_DWARFFILE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/DWARFLinkerParallel/AddressesMap.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/Support/Endian.h"
#include <functional>
#include <memory>

namespace llvm {
namespace dwarflinker_parallel {

/// This class represents DWARF information for source file
/// and it's address map.
///
/// May be used asynchroniously for reading.
class DWARFFile {
public:
  using UnloadCallbackTy = std::function<void(StringRef FileName)>;

  DWARFFile(StringRef Name, std::unique_ptr<DWARFContext> Dwarf,
            std::unique_ptr<AddressesMap> Addresses,
            const std::vector<std::string> &Warnings,
            UnloadCallbackTy UnloadFunc = nullptr)
      : FileName(Name), Dwarf(std::move(Dwarf)),
        Addresses(std::move(Addresses)), Warnings(Warnings),
        UnloadFunc(UnloadFunc) {
    if (this->Dwarf)
      Endianess = this->Dwarf->isLittleEndian() ? support::endianness::little
                                                : support::endianness::big;
  }

  /// Object file name.
  StringRef FileName;

  /// Source DWARF information.
  std::unique_ptr<DWARFContext> Dwarf;

  /// Helpful address information(list of valid address ranges, relocations).
  std::unique_ptr<AddressesMap> Addresses;

  /// Warnings for object file.
  const std::vector<std::string> &Warnings;

  /// Endiannes of source DWARF information.
  support::endianness Endianess = support::endianness::little;

  /// Callback to the module keeping object file to unload.
  UnloadCallbackTy UnloadFunc;

  /// Unloads object file and corresponding AddressesMap and Dwarf Context.
  void unload() {
    Addresses.reset();
    Dwarf.reset();

    if (UnloadFunc)
      UnloadFunc(FileName);
  }
};

} // end namespace dwarflinker_parallel
} // end namespace llvm

#endif // LLVM_DWARFLINKERPARALLEL_DWARFFILE_H
