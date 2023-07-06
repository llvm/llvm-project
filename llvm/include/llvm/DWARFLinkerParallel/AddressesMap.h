//===- AddressesMap.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DWARFLINKERPARALLEL_ADDRESSESMAP_H
#define LLVM_DWARFLINKERPARALLEL_ADDRESSESMAP_H

#include "llvm/ADT/AddressRanges.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFExpression.h"
#include <cstdint>

namespace llvm {
namespace dwarflinker_parallel {

/// Mapped value in the address map is the offset to apply to the
/// linked address.
using RangesTy = AddressRangesMap;

/// AddressesMap represents information about valid addresses used
/// by debug information. Valid addresses are those which points to
/// live code sections. i.e. relocations for these addresses point
/// into sections which would be/are placed into resulting binary.
class AddressesMap {
public:
  virtual ~AddressesMap() = default;

  /// Checks that there are valid relocations in the .debug_info
  /// section.
  virtual bool hasValidRelocs() = 0;

  /// Checks that the specified DWARF expression operand \p Op references live
  /// code section and returns the relocation adjustment value (to get the
  /// linked address this value might be added to the source expression operand
  /// address).
  /// \returns relocation adjustment value or std::nullopt if there is no
  /// corresponding live address.
  virtual std::optional<int64_t>
  getExprOpAddressRelocAdjustment(DWARFUnit &U,
                                  const DWARFExpression::Operation &Op,
                                  uint64_t StartOffset, uint64_t EndOffset) = 0;

  /// Checks that the specified subprogram \p DIE references the live code
  /// section and returns the relocation adjustment value (to get the linked
  /// address this value might be added to the source subprogram address).
  /// Allowed kinds of input DIE: DW_TAG_subprogram, DW_TAG_label.
  /// \returns relocation adjustment value or std::nullopt if there is no
  /// corresponding live address.
  virtual std::optional<int64_t>
  getSubprogramRelocAdjustment(const DWARFDie &DIE) = 0;

  /// Apply the valid relocations to the buffer \p Data, taking into
  /// account that Data is at \p BaseOffset in the .debug_info section.
  ///
  /// \returns true whether any reloc has been applied.
  virtual bool applyValidRelocs(MutableArrayRef<char> Data, uint64_t BaseOffset,
                                bool IsLittleEndian) = 0;

  /// Erases all data.
  virtual void clear() = 0;
};

} // end of namespace dwarflinker_parallel
} // end namespace llvm

#endif // LLVM_DWARFLINKERPARALLEL_ADDRESSESMAP_H
