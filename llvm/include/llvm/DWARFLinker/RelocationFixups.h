//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DWARFLINKER_RELOCATIONFIXUPS_H
#define LLVM_DWARFLINKER_RELOCATIONFIXUPS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DWARFLinker/AddressesMap.h"
#include "llvm/Support/Compiler.h"
#include <cstdint>
#include <optional>

namespace llvm {
namespace dwarf_linker {

/// Where the attributes of a unit were cloned to, so that their relocations can
/// be saved once the output offsets of the unit are known.
class RelocationFixups {
public:
  /// Where the operations of an expression were cloned to. An operation copied
  /// as is keeps the relocations of its bytes, while one rewritten from
  /// DW_OP_addrx or DW_OP_constx takes those of its .debug_addr entry.
  class ClonedExpression {
  public:
    struct Operation {
      RelocSectionKind InputSection;
      /// In .debug_addr, or from the start of the input expression.
      uint64_t InputStart;
      uint64_t InputEnd;
      /// From the start of the output expression, or of the attribute after
      /// addOutputPrefix().
      uint64_t OutputOffset;
    };

    explicit ClonedExpression(uint64_t InputSize) : InputSize(InputSize) {}

    void noteCopiedBytes(uint64_t InputStart, uint64_t InputEnd,
                         uint64_t OutputOffset) {
      Operations.push_back(
          {RelocSectionKind::DebugInfo, InputStart, InputEnd, OutputOffset});
    }

    void noteDebugAddrOperand(uint64_t EntryOffset, uint8_t AddrSize,
                              uint64_t OutputOffset) {
      Operations.push_back({RelocSectionKind::DebugAddr, EntryOffset,
                            EntryOffset + AddrSize, OutputOffset});
    }

    /// Moves the operations by \p Size, the bytes the attribute has ahead of
    /// the expression.
    void addOutputPrefix(uint64_t Size) {
      for (Operation &Op : Operations)
        Op.OutputOffset += Size;
    }

    /// The size of the input expression, which ends its attribute.
    uint64_t InputSize;
    SmallVector<Operation, 2> Operations;
  };

  /// An address attribute cloned from one .debug_addr entry into another.
  struct DebugAddrEntry {
    uint64_t InputOffset;
    /// In the address table of the output unit.
    uint64_t OutputIndex;
  };

  /// What an attribute being cloned was cloned from.
  struct AttributeSource {
    std::optional<ClonedExpression> Expression;
    std::optional<DebugAddrEntry> Entry;

    void clear() {
      Expression.reset();
      Entry.reset();
    }
  };

  /// Notes that the attribute \p Attr of a DIE tagged \p Tag was cloned from
  /// [\p InputStart, \p InputEnd) of .debug_info to \p OutputOffset, which does
  /// not count the abbreviation number of the DIE yet.
  LLVM_ABI void noteAttribute(dwarf::Tag Tag, dwarf::Attribute Attr,
                              uint64_t InputStart, uint64_t InputEnd,
                              uint64_t OutputOffset,
                              const AttributeSource &Source);

  /// Completes the attributes noted since the previous call, once the DIE has
  /// an abbreviation number.
  LLVM_ABI void finishDIE(uint64_t AbbrevNumberSize);

  /// Notes that a location list operation kept an index into .debug_addr.
  void noteDebugAddrEntry(DebugAddrEntry Entry) {
    DebugAddrFixups.push_back({Entry, /*IsLowPC=*/false});
  }

  void setDebugAddrTableOffset(uint64_t Offset) {
    DebugAddrTableOffset = Offset;
  }

  /// Saves the relocations, for the unit at \p UnitOffset in the input.
  LLVM_ABI void save(AddressesMap &Addresses, uint64_t UnitOffset,
                     uint8_t AddrSize) const;

  bool empty() const {
    return DIEFixups.empty() && DebugInfoFixups.empty() &&
           DebugAddrFixups.empty();
  }

  LLVM_ABI void clear();

private:
  /// Relocations moved into .debug_info.
  struct DebugInfoFixup {
    RelocSectionKind InputSection;
    uint64_t InputStart;
    uint64_t InputEnd;
    int64_t LinkedOffset;
  };

  struct DebugAddrFixup {
    DebugAddrEntry Entry;
    bool IsLowPC;
  };

  /// Those of the DIE being cloned.
  SmallVector<DebugInfoFixup> DIEFixups;
  SmallVector<DebugInfoFixup> DebugInfoFixups;
  SmallVector<DebugAddrFixup> DebugAddrFixups;
  std::optional<uint64_t> DebugAddrTableOffset;
};

} // end of namespace dwarf_linker
} // end of namespace llvm

#endif // LLVM_DWARFLINKER_RELOCATIONFIXUPS_H
