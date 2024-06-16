//===- MCAsmLayout.h - Assembly Layout Object -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCASMLAYOUT_H
#define LLVM_MC_MCASMLAYOUT_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {
class MCAssembler;
class MCFragment;
class MCSection;
class MCSymbol;

/// Encapsulates the layout of an assembly file at a particular point in time.
///
/// Assembly may require computing multiple layouts for a particular assembly
/// file as part of the relaxation process. This class encapsulates the layout
/// at a single point in time in such a way that it is always possible to
/// efficiently compute the exact address of any symbol in the assembly file,
/// even during the relaxation process.
class MCAsmLayout {
  MCAssembler &Assembler;

  /// List of sections in layout order.
  llvm::SmallVector<MCSection *, 16> SectionOrder;

  /// Compute the layout for the section if necessary.
  void ensureValid(const MCFragment *F) const;

public:
  MCAsmLayout(MCAssembler &Assembler);

  /// Get the assembler object this is a layout for.
  MCAssembler &getAssembler() const { return Assembler; }

  /// Invalidate the fragments starting with F because it has been
  /// resized. The fragment's size should have already been updated, but
  /// its bundle padding will be recomputed.
  void invalidateFragmentsFrom(MCFragment *F);

  void layoutBundle(MCFragment *F);

  /// \name Section Access (in layout order)
  /// @{

  llvm::SmallVectorImpl<MCSection *> &getSectionOrder() { return SectionOrder; }
  const llvm::SmallVectorImpl<MCSection *> &getSectionOrder() const {
    return SectionOrder;
  }

  /// @}
  /// \name Fragment Layout Data
  /// @{

  /// Get the offset of the given fragment inside its containing section.
  uint64_t getFragmentOffset(const MCFragment *F) const;

  /// @}
  /// \name Utility Functions
  /// @{

  /// Get the address space size of the given section, as it effects
  /// layout. This may differ from the size reported by \see
  /// getSectionFileSize() by not including section tail padding.
  uint64_t getSectionAddressSize(const MCSection *Sec) const;

  /// Get the data size of the given section, as emitted to the object
  /// file. This may include additional padding, or be 0 for virtual sections.
  uint64_t getSectionFileSize(const MCSection *Sec) const;

  /// Get the offset of the given symbol, as computed in the current
  /// layout.
  /// \return True on success.
  bool getSymbolOffset(const MCSymbol &S, uint64_t &Val) const;

  /// Variant that reports a fatal error if the offset is not computable.
  uint64_t getSymbolOffset(const MCSymbol &S) const;

  /// If this symbol is equivalent to A + Constant, return A.
  const MCSymbol *getBaseSymbol(const MCSymbol &Symbol) const;

  /// @}
};

} // end namespace llvm

#endif
