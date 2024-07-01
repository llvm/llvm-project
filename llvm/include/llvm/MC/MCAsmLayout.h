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

public:
  MCAsmLayout(MCAssembler &Assembler);

  /// Get the assembler object this is a layout for.
  MCAssembler &getAssembler() const { return Assembler; }


  /// \name Section Access (in layout order)
  /// @{

  llvm::SmallVectorImpl<MCSection *> &getSectionOrder() { return SectionOrder; }
  const llvm::SmallVectorImpl<MCSection *> &getSectionOrder() const {
    return SectionOrder;
  }

  /// @}
  /// \name Fragment Layout Data
  /// @{

  /// @}
  /// \name Utility Functions
  /// @{

  /// Variant that reports a fatal error if the offset is not computable.
  uint64_t getSymbolOffset(const MCSymbol &S) const;

  /// If this symbol is equivalent to A + Constant, return A.
  const MCSymbol *getBaseSymbol(const MCSymbol &Symbol) const;

  /// @}
};

} // end namespace llvm

#endif
