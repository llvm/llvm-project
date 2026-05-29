//===- MCSectionMetalLib.h - MetalLib MC Sections ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MCSectionMetalLib class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSECTIONMETALLIB_H
#define LLVM_MC_MCSECTIONMETALLIB_H

#include "llvm/MC/MCSection.h"
#include "llvm/MC/SectionKind.h"

namespace llvm {

class MCSymbol;

/// MC section for the MetalLib container.
///
/// The metallib produced by the Metal target's embedder pass is emitted as a
/// single MC section named ".metallib"; this class is the section type the
/// MetalLib object writer locates and dumps to the output stream.
class MCSectionMetalLib final : public MCSection {
  friend class MCContext;

  MCSectionMetalLib(StringRef Name, SectionKind K, MCSymbol *Begin)
      : MCSection(Name, K.isText(), /*IsVirtual=*/false, Begin) {}
};

} // end namespace llvm

#endif // LLVM_MC_MCSECTIONMETALLIB_H
