//===-- SystemZTargetObjectFile.h - SystemZ Object Info ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZTARGETOBJECTFILE_H
#define LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZTARGETOBJECTFILE_H

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"

namespace llvm {

/// This implementation is used for SystemZ ELF targets.
class SystemZELFTargetObjectFile : public TargetLoweringObjectFileELF {
public:
  SystemZELFTargetObjectFile() = default;

  /// Describe a TLS variable address within debug info.
  const MCExpr *getDebugThreadLocalSymbol(const MCSymbol *Sym) const override;
};

/// This implementation is used for SystemZ GOFF targets.
class SystemZGOFFTargetObjectFile : public TargetLoweringObjectFileGOFF {
public:
  SystemZGOFFTargetObjectFile() = default;

  const MCExpr *getDwarfSymbolReference(const MCSymbol *Label,
                                        const TargetMachine &TM,
                                        MCContext &Ctx) const override;
};

} // end namespace llvm

#endif
