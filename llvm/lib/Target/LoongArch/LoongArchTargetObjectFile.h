//=-- LoongArchTargetObjectFile.h - LoongArch Object Info ---------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_LOONGARCH_LOONGARCHTARGETOBJECTFILE_H
#define LLVM_LIB_TARGET_LOONGARCH_LOONGARCHTARGETOBJECTFILE_H

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"

namespace llvm {

/// This implementation is used for LoongArch ELF targets.
class LoongArchELFTargetObjectFile : public TargetLoweringObjectFileELF {
public:
  /// Describe a TLS variable address within debug info.
  const MCExpr *getDebugThreadLocalSymbol(const MCSymbol *Sym) const override;
};

} // end namespace llvm

#endif
