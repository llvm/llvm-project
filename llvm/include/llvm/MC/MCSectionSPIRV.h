//===- MCSectionSPIRV.h - SPIR-V Machine Code Sections ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MCSectionSPIRV class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSECTIONSPIRV_H
#define LLVM_MC_MCSECTIONSPIRV_H

#include "llvm/MC/MCSection.h"
#include "llvm/MC/SectionKind.h"

namespace llvm {

class MCSectionSPIRV final : public MCSection {
  friend class MCContext;

  MCSectionSPIRV()
      : MCSection("", /*IsText=*/true, /*IsVirtual=*/false,
                  /*Begin=*/nullptr) {}
  // TODO: Add StringRef Name to MCSectionSPIRV.
};

} // end namespace llvm

#endif // LLVM_MC_MCSECTIONSPIRV_H
