//===-- AMDGPUDwarfDebug.h - AMDGPU DwarfDebug Implementation -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the AMDGPUDwarfDebug class, the AMDGPU-specific subclass
// of DwarfDebug. It customizes DWARF emission for the AMDGPU DWARF extensions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUDWARFDEBUG_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUDWARFDEBUG_H

#include "../../CodeGen/AsmPrinter/DwarfDebug.h"

namespace llvm {

/// AMDGPU-specific DwarfDebug implementation.
class AMDGPUDwarfDebug : public DwarfDebug {
public:
  AMDGPUDwarfDebug(AsmPrinter *A) : DwarfDebug(A) {}

  /// The AMDGPU DWARF extensions describe a pointer type's address space with
  /// DW_AT_LLVM_address_space rather than DW_AT_address_class.
  dwarf::Attribute getTypeAddressSpaceAttribute() const override {
    return dwarf::DW_AT_LLVM_address_space;
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUDWARFDEBUG_H
