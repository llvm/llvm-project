//===-- PISAHelpers.h - Common helpers for PISA ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_PISAHELPERS_H
#define LLVM_LIB_TARGET_PISA_PISAHELPERS_H

#include "PISADefines.h"

#include "llvm/Support/ErrorHandling.h"

namespace llvm::PISA {

inline unsigned getSwizzleElemCount(PISA::Swizzle Swizzle) {
  switch (Swizzle) {
  case PISA::Swizzle::NONE:
    return 0;
  case PISA::Swizzle::X:
  case PISA::Swizzle::Y:
  case PISA::Swizzle::Z:
  case PISA::Swizzle::W:
    return 1;
  case PISA::Swizzle::XY:
  case PISA::Swizzle::ZW:
    return 2;
  case PISA::Swizzle::XYZW:
    return 4;
  }
  llvm_unreachable("Unknown swizzle!");
}

} // namespace llvm::PISA

#endif // LLVM_LIB_TARGET_PISA_PISAHELPERS_H
