//===- AIRAddressSpaces.h - AIR address-space constants -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Single source of truth for the AIR address-space numbering used by every
/// pass and writer in this target.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRADDRESSSPACES_H
#define LLVM_LIB_TARGET_AIR_AIRADDRESSSPACES_H

#include "llvm/IR/AIR.h"

namespace llvm {
namespace metal {
// The numbering is shared with AIR's consumers (llvm/IR/AIR.h).
namespace AS = air::AS;
} // namespace metal
} // namespace llvm

#endif // LLVM_LIB_TARGET_AIR_AIRADDRESSSPACES_H
