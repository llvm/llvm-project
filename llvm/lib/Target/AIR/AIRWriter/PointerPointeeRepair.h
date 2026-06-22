//===- PointerPointeeRepair.h - Pointer/pointee type agreement --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRWRITER_POINTERPOINTEEREPAIR_H
#define LLVM_LIB_TARGET_AIR_AIRWRITER_POINTERPOINTEEREPAIR_H

#include "PointeeTypeMap.h"
#include "llvm/IR/Module.h"

namespace llvm {
namespace metal {

void removeRedundantBitcasts(Module &M, PointeeTypeMap &PTM);

void fixPhiIncomingTypes(Module &M, PointeeTypeMap &PTM);

void fixMMAPointerSuffixMismatch(Module &M, PointeeTypeMap &PTM);

void fixSelectPointerArms(Module &M, PointeeTypeMap &PTM);

void fixAccessTypeMismatch(Module &M, PointeeTypeMap &PTM);

}
}

#endif
