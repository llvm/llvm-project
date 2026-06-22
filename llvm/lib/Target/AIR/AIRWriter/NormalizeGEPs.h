//===- NormalizeGEPs.h - Normalize GEP source element types -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRWRITER_NORMALIZEGEPS_H
#define LLVM_LIB_TARGET_AIR_AIRWRITER_NORMALIZEGEPS_H

#include "PointeeTypeMap.h"
#include "llvm/IR/Module.h"

namespace llvm {
namespace metal {

void normalizeGEPs(Module &M, PointeeTypeMap &PTM);

void normalizeArrayGlobalGEPs(Module &M);

}
}

#endif
