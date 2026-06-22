//===- AggregateScalarize.h - Scalarize aggregate/bool-vec ops --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRWRITER_AGGREGATESCALARIZE_H
#define LLVM_LIB_TARGET_AIR_AIRWRITER_AGGREGATESCALARIZE_H

#include "PointeeTypeMap.h"
#include "llvm/IR/Module.h"

namespace llvm {
namespace metal {

void scalarizeBoolVectorCasts(Module &M);

void scalarizeAggregateLoads(Module &M, PointeeTypeMap &PTM);

void scalarizeAggregateStores(Module &M, PointeeTypeMap &PTM);

}
}

#endif
