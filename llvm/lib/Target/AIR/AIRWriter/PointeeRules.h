//===- PointeeRules.h - Single-authority pointee-typing rules ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The pointee-typing rules that both the one-shot analysis
// (buildPointeeTypeMap) and the post-mutation repairs (PointerPointeeRepair)
// must agree on, expressed ONCE here. Each rule maps a Value (or an intrinsic
// name / arg position) to the pointee it requires, or null when the rule does
// not apply. The analysis calls these to populate the map; the repairs call the
// SAME functions to re-assert the rule after the IR is rewritten into final
// shape. This is the single authority that removes the analysis-vs-repair
// duplication.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRWRITER_POINTEERULES_H
#define LLVM_LIB_TARGET_AIR_AIRWRITER_POINTEERULES_H

#include "PointeeTypeMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"

namespace llvm {
namespace metal {

Type *atomicPointeeFromUsers(Value *Ptr);

Type *mmaElemFromName(StringRef Name, LLVMContext &Ctx);

Type *requiredSelectPointee(SelectInst *Sel, const PointeeTypeMap &PTM);

Type *requiredPhiPointee(PHINode *PN, const PointeeTypeMap &PTM);

bool reconcileGEPBaseType(GetElementPtrInst *GEP, PointeeTypeMap &PTM);

}
}

#endif
