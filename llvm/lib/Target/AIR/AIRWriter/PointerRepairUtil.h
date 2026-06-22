//===- PointerRepairUtil.h - Shared typed-pointer repair helpers -*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AIR_AIRWRITER_POINTERREPAIRUTIL_H
#define LLVM_LIB_TARGET_AIR_AIRWRITER_POINTERREPAIRUTIL_H

#include "PointeeTypeMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

namespace llvm {
namespace metal {

inline Type *effectivePointee(Value *Base, const PointeeTypeMap &PTM) {
  if (auto *G = dyn_cast<GetElementPtrInst>(Base))
    return G->getResultElementType();
  return PTM.get(Base);
}

inline Value *retypePointerVia(Value *Ptr, Type *NewPointee,
                               Instruction *BeforeI, PointeeTypeMap &PTM) {
  if (effectivePointee(Ptr, PTM) == NewPointee &&
      !isa<ConstantPointerNull>(Ptr))
    return Ptr;
  auto *BC = cast<BitCastInst>(CastInst::Create(
      Instruction::BitCast, Ptr, Ptr->getType(), "", BeforeI->getIterator()));
  PTM.set(BC, NewPointee);
  return BC;
}

template <typename Inst, typename PredT>
SmallVector<Inst *, 8> collectInsts(Module &M, PredT Pred) {
  SmallVector<Inst *, 8> Out;
  for (auto &F : M)
    for (auto &BB : F)
      for (auto &I : BB)
        if (auto *Cast = dyn_cast<Inst>(&I))
          if (Pred(Cast))
            Out.push_back(Cast);
  return Out;
}

}
}

#endif
