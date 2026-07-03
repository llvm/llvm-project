//===- ObjCConstantIvarOffset.cpp - Promote ObjC ivar offsets -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// This pass promotes OBJC_IVAR_$_* offset globals to constants when the full
/// class hierarchy is visible, pre-sliding offsets to match the runtime
/// moveIvars().
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/ObjCConstantIvarOffset.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IR/ObjCClassHierarchy.h"
#include "llvm/IR/Type.h"

using namespace llvm;


// Collects ivar offset globals owned by this class's _class_ro_t,
static SmallVector<GlobalVariable *, 4>
collectIvarOffsetGlobals(GlobalVariable &ROGV) {
  SmallVector<GlobalVariable *, 4> Offsets;
  // Walk ROGV -> ivarList -> ivar[i] -> offset global.
  //   _class_ro_t: { flags, instanceStart, instanceSize, ..., ivarList, ... }
  //   _ivar_list_t: { entsize, count, ivars }
  //   _ivar_t: { offset, name, type, alignment_raw, size }

  auto *ROStruct = dyn_cast<ConstantStruct>(ROGV.getInitializer());
  if (!ROStruct || ROStruct->getNumOperands() < 8)
    return Offsets;

  auto *IvarListGV =
      dyn_cast_or_null<GlobalVariable>(ROStruct->getOperand(7)->stripPointerCasts());
  if (!IvarListGV || !IvarListGV->hasInitializer())
    return Offsets;

  auto *IvarList = dyn_cast<ConstantStruct>(IvarListGV->getInitializer());
  if (!IvarList || IvarList->getNumOperands() < 3)
    return Offsets;

  auto *Ivars = dyn_cast<ConstantArray>(IvarList->getOperand(2));
  if (!Ivars)
    return Offsets;

  for (const auto &Ivar : Ivars->operands()) {
    auto *IvarStruct = dyn_cast<ConstantStruct>(Ivar.get());
    if (!IvarStruct || IvarStruct->getNumOperands() < 5)
      continue;
    if (auto *OffsetGV = dyn_cast_or_null<GlobalVariable>(
            IvarStruct->getOperand(0)->stripPointerCasts()))
      Offsets.push_back(OffsetGV);
  }

  return Offsets;
}

// Apply resolved layout to one class: slide ivar offsets by the delta,
// mark them constant, and update _class_ro_t instanceStart/Size.
static bool
applyResolvedLayout(const ObjCClassHierarchy::ResolvedClass &Class) {
  GlobalVariable &ROGV = *Class.ROGV;
  uint32_t InstanceStart = Class.InstanceStart;
  uint32_t InstanceSize = Class.InstanceSize;

  // Read the original instanceStart/Size from _class_ro_t.
  auto *ROStruct = cast<ConstantStruct>(ROGV.getInitializer());
  uint32_t OldStart =
      cast<ConstantInt>(ROStruct->getOperand(1))->getZExtValue();
  uint32_t OldSize =
      cast<ConstantInt>(ROStruct->getOperand(2))->getZExtValue();

  uint32_t Diff = InstanceStart - OldStart;
  bool Changed = false;

  // Slide each OBJC_IVAR_$_* initializer and mark it constant.
  auto IvarOffsets = collectIvarOffsetGlobals(ROGV);

  for (auto *GV : IvarOffsets) {
    if (Diff != 0 && GV->hasInitializer()) {
      if (const auto *Old = dyn_cast<ConstantInt>(GV->getInitializer())) {
        GV->setInitializer(
            ConstantInt::get(Old->getType(), Old->getSExtValue() + Diff));
        Changed = true;
      }
    }

    if (!GV->isConstant()) {
      GV->setConstant(true);
      Changed = true;
    }
  }

  // Update _class_ro_t if instanceStart or instanceSize changed.
  if (OldStart != InstanceStart || OldSize != InstanceSize) {
    SmallVector<Constant *, 10> Ops;
    for (const auto &Op : ROStruct->operands())
      Ops.push_back(cast<Constant>(Op.get()));
    Type *I32Ty = Type::getInt32Ty(ROGV.getContext());
    Ops[1] = ConstantInt::get(I32Ty, InstanceStart);
    Ops[2] = ConstantInt::get(I32Ty, InstanceSize);
    ROGV.setInitializer(ConstantStruct::get(ROStruct->getType(), Ops));
    Changed = true;
  }

  return Changed;
}

PreservedAnalyses ObjCConstantIvarOffsetPass::run(Module &M,
                                                  ModuleAnalysisManager &MAM) {
  ObjCClassHierarchy Hierarchy(M, ImportSummary);

  bool Changed = false;
  for (const auto &Class : Hierarchy.getResolvedClasses())
    Changed |= applyResolvedLayout(Class);

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
