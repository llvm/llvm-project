//===- AMDGPULDSUtils.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AMDGPU LDS related helper utility functions.
//
//===----------------------------------------------------------------------===//

#include "AMDGPULDSUtils.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/IR/Constants.h"

using namespace llvm;

namespace llvm {

namespace AMDGPU {

bool isKernelCC(const Function *Func) {
  return AMDGPU::isModuleEntryFunctionCC(Func->getCallingConv());
}

Align getAlign(DataLayout const &DL, const GlobalVariable *GV) {
  return DL.getValueOrABITypeAlignment(GV->getPointerAlignment(DL),
                                       GV->getValueType());
}

bool isUsedOnlyFromFunction(const User *U, const Function *F) {
  if (auto *I = dyn_cast<Instruction>(U)) {
    return I->getFunction() == F;
  }

  if (isa<ConstantExpr>(U)) {
    return all_of(U->users(),
                  [F](const User *U) { return isUsedOnlyFromFunction(U, F); });
  }

  return false;
}

bool shouldLowerLDSToStruct(const SmallPtrSetImpl<GlobalValue *> &UsedList,
                            const GlobalVariable &GV, const Function *F) {
  // Any LDS variable can be lowered by moving into the created struct
  // Each variable so lowered is allocated in every kernel, so variables
  // whose users are all known to be safe to lower without the transform
  // are left unchanged.
  bool Ret = false;
  SmallPtrSet<const User *, 8> Visited;
  SmallVector<const User *, 16> Stack(GV.users());

  assert(!F || isKernelCC(F));

  while (!Stack.empty()) {
    const User *V = Stack.pop_back_val();
    Visited.insert(V);

    if (auto *G = dyn_cast<GlobalValue>(V->stripPointerCasts())) {
      if (UsedList.contains(G)) {
        continue;
      }
    }

    if (auto *I = dyn_cast<Instruction>(V)) {
      const Function *UF = I->getFunction();
      if (UF == F) {
        // Used from this kernel, we want to put it into the structure.
        Ret = true;
      } else if (!F) {
        Ret |= !isKernelCC(UF);
      }
      continue;
    }

    if (auto *E = dyn_cast<ConstantExpr>(V)) {
      if (F) {
        // Any use which does not end up an instruction disqualifies a
        // variable to be put into a kernel's LDS structure because later
        // we will need to replace only this kernel's uses for which we
        // need to identify a using function.
        if (!isUsedOnlyFromFunction(E, F))
          return false;
      }
      for (const User *U : E->users()) {
        if (Visited.insert(U).second) {
          Stack.push_back(U);
        }
      }
      continue;
    }

    // Unknown user, conservatively lower the variable.
    // For module LDS conservatively means place it into the module LDS struct.
    // For kernel LDS it means lower as a standalone variable.
    return !F;
  }

  return Ret;
}

std::vector<GlobalVariable *>
findVariablesToLower(Module &M, const SmallPtrSetImpl<GlobalValue *> &UsedList,
                     const Function *F) {
  std::vector<llvm::GlobalVariable *> LocalVars;
  for (auto &GV : M.globals()) {
    if (GV.getType()->getPointerAddressSpace() != AMDGPUAS::LOCAL_ADDRESS) {
      continue;
    }
    if (!GV.hasInitializer()) {
      // addrspace(3) without initializer implies cuda/hip extern __shared__
      // the semantics for such a variable appears to be that all extern
      // __shared__ variables alias one another, in which case this transform
      // is not required
      continue;
    }
    if (!isa<UndefValue>(GV.getInitializer())) {
      // Initializers are unimplemented for local address space.
      // Leave such variables in place for consistent error reporting.
      continue;
    }
    if (GV.isConstant()) {
      // A constant undef variable can't be written to, and any load is
      // undef, so it should be eliminated by the optimizer. It could be
      // dropped by the back end if not. This pass skips over it.
      continue;
    }
    if (!shouldLowerLDSToStruct(UsedList, GV, F)) {
      continue;
    }
    LocalVars.push_back(&GV);
  }
  return LocalVars;
}

SmallPtrSet<GlobalValue *, 32> getUsedList(Module &M) {
  SmallPtrSet<GlobalValue *, 32> UsedList;

  SmallVector<GlobalValue *, 32> TmpVec;
  collectUsedGlobalVariables(M, TmpVec, true);
  UsedList.insert(TmpVec.begin(), TmpVec.end());

  TmpVec.clear();
  collectUsedGlobalVariables(M, TmpVec, false);
  UsedList.insert(TmpVec.begin(), TmpVec.end());

  return UsedList;
}

} // end namespace AMDGPU

} // end namespace llvm
