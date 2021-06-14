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
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/ReplaceConstant.h"

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

static void collectFunctionUses(User *U, const Function *F,
                                SetVector<Instruction *> &InstUsers) {
  SmallVector<User *> Stack{U};

  while (!Stack.empty()) {
    U = Stack.pop_back_val();

    if (auto *I = dyn_cast<Instruction>(U)) {
      if (I->getFunction() == F)
        InstUsers.insert(I);
      continue;
    }

    if (!isa<ConstantExpr>(U))
      continue;

    append_range(Stack, U->users());
  }
}

void replaceConstantUsesInFunction(ConstantExpr *C, const Function *F) {
  SetVector<Instruction *> InstUsers;

  collectFunctionUses(C, F, InstUsers);
  for (Instruction *I : InstUsers) {
    convertConstantExprsToInstructions(I, C);
  }
}

bool hasUserInstruction(const GlobalValue *GV) {
  SmallPtrSet<const User *, 8> Visited;
  SmallVector<const User *, 16> Stack(GV->users());

  while (!Stack.empty()) {
    const User *U = Stack.pop_back_val();

    if (!Visited.insert(U).second)
      continue;

    if (isa<Instruction>(U))
      return true;

    append_range(Stack, U->users());
  }

  return false;
}

bool shouldLowerLDSToStruct(const GlobalVariable &GV, const Function *F) {
  // We are not interested in kernel LDS lowering for module LDS itself.
  if (F && GV.getName() == "llvm.amdgcn.module.lds")
    return false;

  bool Ret = false;
  SmallPtrSet<const User *, 8> Visited;
  SmallVector<const User *, 16> Stack(GV.users());
  SmallPtrSet<const GlobalValue *, 8> GlobalUsers;

  assert(!F || isKernelCC(F));

  while (!Stack.empty()) {
    const User *V = Stack.pop_back_val();
    Visited.insert(V);

    if (auto *G = dyn_cast<GlobalValue>(V)) {
      StringRef GName = G->getName();
      if (F && GName != "llvm.used" && GName != "llvm.compiler.used") {
        // For kernel LDS lowering, if G is not a compiler.used list, then we
        // cannot lower the lds GV since we cannot replace the use of GV within
        // G.
        return false;
      }
      GlobalUsers.insert(G);
      continue;
    }

    if (auto *I = dyn_cast<Instruction>(V)) {
      const Function *UF = I->getFunction();
      if (UF == F) {
        // Used from this kernel, we want to put it into the structure.
        Ret = true;
      } else if (!F) {
        // For module LDS lowering, lowering is required if the user instruction
        // is from non-kernel function.
        Ret |= !isKernelCC(UF);
      }
      continue;
    }

    // User V should be a constant, recursively visit users of V.
    assert(isa<Constant>(V) && "Expected a constant.");
    append_range(Stack, V->users());
  }

  if (!F && !Ret) {
    // For module LDS lowering, we have not yet decided if we should lower GV or
    // not. Explore all global users of GV, and check if atleast one of these
    // global users appear as an use within an instruction (possibly nested use
    // via constant expression), if so, then conservately lower LDS.
    for (auto *G : GlobalUsers)
      Ret |= hasUserInstruction(G);
  }

  return Ret;
}

std::vector<GlobalVariable *> findVariablesToLower(Module &M,
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
    if (!shouldLowerLDSToStruct(GV, F)) {
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
