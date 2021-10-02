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
#include "AMDGPU.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/ReplaceConstant.h"

using namespace llvm;

namespace llvm {

namespace AMDGPU {

// An helper class for collecting all reachable callees for each kernel defined
// within the module.
class CollectReachableCallees {
  Module &M;
  CallGraph CG;
  SmallPtrSet<CallGraphNode *, 8> AddressTakenFunctions;

  // Collect all address taken functions within the module.
  void collectAddressTakenFunctions() {
    auto *ECNode = CG.getExternalCallingNode();

    for (auto GI = ECNode->begin(), GE = ECNode->end(); GI != GE; ++GI) {
      auto *CGN = GI->second;
      auto *F = CGN->getFunction();
      if (!F || F->isDeclaration() || AMDGPU::isKernelCC(F))
        continue;
      AddressTakenFunctions.insert(CGN);
    }
  }

  // For given kernel, collect all its reachable non-kernel functions.
  SmallPtrSet<Function *, 8> collectReachableCallees(Function *K) {
    SmallPtrSet<Function *, 8> ReachableCallees;

    // Call graph node which represents this kernel.
    auto *KCGN = CG[K];

    // Go through all call graph nodes reachable from the node representing this
    // kernel, visit all their call sites, if the call site is direct, add
    // corresponding callee to reachable callee set, if it is indirect, resolve
    // the indirect call site to potential reachable callees, add them to
    // reachable callee set, and repeat the process for the newly added
    // potential callee nodes.
    //
    // FIXME: Need to handle bit-casted function pointers.
    //
    SmallVector<CallGraphNode *, 8> CGNStack(df_begin(KCGN), df_end(KCGN));
    SmallPtrSet<CallGraphNode *, 8> VisitedCGNodes;
    while (!CGNStack.empty()) {
      auto *CGN = CGNStack.pop_back_val();

      if (!VisitedCGNodes.insert(CGN).second)
        continue;

      // Ignore call graph node which does not have associated function or
      // associated function is not a definition.
      if (!CGN->getFunction() || CGN->getFunction()->isDeclaration())
        continue;

      for (auto GI = CGN->begin(), GE = CGN->end(); GI != GE; ++GI) {
        auto *RCB = cast<CallBase>(GI->first.getValue());
        auto *RCGN = GI->second;

        if (auto *DCallee = RCGN->getFunction()) {
          ReachableCallees.insert(DCallee);
        } else if (RCB->isIndirectCall()) {
          auto *RCBFTy = RCB->getFunctionType();
          for (auto *ACGN : AddressTakenFunctions) {
            auto *ACallee = ACGN->getFunction();
            if (ACallee->getFunctionType() == RCBFTy) {
              ReachableCallees.insert(ACallee);
              CGNStack.append(df_begin(ACGN), df_end(ACGN));
            }
          }
        }
      }
    }

    return ReachableCallees;
  }

public:
  explicit CollectReachableCallees(Module &M) : M(M), CG(CallGraph(M)) {
    // Collect address taken functions.
    collectAddressTakenFunctions();
  }

  void collectReachableCallees(
      DenseMap<Function *, SmallPtrSet<Function *, 8>> &KernelToCallees) {
    // Collect reachable callee set for each kernel defined in the module.
    for (Function &F : M.functions()) {
      if (!AMDGPU::isKernelCC(&F))
        continue;
      Function *K = &F;
      KernelToCallees[K] = collectReachableCallees(K);
    }
  }
};

void collectReachableCallees(
    Module &M,
    DenseMap<Function *, SmallPtrSet<Function *, 8>> &KernelToCallees) {
  CollectReachableCallees CRC{M};
  CRC.collectReachableCallees(KernelToCallees);
}

SmallPtrSet<Function *, 8> collectNonKernelAccessorsOfLDS(GlobalVariable *GV) {
  SmallPtrSet<Function *, 8> LDSAccessors;
  SmallVector<User *, 8> UserStack(GV->users());
  SmallPtrSet<User *, 8> VisitedUsers;

  while (!UserStack.empty()) {
    auto *U = UserStack.pop_back_val();

    // `U` is already visited? continue to next one.
    if (!VisitedUsers.insert(U).second)
      continue;

    // `U` is a global variable which is initialized with LDS. Ignore LDS.
    if (isa<GlobalValue>(U))
      return SmallPtrSet<Function *, 8>();

    // Recursively explore constant users.
    if (isa<Constant>(U)) {
      append_range(UserStack, U->users());
      continue;
    }

    // `U` should be an instruction, if it belongs to a non-kernel function F,
    // then collect F.
    Function *F = cast<Instruction>(U)->getFunction();
    if (!AMDGPU::isKernelCC(F))
      LDSAccessors.insert(F);
  }

  return LDSAccessors;
}

DenseMap<Function *, SmallPtrSet<Instruction *, 8>>
getFunctionToInstsMap(User *U, bool CollectKernelInsts) {
  DenseMap<Function *, SmallPtrSet<Instruction *, 8>> FunctionToInsts;
  SmallVector<User *, 8> UserStack;
  SmallPtrSet<User *, 8> VisitedUsers;

  UserStack.push_back(U);

  while (!UserStack.empty()) {
    auto *UU = UserStack.pop_back_val();

    if (!VisitedUsers.insert(UU).second)
      continue;

    if (isa<GlobalValue>(UU))
      continue;

    if (isa<Constant>(UU)) {
      append_range(UserStack, UU->users());
      continue;
    }

    auto *I = cast<Instruction>(UU);
    Function *F = I->getFunction();
    if (CollectKernelInsts) {
      if (!AMDGPU::isKernelCC(F)) {
        continue;
      }
    } else {
      if (AMDGPU::isKernelCC(F)) {
        continue;
      }
    }

    FunctionToInsts.insert(std::make_pair(F, SmallPtrSet<Instruction *, 8>()));
    FunctionToInsts[F].insert(I);
  }

  return FunctionToInsts;
}

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
