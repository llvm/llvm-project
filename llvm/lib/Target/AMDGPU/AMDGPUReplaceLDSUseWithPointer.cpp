//===-- AMDGPUReplaceLDSUseWithPointer.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass replaces all the uses of LDS within non-kernel functions by
// corresponding pointer counter-parts.
//
// The main motivation behind this pass is - to *avoid* subsequent LDS lowering
// pass from directly packing LDS (assume large LDS) into a struct type which
// would otherwise cause allocating huge memory for struct instance within every
// kernel.
//
// Brief sketch of the algorithm implemented in this pass is as below:
//
//   1. Collect all the LDS defined in the module which qualify for pointer
//      replacement, say it is, LDSGlobals set.
//
//   2. Collect all the reachable callees for each kernel defined in the module,
//      say it is, KernelToCallees map.
//
//   3. FOR (each global GV from LDSGlobals set) DO
//        LDSUsedNonKernels = Collect all non-kernel functions which use GV.
//        FOR (each kernel K in KernelToCallees map) DO
//           ReachableCallees = KernelToCallees[K]
//           ReachableAndLDSUsedCallees =
//              SetIntersect(LDSUsedNonKernels, ReachableCallees)
//           IF (ReachableAndLDSUsedCallees is not empty) THEN
//             Pointer = Create a pointer to point-to GV if not created.
//             Initialize Pointer to point-to GV within kernel K.
//           ENDIF
//        ENDFOR
//        Replace all uses of GV within non kernel functions by Pointer.
//      ENFOR
//
// LLVM IR example:
//
//    Input IR:
//
//    @lds = internal addrspace(3) global [4 x i32] undef, align 16
//
//    define internal void @f0() {
//    entry:
//      %gep = getelementptr inbounds [4 x i32], [4 x i32] addrspace(3)* @lds,
//             i32 0, i32 0
//      ret void
//    }
//
//    define protected amdgpu_kernel void @k0() {
//    entry:
//      call void @f0()
//      ret void
//    }
//
//    Output IR:
//
//    @lds = internal addrspace(3) global [4 x i32] undef, align 16
//    @lds.ptr = internal unnamed_addr addrspace(3) global i16 undef, align 2
//
//    define internal void @f0() {
//    entry:
//      %0 = load i16, i16 addrspace(3)* @lds.ptr, align 2
//      %1 = getelementptr i8, i8 addrspace(3)* null, i16 %0
//      %2 = bitcast i8 addrspace(3)* %1 to [4 x i32] addrspace(3)*
//      %gep = getelementptr inbounds [4 x i32], [4 x i32] addrspace(3)* %2,
//             i32 0, i32 0
//      ret void
//    }
//
//    define protected amdgpu_kernel void @k0() {
//    entry:
//      store i16 ptrtoint ([4 x i32] addrspace(3)* @lds to i16),
//            i16 addrspace(3)* @lds.ptr, align 2
//      call void @f0()
//      ret void
//    }
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "Utils/AMDGPUMemoryUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <algorithm>
#include <vector>

#define DEBUG_TYPE "amdgpu-replace-lds-use-with-pointer"

using namespace llvm;

namespace {

namespace AMDGPU {
/// Collect all the instructions where user \p U belongs to. \p U could be
/// instruction itself or it could be a constant expression which is used within
/// an instruction. If \p CollectKernelInsts is true, collect instructions only
/// from kernels, otherwise collect instructions only from non-kernel functions.
DenseMap<Function *, SmallPtrSet<Instruction *, 8>>
getFunctionToInstsMap(User *U, bool CollectKernelInsts);

SmallPtrSet<Function *, 8> collectNonKernelAccessorsOfLDS(GlobalVariable *GV);

} // namespace AMDGPU

class ReplaceLDSUseImpl {
  Module &M;
  LLVMContext &Ctx;
  const DataLayout &DL;
  Constant *LDSMemBaseAddr;

  DenseMap<GlobalVariable *, GlobalVariable *> LDSToPointer;
  DenseMap<GlobalVariable *, SmallPtrSet<Function *, 8>> LDSToNonKernels;
  DenseMap<Function *, SmallPtrSet<Function *, 8>> KernelToCallees;
  DenseMap<Function *, SmallPtrSet<GlobalVariable *, 8>> KernelToLDSPointers;
  DenseMap<Function *, BasicBlock *> KernelToInitBB;
  DenseMap<Function *, DenseMap<GlobalVariable *, Value *>>
      FunctionToLDSToReplaceInst;

  // Collect LDS which requires their uses to be replaced by pointer.
  std::vector<GlobalVariable *> collectLDSRequiringPointerReplace() {
    // Collect LDS which requires module lowering.
    std::vector<GlobalVariable *> LDSGlobals =
        llvm::AMDGPU::findLDSVariablesToLower(M, nullptr);

    // Remove LDS which don't qualify for replacement.
    llvm::erase_if(LDSGlobals, [&](GlobalVariable *GV) {
      return shouldIgnorePointerReplacement(GV);
    });

    return LDSGlobals;
  }

  // Returns true if uses of given LDS global within non-kernel functions should
  // be keep as it is without pointer replacement.
  bool shouldIgnorePointerReplacement(GlobalVariable *GV) {
    // LDS whose size is very small and doesn't exceed pointer size is not worth
    // replacing.
    if (DL.getTypeAllocSize(GV->getValueType()) <= 2)
      return true;

    // LDS which is not used from non-kernel function scope or it is used from
    // global scope does not qualify for replacement.
    LDSToNonKernels[GV] = AMDGPU::collectNonKernelAccessorsOfLDS(GV);
    return LDSToNonKernels[GV].empty();

    // FIXME: When GV is used within all (or within most of the kernels), then
    // it does not make sense to create a pointer for it.
  }

  // Insert new global LDS pointer which points to LDS.
  GlobalVariable *createLDSPointer(GlobalVariable *GV) {
    // LDS pointer which points to LDS is already created? Return it.
    auto PointerEntry = LDSToPointer.insert(std::pair(GV, nullptr));
    if (!PointerEntry.second)
      return PointerEntry.first->second;

    // We need to create new LDS pointer which points to LDS.
    //
    // Each CU owns at max 64K of LDS memory, so LDS address ranges from 0 to
    // 2^16 - 1. Hence 16 bit pointer is enough to hold the LDS address.
    auto *I16Ty = Type::getInt16Ty(Ctx);
    GlobalVariable *LDSPointer = new GlobalVariable(
        M, I16Ty, false, GlobalValue::InternalLinkage, UndefValue::get(I16Ty),
        GV->getName() + Twine(".ptr"), nullptr, GlobalVariable::NotThreadLocal,
        AMDGPUAS::LOCAL_ADDRESS);

    LDSPointer->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    LDSPointer->setAlignment(llvm::AMDGPU::getAlign(DL, LDSPointer));

    // Mark that an associated LDS pointer is created for LDS.
    LDSToPointer[GV] = LDSPointer;

    return LDSPointer;
  }

  // Split entry basic block in such a way that only lane 0 of each wave does
  // the LDS pointer initialization, and return newly created basic block.
  BasicBlock *activateLaneZero(Function *K) {
    // If the entry basic block of kernel K is already split, then return
    // newly created basic block.
    auto BasicBlockEntry = KernelToInitBB.insert(std::pair(K, nullptr));
    if (!BasicBlockEntry.second)
      return BasicBlockEntry.first->second;

    // Split entry basic block of kernel K.
    auto *EI = &(*(K->getEntryBlock().getFirstInsertionPt()));
    IRBuilder<> Builder(EI);

    Value *Mbcnt =
        Builder.CreateIntrinsic(Intrinsic::amdgcn_mbcnt_lo, {},
                                {Builder.getInt32(-1), Builder.getInt32(0)});
    Value *Cond = Builder.CreateICmpEQ(Mbcnt, Builder.getInt32(0));
    Instruction *WB = cast<Instruction>(
        Builder.CreateIntrinsic(Intrinsic::amdgcn_wave_barrier, {}, {}));

    BasicBlock *NBB = SplitBlockAndInsertIfThen(Cond, WB, false)->getParent();

    // Mark that the entry basic block of kernel K is split.
    KernelToInitBB[K] = NBB;

    return NBB;
  }

  // Within given kernel, initialize given LDS pointer to point to given LDS.
  void initializeLDSPointer(Function *K, GlobalVariable *GV,
                            GlobalVariable *LDSPointer) {
    // If LDS pointer is already initialized within K, then nothing to do.
    auto PointerEntry = KernelToLDSPointers.insert(
        std::pair(K, SmallPtrSet<GlobalVariable *, 8>()));
    if (!PointerEntry.second)
      if (PointerEntry.first->second.contains(LDSPointer))
        return;

    // Insert instructions at EI which initialize LDS pointer to point-to LDS
    // within kernel K.
    //
    // That is, convert pointer type of GV to i16, and then store this converted
    // i16 value within LDSPointer which is of type i16*.
    auto *EI = &(*(activateLaneZero(K)->getFirstInsertionPt()));
    IRBuilder<> Builder(EI);
    Builder.CreateStore(Builder.CreatePtrToInt(GV, Type::getInt16Ty(Ctx)),
                        LDSPointer);

    // Mark that LDS pointer is initialized within kernel K.
    KernelToLDSPointers[K].insert(LDSPointer);
  }

  // We have created an LDS pointer for LDS, and initialized it to point-to LDS
  // within all relevant kernels. Now replace all the uses of LDS within
  // non-kernel functions by LDS pointer.
  void replaceLDSUseByPointer(GlobalVariable *GV, GlobalVariable *LDSPointer) {
    SmallVector<User *, 8> LDSUsers(GV->users());
    for (auto *U : LDSUsers) {
      // When `U` is a constant expression, it is possible that same constant
      // expression exists within multiple instructions, and within multiple
      // non-kernel functions. Collect all those non-kernel functions and all
      // those instructions within which `U` exist.
      auto FunctionToInsts =
          AMDGPU::getFunctionToInstsMap(U, false /*=CollectKernelInsts*/);

      for (const auto &FunctionToInst : FunctionToInsts) {
        Function *F = FunctionToInst.first;
        auto &Insts = FunctionToInst.second;
        for (auto *I : Insts) {
          // If `U` is a constant expression, then we need to break the
          // associated instruction into a set of separate instructions by
          // converting constant expressions into instructions.
          SmallPtrSet<Instruction *, 8> UserInsts;

          if (U == I) {
            // `U` is an instruction, conversion from constant expression to
            // set of instructions is *not* required.
            UserInsts.insert(I);
          } else {
            // `U` is a constant expression, convert it into corresponding set
            // of instructions.
            auto *CE = cast<ConstantExpr>(U);
            convertConstantExprsToInstructions(I, CE, &UserInsts);
          }

          // Go through all the user instructions, if LDS exist within them as
          // an operand, then replace it by replace instruction.
          for (auto *II : UserInsts) {
            auto *ReplaceInst = getReplacementInst(F, GV, LDSPointer);
            II->replaceUsesOfWith(GV, ReplaceInst);
          }
        }
      }
    }
  }

  // Create a set of replacement instructions which together replace LDS within
  // non-kernel function F by accessing LDS indirectly using LDS pointer.
  Value *getReplacementInst(Function *F, GlobalVariable *GV,
                            GlobalVariable *LDSPointer) {
    // If the instruction which replaces LDS within F is already created, then
    // return it.
    auto LDSEntry = FunctionToLDSToReplaceInst.insert(
        std::pair(F, DenseMap<GlobalVariable *, Value *>()));
    if (!LDSEntry.second) {
      auto ReplaceInstEntry =
          LDSEntry.first->second.insert(std::pair(GV, nullptr));
      if (!ReplaceInstEntry.second)
        return ReplaceInstEntry.first->second;
    }

    // Get the instruction insertion point within the beginning of the entry
    // block of current non-kernel function.
    auto *EI = &(*(F->getEntryBlock().getFirstInsertionPt()));
    IRBuilder<> Builder(EI);

    // Insert required set of instructions which replace LDS within F.
    auto *V = Builder.CreateBitCast(
        Builder.CreateGEP(
            Builder.getInt8Ty(), LDSMemBaseAddr,
            Builder.CreateLoad(LDSPointer->getValueType(), LDSPointer)),
        GV->getType());

    // Mark that the replacement instruction which replace LDS within F is
    // created.
    FunctionToLDSToReplaceInst[F][GV] = V;

    return V;
  }

public:
  ReplaceLDSUseImpl(Module &M)
      : M(M), Ctx(M.getContext()), DL(M.getDataLayout()) {
    LDSMemBaseAddr = Constant::getIntegerValue(
        PointerType::get(Type::getInt8Ty(M.getContext()),
                         AMDGPUAS::LOCAL_ADDRESS),
        APInt(32, 0));
  }

  // Entry-point function which interface ReplaceLDSUseImpl with outside of the
  // class.
  bool replaceLDSUse();

private:
  // For a given LDS from collected LDS globals set, replace its non-kernel
  // function scope uses by pointer.
  bool replaceLDSUse(GlobalVariable *GV);
};

// For given LDS from collected LDS globals set, replace its non-kernel function
// scope uses by pointer.
bool ReplaceLDSUseImpl::replaceLDSUse(GlobalVariable *GV) {
  // Holds all those non-kernel functions within which LDS is being accessed.
  SmallPtrSet<Function *, 8> &LDSAccessors = LDSToNonKernels[GV];

  // The LDS pointer which points to LDS and replaces all the uses of LDS.
  GlobalVariable *LDSPointer = nullptr;

  // Traverse through each kernel K, check and if required, initialize the
  // LDS pointer to point to LDS within K.
  for (const auto &KernelToCallee : KernelToCallees) {
    Function *K = KernelToCallee.first;
    SmallPtrSet<Function *, 8> Callees = KernelToCallee.second;

    // Compute reachable and LDS used callees for kernel K.
    set_intersect(Callees, LDSAccessors);

    // None of the LDS accessing non-kernel functions are reachable from
    // kernel K. Hence, no need to initialize LDS pointer within kernel K.
    if (Callees.empty())
      continue;

    // We have found reachable and LDS used callees for kernel K, and we need to
    // initialize LDS pointer within kernel K, and we need to replace LDS use
    // within those callees by LDS pointer.
    //
    // But, first check if LDS pointer is already created, if not create one.
    LDSPointer = createLDSPointer(GV);

    // Initialize LDS pointer to point to LDS within kernel K.
    initializeLDSPointer(K, GV, LDSPointer);
  }

  // We have not found reachable and LDS used callees for any of the kernels,
  // and hence we have not created LDS pointer.
  if (!LDSPointer)
    return false;

  // We have created an LDS pointer for LDS, and initialized it to point-to LDS
  // within all relevant kernels. Now replace all the uses of LDS within
  // non-kernel functions by LDS pointer.
  replaceLDSUseByPointer(GV, LDSPointer);

  return true;
}

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

    for (const auto &GI : *ECNode) {
      auto *CGN = GI.second;
      auto *F = CGN->getFunction();
      if (!F || F->isDeclaration() || llvm::AMDGPU::isKernelCC(F))
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
    SmallVector<CallGraphNode *, 8> CGNStack(depth_first(KCGN));
    SmallPtrSet<CallGraphNode *, 8> VisitedCGNodes;
    while (!CGNStack.empty()) {
      auto *CGN = CGNStack.pop_back_val();

      if (!VisitedCGNodes.insert(CGN).second)
        continue;

      // Ignore call graph node which does not have associated function or
      // associated function is not a definition.
      if (!CGN->getFunction() || CGN->getFunction()->isDeclaration())
        continue;

      for (const auto &GI : *CGN) {
        auto *RCB = cast<CallBase>(*GI.first);
        auto *RCGN = GI.second;

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
      if (!llvm::AMDGPU::isKernelCC(&F))
        continue;
      Function *K = &F;
      KernelToCallees[K] = collectReachableCallees(K);
    }
  }
};

/// Collect reachable callees for each kernel defined in the module \p M and
/// return collected callees at \p KernelToCallees.
void collectReachableCallees(
    Module &M,
    DenseMap<Function *, SmallPtrSet<Function *, 8>> &KernelToCallees) {
  CollectReachableCallees CRC{M};
  CRC.collectReachableCallees(KernelToCallees);
}

/// For the given LDS global \p GV, visit all its users and collect all
/// non-kernel functions within which \p GV is used and return collected list of
/// such non-kernel functions.
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
    if (!llvm::AMDGPU::isKernelCC(F))
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
      if (!llvm::AMDGPU::isKernelCC(F)) {
        continue;
      }
    } else {
      if (llvm::AMDGPU::isKernelCC(F)) {
        continue;
      }
    }

    FunctionToInsts.insert(std::pair(F, SmallPtrSet<Instruction *, 8>()));
    FunctionToInsts[F].insert(I);
  }

  return FunctionToInsts;
}

} // namespace AMDGPU

// Entry-point function which interface ReplaceLDSUseImpl with outside of the
// class.
bool ReplaceLDSUseImpl::replaceLDSUse() {
  // Collect LDS which requires their uses to be replaced by pointer.
  std::vector<GlobalVariable *> LDSGlobals =
      collectLDSRequiringPointerReplace();

  // No LDS to pointer-replace. Nothing to do.
  if (LDSGlobals.empty())
    return false;

  // Collect reachable callee set for each kernel defined in the module.
  AMDGPU::collectReachableCallees(M, KernelToCallees);

  if (KernelToCallees.empty()) {
    // Either module does not have any kernel definitions, or none of the kernel
    // has a call to non-kernel functions, or we could not resolve any of the
    // call sites to proper non-kernel functions, because of the situations like
    // inline asm calls. Nothing to replace.
    return false;
  }

  // For every LDS from collected LDS globals set, replace its non-kernel
  // function scope use by pointer.
  bool Changed = false;
  for (auto *GV : LDSGlobals)
    Changed |= replaceLDSUse(GV);

  return Changed;
}

class AMDGPUReplaceLDSUseWithPointer : public ModulePass {
public:
  static char ID;

  AMDGPUReplaceLDSUseWithPointer() : ModulePass(ID) {
    initializeAMDGPUReplaceLDSUseWithPointerPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
  }
};

} // namespace

char AMDGPUReplaceLDSUseWithPointer::ID = 0;
char &llvm::AMDGPUReplaceLDSUseWithPointerID =
    AMDGPUReplaceLDSUseWithPointer::ID;

INITIALIZE_PASS_BEGIN(
    AMDGPUReplaceLDSUseWithPointer, DEBUG_TYPE,
    "Replace within non-kernel function use of LDS with pointer",
    false /*only look at the cfg*/, false /*analysis pass*/)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(
    AMDGPUReplaceLDSUseWithPointer, DEBUG_TYPE,
    "Replace within non-kernel function use of LDS with pointer",
    false /*only look at the cfg*/, false /*analysis pass*/)

bool AMDGPUReplaceLDSUseWithPointer::runOnModule(Module &M) {
  ReplaceLDSUseImpl LDSUseReplacer{M};
  return LDSUseReplacer.replaceLDSUse();
}

ModulePass *llvm::createAMDGPUReplaceLDSUseWithPointerPass() {
  return new AMDGPUReplaceLDSUseWithPointer();
}

PreservedAnalyses
AMDGPUReplaceLDSUseWithPointerPass::run(Module &M, ModuleAnalysisManager &AM) {
  ReplaceLDSUseImpl LDSUseReplacer{M};
  LDSUseReplacer.replaceLDSUse();
  return PreservedAnalyses::all();
}
