//===-- NVPTXLowerAlloca.cpp - Make alloca to use local memory =====--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// For all alloca instructions, and add a pair of cast to local address for
// each of them. For example,
//
//   %A = alloca i32
//   store i32 0, i32* %A ; emits st.u32
//
// will be transformed to
//
//   %A = alloca i32
//   %Local = addrspacecast i32* %A to i32 addrspace(5)*
//   %Generic = addrspacecast i32 addrspace(5)* %A to i32*
//   store i32 0, i32 addrspace(5)* %Generic ; emits st.local.u32
//
// And we will rely on NVPTXInferAddressSpaces to combine the last two
// instructions.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/NVPTXBaseInfo.h"
#include "NVPTX.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"

using namespace llvm;

namespace llvm {
void initializeNVPTXLowerAllocaPass(PassRegistry &);
}

namespace {
class NVPTXLowerAlloca : public FunctionPass {
  bool runOnFunction(Function &F) override;

public:
  static char ID; // Pass identification, replacement for typeid
  NVPTXLowerAlloca() : FunctionPass(ID) {}
  StringRef getPassName() const override {
    return "convert address space of alloca'ed memory to local";
  }
};
} // namespace

char NVPTXLowerAlloca::ID = 1;

INITIALIZE_PASS(NVPTXLowerAlloca, "nvptx-lower-alloca", "Lower Alloca", false,
                false)

// =============================================================================
// Main function for this pass.
// =============================================================================
bool NVPTXLowerAlloca::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  bool Changed = false;
  for (auto &BB : F)
    for (auto &I : BB) {
      if (auto allocaInst = dyn_cast<AllocaInst>(&I)) {
        Changed = true;

        PointerType *AllocInstPtrTy =
            cast<PointerType>(allocaInst->getType()->getScalarType());
        unsigned AllocAddrSpace = AllocInstPtrTy->getAddressSpace();
        assert((AllocAddrSpace == ADDRESS_SPACE_GENERIC ||
                AllocAddrSpace == ADDRESS_SPACE_LOCAL) &&
               "AllocaInst can only be in Generic or Local address space for "
               "NVPTX.");

        Instruction *AllocaInLocalAS = allocaInst;
        auto ETy = allocaInst->getAllocatedType();

        // We need to make sure that LLVM has info that alloca needs to go to
        // ADDRESS_SPACE_LOCAL for InferAddressSpace pass.
        //
        // For allocas in ADDRESS_SPACE_LOCAL, we add addrspacecast to
        // ADDRESS_SPACE_LOCAL and back to ADDRESS_SPACE_GENERIC, so that
        // the alloca's users still use a generic pointer to operate on.
        //
        // For allocas already in ADDRESS_SPACE_LOCAL, we just need
        // addrspacecast to ADDRESS_SPACE_GENERIC.
        if (AllocAddrSpace == ADDRESS_SPACE_GENERIC) {
          auto ASCastToLocalAS = new AddrSpaceCastInst(
              allocaInst, PointerType::get(ETy, ADDRESS_SPACE_LOCAL), "");
          ASCastToLocalAS->insertAfter(allocaInst);
          AllocaInLocalAS = ASCastToLocalAS;
        }

        auto AllocaInGenericAS = new AddrSpaceCastInst(
            AllocaInLocalAS, PointerType::get(ETy, ADDRESS_SPACE_GENERIC), "");
        AllocaInGenericAS->insertAfter(AllocaInLocalAS);

        for (Use &AllocaUse : llvm::make_early_inc_range(allocaInst->uses())) {
          // Check Load, Store, GEP, and BitCast Uses on alloca and make them
          // use the converted generic address, in order to expose non-generic
          // addrspacecast to NVPTXInferAddressSpaces. For other types
          // of instructions this is unnecessary and may introduce redundant
          // address cast.
          auto LI = dyn_cast<LoadInst>(AllocaUse.getUser());
          if (LI && LI->getPointerOperand() == allocaInst &&
              !LI->isVolatile()) {
            LI->setOperand(LI->getPointerOperandIndex(), AllocaInGenericAS);
            continue;
          }
          auto SI = dyn_cast<StoreInst>(AllocaUse.getUser());
          if (SI && SI->getPointerOperand() == allocaInst &&
              !SI->isVolatile()) {
            SI->setOperand(SI->getPointerOperandIndex(), AllocaInGenericAS);
            continue;
          }
          auto GI = dyn_cast<GetElementPtrInst>(AllocaUse.getUser());
          if (GI && GI->getPointerOperand() == allocaInst) {
            GI->setOperand(GI->getPointerOperandIndex(), AllocaInGenericAS);
            continue;
          }
          auto BI = dyn_cast<BitCastInst>(AllocaUse.getUser());
          if (BI && BI->getOperand(0) == allocaInst) {
            BI->setOperand(0, AllocaInGenericAS);
            continue;
          }
        }
      }
    }
  return Changed;
}

FunctionPass *llvm::createNVPTXLowerAllocaPass() {
  return new NVPTXLowerAlloca();
}
