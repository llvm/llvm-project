//===- NVPTXLowerAggrCopies.cpp - ------------------------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// \file
// Lower aggregate copies, memset, memcpy, memmov intrinsics into loops when
// the size is large or is not a compile-time constant.
//
//===----------------------------------------------------------------------===//

#include "NVPTXLowerAggrCopies.h"
#include "llvm/CodeGen/StackProtector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "nvptx"

using namespace llvm;

namespace {

// actual analysis class, which is a functionpass
struct NVPTXLowerAggrCopies : public FunctionPass {
  static char ID;

  NVPTXLowerAggrCopies() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addPreserved<StackProtector>();
  }

  bool runOnFunction(Function &F) override;

  static const unsigned MaxAggrCopySize = 128;

  const char *getPassName() const override {
    return "Lower aggregate copies/intrinsics into loops";
  }
};

char NVPTXLowerAggrCopies::ID = 0;

// Lower memcpy to loop.
void convertMemCpyToLoop(Instruction *ConvertedInst, Value *SrcAddr,
                         Value *DstAddr, Value *CopyLen, bool SrcIsVolatile,
                         bool DstIsVolatile, LLVMContext &Context,
                         Function &F) {
  Type *TypeOfCopyLen = CopyLen->getType();

  BasicBlock *OrigBB = ConvertedInst->getParent();
  BasicBlock *NewBB =
      ConvertedInst->getParent()->splitBasicBlock(ConvertedInst, "split");
  BasicBlock *LoopBB = BasicBlock::Create(Context, "loadstoreloop", &F, NewBB);

  OrigBB->getTerminator()->setSuccessor(0, LoopBB);
  IRBuilder<> Builder(OrigBB->getTerminator());

  // SrcAddr and DstAddr are expected to be pointer types,
  // so no check is made here.
  unsigned SrcAS = cast<PointerType>(SrcAddr->getType())->getAddressSpace();
  unsigned DstAS = cast<PointerType>(DstAddr->getType())->getAddressSpace();

  // Cast pointers to (char *)
  SrcAddr = Builder.CreateBitCast(SrcAddr, Builder.getInt8PtrTy(SrcAS));
  DstAddr = Builder.CreateBitCast(DstAddr, Builder.getInt8PtrTy(DstAS));

  IRBuilder<> LoopBuilder(LoopBB);
  PHINode *LoopIndex = LoopBuilder.CreatePHI(TypeOfCopyLen, 0);
  LoopIndex->addIncoming(ConstantInt::get(TypeOfCopyLen, 0), OrigBB);

  // load from SrcAddr+LoopIndex
  // TODO: we can leverage the align parameter of llvm.memcpy for more efficient
  // word-sized loads and stores.
  Value *Element =
      LoopBuilder.CreateLoad(LoopBuilder.CreateInBoundsGEP(
                                 LoopBuilder.getInt8Ty(), SrcAddr, LoopIndex),
                             SrcIsVolatile);
  // store at DstAddr+LoopIndex
  LoopBuilder.CreateStore(Element,
                          LoopBuilder.CreateInBoundsGEP(LoopBuilder.getInt8Ty(),
                                                        DstAddr, LoopIndex),
                          DstIsVolatile);

  // The value for LoopIndex coming from backedge is (LoopIndex + 1)
  Value *NewIndex =
      LoopBuilder.CreateAdd(LoopIndex, ConstantInt::get(TypeOfCopyLen, 1));
  LoopIndex->addIncoming(NewIndex, LoopBB);

  LoopBuilder.CreateCondBr(LoopBuilder.CreateICmpULT(NewIndex, CopyLen), LoopBB,
                           NewBB);
}

// Lower memmove to IR. memmove is required to correctly copy overlapping memory
// regions; therefore, it has to check the relative positions of the source and
// destination pointers and choose the copy direction accordingly.
//
// The code below is an IR rendition of this C function:
//
// void* memmove(void* dst, const void* src, size_t n) {
//   unsigned char* d = dst;
//   const unsigned char* s = src;
//   if (s < d) {
//     // copy backwards
//     while (n--) {
//       d[n] = s[n];
//     }
//   } else {
//     // copy forward
//     for (size_t i = 0; i < n; ++i) {
//       d[i] = s[i];
//     }
//   }
//   return dst;
// }
void convertMemMoveToLoop(Instruction *ConvertedInst, Value *SrcAddr,
                          Value *DstAddr, Value *CopyLen, bool SrcIsVolatile,
                          bool DstIsVolatile, LLVMContext &Context,
                          Function &F) {
  Type *TypeOfCopyLen = CopyLen->getType();
  BasicBlock *OrigBB = ConvertedInst->getParent();

  // Create the a comparison of src and dst, based on which we jump to either
  // the forward-copy part of the function (if src >= dst) or the backwards-copy
  // part (if src < dst).
  // SplitBlockAndInsertIfThenElse conveniently creates the basic if-then-else
  // structure. Its block terminators (unconditional branches) are replaced by
  // the appropriate conditional branches when the loop is built.
  ICmpInst *PtrCompare = new ICmpInst(ConvertedInst, ICmpInst::ICMP_ULT,
                                      SrcAddr, DstAddr, "compare_src_dst");
  TerminatorInst *ThenTerm, *ElseTerm;
  SplitBlockAndInsertIfThenElse(PtrCompare, ConvertedInst, &ThenTerm,
                                &ElseTerm);

  // Each part of the function consists of two blocks:
  //   copy_backwards:        used to skip the loop when n == 0
  //   copy_backwards_loop:   the actual backwards loop BB
  //   copy_forward:          used to skip the loop when n == 0
  //   copy_forward_loop:     the actual forward loop BB
  BasicBlock *CopyBackwardsBB = ThenTerm->getParent();
  CopyBackwardsBB->setName("copy_backwards");
  BasicBlock *CopyForwardBB = ElseTerm->getParent();
  CopyForwardBB->setName("copy_forward");
  BasicBlock *ExitBB = ConvertedInst->getParent();
  ExitBB->setName("memmove_done");

  // Initial comparison of n == 0 that lets us skip the loops altogether. Shared
  // between both backwards and forward copy clauses.
  ICmpInst *CompareN =
      new ICmpInst(OrigBB->getTerminator(), ICmpInst::ICMP_EQ, CopyLen,
                   ConstantInt::get(TypeOfCopyLen, 0), "compare_n_to_0");

  // Copying backwards.
  BasicBlock *LoopBB =
      BasicBlock::Create(Context, "copy_backwards_loop", &F, CopyForwardBB);
  IRBuilder<> LoopBuilder(LoopBB);
  PHINode *LoopPhi = LoopBuilder.CreatePHI(TypeOfCopyLen, 0);
  Value *IndexPtr = LoopBuilder.CreateSub(
      LoopPhi, ConstantInt::get(TypeOfCopyLen, 1), "index_ptr");
  Value *Element = LoopBuilder.CreateLoad(
      LoopBuilder.CreateInBoundsGEP(SrcAddr, IndexPtr), "element");
  LoopBuilder.CreateStore(Element,
                          LoopBuilder.CreateInBoundsGEP(DstAddr, IndexPtr));
  LoopBuilder.CreateCondBr(
      LoopBuilder.CreateICmpEQ(IndexPtr, ConstantInt::get(TypeOfCopyLen, 0)),
      ExitBB, LoopBB);
  LoopPhi->addIncoming(IndexPtr, LoopBB);
  LoopPhi->addIncoming(CopyLen, CopyBackwardsBB);
  BranchInst::Create(ExitBB, LoopBB, CompareN, ThenTerm);
  ThenTerm->eraseFromParent();

  // Copying forward.
  BasicBlock *FwdLoopBB =
      BasicBlock::Create(Context, "copy_forward_loop", &F, ExitBB);
  IRBuilder<> FwdLoopBuilder(FwdLoopBB);
  PHINode *FwdCopyPhi = FwdLoopBuilder.CreatePHI(TypeOfCopyLen, 0, "index_ptr");
  Value *FwdElement = FwdLoopBuilder.CreateLoad(
      FwdLoopBuilder.CreateInBoundsGEP(SrcAddr, FwdCopyPhi), "element");
  FwdLoopBuilder.CreateStore(
      FwdElement, FwdLoopBuilder.CreateInBoundsGEP(DstAddr, FwdCopyPhi));
  Value *FwdIndexPtr = FwdLoopBuilder.CreateAdd(
      FwdCopyPhi, ConstantInt::get(TypeOfCopyLen, 1), "index_increment");
  FwdLoopBuilder.CreateCondBr(FwdLoopBuilder.CreateICmpEQ(FwdIndexPtr, CopyLen),
                              ExitBB, FwdLoopBB);
  FwdCopyPhi->addIncoming(FwdIndexPtr, FwdLoopBB);
  FwdCopyPhi->addIncoming(ConstantInt::get(TypeOfCopyLen, 0), CopyForwardBB);

  BranchInst::Create(ExitBB, FwdLoopBB, CompareN, ElseTerm);
  ElseTerm->eraseFromParent();
}

// Lower memset to loop.
void convertMemSetToLoop(Instruction *ConvertedInst, Value *DstAddr,
                         Value *CopyLen, Value *SetValue, LLVMContext &Context,
                         Function &F) {
  BasicBlock *OrigBB = ConvertedInst->getParent();
  BasicBlock *NewBB =
      ConvertedInst->getParent()->splitBasicBlock(ConvertedInst, "split");
  BasicBlock *LoopBB = BasicBlock::Create(Context, "loadstoreloop", &F, NewBB);

  OrigBB->getTerminator()->setSuccessor(0, LoopBB);
  IRBuilder<> Builder(OrigBB->getTerminator());

  // Cast pointer to the type of value getting stored
  unsigned dstAS = cast<PointerType>(DstAddr->getType())->getAddressSpace();
  DstAddr = Builder.CreateBitCast(DstAddr,
                                  PointerType::get(SetValue->getType(), dstAS));

  IRBuilder<> LoopBuilder(LoopBB);
  PHINode *LoopIndex = LoopBuilder.CreatePHI(CopyLen->getType(), 0);
  LoopIndex->addIncoming(ConstantInt::get(CopyLen->getType(), 0), OrigBB);

  LoopBuilder.CreateStore(
      SetValue,
      LoopBuilder.CreateInBoundsGEP(SetValue->getType(), DstAddr, LoopIndex),
      false);

  Value *NewIndex =
      LoopBuilder.CreateAdd(LoopIndex, ConstantInt::get(CopyLen->getType(), 1));
  LoopIndex->addIncoming(NewIndex, LoopBB);

  LoopBuilder.CreateCondBr(LoopBuilder.CreateICmpULT(NewIndex, CopyLen), LoopBB,
                           NewBB);
}

bool NVPTXLowerAggrCopies::runOnFunction(Function &F) {
  SmallVector<LoadInst *, 4> AggrLoads;
  SmallVector<MemIntrinsic *, 4> MemCalls;

  const DataLayout &DL = F.getParent()->getDataLayout();
  LLVMContext &Context = F.getParent()->getContext();

  // Collect all aggregate loads and mem* calls.
  for (Function::iterator BI = F.begin(), BE = F.end(); BI != BE; ++BI) {
    for (BasicBlock::iterator II = BI->begin(), IE = BI->end(); II != IE;
         ++II) {
      if (LoadInst *LI = dyn_cast<LoadInst>(II)) {
        if (!LI->hasOneUse())
          continue;

        if (DL.getTypeStoreSize(LI->getType()) < MaxAggrCopySize)
          continue;

        if (StoreInst *SI = dyn_cast<StoreInst>(LI->user_back())) {
          if (SI->getOperand(0) != LI)
            continue;
          AggrLoads.push_back(LI);
        }
      } else if (MemIntrinsic *IntrCall = dyn_cast<MemIntrinsic>(II)) {
        // Convert intrinsic calls with variable size or with constant size
        // larger than the MaxAggrCopySize threshold.
        if (ConstantInt *LenCI = dyn_cast<ConstantInt>(IntrCall->getLength())) {
          if (LenCI->getZExtValue() >= MaxAggrCopySize) {
            MemCalls.push_back(IntrCall);
          }
        } else {
          MemCalls.push_back(IntrCall);
        }
      }
    }
  }

  if (AggrLoads.size() == 0 && MemCalls.size() == 0) {
    return false;
  }

  //
  // Do the transformation of an aggr load/copy/set to a loop
  //
  for (LoadInst *LI : AggrLoads) {
    StoreInst *SI = dyn_cast<StoreInst>(*LI->user_begin());
    Value *SrcAddr = LI->getOperand(0);
    Value *DstAddr = SI->getOperand(1);
    unsigned NumLoads = DL.getTypeStoreSize(LI->getType());
    Value *CopyLen = ConstantInt::get(Type::getInt32Ty(Context), NumLoads);

    convertMemCpyToLoop(/* ConvertedInst */ SI,
                        /* SrcAddr */ SrcAddr, /* DstAddr */ DstAddr,
                        /* CopyLen */ CopyLen,
                        /* SrcIsVolatile */ LI->isVolatile(),
                        /* DstIsVolatile */ SI->isVolatile(),
                        /* Context */ Context,
                        /* Function F */ F);

    SI->eraseFromParent();
    LI->eraseFromParent();
  }

  // Transform mem* intrinsic calls.
  for (MemIntrinsic *MemCall : MemCalls) {
    if (MemCpyInst *Memcpy = dyn_cast<MemCpyInst>(MemCall)) {
      convertMemCpyToLoop(/* ConvertedInst */ Memcpy,
                          /* SrcAddr */ Memcpy->getRawSource(),
                          /* DstAddr */ Memcpy->getRawDest(),
                          /* CopyLen */ Memcpy->getLength(),
                          /* SrcIsVolatile */ Memcpy->isVolatile(),
                          /* DstIsVolatile */ Memcpy->isVolatile(),
                          /* Context */ Context,
                          /* Function F */ F);
    } else if (MemMoveInst *Memmove = dyn_cast<MemMoveInst>(MemCall)) {
      convertMemMoveToLoop(/* ConvertedInst */ Memmove,
                           /* SrcAddr */ Memmove->getRawSource(),
                           /* DstAddr */ Memmove->getRawDest(),
                           /* CopyLen */ Memmove->getLength(),
                           /* SrcIsVolatile */ Memmove->isVolatile(),
                           /* DstIsVolatile */ Memmove->isVolatile(),
                           /* Context */ Context,
                           /* Function F */ F);

    } else if (MemSetInst *Memset = dyn_cast<MemSetInst>(MemCall)) {
      convertMemSetToLoop(/* ConvertedInst */ Memset,
                          /* DstAddr */ Memset->getRawDest(),
                          /* CopyLen */ Memset->getLength(),
                          /* SetValue */ Memset->getValue(),
                          /* Context */ Context,
                          /* Function F */ F);
    }
    MemCall->eraseFromParent();
  }

  return true;
}

} // namespace

namespace llvm {
void initializeNVPTXLowerAggrCopiesPass(PassRegistry &);
}

INITIALIZE_PASS(NVPTXLowerAggrCopies, "nvptx-lower-aggr-copies",
                "Lower aggregate copies, and llvm.mem* intrinsics into loops",
                false, false)

FunctionPass *llvm::createLowerAggrCopies() {
  return new NVPTXLowerAggrCopies();
}
