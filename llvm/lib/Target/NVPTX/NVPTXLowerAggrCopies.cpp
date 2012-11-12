//===- NVPTXLowerAggrCopies.cpp - ------------------------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Lower aggregate copies, memset, memcpy, memmov intrinsics into loops when
// the size is large or is not a compile-time constant.
//
//===----------------------------------------------------------------------===//

#include "NVPTXLowerAggrCopies.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/IRBuilder.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Intrinsics.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/DataLayout.h"

using namespace llvm;

namespace llvm {
FunctionPass *createLowerAggrCopies();
}

char NVPTXLowerAggrCopies::ID = 0;

// Lower MemTransferInst or load-store pair to loop
static void convertTransferToLoop(Instruction *splitAt, Value *srcAddr,
                                  Value *dstAddr, Value *len,
                                  //unsigned numLoads,
                                  bool srcVolatile, bool dstVolatile,
                                  LLVMContext &Context, Function &F) {
  Type *indType = len->getType();

  BasicBlock *origBB = splitAt->getParent();
  BasicBlock *newBB = splitAt->getParent()->splitBasicBlock(splitAt, "split");
  BasicBlock *loopBB = BasicBlock::Create(Context, "loadstoreloop", &F, newBB);

  origBB->getTerminator()->setSuccessor(0, loopBB);
  IRBuilder<> builder(origBB, origBB->getTerminator());

  // srcAddr and dstAddr are expected to be pointer types,
  // so no check is made here.
  unsigned srcAS =
      dyn_cast<PointerType>(srcAddr->getType())->getAddressSpace();
  unsigned dstAS =
      dyn_cast<PointerType>(dstAddr->getType())->getAddressSpace();

  // Cast pointers to (char *)
  srcAddr = builder.CreateBitCast(srcAddr, Type::getInt8PtrTy(Context, srcAS));
  dstAddr = builder.CreateBitCast(dstAddr, Type::getInt8PtrTy(Context, dstAS));

  IRBuilder<> loop(loopBB);
  // The loop index (ind) is a phi node.
  PHINode *ind = loop.CreatePHI(indType, 0);
  // Incoming value for ind is 0
  ind->addIncoming(ConstantInt::get(indType, 0), origBB);

  // load from srcAddr+ind
  Value *val = loop.CreateLoad(loop.CreateGEP(srcAddr, ind), srcVolatile);
  // store at dstAddr+ind
  loop.CreateStore(val, loop.CreateGEP(dstAddr, ind), dstVolatile);

  // The value for ind coming from backedge is (ind + 1)
  Value *newind = loop.CreateAdd(ind, ConstantInt::get(indType, 1));
  ind->addIncoming(newind, loopBB);

  loop.CreateCondBr(loop.CreateICmpULT(newind, len), loopBB, newBB);
}

// Lower MemSetInst to loop
static void convertMemSetToLoop(Instruction *splitAt, Value *dstAddr,
                                Value *len, Value *val, LLVMContext &Context,
                                Function &F) {
  BasicBlock *origBB = splitAt->getParent();
  BasicBlock *newBB = splitAt->getParent()->splitBasicBlock(splitAt, "split");
  BasicBlock *loopBB = BasicBlock::Create(Context, "loadstoreloop", &F, newBB);

  origBB->getTerminator()->setSuccessor(0, loopBB);
  IRBuilder<> builder(origBB, origBB->getTerminator());

  unsigned dstAS =
      dyn_cast<PointerType>(dstAddr->getType())->getAddressSpace();

  // Cast pointer to the type of value getting stored
  dstAddr = builder.CreateBitCast(dstAddr,
                                  PointerType::get(val->getType(), dstAS));

  IRBuilder<> loop(loopBB);
  PHINode *ind = loop.CreatePHI(len->getType(), 0);
  ind->addIncoming(ConstantInt::get(len->getType(), 0), origBB);

  loop.CreateStore(val, loop.CreateGEP(dstAddr, ind), false);

  Value *newind = loop.CreateAdd(ind, ConstantInt::get(len->getType(), 1));
  ind->addIncoming(newind, loopBB);

  loop.CreateCondBr(loop.CreateICmpULT(newind, len), loopBB, newBB);
}

bool NVPTXLowerAggrCopies::runOnFunction(Function &F) {
  SmallVector<LoadInst *, 4> aggrLoads;
  SmallVector<MemTransferInst *, 4> aggrMemcpys;
  SmallVector<MemSetInst *, 4> aggrMemsets;

  DataLayout *TD = &getAnalysis<DataLayout>();
  LLVMContext &Context = F.getParent()->getContext();

  //
  // Collect all the aggrLoads, aggrMemcpys and addrMemsets.
  //
  //const BasicBlock *firstBB = &F.front();  // first BB in F
  for (Function::iterator BI = F.begin(), BE = F.end(); BI != BE; ++BI) {
    //BasicBlock *bb = BI;
    for (BasicBlock::iterator II = BI->begin(), IE = BI->end(); II != IE;
        ++II) {
      if (LoadInst * load = dyn_cast<LoadInst>(II)) {

        if (load->hasOneUse() == false) continue;

        if (TD->getTypeStoreSize(load->getType()) < MaxAggrCopySize) continue;

        User *use = *(load->use_begin());
        if (StoreInst * store = dyn_cast<StoreInst>(use)) {
          if (store->getOperand(0) != load) //getValueOperand
          continue;
          aggrLoads.push_back(load);
        }
      } else if (MemTransferInst * intr = dyn_cast<MemTransferInst>(II)) {
        Value *len = intr->getLength();
        // If the number of elements being copied is greater
        // than MaxAggrCopySize, lower it to a loop
        if (ConstantInt * len_int = dyn_cast < ConstantInt > (len)) {
          if (len_int->getZExtValue() >= MaxAggrCopySize) {
            aggrMemcpys.push_back(intr);
          }
        } else {
          // turn variable length memcpy/memmov into loop
          aggrMemcpys.push_back(intr);
        }
      } else if (MemSetInst * memsetintr = dyn_cast<MemSetInst>(II)) {
        Value *len = memsetintr->getLength();
        if (ConstantInt * len_int = dyn_cast<ConstantInt>(len)) {
          if (len_int->getZExtValue() >= MaxAggrCopySize) {
            aggrMemsets.push_back(memsetintr);
          }
        } else {
          // turn variable length memset into loop
          aggrMemsets.push_back(memsetintr);
        }
      }
    }
  }
  if ((aggrLoads.size() == 0) && (aggrMemcpys.size() == 0)
      && (aggrMemsets.size() == 0)) return false;

  //
  // Do the transformation of an aggr load/copy/set to a loop
  //
  for (unsigned i = 0, e = aggrLoads.size(); i != e; ++i) {
    LoadInst *load = aggrLoads[i];
    StoreInst *store = dyn_cast<StoreInst>(*load->use_begin());
    Value *srcAddr = load->getOperand(0);
    Value *dstAddr = store->getOperand(1);
    unsigned numLoads = TD->getTypeStoreSize(load->getType());
    Value *len = ConstantInt::get(Type::getInt32Ty(Context), numLoads);

    convertTransferToLoop(store, srcAddr, dstAddr, len, load->isVolatile(),
                          store->isVolatile(), Context, F);

    store->eraseFromParent();
    load->eraseFromParent();
  }

  for (unsigned i = 0, e = aggrMemcpys.size(); i != e; ++i) {
    MemTransferInst *cpy = aggrMemcpys[i];
    Value *len = cpy->getLength();
    // llvm 2.7 version of memcpy does not have volatile
    // operand yet. So always making it non-volatile
    // optimistically, so that we don't see unnecessary
    // st.volatile in ptx
    convertTransferToLoop(cpy, cpy->getSource(), cpy->getDest(), len, false,
                          false, Context, F);
    cpy->eraseFromParent();
  }

  for (unsigned i = 0, e = aggrMemsets.size(); i != e; ++i) {
    MemSetInst *memsetinst = aggrMemsets[i];
    Value *len = memsetinst->getLength();
    Value *val = memsetinst->getValue();
    convertMemSetToLoop(memsetinst, memsetinst->getDest(), len, val, Context,
                        F);
    memsetinst->eraseFromParent();
  }

  return true;
}

FunctionPass *llvm::createLowerAggrCopies() {
  return new NVPTXLowerAggrCopies();
}
