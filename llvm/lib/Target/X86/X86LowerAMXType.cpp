//===- llvm/CodeGen/TileShapeInfo.h - ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Pass to transform <256 x i32> load/store
/// <256 x i32> is bitcasted to x86_amx on X86, and AMX instruction set only
/// provides simple operation on x86_amx. The basic elementwise operation
/// is not supported by AMX. Since x86_amx is bitcasted from vector <256 x i32>
/// and only AMX intrinsics can operate on the type, we need transform
/// load/store <256 x i32> instruction to AMX load/store. If the bitcast can
/// not be combined with load/store, we transform the bitcast to amx load/store
/// and <256 x i32> store/load.
//
//===----------------------------------------------------------------------===//
//
#include "X86.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "lower-amx-type"

static AllocaInst *CreateAllocaInst(IRBuilder<> &Builder, BasicBlock *BB) {
  Function &F = *BB->getParent();
  Module *M = BB->getModule();
  const DataLayout &DL = M->getDataLayout();

  Type *V256I32Ty = VectorType::get(Builder.getInt32Ty(), 256, false);
  LLVMContext &Ctx = Builder.getContext();
  auto AllocaAlignment = DL.getPrefTypeAlign(Type::getX86_AMXTy(Ctx));
  unsigned AllocaAS = DL.getAllocaAddrSpace();
  AllocaInst *AllocaRes =
      new AllocaInst(V256I32Ty, AllocaAS, "", &F.getEntryBlock().front());
  AllocaRes->setAlignment(AllocaAlignment);
  return AllocaRes;
}

static std::pair<Value *, Value *> getShape(IntrinsicInst *II, unsigned OpNo) {
  Value *Row = nullptr, *Col = nullptr;
  switch (II->getIntrinsicID()) {
  default:
    llvm_unreachable("Expect amx intrinsics");
  case Intrinsic::x86_tileloadd64_internal:
  case Intrinsic::x86_tilestored64_internal: {
    Row = II->getArgOperand(0);
    Col = II->getArgOperand(1);
    break;
  }
  // a * b + c
  // The shape depends on which operand.
  case Intrinsic::x86_tdpbssd_internal:
  case Intrinsic::x86_tdpbsud_internal:
  case Intrinsic::x86_tdpbusd_internal:
  case Intrinsic::x86_tdpbuud_internal:
  case Intrinsic::x86_tdpbf16ps_internal: {
    switch (OpNo) {
    case 3:
      Row = II->getArgOperand(0);
      Col = II->getArgOperand(1);
      break;
    case 4:
      Row = II->getArgOperand(0);
      Col = II->getArgOperand(2);
      break;
    case 5:
      Row = II->getArgOperand(2);
      Col = II->getArgOperand(1);
      break;
    }
    break;
  }
  }

  return std::make_pair(Row, Col);
}

// %src = load <256 x i32>, <256 x i32>* %addr, align 64
// %2 = bitcast <256 x i32> %src to x86_amx
// -->
// %2 = call x86_amx @llvm.x86.tileloadd64.internal(i16 %row, i16 %col,
// i8* %addr, i64 %stride64)
static void combineLoadBitcast(LoadInst *LD, BitCastInst *Bitcast) {
  Value *Row = nullptr, *Col = nullptr;
  Use &U = *(Bitcast->use_begin());
  unsigned OpNo = U.getOperandNo();
  auto *II = cast<IntrinsicInst>(U.getUser());
  std::tie(Row, Col) = getShape(II, OpNo);
  IRBuilder<> Builder(Bitcast);
  // Use the maximun column as stride.
  Value *Stride = Builder.getInt64(64);
  Value *I8Ptr =
      Builder.CreateBitCast(LD->getOperand(0), Builder.getInt8PtrTy());
  std::array<Value *, 4> Args = {Row, Col, I8Ptr, Stride};

  Value *NewInst =
      Builder.CreateIntrinsic(Intrinsic::x86_tileloadd64_internal, None, Args);
  Bitcast->replaceAllUsesWith(NewInst);
}

// %src = call x86_amx @llvm.x86.tileloadd64.internal(%row, %col, %addr,
//                                                    %stride);
// %13 = bitcast x86_amx %src to <256 x i32>
// store <256 x i32> %13, <256 x i32>* %addr, align 64
// -->
// call void @llvm.x86.tilestored64.internal(%row, %col, %addr,
//                                           %stride64, %13)
static void combineBitcastStore(BitCastInst *Bitcast, StoreInst *ST) {

  Value *Tile = Bitcast->getOperand(0);
  auto *II = cast<IntrinsicInst>(Tile);
  // Tile is output from AMX intrinsic. The first operand of the
  // intrinsic is row, the second operand of the intrinsic is column.
  Value *Row = II->getOperand(0);
  Value *Col = II->getOperand(1);
  IRBuilder<> Builder(ST);
  // Use the maximum column as stride. It must be the same with load
  // stride.
  Value *Stride = Builder.getInt64(64);
  Value *I8Ptr =
      Builder.CreateBitCast(ST->getOperand(1), Builder.getInt8PtrTy());
  std::array<Value *, 5> Args = {Row, Col, I8Ptr, Stride, Tile};
  Builder.CreateIntrinsic(Intrinsic::x86_tilestored64_internal, None, Args);
  if (Bitcast->hasOneUse())
    return;
  // %13 = bitcast x86_amx %src to <256 x i32>
  // store <256 x i32> %13, <256 x i32>* %addr, align 64
  // %add = <256 x i32> %13, <256 x i32> %src2
  // -->
  // %13 = bitcast x86_amx %src to <256 x i32>
  // call void @llvm.x86.tilestored64.internal(%row, %col, %addr,
  //                                           %stride64, %13)
  // %14 = load <256 x i32>, %addr
  // %add = <256 x i32> %14, <256 x i32> %src2
  Value *Vec = Builder.CreateLoad(Bitcast->getType(), ST->getOperand(1));
  Bitcast->replaceAllUsesWith(Vec);
}

// transform bitcast to <store, load> instructions.
static bool transformBitcast(BitCastInst *Bitcast) {
  IRBuilder<> Builder(Bitcast);
  AllocaInst *AllocaAddr;
  Value *I8Ptr, *Stride;
  auto *Src = Bitcast->getOperand(0);

  auto Prepare = [&]() {
    AllocaAddr = CreateAllocaInst(Builder, Bitcast->getParent());
    I8Ptr = Builder.CreateBitCast(AllocaAddr, Builder.getInt8PtrTy());
    Stride = Builder.getInt64(64);
  };

  if (Bitcast->getType()->isX86_AMXTy()) {
    // %2 = bitcast <256 x i32> %src to x86_amx
    // -->
    // %addr = alloca <256 x i32>, align 64
    // store <256 x i32> %src, <256 x i32>* %addr, align 64
    // %addr2 = bitcast <256 x i32>* to i8*
    // %2 = call x86_amx @llvm.x86.tileloadd64.internal(i16 %row, i16 %col,
    //                                                  i8* %addr2,
    //                                                  i64 64)
    Use &U = *(Bitcast->use_begin());
    unsigned OpNo = U.getOperandNo();
    auto *II = dyn_cast<IntrinsicInst>(U.getUser());
    if (!II)
      return false; // May be bitcast from x86amx to <256 x i32>.
    Prepare();
    Builder.CreateStore(Src, AllocaAddr);
    // TODO we can pick an constant operand for the shape.
    Value *Row = nullptr, *Col = nullptr;
    std::tie(Row, Col) = getShape(II, OpNo);
    std::array<Value *, 4> Args = {Row, Col, I8Ptr, Stride};
    Value *NewInst = Builder.CreateIntrinsic(
        Intrinsic::x86_tileloadd64_internal, None, Args);
    Bitcast->replaceAllUsesWith(NewInst);
  } else {
    // %2 = bitcast x86_amx %src to <256 x i32>
    // -->
    // %addr = alloca <256 x i32>, align 64
    // %addr2 = bitcast <256 x i32>* to i8*
    // call void @llvm.x86.tilestored64.internal(i16 %row, i16 %col,
    //                                           i8* %addr2, i64 %stride)
    // %2 = load <256 x i32>, <256 x i32>* %addr, align 64
    auto *II = dyn_cast<IntrinsicInst>(Src);
    if (!II)
      return false; // May be bitcast from <256 x i32> to x86amx.
    Prepare();
    Value *Row = II->getOperand(0);
    Value *Col = II->getOperand(1);
    std::array<Value *, 5> Args = {Row, Col, I8Ptr, Stride, Src};
    Builder.CreateIntrinsic(Intrinsic::x86_tilestored64_internal, None, Args);
    Value *NewInst = Builder.CreateLoad(Bitcast->getType(), AllocaAddr);
    Bitcast->replaceAllUsesWith(NewInst);
  }

  return true;
}

namespace {
class X86LowerAMXType {
  Function &Func;

public:
  X86LowerAMXType(Function &F) : Func(F) {}
  bool visit();
};

bool X86LowerAMXType::visit() {
  SmallVector<Instruction *, 8> DeadInsts;

  for (BasicBlock *BB : post_order(&Func)) {
    for (BasicBlock::reverse_iterator II = BB->rbegin(), IE = BB->rend();
         II != IE;) {
      Instruction &Inst = *II++;
      auto *Bitcast = dyn_cast<BitCastInst>(&Inst);
      if (!Bitcast)
        continue;

      Value *Src = Bitcast->getOperand(0);
      if (Bitcast->getType()->isX86_AMXTy()) {
        if (Bitcast->user_empty()) {
          DeadInsts.push_back(Bitcast);
          continue;
        }
        LoadInst *LD = dyn_cast<LoadInst>(Src);
        if (!LD) {
          if (transformBitcast(Bitcast))
            DeadInsts.push_back(Bitcast);
          continue;
        }
        // If load has mutli-user, duplicate a vector load.
        // %src = load <256 x i32>, <256 x i32>* %addr, align 64
        // %2 = bitcast <256 x i32> %src to x86_amx
        // %add = add <256 x i32> %src, <256 x i32> %src2
        // -->
        // %src = load <256 x i32>, <256 x i32>* %addr, align 64
        // %2 = call x86_amx @llvm.x86.tileloadd64.internal(i16 %row, i16 %col,
        //                                            i8* %addr, i64 %stride64)
        // %add = add <256 x i32> %src, <256 x i32> %src2

        // If load has one user, the load will be eliminated in DAG ISel.
        // %src = load <256 x i32>, <256 x i32>* %addr, align 64
        // %2 = bitcast <256 x i32> %src to x86_amx
        // -->
        // %2 = call x86_amx @llvm.x86.tileloadd64.internal(i16 %row, i16 %col,
        //                                            i8* %addr, i64 %stride64)
        combineLoadBitcast(LD, Bitcast);
        DeadInsts.push_back(Bitcast);
        if (LD->hasOneUse())
          DeadInsts.push_back(LD);
      } else if (Src->getType()->isX86_AMXTy()) {
        if (Bitcast->user_empty()) {
          DeadInsts.push_back(Bitcast);
          continue;
        }
        StoreInst *ST = nullptr;
        for (auto UI = Bitcast->use_begin(), UE = Bitcast->use_end();
             UI != UE;) {
          Value *I = (UI++)->getUser();
          ST = dyn_cast<StoreInst>(I);
          if (ST)
            break;
        }
        if (!ST) {
          if (transformBitcast(Bitcast))
            DeadInsts.push_back(Bitcast);
          continue;
        }
        // If bitcast (%13) has one use, combine bitcast and store to amx store.
        // %src = call x86_amx @llvm.x86.tileloadd64.internal(%row, %col, %addr,
        //                                                    %stride);
        // %13 = bitcast x86_amx %src to <256 x i32>
        // store <256 x i32> %13, <256 x i32>* %addr, align 64
        // -->
        // call void @llvm.x86.tilestored64.internal(%row, %col, %addr,
        //                                           %stride64, %13)
        //
        // If bitcast (%13) has multi-use, transform as below.
        // %13 = bitcast x86_amx %src to <256 x i32>
        // store <256 x i32> %13, <256 x i32>* %addr, align 64
        // %add = <256 x i32> %13, <256 x i32> %src2
        // -->
        // %13 = bitcast x86_amx %src to <256 x i32>
        // call void @llvm.x86.tilestored64.internal(%row, %col, %addr,
        //                                           %stride64, %13)
        // %14 = load <256 x i32>, %addr
        // %add = <256 x i32> %14, <256 x i32> %src2
        //
        combineBitcastStore(Bitcast, ST);
        // Delete user first.
        DeadInsts.push_back(ST);
        DeadInsts.push_back(Bitcast);
      }
    }
  }

  bool C = !DeadInsts.empty();

  for (auto *Inst : DeadInsts)
    Inst->eraseFromParent();

  return C;
}
} // anonymous namespace

namespace {

class X86LowerAMXTypeLegacyPass : public FunctionPass {
public:
  static char ID;

  X86LowerAMXTypeLegacyPass() : FunctionPass(ID) {
    initializeX86LowerAMXTypeLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    TargetMachine *TM = &getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
    if (F.hasFnAttribute(Attribute::OptimizeNone) ||
        TM->getOptLevel() == CodeGenOpt::None)
      return false;
    X86LowerAMXType LAT(F);
    bool C = LAT.visit();
    return C;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<TargetPassConfig>();
  }
};

} // anonymous namespace

static const char PassName[] = "Lower AMX type for load/store";
char X86LowerAMXTypeLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(X86LowerAMXTypeLegacyPass, DEBUG_TYPE, PassName, false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(X86LowerAMXTypeLegacyPass, DEBUG_TYPE, PassName, false,
                    false)

FunctionPass *llvm::createX86LowerAMXTypePass() {
  return new X86LowerAMXTypeLegacyPass();
}
