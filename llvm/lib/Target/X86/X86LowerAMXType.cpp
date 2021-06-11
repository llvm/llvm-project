//===- Target/X86/X86LowerAMXType.cpp - -------------------------*- C++ -*-===//
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
///
/// If Front End not use O0 but the Mid/Back end use O0, (e.g. "Clang -O2 -S
/// -emit-llvm t.c" + "llc t.ll") we should make sure the amx data is volatile,
/// because that is necessary for AMX fast register allocation. (In Fast
/// registera allocation, register will be allocated before spill/reload, so
/// there is no additional register for amx to identify the step in spill.)
/// The volatileTileData() will handle this case.
/// e.g.
/// ----------------------------------------------------------
/// | def %td = ...                                          |
/// | ...                                                    |
/// | "use %td"                                              |
/// ----------------------------------------------------------
/// will transfer to -->
/// ----------------------------------------------------------
/// | def %td = ...                                          |
/// | call void @llvm.x86.tilestored64.internal(mem, %td)    |
/// | ...                                                    |
/// | %td2 = call x86_amx @llvm.x86.tileloadd64.internal(mem)|
/// | "use %td2"                                             |
/// ----------------------------------------------------------
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

static AllocaInst *createAllocaInstAtEntry(IRBuilder<> &Builder,
                                           BasicBlock *BB) {
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

namespace {
class X86LowerAMXType {
  Function &Func;
  TargetMachine *TM = nullptr;

  // In AMX intrinsics we let Shape = {Row, Col}, but the
  // RealCol = Col / ElementSize. We may use the RealCol
  // as a new Row for other new created AMX intrinsics.
  std::map<Value *, Value *> Col2Row;

public:
  X86LowerAMXType(Function &F, TargetMachine *TargetM) : Func(F), TM(TargetM) {}
  bool visit();
  void combineLoadBitcast(LoadInst *LD, BitCastInst *Bitcast);
  void combineBitcastStore(BitCastInst *Bitcast, StoreInst *ST);
  bool transformBitcast(BitCastInst *Bitcast);
  std::pair<Value *, Value *> getShape(IntrinsicInst *II, unsigned OpNo);
  Value *getRowFromCol(Instruction *II, Value *V, unsigned Granularity);
};

Value *X86LowerAMXType::getRowFromCol(Instruction *II, Value *V,
                                      unsigned Granularity) {
  if (Col2Row.count(V))
    return Col2Row[V];
  IRBuilder<> Builder(&*II->getParent()->getFirstInsertionPt());
  if (auto *I = dyn_cast<Instruction>(V)) {
    BasicBlock::iterator Iter = I->getIterator();
    ++Iter;
    Builder.SetInsertPoint(&*Iter);
  }
  ConstantInt *Gran = Builder.getInt16(Granularity);
  Value *RealRow = Builder.CreateUDiv(V, Gran);
  Col2Row[V] = RealRow;
  return RealRow;
}

std::pair<Value *, Value *> X86LowerAMXType::getShape(IntrinsicInst *II,
                                                      unsigned OpNo) {
  Value *Row = nullptr, *Col = nullptr;
  switch (II->getIntrinsicID()) {
  default:
    llvm_unreachable("Expect amx intrinsics");
  case Intrinsic::x86_tileloadd64_internal:
  case Intrinsic::x86_tileloaddt164_internal:
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
      // FIXME: There is a design bug for AMX shape, which the Col should be
      // Col/4 if it will be used as Row, but current Greedy RA can't handle
      // this case well, it may failed if we generate a new Shape definition.
      // So Let's just do it in O0 first.
      // Row = Row / 4
      if (TM->getOptLevel() == CodeGenOpt::None)
        Row = getRowFromCol(II, Row, 4);
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
void X86LowerAMXType::combineLoadBitcast(LoadInst *LD, BitCastInst *Bitcast) {
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
void X86LowerAMXType::combineBitcastStore(BitCastInst *Bitcast, StoreInst *ST) {

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
bool X86LowerAMXType::transformBitcast(BitCastInst *Bitcast) {
  IRBuilder<> Builder(Bitcast);
  AllocaInst *AllocaAddr;
  Value *I8Ptr, *Stride;
  auto *Src = Bitcast->getOperand(0);

  auto Prepare = [&]() {
    AllocaAddr = createAllocaInstAtEntry(Builder, Bitcast->getParent());
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

bool X86LowerAMXType::visit() {
  SmallVector<Instruction *, 8> DeadInsts;
  Col2Row.clear();

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

static Value *getAllocaPos(BasicBlock *BB) {
  Module *M = BB->getModule();
  Function *F = BB->getParent();
  IRBuilder<> Builder(&F->getEntryBlock().front());
  const DataLayout &DL = M->getDataLayout();
  unsigned AllocaAS = DL.getAllocaAddrSpace();
  Type *V256I32Ty = VectorType::get(Builder.getInt32Ty(), 256, false);
  AllocaInst *AllocaRes =
      new AllocaInst(V256I32Ty, AllocaAS, "", &F->getEntryBlock().front());
  BasicBlock::iterator Iter = AllocaRes->getIterator();
  ++Iter;
  Builder.SetInsertPoint(&*Iter);
  Value *I8Ptr = Builder.CreateBitCast(AllocaRes, Builder.getInt8PtrTy());
  return I8Ptr;
}

static Instruction *createTileStore(Instruction *TileDef, Value *Ptr) {
  assert(TileDef->getType()->isX86_AMXTy() && "Not define tile!");
  auto *II = cast<IntrinsicInst>(TileDef);
  assert(II && "Not tile intrinsic!");
  Value *Row = II->getOperand(0);
  Value *Col = II->getOperand(1);

  BasicBlock *BB = TileDef->getParent();
  BasicBlock::iterator Iter = TileDef->getIterator();
  IRBuilder<> Builder(BB, ++Iter);
  Value *Stride = Builder.getInt64(64);
  std::array<Value *, 5> Args = {Row, Col, Ptr, Stride, TileDef};

  Instruction *TileStore =
      Builder.CreateIntrinsic(Intrinsic::x86_tilestored64_internal, None, Args);
  return TileStore;
}

static void replaceWithTileLoad(Use &U, Value *Ptr, bool IsPHI = false) {
  Value *V = U.get();
  assert(V->getType()->isX86_AMXTy() && "Not define tile!");

  // Get tile shape.
  IntrinsicInst *II = nullptr;
  if (IsPHI) {
    Value *PhiOp = dyn_cast<PHINode>(V)->getIncomingValue(0);
    II = cast<IntrinsicInst>(PhiOp);
  } else {
    II = cast<IntrinsicInst>(V);
  }
  Value *Row = II->getOperand(0);
  Value *Col = II->getOperand(1);

  Instruction *UserI = dyn_cast<Instruction>(U.getUser());
  IRBuilder<> Builder(UserI);
  Value *Stride = Builder.getInt64(64);
  std::array<Value *, 4> Args = {Row, Col, Ptr, Stride};

  Value *TileLoad =
      Builder.CreateIntrinsic(Intrinsic::x86_tileloadd64_internal, None, Args);
  UserI->replaceUsesOfWith(V, TileLoad);
}

static bool isIncomingOfPHI(Instruction *I) {
  for (Use &U : I->uses()) {
    User *V = U.getUser();
    if (isa<PHINode>(V))
      return true;
  }
  return false;
}

// Let all AMX tile data become volatile data, shorten the life range
// of each tile register before fast register allocation.
namespace {
class X86VolatileTileData {
  Function &F;

public:
  X86VolatileTileData(Function &Func) : F(Func) {}
  Value *updatePhiIncomings(BasicBlock *BB,
                            SmallVector<Instruction *, 2> &Incomings);
  void replacePhiDefWithLoad(Instruction *PHI, Value *StorePtr);
  bool volatileTileData();
  void volatileTilePHI(PHINode *Inst);
  void volatileTileNonPHI(Instruction *I);
};

Value *X86VolatileTileData::updatePhiIncomings(
    BasicBlock *BB, SmallVector<Instruction *, 2> &Incomings) {
  Value *I8Ptr = getAllocaPos(BB);

  for (auto *I : Incomings) {
    User *Store = createTileStore(I, I8Ptr);

    // All its uses (except phi) should load from stored mem.
    for (Use &U : I->uses()) {
      User *V = U.getUser();
      if (isa<PHINode>(V) || V == Store)
        continue;
      replaceWithTileLoad(U, I8Ptr);
    }
  }
  return I8Ptr;
}

void X86VolatileTileData::replacePhiDefWithLoad(Instruction *PHI,
                                                Value *StorePtr) {
  for (Use &U : PHI->uses())
    replaceWithTileLoad(U, StorePtr, true);
  PHI->eraseFromParent();
}

// Smilar with volatileTileNonPHI, this function only handle PHI Nodes
// and their related AMX intrinsics.
// 1) PHI Def should change to tileload.
// 2) PHI Incoming Values should tilestored in just after their def.
// 3) The mem of these tileload and tilestores should be same.
// e.g.
// ------------------------------------------------------
// bb_dom:
//   ...
//   br i1 %bool.cond, label %if.else, label %if.then
//
// if.then:
//   def %t0 = ...
//   ...
//   use %t0
//   ...
//   br label %if.end
//
// if.else:
//   def %t1 = ...
//   br label %if.end
//
// if.end:
//   %td = phi x86_amx [ %t1, %if.else ], [ %t0, %if.then ]
//   ...
//   use %td
// ------------------------------------------------------
// -->
// ------------------------------------------------------
// bb_entry:
//   %mem = alloca <256 x i32>, align 1024                  *
//   ...
// bb_dom:
//   ...
//   br i1 %bool.cond, label %if.else, label %if.then
//
// if.then:
//   def %t0 = ...
//   call void @llvm.x86.tilestored64.internal(mem, %t0)    *
//   ...
//   %t0` = call x86_amx @llvm.x86.tileloadd64.internal(mem)*
//   use %t0`                                               *
//   ...
//   br label %if.end
//
// if.else:
//   def %t1 = ...
//   call void @llvm.x86.tilestored64.internal(mem, %t1)    *
//   br label %if.end
//
// if.end:
//   ...
//   %td = call x86_amx @llvm.x86.tileloadd64.internal(mem) *
//   use %td
// ------------------------------------------------------
void X86VolatileTileData::volatileTilePHI(PHINode *PHI) {
  BasicBlock *BB = PHI->getParent();
  SmallVector<Instruction *, 2> Incomings;

  for (unsigned I = 0, E = PHI->getNumIncomingValues(); I != E; ++I) {
    Value *Op = PHI->getIncomingValue(I);
    Instruction *Inst = dyn_cast<Instruction>(Op);
    assert(Inst && "We shouldn't fold AMX instrution!");
    Incomings.push_back(Inst);
  }

  Value *StorePtr = updatePhiIncomings(BB, Incomings);
  replacePhiDefWithLoad(PHI, StorePtr);
}

// Store the defined tile and load it before use.
// All its users are not PHI.
// e.g.
// ------------------------------------------------------
// def %td = ...
// ...
// "use %td"
// ------------------------------------------------------
// -->
// ------------------------------------------------------
// def %td = ...
// call void @llvm.x86.tilestored64.internal(mem, %td)
// ...
// %td2 = call x86_amx @llvm.x86.tileloadd64.internal(mem)
// "use %td2"
// ------------------------------------------------------
void X86VolatileTileData::volatileTileNonPHI(Instruction *I) {
  BasicBlock *BB = I->getParent();
  Value *I8Ptr = getAllocaPos(BB);
  User *Store = createTileStore(I, I8Ptr);

  // All its uses should load from stored mem.
  for (Use &U : I->uses()) {
    User *V = U.getUser();
    assert(!isa<PHINode>(V) && "PHI Nodes should be excluded!");
    if (V != Store)
      replaceWithTileLoad(U, I8Ptr);
  }
}

// Volatile Tile Model:
// 1) All the uses of tile data comes from tileload in time.
// 2) All the defs of tile data tilestore into mem immediately.
// For example:
// --------------------------------------------------------------------------
// %t1 = call x86_amx @llvm.x86.tileloadd64.internal(m, k, ...)          key
// %t2 = call x86_amx @llvm.x86.tileloadd64.internal(k, n, ...)
// %t3 = call x86_amx @llvm.x86.tileloadd64.internal(m, n, ...)          amx
// %td = tail call x86_amx @llvm.x86.tdpbssd.internal(m, n, k, t1, t2, t3)
// call void @llvm.x86.tilestored64.internal(... td)                     area
// --------------------------------------------------------------------------
// 3) No terminator, call or other amx instructions in the key amx area.
bool X86VolatileTileData::volatileTileData() {
  bool Changed = false;
  for (BasicBlock &BB : F) {
    SmallVector<Instruction *, 2> PHIInsts;
    SmallVector<Instruction *, 8> AMXDefInsts;

    for (Instruction &I : BB) {
      if (!I.getType()->isX86_AMXTy())
        continue;
      if (isa<PHINode>(&I))
        PHIInsts.push_back(&I);
      else
        AMXDefInsts.push_back(&I);
    }

    // First we "volatile" the non-phi related amx intrinsics.
    for (Instruction *I : AMXDefInsts) {
      if (isIncomingOfPHI(I))
        continue;
      volatileTileNonPHI(I);
      Changed = true;
    }

    for (Instruction *I : PHIInsts) {
      volatileTilePHI(dyn_cast<PHINode>(I));
      Changed = true;
    }
  }
  return Changed;
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

    X86LowerAMXType LAT(F, TM);
    bool C = LAT.visit();

    // Prepare for fast register allocation at O0.
    // Todo: May better check the volatile model of AMX code, not just
    // by checking Attribute::OptimizeNone and CodeGenOpt::None.
    if (TM->getOptLevel() == CodeGenOpt::None) {
      // If Front End not use O0 but the Mid/Back end use O0, (e.g.
      // "Clang -O2 -S -emit-llvm t.c" + "llc t.ll") we should make
      // sure the amx data is volatile, that is nessary for AMX fast
      // register allocation.
      if (!F.hasFnAttribute(Attribute::OptimizeNone)) {
        X86VolatileTileData VTD(F);
        C = VTD.volatileTileData() || C;
      }
    }

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
