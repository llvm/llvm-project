//===- Target/X86/X86PreAMXConfig.cpp - ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// Insert tilecfg for each area of key AMX intrinsic.
/// All the key AMX intrinsic's tile operand must come from tileload. And the
/// def tile of key AMX intrinsic must be tilestored.
/// take tdpbssd for example:
/// --------------------------------------------------------------------------
/// %t1 = call x86_amx @llvm.x86.tileloadd64.internal(...)                key
/// %t2 = call x86_amx @llvm.x86.tileloadd64.internal(...)                 |
/// %t3 = call x86_amx @llvm.x86.tileloadd64.internal(...)                amx
/// %td = tail call x86_amx @llvm.x86.tdpbssd.internal(t1, t2, t3)         |
/// call void @llvm.x86.tilestored64.internal(... td)                     area
/// --------------------------------------------------------------------------
/// This pass will insert tilecfg before every key-amx-area, some like:
/// --------------------------------------------------------------------------
/// %cfgmem = alloca <16 x i32>, align 4                        * allocate mem
/// store <16 x i32> zeroinitializer, <16 x i32>* %cfgmem       * zero init
/// ...
/// ... pre-config shape of %t1                                 *
/// store volatile i8 %m, i8* %amx.tmm.0.shape.row, align 1     *
/// store volatile i16 %k, i16* %amx.tmm.0.shape.col, align 2   * pre-config
/// ...                                                         *
/// ... pre-config shape of %t2                                 * shapes
/// store volatile i8 %k, i8* %amx.tmm.1.shape.row, align 1     *
/// store volatile i16 %n, i16* %amx.tmm.1.shape.col, align 2   *
/// ...
/// call void @llvm.x86.ldtilecfg(i8* %cfgmem)                  * tile config
//
//===----------------------------------------------------------------------===//
//
#include "X86.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
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
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "pre-amx-config"

static bool isAMXIntrinsic(IntrinsicInst *II) {
  for (Value *Operand : II->operands())
    if (Operand->getType()->isX86_AMXTy())
      return true;
  return II->getType()->isX86_AMXTy();
}

static bool isTileLoad(IntrinsicInst *II) {
  return II->getIntrinsicID() == Intrinsic::x86_tileloadd64_internal ||
         II->getIntrinsicID() == Intrinsic::x86_tileloaddt164_internal;
}

static bool isTileStore(IntrinsicInst *II) {
  return II->getIntrinsicID() == Intrinsic::x86_tilestored64_internal;
}

#ifndef NDEBUG
static bool onlyTileDef(IntrinsicInst *II) {
  for (Value *Operand : II->operands())
    if (Operand->getType()->isX86_AMXTy())
      return false;
  return II->getType()->isX86_AMXTy();
}

static bool brokenVolatile(Instruction *I) {
  // Todo: it is weak to identify a normal call here.
  if ((isa<CallInst>(I) && !isa<IntrinsicInst>(I)) || I->isTerminator())
    return true;
  return false;
}
#endif

namespace {
class X86PreAMXConfig {
  using PosAndShapesMap = MapVector<Instruction *, SmallVector<Value *, 8>>;

  Function &F;

public:
  X86PreAMXConfig(Function &Func) : F(Func) {}
  bool preTileConfig();
  void addTileConfig(Instruction *ModelStart, SmallVector<Value *, 8> &Shapes);
  bool findConfigShapes(PosAndShapesMap &PosAndShapes);
  bool getKeyAMXShapes(IntrinsicInst *KeyAMX, SmallVector<Value *, 8> &Shapes);
  void preWriteTileCfg(Value *I8Ptr, IRBuilderBase &Builder,
                       SmallVector<Value *, 8> &Shapes);
  BasicBlock::iterator
  getShapesAndConfigPosEnd(BasicBlock::iterator Iter,
                           SmallVector<Value *, 8> &Shapes);
  bool checkVolatileModel(SmallSet<Value *, 4> &Loads, IntrinsicInst *Store,
                          IntrinsicInst *KeyAMX);
};

// Orderly write the shapes in tilecfg's mem. This maybe not right.
// Because the first shape may not corresponding to the first tmm register,
// so we need to handle at at X86FastTileConfig::materializeTileCfg()
// after register allocation.
// For example:
// --------------------------------------------------------------------------
// zeroinitialize tilecfg's mem (of ldtilecfg)
// --------------------------------------------------------------------------
// ... pre-config shape of %t1                                 *
// %amx.tmm.0.shape.row = getelementptr i8, i8* %mem, i64 48   *
// %amx.tmm.0.shape.col = getelementptr i16, i16* %mem, i64 16 *
// store volatile i8 %m, i8* %amx.tmm.0.shape.row, align 1     *
// store volatile i16 %k, i16* %amx.tmm.0.shape.col, align 2   * pre-config
// ...                                                         *
// ... pre-config shape of %t2                                 *
// %amx.tmm.1.shape.row = getelementptr i8, i8* %mem, i64 49   *
// %amx.tmm.1.shape.col = getelementptr i16, i16* %mem, i64 18 *
// store volatile i8 %k, i8* %amx.tmm.1.shape.row, align 1     * shapes
// store volatile i16 %n, i16* %amx.tmm.1.shape.col, align 2   *
// ...                                                         *
// ... pre-config shape of %t3                                 * of
// %amx.tmm.2.shape.row = getelementptr i8, i8* %mem, i64 50   *
// %amx.tmm.2.shape.col = getelementptr i16, i16* %mem, i64 20 *
// store volatile i8 %m, i8* %amx.tmm.2.shape.row, align 1     *
// store volatile i16 %n, i16* %amx.tmm.2.shape.col, align 2   *
// ...                                                         * tiles
// ... pre-config shape of %td                                 *
// %amx.tmm.3.shape.row = getelementptr i8, i8* %mem, i64 51   *
// %amx.tmm.3.shape.col = getelementptr i16, i16* %mem, i64 22 *
// store volatile i8 %m, i8* %amx.tmm.3.shape.row, align 1     *
// store volatile i16 %n, i16* %amx.tmm.3.shape.col, align 2   *
// --------------------------------------------------------------------------
// call void @llvm.x86.ldtilecfg(i8* %mem)                     * tile config
// --------------------------------------------------------------------------
// %t1 = call x86_amx @llvm.x86.tileloadd64.internal(m, k, ...)          key
// %t2 = call x86_amx @llvm.x86.tileloadd64.internal(k, n, ...)
// %t3 = call x86_amx @llvm.x86.tileloadd64.internal(m, n, ...)          amx
// %td = tail call x86_amx @llvm.x86.tdpbssd.internal(m, n, k, t1, t2, t3)
// call void @llvm.x86.tilestored64.internal(... td)                     area
// --------------------------------------------------------------------------
void X86PreAMXConfig::preWriteTileCfg(Value *I8Ptr, IRBuilderBase &Builder,
                                      SmallVector<Value *, 8> &Shapes) {
  LLVMContext &Ctx = Builder.getContext();
  Type *I8Ty = Type::getInt8Ty(Ctx);
  Type *I16Ty = Type::getInt16Ty(Ctx);

  // TODO: Currently we defaultly set Palette = 1, it may be assigned to
  // other value in the future.
  Value *PaletteOffset = ConstantInt::get(Type::getInt64Ty(Ctx), 0);
  Value *PaletteValue = ConstantInt::get(Type::getInt8Ty(Ctx), 1);
  Value *PalettePos = Builder.CreateGEP(I8Ty, I8Ptr, PaletteOffset);
  Builder.CreateStore(PaletteValue, PalettePos);

  for (int I = 0, E = Shapes.size() / 2; I < E; I++) {
    Value *RowOffset = ConstantInt::get(Type::getInt64Ty(Ctx), 48 + I);
    Value *ColOffset = ConstantInt::get(Type::getInt64Ty(Ctx), 16 + I * 2);
    const std::string ShapeName = "amx.tmm." + itostr(I);
    Value *RowPos = Builder.CreateGEP(I8Ty, I8Ptr, RowOffset,
                                      ShapeName + ".shape.row");
    Value *ColPos = Builder.CreateGEP(I8Ty, I8Ptr, ColOffset);
    ColPos = Builder.CreateBitCast(ColPos, PointerType::get(I16Ty, 0),
                                   ShapeName + ".shape.col");
    Value *Row = Shapes[I * 2];
    Value *Col = Shapes[I * 2 + 1];
    Row = Builder.CreateTrunc(Row, I8Ty);
    Builder.CreateStore(Row, RowPos);
    Builder.CreateStore(Col, ColPos);
  }
}

void X86PreAMXConfig::addTileConfig(Instruction *ModelStart,
                                    SmallVector<Value *, 8> &Shapes) {
  Module *M = F.getParent();
  IRBuilder<> Builder(ModelStart);
  const DataLayout &DL = M->getDataLayout();
  unsigned AddrSpace = DL.getAllocaAddrSpace();
  LLVMContext &Ctx = Builder.getContext();
  Type *V512Ty = VectorType::get(Builder.getInt32Ty(), 16, false);
  Align Alignment = DL.getPrefTypeAlign(Type::getInt32Ty(Ctx));

  AllocaInst *Addr =
      new AllocaInst(V512Ty, AddrSpace, "", &F.getEntryBlock().front());
  Addr->setAlignment(Alignment);
  Value *I8Ptr = Builder.CreateBitCast(Addr, Builder.getInt8PtrTy());

  Builder.CreateAlignedStore(Constant::getNullValue(V512Ty), Addr, Alignment);

  preWriteTileCfg(I8Ptr, Builder, Shapes);

  Builder.CreateIntrinsic(Intrinsic::x86_ldtilecfg_internal, std::nullopt,
                          {I8Ptr});
}

// Todo: We may need to handle "more than one store" case in the future.
bool X86PreAMXConfig::checkVolatileModel(SmallSet<Value *, 4> &Loads,
                                         IntrinsicInst *Store,
                                         IntrinsicInst *KeyAMX) {
  Value *ST = Store->getOperand(4);

  // Only has tileload and tilestore.
  if (!KeyAMX)
    return (Loads.size() == 1) && Loads.contains(ST);

  // All Loads should be operands of KeyAMX.
  // All tile operands of KeyAMX should come from Loads.
  for (Value *Op : KeyAMX->operands()) {
    if (Op->getType()->isX86_AMXTy())
      if (!Loads.erase(Op))
        return false;
  }

  // The def of KeyAMX should be stored into mem.
  // Todo: is it key amx can be no def?
  return Loads.empty() && (ST == cast<Value>(KeyAMX));
}

bool X86PreAMXConfig::getKeyAMXShapes(IntrinsicInst *KeyAMX,
                                      SmallVector<Value *, 8> &Shapes) {
  for (unsigned I = 0; I < KeyAMX->getNumOperands(); I++) {
    Value *Op = KeyAMX->getOperand(I);
    if (!Op->getType()->isX86_AMXTy())
      continue;
    IntrinsicInst *TileDef = dyn_cast<IntrinsicInst>(Op);
    assert((TileDef && isTileLoad(TileDef)) &&
           "All KeyAMX's tile definiation should comes from TileLoad!");
    Shapes.push_back(TileDef->getOperand(0));
    Shapes.push_back(TileDef->getOperand(1));
  }
  if (!isTileStore(KeyAMX)) {
    Shapes.push_back(KeyAMX->getOperand(0));
    Shapes.push_back(KeyAMX->getOperand(1));
  }
  return Shapes.size() != 0;
}

// Collect the shapes and skip the area of current key amx intrinsic.
//
// For example:
// ...
// --------------------------------------------------------------------------
// %t1 = call x86_amx @llvm.x86.tileloadd64.internal(m, k, ...)  record (m,k)
// %t2 = call x86_amx @llvm.x86.tileloadd64.internal(k, n, ...)  record (m,k)
// %t3 = call x86_amx @llvm.x86.tileloadd64.internal(m, n, ...)  record (m,k)
// %td = call x86_amx @llvm.x86.tdpbssd.internal(...t1, t2, t3)
// call void @llvm.x86.tilestored64.internal(m, n,... td) <--PosEnd record (m,k)
// --------------------------------------------------------------------------
BasicBlock::iterator
X86PreAMXConfig::getShapesAndConfigPosEnd(BasicBlock::iterator Iter,
                                          SmallVector<Value *, 8> &Shapes) {
  IntrinsicInst *KeyAMX = nullptr;
  BasicBlock *BB = Iter->getParent();
  BasicBlock::iterator PosEnd = BB->end();
  SmallSet<Value *, 4> Loads;

  // See TileStore as "Config Position End" and check volatile model.
  for (auto I = Iter, E = BB->end(); I != E; ++I) {
    assert(!brokenVolatile(&*I) && "Not reach tile store!");
    IntrinsicInst *II = dyn_cast<IntrinsicInst>(&*I);
    if (!II || !isAMXIntrinsic(II))
      continue;

    if (isTileLoad(II)) {
      Loads.insert(II);
    } else if (isTileStore(II)) {
      if (!checkVolatileModel(Loads, II, KeyAMX))
        report_fatal_error("Not Volatile AMX Model!");
      PosEnd = I;
      break;
    } else {
      assert(!KeyAMX && "Too many key amx intrinsic!");
      KeyAMX = II;
    }
  }
  assert(PosEnd != BB->end() && "Not find TileStore!");

  // See KeyAMX as TileStore if only TileLoad and TileStore.
  if (!KeyAMX)
    KeyAMX = dyn_cast<IntrinsicInst>(&*PosEnd);

  // Get Shapes in order.
  assert(Shapes.empty() && "Shapes should be clean.");
  getKeyAMXShapes(KeyAMX, Shapes);

  return PosEnd;
}

// Record a key amx area's shapes with its position.
// Use the first tileload as its position.
// For example:
// ...
// --------------------------------------------------------------------------
// %t1 = call x86_amx @llvm.x86.tileloadd64.internal(m, k, ...)   <--  pos
// %t2 = call x86_amx @llvm.x86.tileloadd64.internal(k, n, ...)        /
// %t3 = call x86_amx @llvm.x86.tileloadd64.internal(m, n, ...)     shapes:
// %td = call x86_amx @llvm.x86.tdpbssd.internal(...t1, t2, t3)    (m,k)(k,n)
// call void @llvm.x86.tilestored64.internal(m, n,... td)          (m,n)(m,n)
// --------------------------------------------------------------------------
bool X86PreAMXConfig::findConfigShapes(PosAndShapesMap &PosAndShapes) {
  bool Find = false;
  for (BasicBlock &BB : F) {
    for (BasicBlock::iterator I = BB.begin(), E = BB.end(); I != E; ++I) {
      IntrinsicInst *II = dyn_cast<IntrinsicInst>(&*I);
      if (!II)
        continue;
      if (!isAMXIntrinsic(II))
        continue;
      assert(onlyTileDef(II) && "Not volatile model for AMX at O0!");

      I = getShapesAndConfigPosEnd(I, PosAndShapes[&*I]);
      Find = true;
    }
  }
  return Find;
}

// Insert ldtilecfg and preconfig the shapes for each area of key AMX intrinsic.
// e.g. (key amx = tdpbssd)
// --------------------------------------------------------------------------
// %cfgmem = alloca <16 x i32>, align 4                        * allocate mem
// store <16 x i32> zeroinitializer, <16 x i32>* %cfgmem       * zero init
// ...
// ... pre-config shape of %t1                                 *
// store volatile i8 %m, i8* %amx.tmm.0.shape.row, align 1     *
// store volatile i16 %k, i16* %amx.tmm.0.shape.col, align 2   * pre-config
// ...                                                         *
// ... pre-config shape of %t2                                 *
// store volatile i8 %k, i8* %amx.tmm.1.shape.row, align 1     * shapes
// store volatile i16 %n, i16* %amx.tmm.1.shape.col, align 2   *
// ...                                                         *
// ... pre-config shape of %t3                                 * of
// store volatile i8 %m, i8* %amx.tmm.2.shape.row, align 1     *
// store volatile i16 %n, i16* %amx.tmm.2.shape.col, align 2   *
// ...                                                         * tiles
// ... pre-config shape of %td                                 *
// store volatile i8 %m, i8* %amx.tmm.3.shape.row, align 1     *
// store volatile i16 %n, i16* %amx.tmm.3.shape.col, align 2   *
//
// call void @llvm.x86.ldtilecfg(i8* %cfgmem)                  * pre-config
// --------------------------------------------------------------------------
// %t1 = call x86_amx @llvm.x86.tileloadd64.internal(m, k, ...)          key
// %t2 = call x86_amx @llvm.x86.tileloadd64.internal(k, n, ...)
// %t3 = call x86_amx @llvm.x86.tileloadd64.internal(m, n, ...)          amx
// %td = tail call x86_amx @llvm.x86.tdpbssd.internal(m, n, k, t1, t2, t3)
// call void @llvm.x86.tilestored64.internal(... td)                     area
// --------------------------------------------------------------------------
bool X86PreAMXConfig::preTileConfig() {
  PosAndShapesMap PosAndShapes;
  bool NeedCfg = findConfigShapes(PosAndShapes);
  if (!NeedCfg)
    return false;
  for (auto &IPAndShapes : PosAndShapes)
    addTileConfig(IPAndShapes.first, IPAndShapes.second);

  return true;
}
} // anonymous namespace

namespace {

class X86PreAMXConfigPass : public FunctionPass {
public:
  static char ID;

  X86PreAMXConfigPass() : FunctionPass(ID) {
    initializeX86PreAMXConfigPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    TargetMachine *TM = &getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
    bool C = false;

    // Prepare for fast register allocation at O0.
    if (TM->getOptLevel() == CodeGenOptLevel::None) {

      // We pre-config each key AMX intrinsic at O0.
      // In theory, one tile config can cover several AMX intrinsics, but
      // it is very diffcult to classify the tile shapes at O0. So here we
      // let thing be easy, pre-config every key AMX intrinsic.
      X86PreAMXConfig PCFG(F);
      C = PCFG.preTileConfig();
    }

    return C;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<TargetPassConfig>();
  }
};

} // anonymous namespace

static const char PassName[] = "Pre AMX Tile Config";
char X86PreAMXConfigPass::ID = 0;
INITIALIZE_PASS_BEGIN(X86PreAMXConfigPass, DEBUG_TYPE, PassName, false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(X86PreAMXConfigPass, DEBUG_TYPE, PassName, false, false)

FunctionPass *llvm::createX86PreAMXConfigPass() {
  return new X86PreAMXConfigPass();
}
