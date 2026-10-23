//===-- AGXLowerRecursion.cpp - Recursion with an explicit stack ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The backend has no calling convention: every function is inlined, which
// recursive functions cannot be. Each strongly connected component of the
// call graph with a cycle becomes, for each of its members called from
// outside it, one function that runs the whole component as a state machine
// over an explicit per-thread stack of frames in private memory:
//
//  * The blocks of each member are split after its calls into the component.
//    Its arguments, its allocas and its values used outside their block
//    become fields of its frame, after a header of two words: the number of
//    the call site to return to, and the stack pointer of the caller.
//  * A call stores the arguments in the callee's frame, which follows the
//    caller's, and branches to the callee's entry.
//  * A return stores the return value in a slot of the function, pops the
//    frame and switches on the call site number to the code after the call,
//    or out of the function for 0.
//
// The result is not recursive and is inlined like any other function. The
// control flow passes (FixIrreducible, StructurizeCFG) make the state
// machine structured, divergent call depths included. A thread whose call
// would pass the end of the stack sets the global __agx_stack_overflow (a
// 32-bit word of device memory, 0 at load) to 1 and ends; the runtime
// reports the launch as failed.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "Utils/AGXBaseInfo.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

#define DEBUG_TYPE "agx-lower-recursion"

static cl::opt<unsigned> RecursionStackSizeOpt(
    "agx-recursion-stack-size", cl::init(4096), cl::Hidden,
    cl::desc("Bytes of private memory per thread for the frames of "
             "recursive functions (default: four times the module flag "
             "agx-stack-size, else 4096)"));

static unsigned RecursionStackSize = 4096;

namespace {

// The frame header: the call site to return to and the caller's stack pointer.
constexpr uint64_t ReturnSiteOffset = 0;
constexpr uint64_t SavedSPOffset = 4;
constexpr uint64_t HeaderSize = 8;
constexpr uint64_t FrameAlign = 16;

struct Member {
  Function *F = nullptr;
  BasicBlock *Entry = nullptr;
  SmallVector<AllocaInst *, 4> Args;
  SmallVector<uint64_t, 4> ArgOffsets;
  SmallVector<std::pair<AllocaInst *, uint64_t>, 16> Fields;
  uint64_t Size = HeaderSize;
  SmallVector<CallInst *, 4> Calls;
  SmallVector<ReturnInst *, 2> Returns;
  // The continuations of the calls to this member, by call site number.
  SmallVector<std::pair<unsigned, BasicBlock *>, 4> ReturnSites;
  BasicBlock *Dispatch = nullptr;
};

class Machine {
  Module &M;
  const DataLayout &DL;
  LLVMContext &C;
  ArrayRef<Function *> SCC;
  Function *Root;
  Function *G = nullptr;
  AllocaInst *Stack = nullptr;
  AllocaInst *SP = nullptr;
  GlobalVariable *OverflowFlag;
  MapVector<Type *, AllocaInst *> ReturnSlots;
  DenseMap<Function *, unsigned> Index;
  SmallVector<Member, 4> Members;

  IntegerType *i32() { return Type::getInt32Ty(C); }

  Value *field(IRBuilder<> &B, Value *Base, uint64_t Offset) {
    return B.CreateInBoundsGEP(B.getInt8Ty(), Stack,
                               B.CreateAdd(Base, B.getInt32(Offset)));
  }

  bool prepare(Member &Mb, Function *Original);
  void lowerCall(Member &Caller, CallInst *CI, unsigned Site);
  void lowerReturn(Member &Mb, ReturnInst *RI);

public:
  Machine(Module &M, ArrayRef<Function *> SCC, Function *Root,
          GlobalVariable *OverflowFlag)
      : M(M), DL(M.getDataLayout()), C(M.getContext()), SCC(SCC), Root(Root),
        OverflowFlag(OverflowFlag) {}
  Function *build();
};

} // namespace

/// Copies the member into G and turns its state into frame fields.
bool Machine::prepare(Member &Mb, Function *Original) {
  ValueToValueMapTy VMap;
  Function *Clone = CloneFunction(Original, VMap);
  // Frames are plain memory: no lifetimes, and no debug information (the
  // blocks of several functions share one function).
  SmallVector<Instruction *, 16> Erase;
  for (Instruction &I : instructions(Clone)) {
    I.setDebugLoc(DebugLoc());
    I.dropDbgRecords();
    if (isa<DbgInfoIntrinsic>(I) || I.isLifetimeStartOrEnd())
      Erase.push_back(&I);
  }
  for (Instruction *I : Erase)
    I->eraseFromParent();

  BasicBlock &Entry = Clone->getEntryBlock();
  IRBuilder<> B(Entry.getFirstInsertionPt());
  unsigned AS = DL.getAllocaAddrSpace();
  for (Argument &A : Clone->args()) {
    AllocaInst *Slot = B.CreateAlloca(A.getType(), AS, nullptr, A.getName());
    Mb.Args.push_back(Slot);
    A.replaceAllUsesWith(B.CreateLoad(A.getType(), Slot));
  }
  // Placeholders of the return slots of G, replaced once G exists.
  DenseMap<Type *, AllocaInst *> Results;
  auto Result = [&](Type *T) {
    AllocaInst *&Slot = Results[T];
    if (!Slot) {
      IRBuilder<> EB(Entry.begin());
      Slot = EB.CreateAlloca(T, AS, nullptr, "result");
    }
    return Slot;
  };

  SmallVector<Instruction *, 16> Sites;
  for (Instruction &I : instructions(Clone))
    if (isa<ReturnInst, CallBase>(I))
      Sites.push_back(&I);
  for (Instruction *Site : Sites) {
    Instruction &I = *Site;
    if (auto *RI = dyn_cast<ReturnInst>(&I)) {
      if (Value *V = RI->getReturnValue()) {
        new StoreInst(V, Result(V->getType()), RI->getIterator());
        RI->setOperand(0, PoisonValue::get(V->getType()));
      }
      Mb.Returns.push_back(RI);
      continue;
    }
    auto *CB = dyn_cast<CallBase>(&I);
    if (!CB || !Index.contains(CB->getCalledFunction()))
      continue;
    auto *CI = dyn_cast<CallInst>(CB);
    if (!CI || CI->isMustTailCall())
      return false;
    BasicBlock *After = SplitBlock(CI->getParent(), CI->getNextNode());
    if (!CI->getType()->isVoidTy()) {
      IRBuilder<> AB(After->getFirstInsertionPt());
      CI->replaceAllUsesWith(
          AB.CreateLoad(CI->getType(), Result(CI->getType())));
    }
    Mb.Calls.push_back(CI);
  }

  // Values that live across blocks go to memory.
  SmallVector<PHINode *, 8> Phis;
  for (BasicBlock &BB : *Clone)
    for (PHINode &P : BB.phis())
      Phis.push_back(&P);
  for (PHINode *P : Phis)
    DemotePHIToStack(P);
  SmallVector<Instruction *, 32> Live;
  for (Instruction &I : instructions(Clone))
    if (!isa<AllocaInst>(I) && I.isUsedOutsideOfBlock(I.getParent()))
      Live.push_back(&I);
  for (Instruction *I : Live) {
    if (I->getType()->isTokenTy())
      return false;
    DemoteRegToStack(*I);
  }

  // The frame layout.
  for (Instruction &I : instructions(Clone)) {
    auto *AI = dyn_cast<AllocaInst>(&I);
    if (!AI)
      continue;
    if (any_of(Results, [&](auto &R) { return R.second == AI; }))
      continue;
    if (!AI->isStaticAlloca() || AI->getAlign().value() > FrameAlign)
      return false;
    std::optional<TypeSize> Size = AI->getAllocationSize(DL);
    if (!Size || Size->isScalable())
      return false;
    uint64_t Offset = alignTo(Mb.Size, AI->getAlign());
    Mb.Fields.push_back({AI, Offset});
    Mb.Size = Offset + Size->getFixedValue();
  }
  Mb.Size = alignTo(Mb.Size, FrameAlign);
  for (AllocaInst *Slot : Mb.Args)
    Mb.ArgOffsets.push_back(
        find_if(Mb.Fields, [&](auto &F) { return F.first == Slot; })->second);
  Mb.Args.clear();

  Mb.Entry = &Entry;
  G->splice(G->end(), Clone);
  Clone->eraseFromParent();

  for (auto [T, Slot] : Results) {
    AllocaInst *&GSlot = ReturnSlots[T];
    if (!GSlot) {
      IRBuilder<> GB(G->getEntryBlock().begin());
      GSlot = GB.CreateAlloca(T, AS, nullptr, "result");
    }
    Slot->replaceAllUsesWith(GSlot);
    Slot->eraseFromParent();
  }
  // A field is at its offset in the frame of the stack pointer at its use.
  for (auto [AI, Offset] : Mb.Fields) {
    for (Use &U : make_early_inc_range(AI->uses())) {
      IRBuilder<> UB(cast<Instruction>(U.getUser()));
      U.set(field(UB, UB.CreateLoad(i32(), SP), Offset));
    }
  }
  for (auto [AI, Offset] : Mb.Fields)
    AI->eraseFromParent();
  Mb.Fields.clear();
  return true;
}

void Machine::lowerCall(Member &Caller, CallInst *CI, unsigned Site) {
  Member &Callee = Members[Index.lookup(CI->getCalledFunction())];
  BasicBlock *After = CI->getParent()->getSingleSuccessor();
  IRBuilder<> B(CI);
  Value *Old = B.CreateLoad(i32(), SP);
  Value *New = B.CreateAdd(Old, B.getInt32(Caller.Size));
  Value *Overflow = B.CreateICmpUGT(B.CreateAdd(New, B.getInt32(Callee.Size)),
                                    B.getInt32(RecursionStackSize));
  Instruction *Trap = SplitBlockAndInsertIfThen(Overflow, CI, true);
  IRBuilder<> TB(Trap);
  TB.CreateStore(TB.getInt32(1), OverflowFlag, /*isVolatile=*/true);
  TB.CreateIntrinsic(Intrinsic::trap, {});
  B.SetInsertPoint(CI);
  B.CreateStore(B.getInt32(Site), field(B, New, ReturnSiteOffset));
  B.CreateStore(Old, field(B, New, SavedSPOffset));
  for (auto [Arg, Offset] : zip(CI->args(), Callee.ArgOffsets))
    B.CreateStore(Arg, field(B, New, Offset));
  B.CreateStore(New, SP);
  Instruction *Br = CI->getParent()->getTerminator();
  B.SetInsertPoint(Br);
  B.CreateBr(Callee.Entry);
  Br->eraseFromParent();
  CI->eraseFromParent();
  Callee.ReturnSites.push_back({Site, After});
}

void Machine::lowerReturn(Member &Mb, ReturnInst *RI) {
  IRBuilder<> B(RI);
  Value *Frame = B.CreateLoad(i32(), SP);
  Value *Site = B.CreateLoad(i32(), field(B, Frame, ReturnSiteOffset));
  B.CreateStore(B.CreateLoad(i32(), field(B, Frame, SavedSPOffset)), SP);
  B.CreateBr(Mb.Dispatch);
  cast<PHINode>(&Mb.Dispatch->front())->addIncoming(Site, RI->getParent());
  RI->eraseFromParent();
}

Function *Machine::build() {
  G = Function::Create(Root->getFunctionType(), GlobalValue::InternalLinkage,
                       Root->getName() + ".stack", M);
  G->setAttributes(Root->getAttributes());
  G->setCallingConv(Root->getCallingConv());
  G->removeFnAttr(Attribute::Memory);
  G->removeFnAttr(Attribute::NoRecurse);
  unsigned AS = DL.getAllocaAddrSpace();
  BasicBlock *Entry = BasicBlock::Create(C, "entry", G);
  IRBuilder<> B(Entry);
  Stack = B.CreateAlloca(ArrayType::get(B.getInt8Ty(), RecursionStackSize), AS,
                         nullptr, "stack");
  Stack->setAlignment(Align(FrameAlign));
  SP = B.CreateAlloca(i32(), AS, nullptr, "sp");

  for (auto [N, F] : enumerate(SCC))
    Index[F] = N;
  Members.resize(SCC.size());
  for (auto [N, F] : enumerate(SCC)) {
    Members[N].F = F;
    if (!prepare(Members[N], F)) {
      G->eraseFromParent();
      return nullptr;
    }
  }
  Member &RootMb = Members[Index.lookup(Root)];
  if (RootMb.Size > RecursionStackSize) {
    G->eraseFromParent();
    return nullptr;
  }

  // The first frame, for the call of G.
  B.SetInsertPoint(Entry);
  B.CreateStore(B.getInt32(0), SP);
  B.CreateStore(B.getInt32(0), field(B, B.getInt32(0), ReturnSiteOffset));
  B.CreateStore(B.getInt32(0), field(B, B.getInt32(0), SavedSPOffset));
  for (auto [Arg, Offset] : zip(G->args(), RootMb.ArgOffsets))
    B.CreateStore(&Arg, field(B, B.getInt32(0), Offset));
  B.CreateBr(RootMb.Entry);

  BasicBlock *Exit = BasicBlock::Create(C, "exit", G);
  B.SetInsertPoint(Exit);
  Type *RetTy = G->getReturnType();
  if (RetTy->isVoidTy())
    B.CreateRetVoid();
  else
    B.CreateRet(B.CreateLoad(RetTy, ReturnSlots.lookup(RetTy)));

  unsigned Site = 0;
  for (Member &Mb : Members)
    for (CallInst *CI : Mb.Calls)
      lowerCall(Mb, CI, ++Site);
  for (Member &Mb : Members) {
    Mb.Dispatch = BasicBlock::Create(C, Mb.F->getName() + ".return", G);
    IRBuilder<> DB(Mb.Dispatch);
    PHINode *Id = DB.CreatePHI(i32(), Mb.Returns.size(), "site");
    BasicBlock *Default = &Mb == &RootMb ? Exit : Mb.ReturnSites.back().second;
    SwitchInst *SI = DB.CreateSwitch(Id, Default);
    for (auto [S, To] : Mb.ReturnSites)
      if (To != Default)
        SI->addCase(DB.getInt32(S), To);
    for (ReturnInst *RI : Mb.Returns)
      lowerReturn(Mb, RI);
  }
  return G;
}

/// Replaces the recursive functions of M by state machines. Called by
/// AGXAlwaysInline after indirect calls became direct ones.
bool llvm::lowerAGXRecursion(Module &M) {
  RecursionStackSize = RecursionStackSizeOpt;
  if (auto *Flag = mdconst::extract_or_null<ConstantInt>(
          M.getModuleFlag("agx-stack-size"));
      Flag && !RecursionStackSizeOpt.getNumOccurrences())
    RecursionStackSize = 4 * Flag->getZExtValue();
  CallGraph CG(M);
  SmallVector<SmallVector<Function *, 4>, 2> Components;
  for (scc_iterator<CallGraph *> I = scc_begin(&CG); !I.isAtEnd(); ++I) {
    if (!I.hasCycle())
      continue;
    SmallVector<Function *, 4> SCC;
    for (CallGraphNode *N : *I)
      if (Function *F = N->getFunction())
        SCC.push_back(F);
    Components.push_back(SCC);
  }
  bool Changed = false;
  GlobalVariable *OverflowFlag = nullptr;
  auto Flag = [&] {
    if (!OverflowFlag)
      OverflowFlag = M.getGlobalVariable("__agx_stack_overflow",
                                         /*AllowInternal=*/true);
    if (!OverflowFlag) {
      Type *I32 = Type::getInt32Ty(M.getContext());
      OverflowFlag = new GlobalVariable(
          M, I32, false, GlobalValue::ExternalLinkage, ConstantInt::get(I32, 0),
          "__agx_stack_overflow", nullptr, GlobalValue::NotThreadLocal,
          AGX::DEVICE_ADDRESS);
      OverflowFlag->setAlignment(Align(4));
    }
    return OverflowFlag;
  };
  for (ArrayRef<Function *> SCC : Components) {
    SmallPtrSet<Function *, 4> In(SCC.begin(), SCC.end());
    // Only direct calls of defined, fixed-argument functions.
    bool Legal = all_of(SCC, [&](Function *F) {
      if (F->isDeclaration() || F->isVarArg() ||
          F->getCallingConv() == CallingConv::AGX_KERNEL)
        return false;
      return all_of(F->uses(), [&](Use &U) {
        auto *CI = dyn_cast<CallInst>(U.getUser());
        return CI && CI->isCallee(&U);
      });
    });
    if (!Legal)
      continue;
    SetVector<Function *> Roots;
    for (Function *F : SCC)
      for (User *U : F->users())
        if (!In.contains(cast<CallInst>(U)->getFunction()))
          Roots.insert(F);
    SmallVector<std::pair<Function *, Function *>, 2> Built;
    for (Function *Root : Roots) {
      Function *G = Machine(M, SCC, Root, Flag()).build();
      if (!G)
        break;
      Built.push_back({Root, G});
    }
    if (Built.size() != Roots.size()) {
      for (auto [Root, G] : Built)
        G->eraseFromParent();
      continue;
    }
    for (auto [Root, G] : Built)
      for (User *U : make_early_inc_range(Root->users()))
        if (auto *CI = cast<CallInst>(U); !In.contains(CI->getFunction()))
          CI->setCalledFunction(G);
    for (Function *F : SCC)
      F->dropAllReferences();
    for (Function *F : SCC)
      F->eraseFromParent();
    Changed = true;
  }
  return Changed;
}
