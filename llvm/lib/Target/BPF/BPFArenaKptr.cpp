//===-- BPFArenaKptr.cpp - Locals that hold pointers of the kernel --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Locals go to the stack in arena (BPFArenaStack.cpp). This pass runs in front
// of it and marks the locals that have to stay in the stack of the program.
//
// What gets to helpers and kfuncs stays: they take memory that the kernel knows
// about. See BPFKernelMem.
//
// What gets a pointer of the kernel stays. A pointer that the kernel gave to
// the program (context, task_struct, a map value, the stack of the program) is
// a pointer for the verifier while it is in a register or in the stack of
// the program. Arena is not tracked: the pointer would come back as a number.
// The verifier does not let a program that has a stack in arena store it there.
//
// - The pass finds the values that may be pointers of the kernel: the context,
//   what is loaded from it or from an object of the kernel, what helpers and
//   kfuncs return unless its type says arena, the parameters of a callback,
//   the addresses of maps, of variables of the kernel, of global variables that
//   are not in arena and of the locals that stay.
// - It follows them through arguments, results, indirect calls and copies, also
//   when a callee stores them through a pointer, when a function returns them
//   in a pair, and when the copy of a struct moves them as an integer. What is
//   loaded from a local that stays is one of them.
// - A local that gets one of them stays, whatever else the program does with
//   it. It is where it is without the pass.
// - A global variable that gets one does not move to arena (BPFArenaData.cpp)
//   if the program only compares what it loads from there, keeps it as a number
//   or passes it to a helper or a kfunc. If it accesses memory through it
//   the variable moves: outside of arena and of the stack the pointer comes
//   back as a number too, and nothing tells. In arena the verifier can reject
//   the store.
// - Loads that may yield pointers to the stack are marked. BPFArenaCast follows
//   pointers, but not through memory.
//
// If the pass misses a local the program does not load: the verifier rejects
// the store.
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "BPFArena.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "bpf-arena-kptr"

STATISTIC(NumKept, "Number of locals kept in the stack of the program");

/// An offset that is not a constant.
static constexpr int64_t Unknown = INT64_MIN;

using Offsets = SmallSetVector<int64_t, 4>;

/// In a cycle of calls that add a constant to the address the offsets grow for
/// ever. An indirect call makes such a cycle easily: it may call every function
/// of its type.
static bool addOffset(Offsets &S, int64_t Off) {
  if (S.count(Unknown))
    return false;
  return S.insert(S.size() < 32 ? Off : Unknown);
}

namespace {

class ArenaKptr {
  using ArenaPred = function_ref<bool(const GlobalVariable &)>;

  Module &M;
  const DataLayout &DL;
  /// Is that global variable in arena, or does it move there?
  ArenaPred InArena;
  /// Values that may be pointers of the kernel.
  DenseSet<const Value *> K;
  /// Allocas, arguments and global variables that get one of them.
  DenseSet<const Value *> Holders;
  /// Allocas and arguments -> offsets at which such a pointer is stored.
  DenseMap<const Value *, Offsets> Slots;
  /// The part of Slots of an argument that the function itself stores, not
  /// one of its callers.
  DenseMap<const Value *, Offsets> Written;
  /// Functions that may return such a pointer.
  DenseSet<const Function *> RetK;
  /// The functions that an indirect call of that type may call.
  DenseMap<const FunctionType *, SmallVector<Function *, 4>> Taken;
  /// Track pointers to the given locals instead of pointers of the kernel.
  bool StackOnly = false;

  std::pair<const Value *, int64_t> slot(const Value *Ptr) const;
  bool isK(const Value *V) const;
  bool mayLoadK(const Value *Ptr, Type *Ty) const;
  bool store(const Value *Ptr, int64_t Delta = 0);
  SmallVector<Function *, 4> callees(const CallBase &CB) const;
  bool visit(Instruction &I);
  void propagate();
  void seed(SmallVectorImpl<AllocaInst *> &Stay);
  /// What was looked at: as the address of a global variable, and as a value
  /// that was loaded from there.
  struct Visited {
    SmallPtrSet<const Value *, 16> Addrs, Values;
  };
  bool isAccessed(const Value *V, Visited &Seen) const;
  bool isRead(const Value *Ptr, Visited &Seen) const;

public:
  ArenaKptr(Module &M, ArenaPred InArena);
  bool run();
  void globals(SmallPtrSetImpl<const GlobalVariable *> &GVs);
};

} // namespace

static bool isObject(const Value *V) { return isa<AllocaInst, Argument>(V); }

/// The context.
static bool isCtx(const Value *V) {
  auto *A = dyn_cast<Argument>(V);

  return A && !A->getArgNo() && A->getParent()->hasSection();
}

/// A pointer, or an integer that becomes one: (void *)(long)skb->data_end.
static bool usedAsPointer(const Value *V) {
  if (V->getType()->isPointerTy())
    return true;
  return any_of(V->users(), [](const User *U) {
    return isa<IntToPtrInst>(U) ||
           (isa<ZExtInst, SExtInst, FreezeInst>(U) && usedAsPointer(U));
  });
}

ArenaKptr::ArenaKptr(Module &M, ArenaPred InArena)
    : M(M), DL(M.getDataLayout()), InArena(InArena) {
  for (Function &F : M)
    if (!F.isDeclaration() && F.hasAddressTaken())
      Taken[F.getFunctionType()].push_back(&F);
}

/// The functions of the module that \p CB may call.
SmallVector<Function *, 4> ArenaKptr::callees(const CallBase &CB) const {
  SmallVector<Function *, 4> Fns;

  if (CB.isInlineAsm())
    return Fns;
  if (Function *Callee = CB.getCalledFunction())
    Fns.push_back(Callee);
  else
    Fns = Taken.lookup(CB.getFunctionType());
  erase_if(Fns, [&](Function *F) {
    return F->isDeclaration() || F->isVarArg() ||
           F->arg_size() != CB.arg_size();
  });
  return Fns;
}

std::pair<const Value *, int64_t> ArenaKptr::slot(const Value *Ptr) const {
  int64_t Off = 0;
  const Value *Base = GetPointerBaseWithConstantOffset(Ptr, Off, DL);

  if (isObject(Base) || isa<GlobalVariable>(Base))
    return {Base, Off};
  return {getUnderlyingObject(Ptr), Unknown};
}

/// A load of \p Ty from \p Ptr can read one of K: it is not from another field
/// of the variable.
bool ArenaKptr::mayLoadK(const Value *Ptr, Type *Ty) const {
  auto [Obj, Off] = slot(Ptr);
  auto It = Slots.find(Obj);
  int64_t Size = DL.getTypeStoreSize(Ty).getFixedValue();

  if (It == Slots.end() || Off == Unknown || It->second.count(Unknown))
    return true;
  return any_of(It->second, [&, Off = Off](int64_t O) {
    return O < Off + Size && Off < O + 8;
  });
}

bool ArenaKptr::isK(const Value *V) const {
  if (K.count(V))
    return true;
  if (StackOnly || !isa<Constant>(V))
    return false;
  if (auto *CE = dyn_cast<ConstantExpr>(V);
      CE && CE->getOpcode() == Instruction::PtrToInt)
    V = CE->getOperand(0);
  // A map, a variable of the kernel, data that the kernel knows as a map.
  auto *GV = dyn_cast<GlobalVariable>(getUnderlyingObject(V));
  return GV && !InArena(*GV);
}

/// One of K is stored at \p Ptr + \p Delta.
bool ArenaKptr::store(const Value *Ptr, int64_t Delta) {
  SmallVector<const Value *, 4> Objs;
  bool Changed = false;
  auto [Obj, Off] = slot(Ptr);

  if (isObject(Obj) || isa<GlobalVariable>(Obj)) {
    Off = Off == Unknown || Delta == Unknown ? Unknown : Off + Delta;
    if (isa<Argument>(Obj))
      Changed |= addOffset(Written[Obj], Off);
    Changed |= addOffset(Slots[Obj], Off);
  }
  if (StackOnly)
    return Changed;
  getUnderlyingObjects(Ptr, Objs);
  for (const Value *O : Objs) {
    if (!isObject(O) && !isa<GlobalVariable>(O))
      continue;
    Changed |= Holders.insert(O).second;
    // It stays, so its address is a pointer to the stack of the program.
    if (isa<AllocaInst>(O))
      Changed |= K.insert(O).second;
  }
  return Changed;
}

bool ArenaKptr::visit(Instruction &I) {
  bool Changed = false;

  if (auto *LI = dyn_cast<LoadInst>(&I)) {
    bool Ptr = usedAsPointer(LI);

    // The copy of a struct of 8 bytes loads and stores an integer.
    if (!Ptr && !LI->getType()->isIntegerTy(64))
      return false;
    auto [Obj, Off] = slot(LI->getPointerOperand());
    auto It = Slots.find(Obj);
    if (It != Slots.end() &&
        (Off == Unknown || It->second.count(Off) || It->second.count(Unknown)))
      Changed |= K.insert(LI).second;
    // A field of the context or of an object of the kernel.
    if (Ptr && !StackOnly && isK(getUnderlyingObject(LI->getPointerOperand())))
      Changed |= K.insert(LI).second;
  } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
    if (isK(SI->getValueOperand()))
      Changed |= store(SI->getPointerOperand());
  } else if (auto *MT = dyn_cast<MemTransferInst>(&I)) {
    auto [Src, SrcOff] = slot(MT->getSource());
    auto It = Slots.find(Src);
    if (It == Slots.end())
      return false;
    auto *Len = dyn_cast<ConstantInt>(MT->getLength());
    SmallVector<int64_t, 4> Offs(It->second.begin(), It->second.end());
    for (int64_t O : Offs) {
      if (O == Unknown || SrcOff == Unknown || !Len)
        Changed |= store(MT->getDest(), Unknown);
      else if (O >= SrcOff && O < SrcOff + (int64_t)Len->getZExtValue())
        Changed |= store(MT->getDest(), O - SrcOff);
    }
  } else if (auto *CB = dyn_cast<CallBase>(&I)) {
    if (isBPFKernelFunc(*CB)) {
      if (StackOnly)
        return false;
      if (CB->getType()->isPointerTy() &&
          CB->getType()->getPointerAddressSpace() != BPFArenaAS)
        Changed |= K.insert(CB).second;
      // The kernel calls a callback with pointers of its own.
      for (Value *Op : CB->args())
        if (auto *F = dyn_cast<Function>(Op->stripPointerCasts()))
          for (Argument &A : F->args())
            if (A.getType()->isPointerTy())
              Changed |= K.insert(&A).second;
      return Changed;
    }
    for (Function *Callee : callees(*CB)) {
      for (unsigned N = 0; N < CB->arg_size(); N++) {
        Value *Op = CB->getArgOperand(N);
        Argument *A = Callee->getArg(N);

        if (isK(Op))
          Changed |= K.insert(A).second;
        if (!Op->getType()->isPointerTy())
          continue;
        auto [Obj, Off] = slot(Op);
        // What the caller stored is seen by the callee and the other way round.
        if (auto It = Slots.find(Obj); It != Slots.end()) {
          SmallVector<int64_t, 4> Offs(It->second.begin(), It->second.end());
          for (int64_t O : Offs)
            Changed |= addOffset(
                Slots[A], O == Unknown || Off == Unknown ? Unknown : O - Off);
        }
        if (auto It = Written.find(A); It != Written.end()) {
          SmallVector<int64_t, 4> Offs(It->second.begin(), It->second.end());
          for (int64_t O : Offs)
            Changed |= store(Op, O);
        }
      }
      if (RetK.count(Callee))
        Changed |= K.insert(CB).second;
    }
  } else if (auto *RI = dyn_cast<ReturnInst>(&I)) {
    if (RI->getReturnValue() && isK(RI->getReturnValue()))
      Changed |= RetK.insert(I.getFunction()).second;
  } else if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
    if (isK(GEP->getPointerOperand()))
      Changed |= K.insert(&I).second;
  } else if (isa<SelectInst, PHINode, CastInst, FreezeInst, InsertValueInst>(
                 I)) {
    if (any_of(drop_begin(I.operands(), isa<SelectInst>(I) ? 1 : 0),
               [&](const Use &U) { return isK(U.get()); }))
      Changed |= K.insert(&I).second;
  } else if (auto *EV = dyn_cast<ExtractValueInst>(&I)) {
    // A function returns a pair. Which of the two it is, is not tracked.
    if (usedAsPointer(EV) && isK(EV->getAggregateOperand()))
      Changed |= K.insert(&I).second;
  }
  return Changed;
}

void ArenaKptr::propagate() {
  for (bool Changed = true; Changed;) {
    Changed = false;
    for (Function &F : M)
      for (Instruction &I : instructions(F))
        Changed |= visit(I);
  }
}

void ArenaKptr::seed(SmallVectorImpl<AllocaInst *> &Stay) {
  BPFKernelMem KM(M);

  for (Function &F : M) {
    if (!F.isDeclaration() && F.arg_size() && isCtx(F.getArg(0)))
      K.insert(F.getArg(0));
    for (Instruction &I : instructions(F))
      if (auto *AI = dyn_cast<AllocaInst>(&I); AI && KM.isPassed(AI))
        Stay.push_back(AI);
  }
  K.insert(Stay.begin(), Stay.end());
  propagate();
}

/// The program accesses memory through \p V, or \p V gets out of sight as
/// a pointer.
bool ArenaKptr::isAccessed(const Value *V, Visited &Seen) const {
  if (!Seen.Values.insert(V).second)
    return false;
  return any_of(V->uses(), [&](const Use &U) {
    const auto *I = cast<Instruction>(U.getUser());

    // An index is a number.
    if (const auto *GEP = dyn_cast<GetElementPtrInst>(I))
      return &U == &GEP->getOperandUse(0) && isAccessed(I, Seen);
    if (isa<PHINode, SelectInst, CastInst, FreezeInst, BinaryOperator>(I))
      return isAccessed(I, Seen);
    // A copy of a number: a key, a field of an event.
    if (isa<ReturnInst>(I) ||
        (isa<StoreInst>(I) &&
         U.getOperandNo() != StoreInst::getPointerOperandIndex()))
      return V->getType()->isPointerTy();
    const auto *CB = dyn_cast<CallBase>(I);
    // Comparisons are fine.
    if (!CB)
      return !isa<CmpInst>(I);
    // It is a number for the kernel too.
    if (isBPFKernelFunc(*CB) || isa<DbgInfoIntrinsic>(CB))
      return false;
    SmallVector<Function *, 4> Fns = callees(*CB);
    return Fns.empty() || !CB->isArgOperand(&U) ||
           any_of(Fns, [&](Function *F) {
             return isAccessed(F->getArg(CB->getArgOperandNo(&U)), Seen);
           });
  });
}

/// The program accesses memory through what it loads from \p Ptr, or \p Ptr
/// gets out of sight.
bool ArenaKptr::isRead(const Value *Ptr, Visited &Seen) const {
  if (!Seen.Addrs.insert(Ptr).second)
    return false;
  return any_of(Ptr->uses(), [&](const Use &U) {
    const User *Usr = U.getUser();

    if (isa<GEPOperator, BitCastOperator, AddrSpaceCastOperator>(Usr) ||
        isa<PHINode, SelectInst>(Usr))
      return isRead(Usr, Seen);
    if (isa<LoadInst, AtomicRMWInst>(Usr))
      return mayLoadK(U.get(), Usr->getType()) && isAccessed(Usr, Seen);
    if (isa<StoreInst>(Usr))
      return U.getOperandNo() != StoreInst::getPointerOperandIndex();
    if (const auto *MT = dyn_cast<MemTransferInst>(Usr))
      return &U == &MT->getRawSourceUse();
    const auto *CB = dyn_cast<CallBase>(Usr);
    if (!CB)
      return !isa<CmpInst>(Usr);
    if (isBPFKernelFunc(*CB) || isa<IntrinsicInst>(CB))
      return false;
    SmallVector<Function *, 4> Fns = callees(*CB);
    return Fns.empty() || !CB->isArgOperand(&U) ||
           any_of(Fns, [&](Function *F) {
             return isRead(F->getArg(CB->getArgOperandNo(&U)), Seen);
           });
  });
}

/// The global variables that get a pointer of the kernel, unless the program
/// accesses memory through what it loads from there.
void ArenaKptr::globals(SmallPtrSetImpl<const GlobalVariable *> &GVs) {
  SmallVector<AllocaInst *, 16> Stay;

  seed(Stay);
  for (const Value *V : Holders) {
    Visited Seen;
    auto *GV = dyn_cast<GlobalVariable>(V);

    if (GV && !isRead(GV, Seen))
      GVs.insert(GV);
  }
}

bool ArenaKptr::run() {
  MDNode *MD = MDNode::get(M.getContext(), {});
  /// The locals that stay.
  SmallVector<AllocaInst *, 16> Stay;

  seed(Stay);
  for (Function &F : M) {
    for (Instruction &I : instructions(F)) {
      auto *AI = dyn_cast<AllocaInst>(&I);

      if (AI && Holders.count(AI) && !is_contained(Stay, AI)) {
        Stay.push_back(AI);
        NumKept++;
      }
    }
  }
  if (Stay.empty())
    return false;
  for (AllocaInst *AI : Stay) {
    AI->setMetadata(BPFStackKeepMD, MD);
    AI->setMetadata(BPFArenaStackMD, nullptr);
  }

  // Where do pointers to these locals get to through memory?
  ArenaKptr S(M, InArena);

  S.StackOnly = true;
  S.K.insert(Stay.begin(), Stay.end());
  S.propagate();
  for (const Value *V : S.K)
    if (auto *LI = dyn_cast<LoadInst>(V))
      const_cast<LoadInst *>(LI)->setMetadata(BPFStackPtrMD, MD);
  return true;
}

namespace {

struct BPFArenaKptrLegacy : public ModulePass {
  static char ID;
  BPFArenaKptrLegacy() : ModulePass(ID) {}
  StringRef getPassName() const override { return "BPF Arena Kptr"; }

  bool runOnModule(Module &M) override {
    return ArenaKptr(M, isBPFArenaGlobal).run();
  }
};

} // namespace

char BPFArenaKptrLegacy::ID = 0;

INITIALIZE_PASS(BPFArenaKptrLegacy, DEBUG_TYPE, "BPF Arena Kptr", false, false)

ModulePass *llvm::createBPFArenaKptrPass() { return new BPFArenaKptrLegacy(); }

void llvm::findBPFKernelPtrGlobals(
    Module &M, function_ref<bool(const GlobalVariable &)> Moves,
    SmallPtrSetImpl<const GlobalVariable *> &GVs) {
  ArenaKptr(M, Moves).globals(GVs);
}

PreservedAnalyses BPFArenaKptrPass::run(Module &M, ModuleAnalysisManager &AM) {
  if (!ArenaKptr(M, isBPFArenaGlobal).run())
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
