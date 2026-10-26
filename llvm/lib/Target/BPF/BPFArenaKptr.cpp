//===-- BPFArenaKptr.cpp - Locals that hold pointers of the kernel --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A pointer that the kernel gave to the program (context, task_struct, a map
// value) is a pointer for the verifier while it is in a register or in the
// stack of the program. Stored in arena it comes back as a number. A local
// whose address is taken goes to the stack in arena (BPFArenaStack.cpp), so a
// struct or the environment of a closure that holds such a pointer loses it.
//
// This pass runs in front of BPFArenaStack and marks the locals that have to
// stay in the stack of the program.
//
// - It finds the values that may be pointers of the kernel: the context, what
//   is loaded from it or from an object of the kernel, what helpers and kfuncs
//   return. It follows them through arguments, results and fields of locals at
//   constant offsets, also when a callee stores them through a pointer.
// - A local that holds one stays when the pointer is loaded back from it, or
//   may be, because the address of the local is stored or passed to an indirect
//   call, and all uses of the address are visible: loads, stores, small memcpy,
//   calls. The address must not go to arena either, so a local that holds that
//   address stays too.
// - The kernel takes addr_space_cast of a pointer to the stack for a move, but
//   one cast insn cannot see a pointer to the stack and a number. A function
//   that is called with a pointer to the stack in one place and with a pointer
//   to arena in another is copied, until every parameter is of one kind.
// - Loads that yield pointers to the stack are marked. BPFArenaCast follows
//   pointers, but not through memory.
//
// It gives up on phi and select of addresses, offsets that are not constant,
// large locals, and a kfunc that wants a pointer to arena in that argument.
// Such a local goes to arena like it does without the pass.
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

#define DEBUG_TYPE "bpf-arena-kptr"

STATISTIC(NumKept, "Number of locals kept in the stack of the program");
STATISTIC(NumCopies, "Number of functions copied for pointers to the stack");

static cl::opt<bool> DisableArenaKptr(
    "bpf-disable-arena-kptr", cl::Hidden,
    cl::desc("Locals that hold pointers of the kernel go to arena too"));
static cl::opt<unsigned>
    MaxLocalSize("bpf-arena-kptr-max-size", cl::Hidden, cl::init(256),
                 cl::desc("Locals that are larger than that go to arena"));

/// An offset that is not a constant.
static constexpr int64_t Unknown = INT64_MIN;

namespace {

class ArenaKptr {
  Module &M;
  const DataLayout &DL;
  function_ref<const TargetLibraryInfo &(Function &)> GetTLI;
  /// Values that may be pointers of the kernel.
  DenseSet<const Value *> K;
  /// Allocas and arguments -> offsets at which such a pointer is stored.
  DenseMap<const Value *, SmallSetVector<int64_t, 4>> Slots;
  /// The part of Slots of an argument that the function itself stores, not
  /// one of its callers.
  DenseMap<const Value *, SmallSetVector<int64_t, 4>> Written;
  /// Allocas and arguments from which such a pointer is loaded back.
  DenseSet<const Value *> ReadBack;
  /// Functions that may return such a pointer.
  DenseSet<const Function *> RetK;
  /// Track pointers to the given locals instead of pointers of the kernel.
  bool StackOnly = false;
  /// Locals whose address is stored in another local.
  mutable DenseSet<const Value *> AddrStored;

  std::pair<const Value *, int64_t> slot(const Value *Ptr) const;
  bool addSlot(const Value *Ptr, int64_t Delta = 0);
  bool visit(Instruction &I);
  void propagate();
  const char *hidden(Value *P, SmallPtrSetImpl<Value *> &Seen,
                     const Value *Root) const;
  bool escapes(ArrayRef<AllocaInst *> Keep) const;
  static bool specialize(Module &M, const ArenaKptr &S);

public:
  ArenaKptr(Module &M,
            function_ref<const TargetLibraryInfo &(Function &)> GetTLI)
      : M(M), DL(M.getDataLayout()), GetTLI(GetTLI) {}
  bool run();
};

} // namespace

static bool isObject(const Value *V) { return isa<AllocaInst, Argument>(V); }

static bool addressIsStored(const Value *P) {
  for (const Use &U : P->uses()) {
    if (isa<StoreInst>(U.getUser())) {
      if (U.getOperandNo() != StoreInst::getPointerOperandIndex())
        return true;
    } else if (isa<GetElementPtrInst>(U.getUser())) {
      if (addressIsStored(U.getUser()))
        return true;
    }
  }
  return false;
}

/// Which function is called is not known here. It may load the pointer back.
static bool addressGoesToIndirectCall(const Value *P) {
  for (const Use &U : P->uses()) {
    if (auto *CB = dyn_cast<CallBase>(U.getUser())) {
      if (CB->isIndirectCall() && CB->isArgOperand(&U))
        return true;
    } else if (isa<GetElementPtrInst>(U.getUser())) {
      if (addressGoesToIndirectCall(U.getUser()))
        return true;
    }
  }
  return false;
}

/// These kfuncs return memory in arena.
static bool returnsArena(const Function *F) {
  return F && (F->getName() == "bpf_alloc" ||
               F->getName() == "bpf_arena_alloc_pages");
}

std::pair<const Value *, int64_t> ArenaKptr::slot(const Value *Ptr) const {
  int64_t Off = 0;
  const Value *Base = GetPointerBaseWithConstantOffset(Ptr, Off, DL);

  if (isObject(Base))
    return {Base, Off};
  return {getUnderlyingObject(Ptr), Unknown};
}

bool ArenaKptr::addSlot(const Value *Ptr, int64_t Delta) {
  auto [Obj, Off] = slot(Ptr);

  if (!isObject(Obj))
    return false;
  Off = Off == Unknown || Delta == Unknown ? Unknown : Off + Delta;
  if (isa<Argument>(Obj))
    Written[Obj].insert(Off);
  return Slots[Obj].insert(Off);
}

bool ArenaKptr::visit(Instruction &I) {
  bool Changed = false;

  if (auto *LI = dyn_cast<LoadInst>(&I)) {
    if (!LI->getType()->isPointerTy() &&
        !(LI->getType()->isIntegerTy(64) &&
          any_of(LI->users(), IsaPred<IntToPtrInst>)))
      return false;
    auto [Obj, Off] = slot(LI->getPointerOperand());
    auto It = Slots.find(Obj);
    if (It != Slots.end() && (Off == Unknown || It->second.count(Off) ||
                              It->second.count(Unknown))) {
      Changed |= ReadBack.insert(Obj).second;
      Changed |= K.insert(LI).second;
    }
    // A field of an object of the kernel.
    if (!StackOnly && K.count(getUnderlyingObject(LI->getPointerOperand())))
      Changed |= K.insert(LI).second;
  } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
    if (!K.count(SI->getValueOperand()))
      return false;
    Changed |= addSlot(SI->getPointerOperand());
  } else if (auto *MT = dyn_cast<MemTransferInst>(&I)) {
    auto [Src, SrcOff] = slot(MT->getSource());
    auto It = Slots.find(Src);
    if (It == Slots.end())
      return false;
    auto *Len = dyn_cast<ConstantInt>(MT->getLength());
    SmallVector<int64_t, 4> Offs(It->second.begin(), It->second.end());
    for (int64_t O : Offs) {
      if (O == Unknown || SrcOff == Unknown || !Len)
        Changed |= addSlot(MT->getDest(), Unknown);
      else if (O >= SrcOff && O < SrcOff + (int64_t)Len->getZExtValue())
        Changed |= addSlot(MT->getDest(), O - SrcOff);
    }
  } else if (auto *CB = dyn_cast<CallBase>(&I)) {
    Function *Callee = CB->getCalledFunction();

    if (isBPFKernelFunc(*CB, GetTLI(*I.getFunction()))) {
      if (!StackOnly && CB->getType()->isPointerTy() && !returnsArena(Callee))
        Changed |= K.insert(CB).second;
      return Changed;
    }
    if (!Callee || Callee->isDeclaration() || Callee->isVarArg() ||
        Callee->arg_size() != CB->arg_size())
      return false;
    for (unsigned N = 0; N < CB->arg_size(); N++) {
      Value *Op = CB->getArgOperand(N);
      Argument *A = Callee->getArg(N);

      if (K.count(Op))
        Changed |= K.insert(A).second;
      if (!Op->getType()->isPointerTy())
        continue;
      auto [Obj, Off] = slot(Op);
      if (!isObject(Obj))
        continue;
      // What the caller stored is seen by the callee and the other way round.
      if (auto It = Slots.find(Obj); It != Slots.end()) {
        SmallVector<int64_t, 4> Offs(It->second.begin(), It->second.end());
        for (int64_t O : Offs)
          Changed |= Slots[A].insert(O == Unknown || Off == Unknown ? Unknown
                                                                    : O - Off);
      }
      if (auto It = Written.find(A); It != Written.end()) {
        SmallVector<int64_t, 4> Offs(It->second.begin(), It->second.end());
        for (int64_t O : Offs) {
          O = O == Unknown || Off == Unknown ? Unknown : O + Off;
          Changed |= Slots[Obj].insert(O);
          if (isa<Argument>(Obj))
            Written[Obj].insert(O);
        }
      }
      if (ReadBack.count(A))
        Changed |= ReadBack.insert(Obj).second;
    }
    if (RetK.count(Callee) && CB->getType()->isPointerTy())
      Changed |= K.insert(CB).second;
  } else if (auto *RI = dyn_cast<ReturnInst>(&I)) {
    if (RI->getReturnValue() && K.count(RI->getReturnValue()))
      Changed |= RetK.insert(I.getFunction()).second;
  } else if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
    if (K.count(GEP->getPointerOperand()))
      Changed |= K.insert(&I).second;
  } else if (auto *Sel = dyn_cast<SelectInst>(&I)) {
    if (K.count(Sel->getTrueValue()) || K.count(Sel->getFalseValue()))
      Changed |= K.insert(&I).second;
  } else if (isa<PHINode, CastInst, FreezeInst>(I)) {
    if (any_of(I.operands(), [&](const Use &U) { return K.count(U.get()); }))
      Changed |= K.insert(&I).second;
  }
  return Changed;
}

/// Why not all uses of the address \p P can be seen, or null if they can.
const char *ArenaKptr::hidden(Value *P, SmallPtrSetImpl<Value *> &Seen,
                              const Value *Root) const {
  if (!Seen.insert(P).second)
    return nullptr;
  for (Use &U : P->uses()) {
    auto *I = cast<Instruction>(U.getUser());

    if (isa<LoadInst, ICmpInst>(I))
      continue;
    if (auto *SI = dyn_cast<StoreInst>(I)) {
      if (U.getOperandNo() == StoreInst::getPointerOperandIndex())
        continue;
      // Fine in a local that stays in the stack too. What is loaded from there
      // is looked at by escapes().
      auto [Obj, Off] = slot(SI->getPointerOperand());
      if (!isObject(Obj) || Off == Unknown)
        return "stored";
      AddrStored.insert(Root);
      continue;
    }
    if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
      if (!GEP->hasAllConstantIndices())
        return "variable offset";
      if (const char *Why = hidden(GEP, Seen, Root))
        return Why;
      continue;
    }
    if (isa<PHINode, SelectInst>(I))
      return "phi or select";
    auto *CB = dyn_cast<CallBase>(I);
    if (!CB)
      return "other use";
    if (isa<LifetimeIntrinsic, DbgInfoIntrinsic>(CB))
      continue;
    if (auto *MI = dyn_cast<MemIntrinsic>(CB)) {
      // What is expanded to loads and stores.
      auto *Len = dyn_cast<ConstantInt>(MI->getLength());
      if (Len && Len->getZExtValue() <= 128)
        continue;
      return "large memcpy or memset";
    }
    if (isBPFArenaArg(*CB, U))
      return "kfunc that wants arena";
    if (isBPFKernelFunc(*CB, GetTLI(*I->getFunction())))
      continue;
    if (CB->isInlineAsm())
      return "inline asm";
    Function *Callee = CB->getCalledFunction();
    // BPFArenaCast follows the pointer into the functions that may be called.
    if (!Callee && CB->isArgOperand(&U))
      continue;
    if (!Callee)
      return "other use";
    if (Callee->isDeclaration() || Callee->isVarArg() ||
        !Callee->hasLocalLinkage() || !CB->isArgOperand(&U) ||
        Callee->arg_size() != CB->arg_size() || Callee->arg_size() > 64)
      return "call of something else";
    if (const char *Why =
            hidden(Callee->getArg(CB->getArgOperandNo(&U)), Seen, Root))
      return Why;
  }
  return nullptr;
}

/// A pointer to one of the locals in \p Keep is loaded from another local and
/// not all of its uses can be seen.
bool ArenaKptr::escapes(ArrayRef<AllocaInst *> Keep) const {
  ArenaKptr S(M, GetTLI);

  S.StackOnly = true;
  S.K.insert(Keep.begin(), Keep.end());
  S.propagate();
  return any_of(S.K, [&](const Value *V) {
    SmallPtrSet<Value *, 8> Seen;
    auto *LI = const_cast<LoadInst *>(dyn_cast<LoadInst>(V));

    return LI && S.hidden(LI, Seen, LI);
  });
}

void ArenaKptr::propagate() {
  for (bool Changed = true; Changed;) {
    Changed = false;
    for (Function &F : M)
      for (Instruction &I : instructions(F))
        Changed |= visit(I);
  }
}

/// Every parameter has to be of one kind: a pointer to the stack or not.
/// Give the calls that pass pointers to the stack a copy of the function.
bool ArenaKptr::specialize(Module &M, const ArenaKptr &S) {
  MapVector<Function *, MapVector<uint64_t, SmallVector<CallBase *, 4>>> Sites;

  for (Function &F : M) {
    for (Instruction &I : instructions(F)) {
      auto *CB = dyn_cast<CallBase>(&I);
      Function *Callee = CB ? CB->getCalledFunction() : nullptr;
      uint64_t Mask = 0;

      if (!Callee || Callee->isDeclaration() || !Callee->hasLocalLinkage() ||
          Callee->arg_size() != CB->arg_size() || CB->arg_size() > 64)
        continue;
      for (unsigned N = 0; N < CB->arg_size(); N++)
        if (S.K.count(CB->getArgOperand(N)))
          Mask |= 1ULL << N;
      Sites[Callee][Mask].push_back(CB);
    }
  }
  bool Changed = false;
  for (auto &[F, Masks] : Sites) {
    // The function itself is for one of them. Not when its address is taken:
    // indirect calls are not looked at here and may pass other pointers.
    bool First = !Masks.count(0) && !F->hasAddressTaken();
    for (auto &[Mask, Calls] : Masks) {
      if (!Mask || First) {
        First = false;
        continue;
      }
      ValueToValueMapTy VMap;
      Function *New = CloneFunction(F, VMap);
      New->setName(F->getName() + ".stack");
      ++NumCopies;
      for (CallBase *CB : Calls) {
        // A call of the function in itself was copied with it.
        if (CB->getFunction() == F)
          if (auto *C = dyn_cast_or_null<CallBase>(VMap.lookup(CB)))
            C->setCalledFunction(New);
        CB->setCalledFunction(New);
      }
      Changed = true;
    }
  }
  return Changed;
}

bool ArenaKptr::run() {
  // The context.
  for (Function &F : M)
    if (!F.isDeclaration() && F.hasSection() && F.arg_size())
      K.insert(F.getArg(0));
  propagate();

  // Locals that hold such a pointer. Their addresses must not go to arena
  // either, so a local that holds one of them is one more, and so on.
  SmallSetVector<AllocaInst *, 16> Cand;
  while (true) {
    unsigned Before = Cand.size();

    for (Function &F : M) {
      for (Instruction &I : instructions(F)) {
        auto *AI = dyn_cast<AllocaInst>(&I);

        // What is not loaded back as a pointer is fine in arena. It may be
        // loaded through the address when that is stored somewhere or passed
        // to a function that is not known.
        if (AI && AI->isStaticAlloca() && Slots.count(AI) &&
            !bpfStaysOnStack(*AI, GetTLI(F)) &&
            (ReadBack.count(AI) || addressIsStored(AI) ||
             addressGoesToIndirectCall(AI)))
          Cand.insert(AI);
      }
    }
    if (Cand.size() == Before)
      break;
    K.insert(Cand.begin(), Cand.end());
    propagate();
  }

  SmallVector<std::pair<AllocaInst *, const char *>, 16> Verdict;
  SmallVector<AllocaInst *, 16> Keep;

  for (AllocaInst *AI : Cand) {
    SmallPtrSet<Value *, 8> Seen;
    uint64_t Size = AI->getAllocationSize(DL)->getFixedValue();

    Verdict.push_back(
        {AI, Size > MaxLocalSize ? "too large" : hidden(AI, Seen, AI)});
    if (!Verdict.back().second)
      Keep.push_back(AI);
  }
  // The local that holds the address of another one may have gone to arena, or
  // the address gets out of sight after it is loaded from there. Which address
  // that is is not tracked.
  if (Keep.size() != Cand.size() || escapes(Keep)) {
    for (auto &[AI, Reason] : Verdict)
      if (!Reason && AddrStored.count(AI))
        Reason = "its address is stored";
    erase_if(Keep, [&](AllocaInst *AI) { return AddrStored.count(AI); });
  }
  LLVM_DEBUG(for (auto &[AI, Reason] : Verdict) dbgs()
             << AI->getFunction()->getName() << " %" << AI->getName() << ": "
             << (Reason ? Reason : "stays in the stack of the program")
             << "\n");
  if (Keep.empty())
    return false;

  MDNode *MD = MDNode::get(M.getContext(), {});
  for (AllocaInst *AI : Keep) {
    AI->setMetadata(BPFStackKeepMD, MD);
    AI->setMetadata(BPFArenaStackMD, nullptr);
  }
  NumKept += Keep.size();

  // Where do pointers to these locals get to? A copy of a function has calls
  // of its own, so it takes a few rounds.
  for (unsigned Round = 0;; ++Round) {
    ArenaKptr S(M, GetTLI);

    S.StackOnly = true;
    S.K.insert(Keep.begin(), Keep.end());
    S.propagate();
    if (Round < 8 && specialize(M, S))
      continue;

    for (const Value *V : S.K)
      if (auto *LI = dyn_cast<LoadInst>(V))
        const_cast<LoadInst *>(LI)->setMetadata(BPFStackPtrMD, MD);
    break;
  }
  return true;
}

namespace {

struct BPFArenaKptrLegacy : public ModulePass {
  static char ID;
  BPFArenaKptrLegacy() : ModulePass(ID) {}
  StringRef getPassName() const override { return "BPF Arena Kptr"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
  }

  bool runOnModule(Module &M) override {
    auto GetTLI = [this](Function &F) -> const TargetLibraryInfo & {
      return getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
    };
    return !DisableArenaKptr && ArenaKptr(M, GetTLI).run();
  }
};

} // namespace

char BPFArenaKptrLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(BPFArenaKptrLegacy, DEBUG_TYPE, "BPF Arena Kptr", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(BPFArenaKptrLegacy, DEBUG_TYPE, "BPF Arena Kptr", false,
                    false)

ModulePass *llvm::createBPFArenaKptrPass() { return new BPFArenaKptrLegacy(); }

PreservedAnalyses BPFArenaKptrPass::run(Module &M, ModuleAnalysisManager &AM) {
  if (DisableArenaKptr)
    return PreservedAnalyses::all();

  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto GetTLI = [&FAM](Function &F) -> const TargetLibraryInfo & {
    return FAM.getResult<TargetLibraryAnalysis>(F);
  };
  return ArenaKptr(M, GetTLI).run() ? PreservedAnalyses::none()
                                    : PreservedAnalyses::all();
}
