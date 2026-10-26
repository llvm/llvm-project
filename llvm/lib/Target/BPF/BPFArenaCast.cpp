//===-- BPFArenaCast.cpp - Access arena through address space casts -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A program that keeps its memory in arena and is written in a language without
// address spaces has pointers of one kind. The kernel has two. A pointer that
// it gave to the program (ctx, task_struct, a map value, the stack) is good for
// an access as it is. An address in arena is a number until it's told
// otherwise:
//
//   r2 = addr_space_cast(r1, 0, 1)
//   r0 = *(u64 *)(r2 + 0)
//
// Nothing in the source says which pointer is which, and the compiler cannot
// always find out. It relies on the kernel, which knows its own pointers:
// - The cast of a pointer of the kernel is a move. A cast that is not needed
//   does no harm.
// - An access through a number is rejected. A cast that is missing doesn't go
//   unnoticed.
// - A cast insn is a move on all paths or on none. A program where one cast
//   sees a pointer of the kernel on one path and a number on another is
//   rejected. NULL is a number.
//
// So a pointer that is dereferenced is cast, unless it's certain that it's not
// an address in arena. That saves insns, and the last point asks for it:
// 'phi [null], [task]' must not be cast.
//
// Pointers of the kernel are:
// - The argument of a program: ctx.
// - What helpers return.
// - Global variables that are not in arena: maps, ksyms, strings for helpers.
// - Locals in the stack of the program, and what a load that BPFArenaKptr.cpp
//   marked yields: a pointer to such a local.
// - The arguments of a callback. The kernel calls it with pointers of its own.
//
// The rest can be an address in arena:
// - Global variables and locals in arena.
// - What kfuncs return. Most of them return pointers of the kernel,
//   bpf_arena_alloc_pages() and the like return memory in arena, and
//   the declaration doesn't tell.
// - What other loads yield. Programs keep addresses in arena in map values,
//   user space passes them in ctx.
// - What comes from callers that are not seen.
// - The result of arithmetic on integers.
//
// That is followed through address arithmetic, phi, select, arguments and
// results of the functions of the module. An indirect call may get to every
// function of its type whose address is taken. NULL is neither: a pointer of
// the kernel or NULL is a pointer of the kernel.
//
// Pointers to task_struct and the like are loaded from ctx or returned by
// kfuncs. But fields of objects of the kernel are at offsets that are CO-RE
// relocations, so an access at such an offset goes without a cast, wherever
// the pointer comes from. It follows that an object of the kernel that is in
// arena, e.g. struct scx_cmask, cannot be accessed at such offsets: the kernel
// rejects the program.
//
// A pointer is cast once, where it is defined. The result of the cast is good
// for nothing but the access: it's the low 32 bits of the address. What is
// compared, stored or passed is the pointer as it was. Address arithmetic is
// done again on top of the cast: the kernel allows that on pointers to arena.
// Unless the result of the arithmetic is needed as a value as well. Then it's
// computed once and cast.
//
// Locals that are in the stack in arena (see BPFArenaStack.cpp) are accessed
// through r12 without a cast where the offset is a constant and the insn
// is a plain load or store.
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "BPFCORE.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

#define DEBUG_TYPE "bpf-arena-cast"

namespace {

/// What a pointer can be. A set.
enum Kind : unsigned {
  /// NULL or another constant that is not the address of a variable, or a value
  /// that was not looked at yet.
  Nothing = 0,
  /// A pointer that the kernel gave to the program.
  Kernel = 1,
  /// An address in arena, or nothing says that it's not.
  MaybeArena = 2,
};

/// The kinds of all pointers in the module, also of those that are integers
/// for a while. It starts with nothing and adds kinds until nothing changes.
/// No rule takes a kind away when its input gets one more, so the result
/// doesn't depend on the order.
class PointerKinds {
  function_ref<const TargetLibraryInfo &(Function &)> GetTLI;
  /// Arguments and instructions. For a function: what it returns.
  DenseMap<const Value *, unsigned> Kinds;
  /// The functions that an indirect call of that type may call.
  DenseMap<const FunctionType *, SmallVector<const Function *, 4>> Taken;

  bool add(const Value *V, unsigned K);
  SmallVector<const Function *, 4> callees(const CallBase &CB) const;
  unsigned eval(Instruction &I) const;
  bool visit(Instruction &I);

public:
  PointerKinds(Module &M,
               function_ref<const TargetLibraryInfo &(Function &)> GetTLI);
  unsigned get(const Value *V) const;
};

struct Access {
  Instruction *I;
  unsigned Idx;
  /// Address arithmetic, from the access down to the root.
  SmallVector<GetElementPtrInst *, 4> Chain;
};

class ArenaCast {
  Function &F;
  const PointerKinds &Kinds;
  /// Pointers to cast and what is accessed through them.
  MapVector<Value *, SmallVector<Access, 4>> Roots;
  /// Accesses through r12 whose address is computed in another block.
  SmallVector<std::pair<AllocaInst *, Access>, 4> Locals;
  /// Address arithmetic that is done on top of a cast in a block already.
  DenseMap<std::pair<Value *, BasicBlock *>, DenseMap<Value *, Value *>> Clones;
  DenseMap<Value *, bool> OnlyDeref;

  bool onlyDeref(GetElementPtrInst *GEP);
  Value *makeCast(IRBuilderBase &B, Value *V);
  void collect(Instruction &I, unsigned Idx, bool Plain);
  void use(Access &A, Value *K);
  void castConstant(Constant *C, MutableArrayRef<Access> As);
  void castValue(Value *V, MutableArrayRef<Access> As);

public:
  ArenaCast(Function &F, const PointerKinds &Kinds) : F(F), Kinds(Kinds) {}
  bool run();
};

} // namespace

bool PointerKinds::add(const Value *V, unsigned K) {
  unsigned &Old = Kinds[V];

  if ((Old | K) == Old)
    return false;
  Old |= K;
  return true;
}

unsigned PointerKinds::get(const Value *V) const {
  if (!isa<Constant>(V))
    return Kinds.lookup(V);

  const auto *GV = dyn_cast<GlobalVariable>(getUnderlyingObject(V));

  if (!GV)
    return Nothing;
  return isBPFArenaGlobal(*GV) ? MaybeArena : Kernel;
}

/// The functions of the module that the call may get to.
SmallVector<const Function *, 4>
PointerKinds::callees(const CallBase &CB) const {
  const Function *F = CB.getCalledFunction();

  if (F && !F->isDeclaration())
    return {F};
  if (CB.isIndirectCall())
    return Taken.lookup(CB.getFunctionType());
  return {};
}

unsigned PointerKinds::eval(Instruction &I) const {
  if (auto *AI = dyn_cast<AllocaInst>(&I))
    return AI->hasMetadata(BPFArenaStackMD) ? MaybeArena : Kernel;
  if (isa<LoadInst>(I))
    return I.hasMetadata(BPFStackPtrMD) ? Kernel : MaybeArena;
  if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
    return get(GEP->getPointerOperand());
  if (auto *Sel = dyn_cast<SelectInst>(&I))
    return get(Sel->getTrueValue()) | get(Sel->getFalseValue());
  if (isa<PHINode, IntToPtrInst, PtrToIntInst>(I)) {
    unsigned K = Nothing;

    for (const Use &U : I.operands())
      K |= get(U);
    return K;
  }
  if (auto *CB = dyn_cast<CallBase>(&I)) {
    // A kfunc has a name, a helper has a number.
    if (isBPFKernelFunc(*CB, GetTLI(*I.getFunction())))
      return CB->getCalledFunction() ? MaybeArena : Kernel;

    SmallVector<const Function *, 4> Fs = callees(*CB);
    unsigned K = Fs.empty() ? MaybeArena : Nothing;

    for (const Function *F : Fs)
      K |= Kinds.lookup(F);
    return K;
  }
  return MaybeArena;
}

bool PointerKinds::visit(Instruction &I) {
  Type *Ty = I.getType();
  bool Changed = false;

  if (auto *CB = dyn_cast<CallBase>(&I)) {
    for (const Function *F : callees(*CB))
      for (auto [Arg, Op] : zip(F->args(), CB->args()))
        Changed |= add(&Arg, get(Op));
    // The kernel calls a callback with pointers of its own.
    if (isBPFKernelFunc(*CB, GetTLI(*I.getFunction())))
      for (const Use &U : CB->args())
        if (const auto *F = dyn_cast<Function>(U))
          for (const Argument &Arg : F->args())
            Changed |= add(&Arg, Kernel);
  } else if (auto *RI = dyn_cast<ReturnInst>(&I)) {
    if (const Value *V = RI->getReturnValue())
      Changed |= add(I.getFunction(), get(V));
  }
  if (Ty->isPointerTy() || Ty->isIntegerTy(64))
    Changed |= add(&I, eval(I));
  return Changed;
}

PointerKinds::PointerKinds(
    Module &M, function_ref<const TargetLibraryInfo &(Function &)> GetTLI)
    : GetTLI(GetTLI) {
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    // A program.
    if (F.hasSection()) {
      if (F.arg_size())
        add(F.getArg(0), Kernel);
      continue;
    }
    // Not all callers are seen. Calls of memset() and the like appear when
    // insns are selected.
    if (!F.hasLocalLinkage() || GetTLI(F).getLibFunc(F) != NotLibFunc)
      for (Argument &A : F.args())
        add(&A, MaybeArena);
    if (F.hasAddressTaken())
      Taken[F.getFunctionType()].push_back(&F);
  }
  for (bool Changed = true; Changed;) {
    Changed = false;
    for (Function &F : M)
      for (Instruction &I : instructions(F))
        Changed |= visit(I);
  }
}

/// The number is the same in both address spaces. inttoptr is how to say that.
Value *ArenaCast::makeCast(IRBuilderBase &B, Value *V) {
  LLVMContext &C = F.getContext();
  Value *Int = B.CreatePtrToInt(V, Type::getInt64Ty(C));
  Value *User = B.CreateIntToPtr(Int, PointerType::get(C, 1));

  return B.CreateAddrSpaceCast(User, PointerType::get(C, 0));
}

/// The address is not needed for anything but to access memory at it.
bool ArenaCast::onlyDeref(GetElementPtrInst *GEP) {
  auto It = OnlyDeref.find(GEP);

  if (It != OnlyDeref.end())
    return It->second;

  bool Res = all_of(GEP->uses(), [&](const Use &U) {
    User *Usr = U.getUser();

    if (isa<LoadInst>(Usr))
      return true;
    if (isa<StoreInst>(Usr))
      return U.getOperandNo() == StoreInst::getPointerOperandIndex();
    if (isa<AtomicRMWInst>(Usr))
      return U.getOperandNo() == AtomicRMWInst::getPointerOperandIndex();
    if (isa<AtomicCmpXchgInst>(Usr))
      return U.getOperandNo() == AtomicCmpXchgInst::getPointerOperandIndex();
    if (isa<MemIntrinsic>(Usr))
      return U.getOperandNo() < 2;
    if (auto *Next = dyn_cast<GetElementPtrInst>(Usr))
      return onlyDeref(Next);
    return false;
  });
  return OnlyDeref[GEP] = Res;
}

/// A CO-RE relocation looks like a load until it's lowered. GVN makes phis of
/// them.
static bool isFieldOffset(const Value *V,
                          SmallPtrSetImpl<const Value *> &Seen) {
  if (const auto *LI = dyn_cast<LoadInst>(V)) {
    const auto *GV = dyn_cast<GlobalVariable>(LI->getPointerOperand());

    return GV && GV->hasAttribute(BPFCoreSharedInfo::AmaAttr);
  }
  if (!isa<PHINode, ZExtInst>(V))
    return false;
  return !Seen.insert(V).second ||
         all_of(cast<Instruction>(V)->operands(),
                [&](const Use &U) { return isFieldOffset(U, Seen); });
}

/// The address of a field, or of a part of it.
static bool isField(const Value *Ptr) {
  for (; const auto *GEP = dyn_cast<GetElementPtrInst>(Ptr);
       Ptr = GEP->getPointerOperand())
    if (any_of(GEP->indices(), [](const Use &U) {
          SmallPtrSet<const Value *, 4> Seen;

          return isFieldOffset(U, Seen);
        }))
      return true;
  return false;
}

/// \p Plain: a load or a store that is not atomic.
void ArenaCast::collect(Instruction &I, unsigned Idx, bool Plain) {
  Value *Ptr = I.getOperand(Idx);
  Access A = {&I, Idx, {}};
  bool ConstOff = true;

  if (Ptr->getType()->getPointerAddressSpace() != 0)
    return;
  if (Kinds.get(Ptr) == Kernel || isField(Ptr))
    return;
  while (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
    if (!GEP->hasAllConstantIndices()) {
      // It is computed anyway. A constant offset costs nothing to add again.
      if (!onlyDeref(GEP))
        break;
      ConstOff = false;
    }
    A.Chain.push_back(GEP);
    Ptr = GEP->getPointerOperand();
  }
  // It is cast already, or there is nothing to access.
  if (isa<AddrSpaceCastInst, ConstantPointerNull, UndefValue>(Ptr))
    return;
  if (auto *AI = dyn_cast<AllocaInst>(Ptr)) {
    if (Plain && ConstOff) {
      // Insns are selected block by block: the offset gets into the insn when
      // it is added in the block of the access.
      if (any_of(A.Chain, [&](GetElementPtrInst *GEP) {
            return GEP->getParent() != I.getParent();
          }))
        Locals.push_back({AI, std::move(A)});
      return;
    }
  }
  Roots[Ptr].push_back(std::move(A));
}

/// Make the access go through \p K, which is the cast of its root.
void ArenaCast::use(Access &A, Value *K) {
  auto &Done = Clones[{K, A.I->getParent()}];

  for (GetElementPtrInst *GEP : reverse(A.Chain)) {
    Value *&Clone = Done[GEP];

    if (!Clone) {
      auto *New = cast<GetElementPtrInst>(GEP->clone());

      New->setOperand(0, K);
      New->insertBefore(A.I->getIterator());
      Clone = New;
    }
    K = Clone;
  }
  A.I->setOperand(A.Idx, K);
}

/// The cast is a constant expression, every block that uses it has a copy.
void ArenaCast::castConstant(Constant *C, MutableArrayRef<Access> As) {
  const DataLayout &DL = F.getDataLayout();
  APInt Off(DL.getIndexTypeSizeInBits(C->getType()), 0);
  auto *Base =
      cast<Constant>(C->stripAndAccumulateConstantOffsets(DL, Off, true));
  LLVMContext &Ctx = F.getContext();
  Constant *K = ConstantExpr::getAddrSpaceCast(
      ConstantExpr::getIntToPtr(
          ConstantExpr::getPtrToInt(Base, Type::getInt64Ty(Ctx)),
          PointerType::get(Ctx, 1)),
      PointerType::get(Ctx, 0));

  // The offset goes into the insn that accesses memory.
  K = ConstantExpr::getPtrAdd(K, ConstantInt::get(Ctx, Off));
  for (Access &A : As)
    use(A, K);
}

void ArenaCast::castValue(Value *V, MutableArrayRef<Access> As) {
  std::optional<BasicBlock::iterator> IP;

  if (auto *I = dyn_cast<Instruction>(V))
    IP = I->getInsertionPointAfterDef();
  else
    IP = F.getEntryBlock().getFirstNonPHIOrAlloca();

  if (IP) {
    IRBuilder<> B(*IP);
    Value *K = makeCast(B, V);

    for (Access &A : As)
      use(A, K);
    return;
  }

  // There is no place right after the definition, e.g. the result of invoke
  // whose normal destination has other predecessors.
  for (Access &A : As) {
    IRBuilder<> B(A.I);

    use(A, makeCast(B, V));
  }
}

bool ArenaCast::run() {
  for (Instruction &I : instructions(F)) {
    if (auto *LI = dyn_cast<LoadInst>(&I)) {
      collect(I, LoadInst::getPointerOperandIndex(), !LI->isAtomic());
    } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
      collect(I, StoreInst::getPointerOperandIndex(), !SI->isAtomic());
    } else if (isa<AtomicRMWInst>(I)) {
      collect(I, AtomicRMWInst::getPointerOperandIndex(), false);
    } else if (isa<AtomicCmpXchgInst>(I)) {
      collect(I, AtomicCmpXchgInst::getPointerOperandIndex(), false);
    } else if (isa<MemIntrinsic>(I)) {
      // memset is expanded into stores or is a call that takes the address.
      // memcpy is expanded into loads and stores through registers.
      collect(I, 0, isa<MemSetInst>(I));
      if (isa<MemTransferInst>(I))
        collect(I, 1, false);
    }
  }

  SmallVector<WeakTrackingVH, 32> Old;

  for (auto &[AI, A] : Locals) {
    Old.append(A.Chain.begin(), A.Chain.end());
    use(A, AI);
  }
  for (auto &[Root, As] : Roots) {
    LLVM_DEBUG({
      dbgs() << F.getName() << ": cast ";
      Root->printAsOperand(dbgs(), false);
      if (Kinds.get(Root) & Kernel)
        dbgs() << ", can be a pointer of the kernel";
      dbgs() << "\n";
    });
    for (Access &A : As)
      Old.append(A.Chain.begin(), A.Chain.end());
    if (auto *C = dyn_cast<Constant>(Root))
      castConstant(C, As);
    else
      castValue(Root, As);
  }

  // Address arithmetic that nothing uses anymore.
  for (WeakTrackingVH &VH : Old)
    if (VH)
      RecursivelyDeleteTriviallyDeadInstructions(VH);
  return !Roots.empty() || !Locals.empty();
}

static bool
castModule(Module &M,
           function_ref<const TargetLibraryInfo &(Function &)> GetTLI) {
  PointerKinds Kinds(M, GetTLI);
  bool Changed = false;

  for (Function &F : M)
    if (!F.isDeclaration())
      Changed |= ArenaCast(F, Kinds).run();
  return Changed;
}

namespace {

struct BPFArenaCastLegacy : public ModulePass {
  static char ID;
  BPFArenaCastLegacy() : ModulePass(ID) {}
  StringRef getPassName() const override { return "BPF Arena Cast"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.setPreservesCFG();
  }

  bool runOnModule(Module &M) override {
    return castModule(M, [this](Function &F) -> const TargetLibraryInfo & {
      return getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
    });
  }
};

} // namespace

char BPFArenaCastLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(BPFArenaCastLegacy, DEBUG_TYPE, "BPF Arena Cast", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(BPFArenaCastLegacy, DEBUG_TYPE, "BPF Arena Cast", false,
                    false)

ModulePass *llvm::createBPFArenaCastPass() { return new BPFArenaCastLegacy(); }

PreservedAnalyses BPFArenaCastPass::run(Module &M, ModuleAnalysisManager &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  if (!castModule(M, [&FAM](Function &F) -> const TargetLibraryInfo & {
        return FAM.getResult<TargetLibraryAnalysis>(F);
      }))
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
