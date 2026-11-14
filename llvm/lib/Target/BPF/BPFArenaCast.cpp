//===-- BPFArenaCast.cpp - Access arena through address space casts -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A program that keeps its memory in arena and doesn't mark the pointers to it
// has pointers of one kind. The kernel has two. A pointer that it gave to
// the program (ctx, task_struct, a map value, the stack) is good for an access
// as it is. An address in arena is a number until it's told otherwise:
//
//   r2 = addr_space_cast(r1, 0, 1)
//   r0 = *(u64 *)(r2 + 0)
//
// So 'ptr' is a flat pointer here, like the generic pointer of a GPU. What
// resolves it is the verifier, when the program is loaded:
// - The cast of a pointer of the kernel is a move. A cast that is not needed
//   does no harm.
// - An access through a number is rejected. A cast that is missing doesn't go
//   unnoticed.
// - A cast insn is a move on all paths or on none. A program where one cast
//   sees a pointer of the kernel on one path and a number on another is
//   rejected. NULL is in both address spaces.
//
// A pointer that is dereferenced is cast, unless it's certain that it's not
// an address in arena. That saves insns.
//
// Pointers of the kernel are:
// - The argument of a program: ctx.
// - What helpers and kfuncs return, unless the type is 'ptr addrspace(1)'.
// - Global variables of this file that are not in arena, maps, ksyms.
// - Locals in the stack of the program, and what a load that BPFArenaKptr.cpp
//   marked reads from there: a pointer to such a local.
// - The arguments of a callback. The kernel calls it with pointers of its own.
//
// The rest can be an address in arena:
// - Global variables and locals in arena, 'ptr addrspace(1)'.
// - What other loads yield. Programs keep addresses in arena in map values,
//   user space passes them in ctx.
// - What another file of the program defines: variables, the results of its
//   functions, the arguments of a function that it can call.
// - The result of arithmetic on integers.
//
// That is followed through address arithmetic, phi, select, freeze, arguments
// and results of the functions of the module. An indirect call may get to every
// function of its type whose address is taken. NULL is neither. A pointer that
// can be nothing but NULL is cast.
//
// The pass has four steps.
//
// 1. addrspacecast in the source keeps the value. A pointer to arena is
//    the same number with and without the mark, and the program stores,
//    compares and passes that number. Only the casts of step 4 are insns.
// 2. A static function that gets a pointer of the kernel from one caller and
//    an address in arena from another is copied ('f.kern'), until every
//    parameter is of one kind.
// 3. The kernel verifies a function that is not static on its own, from its
//    type in BTF. Pointers in its signature get btf_type_tag("arena"). Not
//    a parameter that the source describes with a decl tag ('arg:ctx'), that
//    gets to a helper, or that a caller in this file passes a pointer of
//    the kernel in.
// 4. A pointer is cast once, where it is defined. The result of the cast is
//    good for nothing but the access: it's the low 32 bits of the address.
//    Address arithmetic is done again on top of the cast: the kernel allows
//    that on pointers to arena. Unless the result of the arithmetic is needed
//    as a value as well. Then it's computed once and cast.
//
// Locals that are in the stack in arena (see BPFArenaStack.cpp) are accessed
// through r12 without a cast where the offset is a constant and the insn
// is a plain load or store.
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "BPFArena.h"
#include "BPFCORE.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

#define DEBUG_TYPE "bpf-arena-cast"

STATISTIC(NumCopies, "Number of functions copied for pointers of the kernel");

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
  bool alignLocals(MemTransferInst &MT);
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
  if (!isa<Constant>(V)) {
    Type *Ty = V->getType();

    if (Ty->isPointerTy() && Ty->getPointerAddressSpace() == BPFArenaAS)
      return MaybeArena;
    return Kinds.lookup(V);
  }

  const Value *Obj = getUnderlyingObject(V);
  const auto *GV = dyn_cast<GlobalVariable>(Obj);

  // A number that the program uses as an address, or NULL.
  if (!GV)
    return Operator::getOpcode(Obj) == Instruction::IntToPtr ? MaybeArena
                                                             : Nothing;
  if (isBPFArenaGlobal(*GV))
    return MaybeArena;
  // The file that defines the variable picks the place for it. A CO-RE
  // relocation looks like such a declaration.
  if (GV->isDeclaration() && !GV->hasSection() &&
      !GV->hasAttribute(BPFCoreSharedInfo::AmaAttr) &&
      !GV->hasAttribute(BPFCoreSharedInfo::TypeIdAttr))
    return MaybeArena;
  return Kernel;
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
  // The stack of the program keeps a pointer, arena makes a number of it.
  if (auto *LI = dyn_cast<LoadInst>(&I))
    return I.hasMetadata(BPFStackPtrMD) ? get(LI->getPointerOperand())
                                        : MaybeArena;
  if (auto *GEP = dyn_cast<GetElementPtrInst>(&I))
    return get(GEP->getPointerOperand());
  if (auto *Sel = dyn_cast<SelectInst>(&I))
    return get(Sel->getTrueValue()) | get(Sel->getFalseValue());
  if (isa<PHINode, IntToPtrInst, PtrToIntInst, FreezeInst>(I)) {
    unsigned K = Nothing;

    for (const Use &U : I.operands())
      K |= get(U);
    return K;
  }
  if (auto *CB = dyn_cast<CallBase>(&I)) {
    // Memory in arena has another type, see get().
    if (isBPFKernelFunc(*CB))
      return Kernel;

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
    if (isBPFKernelFunc(*CB))
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
    Module &M, function_ref<const TargetLibraryInfo &(Function &)> GetTLI) {
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

/// \p Plain: a load or a store that is not atomic.
void ArenaCast::collect(Instruction &I, unsigned Idx, bool Plain) {
  Value *Ptr = I.getOperand(Idx);
  Access A = {&I, Idx, {}};
  bool ConstOff = true;

  if (Ptr->getType()->getPointerAddressSpace() != 0)
    return;
  if (Kinds.get(Ptr) == Kernel)
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
  // There is nothing to access. An access through NULL is one to arena, see
  // BPFArenaNullPass.
  if (isa<UndefValue>(Ptr))
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

/// A copy goes 8 bytes at a time if both sides are aligned. If it takes too
/// many insns it is a call of memcpy(). Insn selection gives a local that it
/// copies to the alignment that it wants, but it does not see the local behind
/// a cast. A copy within the memory of the kernel stays as it is.
bool ArenaCast::alignLocals(MemTransferInst &MT) {
  const DataLayout &DL = F.getDataLayout();
  bool Changed = false;

  if (Kinds.get(MT.getRawDest()) == Kernel &&
      Kinds.get(MT.getRawSource()) == Kernel)
    return false;

  auto Known = [&](Value *Ptr) {
    MaybeAlign Pref;

    if (isa<AllocaInst>(Ptr->stripPointerCasts()))
      Pref = Align::Constant<8>();
    return getOrEnforceKnownAlignment(Ptr, Pref, DL, &MT);
  };
  if (Align A = Known(MT.getRawDest()); A > MT.getDestAlign().valueOrOne()) {
    MT.setDestAlignment(A);
    Changed = true;
  }
  if (Align A = Known(MT.getRawSource());
      A > MT.getSourceAlign().valueOrOne()) {
    MT.setSourceAlignment(A);
    Changed = true;
  }
  return Changed;
}

bool ArenaCast::run() {
  bool Changed = false;

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
      if (auto *MT = dyn_cast<MemTransferInst>(&I))
        Changed |= alignLocals(*MT);
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
  return Changed || !Roots.empty() || !Locals.empty();
}

static Constant *flatten(Constant *C) {
  auto *CE = dyn_cast<ConstantExpr>(C);
  SmallVector<Constant *, 4> Ops;

  if (!CE)
    return C;
  for (Use &U : CE->operands())
    Ops.push_back(flatten(cast<Constant>(U)));
  if (CE->getOpcode() != Instruction::AddrSpaceCast)
    return CE->getWithOperands(Ops);
  return ConstantExpr::getIntToPtr(
      ConstantExpr::getPtrToInt(Ops[0], Type::getInt64Ty(C->getContext())),
      CE->getType());
}

/// Step 1: addrspacecast keeps the value.
static bool flattenCasts(Function &F) {
  bool Changed = false;

  for (Instruction &I : make_early_inc_range(instructions(F))) {
    for (Use &U : I.operands()) {
      auto *C = dyn_cast<ConstantExpr>(U);
      Constant *Flat = C ? flatten(C) : nullptr;

      if (Flat != C) {
        U.set(Flat);
        Changed = true;
      }
    }

    auto *Cast = dyn_cast<AddrSpaceCastInst>(&I);

    if (!Cast || !Cast->getType()->isPointerTy())
      continue;

    IRBuilder<> B(Cast);
    Value *V = B.CreateIntToPtr(
        B.CreatePtrToInt(Cast->getPointerOperand(), B.getInt64Ty()),
        Cast->getType());

    V->takeName(Cast);
    Cast->replaceAllUsesWith(V);
    Cast->eraseFromParent();
    Changed = true;
  }
  return Changed;
}

/// Step 2: the calls that pass pointers of the kernel where other calls pass
/// addresses in arena get a copy of the function.
static bool copyFunctions(Module &M, const PointerKinds &Kinds) {
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
        if (CB->getArgOperand(N)->getType()->isPointerTy() &&
            Kinds.get(CB->getArgOperand(N)) == Kernel &&
            Kinds.get(Callee->getArg(N)) != Kernel)
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

      New->setName(F->getName() + ".kern");
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

static bool hasAnnotation(DINodeArray Annotations, StringRef Name,
                          StringRef Prefix) {
  if (!Annotations)
    return false;
  return any_of(Annotations->operands(), [&](const MDOperand &Op) {
    const auto *MD = cast<MDNode>(Op);

    return cast<MDString>(MD->getOperand(0))->getString() == Name &&
           cast<MDString>(MD->getOperand(1))->getString().starts_with(Prefix);
  });
}

/// The type \p Ty with the tag if it's a pointer.
static DIType *tagPointer(DIBuilder &DIB, DIType *Ty) {
  LLVMContext &C = Ty->getContext();
  auto *D = dyn_cast<DIDerivedType>(Ty);

  while (D && (D->getTag() == dwarf::DW_TAG_typedef ||
               D->getTag() == dwarf::DW_TAG_const_type ||
               D->getTag() == dwarf::DW_TAG_volatile_type ||
               D->getTag() == dwarf::DW_TAG_restrict_type))
    D = dyn_cast_or_null<DIDerivedType>(D->getBaseType());
  if (!D || D->getTag() != dwarf::DW_TAG_pointer_type ||
      hasAnnotation(D->getAnnotations(), "btf_type_tag", "arena"))
    return Ty;

  SmallVector<Metadata *, 2> Annotations;

  if (DINodeArray Old = D->getAnnotations())
    for (const MDOperand &Op : Old->operands())
      Annotations.push_back(Op);
  Annotations.push_back(MDNode::get(
      C, {MDString::get(C, "btf_type_tag"), MDString::get(C, "arena")}));
  return DIB.createPointerType(D->getBaseType(), D->getSizeInBits(),
                               D->getAlignInBits(), std::nullopt, D->getName(),
                               DIB.getOrCreateArray(Annotations));
}

/// Step 3: tell the kernel that the pointers in the signature of a function
/// that is not static are addresses in arena.
static bool tagSignatures(Module &M, const PointerKinds &Kinds) {
  DIBuilder DIB(M, /*AllowUnresolved=*/false);
  BPFKernelMem KM(M);
  bool Changed = false;

  for (Function &F : M) {
    DISubprogram *SP = F.getSubprogram();

    if (F.isDeclaration() || F.hasLocalLinkage() || F.hasSection() || !SP)
      continue;

    DISubroutineType *ST = SP->getType();
    DITypeArray Old = ST->getTypeArray();
    SmallVector<Metadata *, 8> Types;
    bool New = false;

    // The first one is the type of the result. Not a parameter that has to be
    // memory of the kernel, or is for a caller in this file.
    for (unsigned N = 0; N < Old.size(); ++N) {
      const Argument *A = N ? getBPFSourceParam(F, N - 1) : nullptr;
      bool Kern = A && (KM.isKernelParam(*A) || (Kinds.get(A) & Kernel));
      DIType *Ty = Old[N];
      DIType *Tagged = Ty && !Kern ? tagPointer(DIB, Ty) : Ty;

      New |= Tagged != Ty;
      Types.push_back(Tagged);
    }
    if (!New)
      continue;
    SP->replaceType(DIB.createSubroutineType(DIB.getOrCreateTypeArray(Types),
                                             ST->getFlags(), ST->getCC()));
    Changed = true;
  }
  return Changed;
}

static bool
castModule(Module &M,
           function_ref<const TargetLibraryInfo &(Function &)> GetTLI) {
  std::optional<PointerKinds> Kinds;
  bool Changed = false;

  for (Function &F : M)
    Changed |= flattenCasts(F);
  // A copy of a function has calls of its own, so it takes a few rounds.
  for (unsigned Round = 0;; ++Round) {
    Kinds.emplace(M, GetTLI);
    if (Round == 8 || !copyFunctions(M, *Kinds))
      break;
    Changed = true;
  }
  Changed |= tagSignatures(M, *Kinds);

  for (Function &F : M)
    if (!F.isDeclaration())
      Changed |= ArenaCast(F, *Kinds).run();
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
