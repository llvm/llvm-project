//===-- BPFArenaCast.cpp - Access arena through address space casts -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A program that keeps all of its memory in arena and is written in a language
// without address spaces has pointers of one kind. For the kernel an address in
// arena is a number until it's told otherwise:
//
//   r2 = addr_space_cast(r1, 0, 1)
//   r0 = *(u64 *)(r2 + 0)
//
// Nothing in the source says which pointers are addresses in arena, so cast
// every pointer that is dereferenced. The kernel takes the cast of a pointer
// that it knows (ctx, task_struct, map value) for a move. A pointer to the
// stack is the exception, it's not cast.
//
// The result of the cast is good for nothing but the access: it's the low 32
// bits of the address. What is compared, stored or passed is the pointer as
// it was.
//
// There is one cast per pointer and not per access. It's where the pointer is
// defined, and address arithmetic is done again on top of it: the kernel allows
// that on pointers to arena. Unless the result of the arithmetic is needed as
// a value as well. Then it's computed once and cast.
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
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

#define DEBUG_TYPE "bpf-arena-cast"

namespace {

struct Access {
  Instruction *I;
  unsigned Idx;
  /// Address arithmetic, from the access down to the root.
  SmallVector<GetElementPtrInst *, 4> Chain;
};

class ArenaCast {
  Function &F;
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
  ArenaCast(Function &F) : F(F) {}
  bool run();
};

} // namespace

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
  if (auto *Arg = dyn_cast<Argument>(Ptr))
    if (Arg->hasAttribute(BPFStackPtrAttr))
      return;
  if (auto *LI = dyn_cast<LoadInst>(Ptr))
    if (LI->hasMetadata(BPFStackPtrMD))
      return;
  if (auto *AI = dyn_cast<AllocaInst>(Ptr)) {
    if (!AI->hasMetadata(BPFArenaStackMD))
      return;
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

  // Not memory: CO-RE relocations look like loads until they are lowered.
  if (auto *GV = dyn_cast<GlobalVariable>(Base))
    if (GV->hasAttribute(BPFCoreSharedInfo::AmaAttr) ||
        GV->hasAttribute(BPFCoreSharedInfo::TypeIdAttr))
      return;

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

namespace {

struct BPFArenaCastLegacy : public FunctionPass {
  static char ID;
  BPFArenaCastLegacy() : FunctionPass(ID) {}
  StringRef getPassName() const override { return "BPF Arena Cast"; }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }
  bool runOnFunction(Function &F) override { return ArenaCast(F).run(); }
};

} // namespace

char BPFArenaCastLegacy::ID = 0;

INITIALIZE_PASS(BPFArenaCastLegacy, DEBUG_TYPE, "BPF Arena Cast", false, false)

FunctionPass *llvm::createBPFArenaCastPass() {
  return new BPFArenaCastLegacy();
}

PreservedAnalyses BPFArenaCastPass::run(Function &F,
                                        FunctionAnalysisManager &AM) {
  if (!ArenaCast(F).run())
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
