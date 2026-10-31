//===-- AGXLowerGenericAddresses.cpp - Generic pointers to local memories ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AGX has no generic address space: device, threadgroup and private (stack)
// memory are reached by different instructions, and a 64-bit device address
// cannot name the other two. PTX's generic addressing (cvta.shared /
// cvta.local, then ld / st on the generic pointer) is emulated, in the
// functions that convert such pointers to generic ones, with a tagged
// representation:
//
//   generic pointer to threadgroup offset o:  (1 << 63) | o
//   generic pointer to private offset o:      (1 << 62) | o
//   generic pointer to device address a:      a          (device addresses are
//                                                          below 2^48)
//
// The casts become integer operations, and every load, store and atomic
// through a generic pointer that InferAddressSpaces could not resolve tests
// the tags and branches to a threadgroup, private or device access. Functions
// without such casts are unchanged: their generic pointers are device
// pointers. A private pointer is only meaningful in the thread that made it.
//
// With real calls (-agx-experimental-leaf-calls) a called function can receive
// tagged pointers from its callers (Blender Cycles' BVH traversal writes the
// kernel's private Intersection through one), so called functions test the
// tags of every kind the module makes, casts of their own or not.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "Utils/AGXBaseInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "agx-lower-generic-addresses"

namespace {

class AGXLowerGenericAddresses : public FunctionPass {
public:
  static char ID;
  AGXLowerGenericAddresses() : FunctionPass(ID) {}

  StringRef getPassName() const override {
    return "AGX Lower Generic Addresses";
  }
  bool doInitialization(Module &M) override;
  bool runOnFunction(Function &F) override;

private:
  // Whether any function of the module makes generic pointers to threadgroup
  // or private memory, scanned at the first function (doInitialization runs
  // before AGXLowerAIR has given the module AGX's address spaces).
  bool Scanned = false, ModuleHasShared = false, ModuleHasPrivate = false;
  void scanModule(const Module &M);
};

} // namespace

char AGXLowerGenericAddresses::ID = 0;

INITIALIZE_PASS(AGXLowerGenericAddresses, DEBUG_TYPE,
                "AGX Lower Generic Addresses", false, false)

FunctionPass *llvm::createAGXLowerGenericAddressesPass() {
  return new AGXLowerGenericAddresses();
}

static constexpr unsigned Generic = AGX::GENERIC_ADDRESS;

/// The tag of generic pointers to \p AS, or 0 if they are not tagged.
static uint64_t tagOf(unsigned AS) {
  if (AS == AGX::THREADGROUP_ADDRESS)
    return uint64_t(1) << 63;
  if (AS == AGX::PRIVATE_ADDRESS)
    return uint64_t(1) << 62;
  return 0;
}

/// Address-space inference can put casts in constants, including comparison
/// operands, vectors and phi incoming values. Give these the same lowering as
/// instruction casts, rather than leaving an unselectable addrspacecast DAG.
static bool expandConstantLocalCasts(Function &F) {
  SmallVector<Constant *, 16> Work;
  SmallPtrSet<Constant *, 16> Seen;
  SmallVector<Constant *, 8> Casts;
  auto Push = [&](Value *V) {
    if (auto *C = dyn_cast<Constant>(V);
        C && (isa<ConstantExpr>(C) || isa<ConstantAggregate>(C)) &&
        Seen.insert(C).second)
      Work.push_back(C);
  };
  for (Instruction &I : instructions(F))
    for (Value *V : I.operands())
      Push(V);
  while (!Work.empty()) {
    Constant *C = Work.pop_back_val();
    if (auto *CE = dyn_cast<ConstantExpr>(C);
        CE && CE->getOpcode() == Instruction::AddrSpaceCast &&
        CE->getType()->isPointerTy()) {
      unsigned Src = CE->getOperand(0)->getType()->getPointerAddressSpace();
      unsigned Dst = CE->getType()->getPointerAddressSpace();
      if ((Dst == Generic && tagOf(Src)) || (Src == Generic && tagOf(Dst)))
        Casts.push_back(CE);
    }
    for (Value *V : C->operands())
      Push(V);
  }
  // Selected expressions may use each other. Keep the constant DAG alive
  // while the helper walks the roots; normal constant cleanup can reclaim it.
  return convertUsersOfConstantsToInstructions(
      Casts, &F, /*RemoveDeadConstants=*/false, /*IncludeSelf=*/true);
}

/// The pointer operand of a generic memory access, or null.
static Value *genericPointer(Instruction &I) {
  Value *P = nullptr;
  if (auto *L = dyn_cast<LoadInst>(&I))
    P = L->getPointerOperand();
  else if (auto *S = dyn_cast<StoreInst>(&I))
    P = S->getPointerOperand();
  else if (auto *A = dyn_cast<AtomicRMWInst>(&I))
    P = A->getPointerOperand();
  else if (auto *X = dyn_cast<AtomicCmpXchgInst>(&I))
    P = X->getPointerOperand();
  return P && P->getType()->getPointerAddressSpace() == Generic ? P : nullptr;
}

static void setPointer(Instruction &I, Value *P) {
  if (auto *L = dyn_cast<LoadInst>(&I))
    L->setOperand(LoadInst::getPointerOperandIndex(), P);
  else if (auto *S = dyn_cast<StoreInst>(&I))
    S->setOperand(StoreInst::getPointerOperandIndex(), P);
  else if (auto *A = dyn_cast<AtomicRMWInst>(&I))
    A->setOperand(AtomicRMWInst::getPointerOperandIndex(), P);
  else
    cast<AtomicCmpXchgInst>(I).setOperand(
        AtomicCmpXchgInst::getPointerOperandIndex(), P);
}

/// Notes in \p HasShared / \p HasPrivate the tagged spaces that \p V, a cast
/// instruction or constant expression, makes generic pointers to.
static void noteToGeneric(const Value *V, bool &HasShared, bool &HasPrivate) {
  const auto *Op = dyn_cast<Operator>(V);
  if (!Op || Op->getOpcode() != Instruction::AddrSpaceCast ||
      !Op->getType()->isPointerTy() ||
      Op->getType()->getPointerAddressSpace() != Generic)
    return;
  unsigned Src = Op->getOperand(0)->getType()->getPointerAddressSpace();
  if (Src == AGX::THREADGROUP_ADDRESS)
    HasShared = true;
  else if (Src == AGX::PRIVATE_ADDRESS)
    HasPrivate = true;
}

bool AGXLowerGenericAddresses::doInitialization(Module &M) {
  Scanned = ModuleHasShared = ModuleHasPrivate = false;
  return false;
}

void AGXLowerGenericAddresses::scanModule(const Module &M) {
  Scanned = true;
  SmallVector<const Value *, 16> Work;
  SmallPtrSet<const Value *, 16> Seen;
  for (const Function &F : M)
    for (const Instruction &I : instructions(F)) {
      noteToGeneric(&I, ModuleHasShared, ModuleHasPrivate);
      for (const Value *V : I.operands())
        if (isa<ConstantExpr>(V) && Seen.insert(V).second)
          Work.push_back(V);
    }
  while (!Work.empty()) {
    const Value *V = Work.pop_back_val();
    noteToGeneric(V, ModuleHasShared, ModuleHasPrivate);
    for (const Value *O : cast<User>(V)->operands())
      if (isa<ConstantExpr>(O) && Seen.insert(O).second)
        Work.push_back(O);
  }
}

bool AGXLowerGenericAddresses::runOnFunction(Function &F) {
  // (at the first function, before any function's casts are lowered)
  if (agxExperimentalLeafCallsEnabled() && !Scanned)
    scanModule(*F.getParent());
  bool Called = agxExperimentalLeafCallsEnabled() &&
                F.getCallingConv() != CallingConv::AGX_KERNEL;

  // Constant casts of threadgroup globals become instructions first.
  SmallVector<Constant *, 4> Consts;
  for (GlobalVariable &G : F.getParent()->globals())
    if (G.getAddressSpace() == AGX::THREADGROUP_ADDRESS)
      Consts.push_back(&G);
  bool Changed = convertUsersOfConstantsToInstructions(Consts, &F);
  Changed |= expandConstantLocalCasts(F);

  SmallVector<AddrSpaceCastInst *, 8> ToGeneric, FromGeneric;
  bool HasShared = false, HasPrivate = false;
  for (Instruction &I : instructions(F))
    if (auto *C = dyn_cast<AddrSpaceCastInst>(&I)) {
      unsigned Src = C->getSrcAddressSpace(), Dst = C->getDestAddressSpace();
      if (Dst == Generic && tagOf(Src)) {
        ToGeneric.push_back(C);
        (Src == AGX::THREADGROUP_ADDRESS ? HasShared : HasPrivate) = true;
      } else if (Src == Generic && tagOf(Dst)) {
        FromGeneric.push_back(C);
      }
    }
  // A called function may be given the module's tagged pointers.
  if (Called) {
    HasShared |= ModuleHasShared;
    HasPrivate |= ModuleHasPrivate;
  }
  if (ToGeneric.empty() && FromGeneric.empty() && !HasShared && !HasPrivate)
    return Changed;

  LLVMContext &Ctx = F.getContext();
  Type *I64 = Type::getInt64Ty(Ctx), *I32 = Type::getInt32Ty(Ctx);
  PointerType *GenericPtr = PointerType::get(Ctx, Generic);
  PointerType *DevicePtr = PointerType::get(Ctx, AGX::DEVICE_ADDRESS);

  for (AddrSpaceCastInst *C : ToGeneric) {
    IRBuilder<> B(C);
    Value *Off = B.CreateZExt(B.CreatePtrToInt(C->getOperand(0), I32), I64);
    Value *G = B.CreateIntToPtr(B.CreateOr(Off, tagOf(C->getSrcAddressSpace())),
                                GenericPtr);
    C->replaceAllUsesWith(G);
    C->eraseFromParent();
  }
  for (AddrSpaceCastInst *C : FromGeneric) {
    IRBuilder<> B(C);
    Value *Off = B.CreateTrunc(B.CreatePtrToInt(C->getOperand(0), I64), I32);
    C->replaceAllUsesWith(B.CreateIntToPtr(Off, C->getDestTy()));
    C->eraseFromParent();
  }

  // The memories a generic access may reach in this function, the tagged ones
  // first; the last one is device memory.
  SmallVector<unsigned, 3> AllSpaces;
  if (HasShared)
    AllSpaces.push_back(AGX::THREADGROUP_ADDRESS);
  if (HasPrivate)
    AllSpaces.push_back(AGX::PRIVATE_ADDRESS);
  AllSpaces.push_back(AGX::DEVICE_ADDRESS);

  SmallVector<Instruction *, 16> Accesses;
  for (Instruction &I : instructions(F))
    if (genericPointer(I))
      Accesses.push_back(&I);
  for (Instruction *I : Accesses) {
    Value *P = genericPointer(*I);
    // PTX leaves atomics on generic addresses of local memory undefined.
    SmallVector<unsigned, 3> Spaces(AllSpaces);
    if (isa<AtomicRMWInst, AtomicCmpXchgInst>(I))
      llvm::erase(Spaces, AGX::PRIVATE_ADDRESS);
    BasicBlock *Head = I->getParent();
    BasicBlock *Tail = Head->splitBasicBlock(I, "generic.join");
    Head->getTerminator()->eraseFromParent();
    IRBuilder<> B(Head);
    Value *Bits = B.CreatePtrToInt(P, I64);
    // The tags are the top two bits of the high word, cheaper to test than
    // the 64-bit value: 2 or 3 is threadgroup, 1 private, 0 device.
    Value *Top = B.CreateLShr(B.CreateTrunc(B.CreateLShr(Bits, 32), I32), 30);
    PHINode *Phi = nullptr;
    if (!I->getType()->isVoidTy()) {
      Phi = PHINode::Create(I->getType(), Spaces.size(), "", Tail->begin());
      Phi->takeName(I);
    }
    BasicBlock *Test = Head;
    for (unsigned N = 0; N < Spaces.size(); ++N) {
      unsigned AS = Spaces[N];
      BasicBlock *BB = BasicBlock::Create(Ctx, "generic.access", &F, Tail);
      IRBuilder<> BA(BB);
      Instruction *A = I->clone();
      Value *Ptr = AS == AGX::DEVICE_ADDRESS
                       ? BA.CreateIntToPtr(Bits, DevicePtr)
                       : BA.CreateIntToPtr(BA.CreateTrunc(Bits, I32),
                                           PointerType::get(Ctx, AS));
      setPointer(*A, Ptr);
      BA.Insert(A);
      BA.CreateBr(Tail);
      if (Phi)
        Phi->addIncoming(A, BB);
      IRBuilder<> BT(Test);
      if (N + 1 == Spaces.size()) {
        BT.CreateBr(BB);
        break;
      }
      Value *Is = AS == AGX::THREADGROUP_ADDRESS
                      ? BT.CreateICmpUGE(Top, ConstantInt::get(I32, 2))
                      : BT.CreateICmpEQ(Top, ConstantInt::get(I32, 1));
      BasicBlock *Else = BasicBlock::Create(Ctx, "generic.test", &F, Tail);
      BT.CreateCondBr(Is, BB, Else);
      Test = Else;
    }
    if (Phi)
      I->replaceAllUsesWith(Phi);
    I->eraseFromParent();
  }
  return true;
}
