//===-- NVPTXFavorNonGenericAddrSpace.cpp - ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// FIXME: This pass is deprecated in favor of NVPTXInferAddressSpaces, which
// uses a new algorithm that handles pointer induction variables.
//
// When a load/store accesses the generic address space, checks whether the
// address is casted from a non-generic address space. If so, remove this
// addrspacecast because accessing non-generic address spaces is typically
// faster. Besides removing addrspacecasts directly used by loads/stores, this
// optimization also recursively traces into a GEP's pointer operand and a
// bitcast's source to find more eliminable addrspacecasts.
//
// For instance, the code below loads a float from an array allocated in
// addrspace(3).
//
//   %0 = addrspacecast [10 x float] addrspace(3)* @a to [10 x float]*
//   %1 = gep [10 x float]* %0, i64 0, i64 %i
//   %2 = bitcast float* %1 to i32*
//   %3 = load i32* %2 ; emits ld.u32
//
// First, function hoistAddrSpaceCastFrom reorders the addrspacecast, the GEP,
// and the bitcast to expose more optimization opportunities to function
// optimizeMemoryInst. The intermediate code looks like:
//
//   %0 = gep [10 x float] addrspace(3)* @a, i64 0, i64 %i
//   %1 = bitcast float addrspace(3)* %0 to i32 addrspace(3)*
//   %2 = addrspacecast i32 addrspace(3)* %1 to i32*
//   %3 = load i32* %2 ; still emits ld.u32, but will be optimized shortly
//
// Then, function optimizeMemoryInstruction detects a load from addrspacecast'ed
// generic pointers, and folds the load and the addrspacecast into a load from
// the original address space. The final code looks like:
//
//   %0 = gep [10 x float] addrspace(3)* @a, i64 0, i64 %i
//   %1 = bitcast float addrspace(3)* %0 to i32 addrspace(3)*
//   %3 = load i32 addrspace(3)* %1 ; emits ld.shared.f32
//
// This pass may remove an addrspacecast in a different BB. Therefore, we
// implement it as a FunctionPass.
//
// TODO:
// The current implementation doesn't handle PHINodes. Eliminating
// addrspacecasts used by PHINodes is trickier because PHINodes can introduce
// loops in data flow. For example,
//
//     %generic.input = addrspacecast float addrspace(3)* %input to float*
//   loop:
//     %y = phi [ %generic.input, %y2 ]
//     %y2 = getelementptr %y, 1
//     %v = load %y2
//     br ..., label %loop, ...
//
// Marking %y2 shared depends on marking %y shared, but %y also data-flow
// depends on %y2. We probably need an iterative fix-point algorithm on handle
// this case.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

// An option to disable this optimization. Enable it by default.
static cl::opt<bool> DisableFavorNonGeneric(
  "disable-nvptx-favor-non-generic",
  cl::init(false),
  cl::desc("Do not convert generic address space usage "
           "to non-generic address space usage"),
  cl::Hidden);

namespace {
/// \brief NVPTXFavorNonGenericAddrSpaces
class NVPTXFavorNonGenericAddrSpaces : public FunctionPass {
public:
  static char ID;
  NVPTXFavorNonGenericAddrSpaces() : FunctionPass(ID) {}
  bool runOnFunction(Function &F) override;

private:
  /// Optimizes load/store instructions. Idx is the index of the pointer operand
  /// (0 for load, and 1 for store). Returns true if it changes anything.
  bool optimizeMemoryInstruction(Instruction *I, unsigned Idx);
  /// Recursively traces into a GEP's pointer operand or a bitcast's source to
  /// find an eliminable addrspacecast, and hoists that addrspacecast to the
  /// outermost level. For example, this function transforms
  ///   bitcast(gep(gep(addrspacecast(X))))
  /// to
  ///   addrspacecast(bitcast(gep(gep(X)))).
  ///
  /// This reordering exposes to optimizeMemoryInstruction more
  /// optimization opportunities on loads and stores.
  ///
  /// If this function successfully hoists an eliminable addrspacecast or V is
  /// already such an addrspacecast, it returns the transformed value (which is
  /// guaranteed to be an addrspacecast); otherwise, it returns nullptr.
  Value *hoistAddrSpaceCastFrom(Value *V, int Depth = 0);
  /// Helper function for GEPs.
  Value *hoistAddrSpaceCastFromGEP(GEPOperator *GEP, int Depth);
  /// Helper function for bitcasts.
  Value *hoistAddrSpaceCastFromBitCast(BitCastOperator *BC, int Depth);
};
}

char NVPTXFavorNonGenericAddrSpaces::ID = 0;

namespace llvm {
void initializeNVPTXFavorNonGenericAddrSpacesPass(PassRegistry &);
}
INITIALIZE_PASS(NVPTXFavorNonGenericAddrSpaces, "nvptx-favor-non-generic",
                "Remove unnecessary non-generic-to-generic addrspacecasts",
                false, false)

// Decides whether V is an addrspacecast and shortcutting V in load/store is
// valid and beneficial.
static bool isEliminableAddrSpaceCast(Value *V) {
  // Returns false if V is not even an addrspacecast.
  Operator *Cast = dyn_cast<Operator>(V);
  if (Cast == nullptr || Cast->getOpcode() != Instruction::AddrSpaceCast)
    return false;

  Value *Src = Cast->getOperand(0);
  PointerType *SrcTy = cast<PointerType>(Src->getType());
  PointerType *DestTy = cast<PointerType>(Cast->getType());
  // TODO: For now, we only handle the case where the addrspacecast only changes
  // the address space but not the type. If the type also changes, we could
  // still get rid of the addrspacecast by adding an extra bitcast, but we
  // rarely see such scenarios.
  if (SrcTy->getElementType() != DestTy->getElementType())
    return false;

  // Checks whether the addrspacecast is from a non-generic address space to the
  // generic address space.
  return (SrcTy->getAddressSpace() != AddressSpace::ADDRESS_SPACE_GENERIC &&
          DestTy->getAddressSpace() == AddressSpace::ADDRESS_SPACE_GENERIC);
}

Value *NVPTXFavorNonGenericAddrSpaces::hoistAddrSpaceCastFromGEP(
    GEPOperator *GEP, int Depth) {
  Value *NewOperand =
      hoistAddrSpaceCastFrom(GEP->getPointerOperand(), Depth + 1);
  if (NewOperand == nullptr)
    return nullptr;

  // hoistAddrSpaceCastFrom returns an eliminable addrspacecast or nullptr.
  assert(isEliminableAddrSpaceCast(NewOperand));
  Operator *Cast = cast<Operator>(NewOperand);

  SmallVector<Value *, 8> Indices(GEP->idx_begin(), GEP->idx_end());
  Value *NewASC;
  if (Instruction *GEPI = dyn_cast<Instruction>(GEP)) {
    // GEP = gep (addrspacecast X), indices
    // =>
    // NewGEP = gep X, indices
    // NewASC = addrspacecast NewGEP
    GetElementPtrInst *NewGEP = GetElementPtrInst::Create(
        GEP->getSourceElementType(), Cast->getOperand(0), Indices,
        "", GEPI);
    NewGEP->setIsInBounds(GEP->isInBounds());
    NewGEP->takeName(GEP);
    NewASC = new AddrSpaceCastInst(NewGEP, GEP->getType(), "", GEPI);
    // Without RAUWing GEP, the compiler would visit GEP again and emit
    // redundant instructions. This is exercised in test @rauw in
    // access-non-generic.ll.
    GEP->replaceAllUsesWith(NewASC);
  } else {
    // GEP is a constant expression.
    Constant *NewGEP = ConstantExpr::getGetElementPtr(
        GEP->getSourceElementType(), cast<Constant>(Cast->getOperand(0)),
        Indices, GEP->isInBounds());
    NewASC = ConstantExpr::getAddrSpaceCast(NewGEP, GEP->getType());
  }
  return NewASC;
}

Value *NVPTXFavorNonGenericAddrSpaces::hoistAddrSpaceCastFromBitCast(
    BitCastOperator *BC, int Depth) {
  Value *NewOperand = hoistAddrSpaceCastFrom(BC->getOperand(0), Depth + 1);
  if (NewOperand == nullptr)
    return nullptr;

  // hoistAddrSpaceCastFrom returns an eliminable addrspacecast or nullptr.
  assert(isEliminableAddrSpaceCast(NewOperand));
  Operator *Cast = cast<Operator>(NewOperand);

  // Cast  = addrspacecast Src
  // BC    = bitcast Cast
  //   =>
  // Cast' = bitcast Src
  // BC'   = addrspacecast Cast'
  Value *Src = Cast->getOperand(0);
  Type *TypeOfNewCast =
      PointerType::get(BC->getType()->getPointerElementType(),
                       Src->getType()->getPointerAddressSpace());
  Value *NewBC;
  if (BitCastInst *BCI = dyn_cast<BitCastInst>(BC)) {
    Value *NewCast = new BitCastInst(Src, TypeOfNewCast, "", BCI);
    NewBC = new AddrSpaceCastInst(NewCast, BC->getType(), "", BCI);
    NewBC->takeName(BC);
    // Without RAUWing BC, the compiler would visit BC again and emit
    // redundant instructions. This is exercised in test @rauw in
    // access-non-generic.ll.
    BC->replaceAllUsesWith(NewBC);
  } else {
    // BC is a constant expression.
    Constant *NewCast =
        ConstantExpr::getBitCast(cast<Constant>(Src), TypeOfNewCast);
    NewBC = ConstantExpr::getAddrSpaceCast(NewCast, BC->getType());
  }
  return NewBC;
}

Value *NVPTXFavorNonGenericAddrSpaces::hoistAddrSpaceCastFrom(Value *V,
                                                              int Depth) {
  // Returns V if V is already an eliminable addrspacecast.
  if (isEliminableAddrSpaceCast(V))
    return V;

  // Limit the depth to prevent this recursive function from running too long.
  const int MaxDepth = 20;
  if (Depth >= MaxDepth)
    return nullptr;

  // If V is a GEP or bitcast, hoist the addrspacecast if any from its pointer
  // operand. This enables optimizeMemoryInstruction to shortcut addrspacecasts
  // that are not directly used by the load/store.
  if (GEPOperator *GEP = dyn_cast<GEPOperator>(V))
    return hoistAddrSpaceCastFromGEP(GEP, Depth);

  if (BitCastOperator *BC = dyn_cast<BitCastOperator>(V))
    return hoistAddrSpaceCastFromBitCast(BC, Depth);

  return nullptr;
}

bool NVPTXFavorNonGenericAddrSpaces::optimizeMemoryInstruction(Instruction *MI,
                                                               unsigned Idx) {
  Value *NewOperand = hoistAddrSpaceCastFrom(MI->getOperand(Idx));
  if (NewOperand == nullptr)
    return false;

  // load/store (addrspacecast X) => load/store X if shortcutting the
  // addrspacecast is valid and can improve performance.
  //
  // e.g.,
  // %1 = addrspacecast float addrspace(3)* %0 to float*
  // %2 = load float* %1
  // ->
  // %2 = load float addrspace(3)* %0
  //
  // Note: the addrspacecast can also be a constant expression.
  assert(isEliminableAddrSpaceCast(NewOperand));
  Operator *ASC = dyn_cast<Operator>(NewOperand);
  MI->setOperand(Idx, ASC->getOperand(0));
  return true;
}

bool NVPTXFavorNonGenericAddrSpaces::runOnFunction(Function &F) {
  if (DisableFavorNonGeneric || skipFunction(F))
    return false;

  bool Changed = false;
  for (BasicBlock &B : F) {
    for (Instruction &I : B) {
      if (isa<LoadInst>(I)) {
        // V = load P
        Changed |= optimizeMemoryInstruction(&I, 0);
      } else if (isa<StoreInst>(I)) {
        // store V, P
        Changed |= optimizeMemoryInstruction(&I, 1);
      }
    }
  }
  return Changed;
}

FunctionPass *llvm::createNVPTXFavorNonGenericAddrSpacesPass() {
  return new NVPTXFavorNonGenericAddrSpaces();
}
