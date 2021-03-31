//===----- SVEIntrinsicOpts - SVE ACLE Intrinsics Opts --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Performs general IR level optimizations on SVE intrinsics.
//
// This pass performs the following optimizations:
//
// - removes unnecessary reinterpret intrinsics
//   (llvm.aarch64.sve.convert.[to|from].svbool), e.g:
//     %1 = @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %a)
//     %2 = @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %1)
//
// - removes unnecessary ptrue intrinsics (llvm.aarch64.sve.ptrue), e.g:
//     %1 = @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
//     %2 = @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
//     ; (%1 can be replaced with a reinterpret of %2)
//
// - optimizes ptest intrinsics and phi instructions where the operands are
//   being needlessly converted to and from svbool_t.
//
//===----------------------------------------------------------------------===//

#include "Utils/AArch64BaseInfo.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "aarch64-sve-intrinsic-opts"

namespace llvm {
void initializeSVEIntrinsicOptsPass(PassRegistry &);
}

namespace {
struct SVEIntrinsicOpts : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  SVEIntrinsicOpts() : ModulePass(ID) {
    initializeSVEIntrinsicOptsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  static IntrinsicInst *isReinterpretToSVBool(Value *V);

  bool coalescePTrueIntrinsicCalls(BasicBlock &BB,
                                   SmallSetVector<IntrinsicInst *, 4> &PTrues);
  bool optimizePTrueIntrinsicCalls(SmallSetVector<Function *, 4> &Functions);

  /// Operates at the instruction-scope. I.e., optimizations are applied local
  /// to individual instructions.
  static bool optimizeIntrinsic(Instruction *I);
  bool optimizeIntrinsicCalls(SmallSetVector<Function *, 4> &Functions);

  /// Operates at the function-scope. I.e., optimizations are applied local to
  /// the functions themselves.
  bool optimizeFunctions(SmallSetVector<Function *, 4> &Functions);

  static bool optimizeConvertFromSVBool(IntrinsicInst *I);
  static bool optimizePTest(IntrinsicInst *I);
  static bool optimizeVectorMul(IntrinsicInst *I);
  static bool optimizeTBL(IntrinsicInst *I);

  static bool processPhiNode(IntrinsicInst *I);
};
} // end anonymous namespace

void SVEIntrinsicOpts::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.setPreservesCFG();
}

char SVEIntrinsicOpts::ID = 0;
static const char *name = "SVE intrinsics optimizations";
INITIALIZE_PASS_BEGIN(SVEIntrinsicOpts, DEBUG_TYPE, name, false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass);
INITIALIZE_PASS_END(SVEIntrinsicOpts, DEBUG_TYPE, name, false, false)

namespace llvm {
ModulePass *createSVEIntrinsicOptsPass() { return new SVEIntrinsicOpts(); }
} // namespace llvm

/// Returns V if it's a cast from <n x 16 x i1> (aka svbool_t), nullptr
/// otherwise.
IntrinsicInst *SVEIntrinsicOpts::isReinterpretToSVBool(Value *V) {
  IntrinsicInst *I = dyn_cast<IntrinsicInst>(V);
  if (!I)
    return nullptr;

  if (I->getIntrinsicID() != Intrinsic::aarch64_sve_convert_to_svbool)
    return nullptr;

  return I;
}

/// Checks if a ptrue intrinsic call is promoted. The act of promoting a
/// ptrue will introduce zeroing. For example:
///
///     %1 = <vscale x 4 x i1> call @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
///     %2 = <vscale x 16 x i1> call @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %1)
///     %3 = <vscale x 8 x i1> call @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %2)
///
/// %1 is promoted, because it is converted:
///
///     <vscale x 4 x i1> => <vscale x 16 x i1> => <vscale x 8 x i1>
///
/// via a sequence of the SVE reinterpret intrinsics convert.{to,from}.svbool.
bool isPTruePromoted(IntrinsicInst *PTrue) {
  // Find all users of this intrinsic that are calls to convert-to-svbool
  // reinterpret intrinsics.
  SmallVector<IntrinsicInst *, 4> ConvertToUses;
  for (User *User : PTrue->users()) {
    if (match(User, m_Intrinsic<Intrinsic::aarch64_sve_convert_to_svbool>())) {
      ConvertToUses.push_back(cast<IntrinsicInst>(User));
    }
  }

  // If no such calls were found, this is ptrue is not promoted.
  if (ConvertToUses.empty())
    return false;

  // Otherwise, try to find users of the convert-to-svbool intrinsics that are
  // calls to the convert-from-svbool intrinsic, and would result in some lanes
  // being zeroed.
  const auto *PTrueVTy = cast<ScalableVectorType>(PTrue->getType());
  for (IntrinsicInst *ConvertToUse : ConvertToUses) {
    for (User *User : ConvertToUse->users()) {
      auto *IntrUser = dyn_cast<IntrinsicInst>(User);
      if (IntrUser && IntrUser->getIntrinsicID() ==
                          Intrinsic::aarch64_sve_convert_from_svbool) {
        const auto *IntrUserVTy = cast<ScalableVectorType>(IntrUser->getType());

        // Would some lanes become zeroed by the conversion?
        if (IntrUserVTy->getElementCount().getKnownMinValue() >
            PTrueVTy->getElementCount().getKnownMinValue())
          // This is a promoted ptrue.
          return true;
      }
    }
  }

  // If no matching calls were found, this is not a promoted ptrue.
  return false;
}

/// Attempts to coalesce ptrues in a basic block.
bool SVEIntrinsicOpts::coalescePTrueIntrinsicCalls(
    BasicBlock &BB, SmallSetVector<IntrinsicInst *, 4> &PTrues) {
  if (PTrues.size() <= 1)
    return false;

  // Find the ptrue with the most lanes.
  auto *MostEncompassingPTrue = *std::max_element(
      PTrues.begin(), PTrues.end(), [](auto *PTrue1, auto *PTrue2) {
        auto *PTrue1VTy = cast<ScalableVectorType>(PTrue1->getType());
        auto *PTrue2VTy = cast<ScalableVectorType>(PTrue2->getType());
        return PTrue1VTy->getElementCount().getKnownMinValue() <
               PTrue2VTy->getElementCount().getKnownMinValue();
      });

  // Remove the most encompassing ptrue, as well as any promoted ptrues, leaving
  // behind only the ptrues to be coalesced.
  PTrues.remove(MostEncompassingPTrue);
  PTrues.remove_if([](auto *PTrue) { return isPTruePromoted(PTrue); });

  // Hoist MostEncompassingPTrue to the start of the basic block. It is always
  // safe to do this, since ptrue intrinsic calls are guaranteed to have no
  // predecessors.
  MostEncompassingPTrue->moveBefore(BB, BB.getFirstInsertionPt());

  LLVMContext &Ctx = BB.getContext();
  IRBuilder<> Builder(Ctx);
  Builder.SetInsertPoint(&BB, ++MostEncompassingPTrue->getIterator());

  auto *MostEncompassingPTrueVTy =
      cast<VectorType>(MostEncompassingPTrue->getType());
  auto *ConvertToSVBool = Builder.CreateIntrinsic(
      Intrinsic::aarch64_sve_convert_to_svbool, {MostEncompassingPTrueVTy},
      {MostEncompassingPTrue});

  for (auto *PTrue : PTrues) {
    auto *PTrueVTy = cast<VectorType>(PTrue->getType());

    Builder.SetInsertPoint(&BB, ++ConvertToSVBool->getIterator());
    auto *ConvertFromSVBool =
        Builder.CreateIntrinsic(Intrinsic::aarch64_sve_convert_from_svbool,
                                {PTrueVTy}, {ConvertToSVBool});
    PTrue->replaceAllUsesWith(ConvertFromSVBool);
    PTrue->eraseFromParent();
  }

  return true;
}

/// The goal of this function is to remove redundant calls to the SVE ptrue
/// intrinsic in each basic block within the given functions.
///
/// SVE ptrues have two representations in LLVM IR:
/// - a logical representation -- an arbitrary-width scalable vector of i1s,
///   i.e. <vscale x N x i1>.
/// - a physical representation (svbool, <vscale x 16 x i1>) -- a 16-element
///   scalable vector of i1s, i.e. <vscale x 16 x i1>.
///
/// The SVE ptrue intrinsic is used to create a logical representation of an SVE
/// predicate. Suppose that we have two SVE ptrue intrinsic calls: P1 and P2. If
/// P1 creates a logical SVE predicate that is at least as wide as the logical
/// SVE predicate created by P2, then all of the bits that are true in the
/// physical representation of P2 are necessarily also true in the physical
/// representation of P1. P1 'encompasses' P2, therefore, the intrinsic call to
/// P2 is redundant and can be replaced by an SVE reinterpret of P1 via
/// convert.{to,from}.svbool.
///
/// Currently, this pass only coalesces calls to SVE ptrue intrinsics
/// if they match the following conditions:
///
/// - the call to the intrinsic uses either the SV_ALL or SV_POW2 patterns.
///   SV_ALL indicates that all bits of the predicate vector are to be set to
///   true. SV_POW2 indicates that all bits of the predicate vector up to the
///   largest power-of-two are to be set to true.
/// - the result of the call to the intrinsic is not promoted to a wider
///   predicate. In this case, keeping the extra ptrue leads to better codegen
///   -- coalescing here would create an irreducible chain of SVE reinterprets
///   via convert.{to,from}.svbool.
///
/// EXAMPLE:
///
///     %1 = <vscale x 8 x i1> ptrue(i32 SV_ALL)
///     ; Logical:  <1, 1, 1, 1, 1, 1, 1, 1>
///     ; Physical: <1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0>
///     ...
///
///     %2 = <vscale x 4 x i1> ptrue(i32 SV_ALL)
///     ; Logical:  <1, 1, 1, 1>
///     ; Physical: <1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0>
///     ...
///
/// Here, %2 can be replaced by an SVE reinterpret of %1, giving, for instance:
///
///     %1 = <vscale x 8 x i1> ptrue(i32 i31)
///     %2 = <vscale x 16 x i1> convert.to.svbool(<vscale x 8 x i1> %1)
///     %3 = <vscale x 4 x i1> convert.from.svbool(<vscale x 16 x i1> %2)
///
bool SVEIntrinsicOpts::optimizePTrueIntrinsicCalls(
    SmallSetVector<Function *, 4> &Functions) {
  bool Changed = false;

  for (auto *F : Functions) {
    for (auto &BB : *F) {
      SmallSetVector<IntrinsicInst *, 4> SVAllPTrues;
      SmallSetVector<IntrinsicInst *, 4> SVPow2PTrues;

      // For each basic block, collect the used ptrues and try to coalesce them.
      for (Instruction &I : BB) {
        if (I.use_empty())
          continue;

        auto *IntrI = dyn_cast<IntrinsicInst>(&I);
        if (!IntrI || IntrI->getIntrinsicID() != Intrinsic::aarch64_sve_ptrue)
          continue;

        const auto PTruePattern =
            cast<ConstantInt>(IntrI->getOperand(0))->getZExtValue();

        if (PTruePattern == AArch64SVEPredPattern::all)
          SVAllPTrues.insert(IntrI);
        if (PTruePattern == AArch64SVEPredPattern::pow2)
          SVPow2PTrues.insert(IntrI);
      }

      Changed |= coalescePTrueIntrinsicCalls(BB, SVAllPTrues);
      Changed |= coalescePTrueIntrinsicCalls(BB, SVPow2PTrues);
    }
  }

  return Changed;
}

/// The function will remove redundant reinterprets casting in the presence
/// of the control flow
bool SVEIntrinsicOpts::processPhiNode(IntrinsicInst *X) {

  SmallVector<Instruction *, 32> Worklist;
  auto RequiredType = X->getType();

  auto *PN = dyn_cast<PHINode>(X->getArgOperand(0));
  assert(PN && "Expected Phi Node!");

  // Don't create a new Phi unless we can remove the old one.
  if (!PN->hasOneUse())
    return false;

  for (Value *IncValPhi : PN->incoming_values()) {
    auto *Reinterpret = isReinterpretToSVBool(IncValPhi);
    if (!Reinterpret ||
        RequiredType != Reinterpret->getArgOperand(0)->getType())
      return false;
  }

  // Create the new Phi
  LLVMContext &Ctx = PN->getContext();
  IRBuilder<> Builder(Ctx);
  Builder.SetInsertPoint(PN);
  PHINode *NPN = Builder.CreatePHI(RequiredType, PN->getNumIncomingValues());
  Worklist.push_back(PN);

  for (unsigned I = 0; I < PN->getNumIncomingValues(); I++) {
    auto *Reinterpret = cast<Instruction>(PN->getIncomingValue(I));
    NPN->addIncoming(Reinterpret->getOperand(0), PN->getIncomingBlock(I));
    Worklist.push_back(Reinterpret);
  }

  // Cleanup Phi Node and reinterprets
  X->replaceAllUsesWith(NPN);
  X->eraseFromParent();

  for (auto &I : Worklist)
    if (I->use_empty())
      I->eraseFromParent();

  return true;
}

bool SVEIntrinsicOpts::optimizePTest(IntrinsicInst *I) {
  IntrinsicInst *Op1 = dyn_cast<IntrinsicInst>(I->getArgOperand(0));
  IntrinsicInst *Op2 = dyn_cast<IntrinsicInst>(I->getArgOperand(1));

  if (Op1 && Op2 &&
      Op1->getIntrinsicID() == Intrinsic::aarch64_sve_convert_to_svbool &&
      Op2->getIntrinsicID() == Intrinsic::aarch64_sve_convert_to_svbool &&
      Op1->getArgOperand(0)->getType() == Op2->getArgOperand(0)->getType()) {

    Value *Ops[] = {Op1->getArgOperand(0), Op2->getArgOperand(0)};
    Type *Tys[] = {Op1->getArgOperand(0)->getType()};
    Module *M = I->getParent()->getParent()->getParent();

    auto Fn = Intrinsic::getDeclaration(M, I->getIntrinsicID(), Tys);
    auto CI = CallInst::Create(Fn, Ops, I->getName(), I);

    I->replaceAllUsesWith(CI);
    I->eraseFromParent();
    if (Op1->use_empty())
      Op1->eraseFromParent();
    if (Op1 != Op2 && Op2->use_empty())
      Op2->eraseFromParent();

    return true;
  }

  return false;
}

bool SVEIntrinsicOpts::optimizeVectorMul(IntrinsicInst *I) {
  assert((I->getIntrinsicID() == Intrinsic::aarch64_sve_mul ||
          I->getIntrinsicID() == Intrinsic::aarch64_sve_fmul) &&
         "Unexpected opcode");

  auto *OpPredicate = I->getOperand(0);
  auto *OpMultiplicand = I->getOperand(1);
  auto *OpMultiplier = I->getOperand(2);

  // Return true if a given instruction is an aarch64_sve_dup_x intrinsic call
  // with a unit splat value, false otherwise.
  auto IsUnitDupX = [](auto *I) {
    auto *IntrI = dyn_cast<IntrinsicInst>(I);
    if (!IntrI || IntrI->getIntrinsicID() != Intrinsic::aarch64_sve_dup_x)
      return false;

    auto *SplatValue = IntrI->getOperand(0);
    return match(SplatValue, m_FPOne()) || match(SplatValue, m_One());
  };

  // Return true if a given instruction is an aarch64_sve_dup intrinsic call
  // with a unit splat value, false otherwise.
  auto IsUnitDup = [](auto *I) {
    auto *IntrI = dyn_cast<IntrinsicInst>(I);
    if (!IntrI || IntrI->getIntrinsicID() != Intrinsic::aarch64_sve_dup)
      return false;

    auto *SplatValue = IntrI->getOperand(2);
    return match(SplatValue, m_FPOne()) || match(SplatValue, m_One());
  };

  bool Changed = true;

  // The OpMultiplier variable should always point to the dup (if any), so
  // swap if necessary.
  if (IsUnitDup(OpMultiplicand) || IsUnitDupX(OpMultiplicand))
    std::swap(OpMultiplier, OpMultiplicand);

  if (IsUnitDupX(OpMultiplier)) {
    // [f]mul pg (dupx 1) %n => %n
    I->replaceAllUsesWith(OpMultiplicand);
    I->eraseFromParent();
    Changed = true;
  } else if (IsUnitDup(OpMultiplier)) {
    // [f]mul pg (dup pg 1) %n => %n
    auto *DupInst = cast<IntrinsicInst>(OpMultiplier);
    auto *DupPg = DupInst->getOperand(1);
    // TODO: this is naive. The optimization is still valid if DupPg
    // 'encompasses' OpPredicate, not only if they're the same predicate.
    if (OpPredicate == DupPg) {
      I->replaceAllUsesWith(OpMultiplicand);
      I->eraseFromParent();
      Changed = true;
    }
  }

  // If an instruction was optimized out then it is possible that some dangling
  // instructions are left.
  if (Changed) {
    auto *OpPredicateInst = dyn_cast<Instruction>(OpPredicate);
    auto *OpMultiplierInst = dyn_cast<Instruction>(OpMultiplier);
    if (OpMultiplierInst && OpMultiplierInst->use_empty())
      OpMultiplierInst->eraseFromParent();
    if (OpPredicateInst && OpPredicateInst->use_empty())
      OpPredicateInst->eraseFromParent();
  }

  return Changed;
}

bool SVEIntrinsicOpts::optimizeTBL(IntrinsicInst *I) {
  assert(I->getIntrinsicID() == Intrinsic::aarch64_sve_tbl &&
         "Unexpected opcode");

  auto *OpVal = I->getOperand(0);
  auto *OpIndices = I->getOperand(1);
  VectorType *VTy = cast<VectorType>(I->getType());

  // Check whether OpIndices is an aarch64_sve_dup_x intrinsic call with
  // constant splat value < minimal element count of result.
  auto *DupXIntrI = dyn_cast<IntrinsicInst>(OpIndices);
  if (!DupXIntrI || DupXIntrI->getIntrinsicID() != Intrinsic::aarch64_sve_dup_x)
    return false;

  auto *SplatValue = dyn_cast<ConstantInt>(DupXIntrI->getOperand(0));
  if (!SplatValue ||
      SplatValue->getValue().uge(VTy->getElementCount().getKnownMinValue()))
    return false;

  // Convert sve_tbl(OpVal sve_dup_x(SplatValue)) to
  // splat_vector(extractelement(OpVal, SplatValue)) for further optimization.
  LLVMContext &Ctx = I->getContext();
  IRBuilder<> Builder(Ctx);
  Builder.SetInsertPoint(I);
  auto *Extract = Builder.CreateExtractElement(OpVal, SplatValue);
  auto *VectorSplat =
      Builder.CreateVectorSplat(VTy->getElementCount(), Extract);

  I->replaceAllUsesWith(VectorSplat);
  I->eraseFromParent();
  if (DupXIntrI->use_empty())
    DupXIntrI->eraseFromParent();
  return true;
}

bool SVEIntrinsicOpts::optimizeConvertFromSVBool(IntrinsicInst *I) {
  assert(I->getIntrinsicID() == Intrinsic::aarch64_sve_convert_from_svbool &&
         "Unexpected opcode");

  // If the reinterpret instruction operand is a PHI Node
  if (isa<PHINode>(I->getArgOperand(0)))
    return processPhiNode(I);

  SmallVector<Instruction *, 32> CandidatesForRemoval;
  Value *Cursor = I->getOperand(0), *EarliestReplacement = nullptr;

  const auto *IVTy = cast<VectorType>(I->getType());

  // Walk the chain of conversions.
  while (Cursor) {
    // If the type of the cursor has fewer lanes than the final result, zeroing
    // must take place, which breaks the equivalence chain.
    const auto *CursorVTy = cast<VectorType>(Cursor->getType());
    if (CursorVTy->getElementCount().getKnownMinValue() <
        IVTy->getElementCount().getKnownMinValue())
      break;

    // If the cursor has the same type as I, it is a viable replacement.
    if (Cursor->getType() == IVTy)
      EarliestReplacement = Cursor;

    auto *IntrinsicCursor = dyn_cast<IntrinsicInst>(Cursor);

    // If this is not an SVE conversion intrinsic, this is the end of the chain.
    if (!IntrinsicCursor || !(IntrinsicCursor->getIntrinsicID() ==
                                  Intrinsic::aarch64_sve_convert_to_svbool ||
                              IntrinsicCursor->getIntrinsicID() ==
                                  Intrinsic::aarch64_sve_convert_from_svbool))
      break;

    CandidatesForRemoval.insert(CandidatesForRemoval.begin(), IntrinsicCursor);
    Cursor = IntrinsicCursor->getOperand(0);
  }

  // If no viable replacement in the conversion chain was found, there is
  // nothing to do.
  if (!EarliestReplacement)
    return false;

  I->replaceAllUsesWith(EarliestReplacement);
  I->eraseFromParent();

  while (!CandidatesForRemoval.empty()) {
    Instruction *Candidate = CandidatesForRemoval.pop_back_val();
    if (Candidate->use_empty())
      Candidate->eraseFromParent();
  }
  return true;
}

bool SVEIntrinsicOpts::optimizeIntrinsic(Instruction *I) {
  IntrinsicInst *IntrI = dyn_cast<IntrinsicInst>(I);
  if (!IntrI)
    return false;

  switch (IntrI->getIntrinsicID()) {
  case Intrinsic::aarch64_sve_convert_from_svbool:
    return optimizeConvertFromSVBool(IntrI);
  case Intrinsic::aarch64_sve_fmul:
  case Intrinsic::aarch64_sve_mul:
    return optimizeVectorMul(IntrI);
  case Intrinsic::aarch64_sve_ptest_any:
  case Intrinsic::aarch64_sve_ptest_first:
  case Intrinsic::aarch64_sve_ptest_last:
    return optimizePTest(IntrI);
  case Intrinsic::aarch64_sve_tbl:
    return optimizeTBL(IntrI);
  default:
    return false;
  }

  return true;
}

bool SVEIntrinsicOpts::optimizeIntrinsicCalls(
    SmallSetVector<Function *, 4> &Functions) {
  bool Changed = false;
  for (auto *F : Functions) {
    DominatorTree *DT = &getAnalysis<DominatorTreeWrapperPass>(*F).getDomTree();

    // Traverse the DT with an rpo walk so we see defs before uses, allowing
    // simplification to be done incrementally.
    BasicBlock *Root = DT->getRoot();
    ReversePostOrderTraversal<BasicBlock *> RPOT(Root);
    for (auto *BB : RPOT)
      for (Instruction &I : make_early_inc_range(*BB))
        Changed |= optimizeIntrinsic(&I);
  }
  return Changed;
}

bool SVEIntrinsicOpts::optimizeFunctions(
    SmallSetVector<Function *, 4> &Functions) {
  bool Changed = false;

  Changed |= optimizePTrueIntrinsicCalls(Functions);
  Changed |= optimizeIntrinsicCalls(Functions);

  return Changed;
}

bool SVEIntrinsicOpts::runOnModule(Module &M) {
  bool Changed = false;
  SmallSetVector<Function *, 4> Functions;

  // Check for SVE intrinsic declarations first so that we only iterate over
  // relevant functions. Where an appropriate declaration is found, store the
  // function(s) where it is used so we can target these only.
  for (auto &F : M.getFunctionList()) {
    if (!F.isDeclaration())
      continue;

    switch (F.getIntrinsicID()) {
    case Intrinsic::aarch64_sve_convert_from_svbool:
    case Intrinsic::aarch64_sve_ptest_any:
    case Intrinsic::aarch64_sve_ptest_first:
    case Intrinsic::aarch64_sve_ptest_last:
    case Intrinsic::aarch64_sve_ptrue:
    case Intrinsic::aarch64_sve_mul:
    case Intrinsic::aarch64_sve_fmul:
    case Intrinsic::aarch64_sve_tbl:
      for (User *U : F.users())
        Functions.insert(cast<Instruction>(U)->getFunction());
      break;
    default:
      break;
    }
  }

  if (!Functions.empty())
    Changed |= optimizeFunctions(Functions);

  return Changed;
}
