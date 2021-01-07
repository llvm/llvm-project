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
// The main goal of this pass is to remove unnecessary reinterpret
// intrinsics (llvm.aarch64.sve.convert.[to|from].svbool), e.g:
//
//   %1 = @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 4 x i1> %a)
//   %2 = @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %1)
//
// This pass also looks for ptest intrinsics & phi instructions where the
// operands are being needlessly converted to and from svbool_t.
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

  static bool optimizeIntrinsic(Instruction *I);

  bool optimizeFunctions(SmallSetVector<Function *, 4> &Functions);

  static bool optimizeConvertFromSVBool(IntrinsicInst *I);
  static bool optimizePTest(IntrinsicInst *I);

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

bool SVEIntrinsicOpts::optimizeConvertFromSVBool(IntrinsicInst *I) {
  assert(I->getIntrinsicID() == Intrinsic::aarch64_sve_convert_from_svbool &&
         "Unexpected opcode");

  // If the reinterpret instruction operand is a PHI Node
  if (isa<PHINode>(I->getArgOperand(0)))
    return processPhiNode(I);

  // If we have a reinterpret intrinsic I of type A which is converting from
  // another reinterpret Y of type B, and the source type of Y is A, then we can
  // elide away both reinterprets if there are no other users of Y.
  auto *Y = isReinterpretToSVBool(I->getArgOperand(0));
  if (!Y)
    return false;

  Value *SourceVal = Y->getArgOperand(0);
  if (I->getType() != SourceVal->getType())
    return false;

  I->replaceAllUsesWith(SourceVal);
  I->eraseFromParent();
  if (Y->use_empty())
    Y->eraseFromParent();

  return true;
}

bool SVEIntrinsicOpts::optimizeIntrinsic(Instruction *I) {
  IntrinsicInst *IntrI = dyn_cast<IntrinsicInst>(I);
  if (!IntrI)
    return false;

  switch (IntrI->getIntrinsicID()) {
  case Intrinsic::aarch64_sve_convert_from_svbool:
    return optimizeConvertFromSVBool(IntrI);
  case Intrinsic::aarch64_sve_ptest_any:
  case Intrinsic::aarch64_sve_ptest_first:
  case Intrinsic::aarch64_sve_ptest_last:
    return optimizePTest(IntrI);
  default:
    return false;
  }

  return true;
}

bool SVEIntrinsicOpts::optimizeFunctions(
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
