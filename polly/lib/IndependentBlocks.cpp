//===------ IndependentBlocks.cpp - Create Independent Blocks in Regions --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Create independent blocks in the regions detected by ScopDetection.
//
//===----------------------------------------------------------------------===//
//
#include "polly/LinkAllPasses.h"
#include "polly/CodeGen/BlockGenerators.h"
#include "polly/CodeGen/Cloog.h"
#include "polly/ScopDetection.h"
#include "polly/Support/ScopHelper.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Transforms/Utils/Local.h"

#define DEBUG_TYPE "polly-independent"
#include "llvm/Support/Debug.h"

#include <vector>

using namespace polly;
using namespace llvm;

namespace {
struct IndependentBlocks : public FunctionPass {
  RegionInfo *RI;
  ScalarEvolution *SE;
  ScopDetection *SD;
  LoopInfo *LI;

  BasicBlock *AllocaBlock;

  static char ID;

  IndependentBlocks() : FunctionPass(ID) {}

  // Create new code for every instruction operator that can be expressed by a
  // SCEV.  Like this there are just two types of instructions left:
  //
  // 1. Instructions that only reference loop ivs or parameters outside the
  // region.
  //
  // 2. Instructions that are not used for any memory modification. (These
  //    will be ignored later on.)
  //
  // Blocks containing only these kind of instructions are called independent
  // blocks as they can be scheduled arbitrarily.
  bool createIndependentBlocks(BasicBlock *BB, const Region *R);
  bool createIndependentBlocks(const Region *R);

  // Elimination on the Scop to eliminate the scalar dependences come with
  // trivially dead instructions.
  bool eliminateDeadCode(const Region *R);

  //===--------------------------------------------------------------------===//
  /// Non trivial scalar dependences checking functions.
  /// Non trivial scalar dependences occur when the def and use are located in
  /// different BBs and we can not move them into the same one. This will
  /// prevent use from schedule BBs arbitrarily.
  ///
  /// @brief This function checks if a scalar value that is part of the
  ///        Scop is used outside of the Scop.
  ///
  /// @param Use  The use of the instruction.
  /// @param R    The maximum region in the Scop.
  ///
  /// @return Return true if the Use of an instruction and the instruction
  ///         itself form a non trivial scalar dependence.
  static bool isEscapeUse(const Value *Use, const Region *R);

  /// @brief This function just checks if a Value is either defined in the same
  ///        basic block or outside the region, such that there are no scalar
  ///        dependences between basic blocks that are both part of the same
  ///        region.
  ///
  /// @param Operand  The operand of the instruction.
  /// @param CurBB    The BasicBlock that contains the instruction.
  /// @param R        The maximum region in the Scop.
  ///
  /// @return Return true if the Operand of an instruction and the instruction
  ///         itself form a non trivial scalar (true) dependence.
  bool isEscapeOperand(const Value *Operand, const BasicBlock *CurBB,
                       const Region *R) const;

  //===--------------------------------------------------------------------===//
  /// Operand tree moving functions.
  /// Trivial scalar dependences can eliminate by move the def to the same BB
  /// that containing use.
  ///
  /// @brief Check if the instruction can be moved to another place safely.
  ///
  /// @param Inst The instruction.
  ///
  /// @return Return true if the instruction can be moved safely, false
  ///         otherwise.
  static bool isSafeToMove(Instruction *Inst);

  typedef std::map<Instruction *, Instruction *> ReplacedMapType;

  /// @brief Move all safe to move instructions in the Operand Tree (DAG) to
  ///        eliminate trivial scalar dependences.
  ///
  /// @param Inst         The root of the operand Tree.
  /// @param R            The maximum region in the Scop.
  /// @param ReplacedMap  The map that mapping original instruction to the moved
  ///                     instruction.
  /// @param InsertPos    The insert position of the moved instructions.
  void moveOperandTree(Instruction *Inst, const Region *R,
                       ReplacedMapType &ReplacedMap, Instruction *InsertPos);

  bool isIndependentBlock(const Region *R, BasicBlock *BB) const;
  bool areAllBlocksIndependent(const Region *R) const;

  // Split the exit block to hold load instructions.
  bool splitExitBlock(Region *R);
  bool onlyUsedInRegion(Instruction *Inst, const Region *R);
  bool translateScalarToArray(BasicBlock *BB, const Region *R);
  bool translateScalarToArray(Instruction *Inst, const Region *R);
  bool translateScalarToArray(const Region *R);

  bool runOnFunction(Function &F);
  void verifyAnalysis() const;
  void verifyScop(const Region *R) const;
  void getAnalysisUsage(AnalysisUsage &AU) const;
};
}

bool IndependentBlocks::isSafeToMove(Instruction *Inst) {
  if (Inst->mayReadFromMemory() || Inst->mayWriteToMemory())
    return false;

  return isSafeToSpeculativelyExecute(Inst);
}

void IndependentBlocks::moveOperandTree(Instruction *Inst, const Region *R,
                                        ReplacedMapType &ReplacedMap,
                                        Instruction *InsertPos) {
  BasicBlock *CurBB = Inst->getParent();

  // Depth first traverse the operand tree (or operand dag, because we will
  // stop at PHINodes, so there are no cycle).
  typedef Instruction::op_iterator ChildIt;
  std::vector<std::pair<Instruction *, ChildIt> > WorkStack;

  WorkStack.push_back(std::make_pair(Inst, Inst->op_begin()));
  DenseSet<Instruction *> VisitedSet;

  while (!WorkStack.empty()) {
    Instruction *CurInst = WorkStack.back().first;
    ChildIt It = WorkStack.back().second;
    DEBUG(dbgs() << "Checking Operand of Node:\n" << *CurInst << "\n------>\n");
    if (It == CurInst->op_end()) {
      // Insert the new instructions in topological order.
      if (!CurInst->getParent()) {
        CurInst->insertBefore(InsertPos);
        SE->forgetValue(CurInst);
      }

      WorkStack.pop_back();
    } else {
      // for each node N,
      Instruction *Operand = dyn_cast<Instruction>(*It);
      ++WorkStack.back().second;

      // Can not move no instruction value.
      if (Operand == 0)
        continue;

      DEBUG(dbgs() << "For Operand:\n" << *Operand << "\n--->");

      // If the Scop Region does not contain N, skip it and all its operands and
      // continue: because we reach a "parameter".
      // FIXME: we must keep the predicate instruction inside the Scop,
      // otherwise it will be translated to a load instruction, and we can not
      // handle load as affine predicate at this moment.
      if (!R->contains(Operand) && !isa<TerminatorInst>(CurInst)) {
        DEBUG(dbgs() << "Out of region.\n");
        continue;
      }

      if (canSynthesize(Operand, LI, SE, R)) {
        DEBUG(dbgs() << "is IV.\n");
        continue;
      }

      // We can not move the operand, a non trivial scalar dependence found!
      if (!isSafeToMove(Operand)) {
        DEBUG(dbgs() << "Can not move!\n");
        continue;
      }

      // Do not need to move instruction if it is contained in the same BB with
      // the root instruction.
      if (Operand->getParent() == CurBB) {
        DEBUG(dbgs() << "No need to move.\n");
        // Try to move its operand, but do not visit an instuction twice.
        if (VisitedSet.insert(Operand).second)
          WorkStack.push_back(std::make_pair(Operand, Operand->op_begin()));
        continue;
      }

      // Now we need to move Operand to CurBB.
      // Check if we already moved it.
      ReplacedMapType::iterator At = ReplacedMap.find(Operand);
      if (At != ReplacedMap.end()) {
        DEBUG(dbgs() << "Moved.\n");
        Instruction *MovedOp = At->second;
        It->set(MovedOp);
        SE->forgetValue(MovedOp);
      } else {
        // Note that NewOp is not inserted in any BB now, we will insert it when
        // it popped form the work stack, so it will be inserted in topological
        // order.
        Instruction *NewOp = Operand->clone();
        NewOp->setName(Operand->getName() + ".moved.to." + CurBB->getName());
        DEBUG(dbgs() << "Move to " << *NewOp << "\n");
        It->set(NewOp);
        ReplacedMap.insert(std::make_pair(Operand, NewOp));
        SE->forgetValue(Operand);

        // Process its operands, but do not visit an instuction twice.
        if (VisitedSet.insert(NewOp).second)
          WorkStack.push_back(std::make_pair(NewOp, NewOp->op_begin()));
      }
    }
  }

  SE->forgetValue(Inst);
}

bool IndependentBlocks::createIndependentBlocks(BasicBlock *BB,
                                                const Region *R) {
  std::vector<Instruction *> WorkList;
  for (BasicBlock::iterator II = BB->begin(), IE = BB->end(); II != IE; ++II)
    if (!isSafeToMove(II) && !canSynthesize(II, LI, SE, R))
      WorkList.push_back(II);

  ReplacedMapType ReplacedMap;
  Instruction *InsertPos = BB->getFirstNonPHIOrDbg();

  for (std::vector<Instruction *>::iterator I = WorkList.begin(),
                                            E = WorkList.end();
       I != E; ++I)
    moveOperandTree(*I, R, ReplacedMap, InsertPos);

  // The BB was changed if we replaced any operand.
  return !ReplacedMap.empty();
}

bool IndependentBlocks::createIndependentBlocks(const Region *R) {
  bool Changed = false;

  for (Region::const_block_iterator SI = R->block_begin(), SE = R->block_end();
       SI != SE; ++SI)
    Changed |= createIndependentBlocks(*SI, R);

  return Changed;
}

bool IndependentBlocks::eliminateDeadCode(const Region *R) {
  std::vector<Instruction *> WorkList;

  // Find all trivially dead instructions.
  for (Region::const_block_iterator SI = R->block_begin(), SE = R->block_end();
       SI != SE; ++SI)
    for (BasicBlock::iterator I = (*SI)->begin(), E = (*SI)->end(); I != E; ++I)
      if (isInstructionTriviallyDead(I))
        WorkList.push_back(I);

  if (WorkList.empty())
    return false;

  // Delete them so the cross BB scalar dependences come with them will
  // also be eliminated.
  while (!WorkList.empty()) {
    RecursivelyDeleteTriviallyDeadInstructions(WorkList.back());
    WorkList.pop_back();
  }

  return true;
}

bool IndependentBlocks::isEscapeUse(const Value *Use, const Region *R) {
  // Non-instruction user will never escape.
  if (!isa<Instruction>(Use))
    return false;

  return !R->contains(cast<Instruction>(Use));
}

bool IndependentBlocks::isEscapeOperand(const Value *Operand,
                                        const BasicBlock *CurBB,
                                        const Region *R) const {
  const Instruction *OpInst = dyn_cast<Instruction>(Operand);

  // Non-instruction operand will never escape.
  if (OpInst == 0)
    return false;

  // Induction variables are valid operands.
  if (canSynthesize(OpInst, LI, SE, R))
    return false;

  // A value from a different BB is used in the same region.
  return R->contains(OpInst) && (OpInst->getParent() != CurBB);
}

bool IndependentBlocks::splitExitBlock(Region *R) {
  // Split the exit BB to place the load instruction of escaped users.
  BasicBlock *ExitBB = R->getExit();
  Region *ExitRegion = RI->getRegionFor(ExitBB);

  if (ExitBB != ExitRegion->getEntry())
    return false;

  BasicBlock *NewExit = createSingleExitEdge(R, this);

  std::vector<Region *> toUpdate;
  toUpdate.push_back(R);

  while (!toUpdate.empty()) {
    Region *Reg = toUpdate.back();
    toUpdate.pop_back();

    for (Region::iterator I = Reg->begin(), E = Reg->end(); I != E; ++I) {
      Region *SubR = *I;

      if (SubR->getExit() == ExitBB)
        toUpdate.push_back(SubR);
    }

    Reg->replaceExit(NewExit);
  }

  RI->setRegionFor(NewExit, R->getParent());
  return true;
}

bool IndependentBlocks::translateScalarToArray(const Region *R) {
  bool Changed = false;

  for (Region::const_block_iterator SI = R->block_begin(), SE = R->block_end();
       SI != SE; ++SI)
    Changed |= translateScalarToArray(*SI, R);

  return Changed;
}

// Returns true when Inst is only used inside region R.
bool IndependentBlocks::onlyUsedInRegion(Instruction *Inst, const Region *R) {
  for (Instruction::use_iterator UI = Inst->use_begin(), UE = Inst->use_end();
       UI != UE; ++UI)
    if (Instruction *U = dyn_cast<Instruction>(*UI))
      if (isEscapeUse(U, R))
        return false;

  return true;
}

bool IndependentBlocks::translateScalarToArray(Instruction *Inst,
                                               const Region *R) {
  if (canSynthesize(Inst, LI, SE, R) && onlyUsedInRegion(Inst, R))
    return false;

  SmallVector<Instruction *, 4> LoadInside, LoadOutside;
  for (Instruction::use_iterator UI = Inst->use_begin(), UE = Inst->use_end();
       UI != UE; ++UI)
    // Inst is referenced outside or referenced as an escaped operand.
    if (Instruction *U = dyn_cast<Instruction>(*UI)) {
      if (isEscapeUse(U, R))
        LoadOutside.push_back(U);

      if (canSynthesize(U, LI, SE, R))
        continue;

      BasicBlock *UParent = U->getParent();
      if (R->contains(UParent) && isEscapeOperand(Inst, UParent, R))
        LoadInside.push_back(U);
    }

  if (LoadOutside.empty() && LoadInside.empty())
    return false;

  // Create the alloca.
  AllocaInst *Slot = new AllocaInst(
      Inst->getType(), 0, Inst->getName() + ".s2a", AllocaBlock->begin());
  assert(!isa<InvokeInst>(Inst) && "Unexpect Invoke in Scop!");

  // Store right after Inst, and make sure the position is after all phi nodes.
  BasicBlock::iterator StorePos;
  if (isa<PHINode>(Inst)) {
    StorePos = Inst->getParent()->getFirstNonPHI();
  } else {
    StorePos = Inst;
    StorePos++;
  }
  (void) new StoreInst(Inst, Slot, StorePos);

  if (!LoadOutside.empty()) {
    LoadInst *ExitLoad = new LoadInst(Slot, Inst->getName() + ".loadoutside",
                                      false, R->getExit()->getFirstNonPHI());

    while (!LoadOutside.empty()) {
      Instruction *U = LoadOutside.pop_back_val();
      assert(!isa<PHINode>(U) && "Can not handle PHI node outside!");
      SE->forgetValue(U);
      U->replaceUsesOfWith(Inst, ExitLoad);
    }
  }

  while (!LoadInside.empty()) {
    Instruction *U = LoadInside.pop_back_val();
    assert(!isa<PHINode>(U) && "Can not handle PHI node inside!");
    SE->forgetValue(U);
    LoadInst *L = new LoadInst(Slot, Inst->getName() + ".loadarray", false, U);
    U->replaceUsesOfWith(Inst, L);
  }

  return true;
}

bool IndependentBlocks::translateScalarToArray(BasicBlock *BB,
                                               const Region *R) {
  bool changed = false;

  SmallVector<Instruction *, 32> Insts;
  for (BasicBlock::iterator II = BB->begin(), IE = --BB->end(); II != IE; ++II)
    Insts.push_back(II);

  while (!Insts.empty()) {
    Instruction *Inst = Insts.pop_back_val();
    changed |= translateScalarToArray(Inst, R);
  }

  return changed;
}

bool IndependentBlocks::isIndependentBlock(const Region *R,
                                           BasicBlock *BB) const {
  for (BasicBlock::iterator II = BB->begin(), IE = --BB->end(); II != IE;
       ++II) {
    Instruction *Inst = &*II;

    if (canSynthesize(Inst, LI, SE, R))
      continue;

    // A value inside the Scop is referenced outside.
    for (Instruction::use_iterator UI = Inst->use_begin(), UE = Inst->use_end();
         UI != UE; ++UI) {
      if (isEscapeUse(*UI, R)) {
        DEBUG(dbgs() << "Instruction not independent:\n");
        DEBUG(dbgs() << "Instruction used outside the Scop!\n");
        DEBUG(Inst->print(dbgs()));
        DEBUG(dbgs() << "\n");
        return false;
      }
    }

    for (Instruction::op_iterator OI = Inst->op_begin(), OE = Inst->op_end();
         OI != OE; ++OI) {
      if (isEscapeOperand(*OI, BB, R)) {
        DEBUG(dbgs() << "Instruction in function '";
              WriteAsOperand(dbgs(), BB->getParent(), false);
              dbgs() << "' not independent:\n");
        DEBUG(dbgs() << "Uses invalid operator\n");
        DEBUG(Inst->print(dbgs()));
        DEBUG(dbgs() << "\n");
        DEBUG(dbgs() << "Invalid operator is: ";
              WriteAsOperand(dbgs(), *OI, false); dbgs() << "\n");
        return false;
      }
    }
  }

  return true;
}

bool IndependentBlocks::areAllBlocksIndependent(const Region *R) const {
  for (Region::const_block_iterator SI = R->block_begin(), SE = R->block_end();
       SI != SE; ++SI)
    if (!isIndependentBlock(R, *SI))
      return false;

  return true;
}

void IndependentBlocks::getAnalysisUsage(AnalysisUsage &AU) const {
  // FIXME: If we set preserves cfg, the cfg only passes do not need to
  // be "addPreserved"?
  AU.addPreserved<DominatorTree>();
  AU.addPreserved<DominanceFrontier>();
  AU.addPreserved<PostDominatorTree>();
  AU.addRequired<RegionInfo>();
  AU.addPreserved<RegionInfo>();
  AU.addRequired<LoopInfo>();
  AU.addPreserved<LoopInfo>();
  AU.addRequired<ScalarEvolution>();
  AU.addPreserved<ScalarEvolution>();
  AU.addRequired<ScopDetection>();
  AU.addPreserved<ScopDetection>();
#ifdef CLOOG_FOUND
  AU.addPreserved<CloogInfo>();
#endif
}

bool IndependentBlocks::runOnFunction(llvm::Function &F) {
  bool Changed = false;

  RI = &getAnalysis<RegionInfo>();
  LI = &getAnalysis<LoopInfo>();
  SD = &getAnalysis<ScopDetection>();
  SE = &getAnalysis<ScalarEvolution>();

  AllocaBlock = &F.getEntryBlock();

  DEBUG(dbgs() << "Run IndepBlock on " << F.getName() << '\n');

  for (ScopDetection::iterator I = SD->begin(), E = SD->end(); I != E; ++I) {
    const Region *R = *I;
    Changed |= createIndependentBlocks(R);
    Changed |= eliminateDeadCode(R);
    // This may change the RegionTree.
    Changed |= splitExitBlock(const_cast<Region *>(R));
  }

  DEBUG(dbgs() << "Before Scalar to Array------->\n");
  DEBUG(F.dump());

  for (ScopDetection::iterator I = SD->begin(), E = SD->end(); I != E; ++I)
    Changed |= translateScalarToArray(*I);

  DEBUG(dbgs() << "After Independent Blocks------------->\n");
  DEBUG(F.dump());

  verifyAnalysis();

  return Changed;
}

void IndependentBlocks::verifyAnalysis() const {
  for (ScopDetection::const_iterator I = SD->begin(), E = SD->end(); I != E;
       ++I)
    verifyScop(*I);
}

void IndependentBlocks::verifyScop(const Region *R) const {
  assert(areAllBlocksIndependent(R) && "Cannot generate independent blocks");
}

char IndependentBlocks::ID = 0;
char &polly::IndependentBlocksID = IndependentBlocks::ID;

Pass *polly::createIndependentBlocksPass() { return new IndependentBlocks(); }

INITIALIZE_PASS_BEGIN(IndependentBlocks, "polly-independent",
                      "Polly - Create independent blocks", false, false);
INITIALIZE_PASS_DEPENDENCY(LoopInfo);
INITIALIZE_PASS_DEPENDENCY(RegionInfo);
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution);
INITIALIZE_PASS_DEPENDENCY(ScopDetection);
INITIALIZE_PASS_END(IndependentBlocks, "polly-independent",
                    "Polly - Create independent blocks", false, false)
