//===- IndVarSimplify.cpp - Induction Variable Elimination ----------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Guarantees that all loops with identifiable, linear, induction variables will
// be transformed to have a single, canonical, induction variable.  After this
// pass runs, it guarantees the the first PHI node of the header block in the
// loop is the canonical induction variable if there is one.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/Constants.h"
#include "llvm/Type.h"
#include "llvm/iPHINode.h"
#include "llvm/iOther.h"
#include "llvm/Analysis/InductionVariable.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/CFG.h"
#include "llvm/Transforms/Utils/Local.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
#include "Support/STLExtras.h"
#include <algorithm>
using namespace llvm;

namespace {
  Statistic<> NumRemoved ("indvars", "Number of aux indvars removed");
  Statistic<> NumInserted("indvars", "Number of canonical indvars added");
}

// InsertCast - Cast Val to Ty, setting a useful name on the cast if Val has a
// name...
//
static Instruction *InsertCast(Value *Val, const Type *Ty,
                               Instruction *InsertBefore) {
  return new CastInst(Val, Ty, Val->getName()+"-casted", InsertBefore);
}

static bool TransformLoop(LoopInfo *Loops, Loop *Loop) {
  // Transform all subloops before this loop...
  bool Changed = reduce_apply_bool(Loop->getSubLoops().begin(),
                                   Loop->getSubLoops().end(),
                              std::bind1st(std::ptr_fun(TransformLoop), Loops));
  // Get the header node for this loop.  All of the phi nodes that could be
  // induction variables must live in this basic block.
  //
  BasicBlock *Header = Loop->getHeader();
  
  // Loop over all of the PHI nodes in the basic block, calculating the
  // induction variables that they represent... stuffing the induction variable
  // info into a vector...
  //
  std::vector<InductionVariable> IndVars;    // Induction variables for block
  BasicBlock::iterator AfterPHIIt = Header->begin();
  for (; PHINode *PN = dyn_cast<PHINode>(AfterPHIIt); ++AfterPHIIt)
    IndVars.push_back(InductionVariable(PN, Loops));
  // AfterPHIIt now points to first non-phi instruction...

  // If there are no phi nodes in this basic block, there can't be indvars...
  if (IndVars.empty()) return Changed;
  
  // Loop over the induction variables, looking for a canonical induction
  // variable, and checking to make sure they are not all unknown induction
  // variables.
  //
  bool FoundIndVars = false;
  InductionVariable *Canonical = 0;
  for (unsigned i = 0; i < IndVars.size(); ++i) {
    if (IndVars[i].InductionType == InductionVariable::Canonical &&
        !isa<PointerType>(IndVars[i].Phi->getType()))
      Canonical = &IndVars[i];
    if (IndVars[i].InductionType != InductionVariable::Unknown)
      FoundIndVars = true;
  }

  // No induction variables, bail early... don't add a canonical indvar
  if (!FoundIndVars) return Changed;

  // Okay, we want to convert other induction variables to use a canonical
  // indvar.  If we don't have one, add one now...
  if (!Canonical) {
    // Create the PHI node for the new induction variable, and insert the phi
    // node at the start of the PHI nodes...
    PHINode *PN = new PHINode(Type::UIntTy, "cann-indvar", Header->begin());

    // Create the increment instruction to add one to the counter...
    Instruction *Add = BinaryOperator::create(Instruction::Add, PN,
                                              ConstantUInt::get(Type::UIntTy,1),
                                              "add1-indvar", AfterPHIIt);

    // Figure out which block is incoming and which is the backedge for the loop
    BasicBlock *Incoming, *BackEdgeBlock;
    pred_iterator PI = pred_begin(Header);
    assert(PI != pred_end(Header) && "Loop headers should have 2 preds!");
    if (Loop->contains(*PI)) {  // First pred is back edge...
      BackEdgeBlock = *PI++;
      Incoming      = *PI++;
    } else {
      Incoming      = *PI++;
      BackEdgeBlock = *PI++;
    }
    assert(PI == pred_end(Header) && "Loop headers should have 2 preds!");
    
    // Add incoming values for the PHI node...
    PN->addIncoming(Constant::getNullValue(Type::UIntTy), Incoming);
    PN->addIncoming(Add, BackEdgeBlock);

    // Analyze the new induction variable...
    IndVars.push_back(InductionVariable(PN, Loops));
    assert(IndVars.back().InductionType == InductionVariable::Canonical &&
           "Just inserted canonical indvar that is not canonical!");
    Canonical = &IndVars.back();
    ++NumInserted;
    Changed = true;
  } else {
    // If we have a canonical induction variable, make sure that it is the first
    // one in the basic block.
    if (&Header->front() != Canonical->Phi)
      Header->getInstList().splice(Header->begin(), Header->getInstList(),
                                   Canonical->Phi);
  }

  DEBUG(std::cerr << "Induction variables:\n");

  // Get the current loop iteration count, which is always the value of the
  // canonical phi node...
  //
  PHINode *IterCount = Canonical->Phi;

  // Loop through and replace all of the auxiliary induction variables with
  // references to the canonical induction variable...
  //
  for (unsigned i = 0; i < IndVars.size(); ++i) {
    InductionVariable *IV = &IndVars[i];

    DEBUG(IV->print(std::cerr));

    while (isa<PHINode>(AfterPHIIt)) ++AfterPHIIt;

    // Don't do math with pointers...
    const Type *IVTy = IV->Phi->getType();
    if (isa<PointerType>(IVTy)) IVTy = Type::ULongTy;

    // Don't modify the canonical indvar or unrecognized indvars...
    if (IV != Canonical && IV->InductionType != InductionVariable::Unknown) {
      Instruction *Val = IterCount;
      if (!isa<ConstantInt>(IV->Step) ||   // If the step != 1
          !cast<ConstantInt>(IV->Step)->equalsInt(1)) {

        // If the types are not compatible, insert a cast now...
        if (Val->getType() != IVTy)
          Val = InsertCast(Val, IVTy, AfterPHIIt);
        if (IV->Step->getType() != IVTy)
          IV->Step = InsertCast(IV->Step, IVTy, AfterPHIIt);

        Val = BinaryOperator::create(Instruction::Mul, Val, IV->Step,
                                     IV->Phi->getName()+"-scale", AfterPHIIt);
      }

      // If the start != 0
      if (IV->Start != Constant::getNullValue(IV->Start->getType())) {
        // If the types are not compatible, insert a cast now...
        if (Val->getType() != IVTy)
          Val = InsertCast(Val, IVTy, AfterPHIIt);
        if (IV->Start->getType() != IVTy)
          IV->Start = InsertCast(IV->Start, IVTy, AfterPHIIt);

        // Insert the instruction after the phi nodes...
        Val = BinaryOperator::create(Instruction::Add, Val, IV->Start,
                                     IV->Phi->getName()+"-offset", AfterPHIIt);
      }

      // If the PHI node has a different type than val is, insert a cast now...
      if (Val->getType() != IV->Phi->getType())
        Val = InsertCast(Val, IV->Phi->getType(), AfterPHIIt);
      
      // Replace all uses of the old PHI node with the new computed value...
      IV->Phi->replaceAllUsesWith(Val);

      // Move the PHI name to it's new equivalent value...
      std::string OldName = IV->Phi->getName();
      IV->Phi->setName("");
      Val->setName(OldName);

      // Get the incoming values used by the PHI node
      std::vector<Value*> PHIOps;
      PHIOps.reserve(IV->Phi->getNumIncomingValues());
      for (unsigned i = 0, e = IV->Phi->getNumIncomingValues(); i != e; ++i)
        PHIOps.push_back(IV->Phi->getIncomingValue(i));

      // Delete the old, now unused, phi node...
      Header->getInstList().erase(IV->Phi);

      // If the PHI is the last user of any instructions for computing PHI nodes
      // that are irrelevant now, delete those instructions.
      while (!PHIOps.empty()) {
        Instruction *MaybeDead = dyn_cast<Instruction>(PHIOps.back());
        PHIOps.pop_back();

        if (MaybeDead && isInstructionTriviallyDead(MaybeDead)) {
          PHIOps.insert(PHIOps.end(), MaybeDead->op_begin(),
                        MaybeDead->op_end());
          MaybeDead->getParent()->getInstList().erase(MaybeDead);
          
          // Erase any duplicates entries in the PHIOps list.
          std::vector<Value*>::iterator It =
            std::find(PHIOps.begin(), PHIOps.end(), MaybeDead);
          while (It != PHIOps.end()) {
            PHIOps.erase(It);
            It = std::find(PHIOps.begin(), PHIOps.end(), MaybeDead);
          }

          // Erasing the instruction could invalidate the AfterPHI iterator!
          AfterPHIIt = Header->begin();
        }
      }

      Changed = true;
      ++NumRemoved;
    }
  }

  return Changed;
}

namespace {
  struct InductionVariableSimplify : public FunctionPass {
    virtual bool runOnFunction(Function &) {
      LoopInfo &LI = getAnalysis<LoopInfo>();

      // Induction Variables live in the header nodes of loops
      return reduce_apply_bool(LI.getTopLevelLoops().begin(),
                               LI.getTopLevelLoops().end(),
                               std::bind1st(std::ptr_fun(TransformLoop), &LI));
    }
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<LoopInfo>();
      AU.addRequiredID(LoopSimplifyID);
      AU.setPreservesCFG();
    }
  };
  RegisterOpt<InductionVariableSimplify> X("indvars",
                                           "Canonicalize Induction Variables");
}

Pass *llvm::createIndVarSimplifyPass() {
  return new InductionVariableSimplify();
}

