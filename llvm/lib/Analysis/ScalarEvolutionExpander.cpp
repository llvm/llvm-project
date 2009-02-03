//===- ScalarEvolutionExpander.cpp - Scalar Evolution Analysis --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the scalar evolution expander,
// which is used to generate the code corresponding to a given scalar evolution
// expression.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/LoopInfo.h"
using namespace llvm;

/// InsertCastOfTo - Insert a cast of V to the specified type, doing what
/// we can to share the casts.
Value *SCEVExpander::InsertCastOfTo(Instruction::CastOps opcode, Value *V, 
                                    const Type *Ty) {
  // FIXME: keep track of the cast instruction.
  if (Constant *C = dyn_cast<Constant>(V))
    return ConstantExpr::getCast(opcode, C, Ty);
  
  if (Argument *A = dyn_cast<Argument>(V)) {
    // Check to see if there is already a cast!
    for (Value::use_iterator UI = A->use_begin(), E = A->use_end();
         UI != E; ++UI) {
      if ((*UI)->getType() == Ty)
        if (CastInst *CI = dyn_cast<CastInst>(cast<Instruction>(*UI)))
          if (CI->getOpcode() == opcode) {
            // If the cast isn't the first instruction of the function, move it.
            if (BasicBlock::iterator(CI) != 
                A->getParent()->getEntryBlock().begin()) {
              CI->moveBefore(A->getParent()->getEntryBlock().begin());
            }
            return CI;
          }
    }
    return CastInst::Create(opcode, V, Ty, V->getName(), 
                            A->getParent()->getEntryBlock().begin());
  }

  Instruction *I = cast<Instruction>(V);

  // Check to see if there is already a cast.  If there is, use it.
  for (Value::use_iterator UI = I->use_begin(), E = I->use_end();
       UI != E; ++UI) {
    if ((*UI)->getType() == Ty)
      if (CastInst *CI = dyn_cast<CastInst>(cast<Instruction>(*UI)))
        if (CI->getOpcode() == opcode) {
          BasicBlock::iterator It = I; ++It;
          if (isa<InvokeInst>(I))
            It = cast<InvokeInst>(I)->getNormalDest()->begin();
          while (isa<PHINode>(It)) ++It;
          if (It != BasicBlock::iterator(CI)) {
            // Splice the cast immediately after the operand in question.
            CI->moveBefore(It);
          }
          return CI;
        }
  }
  BasicBlock::iterator IP = I; ++IP;
  if (InvokeInst *II = dyn_cast<InvokeInst>(I))
    IP = II->getNormalDest()->begin();
  while (isa<PHINode>(IP)) ++IP;
  return CastInst::Create(opcode, V, Ty, V->getName(), IP);
}

/// InsertBinop - Insert the specified binary operator, doing a small amount
/// of work to avoid inserting an obviously redundant operation.
Value *SCEVExpander::InsertBinop(Instruction::BinaryOps Opcode, Value *LHS,
                                 Value *RHS, Instruction *InsertPt) {
  // Fold a binop with constant operands.
  if (Constant *CLHS = dyn_cast<Constant>(LHS))
    if (Constant *CRHS = dyn_cast<Constant>(RHS))
      return ConstantExpr::get(Opcode, CLHS, CRHS);

  // Do a quick scan to see if we have this binop nearby.  If so, reuse it.
  unsigned ScanLimit = 6;
  BasicBlock::iterator BlockBegin = InsertPt->getParent()->begin();
  if (InsertPt != BlockBegin) {
    // Scanning starts from the last instruction before InsertPt.
    BasicBlock::iterator IP = InsertPt;
    --IP;
    for (; ScanLimit; --IP, --ScanLimit) {
      if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(IP))
        if (BinOp->getOpcode() == Opcode && BinOp->getOperand(0) == LHS &&
            BinOp->getOperand(1) == RHS)
          return BinOp;
      if (IP == BlockBegin) break;
    }
  }
  
  // If we haven't found this binop, insert it.
  return BinaryOperator::Create(Opcode, LHS, RHS, "tmp", InsertPt);
}

Value *SCEVExpander::visitAddExpr(SCEVAddExpr *S) {
  Value *V = expand(S->getOperand(S->getNumOperands()-1));

  // Emit a bunch of add instructions
  for (int i = S->getNumOperands()-2; i >= 0; --i)
    V = InsertBinop(Instruction::Add, V, expand(S->getOperand(i)),
                    InsertPt);
  return V;
}
    
Value *SCEVExpander::visitMulExpr(SCEVMulExpr *S) {
  int FirstOp = 0;  // Set if we should emit a subtract.
  if (SCEVConstant *SC = dyn_cast<SCEVConstant>(S->getOperand(0)))
    if (SC->getValue()->isAllOnesValue())
      FirstOp = 1;

  int i = S->getNumOperands()-2;
  Value *V = expand(S->getOperand(i+1));

  // Emit a bunch of multiply instructions
  for (; i >= FirstOp; --i)
    V = InsertBinop(Instruction::Mul, V, expand(S->getOperand(i)),
                    InsertPt);
  // -1 * ...  --->  0 - ...
  if (FirstOp == 1)
    V = InsertBinop(Instruction::Sub, Constant::getNullValue(V->getType()), V,
                    InsertPt);
  return V;
}

Value *SCEVExpander::visitUDivExpr(SCEVUDivExpr *S) {
  Value *LHS = expand(S->getLHS());
  if (SCEVConstant *SC = dyn_cast<SCEVConstant>(S->getRHS())) {
    const APInt &RHS = SC->getValue()->getValue();
    if (RHS.isPowerOf2())
      return InsertBinop(Instruction::LShr, LHS,
                         ConstantInt::get(S->getType(), RHS.logBase2()),
                         InsertPt);
  }

  Value *RHS = expand(S->getRHS());
  return InsertBinop(Instruction::UDiv, LHS, RHS, InsertPt);
}

Value *SCEVExpander::visitAddRecExpr(SCEVAddRecExpr *S) {
  const Type *Ty = S->getType();
  const Loop *L = S->getLoop();
  // We cannot yet do fp recurrences, e.g. the xform of {X,+,F} --> X+{0,+,F}
  assert(Ty->isInteger() && "Cannot expand fp recurrences yet!");

  // {X,+,F} --> X + {0,+,F}
  if (!S->getStart()->isZero()) {
    Value *Start = expand(S->getStart());
    std::vector<SCEVHandle> NewOps(S->op_begin(), S->op_end());
    NewOps[0] = SE.getIntegerSCEV(0, Ty);
    Value *Rest = expand(SE.getAddRecExpr(NewOps, L));

    // FIXME: look for an existing add to use.
    return InsertBinop(Instruction::Add, Rest, Start, InsertPt);
  }

  // {0,+,1} --> Insert a canonical induction variable into the loop!
  if (S->isAffine() &&
      S->getOperand(1) == SE.getIntegerSCEV(1, Ty)) {
    // Create and insert the PHI node for the induction variable in the
    // specified loop.
    BasicBlock *Header = L->getHeader();
    PHINode *PN = PHINode::Create(Ty, "indvar", Header->begin());
    PN->addIncoming(Constant::getNullValue(Ty), L->getLoopPreheader());

    pred_iterator HPI = pred_begin(Header);
    assert(HPI != pred_end(Header) && "Loop with zero preds???");
    if (!L->contains(*HPI)) ++HPI;
    assert(HPI != pred_end(Header) && L->contains(*HPI) &&
           "No backedge in loop?");

    // Insert a unit add instruction right before the terminator corresponding
    // to the back-edge.
    Constant *One = ConstantInt::get(Ty, 1);
    Instruction *Add = BinaryOperator::CreateAdd(PN, One, "indvar.next",
                                                 (*HPI)->getTerminator());

    pred_iterator PI = pred_begin(Header);
    if (*PI == L->getLoopPreheader())
      ++PI;
    PN->addIncoming(Add, *PI);
    return PN;
  }

  // Get the canonical induction variable I for this loop.
  Value *I = getOrInsertCanonicalInductionVariable(L, Ty);

  // If this is a simple linear addrec, emit it now as a special case.
  if (S->isAffine()) {   // {0,+,F} --> i*F
    Value *F = expand(S->getOperand(1));
    
    // IF the step is by one, just return the inserted IV.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(F))
      if (CI->getValue() == 1)
        return I;
    
    // If the insert point is directly inside of the loop, emit the multiply at
    // the insert point.  Otherwise, L is a loop that is a parent of the insert
    // point loop.  If we can, move the multiply to the outer most loop that it
    // is safe to be in.
    Instruction *MulInsertPt = InsertPt;
    Loop *InsertPtLoop = LI.getLoopFor(MulInsertPt->getParent());
    if (InsertPtLoop != L && InsertPtLoop &&
        L->contains(InsertPtLoop->getHeader())) {
      do {
        // If we cannot hoist the multiply out of this loop, don't.
        if (!InsertPtLoop->isLoopInvariant(F)) break;

        BasicBlock *InsertPtLoopPH = InsertPtLoop->getLoopPreheader();

        // If this loop hasn't got a preheader, we aren't able to hoist the
        // multiply.
        if (!InsertPtLoopPH)
          break;

        // Otherwise, move the insert point to the preheader.
        MulInsertPt = InsertPtLoopPH->getTerminator();
        InsertPtLoop = InsertPtLoop->getParentLoop();
      } while (InsertPtLoop != L);
    }
    
    return InsertBinop(Instruction::Mul, I, F, MulInsertPt);
  }

  // If this is a chain of recurrences, turn it into a closed form, using the
  // folders, then expandCodeFor the closed form.  This allows the folders to
  // simplify the expression without having to build a bunch of special code
  // into this folder.
  SCEVHandle IH = SE.getUnknown(I);   // Get I as a "symbolic" SCEV.

  SCEVHandle V = S->evaluateAtIteration(IH, SE);
  //cerr << "Evaluated: " << *this << "\n     to: " << *V << "\n";

  return expand(V);
}

Value *SCEVExpander::visitTruncateExpr(SCEVTruncateExpr *S) {
  Value *V = expand(S->getOperand());
  return CastInst::CreateTruncOrBitCast(V, S->getType(), "tmp.", InsertPt);
}

Value *SCEVExpander::visitZeroExtendExpr(SCEVZeroExtendExpr *S) {
  Value *V = expand(S->getOperand());
  return CastInst::CreateZExtOrBitCast(V, S->getType(), "tmp.", InsertPt);
}

Value *SCEVExpander::visitSignExtendExpr(SCEVSignExtendExpr *S) {
  Value *V = expand(S->getOperand());
  return CastInst::CreateSExtOrBitCast(V, S->getType(), "tmp.", InsertPt);
}

Value *SCEVExpander::visitSMaxExpr(SCEVSMaxExpr *S) {
  Value *LHS = expand(S->getOperand(0));
  for (unsigned i = 1; i < S->getNumOperands(); ++i) {
    Value *RHS = expand(S->getOperand(i));
    Value *ICmp = new ICmpInst(ICmpInst::ICMP_SGT, LHS, RHS, "tmp", InsertPt);
    LHS = SelectInst::Create(ICmp, LHS, RHS, "smax", InsertPt);
  }
  return LHS;
}

Value *SCEVExpander::visitUMaxExpr(SCEVUMaxExpr *S) {
  Value *LHS = expand(S->getOperand(0));
  for (unsigned i = 1; i < S->getNumOperands(); ++i) {
    Value *RHS = expand(S->getOperand(i));
    Value *ICmp = new ICmpInst(ICmpInst::ICMP_UGT, LHS, RHS, "tmp", InsertPt);
    LHS = SelectInst::Create(ICmp, LHS, RHS, "umax", InsertPt);
  }
  return LHS;
}

Value *SCEVExpander::expandCodeFor(SCEVHandle SH, Instruction *IP) {
  // Expand the code for this SCEV.
  this->InsertPt = IP;
  return expand(SH);
}

Value *SCEVExpander::expand(SCEV *S) {
  // Check to see if we already expanded this.
  std::map<SCEVHandle, Value*>::iterator I = InsertedExpressions.find(S);
  if (I != InsertedExpressions.end())
    return I->second;
  
  Value *V = visit(S);
  InsertedExpressions[S] = V;
  return V;
}
