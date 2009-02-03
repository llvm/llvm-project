//===---- llvm/Analysis/ScalarEvolutionExpander.h - SCEV Exprs --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the classes used to generate code from scalar expressions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SCALAREVOLUTION_EXPANDER_H
#define LLVM_ANALYSIS_SCALAREVOLUTION_EXPANDER_H

#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"

namespace llvm {
  /// SCEVExpander - This class uses information about analyze scalars to
  /// rewrite expressions in canonical form.
  ///
  /// Clients should create an instance of this class when rewriting is needed,
  /// and destroy it when finished to allow the release of the associated 
  /// memory.
  struct SCEVExpander : public SCEVVisitor<SCEVExpander, Value*> {
    ScalarEvolution &SE;
    LoopInfo &LI;
    std::map<SCEVHandle, Value*> InsertedExpressions;
    std::set<Instruction*> InsertedInstructions;

    Instruction *InsertPt;

    friend struct SCEVVisitor<SCEVExpander, Value*>;
  public:
    SCEVExpander(ScalarEvolution &se, LoopInfo &li) : SE(se), LI(li) {}

    LoopInfo &getLoopInfo() const { return LI; }

    /// clear - Erase the contents of the InsertedExpressions map so that users
    /// trying to expand the same expression into multiple BasicBlocks or
    /// different places within the same BasicBlock can do so.
    void clear() { InsertedExpressions.clear(); }

    /// isInsertedInstruction - Return true if the specified instruction was
    /// inserted by the code rewriter.  If so, the client should not modify the
    /// instruction.
    bool isInsertedInstruction(Instruction *I) const {
      return InsertedInstructions.count(I);
    }

    /// getOrInsertCanonicalInductionVariable - This method returns the
    /// canonical induction variable of the specified type for the specified
    /// loop (inserting one if there is none).  A canonical induction variable
    /// starts at zero and steps by one on each iteration.
    Value *getOrInsertCanonicalInductionVariable(const Loop *L, const Type *Ty){
      assert(Ty->isInteger() && "Can only insert integer induction variables!");
      SCEVHandle H = SE.getAddRecExpr(SE.getIntegerSCEV(0, Ty),
                                      SE.getIntegerSCEV(1, Ty), L);
      return expand(H);
    }

    /// addInsertedValue - Remember the specified instruction as being the
    /// canonical form for the specified SCEV.
    void addInsertedValue(Instruction *I, SCEV *S) {
      InsertedExpressions[S] = (Value*)I;
      InsertedInstructions.insert(I);
    }

    Instruction *getInsertionPoint() const { return InsertPt; }
    
    /// expandCodeFor - Insert code to directly compute the specified SCEV
    /// expression into the program.  The inserted code is inserted into the
    /// specified block.
    Value *expandCodeFor(SCEVHandle SH, Instruction *IP);

    /// InsertCastOfTo - Insert a cast of V to the specified type, doing what
    /// we can to share the casts.
    static Value *InsertCastOfTo(Instruction::CastOps opcode, Value *V, 
                                 const Type *Ty);
    /// InsertBinop - Insert the specified binary operator, doing a small amount
    /// of work to avoid inserting an obviously redundant operation.
    static Value *InsertBinop(Instruction::BinaryOps Opcode, Value *LHS,
                              Value *RHS, Instruction *InsertPt);
  protected:
    Value *expand(SCEV *S);
    
    Value *visitConstant(SCEVConstant *S) {
      return S->getValue();
    }

    Value *visitTruncateExpr(SCEVTruncateExpr *S);

    Value *visitZeroExtendExpr(SCEVZeroExtendExpr *S);

    Value *visitSignExtendExpr(SCEVSignExtendExpr *S);

    Value *visitAddExpr(SCEVAddExpr *S);

    Value *visitMulExpr(SCEVMulExpr *S);

    Value *visitUDivExpr(SCEVUDivExpr *S);

    Value *visitAddRecExpr(SCEVAddRecExpr *S);

    Value *visitSMaxExpr(SCEVSMaxExpr *S);

    Value *visitUMaxExpr(SCEVUMaxExpr *S);

    Value *visitUnknown(SCEVUnknown *S) {
      return S->getValue();
    }
  };
}

#endif

