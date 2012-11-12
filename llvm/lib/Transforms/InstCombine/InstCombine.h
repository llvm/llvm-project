//===- InstCombine.h - Main InstCombine pass definition -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef INSTCOMBINE_INSTCOMBINE_H
#define INSTCOMBINE_INSTCOMBINE_H

#include "InstCombineWorklist.h"
#include "llvm/IRBuilder.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Operator.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/TargetFolder.h"
#include "llvm/Transforms/Utils/SimplifyLibCalls.h"

namespace llvm {
  class CallSite;
  class DataLayout;
  class TargetLibraryInfo;
  class DbgDeclareInst;
  class MemIntrinsic;
  class MemSetInst;
  
/// SelectPatternFlavor - We can match a variety of different patterns for
/// select operations.
enum SelectPatternFlavor {
  SPF_UNKNOWN = 0,
  SPF_SMIN, SPF_UMIN,
  SPF_SMAX, SPF_UMAX
  //SPF_ABS - TODO.
};
  
/// getComplexity:  Assign a complexity or rank value to LLVM Values...
///   0 -> undef, 1 -> Const, 2 -> Other, 3 -> Arg, 3 -> Unary, 4 -> OtherInst
static inline unsigned getComplexity(Value *V) {
  if (isa<Instruction>(V)) {
    if (BinaryOperator::isNeg(V) ||
        BinaryOperator::isFNeg(V) ||
        BinaryOperator::isNot(V))
      return 3;
    return 4;
  }
  if (isa<Argument>(V)) return 3;
  return isa<Constant>(V) ? (isa<UndefValue>(V) ? 0 : 1) : 2;
}

  
/// InstCombineIRInserter - This is an IRBuilder insertion helper that works
/// just like the normal insertion helper, but also adds any new instructions
/// to the instcombine worklist.
class LLVM_LIBRARY_VISIBILITY InstCombineIRInserter 
    : public IRBuilderDefaultInserter<true> {
  InstCombineWorklist &Worklist;
public:
  InstCombineIRInserter(InstCombineWorklist &WL) : Worklist(WL) {}
  
  void InsertHelper(Instruction *I, const Twine &Name,
                    BasicBlock *BB, BasicBlock::iterator InsertPt) const {
    IRBuilderDefaultInserter<true>::InsertHelper(I, Name, BB, InsertPt);
    Worklist.Add(I);
  }
};
  
/// InstCombiner - The -instcombine pass.
class LLVM_LIBRARY_VISIBILITY InstCombiner
                             : public FunctionPass,
                               public InstVisitor<InstCombiner, Instruction*> {
  DataLayout *TD;
  TargetLibraryInfo *TLI;
  bool MadeIRChange;
  LibCallSimplifier *Simplifier;
public:
  /// Worklist - All of the instructions that need to be simplified.
  InstCombineWorklist Worklist;

  /// Builder - This is an IRBuilder that automatically inserts new
  /// instructions into the worklist when they are created.
  typedef IRBuilder<true, TargetFolder, InstCombineIRInserter> BuilderTy;
  BuilderTy *Builder;
      
  static char ID; // Pass identification, replacement for typeid
  InstCombiner() : FunctionPass(ID), TD(0), Builder(0) {
    initializeInstCombinerPass(*PassRegistry::getPassRegistry());
  }

public:
  virtual bool runOnFunction(Function &F);
  
  bool DoOneIteration(Function &F, unsigned ItNum);

  virtual void getAnalysisUsage(AnalysisUsage &AU) const;

  DataLayout *getDataLayout() const { return TD; }

  TargetLibraryInfo *getTargetLibraryInfo() const { return TLI; }

  // Visitation implementation - Implement instruction combining for different
  // instruction types.  The semantics are as follows:
  // Return Value:
  //    null        - No change was made
  //     I          - Change was made, I is still valid, I may be dead though
  //   otherwise    - Change was made, replace I with returned instruction
  //
  Instruction *visitAdd(BinaryOperator &I);
  Instruction *visitFAdd(BinaryOperator &I);
  Value *OptimizePointerDifference(Value *LHS, Value *RHS, Type *Ty);
  Instruction *visitSub(BinaryOperator &I);
  Instruction *visitFSub(BinaryOperator &I);
  Instruction *visitMul(BinaryOperator &I);
  Instruction *visitFMul(BinaryOperator &I);
  Instruction *visitURem(BinaryOperator &I);
  Instruction *visitSRem(BinaryOperator &I);
  Instruction *visitFRem(BinaryOperator &I);
  bool SimplifyDivRemOfSelect(BinaryOperator &I);
  Instruction *commonRemTransforms(BinaryOperator &I);
  Instruction *commonIRemTransforms(BinaryOperator &I);
  Instruction *commonDivTransforms(BinaryOperator &I);
  Instruction *commonIDivTransforms(BinaryOperator &I);
  Instruction *visitUDiv(BinaryOperator &I);
  Instruction *visitSDiv(BinaryOperator &I);
  Instruction *visitFDiv(BinaryOperator &I);
  Value *FoldAndOfICmps(ICmpInst *LHS, ICmpInst *RHS);
  Value *FoldAndOfFCmps(FCmpInst *LHS, FCmpInst *RHS);
  Instruction *visitAnd(BinaryOperator &I);
  Value *FoldOrOfICmps(ICmpInst *LHS, ICmpInst *RHS);
  Value *FoldOrOfFCmps(FCmpInst *LHS, FCmpInst *RHS);
  Instruction *FoldOrWithConstants(BinaryOperator &I, Value *Op,
                                   Value *A, Value *B, Value *C);
  Instruction *visitOr (BinaryOperator &I);
  Instruction *visitXor(BinaryOperator &I);
  Instruction *visitShl(BinaryOperator &I);
  Instruction *visitAShr(BinaryOperator &I);
  Instruction *visitLShr(BinaryOperator &I);
  Instruction *commonShiftTransforms(BinaryOperator &I);
  Instruction *FoldFCmp_IntToFP_Cst(FCmpInst &I, Instruction *LHSI,
                                    Constant *RHSC);
  Instruction *FoldCmpLoadFromIndexedGlobal(GetElementPtrInst *GEP,
                                            GlobalVariable *GV, CmpInst &ICI,
                                            ConstantInt *AndCst = 0);
  Instruction *visitFCmpInst(FCmpInst &I);
  Instruction *visitICmpInst(ICmpInst &I);
  Instruction *visitICmpInstWithCastAndCast(ICmpInst &ICI);
  Instruction *visitICmpInstWithInstAndIntCst(ICmpInst &ICI,
                                              Instruction *LHS,
                                              ConstantInt *RHS);
  Instruction *FoldICmpDivCst(ICmpInst &ICI, BinaryOperator *DivI,
                              ConstantInt *DivRHS);
  Instruction *FoldICmpShrCst(ICmpInst &ICI, BinaryOperator *DivI,
                              ConstantInt *DivRHS);
  Instruction *FoldICmpAddOpCst(ICmpInst &ICI, Value *X, ConstantInt *CI,
                                ICmpInst::Predicate Pred, Value *TheAdd);
  Instruction *FoldGEPICmp(GEPOperator *GEPLHS, Value *RHS,
                           ICmpInst::Predicate Cond, Instruction &I);
  Instruction *FoldShiftByConstant(Value *Op0, ConstantInt *Op1,
                                   BinaryOperator &I);
  Instruction *commonCastTransforms(CastInst &CI);
  Instruction *commonPointerCastTransforms(CastInst &CI);
  Instruction *visitTrunc(TruncInst &CI);
  Instruction *visitZExt(ZExtInst &CI);
  Instruction *visitSExt(SExtInst &CI);
  Instruction *visitFPTrunc(FPTruncInst &CI);
  Instruction *visitFPExt(CastInst &CI);
  Instruction *visitFPToUI(FPToUIInst &FI);
  Instruction *visitFPToSI(FPToSIInst &FI);
  Instruction *visitUIToFP(CastInst &CI);
  Instruction *visitSIToFP(CastInst &CI);
  Instruction *visitPtrToInt(PtrToIntInst &CI);
  Instruction *visitIntToPtr(IntToPtrInst &CI);
  Instruction *visitBitCast(BitCastInst &CI);
  Instruction *FoldSelectOpOp(SelectInst &SI, Instruction *TI,
                              Instruction *FI);
  Instruction *FoldSelectIntoOp(SelectInst &SI, Value*, Value*);
  Instruction *FoldSPFofSPF(Instruction *Inner, SelectPatternFlavor SPF1,
                            Value *A, Value *B, Instruction &Outer,
                            SelectPatternFlavor SPF2, Value *C);
  Instruction *visitSelectInst(SelectInst &SI);
  Instruction *visitSelectInstWithICmp(SelectInst &SI, ICmpInst *ICI);
  Instruction *visitCallInst(CallInst &CI);
  Instruction *visitInvokeInst(InvokeInst &II);

  Instruction *SliceUpIllegalIntegerPHI(PHINode &PN);
  Instruction *visitPHINode(PHINode &PN);
  Instruction *visitGetElementPtrInst(GetElementPtrInst &GEP);
  Instruction *visitAllocaInst(AllocaInst &AI);
  Instruction *visitAllocSite(Instruction &FI);
  Instruction *visitFree(CallInst &FI);
  Instruction *visitLoadInst(LoadInst &LI);
  Instruction *visitStoreInst(StoreInst &SI);
  Instruction *visitBranchInst(BranchInst &BI);
  Instruction *visitSwitchInst(SwitchInst &SI);
  Instruction *visitInsertElementInst(InsertElementInst &IE);
  Instruction *visitExtractElementInst(ExtractElementInst &EI);
  Instruction *visitShuffleVectorInst(ShuffleVectorInst &SVI);
  Instruction *visitExtractValueInst(ExtractValueInst &EV);
  Instruction *visitLandingPadInst(LandingPadInst &LI);

  // visitInstruction - Specify what to return for unhandled instructions...
  Instruction *visitInstruction(Instruction &I) { return 0; }

private:
  bool ShouldChangeType(Type *From, Type *To) const;
  Value *dyn_castNegVal(Value *V) const;
  Value *dyn_castFNegVal(Value *V) const;
  Type *FindElementAtOffset(Type *Ty, int64_t Offset, 
                                  SmallVectorImpl<Value*> &NewIndices);
  Instruction *FoldOpIntoSelect(Instruction &Op, SelectInst *SI);
                                 
  /// ShouldOptimizeCast - Return true if the cast from "V to Ty" actually
  /// results in any code being generated and is interesting to optimize out. If
  /// the cast can be eliminated by some other simple transformation, we prefer
  /// to do the simplification first.
  bool ShouldOptimizeCast(Instruction::CastOps opcode,const Value *V,
                          Type *Ty);

  Instruction *visitCallSite(CallSite CS);
  Instruction *tryOptimizeCall(CallInst *CI, const DataLayout *TD);
  bool transformConstExprCastCall(CallSite CS);
  Instruction *transformCallThroughTrampoline(CallSite CS,
                                              IntrinsicInst *Tramp);
  Instruction *transformZExtICmp(ICmpInst *ICI, Instruction &CI,
                                 bool DoXform = true);
  Instruction *transformSExtICmp(ICmpInst *ICI, Instruction &CI);
  bool WillNotOverflowSignedAdd(Value *LHS, Value *RHS);
  Value *EmitGEPOffset(User *GEP);

public:
  // InsertNewInstBefore - insert an instruction New before instruction Old
  // in the program.  Add the new instruction to the worklist.
  //
  Instruction *InsertNewInstBefore(Instruction *New, Instruction &Old) {
    assert(New && New->getParent() == 0 &&
           "New instruction already inserted into a basic block!");
    BasicBlock *BB = Old.getParent();
    BB->getInstList().insert(&Old, New);  // Insert inst
    Worklist.Add(New);
    return New;
  }

  // InsertNewInstWith - same as InsertNewInstBefore, but also sets the 
  // debug loc.
  //
  Instruction *InsertNewInstWith(Instruction *New, Instruction &Old) {
    New->setDebugLoc(Old.getDebugLoc());
    return InsertNewInstBefore(New, Old);
  }

  // ReplaceInstUsesWith - This method is to be used when an instruction is
  // found to be dead, replacable with another preexisting expression.  Here
  // we add all uses of I to the worklist, replace all uses of I with the new
  // value, then return I, so that the inst combiner will know that I was
  // modified.
  //
  Instruction *ReplaceInstUsesWith(Instruction &I, Value *V) {
    Worklist.AddUsersToWorkList(I);   // Add all modified instrs to worklist.
    
    // If we are replacing the instruction with itself, this must be in a
    // segment of unreachable code, so just clobber the instruction.
    if (&I == V) 
      V = UndefValue::get(I.getType());

    DEBUG(errs() << "IC: Replacing " << I << "\n"
                    "    with " << *V << '\n');

    I.replaceAllUsesWith(V);
    return &I;
  }

  // EraseInstFromFunction - When dealing with an instruction that has side
  // effects or produces a void value, we can't rely on DCE to delete the
  // instruction.  Instead, visit methods should return the value returned by
  // this function.
  Instruction *EraseInstFromFunction(Instruction &I) {
    DEBUG(errs() << "IC: ERASE " << I << '\n');

    assert(I.use_empty() && "Cannot erase instruction that is used!");
    // Make sure that we reprocess all operands now that we reduced their
    // use counts.
    if (I.getNumOperands() < 8) {
      for (User::op_iterator i = I.op_begin(), e = I.op_end(); i != e; ++i)
        if (Instruction *Op = dyn_cast<Instruction>(*i))
          Worklist.Add(Op);
    }
    Worklist.Remove(&I);
    I.eraseFromParent();
    MadeIRChange = true;
    return 0;  // Don't do anything with FI
  }
      
  void ComputeMaskedBits(Value *V, APInt &KnownZero,
                         APInt &KnownOne, unsigned Depth = 0) const {
    return llvm::ComputeMaskedBits(V, KnownZero, KnownOne, TD, Depth);
  }
  
  bool MaskedValueIsZero(Value *V, const APInt &Mask, 
                         unsigned Depth = 0) const {
    return llvm::MaskedValueIsZero(V, Mask, TD, Depth);
  }
  unsigned ComputeNumSignBits(Value *Op, unsigned Depth = 0) const {
    return llvm::ComputeNumSignBits(Op, TD, Depth);
  }

private:

  /// SimplifyAssociativeOrCommutative - This performs a few simplifications for
  /// operators which are associative or commutative.
  bool SimplifyAssociativeOrCommutative(BinaryOperator &I);

  /// SimplifyUsingDistributiveLaws - This tries to simplify binary operations
  /// which some other binary operation distributes over either by factorizing
  /// out common terms (eg "(A*B)+(A*C)" -> "A*(B+C)") or expanding out if this
  /// results in simplifications (eg: "A & (B | C) -> (A&B) | (A&C)" if this is
  /// a win).  Returns the simplified value, or null if it didn't simplify.
  Value *SimplifyUsingDistributiveLaws(BinaryOperator &I);

  /// SimplifyDemandedUseBits - Attempts to replace V with a simpler value
  /// based on the demanded bits.
  Value *SimplifyDemandedUseBits(Value *V, APInt DemandedMask, 
                                 APInt& KnownZero, APInt& KnownOne,
                                 unsigned Depth);
  bool SimplifyDemandedBits(Use &U, APInt DemandedMask, 
                            APInt& KnownZero, APInt& KnownOne,
                            unsigned Depth=0);
      
  /// SimplifyDemandedInstructionBits - Inst is an integer instruction that
  /// SimplifyDemandedBits knows about.  See if the instruction has any
  /// properties that allow us to simplify its operands.
  bool SimplifyDemandedInstructionBits(Instruction &Inst);
      
  Value *SimplifyDemandedVectorElts(Value *V, APInt DemandedElts,
                                    APInt& UndefElts, unsigned Depth = 0);
    
  // FoldOpIntoPhi - Given a binary operator, cast instruction, or select
  // which has a PHI node as operand #0, see if we can fold the instruction
  // into the PHI (which is only possible if all operands to the PHI are
  // constants).
  //
  Instruction *FoldOpIntoPhi(Instruction &I);

  // FoldPHIArgOpIntoPHI - If all operands to a PHI node are the same "unary"
  // operator and they all are only used by the PHI, PHI together their
  // inputs, and do the operation once, to the result of the PHI.
  Instruction *FoldPHIArgOpIntoPHI(PHINode &PN);
  Instruction *FoldPHIArgBinOpIntoPHI(PHINode &PN);
  Instruction *FoldPHIArgGEPIntoPHI(PHINode &PN);
  Instruction *FoldPHIArgLoadIntoPHI(PHINode &PN);

  
  Instruction *OptAndOp(Instruction *Op, ConstantInt *OpRHS,
                        ConstantInt *AndRHS, BinaryOperator &TheAnd);
  
  Value *FoldLogicalPlusAnd(Value *LHS, Value *RHS, ConstantInt *Mask,
                            bool isSub, Instruction &I);
  Value *InsertRangeTest(Value *V, Constant *Lo, Constant *Hi,
                         bool isSigned, bool Inside);
  Instruction *PromoteCastOfAllocation(BitCastInst &CI, AllocaInst &AI);
  Instruction *MatchBSwap(BinaryOperator &I);
  bool SimplifyStoreAtEndOfBlock(StoreInst &SI);
  Instruction *SimplifyMemTransfer(MemIntrinsic *MI);
  Instruction *SimplifyMemSet(MemSetInst *MI);


  Value *EvaluateInDifferentType(Value *V, Type *Ty, bool isSigned);

  /// Descale - Return a value X such that Val = X * Scale, or null if none.  If
  /// the multiplication is known not to overflow then NoSignedWrap is set.
  Value *Descale(Value *Val, APInt Scale, bool &NoSignedWrap);
};

      
  
} // end namespace llvm.

#endif
