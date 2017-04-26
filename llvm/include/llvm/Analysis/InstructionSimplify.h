//===-- InstructionSimplify.h - Fold instrs into simpler forms --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares routines for folding instructions into simpler forms
// that do not require creating new instructions.  This does constant folding
// ("add i32 1, 1" -> "2") but can also handle non-constant operands, either
// returning a constant ("and i32 %x, 0" -> "0") or an already existing value
// ("and i32 %x, %x" -> "%x").  If the simplification is also an instruction
// then it dominates the original instruction.
//
// These routines implicitly resolve undef uses. The easiest way to be safe when
// using these routines to obtain simplified values for existing instructions is
// to always replace all uses of the instructions with the resulting simplified
// values. This will prevent other code from seeing the same undef uses and
// resolving them to different values.
//
// These routines are designed to tolerate moderately incomplete IR, such as
// instructions that are not connected to basic blocks yet. However, they do
// require that all the IR that they encounter be valid. In particular, they
// require that all non-constant values be defined in the same function, and the
// same call context of that function (and not split between caller and callee
// contexts of a directly recursive call, for example).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_INSTRUCTIONSIMPLIFY_H
#define LLVM_ANALYSIS_INSTRUCTIONSIMPLIFY_H

#include "llvm/IR/User.h"

namespace llvm {
  template<typename T>
  class ArrayRef;
  class AssumptionCache;
  class DominatorTree;
  class Instruction;
  class DataLayout;
  class FastMathFlags;
  class OptimizationRemarkEmitter;
  class TargetLibraryInfo;
  class Type;
  class Value;

  struct SimplifyQuery {
    const DataLayout &DL;
    const TargetLibraryInfo *TLI = nullptr;
    const DominatorTree *DT = nullptr;
    AssumptionCache *AC = nullptr;
    const Instruction *CxtI = nullptr;
    SimplifyQuery(const DataLayout &DL) : DL(DL) {}

    SimplifyQuery(const DataLayout &DL, const TargetLibraryInfo *TLI,
                  const DominatorTree *DT, AssumptionCache *AC = nullptr,
                  const Instruction *CXTI = nullptr)
        : DL(DL), TLI(TLI), DT(DT), AC(AC), CxtI(CXTI) {}
  };

  // NOTE: the explicit multiple argument versions of these functions are
  // deprecated.
  // Please use the SimplifyQuery versions in new code.

  /// Given operands for an Add, fold the result or return null.
  Value *SimplifyAddInst(Value *LHS, Value *RHS, bool isNSW, bool isNUW,
                       const SimplifyQuery &Q);
  Value *SimplifyAddInst(Value *LHS, Value *RHS, bool isNSW, bool isNUW,
                         const DataLayout &DL,
                         const TargetLibraryInfo *TLI = nullptr,
                         const DominatorTree *DT = nullptr,
                         AssumptionCache *AC = nullptr,
                         const Instruction *CxtI = nullptr);

  /// Given operands for a Sub, fold the result or return null.
  Value *SimplifySubInst(Value *LHS, Value *RHS, bool isNSW, bool isNUW,
                         const SimplifyQuery &Q);
  Value *SimplifySubInst(Value *LHS, Value *RHS, bool isNSW, bool isNUW,
                         const DataLayout &DL,
                         const TargetLibraryInfo *TLI = nullptr,
                         const DominatorTree *DT = nullptr,
                         AssumptionCache *AC = nullptr,
                         const Instruction *CxtI = nullptr);

  /// Given operands for an FAdd, fold the result or return null.
  Value *SimplifyFAddInst(Value *LHS, Value *RHS, FastMathFlags FMF,
                          const SimplifyQuery &Q);
  Value *SimplifyFAddInst(Value *LHS, Value *RHS, FastMathFlags FMF,
                          const DataLayout &DL,
                          const TargetLibraryInfo *TLI = nullptr,
                          const DominatorTree *DT = nullptr,
                          AssumptionCache *AC = nullptr,
                          const Instruction *CxtI = nullptr);

  /// Given operands for an FSub, fold the result or return null.
  Value *SimplifyFSubInst(Value *LHS, Value *RHS, FastMathFlags FMF,
                          const SimplifyQuery &Q);
  Value *SimplifyFSubInst(Value *LHS, Value *RHS, FastMathFlags FMF,
                          const DataLayout &DL,
                          const TargetLibraryInfo *TLI = nullptr,
                          const DominatorTree *DT = nullptr,
                          AssumptionCache *AC = nullptr,
                          const Instruction *CxtI = nullptr);

  /// Given operands for an FMul, fold the result or return null.
  Value *SimplifyFMulInst(Value *LHS, Value *RHS, FastMathFlags FMF,
                          const SimplifyQuery &Q);
  Value *SimplifyFMulInst(Value *LHS, Value *RHS, FastMathFlags FMF,
                          const DataLayout &DL,
                          const TargetLibraryInfo *TLI = nullptr,
                          const DominatorTree *DT = nullptr,
                          AssumptionCache *AC = nullptr,
                          const Instruction *CxtI = nullptr);

  /// Given operands for a Mul, fold the result or return null.
  Value *SimplifyMulInst(Value *LHS, Value *RHS, const SimplifyQuery &Q);
  Value *SimplifyMulInst(Value *LHS, Value *RHS, const DataLayout &DL,
                         const TargetLibraryInfo *TLI = nullptr,
                         const DominatorTree *DT = nullptr,
                         AssumptionCache *AC = nullptr,
                         const Instruction *CxtI = nullptr);

  /// Given operands for an SDiv, fold the result or return null.
  Value *SimplifySDivInst(Value *LHS, Value *RHS, const SimplifyQuery &Q);
  Value *SimplifySDivInst(Value *LHS, Value *RHS, const DataLayout &DL,
                          const TargetLibraryInfo *TLI = nullptr,
                          const DominatorTree *DT = nullptr,
                          AssumptionCache *AC = nullptr,
                          const Instruction *CxtI = nullptr);

  /// Given operands for a UDiv, fold the result or return null.
  Value *SimplifyUDivInst(Value *LHS, Value *RHS, const SimplifyQuery &Q);
  Value *SimplifyUDivInst(Value *LHS, Value *RHS, const DataLayout &DL,
                          const TargetLibraryInfo *TLI = nullptr,
                          const DominatorTree *DT = nullptr,
                          AssumptionCache *AC = nullptr,
                          const Instruction *CxtI = nullptr);

  /// Given operands for an FDiv, fold the result or return null.
  Value *SimplifyFDivInst(Value *LHS, Value *RHS, FastMathFlags FMF,
                          const SimplifyQuery &Q);
  Value *SimplifyFDivInst(Value *LHS, Value *RHS, FastMathFlags FMF,
                          const DataLayout &DL,
                          const TargetLibraryInfo *TLI = nullptr,
                          const DominatorTree *DT = nullptr,
                          AssumptionCache *AC = nullptr,
                          const Instruction *CxtI = nullptr);

  /// Given operands for an SRem, fold the result or return null.
  Value *SimplifySRemInst(Value *LHS, Value *RHS, const SimplifyQuery &Q);
  Value *SimplifySRemInst(Value *LHS, Value *RHS, const DataLayout &DL,
                          const TargetLibraryInfo *TLI = nullptr,
                          const DominatorTree *DT = nullptr,
                          AssumptionCache *AC = nullptr,
                          const Instruction *CxtI = nullptr);

  /// Given operands for a URem, fold the result or return null.
  Value *SimplifyURemInst(Value *LHS, Value *RHS, const SimplifyQuery &Q);
  Value *SimplifyURemInst(Value *LHS, Value *RHS, const DataLayout &DL,
                          const TargetLibraryInfo *TLI = nullptr,
                          const DominatorTree *DT = nullptr,
                          AssumptionCache *AC = nullptr,
                          const Instruction *CxtI = nullptr);

  /// Given operands for an FRem, fold the result or return null.
  Value *SimplifyFRemInst(Value *LHS, Value *RHS, FastMathFlags FMF,
                          const SimplifyQuery &Q);
  Value *SimplifyFRemInst(Value *LHS, Value *RHS, FastMathFlags FMF,
                          const DataLayout &DL,
                          const TargetLibraryInfo *TLI = nullptr,
                          const DominatorTree *DT = nullptr,
                          AssumptionCache *AC = nullptr,
                          const Instruction *CxtI = nullptr);

  /// Given operands for a Shl, fold the result or return null.
  Value *SimplifyShlInst(Value *Op0, Value *Op1, bool isNSW, bool isNUW,
                         const SimplifyQuery &Q);
  Value *SimplifyShlInst(Value *Op0, Value *Op1, bool isNSW, bool isNUW,
                         const DataLayout &DL,
                         const TargetLibraryInfo *TLI = nullptr,
                         const DominatorTree *DT = nullptr,
                         AssumptionCache *AC = nullptr,
                         const Instruction *CxtI = nullptr);

  /// Given operands for a LShr, fold the result or return null.
  Value *SimplifyLShrInst(Value *Op0, Value *Op1, bool isExact,
                          const SimplifyQuery &Q);
  Value *SimplifyLShrInst(Value *Op0, Value *Op1, bool isExact,
                          const DataLayout &DL,
                          const TargetLibraryInfo *TLI = nullptr,
                          const DominatorTree *DT = nullptr,
                          AssumptionCache *AC = nullptr,
                          const Instruction *CxtI = nullptr);

  /// Given operands for a AShr, fold the result or return nulll.
  Value *SimplifyAShrInst(Value *Op0, Value *Op1, bool isExact,
                          const SimplifyQuery &Q);
  Value *SimplifyAShrInst(Value *Op0, Value *Op1, bool isExact,
                          const DataLayout &DL,
                          const TargetLibraryInfo *TLI = nullptr,
                          const DominatorTree *DT = nullptr,
                          AssumptionCache *AC = nullptr,
                          const Instruction *CxtI = nullptr);

  /// Given operands for an And, fold the result or return null.
  Value *SimplifyAndInst(Value *LHS, Value *RHS, const SimplifyQuery &Q);
  Value *SimplifyAndInst(Value *LHS, Value *RHS, const DataLayout &DL,
                         const TargetLibraryInfo *TLI = nullptr,
                         const DominatorTree *DT = nullptr,
                         AssumptionCache *AC = nullptr,
                         const Instruction *CxtI = nullptr);

  /// Given operands for an Or, fold the result or return null.
  Value *SimplifyOrInst(Value *LHS, Value *RHS, const SimplifyQuery &Q);
  Value *SimplifyOrInst(Value *LHS, Value *RHS, const DataLayout &DL,
                        const TargetLibraryInfo *TLI = nullptr,
                        const DominatorTree *DT = nullptr,
                        AssumptionCache *AC = nullptr,
                        const Instruction *CxtI = nullptr);

  /// Given operands for an Xor, fold the result or return null.
  Value *SimplifyXorInst(Value *LHS, Value *RHS, const SimplifyQuery &Q);
  Value *SimplifyXorInst(Value *LHS, Value *RHS, const DataLayout &DL,
                         const TargetLibraryInfo *TLI = nullptr,
                         const DominatorTree *DT = nullptr,
                         AssumptionCache *AC = nullptr,
                         const Instruction *CxtI = nullptr);

  /// Given operands for an ICmpInst, fold the result or return null.
  Value *SimplifyICmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                          const SimplifyQuery &Q);
  Value *SimplifyICmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                          const DataLayout &DL,
                          const TargetLibraryInfo *TLI = nullptr,
                          const DominatorTree *DT = nullptr,
                          AssumptionCache *AC = nullptr,
                          const Instruction *CxtI = nullptr);

  /// Given operands for an FCmpInst, fold the result or return null.
  Value *SimplifyFCmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                          FastMathFlags FMF, const SimplifyQuery &Q);
  Value *SimplifyFCmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                          FastMathFlags FMF, const DataLayout &DL,
                          const TargetLibraryInfo *TLI = nullptr,
                          const DominatorTree *DT = nullptr,
                          AssumptionCache *AC = nullptr,
                          const Instruction *CxtI = nullptr);

  /// Given operands for a SelectInst, fold the result or return null.
  Value *SimplifySelectInst(Value *Cond, Value *TrueVal, Value *FalseVal,
                            const SimplifyQuery &Q);
  Value *SimplifySelectInst(Value *Cond, Value *TrueVal, Value *FalseVal,
                            const DataLayout &DL,
                            const TargetLibraryInfo *TLI = nullptr,
                            const DominatorTree *DT = nullptr,
                            AssumptionCache *AC = nullptr,
                            const Instruction *CxtI = nullptr);

  /// Given operands for a GetElementPtrInst, fold the result or return null. 
  Value *SimplifyGEPInst(Type *SrcTy, ArrayRef<Value *> Ops,
                         const SimplifyQuery &Q);
  Value *SimplifyGEPInst(Type *SrcTy, ArrayRef<Value *> Ops,
                         const DataLayout &DL,
                         const TargetLibraryInfo *TLI = nullptr,
                         const DominatorTree *DT = nullptr,
                         AssumptionCache *AC = nullptr,
                         const Instruction *CxtI = nullptr);

  /// Given operands for an InsertValueInst, fold the result or return null.
  Value *SimplifyInsertValueInst(Value *Agg, Value *Val,
                                 ArrayRef<unsigned> Idxs,
                                 const SimplifyQuery &Q);
  Value *SimplifyInsertValueInst(Value *Agg, Value *Val,
                                 ArrayRef<unsigned> Idxs, const DataLayout &DL,
                                 const TargetLibraryInfo *TLI = nullptr,
                                 const DominatorTree *DT = nullptr,
                                 AssumptionCache *AC = nullptr,
                                 const Instruction *CxtI = nullptr);

  /// Given operands for an ExtractValueInst, fold the result or return null.
  Value *SimplifyExtractValueInst(Value *Agg, ArrayRef<unsigned> Idxs,
                                  const SimplifyQuery &Q);
  Value *SimplifyExtractValueInst(Value *Agg, ArrayRef<unsigned> Idxs,
                                  const DataLayout &DL,
                                  const TargetLibraryInfo *TLI = nullptr,
                                  const DominatorTree *DT = nullptr,
                                  AssumptionCache *AC = nullptr,
                                  const Instruction *CxtI = nullptr);

  /// Given operands for an ExtractElementInst, fold the result or return null.
  Value *SimplifyExtractElementInst(Value *Vec, Value *Idx,
                                    const SimplifyQuery &Q);
  Value *SimplifyExtractElementInst(Value *Vec, Value *Idx,
                                    const DataLayout &DL,
                                    const TargetLibraryInfo *TLI = nullptr,
                                    const DominatorTree *DT = nullptr,
                                    AssumptionCache *AC = nullptr,
                                    const Instruction *CxtI = nullptr);

  /// Given operands for a CastInst, fold the result or return null.
  Value *SimplifyCastInst(unsigned CastOpc, Value *Op, Type *Ty,
                          const SimplifyQuery &Q);
  Value *SimplifyCastInst(unsigned CastOpc, Value *Op, Type *Ty,
                          const DataLayout &DL,
                          const TargetLibraryInfo *TLI = nullptr,
                          const DominatorTree *DT = nullptr,
                          AssumptionCache *AC = nullptr,
                          const Instruction *CxtI = nullptr);

  /// Given operands for a ShuffleVectorInst, fold the result or return null.
  Value *SimplifyShuffleVectorInst(Value *Op0, Value *Op1, Constant *Mask,
                                   Type *RetTy, const SimplifyQuery &Q);
  Value *SimplifyShuffleVectorInst(Value *Op0, Value *Op1, Constant *Mask,
                                   Type *RetTy, const DataLayout &DL,
                                   const TargetLibraryInfo *TLI = nullptr,
                                   const DominatorTree *DT = nullptr,
                                   AssumptionCache *AC = nullptr,
                                   const Instruction *CxtI = nullptr);

  //=== Helper functions for higher up the class hierarchy.


  /// Given operands for a CmpInst, fold the result or return null.
  Value *SimplifyCmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                         const SimplifyQuery &Q);
  Value *SimplifyCmpInst(unsigned Predicate, Value *LHS, Value *RHS,
                         const DataLayout &DL,
                         const TargetLibraryInfo *TLI = nullptr,
                         const DominatorTree *DT = nullptr,
                         AssumptionCache *AC = nullptr,
                         const Instruction *CxtI = nullptr);

  /// Given operands for a BinaryOperator, fold the result or return null.
  Value *SimplifyBinOp(unsigned Opcode, Value *LHS, Value *RHS,
                       const SimplifyQuery &Q);
  Value *SimplifyBinOp(unsigned Opcode, Value *LHS, Value *RHS,
                       const DataLayout &DL,
                       const TargetLibraryInfo *TLI = nullptr,
                       const DominatorTree *DT = nullptr,
                       AssumptionCache *AC = nullptr,
                       const Instruction *CxtI = nullptr);

  /// Given operands for an FP BinaryOperator, fold the result or return null.
  /// In contrast to SimplifyBinOp, try to use FastMathFlag when folding the
  /// result. In case we don't need FastMathFlags, simply fall to SimplifyBinOp.
  Value *SimplifyFPBinOp(unsigned Opcode, Value *LHS, Value *RHS,
                         FastMathFlags FMF, const SimplifyQuery &Q);
  Value *SimplifyFPBinOp(unsigned Opcode, Value *LHS, Value *RHS,
                         FastMathFlags FMF, const DataLayout &DL,
                         const TargetLibraryInfo *TLI = nullptr,
                         const DominatorTree *DT = nullptr,
                         AssumptionCache *AC = nullptr,
                         const Instruction *CxtI = nullptr);

  /// Given a function and iterators over arguments, fold the result or return
  /// null.
  Value *SimplifyCall(Value *V, User::op_iterator ArgBegin,
                      User::op_iterator ArgEnd, const SimplifyQuery &Q);
  Value *SimplifyCall(Value *V, User::op_iterator ArgBegin,
                      User::op_iterator ArgEnd, const DataLayout &DL,
                      const TargetLibraryInfo *TLI = nullptr,
                      const DominatorTree *DT = nullptr,
                      AssumptionCache *AC = nullptr,
                      const Instruction *CxtI = nullptr);

  /// Given a function and set of arguments, fold the result or return null.
  Value *SimplifyCall(Value *V, ArrayRef<Value *> Args, const SimplifyQuery &Q);
  Value *SimplifyCall(Value *V, ArrayRef<Value *> Args, const DataLayout &DL,
                      const TargetLibraryInfo *TLI = nullptr,
                      const DominatorTree *DT = nullptr,
                      AssumptionCache *AC = nullptr,
                      const Instruction *CxtI = nullptr);

  /// See if we can compute a simplified version of this instruction. If not,
  /// return null.
  Value *SimplifyInstruction(Instruction *I, const SimplifyQuery &Q,
                             OptimizationRemarkEmitter *ORE = nullptr);
  Value *SimplifyInstruction(Instruction *I, const DataLayout &DL,
                             const TargetLibraryInfo *TLI = nullptr,
                             const DominatorTree *DT = nullptr,
                             AssumptionCache *AC = nullptr,
                             OptimizationRemarkEmitter *ORE = nullptr);

  /// Replace all uses of 'I' with 'SimpleV' and simplify the uses recursively.
  ///
  /// This first performs a normal RAUW of I with SimpleV. It then recursively
  /// attempts to simplify those users updated by the operation. The 'I'
  /// instruction must not be equal to the simplified value 'SimpleV'.
  ///
  /// The function returns true if any simplifications were performed.
  bool replaceAndRecursivelySimplify(Instruction *I, Value *SimpleV,
                                     const TargetLibraryInfo *TLI = nullptr,
                                     const DominatorTree *DT = nullptr,
                                     AssumptionCache *AC = nullptr);

  /// Recursively attempt to simplify an instruction.
  ///
  /// This routine uses SimplifyInstruction to simplify 'I', and if successful
  /// replaces uses of 'I' with the simplified value. It then recurses on each
  /// of the users impacted. It returns true if any simplifications were
  /// performed.
  bool recursivelySimplifyInstruction(Instruction *I,
                                      const TargetLibraryInfo *TLI = nullptr,
                                      const DominatorTree *DT = nullptr,
                                      AssumptionCache *AC = nullptr);
} // end namespace llvm

#endif

