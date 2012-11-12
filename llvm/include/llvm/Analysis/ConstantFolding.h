//===-- ConstantFolding.h - Fold instructions into constants --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares routines for folding instructions into constants when all
// operands are constants, for example "sub i32 1, 0" -> "1".
//
// Also, to supplement the basic VMCore ConstantExpr simplifications,
// this file declares some additional folding routines that can make use of
// DataLayout information. These functions cannot go in VMCore due to library
// dependency issues.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CONSTANTFOLDING_H
#define LLVM_ANALYSIS_CONSTANTFOLDING_H

namespace llvm {
  class Constant;
  class ConstantExpr;
  class Instruction;
  class DataLayout;
  class TargetLibraryInfo;
  class Function;
  class Type;
  template<typename T>
  class ArrayRef;

/// ConstantFoldInstruction - Try to constant fold the specified instruction.
/// If successful, the constant result is returned, if not, null is returned.
/// Note that this fails if not all of the operands are constant.  Otherwise,
/// this function can only fail when attempting to fold instructions like loads
/// and stores, which have no constant expression form.
Constant *ConstantFoldInstruction(Instruction *I, const DataLayout *TD = 0,
                                  const TargetLibraryInfo *TLI = 0);

/// ConstantFoldConstantExpression - Attempt to fold the constant expression
/// using the specified DataLayout.  If successful, the constant result is
/// result is returned, if not, null is returned.
Constant *ConstantFoldConstantExpression(const ConstantExpr *CE,
                                         const DataLayout *TD = 0,
                                         const TargetLibraryInfo *TLI = 0);

/// ConstantFoldInstOperands - Attempt to constant fold an instruction with the
/// specified operands.  If successful, the constant result is returned, if not,
/// null is returned.  Note that this function can fail when attempting to 
/// fold instructions like loads and stores, which have no constant expression 
/// form.
///
Constant *ConstantFoldInstOperands(unsigned Opcode, Type *DestTy,
                                   ArrayRef<Constant *> Ops,
                                   const DataLayout *TD = 0,
                                   const TargetLibraryInfo *TLI = 0);

/// ConstantFoldCompareInstOperands - Attempt to constant fold a compare
/// instruction (icmp/fcmp) with the specified operands.  If it fails, it
/// returns a constant expression of the specified operands.
///
Constant *ConstantFoldCompareInstOperands(unsigned Predicate,
                                          Constant *LHS, Constant *RHS,
                                          const DataLayout *TD = 0,
                                          const TargetLibraryInfo *TLI = 0);

/// ConstantFoldInsertValueInstruction - Attempt to constant fold an insertvalue
/// instruction with the specified operands and indices.  The constant result is
/// returned if successful; if not, null is returned.
Constant *ConstantFoldInsertValueInstruction(Constant *Agg, Constant *Val,
                                             ArrayRef<unsigned> Idxs);

/// ConstantFoldLoadFromConstPtr - Return the value that a load from C would
/// produce if it is constant and determinable.  If this is not determinable,
/// return null.
Constant *ConstantFoldLoadFromConstPtr(Constant *C, const DataLayout *TD = 0);

/// ConstantFoldLoadThroughGEPConstantExpr - Given a constant and a
/// getelementptr constantexpr, return the constant value being addressed by the
/// constant expression, or null if something is funny and we can't decide.
Constant *ConstantFoldLoadThroughGEPConstantExpr(Constant *C, ConstantExpr *CE);

/// ConstantFoldLoadThroughGEPIndices - Given a constant and getelementptr
/// indices (with an *implied* zero pointer index that is not in the list),
/// return the constant value being addressed by a virtual load, or null if
/// something is funny and we can't decide.
Constant *ConstantFoldLoadThroughGEPIndices(Constant *C,
                                            ArrayRef<Constant*> Indices);

/// canConstantFoldCallTo - Return true if its even possible to fold a call to
/// the specified function.
bool canConstantFoldCallTo(const Function *F);

/// ConstantFoldCall - Attempt to constant fold a call to the specified function
/// with the specified arguments, returning null if unsuccessful.
Constant *ConstantFoldCall(Function *F, ArrayRef<Constant *> Operands,
                           const TargetLibraryInfo *TLI = 0);
}

#endif
