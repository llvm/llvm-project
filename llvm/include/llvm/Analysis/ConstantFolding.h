//===-- ConstantFolding.h - Analyze constant folding possibilities --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of functions determines the possibility of performing constant
// folding.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CONSTANTFOLDING_H
#define LLVM_ANALYSIS_CONSTANTFOLDING_H

namespace llvm {
  class Constant;
  class ConstantExpr;
  class Instruction;
  class TargetData;
  class Function;
  class Type;

/// ConstantFoldInstruction - Attempt to constant fold the specified
/// instruction.  If successful, the constant result is returned, if not, null
/// is returned.  Note that this function can only fail when attempting to fold
/// instructions like loads and stores, which have no constant expression form.
///
Constant *ConstantFoldInstruction(Instruction *I, const TargetData *TD = 0);

/// ConstantFoldConstantExpression - Attempt to fold the constant expression
/// using the specified TargetData.  If successful, the constant result is
/// result is returned, if not, null is returned.
Constant *ConstantFoldConstantExpression(ConstantExpr *CE,
                                         const TargetData *TD);

/// ConstantFoldInstOperands - Attempt to constant fold an instruction with the
/// specified operands.  If successful, the constant result is returned, if not,
/// null is returned.  Note that this function can fail when attempting to 
/// fold instructions like loads and stores, which have no constant expression 
/// form.
///
Constant *ConstantFoldInstOperands(unsigned Opcode, const Type *DestTy,
                                   Constant*const * Ops, unsigned NumOps,
                                   const TargetData *TD = 0);

/// ConstantFoldCompareInstOperands - Attempt to constant fold a compare
/// instruction (icmp/fcmp) with the specified operands.  If it fails, it
/// returns a constant expression of the specified operands.
///
Constant *ConstantFoldCompareInstOperands(unsigned Predicate,
                                          Constant*const * Ops, unsigned NumOps,
                                          const TargetData *TD = 0);


/// ConstantFoldLoadThroughGEPConstantExpr - Given a constant and a
/// getelementptr constantexpr, return the constant value being addressed by the
/// constant expression, or null if something is funny and we can't decide.
Constant *ConstantFoldLoadThroughGEPConstantExpr(Constant *C, ConstantExpr *CE);
  
/// canConstantFoldCallTo - Return true if its even possible to fold a call to
/// the specified function.
bool canConstantFoldCallTo(const Function *F);

/// ConstantFoldCall - Attempt to constant fold a call to the specified function
/// with the specified arguments, returning null if unsuccessful.
Constant *
ConstantFoldCall(Function *F, Constant* const* Operands, unsigned NumOperands);
}

#endif
