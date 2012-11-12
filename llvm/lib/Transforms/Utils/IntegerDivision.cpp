//===-- IntegerDivision.cpp - Expand integer division ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains an implementation of 32bit scalar integer division for
// targets that don't have native support. It's largely derived from
// compiler-rt's implementation of __udivsi3, but hand-tuned to reduce the
// amount of control flow
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "integer-division"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/IRBuilder.h"
#include "llvm/Transforms/Utils/IntegerDivision.h"

using namespace llvm;

/// Generate code to compute the remainder of two signed integers. Returns the
/// remainder, which will have the sign of the dividend. Builder's insert point
/// should be pointing where the caller wants code generated, e.g. at the srem
/// instruction. This will generate a urem in the process, and Builder's insert
/// point will be pointing at the uren (if present, i.e. not folded), ready to
/// be expanded if the user wishes
static Value *generateSignedRemainderCode(Value *Dividend, Value *Divisor,
                                          IRBuilder<> &Builder) {
  ConstantInt *ThirtyOne = Builder.getInt32(31);

  // ;   %dividend_sgn = ashr i32 %dividend, 31
  // ;   %divisor_sgn  = ashr i32 %divisor, 31
  // ;   %dvd_xor      = xor i32 %dividend, %dividend_sgn
  // ;   %dvs_xor      = xor i32 %divisor, %divisor_sgn
  // ;   %u_dividend   = sub i32 %dvd_xor, %dividend_sgn
  // ;   %u_divisor    = sub i32 %dvs_xor, %divisor_sgn
  // ;   %urem         = urem i32 %dividend, %divisor
  // ;   %xored        = xor i32 %urem, %dividend_sgn
  // ;   %srem         = sub i32 %xored, %dividend_sgn
  Value *DividendSign = Builder.CreateAShr(Dividend, ThirtyOne);
  Value *DivisorSign  = Builder.CreateAShr(Divisor, ThirtyOne);
  Value *DvdXor       = Builder.CreateXor(Dividend, DividendSign);
  Value *DvsXor       = Builder.CreateXor(Divisor, DivisorSign);
  Value *UDividend    = Builder.CreateSub(DvdXor, DividendSign);
  Value *UDivisor     = Builder.CreateSub(DvsXor, DivisorSign);
  Value *URem         = Builder.CreateURem(UDividend, UDivisor);
  Value *Xored        = Builder.CreateXor(URem, DividendSign);
  Value *SRem         = Builder.CreateSub(Xored, DividendSign);

  if (Instruction *URemInst = dyn_cast<Instruction>(URem))
    Builder.SetInsertPoint(URemInst);

  return SRem;
}


/// Generate code to compute the remainder of two unsigned integers. Returns the
/// remainder. Builder's insert point should be pointing where the caller wants
/// code generated, e.g. at the urem instruction. This will generate a udiv in
/// the process, and Builder's insert point will be pointing at the udiv (if
/// present, i.e. not folded), ready to be expanded if the user wishes
static Value *generatedUnsignedRemainderCode(Value *Dividend, Value *Divisor,
                                             IRBuilder<> &Builder) {
  // Remainder = Dividend - Quotient*Divisor

  // ;   %quotient  = udiv i32 %dividend, %divisor
  // ;   %product   = mul i32 %divisor, %quotient
  // ;   %remainder = sub i32 %dividend, %product
  Value *Quotient  = Builder.CreateUDiv(Dividend, Divisor);
  Value *Product   = Builder.CreateMul(Divisor, Quotient);
  Value *Remainder = Builder.CreateSub(Dividend, Product);

  if (Instruction *UDiv = dyn_cast<Instruction>(Quotient))
    Builder.SetInsertPoint(UDiv);

  return Remainder;
}

/// Generate code to divide two signed integers. Returns the quotient, rounded
/// towards 0. Builder's insert point should be pointing where the caller wants
/// code generated, e.g. at the sdiv instruction. This will generate a udiv in
/// the process, and Builder's insert point will be pointing at the udiv (if
/// present, i.e. not folded), ready to be expanded if the user wishes.
static Value *generateSignedDivisionCode(Value *Dividend, Value *Divisor,
                                         IRBuilder<> &Builder) {
  // Implementation taken from compiler-rt's __divsi3

  ConstantInt *ThirtyOne = Builder.getInt32(31);

  // ;   %tmp    = ashr i32 %dividend, 31
  // ;   %tmp1   = ashr i32 %divisor, 31
  // ;   %tmp2   = xor i32 %tmp, %dividend
  // ;   %u_dvnd = sub nsw i32 %tmp2, %tmp
  // ;   %tmp3   = xor i32 %tmp1, %divisor
  // ;   %u_dvsr = sub nsw i32 %tmp3, %tmp1
  // ;   %q_sgn  = xor i32 %tmp1, %tmp
  // ;   %q_mag  = udiv i32 %u_dvnd, %u_dvsr
  // ;   %tmp4   = xor i32 %q_mag, %q_sgn
  // ;   %q      = sub i32 %tmp4, %q_sgn
  Value *Tmp    = Builder.CreateAShr(Dividend, ThirtyOne);
  Value *Tmp1   = Builder.CreateAShr(Divisor, ThirtyOne);
  Value *Tmp2   = Builder.CreateXor(Tmp, Dividend);
  Value *U_Dvnd = Builder.CreateSub(Tmp2, Tmp);
  Value *Tmp3   = Builder.CreateXor(Tmp1, Divisor);
  Value *U_Dvsr = Builder.CreateSub(Tmp3, Tmp1);
  Value *Q_Sgn  = Builder.CreateXor(Tmp1, Tmp);
  Value *Q_Mag  = Builder.CreateUDiv(U_Dvnd, U_Dvsr);
  Value *Tmp4   = Builder.CreateXor(Q_Mag, Q_Sgn);
  Value *Q      = Builder.CreateSub(Tmp4, Q_Sgn);

  if (Instruction *UDiv = dyn_cast<Instruction>(Q_Mag))
    Builder.SetInsertPoint(UDiv);

  return Q;
}

/// Generates code to divide two unsigned scalar 32-bit integers. Returns the
/// quotient, rounded towards 0. Builder's insert point should be pointing where
/// the caller wants code generated, e.g. at the udiv instruction.
static Value *generateUnsignedDivisionCode(Value *Dividend, Value *Divisor,
                                           IRBuilder<> &Builder) {
  // The basic algorithm can be found in the compiler-rt project's
  // implementation of __udivsi3.c. Here, we do a lower-level IR based approach
  // that's been hand-tuned to lessen the amount of control flow involved.

  // Some helper values
  IntegerType *I32Ty = Builder.getInt32Ty();

  ConstantInt *Zero      = Builder.getInt32(0);
  ConstantInt *One       = Builder.getInt32(1);
  ConstantInt *ThirtyOne = Builder.getInt32(31);
  ConstantInt *NegOne    = ConstantInt::getSigned(I32Ty, -1);
  ConstantInt *True      = Builder.getTrue();

  BasicBlock *IBB = Builder.GetInsertBlock();
  Function *F = IBB->getParent();
  Function *CTLZi32 = Intrinsic::getDeclaration(F->getParent(), Intrinsic::ctlz,
                                                I32Ty);

  // Our CFG is going to look like:
  // +---------------------+
  // | special-cases       |
  // |   ...               |
  // +---------------------+
  //  |       |
  //  |   +----------+
  //  |   |  bb1     |
  //  |   |  ...     |
  //  |   +----------+
  //  |    |      |
  //  |    |  +------------+
  //  |    |  |  preheader |
  //  |    |  |  ...       |
  //  |    |  +------------+
  //  |    |      |
  //  |    |      |      +---+
  //  |    |      |      |   |
  //  |    |  +------------+ |
  //  |    |  |  do-while  | |
  //  |    |  |  ...       | |
  //  |    |  +------------+ |
  //  |    |      |      |   |
  //  |   +-----------+  +---+
  //  |   | loop-exit |
  //  |   |  ...      |
  //  |   +-----------+
  //  |     |
  // +-------+
  // | ...   |
  // | end   |
  // +-------+
  BasicBlock *SpecialCases = Builder.GetInsertBlock();
  SpecialCases->setName(Twine(SpecialCases->getName(), "_udiv-special-cases"));
  BasicBlock *End = SpecialCases->splitBasicBlock(Builder.GetInsertPoint(),
                                                  "udiv-end");
  BasicBlock *LoopExit  = BasicBlock::Create(Builder.getContext(),
                                             "udiv-loop-exit", F, End);
  BasicBlock *DoWhile   = BasicBlock::Create(Builder.getContext(),
                                             "udiv-do-while", F, End);
  BasicBlock *Preheader = BasicBlock::Create(Builder.getContext(),
                                             "udiv-preheader", F, End);
  BasicBlock *BB1       = BasicBlock::Create(Builder.getContext(),
                                             "udiv-bb1", F, End);

  // We'll be overwriting the terminator to insert our extra blocks
  SpecialCases->getTerminator()->eraseFromParent();

  // First off, check for special cases: dividend or divisor is zero, divisor
  // is greater than dividend, and divisor is 1.
  // ; special-cases:
  // ;   %ret0_1      = icmp eq i32 %divisor, 0
  // ;   %ret0_2      = icmp eq i32 %dividend, 0
  // ;   %ret0_3      = or i1 %ret0_1, %ret0_2
  // ;   %tmp0        = tail call i32 @llvm.ctlz.i32(i32 %divisor, i1 true)
  // ;   %tmp1        = tail call i32 @llvm.ctlz.i32(i32 %dividend, i1 true)
  // ;   %sr          = sub nsw i32 %tmp0, %tmp1
  // ;   %ret0_4      = icmp ugt i32 %sr, 31
  // ;   %ret0        = or i1 %ret0_3, %ret0_4
  // ;   %retDividend = icmp eq i32 %sr, 31
  // ;   %retVal      = select i1 %ret0, i32 0, i32 %dividend
  // ;   %earlyRet    = or i1 %ret0, %retDividend
  // ;   br i1 %earlyRet, label %end, label %bb1
  Builder.SetInsertPoint(SpecialCases);
  Value *Ret0_1      = Builder.CreateICmpEQ(Divisor, Zero);
  Value *Ret0_2      = Builder.CreateICmpEQ(Dividend, Zero);
  Value *Ret0_3      = Builder.CreateOr(Ret0_1, Ret0_2);
  Value *Tmp0        = Builder.CreateCall2(CTLZi32, Divisor, True);
  Value *Tmp1        = Builder.CreateCall2(CTLZi32, Dividend, True);
  Value *SR          = Builder.CreateSub(Tmp0, Tmp1);
  Value *Ret0_4      = Builder.CreateICmpUGT(SR, ThirtyOne);
  Value *Ret0        = Builder.CreateOr(Ret0_3, Ret0_4);
  Value *RetDividend = Builder.CreateICmpEQ(SR, ThirtyOne);
  Value *RetVal      = Builder.CreateSelect(Ret0, Zero, Dividend);
  Value *EarlyRet    = Builder.CreateOr(Ret0, RetDividend);
  Builder.CreateCondBr(EarlyRet, End, BB1);

  // ; bb1:                                             ; preds = %special-cases
  // ;   %sr_1     = add i32 %sr, 1
  // ;   %tmp2     = sub i32 31, %sr
  // ;   %q        = shl i32 %dividend, %tmp2
  // ;   %skipLoop = icmp eq i32 %sr_1, 0
  // ;   br i1 %skipLoop, label %loop-exit, label %preheader
  Builder.SetInsertPoint(BB1);
  Value *SR_1     = Builder.CreateAdd(SR, One);
  Value *Tmp2     = Builder.CreateSub(ThirtyOne, SR);
  Value *Q        = Builder.CreateShl(Dividend, Tmp2);
  Value *SkipLoop = Builder.CreateICmpEQ(SR_1, Zero);
  Builder.CreateCondBr(SkipLoop, LoopExit, Preheader);

  // ; preheader:                                           ; preds = %bb1
  // ;   %tmp3 = lshr i32 %dividend, %sr_1
  // ;   %tmp4 = add i32 %divisor, -1
  // ;   br label %do-while
  Builder.SetInsertPoint(Preheader);
  Value *Tmp3 = Builder.CreateLShr(Dividend, SR_1);
  Value *Tmp4 = Builder.CreateAdd(Divisor, NegOne);
  Builder.CreateBr(DoWhile);

  // ; do-while:                                 ; preds = %do-while, %preheader
  // ;   %carry_1 = phi i32 [ 0, %preheader ], [ %carry, %do-while ]
  // ;   %sr_3    = phi i32 [ %sr_1, %preheader ], [ %sr_2, %do-while ]
  // ;   %r_1     = phi i32 [ %tmp3, %preheader ], [ %r, %do-while ]
  // ;   %q_2     = phi i32 [ %q, %preheader ], [ %q_1, %do-while ]
  // ;   %tmp5  = shl i32 %r_1, 1
  // ;   %tmp6  = lshr i32 %q_2, 31
  // ;   %tmp7  = or i32 %tmp5, %tmp6
  // ;   %tmp8  = shl i32 %q_2, 1
  // ;   %q_1   = or i32 %carry_1, %tmp8
  // ;   %tmp9  = sub i32 %tmp4, %tmp7
  // ;   %tmp10 = ashr i32 %tmp9, 31
  // ;   %carry = and i32 %tmp10, 1
  // ;   %tmp11 = and i32 %tmp10, %divisor
  // ;   %r     = sub i32 %tmp7, %tmp11
  // ;   %sr_2  = add i32 %sr_3, -1
  // ;   %tmp12 = icmp eq i32 %sr_2, 0
  // ;   br i1 %tmp12, label %loop-exit, label %do-while
  Builder.SetInsertPoint(DoWhile);
  PHINode *Carry_1 = Builder.CreatePHI(I32Ty, 2);
  PHINode *SR_3    = Builder.CreatePHI(I32Ty, 2);
  PHINode *R_1     = Builder.CreatePHI(I32Ty, 2);
  PHINode *Q_2     = Builder.CreatePHI(I32Ty, 2);
  Value *Tmp5  = Builder.CreateShl(R_1, One);
  Value *Tmp6  = Builder.CreateLShr(Q_2, ThirtyOne);
  Value *Tmp7  = Builder.CreateOr(Tmp5, Tmp6);
  Value *Tmp8  = Builder.CreateShl(Q_2, One);
  Value *Q_1   = Builder.CreateOr(Carry_1, Tmp8);
  Value *Tmp9  = Builder.CreateSub(Tmp4, Tmp7);
  Value *Tmp10 = Builder.CreateAShr(Tmp9, 31);
  Value *Carry = Builder.CreateAnd(Tmp10, One);
  Value *Tmp11 = Builder.CreateAnd(Tmp10, Divisor);
  Value *R     = Builder.CreateSub(Tmp7, Tmp11);
  Value *SR_2  = Builder.CreateAdd(SR_3, NegOne);
  Value *Tmp12 = Builder.CreateICmpEQ(SR_2, Zero);
  Builder.CreateCondBr(Tmp12, LoopExit, DoWhile);

  // ; loop-exit:                                      ; preds = %do-while, %bb1
  // ;   %carry_2 = phi i32 [ 0, %bb1 ], [ %carry, %do-while ]
  // ;   %q_3     = phi i32 [ %q, %bb1 ], [ %q_1, %do-while ]
  // ;   %tmp13 = shl i32 %q_3, 1
  // ;   %q_4   = or i32 %carry_2, %tmp13
  // ;   br label %end
  Builder.SetInsertPoint(LoopExit);
  PHINode *Carry_2 = Builder.CreatePHI(I32Ty, 2);
  PHINode *Q_3     = Builder.CreatePHI(I32Ty, 2);
  Value *Tmp13 = Builder.CreateShl(Q_3, One);
  Value *Q_4   = Builder.CreateOr(Carry_2, Tmp13);
  Builder.CreateBr(End);

  // ; end:                                 ; preds = %loop-exit, %special-cases
  // ;   %q_5 = phi i32 [ %q_4, %loop-exit ], [ %retVal, %special-cases ]
  // ;   ret i32 %q_5
  Builder.SetInsertPoint(End, End->begin());
  PHINode *Q_5 = Builder.CreatePHI(I32Ty, 2);

  // Populate the Phis, since all values have now been created. Our Phis were:
  // ;   %carry_1 = phi i32 [ 0, %preheader ], [ %carry, %do-while ]
  Carry_1->addIncoming(Zero, Preheader);
  Carry_1->addIncoming(Carry, DoWhile);
  // ;   %sr_3 = phi i32 [ %sr_1, %preheader ], [ %sr_2, %do-while ]
  SR_3->addIncoming(SR_1, Preheader);
  SR_3->addIncoming(SR_2, DoWhile);
  // ;   %r_1 = phi i32 [ %tmp3, %preheader ], [ %r, %do-while ]
  R_1->addIncoming(Tmp3, Preheader);
  R_1->addIncoming(R, DoWhile);
  // ;   %q_2 = phi i32 [ %q, %preheader ], [ %q_1, %do-while ]
  Q_2->addIncoming(Q, Preheader);
  Q_2->addIncoming(Q_1, DoWhile);
  // ;   %carry_2 = phi i32 [ 0, %bb1 ], [ %carry, %do-while ]
  Carry_2->addIncoming(Zero, BB1);
  Carry_2->addIncoming(Carry, DoWhile);
  // ;   %q_3 = phi i32 [ %q, %bb1 ], [ %q_1, %do-while ]
  Q_3->addIncoming(Q, BB1);
  Q_3->addIncoming(Q_1, DoWhile);
  // ;   %q_5 = phi i32 [ %q_4, %loop-exit ], [ %retVal, %special-cases ]
  Q_5->addIncoming(Q_4, LoopExit);
  Q_5->addIncoming(RetVal, SpecialCases);

  return Q_5;
}

/// Generate code to calculate the remainder of two integers, replacing Rem with
/// the generated code. This currently generates code using the udiv expansion,
/// but future work includes generating more specialized code, e.g. when more
/// information about the operands are known. Currently only implements 32bit
/// scalar division (due to udiv's limitation), but future work is removing this
/// limitation.
///
/// @brief Replace Rem with generated code.
bool llvm::expandRemainder(BinaryOperator *Rem) {
  assert((Rem->getOpcode() == Instruction::SRem ||
          Rem->getOpcode() == Instruction::URem) &&
         "Trying to expand remainder from a non-remainder function");

  IRBuilder<> Builder(Rem);

  // First prepare the sign if it's a signed remainder
  if (Rem->getOpcode() == Instruction::SRem) {
    Value *Remainder = generateSignedRemainderCode(Rem->getOperand(0),
                                                   Rem->getOperand(1), Builder);

    Rem->replaceAllUsesWith(Remainder);
    Rem->dropAllReferences();
    Rem->eraseFromParent();

    // If we didn't actually generate a udiv instruction, we're done
    BinaryOperator *BO = dyn_cast<BinaryOperator>(Builder.GetInsertPoint());
    if (!BO || BO->getOpcode() != Instruction::URem)
      return true;

    Rem = BO;
  }

  Value *Remainder = generatedUnsignedRemainderCode(Rem->getOperand(0),
                                                    Rem->getOperand(1),
                                                    Builder);

  Rem->replaceAllUsesWith(Remainder);
  Rem->dropAllReferences();
  Rem->eraseFromParent();

  // Expand the udiv
  if (BinaryOperator *UDiv = dyn_cast<BinaryOperator>(Builder.GetInsertPoint())) {
    assert(UDiv->getOpcode() == Instruction::UDiv && "Non-udiv in expansion?");
    expandDivision(UDiv);
  }

  return true;
}


/// Generate code to divide two integers, replacing Div with the generated
/// code. This currently generates code similarly to compiler-rt's
/// implementations, but future work includes generating more specialized code
/// when more information about the operands are known. Currently only
/// implements 32bit scalar division, but future work is removing this
/// limitation.
///
/// @brief Replace Div with generated code.
bool llvm::expandDivision(BinaryOperator *Div) {
  assert((Div->getOpcode() == Instruction::SDiv ||
          Div->getOpcode() == Instruction::UDiv) &&
         "Trying to expand division from a non-division function");

  IRBuilder<> Builder(Div);

  if (Div->getType()->isVectorTy())
    llvm_unreachable("Div over vectors not supported");

  // First prepare the sign if it's a signed division
  if (Div->getOpcode() == Instruction::SDiv) {
    // Lower the code to unsigned division, and reset Div to point to the udiv.
    Value *Quotient = generateSignedDivisionCode(Div->getOperand(0),
                                                 Div->getOperand(1), Builder);
    Div->replaceAllUsesWith(Quotient);
    Div->dropAllReferences();
    Div->eraseFromParent();

    // If we didn't actually generate a udiv instruction, we're done
    BinaryOperator *BO = dyn_cast<BinaryOperator>(Builder.GetInsertPoint());
    if (!BO || BO->getOpcode() != Instruction::UDiv)
      return true;

    Div = BO;
  }

  // Insert the unsigned division code
  Value *Quotient = generateUnsignedDivisionCode(Div->getOperand(0),
                                                 Div->getOperand(1),
                                                 Builder);
  Div->replaceAllUsesWith(Quotient);
  Div->dropAllReferences();
  Div->eraseFromParent();

  return true;
}
