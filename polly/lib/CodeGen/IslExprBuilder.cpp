//===------ IslExprBuilder.cpp ----- Code generate isl AST expressions ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/IslExprBuilder.h"
#include "polly/ScopInfo.h"
#include "polly/Support/GICHelper.h"
#include "polly/Support/ScopHelper.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;
using namespace polly;

Type *IslExprBuilder::getWidestType(Type *T1, Type *T2) {
  assert(isa<IntegerType>(T1) && isa<IntegerType>(T2));

  if (T1->getPrimitiveSizeInBits() < T2->getPrimitiveSizeInBits())
    return T2;
  else
    return T1;
}

Value *IslExprBuilder::createOpUnary(__isl_take isl_ast_expr *Expr) {
  assert(isl_ast_expr_get_op_type(Expr) == isl_ast_op_minus &&
         "Unsupported unary operation");

  Value *V;
  Type *MaxType = getType(Expr);
  assert(MaxType->isIntegerTy() &&
         "Unary expressions can only be created for integer types");

  V = create(isl_ast_expr_get_op_arg(Expr, 0));
  MaxType = getWidestType(MaxType, V->getType());

  if (MaxType != V->getType())
    V = Builder.CreateSExt(V, MaxType);

  isl_ast_expr_free(Expr);
  return Builder.CreateNSWNeg(V);
}

Value *IslExprBuilder::createOpNAry(__isl_take isl_ast_expr *Expr) {
  assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_op &&
         "isl ast expression not of type isl_ast_op");
  assert(isl_ast_expr_get_op_n_arg(Expr) >= 2 &&
         "We need at least two operands in an n-ary operation");

  Value *V;

  V = create(isl_ast_expr_get_op_arg(Expr, 0));

  for (int i = 0; i < isl_ast_expr_get_op_n_arg(Expr); ++i) {
    Value *OpV;
    OpV = create(isl_ast_expr_get_op_arg(Expr, i));

    Type *Ty = getWidestType(V->getType(), OpV->getType());

    if (Ty != OpV->getType())
      OpV = Builder.CreateSExt(OpV, Ty);

    if (Ty != V->getType())
      V = Builder.CreateSExt(V, Ty);

    switch (isl_ast_expr_get_op_type(Expr)) {
    default:
      llvm_unreachable("This is no n-ary isl ast expression");

    case isl_ast_op_max: {
      Value *Cmp = Builder.CreateICmpSGT(V, OpV);
      V = Builder.CreateSelect(Cmp, V, OpV);
      continue;
    }
    case isl_ast_op_min: {
      Value *Cmp = Builder.CreateICmpSLT(V, OpV);
      V = Builder.CreateSelect(Cmp, V, OpV);
      continue;
    }
    }
  }

  // TODO: We can truncate the result, if it fits into a smaller type. This can
  // help in cases where we have larger operands (e.g. i67) but the result is
  // known to fit into i64. Without the truncation, the larger i67 type may
  // force all subsequent operations to be performed on a non-native type.
  isl_ast_expr_free(Expr);
  return V;
}

Value *IslExprBuilder::createAccessAddress(isl_ast_expr *Expr) {
  assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_op &&
         "isl ast expression not of type isl_ast_op");
  assert(isl_ast_expr_get_op_type(Expr) == isl_ast_op_access &&
         "not an access isl ast expression");
  assert(isl_ast_expr_get_op_n_arg(Expr) >= 2 &&
         "We need at least two operands to create a member access.");

  Value *Base, *IndexOp, *Access;
  isl_ast_expr *BaseExpr;
  isl_id *BaseId;

  BaseExpr = isl_ast_expr_get_op_arg(Expr, 0);
  BaseId = isl_ast_expr_get_id(BaseExpr);
  isl_ast_expr_free(BaseExpr);

  const ScopArrayInfo *SAI = ScopArrayInfo::getFromId(BaseId);
  Base = SAI->getBasePtr();

  if (auto NewBase = GlobalMap.lookup(Base))
    Base = NewBase;

  assert(Base->getType()->isPointerTy() && "Access base should be a pointer");
  StringRef BaseName = Base->getName();

  auto PointerTy = PointerType::get(SAI->getElementType(),
                                    Base->getType()->getPointerAddressSpace());
  if (Base->getType() != PointerTy) {
    Base =
        Builder.CreateBitCast(Base, PointerTy, "polly.access.cast." + BaseName);
  }

  IndexOp = nullptr;
  for (unsigned u = 1, e = isl_ast_expr_get_op_n_arg(Expr); u < e; u++) {
    Value *NextIndex = create(isl_ast_expr_get_op_arg(Expr, u));
    assert(NextIndex->getType()->isIntegerTy() &&
           "Access index should be an integer");

    if (!IndexOp) {
      IndexOp = NextIndex;
    } else {
      Type *Ty = getWidestType(NextIndex->getType(), IndexOp->getType());

      if (Ty != NextIndex->getType())
        NextIndex = Builder.CreateIntCast(NextIndex, Ty, true);
      if (Ty != IndexOp->getType())
        IndexOp = Builder.CreateIntCast(IndexOp, Ty, true);

      IndexOp =
          Builder.CreateAdd(IndexOp, NextIndex, "polly.access.add." + BaseName);
    }

    // For every but the last dimension multiply the size, for the last
    // dimension we can exit the loop.
    if (u + 1 >= e)
      break;

    const SCEV *DimSCEV = SAI->getDimensionSize(u);

    llvm::ValueToValueMap Map(GlobalMap.begin(), GlobalMap.end());
    DimSCEV = SCEVParameterRewriter::rewrite(DimSCEV, SE, Map);
    Value *DimSize =
        expandCodeFor(S, SE, DL, "polly", DimSCEV, DimSCEV->getType(),
                      &*Builder.GetInsertPoint());

    Type *Ty = getWidestType(DimSize->getType(), IndexOp->getType());

    if (Ty != IndexOp->getType())
      IndexOp = Builder.CreateSExtOrTrunc(IndexOp, Ty,
                                          "polly.access.sext." + BaseName);
    if (Ty != DimSize->getType())
      DimSize = Builder.CreateSExtOrTrunc(DimSize, Ty,
                                          "polly.access.sext." + BaseName);
    IndexOp =
        Builder.CreateMul(IndexOp, DimSize, "polly.access.mul." + BaseName);
  }

  Access = Builder.CreateGEP(Base, IndexOp, "polly.access." + BaseName);

  isl_ast_expr_free(Expr);
  return Access;
}

Value *IslExprBuilder::createOpAccess(isl_ast_expr *Expr) {
  Value *Addr = createAccessAddress(Expr);
  assert(Addr && "Could not create op access address");
  return Builder.CreateLoad(Addr, Addr->getName() + ".load");
}

Value *IslExprBuilder::createOpBin(__isl_take isl_ast_expr *Expr) {
  Value *LHS, *RHS, *Res;
  Type *MaxType;
  isl_ast_expr *LOp, *ROp;
  isl_ast_op_type OpType;

  assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_op &&
         "isl ast expression not of type isl_ast_op");
  assert(isl_ast_expr_get_op_n_arg(Expr) == 2 &&
         "not a binary isl ast expression");

  OpType = isl_ast_expr_get_op_type(Expr);

  LOp = isl_ast_expr_get_op_arg(Expr, 0);
  ROp = isl_ast_expr_get_op_arg(Expr, 1);

  // Catch the special case ((-<pointer>) + <pointer>) which is for
  // isl the same as (<pointer> - <pointer>). We have to treat it here because
  // there is no valid semantics for the (-<pointer>) expression, hence in
  // createOpUnary such an expression will trigger a crash.
  // FIXME: The same problem can now be triggered by a subexpression of the LHS,
  //        however it is much less likely.
  if (OpType == isl_ast_op_add &&
      isl_ast_expr_get_type(LOp) == isl_ast_expr_op &&
      isl_ast_expr_get_op_type(LOp) == isl_ast_op_minus) {
    // Change the binary addition to a substraction.
    OpType = isl_ast_op_sub;

    // Extract the unary operand of the LHS.
    auto *LOpOp = isl_ast_expr_get_op_arg(LOp, 0);
    isl_ast_expr_free(LOp);

    // Swap the unary operand of the LHS and the RHS.
    LOp = ROp;
    ROp = LOpOp;
  }

  LHS = create(LOp);
  RHS = create(ROp);

  Type *LHSType = LHS->getType();
  Type *RHSType = RHS->getType();

  // Handle <pointer> - <pointer>
  if (LHSType->isPointerTy() && RHSType->isPointerTy()) {
    isl_ast_expr_free(Expr);
    assert(OpType == isl_ast_op_sub && "Substraction is the only valid binary "
                                       "pointer <-> pointer operation.");

    return Builder.CreatePtrDiff(LHS, RHS);
  }

  // Handle <pointer> +/- <integer> and <integer> +/- <pointer>
  if (LHSType->isPointerTy() || RHSType->isPointerTy()) {
    isl_ast_expr_free(Expr);

    assert((LHSType->isIntegerTy() || RHSType->isIntegerTy()) &&
           "Arithmetic operations might only performed on one but not two "
           "pointer types.");

    if (LHSType->isIntegerTy())
      std::swap(LHS, RHS);

    switch (OpType) {
    default:
      llvm_unreachable(
          "Only additive binary operations are allowed on pointer types.");
    case isl_ast_op_sub:
      RHS = Builder.CreateNeg(RHS);
    // Fall through
    case isl_ast_op_add:
      return Builder.CreateGEP(LHS, RHS);
    }
  }

  MaxType = getWidestType(LHSType, RHSType);

  // Take the result into account when calculating the widest type.
  //
  // For operations such as '+' the result may require a type larger than
  // the type of the individual operands. For other operations such as '/', the
  // result type cannot be larger than the type of the individual operand. isl
  // does not calculate correct types for these operations and we consequently
  // exclude those operations here.
  switch (OpType) {
  case isl_ast_op_pdiv_q:
  case isl_ast_op_pdiv_r:
  case isl_ast_op_div:
  case isl_ast_op_fdiv_q:
  case isl_ast_op_zdiv_r:
    // Do nothing
    break;
  case isl_ast_op_add:
  case isl_ast_op_sub:
  case isl_ast_op_mul:
    MaxType = getWidestType(MaxType, getType(Expr));
    break;
  default:
    llvm_unreachable("This is no binary isl ast expression");
  }

  if (MaxType != RHS->getType())
    RHS = Builder.CreateSExt(RHS, MaxType);

  if (MaxType != LHS->getType())
    LHS = Builder.CreateSExt(LHS, MaxType);

  switch (OpType) {
  default:
    llvm_unreachable("This is no binary isl ast expression");
  case isl_ast_op_add:
    Res = Builder.CreateNSWAdd(LHS, RHS);
    break;
  case isl_ast_op_sub:
    Res = Builder.CreateNSWSub(LHS, RHS);
    break;
  case isl_ast_op_mul:
    Res = Builder.CreateNSWMul(LHS, RHS);
    break;
  case isl_ast_op_div:
    Res = Builder.CreateSDiv(LHS, RHS, "pexp.div", true);
    break;
  case isl_ast_op_pdiv_q: // Dividend is non-negative
    Res = Builder.CreateUDiv(LHS, RHS, "pexp.p_div_q");
    break;
  case isl_ast_op_fdiv_q: { // Round towards -infty
    if (auto *Const = dyn_cast<ConstantInt>(RHS)) {
      auto &Val = Const->getValue();
      if (Val.isPowerOf2() && Val.isNonNegative()) {
        Res = Builder.CreateAShr(LHS, Val.ceilLogBase2(), "polly.fdiv_q.shr");
        break;
      }
    }
    // TODO: Review code and check that this calculation does not yield
    //       incorrect overflow in some bordercases.
    //
    // floord(n,d) ((n < 0) ? (n - d + 1) : n) / d
    Value *One = ConstantInt::get(MaxType, 1);
    Value *Zero = ConstantInt::get(MaxType, 0);
    Value *Sum1 = Builder.CreateSub(LHS, RHS, "pexp.fdiv_q.0");
    Value *Sum2 = Builder.CreateAdd(Sum1, One, "pexp.fdiv_q.1");
    Value *isNegative = Builder.CreateICmpSLT(LHS, Zero, "pexp.fdiv_q.2");
    Value *Dividend =
        Builder.CreateSelect(isNegative, Sum2, LHS, "pexp.fdiv_q.3");
    Res = Builder.CreateSDiv(Dividend, RHS, "pexp.fdiv_q.4");
    break;
  }
  case isl_ast_op_pdiv_r: // Dividend is non-negative
    Res = Builder.CreateURem(LHS, RHS, "pexp.pdiv_r");
    break;

  case isl_ast_op_zdiv_r: // Result only compared against zero
    Res = Builder.CreateURem(LHS, RHS, "pexp.zdiv_r");
    break;
  }

  // TODO: We can truncate the result, if it fits into a smaller type. This can
  // help in cases where we have larger operands (e.g. i67) but the result is
  // known to fit into i64. Without the truncation, the larger i67 type may
  // force all subsequent operations to be performed on a non-native type.
  isl_ast_expr_free(Expr);
  return Res;
}

Value *IslExprBuilder::createOpSelect(__isl_take isl_ast_expr *Expr) {
  assert(isl_ast_expr_get_op_type(Expr) == isl_ast_op_select &&
         "Unsupported unary isl ast expression");
  Value *LHS, *RHS, *Cond;
  Type *MaxType = getType(Expr);

  Cond = create(isl_ast_expr_get_op_arg(Expr, 0));
  if (!Cond->getType()->isIntegerTy(1))
    Cond = Builder.CreateIsNotNull(Cond);

  LHS = create(isl_ast_expr_get_op_arg(Expr, 1));
  RHS = create(isl_ast_expr_get_op_arg(Expr, 2));

  MaxType = getWidestType(MaxType, LHS->getType());
  MaxType = getWidestType(MaxType, RHS->getType());

  if (MaxType != RHS->getType())
    RHS = Builder.CreateSExt(RHS, MaxType);

  if (MaxType != LHS->getType())
    LHS = Builder.CreateSExt(LHS, MaxType);

  // TODO: Do we want to truncate the result?
  isl_ast_expr_free(Expr);
  return Builder.CreateSelect(Cond, LHS, RHS);
}

Value *IslExprBuilder::createOpICmp(__isl_take isl_ast_expr *Expr) {
  assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_op &&
         "Expected an isl_ast_expr_op expression");

  Value *LHS, *RHS, *Res;

  LHS = create(isl_ast_expr_get_op_arg(Expr, 0));
  RHS = create(isl_ast_expr_get_op_arg(Expr, 1));

  bool IsPtrType =
      LHS->getType()->isPointerTy() || RHS->getType()->isPointerTy();

  if (LHS->getType() != RHS->getType()) {
    if (IsPtrType) {
      Type *I8PtrTy = Builder.getInt8PtrTy();
      if (!LHS->getType()->isPointerTy())
        LHS = Builder.CreateIntToPtr(LHS, I8PtrTy);
      if (!RHS->getType()->isPointerTy())
        RHS = Builder.CreateIntToPtr(RHS, I8PtrTy);
      if (LHS->getType() != I8PtrTy)
        LHS = Builder.CreateBitCast(LHS, I8PtrTy);
      if (RHS->getType() != I8PtrTy)
        RHS = Builder.CreateBitCast(RHS, I8PtrTy);
    } else {
      Type *MaxType = LHS->getType();
      MaxType = getWidestType(MaxType, RHS->getType());

      if (MaxType != RHS->getType())
        RHS = Builder.CreateSExt(RHS, MaxType);

      if (MaxType != LHS->getType())
        LHS = Builder.CreateSExt(LHS, MaxType);
    }
  }

  isl_ast_op_type OpType = isl_ast_expr_get_op_type(Expr);
  assert(OpType >= isl_ast_op_eq && OpType <= isl_ast_op_gt &&
         "Unsupported ICmp isl ast expression");
  assert(isl_ast_op_eq + 4 == isl_ast_op_gt &&
         "Isl ast op type interface changed");

  CmpInst::Predicate Predicates[5][2] = {
      {CmpInst::ICMP_EQ, CmpInst::ICMP_EQ},
      {CmpInst::ICMP_SLE, CmpInst::ICMP_ULE},
      {CmpInst::ICMP_SLT, CmpInst::ICMP_ULT},
      {CmpInst::ICMP_SGE, CmpInst::ICMP_UGE},
      {CmpInst::ICMP_SGT, CmpInst::ICMP_UGT},
  };

  Res = Builder.CreateICmp(Predicates[OpType - isl_ast_op_eq][IsPtrType], LHS,
                           RHS);

  isl_ast_expr_free(Expr);
  return Res;
}

Value *IslExprBuilder::createOpBoolean(__isl_take isl_ast_expr *Expr) {
  assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_op &&
         "Expected an isl_ast_expr_op expression");

  Value *LHS, *RHS, *Res;
  isl_ast_op_type OpType;

  OpType = isl_ast_expr_get_op_type(Expr);

  assert((OpType == isl_ast_op_and || OpType == isl_ast_op_or) &&
         "Unsupported isl_ast_op_type");

  LHS = create(isl_ast_expr_get_op_arg(Expr, 0));
  RHS = create(isl_ast_expr_get_op_arg(Expr, 1));

  // Even though the isl pretty printer prints the expressions as 'exp && exp'
  // or 'exp || exp', we actually code generate the bitwise expressions
  // 'exp & exp' or 'exp | exp'. This forces the evaluation of both branches,
  // but it is, due to the use of i1 types, otherwise equivalent. The reason
  // to go for bitwise operations is, that we assume the reduced control flow
  // will outweight the overhead introduced by evaluating unneeded expressions.
  // The isl code generation currently does not take advantage of the fact that
  // the expression after an '||' or '&&' is in some cases not evaluated.
  // Evaluating it anyways does not cause any undefined behaviour.
  //
  // TODO: Document in isl itself, that the unconditionally evaluating the
  // second part of '||' or '&&' expressions is safe.
  if (!LHS->getType()->isIntegerTy(1))
    LHS = Builder.CreateIsNotNull(LHS);
  if (!RHS->getType()->isIntegerTy(1))
    RHS = Builder.CreateIsNotNull(RHS);

  switch (OpType) {
  default:
    llvm_unreachable("Unsupported boolean expression");
  case isl_ast_op_and:
    Res = Builder.CreateAnd(LHS, RHS);
    break;
  case isl_ast_op_or:
    Res = Builder.CreateOr(LHS, RHS);
    break;
  }

  isl_ast_expr_free(Expr);
  return Res;
}

Value *
IslExprBuilder::createOpBooleanConditional(__isl_take isl_ast_expr *Expr) {
  assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_op &&
         "Expected an isl_ast_expr_op expression");

  Value *LHS, *RHS;
  isl_ast_op_type OpType;

  Function *F = Builder.GetInsertBlock()->getParent();
  LLVMContext &Context = F->getContext();

  OpType = isl_ast_expr_get_op_type(Expr);

  assert((OpType == isl_ast_op_and_then || OpType == isl_ast_op_or_else) &&
         "Unsupported isl_ast_op_type");

  auto InsertBB = Builder.GetInsertBlock();
  auto InsertPoint = Builder.GetInsertPoint();
  auto NextBB = SplitBlock(InsertBB, &*InsertPoint, &DT, &LI);
  BasicBlock *CondBB = BasicBlock::Create(Context, "polly.cond", F);
  LI.changeLoopFor(CondBB, LI.getLoopFor(InsertBB));
  DT.addNewBlock(CondBB, InsertBB);

  InsertBB->getTerminator()->eraseFromParent();
  Builder.SetInsertPoint(InsertBB);
  auto BR = Builder.CreateCondBr(Builder.getTrue(), NextBB, CondBB);

  Builder.SetInsertPoint(CondBB);
  Builder.CreateBr(NextBB);

  Builder.SetInsertPoint(InsertBB->getTerminator());

  LHS = create(isl_ast_expr_get_op_arg(Expr, 0));
  if (!LHS->getType()->isIntegerTy(1))
    LHS = Builder.CreateIsNotNull(LHS);
  auto LeftBB = Builder.GetInsertBlock();

  if (OpType == isl_ast_op_and || OpType == isl_ast_op_and_then)
    BR->setCondition(Builder.CreateNeg(LHS));
  else
    BR->setCondition(LHS);

  Builder.SetInsertPoint(CondBB->getTerminator());
  RHS = create(isl_ast_expr_get_op_arg(Expr, 1));
  if (!RHS->getType()->isIntegerTy(1))
    RHS = Builder.CreateIsNotNull(RHS);
  auto RightBB = Builder.GetInsertBlock();

  Builder.SetInsertPoint(NextBB->getTerminator());
  auto PHI = Builder.CreatePHI(Builder.getInt1Ty(), 2);
  PHI->addIncoming(OpType == isl_ast_op_and_then ? Builder.getFalse()
                                                 : Builder.getTrue(),
                   LeftBB);
  PHI->addIncoming(RHS, RightBB);

  isl_ast_expr_free(Expr);
  return PHI;
}

Value *IslExprBuilder::createOp(__isl_take isl_ast_expr *Expr) {
  assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_op &&
         "Expression not of type isl_ast_expr_op");
  switch (isl_ast_expr_get_op_type(Expr)) {
  case isl_ast_op_error:
  case isl_ast_op_cond:
  case isl_ast_op_call:
  case isl_ast_op_member:
    llvm_unreachable("Unsupported isl ast expression");
  case isl_ast_op_access:
    return createOpAccess(Expr);
  case isl_ast_op_max:
  case isl_ast_op_min:
    return createOpNAry(Expr);
  case isl_ast_op_add:
  case isl_ast_op_sub:
  case isl_ast_op_mul:
  case isl_ast_op_div:
  case isl_ast_op_fdiv_q: // Round towards -infty
  case isl_ast_op_pdiv_q: // Dividend is non-negative
  case isl_ast_op_pdiv_r: // Dividend is non-negative
  case isl_ast_op_zdiv_r: // Result only compared against zero
    return createOpBin(Expr);
  case isl_ast_op_minus:
    return createOpUnary(Expr);
  case isl_ast_op_select:
    return createOpSelect(Expr);
  case isl_ast_op_and:
  case isl_ast_op_or:
    return createOpBoolean(Expr);
  case isl_ast_op_and_then:
  case isl_ast_op_or_else:
    return createOpBooleanConditional(Expr);
  case isl_ast_op_eq:
  case isl_ast_op_le:
  case isl_ast_op_lt:
  case isl_ast_op_ge:
  case isl_ast_op_gt:
    return createOpICmp(Expr);
  case isl_ast_op_address_of:
    return createOpAddressOf(Expr);
  }

  llvm_unreachable("Unsupported isl_ast_expr_op kind.");
}

Value *IslExprBuilder::createOpAddressOf(__isl_take isl_ast_expr *Expr) {
  assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_op &&
         "Expected an isl_ast_expr_op expression.");
  assert(isl_ast_expr_get_op_n_arg(Expr) == 1 && "Address of should be unary.");

  isl_ast_expr *Op = isl_ast_expr_get_op_arg(Expr, 0);
  assert(isl_ast_expr_get_type(Op) == isl_ast_expr_op &&
         "Expected address of operator to be an isl_ast_expr_op expression.");
  assert(isl_ast_expr_get_op_type(Op) == isl_ast_op_access &&
         "Expected address of operator to be an access expression.");

  Value *V = createAccessAddress(Op);

  isl_ast_expr_free(Expr);

  return V;
}

Value *IslExprBuilder::createId(__isl_take isl_ast_expr *Expr) {
  assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_id &&
         "Expression not of type isl_ast_expr_ident");

  isl_id *Id;
  Value *V;

  Id = isl_ast_expr_get_id(Expr);

  assert(IDToValue.count(Id) && "Identifier not found");

  V = IDToValue[Id];
  if (!V)
    V = UndefValue::get(getType(Expr));

  assert(V && "Unknown parameter id found");

  isl_id_free(Id);
  isl_ast_expr_free(Expr);

  return V;
}

IntegerType *IslExprBuilder::getType(__isl_keep isl_ast_expr *Expr) {
  // XXX: We assume i64 is large enough. This is often true, but in general
  //      incorrect. Also, on 32bit architectures, it would be beneficial to
  //      use a smaller type. We can and should directly derive this information
  //      during code generation.
  return IntegerType::get(Builder.getContext(), 64);
}

Value *IslExprBuilder::createInt(__isl_take isl_ast_expr *Expr) {
  assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_int &&
         "Expression not of type isl_ast_expr_int");
  isl_val *Val;
  Value *V;
  APInt APValue;
  IntegerType *T;

  Val = isl_ast_expr_get_val(Expr);
  APValue = APIntFromVal(Val);

  auto BitWidth = APValue.getBitWidth();
  if (BitWidth <= 64)
    T = getType(Expr);
  else
    T = Builder.getIntNTy(BitWidth);

  APValue = APValue.sextOrSelf(T->getBitWidth());
  V = ConstantInt::get(T, APValue);

  isl_ast_expr_free(Expr);
  return V;
}

Value *IslExprBuilder::create(__isl_take isl_ast_expr *Expr) {
  switch (isl_ast_expr_get_type(Expr)) {
  case isl_ast_expr_error:
    llvm_unreachable("Code generation error");
  case isl_ast_expr_op:
    return createOp(Expr);
  case isl_ast_expr_id:
    return createId(Expr);
  case isl_ast_expr_int:
    return createInt(Expr);
  }

  llvm_unreachable("Unexpected enum value");
}
