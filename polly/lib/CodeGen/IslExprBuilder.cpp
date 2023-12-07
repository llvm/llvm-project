//===------ IslExprBuilder.cpp ----- Code generate isl AST expressions ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/IslExprBuilder.h"
#include "polly/CodeGen/RuntimeDebugBuilder.h"
#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "polly/Support/GICHelper.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;
using namespace polly;

/// Different overflow tracking modes.
enum OverflowTrackingChoice {
  OT_NEVER,   ///< Never tack potential overflows.
  OT_REQUEST, ///< Track potential overflows if requested.
  OT_ALWAYS   ///< Always track potential overflows.
};

static cl::opt<OverflowTrackingChoice> OTMode(
    "polly-overflow-tracking",
    cl::desc("Define where potential integer overflows in generated "
             "expressions should be tracked."),
    cl::values(clEnumValN(OT_NEVER, "never", "Never track the overflow bit."),
               clEnumValN(OT_REQUEST, "request",
                          "Track the overflow bit if requested."),
               clEnumValN(OT_ALWAYS, "always",
                          "Always track the overflow bit.")),
    cl::Hidden, cl::init(OT_REQUEST), cl::cat(PollyCategory));

IslExprBuilder::IslExprBuilder(Scop &S, PollyIRBuilder &Builder,
                               IDToValueTy &IDToValue, ValueMapT &GlobalMap,
                               const DataLayout &DL, ScalarEvolution &SE,
                               DominatorTree &DT, LoopInfo &LI,
                               BasicBlock *StartBlock)
    : S(S), Builder(Builder), IDToValue(IDToValue), GlobalMap(GlobalMap),
      DL(DL), SE(SE), DT(DT), LI(LI), StartBlock(StartBlock) {
  OverflowState = (OTMode == OT_ALWAYS) ? Builder.getFalse() : nullptr;
}

void IslExprBuilder::setTrackOverflow(bool Enable) {
  // If potential overflows are tracked always or never we ignore requests
  // to change the behavior.
  if (OTMode != OT_REQUEST)
    return;

  if (Enable) {
    // If tracking should be enabled initialize the OverflowState.
    OverflowState = Builder.getFalse();
  } else {
    // If tracking should be disabled just unset the OverflowState.
    OverflowState = nullptr;
  }
}

Value *IslExprBuilder::getOverflowState() const {
  // If the overflow tracking was requested but it is disabled we avoid the
  // additional nullptr checks at the call sides but instead provide a
  // meaningful result.
  if (OTMode == OT_NEVER)
    return Builder.getFalse();
  return OverflowState;
}

bool IslExprBuilder::hasLargeInts(isl::ast_expr Expr) {
  enum isl_ast_expr_type Type = isl_ast_expr_get_type(Expr.get());

  if (Type == isl_ast_expr_id)
    return false;

  if (Type == isl_ast_expr_int) {
    isl::val Val = Expr.get_val();
    APInt APValue = APIntFromVal(Val);
    auto BitWidth = APValue.getBitWidth();
    return BitWidth >= 64;
  }

  assert(Type == isl_ast_expr_op && "Expected isl_ast_expr of type operation");

  int NumArgs = isl_ast_expr_get_op_n_arg(Expr.get());

  for (int i = 0; i < NumArgs; i++) {
    isl::ast_expr Operand = Expr.get_op_arg(i);
    if (hasLargeInts(Operand))
      return true;
  }

  return false;
}

Value *IslExprBuilder::createBinOp(BinaryOperator::BinaryOps Opc, Value *LHS,
                                   Value *RHS, const Twine &Name) {
  // Handle the plain operation (without overflow tracking) first.
  if (!OverflowState) {
    switch (Opc) {
    case Instruction::Add:
      return Builder.CreateNSWAdd(LHS, RHS, Name);
    case Instruction::Sub:
      return Builder.CreateNSWSub(LHS, RHS, Name);
    case Instruction::Mul:
      return Builder.CreateNSWMul(LHS, RHS, Name);
    default:
      llvm_unreachable("Unknown binary operator!");
    }
  }

  Function *F = nullptr;
  Module *M = Builder.GetInsertBlock()->getModule();
  switch (Opc) {
  case Instruction::Add:
    F = Intrinsic::getDeclaration(M, Intrinsic::sadd_with_overflow,
                                  {LHS->getType()});
    break;
  case Instruction::Sub:
    F = Intrinsic::getDeclaration(M, Intrinsic::ssub_with_overflow,
                                  {LHS->getType()});
    break;
  case Instruction::Mul:
    F = Intrinsic::getDeclaration(M, Intrinsic::smul_with_overflow,
                                  {LHS->getType()});
    break;
  default:
    llvm_unreachable("No overflow intrinsic for binary operator found!");
  }

  auto *ResultStruct = Builder.CreateCall(F, {LHS, RHS}, Name);
  assert(ResultStruct->getType()->isStructTy());

  auto *OverflowFlag =
      Builder.CreateExtractValue(ResultStruct, 1, Name + ".obit");

  // If all overflows are tracked we do not combine the results as this could
  // cause dominance problems. Instead we will always keep the last overflow
  // flag as current state.
  if (OTMode == OT_ALWAYS)
    OverflowState = OverflowFlag;
  else
    OverflowState =
        Builder.CreateOr(OverflowState, OverflowFlag, "polly.overflow.state");

  return Builder.CreateExtractValue(ResultStruct, 0, Name + ".res");
}

Value *IslExprBuilder::createAdd(Value *LHS, Value *RHS, const Twine &Name) {
  return createBinOp(Instruction::Add, LHS, RHS, Name);
}

Value *IslExprBuilder::createSub(Value *LHS, Value *RHS, const Twine &Name) {
  return createBinOp(Instruction::Sub, LHS, RHS, Name);
}

Value *IslExprBuilder::createMul(Value *LHS, Value *RHS, const Twine &Name) {
  return createBinOp(Instruction::Mul, LHS, RHS, Name);
}

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
  return createSub(ConstantInt::getNullValue(MaxType), V);
}

Value *IslExprBuilder::createOpNAry(__isl_take isl_ast_expr *Expr) {
  assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_op &&
         "isl ast expression not of type isl_ast_op");
  assert(isl_ast_expr_get_op_n_arg(Expr) >= 2 &&
         "We need at least two operands in an n-ary operation");

  CmpInst::Predicate Pred;
  switch (isl_ast_expr_get_op_type(Expr)) {
  default:
    llvm_unreachable("This is not a an n-ary isl ast expression");
  case isl_ast_op_max:
    Pred = CmpInst::ICMP_SGT;
    break;
  case isl_ast_op_min:
    Pred = CmpInst::ICMP_SLT;
    break;
  }

  Value *V = create(isl_ast_expr_get_op_arg(Expr, 0));

  for (int i = 1; i < isl_ast_expr_get_op_n_arg(Expr); ++i) {
    Value *OpV = create(isl_ast_expr_get_op_arg(Expr, i));
    Type *Ty = getWidestType(V->getType(), OpV->getType());

    if (Ty != OpV->getType())
      OpV = Builder.CreateSExt(OpV, Ty);

    if (Ty != V->getType())
      V = Builder.CreateSExt(V, Ty);

    Value *Cmp = Builder.CreateICmp(Pred, V, OpV);
    V = Builder.CreateSelect(Cmp, V, OpV);
  }

  // TODO: We can truncate the result, if it fits into a smaller type. This can
  // help in cases where we have larger operands (e.g. i67) but the result is
  // known to fit into i64. Without the truncation, the larger i67 type may
  // force all subsequent operations to be performed on a non-native type.
  isl_ast_expr_free(Expr);
  return V;
}

std::pair<Value *, Type *>
IslExprBuilder::createAccessAddress(__isl_take isl_ast_expr *Expr) {
  assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_op &&
         "isl ast expression not of type isl_ast_op");
  assert(isl_ast_expr_get_op_type(Expr) == isl_ast_op_access &&
         "not an access isl ast expression");
  assert(isl_ast_expr_get_op_n_arg(Expr) >= 1 &&
         "We need at least two operands to create a member access.");

  Value *Base, *IndexOp, *Access;
  isl_ast_expr *BaseExpr;
  isl_id *BaseId;

  BaseExpr = isl_ast_expr_get_op_arg(Expr, 0);
  BaseId = isl_ast_expr_get_id(BaseExpr);
  isl_ast_expr_free(BaseExpr);

  const ScopArrayInfo *SAI = nullptr;

  if (PollyDebugPrinting)
    RuntimeDebugBuilder::createCPUPrinter(Builder, isl_id_get_name(BaseId));

  if (IDToSAI)
    SAI = (*IDToSAI)[BaseId];

  if (!SAI)
    SAI = ScopArrayInfo::getFromId(isl::manage(BaseId));
  else
    isl_id_free(BaseId);

  assert(SAI && "No ScopArrayInfo found for this isl_id.");

  Base = SAI->getBasePtr();

  if (auto NewBase = GlobalMap.lookup(Base))
    Base = NewBase;

  assert(Base->getType()->isPointerTy() && "Access base should be a pointer");
  StringRef BaseName = Base->getName();

  if (isl_ast_expr_get_op_n_arg(Expr) == 1) {
    isl_ast_expr_free(Expr);
    if (PollyDebugPrinting)
      RuntimeDebugBuilder::createCPUPrinter(Builder, "\n");
    return {Base, SAI->getElementType()};
  }

  IndexOp = nullptr;
  for (unsigned u = 1, e = isl_ast_expr_get_op_n_arg(Expr); u < e; u++) {
    Value *NextIndex = create(isl_ast_expr_get_op_arg(Expr, u));
    assert(NextIndex->getType()->isIntegerTy() &&
           "Access index should be an integer");

    if (PollyDebugPrinting)
      RuntimeDebugBuilder::createCPUPrinter(Builder, "[", NextIndex, "]");

    if (!IndexOp) {
      IndexOp = NextIndex;
    } else {
      Type *Ty = getWidestType(NextIndex->getType(), IndexOp->getType());

      if (Ty != NextIndex->getType())
        NextIndex = Builder.CreateIntCast(NextIndex, Ty, true);
      if (Ty != IndexOp->getType())
        IndexOp = Builder.CreateIntCast(IndexOp, Ty, true);

      IndexOp = createAdd(IndexOp, NextIndex, "polly.access.add." + BaseName);
    }

    // For every but the last dimension multiply the size, for the last
    // dimension we can exit the loop.
    if (u + 1 >= e)
      break;

    const SCEV *DimSCEV = SAI->getDimensionSize(u);

    llvm::ValueToSCEVMapTy Map;
    for (auto &KV : GlobalMap)
      Map[KV.first] = SE.getSCEV(KV.second);
    DimSCEV = SCEVParameterRewriter::rewrite(DimSCEV, SE, Map);
    Value *DimSize =
        expandCodeFor(S, SE, DL, "polly", DimSCEV, DimSCEV->getType(),
                      &*Builder.GetInsertPoint(), nullptr,
                      StartBlock->getSinglePredecessor());

    Type *Ty = getWidestType(DimSize->getType(), IndexOp->getType());

    if (Ty != IndexOp->getType())
      IndexOp = Builder.CreateSExtOrTrunc(IndexOp, Ty,
                                          "polly.access.sext." + BaseName);
    if (Ty != DimSize->getType())
      DimSize = Builder.CreateSExtOrTrunc(DimSize, Ty,
                                          "polly.access.sext." + BaseName);
    IndexOp = createMul(IndexOp, DimSize, "polly.access.mul." + BaseName);
  }

  Access = Builder.CreateGEP(SAI->getElementType(), Base, IndexOp,
                             "polly.access." + BaseName);

  if (PollyDebugPrinting)
    RuntimeDebugBuilder::createCPUPrinter(Builder, "\n");
  isl_ast_expr_free(Expr);
  return {Access, SAI->getElementType()};
}

Value *IslExprBuilder::createOpAccess(__isl_take isl_ast_expr *Expr) {
  auto Info = createAccessAddress(Expr);
  assert(Info.first && "Could not create op access address");
  return Builder.CreateLoad(Info.second, Info.first,
                            Info.first->getName() + ".load");
}

Value *IslExprBuilder::createOpBin(__isl_take isl_ast_expr *Expr) {
  Value *LHS, *RHS, *Res;
  Type *MaxType;
  isl_ast_op_type OpType;

  assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_op &&
         "isl ast expression not of type isl_ast_op");
  assert(isl_ast_expr_get_op_n_arg(Expr) == 2 &&
         "not a binary isl ast expression");

  OpType = isl_ast_expr_get_op_type(Expr);

  LHS = create(isl_ast_expr_get_op_arg(Expr, 0));
  RHS = create(isl_ast_expr_get_op_arg(Expr, 1));

  Type *LHSType = LHS->getType();
  Type *RHSType = RHS->getType();

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
    Res = createAdd(LHS, RHS);
    break;
  case isl_ast_op_sub:
    Res = createSub(LHS, RHS);
    break;
  case isl_ast_op_mul:
    Res = createMul(LHS, RHS);
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
    //       incorrect overflow in some edge cases.
    //
    // floord(n,d) ((n < 0) ? (n - d + 1) : n) / d
    Value *One = ConstantInt::get(MaxType, 1);
    Value *Zero = ConstantInt::get(MaxType, 0);
    Value *Sum1 = createSub(LHS, RHS, "pexp.fdiv_q.0");
    Value *Sum2 = createAdd(Sum1, One, "pexp.fdiv_q.1");
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
    Res = Builder.CreateSRem(LHS, RHS, "pexp.zdiv_r");
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

  auto *Op0 = isl_ast_expr_get_op_arg(Expr, 0);
  auto *Op1 = isl_ast_expr_get_op_arg(Expr, 1);
  bool HasNonAddressOfOperand =
      isl_ast_expr_get_type(Op0) != isl_ast_expr_op ||
      isl_ast_expr_get_type(Op1) != isl_ast_expr_op ||
      isl_ast_expr_get_op_type(Op0) != isl_ast_op_address_of ||
      isl_ast_expr_get_op_type(Op1) != isl_ast_op_address_of;

  LHS = create(Op0);
  RHS = create(Op1);

  auto *LHSTy = LHS->getType();
  auto *RHSTy = RHS->getType();
  bool IsPtrType = LHSTy->isPointerTy() || RHSTy->isPointerTy();
  bool UseUnsignedCmp = IsPtrType && !HasNonAddressOfOperand;

  auto *PtrAsIntTy = Builder.getIntNTy(DL.getPointerSizeInBits());
  if (LHSTy->isPointerTy())
    LHS = Builder.CreatePtrToInt(LHS, PtrAsIntTy);
  if (RHSTy->isPointerTy())
    RHS = Builder.CreatePtrToInt(RHS, PtrAsIntTy);

  if (LHS->getType() != RHS->getType()) {
    Type *MaxType = LHS->getType();
    MaxType = getWidestType(MaxType, RHS->getType());

    if (MaxType != RHS->getType())
      RHS = Builder.CreateSExt(RHS, MaxType);

    if (MaxType != LHS->getType())
      LHS = Builder.CreateSExt(LHS, MaxType);
  }

  isl_ast_op_type OpType = isl_ast_expr_get_op_type(Expr);
  assert(OpType >= isl_ast_op_eq && OpType <= isl_ast_op_gt &&
         "Unsupported ICmp isl ast expression");
  static_assert(isl_ast_op_eq + 4 == isl_ast_op_gt,
                "Isl ast op type interface changed");

  CmpInst::Predicate Predicates[5][2] = {
      {CmpInst::ICMP_EQ, CmpInst::ICMP_EQ},
      {CmpInst::ICMP_SLE, CmpInst::ICMP_ULE},
      {CmpInst::ICMP_SLT, CmpInst::ICMP_ULT},
      {CmpInst::ICMP_SGE, CmpInst::ICMP_UGE},
      {CmpInst::ICMP_SGT, CmpInst::ICMP_UGT},
  };

  Res = Builder.CreateICmp(Predicates[OpType - isl_ast_op_eq][UseUnsignedCmp],
                           LHS, RHS);

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
  // will outweigh the overhead introduced by evaluating unneeded expressions.
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

  Value *V = createAccessAddress(Op).first;

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

  if (V->getType()->isPointerTy())
    V = Builder.CreatePtrToInt(V, Builder.getIntNTy(DL.getPointerSizeInBits()));

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

  APValue = APValue.sext(T->getBitWidth());
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
