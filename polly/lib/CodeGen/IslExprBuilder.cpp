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
#include "polly/Options.h"
#include "polly/ScopInfo.h"
#include "polly/Support/GICHelper.h"
#include "polly/Support/ScopHelper.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;
using namespace polly;

/// @brief Different overflow tracking modes.
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
                          "Always track the overflow bit."),
               clEnumValEnd),
    cl::Hidden, cl::init(OT_REQUEST), cl::ZeroOrMore, cl::cat(PollyCategory));

// @TODO This should actually be derived from the DataLayout.
static cl::opt<unsigned> PollyMaxAllowedBitWidth(
    "polly-max-expr-bit-width",
    cl::desc("The maximal bit with for generated expressions."), cl::Hidden,
    cl::ZeroOrMore, cl::init(64), cl::cat(PollyCategory));

IslExprBuilder::IslExprBuilder(Scop &S, PollyIRBuilder &Builder,
                               IDToValueTy &IDToValue, ValueMapT &GlobalMap,
                               const DataLayout &DL, ScalarEvolution &SE,
                               DominatorTree &DT, LoopInfo &LI)
    : S(S), Builder(Builder), IDToValue(IDToValue), GlobalMap(GlobalMap),
      DL(DL), SE(SE), DT(DT), LI(LI) {
  OverflowState = (OTMode == OT_ALWAYS) ? Builder.getFalse() : nullptr;
}

void IslExprBuilder::setTrackOverflow(bool Enable) {
  // If potential overflows are tracked always or never we ignore requests
  // to change the behaviour.
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

Value *IslExprBuilder::createBinOp(BinaryOperator::BinaryOps Opc, Value *LHS,
                                   Value *RHS, const Twine &Name) {
  // Flag that is true if the computation cannot overflow.
  bool IsSafeToCompute = false;
  switch (Opc) {
  case Instruction::Add:
  case Instruction::Sub:
    IsSafeToCompute = adjustTypesForSafeAddition(LHS, RHS);
    break;
  case Instruction::Mul:
    IsSafeToCompute = adjustTypesForSafeMultiplication(LHS, RHS);
    break;
  default:
    llvm_unreachable("Unknown binary operator!");
  }

  // Handle the plain operation (without overflow tracking or a safe
  // computation) first.
  if (!OverflowState || (IsSafeToCompute && (OTMode != OT_ALWAYS))) {
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

static Type *getWidestType(Type *T1, Type *T2) {
  assert(isa<IntegerType>(T1) && isa<IntegerType>(T2));

  if (T1->getPrimitiveSizeInBits() < T2->getPrimitiveSizeInBits())
    return T2;
  else
    return T1;
}

void IslExprBuilder::unifyTypes(Value *&V0, Value *&V1, Value *&V2) {
  auto *T0 = V0->getType();
  auto *T1 = V1->getType();
  auto *T2 = V2->getType();
  if (T0 == T1 && T1 == T2)
    return;
  auto *MaxT = getWidestType(T0, T1);
  MaxT = getWidestType(MaxT, T2);
  V0 = Builder.CreateSExt(V0, MaxT);
  V1 = Builder.CreateSExt(V1, MaxT);
  V2 = Builder.CreateSExt(V2, MaxT);
}

bool IslExprBuilder::adjustTypesForSafeComputation(Value *&LHS, Value *&RHS,
                                                   unsigned RequiredBitWidth) {
  unsigned LBitWidth = LHS->getType()->getPrimitiveSizeInBits();
  unsigned RBitWidth = RHS->getType()->getPrimitiveSizeInBits();
  unsigned MaxUsedBitWidth = std::max(LBitWidth, RBitWidth);

  // @TODO For now use the maximal bit width if the required one is to large but
  //       note that this is not sound.
  unsigned MaxAllowedBitWidth = PollyMaxAllowedBitWidth;
  unsigned NewBitWidth =
      std::max(MaxUsedBitWidth, std::min(MaxAllowedBitWidth, RequiredBitWidth));

  Type *Ty = Builder.getIntNTy(NewBitWidth);
  LHS = Builder.CreateSExt(LHS, Ty);
  RHS = Builder.CreateSExt(RHS, Ty);

  // If the new bit width is not large enough the computation is not sound.
  return NewBitWidth == RequiredBitWidth;
}

bool IslExprBuilder::adjustTypesForSafeAddition(Value *&LHS, Value *&RHS) {
  unsigned LBitWidth = LHS->getType()->getPrimitiveSizeInBits();
  unsigned RBitWidth = RHS->getType()->getPrimitiveSizeInBits();
  return adjustTypesForSafeComputation(LHS, RHS,
                                       std::max(LBitWidth, RBitWidth) + 1);
}

bool IslExprBuilder::adjustTypesForSafeMultiplication(Value *&LHS,
                                                      Value *&RHS) {
  unsigned LBitWidth = LHS->getType()->getPrimitiveSizeInBits();
  unsigned RBitWidth = RHS->getType()->getPrimitiveSizeInBits();
  return adjustTypesForSafeComputation(LHS, RHS, LBitWidth + RBitWidth);
}

Value *IslExprBuilder::createOpUnary(__isl_take isl_ast_expr *Expr) {
  assert(isl_ast_expr_get_op_type(Expr) == isl_ast_op_minus &&
         "Unsupported unary operation");

  auto *V = create(isl_ast_expr_get_op_arg(Expr, 0));
  assert(V->getType()->isIntegerTy() &&
         "Unary expressions can only be created for integer types");

  isl_ast_expr_free(Expr);
  return createSub(ConstantInt::getNullValue(V->getType()), V);
}

Value *IslExprBuilder::createOpNAry(__isl_take isl_ast_expr *Expr) {
  assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_op &&
         "isl ast expression not of type isl_ast_op");
  assert(isl_ast_expr_get_op_n_arg(Expr) >= 2 &&
         "We need at least two operands in an n-ary operation");
  assert((isl_ast_expr_get_op_type(Expr) == isl_ast_op_max ||
          isl_ast_expr_get_op_type(Expr) == isl_ast_op_min) &&
         "This is no n-ary isl ast expression");

  bool IsMax = isl_ast_expr_get_op_type(Expr) == isl_ast_op_max;
  auto Pred = IsMax ? CmpInst::ICMP_SGT : CmpInst::ICMP_SLT;
  auto *V = create(isl_ast_expr_get_op_arg(Expr, 0));

  for (int i = 1; i < isl_ast_expr_get_op_n_arg(Expr); ++i) {
    auto *OpV = create(isl_ast_expr_get_op_arg(Expr, i));
    unifyTypes(V, OpV);
    V = Builder.CreateSelect(Builder.CreateICmp(Pred, V, OpV), V, OpV);
  }

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

    IndexOp = !IndexOp ? NextIndex : createAdd(IndexOp, NextIndex,
                                               "polly.access.add." + BaseName);

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

    IndexOp = createMul(IndexOp, DimSize, "polly.access.mul." + BaseName);
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

Value *IslExprBuilder::createDiv(Value *LHS, Value *RHS, DivisionMode DM) {
  auto *ConstRHS = dyn_cast<ConstantInt>(RHS);
  unsigned UnusedBits = 0;
  Value *Res = nullptr;

  if (ConstRHS)
    UnusedBits = ConstRHS->getValue().logBase2();

  if (ConstRHS && ConstRHS->getValue().isPowerOf2() &&
      ConstRHS->getValue().isNonNegative())
    Res = Builder.CreateAShr(LHS, UnusedBits, "polly.div.shr");
  else if (DM == DM_SIGNED)
    Res = Builder.CreateSDiv(LHS, RHS, "pexp.div", true);
  else if (DM == DM_UNSIGNED)
    Res = Builder.CreateUDiv(LHS, RHS, "pexp.p_div_q");
  else {
    assert(DM == DM_FLOORED);
    // TODO: Review code and check that this calculation does not yield
    //       incorrect overflow in some bordercases.
    //
    // floord(n,d) ((n < 0) ? (n - d + 1) : n) / d
    Value *Sum1 = createSub(LHS, RHS, "pexp.fdiv_q.0");
    Value *One = ConstantInt::get(Sum1->getType(), 1);
    Value *Sum2 = createAdd(Sum1, One, "pexp.fdiv_q.1");
    Value *Zero = ConstantInt::get(LHS->getType(), 0);
    Value *isNegative = Builder.CreateICmpSLT(LHS, Zero, "pexp.fdiv_q.2");
    unifyTypes(LHS, Sum2);
    Value *Dividend =
        Builder.CreateSelect(isNegative, Sum2, LHS, "pexp.fdiv_q.3");
    unifyTypes(Dividend, RHS);
    Res = Builder.CreateSDiv(Dividend, RHS, "pexp.fdiv_q.4");
  }

  if (UnusedBits) {
    auto RequiredBits = Res->getType()->getPrimitiveSizeInBits() - UnusedBits;
    Res = Builder.CreateTrunc(Res, Builder.getIntNTy(RequiredBits),
                              "polly.div.trunc");
  }

  return Res;
}

Value *IslExprBuilder::createOpBin(__isl_take isl_ast_expr *Expr) {
  Value *LHS, *RHS, *Res;
  isl_ast_op_type OpType;

  assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_op &&
         "isl ast expression not of type isl_ast_op");
  assert(isl_ast_expr_get_op_n_arg(Expr) == 2 &&
         "not a binary isl ast expression");

  OpType = isl_ast_expr_get_op_type(Expr);

  LHS = create(isl_ast_expr_get_op_arg(Expr, 0));
  RHS = create(isl_ast_expr_get_op_arg(Expr, 1));

  // For possibly overflowing operations we will later adjust types but
  // for others we do it now as we will directly create the operations.
  switch (OpType) {
  case isl_ast_op_pdiv_q:
  case isl_ast_op_pdiv_r:
  case isl_ast_op_div:
  case isl_ast_op_fdiv_q:
  case isl_ast_op_zdiv_r:
    unifyTypes(LHS, RHS);
    break;
  case isl_ast_op_add:
  case isl_ast_op_sub:
  case isl_ast_op_mul:
    // Do nothing
    break;
  default:
    llvm_unreachable("This is no binary isl ast expression");
  }

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
    Res = createDiv(LHS, RHS, DM_SIGNED);
    break;
  case isl_ast_op_pdiv_q: // Dividend is non-negative
    Res = createDiv(LHS, RHS, DM_UNSIGNED);
    break;
  case isl_ast_op_fdiv_q: // Round towards -infty
    Res = createDiv(LHS, RHS, DM_FLOORED);
    break;
  case isl_ast_op_pdiv_r: // Dividend is non-negative
    Res = Builder.CreateURem(LHS, RHS, "pexp.pdiv_r");
    break;
  case isl_ast_op_zdiv_r: // Result only compared against zero
    Res = Builder.CreateSRem(LHS, RHS, "pexp.zdiv_r");
    break;
  }

  isl_ast_expr_free(Expr);
  return Res;
}

Value *IslExprBuilder::createOpSelect(__isl_take isl_ast_expr *Expr) {
  assert(isl_ast_expr_get_op_type(Expr) == isl_ast_op_select &&
         "Unsupported unary isl ast expression");
  Value *LHS, *RHS, *Cond;

  Cond = create(isl_ast_expr_get_op_arg(Expr, 0));
  if (!Cond->getType()->isIntegerTy(1))
    Cond = Builder.CreateIsNotNull(Cond);

  LHS = create(isl_ast_expr_get_op_arg(Expr, 1));
  RHS = create(isl_ast_expr_get_op_arg(Expr, 2));
  unifyTypes(LHS, RHS);

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

  unifyTypes(LHS, RHS);

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
    V = UndefValue::get(Builder.getInt1Ty());

  if (V->getType()->isPointerTy())
    V = Builder.CreatePtrToInt(V, Builder.getIntNTy(DL.getPointerSizeInBits()));

  assert(V && "Unknown parameter id found");

  isl_id_free(Id);
  isl_ast_expr_free(Expr);

  return V;
}

Value *IslExprBuilder::createInt(__isl_take isl_ast_expr *Expr) {
  assert(isl_ast_expr_get_type(Expr) == isl_ast_expr_int &&
         "Expression not of type isl_ast_expr_int");

  auto *Val = isl_ast_expr_get_val(Expr);
  auto *V = ConstantInt::get(Builder.getContext(), APIntFromVal(Val));

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
