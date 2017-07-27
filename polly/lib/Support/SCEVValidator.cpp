
#include "polly/Support/SCEVValidator.h"
#include "polly/ScopInfo.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace polly;

#define DEBUG_TYPE "polly-scev-validator"

namespace SCEVType {
/// The type of a SCEV
///
/// To check for the validity of a SCEV we assign to each SCEV a type. The
/// possible types are INT, PARAM, IV and INVALID. The order of the types is
/// important. The subexpressions of SCEV with a type X can only have a type
/// that is smaller or equal than X.
enum TYPE {
  // An integer value.
  INT,

  // An expression that is constant during the execution of the Scop,
  // but that may depend on parameters unknown at compile time.
  PARAM,

  // An expression that may change during the execution of the SCoP.
  IV,

  // An invalid expression.
  INVALID
};
} // namespace SCEVType

/// The result the validator returns for a SCEV expression.
class ValidatorResult {
  /// The type of the expression
  SCEVType::TYPE Type;

  /// The set of Parameters in the expression.
  ParameterSetTy Parameters;

public:
  /// The copy constructor
  ValidatorResult(const ValidatorResult &Source) {
    Type = Source.Type;
    Parameters = Source.Parameters;
  }

  /// Construct a result with a certain type and no parameters.
  ValidatorResult(SCEVType::TYPE Type) : Type(Type) {
    assert(Type != SCEVType::PARAM && "Did you forget to pass the parameter");
  }

  /// Construct a result with a certain type and a single parameter.
  ValidatorResult(SCEVType::TYPE Type, const SCEV *Expr) : Type(Type) {
    Parameters.insert(Expr);
  }

  /// Get the type of the ValidatorResult.
  SCEVType::TYPE getType() { return Type; }

  /// Is the analyzed SCEV constant during the execution of the SCoP.
  bool isConstant() { return Type == SCEVType::INT || Type == SCEVType::PARAM; }

  /// Is the analyzed SCEV valid.
  bool isValid() { return Type != SCEVType::INVALID; }

  /// Is the analyzed SCEV of Type IV.
  bool isIV() { return Type == SCEVType::IV; }

  /// Is the analyzed SCEV of Type INT.
  bool isINT() { return Type == SCEVType::INT; }

  /// Is the analyzed SCEV of Type PARAM.
  bool isPARAM() { return Type == SCEVType::PARAM; }

  /// Get the parameters of this validator result.
  const ParameterSetTy &getParameters() { return Parameters; }

  /// Add the parameters of Source to this result.
  void addParamsFrom(const ValidatorResult &Source) {
    Parameters.insert(Source.Parameters.begin(), Source.Parameters.end());
  }

  /// Merge a result.
  ///
  /// This means to merge the parameters and to set the Type to the most
  /// specific Type that matches both.
  void merge(const ValidatorResult &ToMerge) {
    Type = std::max(Type, ToMerge.Type);
    addParamsFrom(ToMerge);
  }

  void print(raw_ostream &OS) {
    switch (Type) {
    case SCEVType::INT:
      OS << "SCEVType::INT";
      break;
    case SCEVType::PARAM:
      OS << "SCEVType::PARAM";
      break;
    case SCEVType::IV:
      OS << "SCEVType::IV";
      break;
    case SCEVType::INVALID:
      OS << "SCEVType::INVALID";
      break;
    }
  }
};

raw_ostream &operator<<(raw_ostream &OS, class ValidatorResult &VR) {
  VR.print(OS);
  return OS;
}

bool polly::isConstCall(llvm::CallInst *Call) {
  if (Call->mayReadOrWriteMemory())
    return false;

  for (auto &Operand : Call->arg_operands())
    if (!isa<ConstantInt>(&Operand))
      return false;

  return true;
}

/// Check if a SCEV is valid in a SCoP.
struct SCEVValidator
    : public SCEVVisitor<SCEVValidator, class ValidatorResult> {
private:
  const Region *R;
  Loop *Scope;
  ScalarEvolution &SE;
  InvariantLoadsSetTy *ILS;

public:
  SCEVValidator(const Region *R, Loop *Scope, ScalarEvolution &SE,
                InvariantLoadsSetTy *ILS)
      : R(R), Scope(Scope), SE(SE), ILS(ILS) {}

  class ValidatorResult visitConstant(const SCEVConstant *Constant) {
    return ValidatorResult(SCEVType::INT);
  }

  class ValidatorResult visitZeroExtendOrTruncateExpr(const SCEV *Expr,
                                                      const SCEV *Operand) {
    ValidatorResult Op = visit(Operand);
    auto Type = Op.getType();

    // If unsigned operations are allowed return the operand, otherwise
    // check if we can model the expression without unsigned assumptions.
    if (PollyAllowUnsignedOperations || Type == SCEVType::INVALID)
      return Op;

    if (Type == SCEVType::IV)
      return ValidatorResult(SCEVType::INVALID);
    return ValidatorResult(SCEVType::PARAM, Expr);
  }

  class ValidatorResult visitTruncateExpr(const SCEVTruncateExpr *Expr) {
    return visitZeroExtendOrTruncateExpr(Expr, Expr->getOperand());
  }

  class ValidatorResult visitZeroExtendExpr(const SCEVZeroExtendExpr *Expr) {
    return visitZeroExtendOrTruncateExpr(Expr, Expr->getOperand());
  }

  class ValidatorResult visitSignExtendExpr(const SCEVSignExtendExpr *Expr) {
    return visit(Expr->getOperand());
  }

  class ValidatorResult visitAddExpr(const SCEVAddExpr *Expr) {
    ValidatorResult Return(SCEVType::INT);

    for (int i = 0, e = Expr->getNumOperands(); i < e; ++i) {
      ValidatorResult Op = visit(Expr->getOperand(i));
      Return.merge(Op);

      // Early exit.
      if (!Return.isValid())
        break;
    }

    return Return;
  }

  class ValidatorResult visitMulExpr(const SCEVMulExpr *Expr) {
    ValidatorResult Return(SCEVType::INT);

    bool HasMultipleParams = false;

    for (int i = 0, e = Expr->getNumOperands(); i < e; ++i) {
      ValidatorResult Op = visit(Expr->getOperand(i));

      if (Op.isINT())
        continue;

      if (Op.isPARAM() && Return.isPARAM()) {
        HasMultipleParams = true;
        continue;
      }

      if ((Op.isIV() || Op.isPARAM()) && !Return.isINT()) {
        DEBUG(dbgs() << "INVALID: More than one non-int operand in MulExpr\n"
                     << "\tExpr: " << *Expr << "\n"
                     << "\tPrevious expression type: " << Return << "\n"
                     << "\tNext operand (" << Op
                     << "): " << *Expr->getOperand(i) << "\n");

        return ValidatorResult(SCEVType::INVALID);
      }

      Return.merge(Op);
    }

    if (HasMultipleParams && Return.isValid())
      return ValidatorResult(SCEVType::PARAM, Expr);

    return Return;
  }

  class ValidatorResult visitAddRecExpr(const SCEVAddRecExpr *Expr) {
    if (!Expr->isAffine()) {
      DEBUG(dbgs() << "INVALID: AddRec is not affine");
      return ValidatorResult(SCEVType::INVALID);
    }

    ValidatorResult Start = visit(Expr->getStart());
    ValidatorResult Recurrence = visit(Expr->getStepRecurrence(SE));

    if (!Start.isValid())
      return Start;

    if (!Recurrence.isValid())
      return Recurrence;

    auto *L = Expr->getLoop();
    if (R->contains(L) && (!Scope || !L->contains(Scope))) {
      DEBUG(dbgs() << "INVALID: Loop of AddRec expression boxed in an a "
                      "non-affine subregion or has a non-synthesizable exit "
                      "value.");
      return ValidatorResult(SCEVType::INVALID);
    }

    if (R->contains(L)) {
      if (Recurrence.isINT()) {
        ValidatorResult Result(SCEVType::IV);
        Result.addParamsFrom(Start);
        return Result;
      }

      DEBUG(dbgs() << "INVALID: AddRec within scop has non-int"
                      "recurrence part");
      return ValidatorResult(SCEVType::INVALID);
    }

    assert(Recurrence.isConstant() && "Expected 'Recurrence' to be constant");

    // Directly generate ValidatorResult for Expr if 'start' is zero.
    if (Expr->getStart()->isZero())
      return ValidatorResult(SCEVType::PARAM, Expr);

    // Translate AddRecExpr from '{start, +, inc}' into 'start + {0, +, inc}'
    // if 'start' is not zero.
    const SCEV *ZeroStartExpr = SE.getAddRecExpr(
        SE.getConstant(Expr->getStart()->getType(), 0),
        Expr->getStepRecurrence(SE), Expr->getLoop(), Expr->getNoWrapFlags());

    ValidatorResult ZeroStartResult =
        ValidatorResult(SCEVType::PARAM, ZeroStartExpr);
    ZeroStartResult.addParamsFrom(Start);

    return ZeroStartResult;
  }

  class ValidatorResult visitSMaxExpr(const SCEVSMaxExpr *Expr) {
    ValidatorResult Return(SCEVType::INT);

    for (int i = 0, e = Expr->getNumOperands(); i < e; ++i) {
      ValidatorResult Op = visit(Expr->getOperand(i));

      if (!Op.isValid())
        return Op;

      Return.merge(Op);
    }

    return Return;
  }

  class ValidatorResult visitUMaxExpr(const SCEVUMaxExpr *Expr) {
    // We do not support unsigned max operations. If 'Expr' is constant during
    // Scop execution we treat this as a parameter, otherwise we bail out.
    for (int i = 0, e = Expr->getNumOperands(); i < e; ++i) {
      ValidatorResult Op = visit(Expr->getOperand(i));

      if (!Op.isConstant()) {
        DEBUG(dbgs() << "INVALID: UMaxExpr has a non-constant operand");
        return ValidatorResult(SCEVType::INVALID);
      }
    }

    return ValidatorResult(SCEVType::PARAM, Expr);
  }

  ValidatorResult visitGenericInst(Instruction *I, const SCEV *S) {
    if (R->contains(I)) {
      DEBUG(dbgs() << "INVALID: UnknownExpr references an instruction "
                      "within the region\n");
      return ValidatorResult(SCEVType::INVALID);
    }

    return ValidatorResult(SCEVType::PARAM, S);
  }

  ValidatorResult visitCallInstruction(Instruction *I, const SCEV *S) {
    assert(I->getOpcode() == Instruction::Call && "Call instruction expected");

    if (R->contains(I)) {
      auto Call = cast<CallInst>(I);

      if (!isConstCall(Call))
        return ValidatorResult(SCEVType::INVALID, S);
    }
    return ValidatorResult(SCEVType::PARAM, S);
  }

  ValidatorResult visitLoadInstruction(Instruction *I, const SCEV *S) {
    if (R->contains(I) && ILS) {
      ILS->insert(cast<LoadInst>(I));
      return ValidatorResult(SCEVType::PARAM, S);
    }

    return visitGenericInst(I, S);
  }

  ValidatorResult visitDivision(const SCEV *Dividend, const SCEV *Divisor,
                                const SCEV *DivExpr,
                                Instruction *SDiv = nullptr) {

    // First check if we might be able to model the division, thus if the
    // divisor is constant. If so, check the dividend, otherwise check if
    // the whole division can be seen as a parameter.
    if (isa<SCEVConstant>(Divisor) && !Divisor->isZero())
      return visit(Dividend);

    // For signed divisions use the SDiv instruction to check for a parameter
    // division, for unsigned divisions check the operands.
    if (SDiv)
      return visitGenericInst(SDiv, DivExpr);

    ValidatorResult LHS = visit(Dividend);
    ValidatorResult RHS = visit(Divisor);
    if (LHS.isConstant() && RHS.isConstant())
      return ValidatorResult(SCEVType::PARAM, DivExpr);

    DEBUG(dbgs() << "INVALID: unsigned division of non-constant expressions");
    return ValidatorResult(SCEVType::INVALID);
  }

  ValidatorResult visitUDivExpr(const SCEVUDivExpr *Expr) {
    if (!PollyAllowUnsignedOperations)
      return ValidatorResult(SCEVType::INVALID);

    auto *Dividend = Expr->getLHS();
    auto *Divisor = Expr->getRHS();
    return visitDivision(Dividend, Divisor, Expr);
  }

  ValidatorResult visitSDivInstruction(Instruction *SDiv, const SCEV *Expr) {
    assert(SDiv->getOpcode() == Instruction::SDiv &&
           "Assumed SDiv instruction!");

    auto *Dividend = SE.getSCEV(SDiv->getOperand(0));
    auto *Divisor = SE.getSCEV(SDiv->getOperand(1));
    return visitDivision(Dividend, Divisor, Expr, SDiv);
  }

  ValidatorResult visitSRemInstruction(Instruction *SRem, const SCEV *S) {
    assert(SRem->getOpcode() == Instruction::SRem &&
           "Assumed SRem instruction!");

    auto *Divisor = SRem->getOperand(1);
    auto *CI = dyn_cast<ConstantInt>(Divisor);
    if (!CI || CI->isZeroValue())
      return visitGenericInst(SRem, S);

    auto *Dividend = SRem->getOperand(0);
    auto *DividendSCEV = SE.getSCEV(Dividend);
    return visit(DividendSCEV);
  }

  ValidatorResult visitUnknown(const SCEVUnknown *Expr) {
    Value *V = Expr->getValue();

    if (!Expr->getType()->isIntegerTy() && !Expr->getType()->isPointerTy()) {
      DEBUG(dbgs() << "INVALID: UnknownExpr is not an integer or pointer");
      return ValidatorResult(SCEVType::INVALID);
    }

    if (isa<UndefValue>(V)) {
      DEBUG(dbgs() << "INVALID: UnknownExpr references an undef value");
      return ValidatorResult(SCEVType::INVALID);
    }

    if (Instruction *I = dyn_cast<Instruction>(Expr->getValue())) {
      switch (I->getOpcode()) {
      case Instruction::IntToPtr:
        return visit(SE.getSCEVAtScope(I->getOperand(0), Scope));
      case Instruction::PtrToInt:
        return visit(SE.getSCEVAtScope(I->getOperand(0), Scope));
      case Instruction::Load:
        return visitLoadInstruction(I, Expr);
      case Instruction::SDiv:
        return visitSDivInstruction(I, Expr);
      case Instruction::SRem:
        return visitSRemInstruction(I, Expr);
      case Instruction::Call:
        return visitCallInstruction(I, Expr);
      default:
        return visitGenericInst(I, Expr);
      }
    }

    return ValidatorResult(SCEVType::PARAM, Expr);
  }
};

class SCEVHasIVParams {
  bool HasIVParams = false;

public:
  SCEVHasIVParams() {}

  bool follow(const SCEV *S) {
    const SCEVUnknown *Unknown = dyn_cast<SCEVUnknown>(S);
    if (!Unknown)
      return true;

    CallInst *Call = dyn_cast<CallInst>(Unknown->getValue());

    if (!Call)
      return true;

    if (isConstCall(Call)) {
      HasIVParams = true;
      return false;
    }

    return true;
  }

  bool isDone() { return HasIVParams; }
  bool hasIVParams() { return HasIVParams; }
};

/// Check whether a SCEV refers to an SSA name defined inside a region.
class SCEVInRegionDependences {
  const Region *R;
  Loop *Scope;
  const InvariantLoadsSetTy &ILS;
  bool AllowLoops;
  bool HasInRegionDeps = false;

public:
  SCEVInRegionDependences(const Region *R, Loop *Scope, bool AllowLoops,
                          const InvariantLoadsSetTy &ILS)
      : R(R), Scope(Scope), ILS(ILS), AllowLoops(AllowLoops) {}

  bool follow(const SCEV *S) {
    if (auto Unknown = dyn_cast<SCEVUnknown>(S)) {
      Instruction *Inst = dyn_cast<Instruction>(Unknown->getValue());

      CallInst *Call = dyn_cast<CallInst>(Unknown->getValue());

      if (Call && isConstCall(Call))
        return false;

      if (Inst) {
        // When we invariant load hoist a load, we first make sure that there
        // can be no dependences created by it in the Scop region. So, we should
        // not consider scalar dependences to `LoadInst`s that are invariant
        // load hoisted.
        //
        // If this check is not present, then we create data dependences which
        // are strictly not necessary by tracking the invariant load as a
        // scalar.
        LoadInst *LI = dyn_cast<LoadInst>(Inst);
        if (LI && ILS.count(LI) > 0)
          return false;
      }

      // Return true when Inst is defined inside the region R.
      if (!Inst || !R->contains(Inst))
        return true;

      HasInRegionDeps = true;
      return false;
    }

    if (auto AddRec = dyn_cast<SCEVAddRecExpr>(S)) {
      if (AllowLoops)
        return true;

      auto *L = AddRec->getLoop();
      if (R->contains(L) && !L->contains(Scope)) {
        HasInRegionDeps = true;
        return false;
      }
    }

    return true;
  }
  bool isDone() { return false; }
  bool hasDependences() { return HasInRegionDeps; }
};

namespace polly {
/// Find all loops referenced in SCEVAddRecExprs.
class SCEVFindLoops {
  SetVector<const Loop *> &Loops;

public:
  SCEVFindLoops(SetVector<const Loop *> &Loops) : Loops(Loops) {}

  bool follow(const SCEV *S) {
    if (const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(S))
      Loops.insert(AddRec->getLoop());
    return true;
  }
  bool isDone() { return false; }
};

void findLoops(const SCEV *Expr, SetVector<const Loop *> &Loops) {
  SCEVFindLoops FindLoops(Loops);
  SCEVTraversal<SCEVFindLoops> ST(FindLoops);
  ST.visitAll(Expr);
}

/// Find all values referenced in SCEVUnknowns.
class SCEVFindValues {
  ScalarEvolution &SE;
  SetVector<Value *> &Values;

public:
  SCEVFindValues(ScalarEvolution &SE, SetVector<Value *> &Values)
      : SE(SE), Values(Values) {}

  bool follow(const SCEV *S) {
    const SCEVUnknown *Unknown = dyn_cast<SCEVUnknown>(S);
    if (!Unknown)
      return true;

    Values.insert(Unknown->getValue());
    Instruction *Inst = dyn_cast<Instruction>(Unknown->getValue());
    if (!Inst || (Inst->getOpcode() != Instruction::SRem &&
                  Inst->getOpcode() != Instruction::SDiv))
      return false;

    auto *Dividend = SE.getSCEV(Inst->getOperand(1));
    if (!isa<SCEVConstant>(Dividend))
      return false;

    auto *Divisor = SE.getSCEV(Inst->getOperand(0));
    SCEVFindValues FindValues(SE, Values);
    SCEVTraversal<SCEVFindValues> ST(FindValues);
    ST.visitAll(Dividend);
    ST.visitAll(Divisor);

    return false;
  }
  bool isDone() { return false; }
};

void findValues(const SCEV *Expr, ScalarEvolution &SE,
                SetVector<Value *> &Values) {
  SCEVFindValues FindValues(SE, Values);
  SCEVTraversal<SCEVFindValues> ST(FindValues);
  ST.visitAll(Expr);
}

bool hasIVParams(const SCEV *Expr) {
  SCEVHasIVParams HasIVParams;
  SCEVTraversal<SCEVHasIVParams> ST(HasIVParams);
  ST.visitAll(Expr);
  return HasIVParams.hasIVParams();
}

bool hasScalarDepsInsideRegion(const SCEV *Expr, const Region *R,
                               llvm::Loop *Scope, bool AllowLoops,
                               const InvariantLoadsSetTy &ILS) {
  SCEVInRegionDependences InRegionDeps(R, Scope, AllowLoops, ILS);
  SCEVTraversal<SCEVInRegionDependences> ST(InRegionDeps);
  ST.visitAll(Expr);
  return InRegionDeps.hasDependences();
}

bool isAffineExpr(const Region *R, llvm::Loop *Scope, const SCEV *Expr,
                  ScalarEvolution &SE, InvariantLoadsSetTy *ILS) {
  if (isa<SCEVCouldNotCompute>(Expr))
    return false;

  SCEVValidator Validator(R, Scope, SE, ILS);
  DEBUG({
    dbgs() << "\n";
    dbgs() << "Expr: " << *Expr << "\n";
    dbgs() << "Region: " << R->getNameStr() << "\n";
    dbgs() << " -> ";
  });

  ValidatorResult Result = Validator.visit(Expr);

  DEBUG({
    if (Result.isValid())
      dbgs() << "VALID\n";
    dbgs() << "\n";
  });

  return Result.isValid();
}

static bool isAffineExpr(Value *V, const Region *R, Loop *Scope,
                         ScalarEvolution &SE, ParameterSetTy &Params) {
  auto *E = SE.getSCEV(V);
  if (isa<SCEVCouldNotCompute>(E))
    return false;

  SCEVValidator Validator(R, Scope, SE, nullptr);
  ValidatorResult Result = Validator.visit(E);
  if (!Result.isValid())
    return false;

  auto ResultParams = Result.getParameters();
  Params.insert(ResultParams.begin(), ResultParams.end());

  return true;
}

bool isAffineConstraint(Value *V, const Region *R, llvm::Loop *Scope,
                        ScalarEvolution &SE, ParameterSetTy &Params,
                        bool OrExpr) {
  if (auto *ICmp = dyn_cast<ICmpInst>(V)) {
    return isAffineConstraint(ICmp->getOperand(0), R, Scope, SE, Params,
                              true) &&
           isAffineConstraint(ICmp->getOperand(1), R, Scope, SE, Params, true);
  } else if (auto *BinOp = dyn_cast<BinaryOperator>(V)) {
    auto Opcode = BinOp->getOpcode();
    if (Opcode == Instruction::And || Opcode == Instruction::Or)
      return isAffineConstraint(BinOp->getOperand(0), R, Scope, SE, Params,
                                false) &&
             isAffineConstraint(BinOp->getOperand(1), R, Scope, SE, Params,
                                false);
    /* Fall through */
  }

  if (!OrExpr)
    return false;

  return isAffineExpr(V, R, Scope, SE, Params);
}

ParameterSetTy getParamsInAffineExpr(const Region *R, Loop *Scope,
                                     const SCEV *Expr, ScalarEvolution &SE) {
  if (isa<SCEVCouldNotCompute>(Expr))
    return ParameterSetTy();

  InvariantLoadsSetTy ILS;
  SCEVValidator Validator(R, Scope, SE, &ILS);
  ValidatorResult Result = Validator.visit(Expr);
  assert(Result.isValid() && "Requested parameters for an invalid SCEV!");

  return Result.getParameters();
}

std::pair<const SCEVConstant *, const SCEV *>
extractConstantFactor(const SCEV *S, ScalarEvolution &SE) {
  auto *ConstPart = cast<SCEVConstant>(SE.getConstant(S->getType(), 1));

  if (auto *Constant = dyn_cast<SCEVConstant>(S))
    return std::make_pair(Constant, SE.getConstant(S->getType(), 1));

  auto *AddRec = dyn_cast<SCEVAddRecExpr>(S);
  if (AddRec) {
    auto *StartExpr = AddRec->getStart();
    if (StartExpr->isZero()) {
      auto StepPair = extractConstantFactor(AddRec->getStepRecurrence(SE), SE);
      auto *LeftOverAddRec =
          SE.getAddRecExpr(StartExpr, StepPair.second, AddRec->getLoop(),
                           AddRec->getNoWrapFlags());
      return std::make_pair(StepPair.first, LeftOverAddRec);
    }
    return std::make_pair(ConstPart, S);
  }

  if (auto *Add = dyn_cast<SCEVAddExpr>(S)) {
    SmallVector<const SCEV *, 4> LeftOvers;
    auto Op0Pair = extractConstantFactor(Add->getOperand(0), SE);
    auto *Factor = Op0Pair.first;
    if (SE.isKnownNegative(Factor)) {
      Factor = cast<SCEVConstant>(SE.getNegativeSCEV(Factor));
      LeftOvers.push_back(SE.getNegativeSCEV(Op0Pair.second));
    } else {
      LeftOvers.push_back(Op0Pair.second);
    }

    for (unsigned u = 1, e = Add->getNumOperands(); u < e; u++) {
      auto OpUPair = extractConstantFactor(Add->getOperand(u), SE);
      // TODO: Use something smarter than equality here, e.g., gcd.
      if (Factor == OpUPair.first)
        LeftOvers.push_back(OpUPair.second);
      else if (Factor == SE.getNegativeSCEV(OpUPair.first))
        LeftOvers.push_back(SE.getNegativeSCEV(OpUPair.second));
      else
        return std::make_pair(ConstPart, S);
    }

    auto *NewAdd = SE.getAddExpr(LeftOvers, Add->getNoWrapFlags());
    return std::make_pair(Factor, NewAdd);
  }

  auto *Mul = dyn_cast<SCEVMulExpr>(S);
  if (!Mul)
    return std::make_pair(ConstPart, S);

  SmallVector<const SCEV *, 4> LeftOvers;
  for (auto *Op : Mul->operands())
    if (isa<SCEVConstant>(Op))
      ConstPart = cast<SCEVConstant>(SE.getMulExpr(ConstPart, Op));
    else
      LeftOvers.push_back(Op);

  return std::make_pair(ConstPart, SE.getMulExpr(LeftOvers));
}
} // namespace polly
