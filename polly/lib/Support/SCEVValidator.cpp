
#include "polly/Support/SCEVValidator.h"
#include "polly/ScopInfo.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Support/Debug.h"
#include <vector>

using namespace llvm;
using namespace polly;

#define DEBUG_TYPE "polly-scev-validator"

namespace SCEVType {
/// @brief The type of a SCEV
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
}

/// @brief The result the validator returns for a SCEV expression.
class ValidatorResult {
  /// @brief The type of the expression
  SCEVType::TYPE Type;

  /// @brief The set of Parameters in the expression.
  std::vector<const SCEV *> Parameters;

public:
  /// @brief The copy constructor
  ValidatorResult(const ValidatorResult &Source) {
    Type = Source.Type;
    Parameters = Source.Parameters;
  }

  /// @brief Construct a result with a certain type and no parameters.
  ValidatorResult(SCEVType::TYPE Type) : Type(Type) {
    assert(Type != SCEVType::PARAM && "Did you forget to pass the parameter");
  }

  /// @brief Construct a result with a certain type and a single parameter.
  ValidatorResult(SCEVType::TYPE Type, const SCEV *Expr) : Type(Type) {
    Parameters.push_back(Expr);
  }

  /// @brief Get the type of the ValidatorResult.
  SCEVType::TYPE getType() { return Type; }

  /// @brief Is the analyzed SCEV constant during the execution of the SCoP.
  bool isConstant() { return Type == SCEVType::INT || Type == SCEVType::PARAM; }

  /// @brief Is the analyzed SCEV valid.
  bool isValid() { return Type != SCEVType::INVALID; }

  /// @brief Is the analyzed SCEV of Type IV.
  bool isIV() { return Type == SCEVType::IV; }

  /// @brief Is the analyzed SCEV of Type INT.
  bool isINT() { return Type == SCEVType::INT; }

  /// @brief Is the analyzed SCEV of Type PARAM.
  bool isPARAM() { return Type == SCEVType::PARAM; }

  /// @brief Get the parameters of this validator result.
  std::vector<const SCEV *> getParameters() { return Parameters; }

  /// @brief Add the parameters of Source to this result.
  void addParamsFrom(const ValidatorResult &Source) {
    Parameters.insert(Parameters.end(), Source.Parameters.begin(),
                      Source.Parameters.end());
  }

  /// @brief Merge a result.
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

/// Check if a SCEV is valid in a SCoP.
struct SCEVValidator
    : public SCEVVisitor<SCEVValidator, class ValidatorResult> {
private:
  const Region *R;
  ScalarEvolution &SE;
  const Value *BaseAddress;
  InvariantLoadsSetTy *ILS;

public:
  SCEVValidator(const Region *R, ScalarEvolution &SE, const Value *BaseAddress,
                InvariantLoadsSetTy *ILS)
      : R(R), SE(SE), BaseAddress(BaseAddress), ILS(ILS) {}

  class ValidatorResult visitConstant(const SCEVConstant *Constant) {
    return ValidatorResult(SCEVType::INT);
  }

  class ValidatorResult visitTruncateExpr(const SCEVTruncateExpr *Expr) {
    ValidatorResult Op = visit(Expr->getOperand());

    switch (Op.getType()) {
    case SCEVType::INT:
    case SCEVType::PARAM:
      // We currently do not represent a truncate expression as an affine
      // expression. If it is constant during Scop execution, we treat it as a
      // parameter.
      return ValidatorResult(SCEVType::PARAM, Expr);
    case SCEVType::IV:
      DEBUG(dbgs() << "INVALID: Truncation of SCEVType::IV expression");
      return ValidatorResult(SCEVType::INVALID);
    case SCEVType::INVALID:
      return Op;
    }

    llvm_unreachable("Unknown SCEVType");
  }

  class ValidatorResult visitZeroExtendExpr(const SCEVZeroExtendExpr *Expr) {
    ValidatorResult Op = visit(Expr->getOperand());

    switch (Op.getType()) {
    case SCEVType::INT:
    case SCEVType::PARAM:
      // We currently do not represent a truncate expression as an affine
      // expression. If it is constant during Scop execution, we treat it as a
      // parameter.
      return ValidatorResult(SCEVType::PARAM, Expr);
    case SCEVType::IV:
      DEBUG(dbgs() << "INVALID: ZeroExtend of SCEVType::IV expression");
      return ValidatorResult(SCEVType::INVALID);
    case SCEVType::INVALID:
      return Op;
    }

    llvm_unreachable("Unknown SCEVType");
  }

  class ValidatorResult visitSignExtendExpr(const SCEVSignExtendExpr *Expr) {
    // We currently allow only signed SCEV expressions. In the case of a
    // signed value, a sign extend is a noop.
    //
    // TODO: Reconsider this when we add support for unsigned values.
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

    // TODO: Check for NSW and NUW.
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

    // TODO: Check for NSW and NUW.
    return Return;
  }

  class ValidatorResult visitUDivExpr(const SCEVUDivExpr *Expr) {
    ValidatorResult LHS = visit(Expr->getLHS());
    ValidatorResult RHS = visit(Expr->getRHS());

    // We currently do not represent an unsigned division as an affine
    // expression. If the division is constant during Scop execution we treat it
    // as a parameter, otherwise we bail out.
    if (LHS.isConstant() && RHS.isConstant())
      return ValidatorResult(SCEVType::PARAM, Expr);

    DEBUG(dbgs() << "INVALID: unsigned division of non-constant expressions");
    return ValidatorResult(SCEVType::INVALID);
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

    if (R->contains(Expr->getLoop())) {
      if (Recurrence.isINT()) {
        ValidatorResult Result(SCEVType::IV);
        Result.addParamsFrom(Start);
        return Result;
      }

      DEBUG(dbgs() << "INVALID: AddRec within scop has non-int"
                      "recurrence part");
      return ValidatorResult(SCEVType::INVALID);
    }

    assert(Start.isConstant() && Recurrence.isConstant() &&
           "Expected 'Start' and 'Recurrence' to be constant");

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
    // We do not support unsigned operations. If 'Expr' is constant during Scop
    // execution we treat this as a parameter, otherwise we bail out.
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

  ValidatorResult visitLoadInstruction(Instruction *I, const SCEV *S) {
    if (R->contains(I) && ILS) {
      ILS->insert(cast<LoadInst>(I));
      return ValidatorResult(SCEVType::PARAM, S);
    }

    return visitGenericInst(I, S);
  }

  ValidatorResult visitSDivInstruction(Instruction *SDiv, const SCEV *S) {
    assert(SDiv->getOpcode() == Instruction::SDiv &&
           "Assumed SDiv instruction!");

    auto *Divisor = SDiv->getOperand(1);
    auto *CI = dyn_cast<ConstantInt>(Divisor);
    if (!CI)
      return visitGenericInst(SDiv, S);

    auto *Dividend = SDiv->getOperand(0);
    auto *DividendSCEV = SE.getSCEV(Dividend);
    return visit(DividendSCEV);
  }

  ValidatorResult visitSRemInstruction(Instruction *SRem, const SCEV *S) {
    assert(SRem->getOpcode() == Instruction::SRem &&
           "Assumed SRem instruction!");

    auto *Divisor = SRem->getOperand(1);
    auto *CI = dyn_cast<ConstantInt>(Divisor);
    if (!CI)
      return visitGenericInst(SRem, S);

    auto *Dividend = SRem->getOperand(0);
    auto *DividendSCEV = SE.getSCEV(Dividend);
    return visit(DividendSCEV);
  }

  ValidatorResult visitUnknown(const SCEVUnknown *Expr) {
    Value *V = Expr->getValue();

    // TODO: FIXME: IslExprBuilder is not capable of producing valid code
    //              for arbitrary pointer expressions at the moment. Until
    //              this is fixed we disallow pointer expressions completely.
    if (Expr->getType()->isPointerTy()) {
      DEBUG(dbgs() << "INVALID: UnknownExpr is a pointer type [FIXME]");
      return ValidatorResult(SCEVType::INVALID);
    }

    if (!Expr->getType()->isIntegerTy()) {
      DEBUG(dbgs() << "INVALID: UnknownExpr is not an integer");
      return ValidatorResult(SCEVType::INVALID);
    }

    if (isa<UndefValue>(V)) {
      DEBUG(dbgs() << "INVALID: UnknownExpr references an undef value");
      return ValidatorResult(SCEVType::INVALID);
    }

    if (BaseAddress == V) {
      DEBUG(dbgs() << "INVALID: UnknownExpr references BaseAddress\n");
      return ValidatorResult(SCEVType::INVALID);
    }

    if (Instruction *I = dyn_cast<Instruction>(Expr->getValue())) {
      switch (I->getOpcode()) {
      case Instruction::Load:
        return visitLoadInstruction(I, Expr);
      case Instruction::SDiv:
        return visitSDivInstruction(I, Expr);
      case Instruction::SRem:
        return visitSRemInstruction(I, Expr);
      default:
        return visitGenericInst(I, Expr);
      }
    }

    return ValidatorResult(SCEVType::PARAM, Expr);
  }
};

/// @brief Check whether a SCEV refers to an SSA name defined inside a region.
///
struct SCEVInRegionDependences
    : public SCEVVisitor<SCEVInRegionDependences, bool> {
public:
  /// Returns true when the SCEV has SSA names defined in region R.
  static bool hasDependences(const SCEV *S, const Region *R) {
    SCEVInRegionDependences Ignore(R);
    return Ignore.visit(S);
  }

  SCEVInRegionDependences(const Region *R) : R(R) {}

  bool visit(const SCEV *Expr) {
    return SCEVVisitor<SCEVInRegionDependences, bool>::visit(Expr);
  }

  bool visitConstant(const SCEVConstant *Constant) { return false; }

  bool visitTruncateExpr(const SCEVTruncateExpr *Expr) {
    return visit(Expr->getOperand());
  }

  bool visitZeroExtendExpr(const SCEVZeroExtendExpr *Expr) {
    return visit(Expr->getOperand());
  }

  bool visitSignExtendExpr(const SCEVSignExtendExpr *Expr) {
    return visit(Expr->getOperand());
  }

  bool visitAddExpr(const SCEVAddExpr *Expr) {
    for (int i = 0, e = Expr->getNumOperands(); i < e; ++i)
      if (visit(Expr->getOperand(i)))
        return true;

    return false;
  }

  bool visitMulExpr(const SCEVMulExpr *Expr) {
    for (int i = 0, e = Expr->getNumOperands(); i < e; ++i)
      if (visit(Expr->getOperand(i)))
        return true;

    return false;
  }

  bool visitUDivExpr(const SCEVUDivExpr *Expr) {
    if (visit(Expr->getLHS()))
      return true;

    if (visit(Expr->getRHS()))
      return true;

    return false;
  }

  bool visitAddRecExpr(const SCEVAddRecExpr *Expr) {
    if (visit(Expr->getStart()))
      return true;

    for (size_t i = 0; i < Expr->getNumOperands(); ++i)
      if (visit(Expr->getOperand(i)))
        return true;

    return false;
  }

  bool visitSMaxExpr(const SCEVSMaxExpr *Expr) {
    for (size_t i = 0; i < Expr->getNumOperands(); ++i)
      if (visit(Expr->getOperand(i)))
        return true;

    return false;
  }

  bool visitUMaxExpr(const SCEVUMaxExpr *Expr) {
    for (size_t i = 0; i < Expr->getNumOperands(); ++i)
      if (visit(Expr->getOperand(i)))
        return true;

    return false;
  }

  bool visitUnknown(const SCEVUnknown *Expr) {
    Instruction *Inst = dyn_cast<Instruction>(Expr->getValue());

    // Return true when Inst is defined inside the region R.
    if (Inst && R->contains(Inst))
      return true;

    return false;
  }

private:
  const Region *R;
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
  SetVector<Value *> &Values;

public:
  SCEVFindValues(SetVector<Value *> &Values) : Values(Values) {}

  bool follow(const SCEV *S) {
    if (const SCEVUnknown *Unknown = dyn_cast<SCEVUnknown>(S))
      Values.insert(Unknown->getValue());
    return true;
  }
  bool isDone() { return false; }
};

void findValues(const SCEV *Expr, SetVector<Value *> &Values) {
  SCEVFindValues FindValues(Values);
  SCEVTraversal<SCEVFindValues> ST(FindValues);
  ST.visitAll(Expr);
}

bool hasScalarDepsInsideRegion(const SCEV *Expr, const Region *R) {
  return SCEVInRegionDependences::hasDependences(Expr, R);
}

bool isAffineExpr(const Region *R, const SCEV *Expr, ScalarEvolution &SE,
                  const Value *BaseAddress, InvariantLoadsSetTy *ILS) {
  if (isa<SCEVCouldNotCompute>(Expr))
    return false;

  SCEVValidator Validator(R, SE, BaseAddress, ILS);
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

static bool isAffineParamExpr(Value *V, const Region *R, ScalarEvolution &SE,
                              std::vector<const SCEV *> &Params) {
  auto *E = SE.getSCEV(V);
  if (isa<SCEVCouldNotCompute>(E))
    return false;

  SCEVValidator Validator(R, SE, nullptr, nullptr);
  ValidatorResult Result = Validator.visit(E);
  if (!Result.isConstant())
    return false;

  auto ResultParams = Result.getParameters();
  Params.insert(Params.end(), ResultParams.begin(), ResultParams.end());

  return true;
}

bool isAffineParamConstraint(Value *V, const Region *R, ScalarEvolution &SE,
                             std::vector<const SCEV *> &Params, bool OrExpr) {
  if (auto *ICmp = dyn_cast<ICmpInst>(V)) {
    return isAffineParamConstraint(ICmp->getOperand(0), R, SE, Params, true) &&
           isAffineParamConstraint(ICmp->getOperand(1), R, SE, Params, true);
  } else if (auto *BinOp = dyn_cast<BinaryOperator>(V)) {
    auto Opcode = BinOp->getOpcode();
    if (Opcode == Instruction::And || Opcode == Instruction::Or)
      return isAffineParamConstraint(BinOp->getOperand(0), R, SE, Params,
                                     false) &&
             isAffineParamConstraint(BinOp->getOperand(1), R, SE, Params,
                                     false);
    /* Fall through */
  }

  if (!OrExpr)
    return false;

  return isAffineParamExpr(V, R, SE, Params);
}

std::vector<const SCEV *> getParamsInAffineExpr(const Region *R,
                                                const SCEV *Expr,
                                                ScalarEvolution &SE,
                                                const Value *BaseAddress) {
  if (isa<SCEVCouldNotCompute>(Expr))
    return std::vector<const SCEV *>();

  InvariantLoadsSetTy ILS;
  SCEVValidator Validator(R, SE, BaseAddress, &ILS);
  ValidatorResult Result = Validator.visit(Expr);
  assert(Result.isValid() && "Requested parameters for an invalid SCEV!");

  return Result.getParameters();
}

std::pair<const SCEV *, const SCEV *>
extractConstantFactor(const SCEV *S, ScalarEvolution &SE) {

  const SCEV *LeftOver = SE.getConstant(S->getType(), 1);
  const SCEV *ConstPart = SE.getConstant(S->getType(), 1);

  const SCEVMulExpr *M = dyn_cast<SCEVMulExpr>(S);
  if (!M)
    return std::make_pair(ConstPart, S);

  for (const SCEV *Op : M->operands())
    if (isa<SCEVConstant>(Op))
      ConstPart = SE.getMulExpr(ConstPart, Op);
    else
      LeftOver = SE.getMulExpr(LeftOver, Op);

  return std::make_pair(ConstPart, LeftOver);
}
}
