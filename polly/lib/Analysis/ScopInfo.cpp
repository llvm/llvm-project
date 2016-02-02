//===--------- ScopInfo.cpp  - Create Scops from LLVM IR ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Create a polyhedral description for a static control flow region.
//
// The pass creates a polyhedral description of the Scops detected by the Scop
// detection derived from their LLVM-IR code.
//
// This representation is shared among several tools in the polyhedral
// community, which are e.g. Cloog, Pluto, Loopo, Graphite.
//
//===----------------------------------------------------------------------===//

#include "polly/ScopInfo.h"
#include "polly/LinkAllPasses.h"
#include "polly/Options.h"
#include "polly/Support/GICHelper.h"
#include "polly/Support/SCEVValidator.h"
#include "polly/Support/ScopHelper.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/RegionIterator.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Support/Debug.h"
#include "isl/aff.h"
#include "isl/constraint.h"
#include "isl/local_space.h"
#include "isl/map.h"
#include "isl/options.h"
#include "isl/printer.h"
#include "isl/schedule.h"
#include "isl/schedule_node.h"
#include "isl/set.h"
#include "isl/union_map.h"
#include "isl/union_set.h"
#include "isl/val.h"
#include <sstream>
#include <string>
#include <vector>

using namespace llvm;
using namespace polly;

#define DEBUG_TYPE "polly-scops"

STATISTIC(ScopFound, "Number of valid Scops");
STATISTIC(RichScopFound, "Number of Scops containing a loop");

// The maximal number of basic sets we allow during domain construction to
// be created. More complex scops will result in very high compile time and
// are also unlikely to result in good code
static int const MaxConjunctsInDomain = 20;

static cl::opt<bool> ModelReadOnlyScalars(
    "polly-analyze-read-only-scalars",
    cl::desc("Model read-only scalar values in the scop description"),
    cl::Hidden, cl::ZeroOrMore, cl::init(true), cl::cat(PollyCategory));

// Multiplicative reductions can be disabled separately as these kind of
// operations can overflow easily. Additive reductions and bit operations
// are in contrast pretty stable.
static cl::opt<bool> DisableMultiplicativeReductions(
    "polly-disable-multiplicative-reductions",
    cl::desc("Disable multiplicative reductions"), cl::Hidden, cl::ZeroOrMore,
    cl::init(false), cl::cat(PollyCategory));

static cl::opt<unsigned> RunTimeChecksMaxParameters(
    "polly-rtc-max-parameters",
    cl::desc("The maximal number of parameters allowed in RTCs."), cl::Hidden,
    cl::ZeroOrMore, cl::init(8), cl::cat(PollyCategory));

static cl::opt<unsigned> RunTimeChecksMaxArraysPerGroup(
    "polly-rtc-max-arrays-per-group",
    cl::desc("The maximal number of arrays to compare in each alias group."),
    cl::Hidden, cl::ZeroOrMore, cl::init(20), cl::cat(PollyCategory));
static cl::opt<std::string> UserContextStr(
    "polly-context", cl::value_desc("isl parameter set"),
    cl::desc("Provide additional constraints on the context parameters"),
    cl::init(""), cl::cat(PollyCategory));

static cl::opt<bool> DetectReductions("polly-detect-reductions",
                                      cl::desc("Detect and exploit reductions"),
                                      cl::Hidden, cl::ZeroOrMore,
                                      cl::init(true), cl::cat(PollyCategory));

static cl::opt<int> MaxDisjunctsAssumed(
    "polly-max-disjuncts-assumed",
    cl::desc("The maximal number of disjuncts we allow in the assumption "
             "context (this bounds compile time)"),
    cl::Hidden, cl::ZeroOrMore, cl::init(150), cl::cat(PollyCategory));

static cl::opt<bool> IgnoreIntegerWrapping(
    "polly-ignore-integer-wrapping",
    cl::desc("Do not build run-time checks to proof absence of integer "
             "wrapping"),
    cl::Hidden, cl::ZeroOrMore, cl::init(false), cl::cat(PollyCategory));

//===----------------------------------------------------------------------===//

// Create a sequence of two schedules. Either argument may be null and is
// interpreted as the empty schedule. Can also return null if both schedules are
// empty.
static __isl_give isl_schedule *
combineInSequence(__isl_take isl_schedule *Prev,
                  __isl_take isl_schedule *Succ) {
  if (!Prev)
    return Succ;
  if (!Succ)
    return Prev;

  return isl_schedule_sequence(Prev, Succ);
}

static __isl_give isl_set *addRangeBoundsToSet(__isl_take isl_set *S,
                                               const ConstantRange &Range,
                                               int dim,
                                               enum isl_dim_type type) {
  isl_val *V;
  isl_ctx *ctx = isl_set_get_ctx(S);

  bool useLowerUpperBound = Range.isSignWrappedSet() && !Range.isFullSet();
  const auto LB = useLowerUpperBound ? Range.getLower() : Range.getSignedMin();
  V = isl_valFromAPInt(ctx, LB, true);
  isl_set *SLB = isl_set_lower_bound_val(isl_set_copy(S), type, dim, V);

  const auto UB = useLowerUpperBound ? Range.getUpper() : Range.getSignedMax();
  V = isl_valFromAPInt(ctx, UB, true);
  if (useLowerUpperBound)
    V = isl_val_sub_ui(V, 1);
  isl_set *SUB = isl_set_upper_bound_val(S, type, dim, V);

  if (useLowerUpperBound)
    return isl_set_union(SLB, SUB);
  else
    return isl_set_intersect(SLB, SUB);
}

static const ScopArrayInfo *identifyBasePtrOriginSAI(Scop *S, Value *BasePtr) {
  LoadInst *BasePtrLI = dyn_cast<LoadInst>(BasePtr);
  if (!BasePtrLI)
    return nullptr;

  if (!S->getRegion().contains(BasePtrLI))
    return nullptr;

  ScalarEvolution &SE = *S->getSE();

  auto *OriginBaseSCEV =
      SE.getPointerBase(SE.getSCEV(BasePtrLI->getPointerOperand()));
  if (!OriginBaseSCEV)
    return nullptr;

  auto *OriginBaseSCEVUnknown = dyn_cast<SCEVUnknown>(OriginBaseSCEV);
  if (!OriginBaseSCEVUnknown)
    return nullptr;

  return S->getScopArrayInfo(OriginBaseSCEVUnknown->getValue(),
                             ScopArrayInfo::MK_Array);
}

ScopArrayInfo::ScopArrayInfo(Value *BasePtr, Type *ElementType, isl_ctx *Ctx,
                             ArrayRef<const SCEV *> Sizes, enum MemoryKind Kind,
                             const DataLayout &DL, Scop *S)
    : BasePtr(BasePtr), ElementType(ElementType), Kind(Kind), DL(DL), S(*S) {
  std::string BasePtrName =
      getIslCompatibleName("MemRef_", BasePtr, Kind == MK_PHI ? "__phi" : "");
  Id = isl_id_alloc(Ctx, BasePtrName.c_str(), this);

  updateSizes(Sizes);
  BasePtrOriginSAI = identifyBasePtrOriginSAI(S, BasePtr);
  if (BasePtrOriginSAI)
    const_cast<ScopArrayInfo *>(BasePtrOriginSAI)->addDerivedSAI(this);
}

__isl_give isl_space *ScopArrayInfo::getSpace() const {
  auto Space =
      isl_space_set_alloc(isl_id_get_ctx(Id), 0, getNumberOfDimensions());
  Space = isl_space_set_tuple_id(Space, isl_dim_set, isl_id_copy(Id));
  return Space;
}

bool ScopArrayInfo::updateSizes(ArrayRef<const SCEV *> NewSizes) {
  int SharedDims = std::min(NewSizes.size(), DimensionSizes.size());
  int ExtraDimsNew = NewSizes.size() - SharedDims;
  int ExtraDimsOld = DimensionSizes.size() - SharedDims;
  for (int i = 0; i < SharedDims; i++)
    if (NewSizes[i + ExtraDimsNew] != DimensionSizes[i + ExtraDimsOld])
      return false;

  if (DimensionSizes.size() >= NewSizes.size())
    return true;

  DimensionSizes.clear();
  DimensionSizes.insert(DimensionSizes.begin(), NewSizes.begin(),
                        NewSizes.end());
  for (isl_pw_aff *Size : DimensionSizesPw)
    isl_pw_aff_free(Size);
  DimensionSizesPw.clear();
  for (const SCEV *Expr : DimensionSizes) {
    isl_pw_aff *Size = S.getPwAff(Expr);
    DimensionSizesPw.push_back(Size);
  }
  return true;
}

ScopArrayInfo::~ScopArrayInfo() {
  isl_id_free(Id);
  for (isl_pw_aff *Size : DimensionSizesPw)
    isl_pw_aff_free(Size);
}

std::string ScopArrayInfo::getName() const { return isl_id_get_name(Id); }

int ScopArrayInfo::getElemSizeInBytes() const {
  return DL.getTypeAllocSize(ElementType);
}

isl_id *ScopArrayInfo::getBasePtrId() const { return isl_id_copy(Id); }

void ScopArrayInfo::dump() const { print(errs()); }

void ScopArrayInfo::print(raw_ostream &OS, bool SizeAsPwAff) const {
  OS.indent(8) << *getElementType() << " " << getName();
  if (getNumberOfDimensions() > 0)
    OS << "[*]";
  for (unsigned u = 1; u < getNumberOfDimensions(); u++) {
    OS << "[";

    if (SizeAsPwAff) {
      auto Size = getDimensionSizePw(u);
      OS << " " << Size << " ";
      isl_pw_aff_free(Size);
    } else {
      OS << *getDimensionSize(u);
    }

    OS << "]";
  }

  OS << ";";

  if (BasePtrOriginSAI)
    OS << " [BasePtrOrigin: " << BasePtrOriginSAI->getName() << "]";

  OS << " // Element size " << getElemSizeInBytes() << "\n";
}

const ScopArrayInfo *
ScopArrayInfo::getFromAccessFunction(__isl_keep isl_pw_multi_aff *PMA) {
  isl_id *Id = isl_pw_multi_aff_get_tuple_id(PMA, isl_dim_out);
  assert(Id && "Output dimension didn't have an ID");
  return getFromId(Id);
}

const ScopArrayInfo *ScopArrayInfo::getFromId(isl_id *Id) {
  void *User = isl_id_get_user(Id);
  const ScopArrayInfo *SAI = static_cast<ScopArrayInfo *>(User);
  isl_id_free(Id);
  return SAI;
}

void MemoryAccess::updateDimensionality() {
  auto ArraySpace = getScopArrayInfo()->getSpace();
  auto AccessSpace = isl_space_range(isl_map_get_space(AccessRelation));

  auto DimsArray = isl_space_dim(ArraySpace, isl_dim_set);
  auto DimsAccess = isl_space_dim(AccessSpace, isl_dim_set);
  auto DimsMissing = DimsArray - DimsAccess;

  auto Map = isl_map_from_domain_and_range(isl_set_universe(AccessSpace),
                                           isl_set_universe(ArraySpace));

  for (unsigned i = 0; i < DimsMissing; i++)
    Map = isl_map_fix_si(Map, isl_dim_out, i, 0);

  for (unsigned i = DimsMissing; i < DimsArray; i++)
    Map = isl_map_equate(Map, isl_dim_in, i - DimsMissing, isl_dim_out, i);

  AccessRelation = isl_map_apply_range(AccessRelation, Map);

  assumeNoOutOfBound();
}

const std::string
MemoryAccess::getReductionOperatorStr(MemoryAccess::ReductionType RT) {
  switch (RT) {
  case MemoryAccess::RT_NONE:
    llvm_unreachable("Requested a reduction operator string for a memory "
                     "access which isn't a reduction");
  case MemoryAccess::RT_ADD:
    return "+";
  case MemoryAccess::RT_MUL:
    return "*";
  case MemoryAccess::RT_BOR:
    return "|";
  case MemoryAccess::RT_BXOR:
    return "^";
  case MemoryAccess::RT_BAND:
    return "&";
  }
  llvm_unreachable("Unknown reduction type");
  return "";
}

/// @brief Return the reduction type for a given binary operator
static MemoryAccess::ReductionType getReductionType(const BinaryOperator *BinOp,
                                                    const Instruction *Load) {
  if (!BinOp)
    return MemoryAccess::RT_NONE;
  switch (BinOp->getOpcode()) {
  case Instruction::FAdd:
    if (!BinOp->hasUnsafeAlgebra())
      return MemoryAccess::RT_NONE;
  // Fall through
  case Instruction::Add:
    return MemoryAccess::RT_ADD;
  case Instruction::Or:
    return MemoryAccess::RT_BOR;
  case Instruction::Xor:
    return MemoryAccess::RT_BXOR;
  case Instruction::And:
    return MemoryAccess::RT_BAND;
  case Instruction::FMul:
    if (!BinOp->hasUnsafeAlgebra())
      return MemoryAccess::RT_NONE;
  // Fall through
  case Instruction::Mul:
    if (DisableMultiplicativeReductions)
      return MemoryAccess::RT_NONE;
    return MemoryAccess::RT_MUL;
  default:
    return MemoryAccess::RT_NONE;
  }
}

/// @brief Derive the individual index expressions from a GEP instruction
///
/// This function optimistically assumes the GEP references into a fixed size
/// array. If this is actually true, this function returns a list of array
/// subscript expressions as SCEV as well as a list of integers describing
/// the size of the individual array dimensions. Both lists have either equal
/// length of the size list is one element shorter in case there is no known
/// size available for the outermost array dimension.
///
/// @param GEP The GetElementPtr instruction to analyze.
///
/// @return A tuple with the subscript expressions and the dimension sizes.
static std::tuple<std::vector<const SCEV *>, std::vector<int>>
getIndexExpressionsFromGEP(GetElementPtrInst *GEP, ScalarEvolution &SE) {
  std::vector<const SCEV *> Subscripts;
  std::vector<int> Sizes;

  Type *Ty = GEP->getPointerOperandType();

  bool DroppedFirstDim = false;

  for (unsigned i = 1; i < GEP->getNumOperands(); i++) {

    const SCEV *Expr = SE.getSCEV(GEP->getOperand(i));

    if (i == 1) {
      if (auto PtrTy = dyn_cast<PointerType>(Ty)) {
        Ty = PtrTy->getElementType();
      } else if (auto ArrayTy = dyn_cast<ArrayType>(Ty)) {
        Ty = ArrayTy->getElementType();
      } else {
        Subscripts.clear();
        Sizes.clear();
        break;
      }
      if (auto Const = dyn_cast<SCEVConstant>(Expr))
        if (Const->getValue()->isZero()) {
          DroppedFirstDim = true;
          continue;
        }
      Subscripts.push_back(Expr);
      continue;
    }

    auto ArrayTy = dyn_cast<ArrayType>(Ty);
    if (!ArrayTy) {
      Subscripts.clear();
      Sizes.clear();
      break;
    }

    Subscripts.push_back(Expr);
    if (!(DroppedFirstDim && i == 2))
      Sizes.push_back(ArrayTy->getNumElements());

    Ty = ArrayTy->getElementType();
  }

  return std::make_tuple(Subscripts, Sizes);
}

MemoryAccess::~MemoryAccess() {
  isl_id_free(Id);
  isl_map_free(AccessRelation);
  isl_map_free(NewAccessRelation);
}

const ScopArrayInfo *MemoryAccess::getScopArrayInfo() const {
  isl_id *ArrayId = getArrayId();
  void *User = isl_id_get_user(ArrayId);
  const ScopArrayInfo *SAI = static_cast<ScopArrayInfo *>(User);
  isl_id_free(ArrayId);
  return SAI;
}

__isl_give isl_id *MemoryAccess::getArrayId() const {
  return isl_map_get_tuple_id(AccessRelation, isl_dim_out);
}

__isl_give isl_pw_multi_aff *MemoryAccess::applyScheduleToAccessRelation(
    __isl_take isl_union_map *USchedule) const {
  isl_map *Schedule, *ScheduledAccRel;
  isl_union_set *UDomain;

  UDomain = isl_union_set_from_set(getStatement()->getDomain());
  USchedule = isl_union_map_intersect_domain(USchedule, UDomain);
  Schedule = isl_map_from_union_map(USchedule);
  ScheduledAccRel = isl_map_apply_domain(getAccessRelation(), Schedule);
  return isl_pw_multi_aff_from_map(ScheduledAccRel);
}

__isl_give isl_map *MemoryAccess::getOriginalAccessRelation() const {
  return isl_map_copy(AccessRelation);
}

std::string MemoryAccess::getOriginalAccessRelationStr() const {
  return stringFromIslObj(AccessRelation);
}

__isl_give isl_space *MemoryAccess::getOriginalAccessRelationSpace() const {
  return isl_map_get_space(AccessRelation);
}

__isl_give isl_map *MemoryAccess::getNewAccessRelation() const {
  return isl_map_copy(NewAccessRelation);
}

std::string MemoryAccess::getNewAccessRelationStr() const {
  return stringFromIslObj(NewAccessRelation);
}

__isl_give isl_basic_map *
MemoryAccess::createBasicAccessMap(ScopStmt *Statement) {
  isl_space *Space = isl_space_set_alloc(Statement->getIslCtx(), 0, 1);
  Space = isl_space_align_params(Space, Statement->getDomainSpace());

  return isl_basic_map_from_domain_and_range(
      isl_basic_set_universe(Statement->getDomainSpace()),
      isl_basic_set_universe(Space));
}

// Formalize no out-of-bound access assumption
//
// When delinearizing array accesses we optimistically assume that the
// delinearized accesses do not access out of bound locations (the subscript
// expression of each array evaluates for each statement instance that is
// executed to a value that is larger than zero and strictly smaller than the
// size of the corresponding dimension). The only exception is the outermost
// dimension for which we do not need to assume any upper bound.  At this point
// we formalize this assumption to ensure that at code generation time the
// relevant run-time checks can be generated.
//
// To find the set of constraints necessary to avoid out of bound accesses, we
// first build the set of data locations that are not within array bounds. We
// then apply the reverse access relation to obtain the set of iterations that
// may contain invalid accesses and reduce this set of iterations to the ones
// that are actually executed by intersecting them with the domain of the
// statement. If we now project out all loop dimensions, we obtain a set of
// parameters that may cause statement instances to be executed that may
// possibly yield out of bound memory accesses. The complement of these
// constraints is the set of constraints that needs to be assumed to ensure such
// statement instances are never executed.
void MemoryAccess::assumeNoOutOfBound() {
  isl_space *Space = isl_space_range(getOriginalAccessRelationSpace());
  isl_set *Outside = isl_set_empty(isl_space_copy(Space));
  for (int i = 1, Size = isl_space_dim(Space, isl_dim_set); i < Size; ++i) {
    isl_local_space *LS = isl_local_space_from_space(isl_space_copy(Space));
    isl_pw_aff *Var =
        isl_pw_aff_var_on_domain(isl_local_space_copy(LS), isl_dim_set, i);
    isl_pw_aff *Zero = isl_pw_aff_zero_on_domain(LS);

    isl_set *DimOutside;

    DimOutside = isl_pw_aff_lt_set(isl_pw_aff_copy(Var), Zero);
    isl_pw_aff *SizeE = getScopArrayInfo()->getDimensionSizePw(i);
    SizeE = isl_pw_aff_add_dims(SizeE, isl_dim_in,
                                isl_space_dim(Space, isl_dim_set));
    SizeE = isl_pw_aff_set_tuple_id(SizeE, isl_dim_in,
                                    isl_space_get_tuple_id(Space, isl_dim_set));

    DimOutside = isl_set_union(DimOutside, isl_pw_aff_le_set(SizeE, Var));

    Outside = isl_set_union(Outside, DimOutside);
  }

  Outside = isl_set_apply(Outside, isl_map_reverse(getAccessRelation()));
  Outside = isl_set_intersect(Outside, Statement->getDomain());
  Outside = isl_set_params(Outside);

  // Remove divs to avoid the construction of overly complicated assumptions.
  // Doing so increases the set of parameter combinations that are assumed to
  // not appear. This is always save, but may make the resulting run-time check
  // bail out more often than strictly necessary.
  Outside = isl_set_remove_divs(Outside);
  Outside = isl_set_complement(Outside);
  Statement->getParent()->addAssumption(
      INBOUNDS, Outside,
      getAccessInstruction() ? getAccessInstruction()->getDebugLoc() : nullptr);
  isl_space_free(Space);
}

void MemoryAccess::computeBoundsOnAccessRelation(unsigned ElementSize) {
  ScalarEvolution *SE = Statement->getParent()->getSE();

  Value *Ptr = MemAccInst(getAccessInstruction()).getPointerOperand();
  if (!Ptr || !SE->isSCEVable(Ptr->getType()))
    return;

  auto *PtrSCEV = SE->getSCEV(Ptr);
  if (isa<SCEVCouldNotCompute>(PtrSCEV))
    return;

  auto *BasePtrSCEV = SE->getPointerBase(PtrSCEV);
  if (BasePtrSCEV && !isa<SCEVCouldNotCompute>(BasePtrSCEV))
    PtrSCEV = SE->getMinusSCEV(PtrSCEV, BasePtrSCEV);

  const ConstantRange &Range = SE->getSignedRange(PtrSCEV);
  if (Range.isFullSet())
    return;

  bool isWrapping = Range.isSignWrappedSet();
  unsigned BW = Range.getBitWidth();
  const auto LB = isWrapping ? Range.getLower() : Range.getSignedMin();
  const auto UB = isWrapping ? Range.getUpper() : Range.getSignedMax();

  auto Min = LB.sdiv(APInt(BW, ElementSize));
  auto Max = (UB - APInt(BW, 1)).sdiv(APInt(BW, ElementSize));

  isl_set *AccessRange = isl_map_range(isl_map_copy(AccessRelation));
  AccessRange =
      addRangeBoundsToSet(AccessRange, ConstantRange(Min, Max), 0, isl_dim_set);
  AccessRelation = isl_map_intersect_range(AccessRelation, AccessRange);
}

__isl_give isl_map *MemoryAccess::foldAccess(__isl_take isl_map *AccessRelation,
                                             ScopStmt *Statement) {
  int Size = Subscripts.size();

  for (int i = Size - 2; i >= 0; --i) {
    isl_space *Space;
    isl_map *MapOne, *MapTwo;
    isl_pw_aff *DimSize = Statement->getPwAff(Sizes[i]);

    isl_space *SpaceSize = isl_pw_aff_get_space(DimSize);
    isl_pw_aff_free(DimSize);
    isl_id *ParamId = isl_space_get_dim_id(SpaceSize, isl_dim_param, 0);

    Space = isl_map_get_space(AccessRelation);
    Space = isl_space_map_from_set(isl_space_range(Space));
    Space = isl_space_align_params(Space, SpaceSize);

    int ParamLocation = isl_space_find_dim_by_id(Space, isl_dim_param, ParamId);
    isl_id_free(ParamId);

    MapOne = isl_map_universe(isl_space_copy(Space));
    for (int j = 0; j < Size; ++j)
      MapOne = isl_map_equate(MapOne, isl_dim_in, j, isl_dim_out, j);
    MapOne = isl_map_lower_bound_si(MapOne, isl_dim_in, i + 1, 0);

    MapTwo = isl_map_universe(isl_space_copy(Space));
    for (int j = 0; j < Size; ++j)
      if (j < i || j > i + 1)
        MapTwo = isl_map_equate(MapTwo, isl_dim_in, j, isl_dim_out, j);

    isl_local_space *LS = isl_local_space_from_space(Space);
    isl_constraint *C;
    C = isl_equality_alloc(isl_local_space_copy(LS));
    C = isl_constraint_set_constant_si(C, -1);
    C = isl_constraint_set_coefficient_si(C, isl_dim_in, i, 1);
    C = isl_constraint_set_coefficient_si(C, isl_dim_out, i, -1);
    MapTwo = isl_map_add_constraint(MapTwo, C);
    C = isl_equality_alloc(LS);
    C = isl_constraint_set_coefficient_si(C, isl_dim_in, i + 1, 1);
    C = isl_constraint_set_coefficient_si(C, isl_dim_out, i + 1, -1);
    C = isl_constraint_set_coefficient_si(C, isl_dim_param, ParamLocation, 1);
    MapTwo = isl_map_add_constraint(MapTwo, C);
    MapTwo = isl_map_upper_bound_si(MapTwo, isl_dim_in, i + 1, -1);

    MapOne = isl_map_union(MapOne, MapTwo);
    AccessRelation = isl_map_apply_range(AccessRelation, MapOne);
  }
  return AccessRelation;
}

/// @brief Check if @p Expr is divisible by @p Size.
static bool isDivisible(const SCEV *Expr, unsigned Size, ScalarEvolution &SE) {

  // Only one factor needs to be divisible.
  if (auto *MulExpr = dyn_cast<SCEVMulExpr>(Expr)) {
    for (auto *FactorExpr : MulExpr->operands())
      if (isDivisible(FactorExpr, Size, SE))
        return true;
    return false;
  }

  // For other n-ary expressions (Add, AddRec, Max,...) all operands need
  // to be divisble.
  if (auto *NAryExpr = dyn_cast<SCEVNAryExpr>(Expr)) {
    for (auto *OpExpr : NAryExpr->operands())
      if (!isDivisible(OpExpr, Size, SE))
        return false;
    return true;
  }

  auto *SizeSCEV = SE.getConstant(Expr->getType(), Size);
  auto *UDivSCEV = SE.getUDivExpr(Expr, SizeSCEV);
  auto *MulSCEV = SE.getMulExpr(UDivSCEV, SizeSCEV);
  return MulSCEV == Expr;
}

void MemoryAccess::buildAccessRelation(const ScopArrayInfo *SAI) {
  assert(!AccessRelation && "AccessReltation already built");

  isl_ctx *Ctx = isl_id_get_ctx(Id);
  isl_id *BaseAddrId = SAI->getBasePtrId();

  if (!isAffine()) {
    // We overapproximate non-affine accesses with a possible access to the
    // whole array. For read accesses it does not make a difference, if an
    // access must or may happen. However, for write accesses it is important to
    // differentiate between writes that must happen and writes that may happen.
    AccessRelation = isl_map_from_basic_map(createBasicAccessMap(Statement));
    AccessRelation =
        isl_map_set_tuple_id(AccessRelation, isl_dim_out, BaseAddrId);

    computeBoundsOnAccessRelation(getElemSizeInBytes());
    return;
  }

  Scop &S = *getStatement()->getParent();
  isl_space *Space = isl_space_alloc(Ctx, 0, Statement->getNumIterators(), 0);
  AccessRelation = isl_map_universe(Space);

  for (int i = 0, Size = Subscripts.size(); i < Size; ++i) {
    isl_pw_aff *Affine = Statement->getPwAff(Subscripts[i]);

    if (Size == 1) {
      // For the non delinearized arrays, divide the access function of the last
      // subscript by the size of the elements in the array.
      //
      // A stride one array access in C expressed as A[i] is expressed in
      // LLVM-IR as something like A[i * elementsize]. This hides the fact that
      // two subsequent values of 'i' index two values that are stored next to
      // each other in memory. By this division we make this characteristic
      // obvious again. However, if the index is not divisible by the element
      // size we will bail out.
      isl_val *v = isl_val_int_from_si(Ctx, getElemSizeInBytes());
      Affine = isl_pw_aff_scale_down_val(Affine, v);

      if (!isDivisible(Subscripts[0], getElemSizeInBytes(), *S.getSE()))
        S.invalidate(ALIGNMENT, AccessInstruction->getDebugLoc());
    }

    isl_map *SubscriptMap = isl_map_from_pw_aff(Affine);

    AccessRelation = isl_map_flat_range_product(AccessRelation, SubscriptMap);
  }

  if (Sizes.size() > 1 && !isa<SCEVConstant>(Sizes[0]))
    AccessRelation = foldAccess(AccessRelation, Statement);

  Space = Statement->getDomainSpace();
  AccessRelation = isl_map_set_tuple_id(
      AccessRelation, isl_dim_in, isl_space_get_tuple_id(Space, isl_dim_set));
  AccessRelation =
      isl_map_set_tuple_id(AccessRelation, isl_dim_out, BaseAddrId);

  AccessRelation = isl_map_gist_domain(AccessRelation, Statement->getDomain());
  isl_space_free(Space);
}

MemoryAccess::MemoryAccess(ScopStmt *Stmt, Instruction *AccessInst,
                           AccessType Type, Value *BaseAddress,
                           unsigned ElemBytes, bool Affine,
                           ArrayRef<const SCEV *> Subscripts,
                           ArrayRef<const SCEV *> Sizes, Value *AccessValue,
                           ScopArrayInfo::MemoryKind Kind, StringRef BaseName)
    : Kind(Kind), AccType(Type), RedType(RT_NONE), Statement(Stmt),
      BaseAddr(BaseAddress), BaseName(BaseName), ElemBytes(ElemBytes),
      Sizes(Sizes.begin(), Sizes.end()), AccessInstruction(AccessInst),
      AccessValue(AccessValue), IsAffine(Affine),
      Subscripts(Subscripts.begin(), Subscripts.end()), AccessRelation(nullptr),
      NewAccessRelation(nullptr) {

  std::string IdName = "__polly_array_ref";
  Id = isl_id_alloc(Stmt->getParent()->getIslCtx(), IdName.c_str(), this);
}

void MemoryAccess::realignParams() {
  isl_space *ParamSpace = Statement->getParent()->getParamSpace();
  AccessRelation = isl_map_align_params(AccessRelation, ParamSpace);
}

const std::string MemoryAccess::getReductionOperatorStr() const {
  return MemoryAccess::getReductionOperatorStr(getReductionType());
}

__isl_give isl_id *MemoryAccess::getId() const { return isl_id_copy(Id); }

raw_ostream &polly::operator<<(raw_ostream &OS,
                               MemoryAccess::ReductionType RT) {
  if (RT == MemoryAccess::RT_NONE)
    OS << "NONE";
  else
    OS << MemoryAccess::getReductionOperatorStr(RT);
  return OS;
}

void MemoryAccess::print(raw_ostream &OS) const {
  switch (AccType) {
  case READ:
    OS.indent(12) << "ReadAccess :=\t";
    break;
  case MUST_WRITE:
    OS.indent(12) << "MustWriteAccess :=\t";
    break;
  case MAY_WRITE:
    OS.indent(12) << "MayWriteAccess :=\t";
    break;
  }
  OS << "[Reduction Type: " << getReductionType() << "] ";
  OS << "[Scalar: " << isScalarKind() << "]\n";
  OS.indent(16) << getOriginalAccessRelationStr() << ";\n";
  if (hasNewAccessRelation())
    OS.indent(11) << "new: " << getNewAccessRelationStr() << ";\n";
}

void MemoryAccess::dump() const { print(errs()); }

// Create a map in the size of the provided set domain, that maps from the
// one element of the provided set domain to another element of the provided
// set domain.
// The mapping is limited to all points that are equal in all but the last
// dimension and for which the last dimension of the input is strict smaller
// than the last dimension of the output.
//
//   getEqualAndLarger(set[i0, i1, ..., iX]):
//
//   set[i0, i1, ..., iX] -> set[o0, o1, ..., oX]
//     : i0 = o0, i1 = o1, ..., i(X-1) = o(X-1), iX < oX
//
static isl_map *getEqualAndLarger(isl_space *setDomain) {
  isl_space *Space = isl_space_map_from_set(setDomain);
  isl_map *Map = isl_map_universe(Space);
  unsigned lastDimension = isl_map_dim(Map, isl_dim_in) - 1;

  // Set all but the last dimension to be equal for the input and output
  //
  //   input[i0, i1, ..., iX] -> output[o0, o1, ..., oX]
  //     : i0 = o0, i1 = o1, ..., i(X-1) = o(X-1)
  for (unsigned i = 0; i < lastDimension; ++i)
    Map = isl_map_equate(Map, isl_dim_in, i, isl_dim_out, i);

  // Set the last dimension of the input to be strict smaller than the
  // last dimension of the output.
  //
  //   input[?,?,?,...,iX] -> output[?,?,?,...,oX] : iX < oX
  Map = isl_map_order_lt(Map, isl_dim_in, lastDimension, isl_dim_out,
                         lastDimension);
  return Map;
}

__isl_give isl_set *
MemoryAccess::getStride(__isl_take const isl_map *Schedule) const {
  isl_map *S = const_cast<isl_map *>(Schedule);
  isl_map *AccessRelation = getAccessRelation();
  isl_space *Space = isl_space_range(isl_map_get_space(S));
  isl_map *NextScatt = getEqualAndLarger(Space);

  S = isl_map_reverse(S);
  NextScatt = isl_map_lexmin(NextScatt);

  NextScatt = isl_map_apply_range(NextScatt, isl_map_copy(S));
  NextScatt = isl_map_apply_range(NextScatt, isl_map_copy(AccessRelation));
  NextScatt = isl_map_apply_domain(NextScatt, S);
  NextScatt = isl_map_apply_domain(NextScatt, AccessRelation);

  isl_set *Deltas = isl_map_deltas(NextScatt);
  return Deltas;
}

bool MemoryAccess::isStrideX(__isl_take const isl_map *Schedule,
                             int StrideWidth) const {
  isl_set *Stride, *StrideX;
  bool IsStrideX;

  Stride = getStride(Schedule);
  StrideX = isl_set_universe(isl_set_get_space(Stride));
  for (unsigned i = 0; i < isl_set_dim(StrideX, isl_dim_set) - 1; i++)
    StrideX = isl_set_fix_si(StrideX, isl_dim_set, i, 0);
  StrideX = isl_set_fix_si(StrideX, isl_dim_set,
                           isl_set_dim(StrideX, isl_dim_set) - 1, StrideWidth);
  IsStrideX = isl_set_is_subset(Stride, StrideX);

  isl_set_free(StrideX);
  isl_set_free(Stride);

  return IsStrideX;
}

bool MemoryAccess::isStrideZero(const isl_map *Schedule) const {
  return isStrideX(Schedule, 0);
}

bool MemoryAccess::isStrideOne(const isl_map *Schedule) const {
  return isStrideX(Schedule, 1);
}

void MemoryAccess::setNewAccessRelation(isl_map *NewAccess) {
  isl_map_free(NewAccessRelation);
  NewAccessRelation = NewAccess;
}

//===----------------------------------------------------------------------===//

isl_map *ScopStmt::getSchedule() const {
  isl_set *Domain = getDomain();
  if (isl_set_is_empty(Domain)) {
    isl_set_free(Domain);
    return isl_map_from_aff(
        isl_aff_zero_on_domain(isl_local_space_from_space(getDomainSpace())));
  }
  auto *Schedule = getParent()->getSchedule();
  Schedule = isl_union_map_intersect_domain(
      Schedule, isl_union_set_from_set(isl_set_copy(Domain)));
  if (isl_union_map_is_empty(Schedule)) {
    isl_set_free(Domain);
    isl_union_map_free(Schedule);
    return isl_map_from_aff(
        isl_aff_zero_on_domain(isl_local_space_from_space(getDomainSpace())));
  }
  auto *M = isl_map_from_union_map(Schedule);
  M = isl_map_coalesce(M);
  M = isl_map_gist_domain(M, Domain);
  M = isl_map_coalesce(M);
  return M;
}

__isl_give isl_pw_aff *ScopStmt::getPwAff(const SCEV *E) {
  return getParent()->getPwAff(E, isBlockStmt() ? getBasicBlock()
                                                : getRegion()->getEntry());
}

void ScopStmt::restrictDomain(__isl_take isl_set *NewDomain) {
  assert(isl_set_is_subset(NewDomain, Domain) &&
         "New domain is not a subset of old domain!");
  isl_set_free(Domain);
  Domain = NewDomain;
}

void ScopStmt::buildAccessRelations() {
  for (MemoryAccess *Access : MemAccs) {
    Type *ElementType = Access->getAccessValue()->getType();

    ScopArrayInfo::MemoryKind Ty;
    if (Access->isPHIKind())
      Ty = ScopArrayInfo::MK_PHI;
    else if (Access->isExitPHIKind())
      Ty = ScopArrayInfo::MK_ExitPHI;
    else if (Access->isValueKind())
      Ty = ScopArrayInfo::MK_Value;
    else
      Ty = ScopArrayInfo::MK_Array;

    const ScopArrayInfo *SAI = getParent()->getOrCreateScopArrayInfo(
        Access->getBaseAddr(), ElementType, Access->Sizes, Ty);

    Access->buildAccessRelation(SAI);
  }
}

void ScopStmt::addAccess(MemoryAccess *Access) {
  Instruction *AccessInst = Access->getAccessInstruction();

  if (Access->isArrayKind()) {
    MemoryAccessList &MAL = InstructionToAccess[AccessInst];
    MAL.emplace_front(Access);
  } else if (Access->isValueKind() && Access->isWrite()) {
    Instruction *AccessVal = cast<Instruction>(Access->getAccessValue());
    assert(Parent.getStmtForBasicBlock(AccessVal->getParent()) == this);
    assert(!ValueWrites.lookup(AccessVal));

    ValueWrites[AccessVal] = Access;
  } else if (Access->isValueKind() && Access->isRead()) {
    Value *AccessVal = Access->getAccessValue();
    assert(!ValueReads.lookup(AccessVal));

    ValueReads[AccessVal] = Access;
  } else if (Access->isAnyPHIKind() && Access->isWrite()) {
    PHINode *PHI = cast<PHINode>(Access->getBaseAddr());
    assert(!PHIWrites.lookup(PHI));

    PHIWrites[PHI] = Access;
  }

  MemAccs.push_back(Access);
}

void ScopStmt::realignParams() {
  for (MemoryAccess *MA : *this)
    MA->realignParams();

  Domain = isl_set_align_params(Domain, Parent.getParamSpace());
}

/// @brief Add @p BSet to the set @p User if @p BSet is bounded.
static isl_stat collectBoundedParts(__isl_take isl_basic_set *BSet,
                                    void *User) {
  isl_set **BoundedParts = static_cast<isl_set **>(User);
  if (isl_basic_set_is_bounded(BSet))
    *BoundedParts = isl_set_union(*BoundedParts, isl_set_from_basic_set(BSet));
  else
    isl_basic_set_free(BSet);
  return isl_stat_ok;
}

/// @brief Return the bounded parts of @p S.
static __isl_give isl_set *collectBoundedParts(__isl_take isl_set *S) {
  isl_set *BoundedParts = isl_set_empty(isl_set_get_space(S));
  isl_set_foreach_basic_set(S, collectBoundedParts, &BoundedParts);
  isl_set_free(S);
  return BoundedParts;
}

/// @brief Compute the (un)bounded parts of @p S wrt. to dimension @p Dim.
///
/// @returns A separation of @p S into first an unbounded then a bounded subset,
///          both with regards to the dimension @p Dim.
static std::pair<__isl_give isl_set *, __isl_give isl_set *>
partitionSetParts(__isl_take isl_set *S, unsigned Dim) {

  for (unsigned u = 0, e = isl_set_n_dim(S); u < e; u++)
    S = isl_set_lower_bound_si(S, isl_dim_set, u, 0);

  unsigned NumDimsS = isl_set_n_dim(S);
  isl_set *OnlyDimS = isl_set_copy(S);

  // Remove dimensions that are greater than Dim as they are not interesting.
  assert(NumDimsS >= Dim + 1);
  OnlyDimS =
      isl_set_project_out(OnlyDimS, isl_dim_set, Dim + 1, NumDimsS - Dim - 1);

  // Create artificial parametric upper bounds for dimensions smaller than Dim
  // as we are not interested in them.
  OnlyDimS = isl_set_insert_dims(OnlyDimS, isl_dim_param, 0, Dim);
  for (unsigned u = 0; u < Dim; u++) {
    isl_constraint *C = isl_inequality_alloc(
        isl_local_space_from_space(isl_set_get_space(OnlyDimS)));
    C = isl_constraint_set_coefficient_si(C, isl_dim_param, u, 1);
    C = isl_constraint_set_coefficient_si(C, isl_dim_set, u, -1);
    OnlyDimS = isl_set_add_constraint(OnlyDimS, C);
  }

  // Collect all bounded parts of OnlyDimS.
  isl_set *BoundedParts = collectBoundedParts(OnlyDimS);

  // Create the dimensions greater than Dim again.
  BoundedParts = isl_set_insert_dims(BoundedParts, isl_dim_set, Dim + 1,
                                     NumDimsS - Dim - 1);

  // Remove the artificial upper bound parameters again.
  BoundedParts = isl_set_remove_dims(BoundedParts, isl_dim_param, 0, Dim);

  isl_set *UnboundedParts = isl_set_subtract(S, isl_set_copy(BoundedParts));
  return std::make_pair(UnboundedParts, BoundedParts);
}

/// @brief Set the dimension Ids from @p From in @p To.
static __isl_give isl_set *setDimensionIds(__isl_keep isl_set *From,
                                           __isl_take isl_set *To) {
  for (unsigned u = 0, e = isl_set_n_dim(From); u < e; u++) {
    isl_id *DimId = isl_set_get_dim_id(From, isl_dim_set, u);
    To = isl_set_set_dim_id(To, isl_dim_set, u, DimId);
  }
  return To;
}

/// @brief Create the conditions under which @p L @p Pred @p R is true.
static __isl_give isl_set *buildConditionSet(ICmpInst::Predicate Pred,
                                             __isl_take isl_pw_aff *L,
                                             __isl_take isl_pw_aff *R) {
  switch (Pred) {
  case ICmpInst::ICMP_EQ:
    return isl_pw_aff_eq_set(L, R);
  case ICmpInst::ICMP_NE:
    return isl_pw_aff_ne_set(L, R);
  case ICmpInst::ICMP_SLT:
    return isl_pw_aff_lt_set(L, R);
  case ICmpInst::ICMP_SLE:
    return isl_pw_aff_le_set(L, R);
  case ICmpInst::ICMP_SGT:
    return isl_pw_aff_gt_set(L, R);
  case ICmpInst::ICMP_SGE:
    return isl_pw_aff_ge_set(L, R);
  case ICmpInst::ICMP_ULT:
    return isl_pw_aff_lt_set(L, R);
  case ICmpInst::ICMP_UGT:
    return isl_pw_aff_gt_set(L, R);
  case ICmpInst::ICMP_ULE:
    return isl_pw_aff_le_set(L, R);
  case ICmpInst::ICMP_UGE:
    return isl_pw_aff_ge_set(L, R);
  default:
    llvm_unreachable("Non integer predicate not supported");
  }
}

/// @brief Create the conditions under which @p L @p Pred @p R is true.
///
/// Helper function that will make sure the dimensions of the result have the
/// same isl_id's as the @p Domain.
static __isl_give isl_set *buildConditionSet(ICmpInst::Predicate Pred,
                                             __isl_take isl_pw_aff *L,
                                             __isl_take isl_pw_aff *R,
                                             __isl_keep isl_set *Domain) {
  isl_set *ConsequenceCondSet = buildConditionSet(Pred, L, R);
  return setDimensionIds(Domain, ConsequenceCondSet);
}

/// @brief Build the conditions sets for the switch @p SI in the @p Domain.
///
/// This will fill @p ConditionSets with the conditions under which control
/// will be moved from @p SI to its successors. Hence, @p ConditionSets will
/// have as many elements as @p SI has successors.
static void
buildConditionSets(Scop &S, SwitchInst *SI, Loop *L, __isl_keep isl_set *Domain,
                   SmallVectorImpl<__isl_give isl_set *> &ConditionSets) {

  Value *Condition = getConditionFromTerminator(SI);
  assert(Condition && "No condition for switch");

  ScalarEvolution &SE = *S.getSE();
  BasicBlock *BB = SI->getParent();
  isl_pw_aff *LHS, *RHS;
  LHS = S.getPwAff(SE.getSCEVAtScope(Condition, L), BB);

  unsigned NumSuccessors = SI->getNumSuccessors();
  ConditionSets.resize(NumSuccessors);
  for (auto &Case : SI->cases()) {
    unsigned Idx = Case.getSuccessorIndex();
    ConstantInt *CaseValue = Case.getCaseValue();

    RHS = S.getPwAff(SE.getSCEV(CaseValue), BB);
    isl_set *CaseConditionSet =
        buildConditionSet(ICmpInst::ICMP_EQ, isl_pw_aff_copy(LHS), RHS, Domain);
    ConditionSets[Idx] = isl_set_coalesce(
        isl_set_intersect(CaseConditionSet, isl_set_copy(Domain)));
  }

  assert(ConditionSets[0] == nullptr && "Default condition set was set");
  isl_set *ConditionSetUnion = isl_set_copy(ConditionSets[1]);
  for (unsigned u = 2; u < NumSuccessors; u++)
    ConditionSetUnion =
        isl_set_union(ConditionSetUnion, isl_set_copy(ConditionSets[u]));
  ConditionSets[0] = setDimensionIds(
      Domain, isl_set_subtract(isl_set_copy(Domain), ConditionSetUnion));

  S.markAsOptimized();
  isl_pw_aff_free(LHS);
}

/// @brief Build the conditions sets for the branch condition @p Condition in
///        the @p Domain.
///
/// This will fill @p ConditionSets with the conditions under which control
/// will be moved from @p TI to its successors. Hence, @p ConditionSets will
/// have as many elements as @p TI has successors. If @p TI is nullptr the
/// context under which @p Condition is true/false will be returned as the
/// new elements of @p ConditionSets.
static void
buildConditionSets(Scop &S, Value *Condition, TerminatorInst *TI, Loop *L,
                   __isl_keep isl_set *Domain,
                   SmallVectorImpl<__isl_give isl_set *> &ConditionSets) {

  isl_set *ConsequenceCondSet = nullptr;
  if (auto *CCond = dyn_cast<ConstantInt>(Condition)) {
    if (CCond->isZero())
      ConsequenceCondSet = isl_set_empty(isl_set_get_space(Domain));
    else
      ConsequenceCondSet = isl_set_universe(isl_set_get_space(Domain));
  } else if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(Condition)) {
    auto Opcode = BinOp->getOpcode();
    assert(Opcode == Instruction::And || Opcode == Instruction::Or);

    buildConditionSets(S, BinOp->getOperand(0), TI, L, Domain, ConditionSets);
    buildConditionSets(S, BinOp->getOperand(1), TI, L, Domain, ConditionSets);

    isl_set_free(ConditionSets.pop_back_val());
    isl_set *ConsCondPart0 = ConditionSets.pop_back_val();
    isl_set_free(ConditionSets.pop_back_val());
    isl_set *ConsCondPart1 = ConditionSets.pop_back_val();

    if (Opcode == Instruction::And)
      ConsequenceCondSet = isl_set_intersect(ConsCondPart0, ConsCondPart1);
    else
      ConsequenceCondSet = isl_set_union(ConsCondPart0, ConsCondPart1);
  } else {
    auto *ICond = dyn_cast<ICmpInst>(Condition);
    assert(ICond &&
           "Condition of exiting branch was neither constant nor ICmp!");

    ScalarEvolution &SE = *S.getSE();
    BasicBlock *BB = TI ? TI->getParent() : nullptr;
    isl_pw_aff *LHS, *RHS;
    LHS = S.getPwAff(SE.getSCEVAtScope(ICond->getOperand(0), L), BB);
    RHS = S.getPwAff(SE.getSCEVAtScope(ICond->getOperand(1), L), BB);
    ConsequenceCondSet =
        buildConditionSet(ICond->getPredicate(), LHS, RHS, Domain);
  }

  // If no terminator was given we are only looking for parameter constraints
  // under which @p Condition is true/false.
  if (!TI)
    ConsequenceCondSet = isl_set_params(ConsequenceCondSet);

  assert(ConsequenceCondSet);
  isl_set *AlternativeCondSet =
      isl_set_complement(isl_set_copy(ConsequenceCondSet));

  ConditionSets.push_back(isl_set_coalesce(
      isl_set_intersect(ConsequenceCondSet, isl_set_copy(Domain))));
  ConditionSets.push_back(isl_set_coalesce(
      isl_set_intersect(AlternativeCondSet, isl_set_copy(Domain))));
}

/// @brief Build the conditions sets for the terminator @p TI in the @p Domain.
///
/// This will fill @p ConditionSets with the conditions under which control
/// will be moved from @p TI to its successors. Hence, @p ConditionSets will
/// have as many elements as @p TI has successors.
static void
buildConditionSets(Scop &S, TerminatorInst *TI, Loop *L,
                   __isl_keep isl_set *Domain,
                   SmallVectorImpl<__isl_give isl_set *> &ConditionSets) {

  if (SwitchInst *SI = dyn_cast<SwitchInst>(TI))
    return buildConditionSets(S, SI, L, Domain, ConditionSets);

  assert(isa<BranchInst>(TI) && "Terminator was neither branch nor switch.");

  if (TI->getNumSuccessors() == 1) {
    ConditionSets.push_back(isl_set_copy(Domain));
    return;
  }

  Value *Condition = getConditionFromTerminator(TI);
  assert(Condition && "No condition for Terminator");

  return buildConditionSets(S, Condition, TI, L, Domain, ConditionSets);
}

void ScopStmt::buildDomain() {
  isl_id *Id;

  Id = isl_id_alloc(getIslCtx(), getBaseName(), this);

  Domain = getParent()->getDomainConditions(this);
  Domain = isl_set_set_tuple_id(Domain, Id);
}

void ScopStmt::deriveAssumptionsFromGEP(GetElementPtrInst *GEP) {
  isl_ctx *Ctx = Parent.getIslCtx();
  isl_local_space *LSpace = isl_local_space_from_space(getDomainSpace());
  Type *Ty = GEP->getPointerOperandType();
  ScalarEvolution &SE = *Parent.getSE();
  ScopDetection &SD = Parent.getSD();

  // The set of loads that are required to be invariant.
  auto &ScopRIL = *SD.getRequiredInvariantLoads(&Parent.getRegion());

  std::vector<const SCEV *> Subscripts;
  std::vector<int> Sizes;

  std::tie(Subscripts, Sizes) = getIndexExpressionsFromGEP(GEP, SE);

  if (auto *PtrTy = dyn_cast<PointerType>(Ty)) {
    Ty = PtrTy->getElementType();
  }

  int IndexOffset = Subscripts.size() - Sizes.size();

  assert(IndexOffset <= 1 && "Unexpected large index offset");

  for (size_t i = 0; i < Sizes.size(); i++) {
    auto Expr = Subscripts[i + IndexOffset];
    auto Size = Sizes[i];

    InvariantLoadsSetTy AccessILS;
    if (!isAffineExpr(&Parent.getRegion(), Expr, SE, nullptr, &AccessILS))
      continue;

    bool NonAffine = false;
    for (LoadInst *LInst : AccessILS)
      if (!ScopRIL.count(LInst))
        NonAffine = true;

    if (NonAffine)
      continue;

    isl_pw_aff *AccessOffset = getPwAff(Expr);
    AccessOffset =
        isl_pw_aff_set_tuple_id(AccessOffset, isl_dim_in, getDomainId());

    isl_pw_aff *DimSize = isl_pw_aff_from_aff(isl_aff_val_on_domain(
        isl_local_space_copy(LSpace), isl_val_int_from_si(Ctx, Size)));

    isl_set *OutOfBound = isl_pw_aff_ge_set(AccessOffset, DimSize);
    OutOfBound = isl_set_intersect(getDomain(), OutOfBound);
    OutOfBound = isl_set_params(OutOfBound);
    isl_set *InBound = isl_set_complement(OutOfBound);
    isl_set *Executed = isl_set_params(getDomain());

    // A => B == !A or B
    isl_set *InBoundIfExecuted =
        isl_set_union(isl_set_complement(Executed), InBound);

    InBoundIfExecuted = isl_set_coalesce(InBoundIfExecuted);
    Parent.addAssumption(INBOUNDS, InBoundIfExecuted, GEP->getDebugLoc());
  }

  isl_local_space_free(LSpace);
}

void ScopStmt::deriveAssumptions(BasicBlock *Block) {
  for (Instruction &Inst : *Block)
    if (auto *GEP = dyn_cast<GetElementPtrInst>(&Inst))
      deriveAssumptionsFromGEP(GEP);
}

void ScopStmt::collectSurroundingLoops() {
  for (unsigned u = 0, e = isl_set_n_dim(Domain); u < e; u++) {
    isl_id *DimId = isl_set_get_dim_id(Domain, isl_dim_set, u);
    NestLoops.push_back(static_cast<Loop *>(isl_id_get_user(DimId)));
    isl_id_free(DimId);
  }
}

ScopStmt::ScopStmt(Scop &parent, Region &R)
    : Parent(parent), Domain(nullptr), BB(nullptr), R(&R), Build(nullptr) {

  BaseName = getIslCompatibleName("Stmt_", R.getNameStr(), "");
}

ScopStmt::ScopStmt(Scop &parent, BasicBlock &bb)
    : Parent(parent), Domain(nullptr), BB(&bb), R(nullptr), Build(nullptr) {

  BaseName = getIslCompatibleName("Stmt_", &bb, "");
}

void ScopStmt::init() {
  assert(!Domain && "init must be called only once");

  buildDomain();
  collectSurroundingLoops();
  buildAccessRelations();

  if (BB) {
    deriveAssumptions(BB);
  } else {
    for (BasicBlock *Block : R->blocks()) {
      deriveAssumptions(Block);
    }
  }

  if (DetectReductions)
    checkForReductions();
}

/// @brief Collect loads which might form a reduction chain with @p StoreMA
///
/// Check if the stored value for @p StoreMA is a binary operator with one or
/// two loads as operands. If the binary operand is commutative & associative,
/// used only once (by @p StoreMA) and its load operands are also used only
/// once, we have found a possible reduction chain. It starts at an operand
/// load and includes the binary operator and @p StoreMA.
///
/// Note: We allow only one use to ensure the load and binary operator cannot
///       escape this block or into any other store except @p StoreMA.
void ScopStmt::collectCandiateReductionLoads(
    MemoryAccess *StoreMA, SmallVectorImpl<MemoryAccess *> &Loads) {
  auto *Store = dyn_cast<StoreInst>(StoreMA->getAccessInstruction());
  if (!Store)
    return;

  // Skip if there is not one binary operator between the load and the store
  auto *BinOp = dyn_cast<BinaryOperator>(Store->getValueOperand());
  if (!BinOp)
    return;

  // Skip if the binary operators has multiple uses
  if (BinOp->getNumUses() != 1)
    return;

  // Skip if the opcode of the binary operator is not commutative/associative
  if (!BinOp->isCommutative() || !BinOp->isAssociative())
    return;

  // Skip if the binary operator is outside the current SCoP
  if (BinOp->getParent() != Store->getParent())
    return;

  // Skip if it is a multiplicative reduction and we disabled them
  if (DisableMultiplicativeReductions &&
      (BinOp->getOpcode() == Instruction::Mul ||
       BinOp->getOpcode() == Instruction::FMul))
    return;

  // Check the binary operator operands for a candidate load
  auto *PossibleLoad0 = dyn_cast<LoadInst>(BinOp->getOperand(0));
  auto *PossibleLoad1 = dyn_cast<LoadInst>(BinOp->getOperand(1));
  if (!PossibleLoad0 && !PossibleLoad1)
    return;

  // A load is only a candidate if it cannot escape (thus has only this use)
  if (PossibleLoad0 && PossibleLoad0->getNumUses() == 1)
    if (PossibleLoad0->getParent() == Store->getParent())
      Loads.push_back(&getArrayAccessFor(PossibleLoad0));
  if (PossibleLoad1 && PossibleLoad1->getNumUses() == 1)
    if (PossibleLoad1->getParent() == Store->getParent())
      Loads.push_back(&getArrayAccessFor(PossibleLoad1));
}

/// @brief Check for reductions in this ScopStmt
///
/// Iterate over all store memory accesses and check for valid binary reduction
/// like chains. For all candidates we check if they have the same base address
/// and there are no other accesses which overlap with them. The base address
/// check rules out impossible reductions candidates early. The overlap check,
/// together with the "only one user" check in collectCandiateReductionLoads,
/// guarantees that none of the intermediate results will escape during
/// execution of the loop nest. We basically check here that no other memory
/// access can access the same memory as the potential reduction.
void ScopStmt::checkForReductions() {
  SmallVector<MemoryAccess *, 2> Loads;
  SmallVector<std::pair<MemoryAccess *, MemoryAccess *>, 4> Candidates;

  // First collect candidate load-store reduction chains by iterating over all
  // stores and collecting possible reduction loads.
  for (MemoryAccess *StoreMA : MemAccs) {
    if (StoreMA->isRead())
      continue;

    Loads.clear();
    collectCandiateReductionLoads(StoreMA, Loads);
    for (MemoryAccess *LoadMA : Loads)
      Candidates.push_back(std::make_pair(LoadMA, StoreMA));
  }

  // Then check each possible candidate pair.
  for (const auto &CandidatePair : Candidates) {
    bool Valid = true;
    isl_map *LoadAccs = CandidatePair.first->getAccessRelation();
    isl_map *StoreAccs = CandidatePair.second->getAccessRelation();

    // Skip those with obviously unequal base addresses.
    if (!isl_map_has_equal_space(LoadAccs, StoreAccs)) {
      isl_map_free(LoadAccs);
      isl_map_free(StoreAccs);
      continue;
    }

    // And check if the remaining for overlap with other memory accesses.
    isl_map *AllAccsRel = isl_map_union(LoadAccs, StoreAccs);
    AllAccsRel = isl_map_intersect_domain(AllAccsRel, getDomain());
    isl_set *AllAccs = isl_map_range(AllAccsRel);

    for (MemoryAccess *MA : MemAccs) {
      if (MA == CandidatePair.first || MA == CandidatePair.second)
        continue;

      isl_map *AccRel =
          isl_map_intersect_domain(MA->getAccessRelation(), getDomain());
      isl_set *Accs = isl_map_range(AccRel);

      if (isl_set_has_equal_space(AllAccs, Accs) || isl_set_free(Accs)) {
        isl_set *OverlapAccs = isl_set_intersect(Accs, isl_set_copy(AllAccs));
        Valid = Valid && isl_set_is_empty(OverlapAccs);
        isl_set_free(OverlapAccs);
      }
    }

    isl_set_free(AllAccs);
    if (!Valid)
      continue;

    const LoadInst *Load =
        dyn_cast<const LoadInst>(CandidatePair.first->getAccessInstruction());
    MemoryAccess::ReductionType RT =
        getReductionType(dyn_cast<BinaryOperator>(Load->user_back()), Load);

    // If no overlapping access was found we mark the load and store as
    // reduction like.
    CandidatePair.first->markAsReductionLike(RT);
    CandidatePair.second->markAsReductionLike(RT);
  }
}

std::string ScopStmt::getDomainStr() const { return stringFromIslObj(Domain); }

std::string ScopStmt::getScheduleStr() const {
  auto *S = getSchedule();
  auto Str = stringFromIslObj(S);
  isl_map_free(S);
  return Str;
}

unsigned ScopStmt::getNumParams() const { return Parent.getNumParams(); }

unsigned ScopStmt::getNumIterators() const { return NestLoops.size(); }

const char *ScopStmt::getBaseName() const { return BaseName.c_str(); }

const Loop *ScopStmt::getLoopForDimension(unsigned Dimension) const {
  return NestLoops[Dimension];
}

isl_ctx *ScopStmt::getIslCtx() const { return Parent.getIslCtx(); }

__isl_give isl_set *ScopStmt::getDomain() const { return isl_set_copy(Domain); }

__isl_give isl_space *ScopStmt::getDomainSpace() const {
  return isl_set_get_space(Domain);
}

__isl_give isl_id *ScopStmt::getDomainId() const {
  return isl_set_get_tuple_id(Domain);
}

ScopStmt::~ScopStmt() { isl_set_free(Domain); }

void ScopStmt::print(raw_ostream &OS) const {
  OS << "\t" << getBaseName() << "\n";
  OS.indent(12) << "Domain :=\n";

  if (Domain) {
    OS.indent(16) << getDomainStr() << ";\n";
  } else
    OS.indent(16) << "n/a\n";

  OS.indent(12) << "Schedule :=\n";

  if (Domain) {
    OS.indent(16) << getScheduleStr() << ";\n";
  } else
    OS.indent(16) << "n/a\n";

  for (MemoryAccess *Access : MemAccs)
    Access->print(OS);
}

void ScopStmt::dump() const { print(dbgs()); }

void ScopStmt::removeMemoryAccesses(MemoryAccessList &InvMAs) {
  // Remove all memory accesses in @p InvMAs from this statement
  // together with all scalar accesses that were caused by them.
  // MK_Value READs have no access instruction, hence would not be removed by
  // this function. However, it is only used for invariant LoadInst accesses,
  // its arguments are always affine, hence synthesizable, and therefore there
  // are no MK_Value READ accesses to be removed.
  for (MemoryAccess *MA : InvMAs) {
    auto Predicate = [&](MemoryAccess *Acc) {
      return Acc->getAccessInstruction() == MA->getAccessInstruction();
    };
    MemAccs.erase(std::remove_if(MemAccs.begin(), MemAccs.end(), Predicate),
                  MemAccs.end());
    InstructionToAccess.erase(MA->getAccessInstruction());
  }
}

//===----------------------------------------------------------------------===//
/// Scop class implement

void Scop::setContext(__isl_take isl_set *NewContext) {
  NewContext = isl_set_align_params(NewContext, isl_set_get_space(Context));
  isl_set_free(Context);
  Context = NewContext;
}

/// @brief Remap parameter values but keep AddRecs valid wrt. invariant loads.
struct SCEVSensitiveParameterRewriter
    : public SCEVVisitor<SCEVSensitiveParameterRewriter, const SCEV *> {
  ValueToValueMap &VMap;
  ScalarEvolution &SE;

public:
  SCEVSensitiveParameterRewriter(ValueToValueMap &VMap, ScalarEvolution &SE)
      : VMap(VMap), SE(SE) {}

  static const SCEV *rewrite(const SCEV *E, ScalarEvolution &SE,
                             ValueToValueMap &VMap) {
    SCEVSensitiveParameterRewriter SSPR(VMap, SE);
    return SSPR.visit(E);
  }

  const SCEV *visit(const SCEV *E) {
    return SCEVVisitor<SCEVSensitiveParameterRewriter, const SCEV *>::visit(E);
  }

  const SCEV *visitConstant(const SCEVConstant *E) { return E; }

  const SCEV *visitTruncateExpr(const SCEVTruncateExpr *E) {
    return SE.getTruncateExpr(visit(E->getOperand()), E->getType());
  }

  const SCEV *visitZeroExtendExpr(const SCEVZeroExtendExpr *E) {
    return SE.getZeroExtendExpr(visit(E->getOperand()), E->getType());
  }

  const SCEV *visitSignExtendExpr(const SCEVSignExtendExpr *E) {
    return SE.getSignExtendExpr(visit(E->getOperand()), E->getType());
  }

  const SCEV *visitAddExpr(const SCEVAddExpr *E) {
    SmallVector<const SCEV *, 4> Operands;
    for (int i = 0, e = E->getNumOperands(); i < e; ++i)
      Operands.push_back(visit(E->getOperand(i)));
    return SE.getAddExpr(Operands);
  }

  const SCEV *visitMulExpr(const SCEVMulExpr *E) {
    SmallVector<const SCEV *, 4> Operands;
    for (int i = 0, e = E->getNumOperands(); i < e; ++i)
      Operands.push_back(visit(E->getOperand(i)));
    return SE.getMulExpr(Operands);
  }

  const SCEV *visitSMaxExpr(const SCEVSMaxExpr *E) {
    SmallVector<const SCEV *, 4> Operands;
    for (int i = 0, e = E->getNumOperands(); i < e; ++i)
      Operands.push_back(visit(E->getOperand(i)));
    return SE.getSMaxExpr(Operands);
  }

  const SCEV *visitUMaxExpr(const SCEVUMaxExpr *E) {
    SmallVector<const SCEV *, 4> Operands;
    for (int i = 0, e = E->getNumOperands(); i < e; ++i)
      Operands.push_back(visit(E->getOperand(i)));
    return SE.getUMaxExpr(Operands);
  }

  const SCEV *visitUDivExpr(const SCEVUDivExpr *E) {
    return SE.getUDivExpr(visit(E->getLHS()), visit(E->getRHS()));
  }

  const SCEV *visitAddRecExpr(const SCEVAddRecExpr *E) {
    auto *Start = visit(E->getStart());
    auto *AddRec = SE.getAddRecExpr(SE.getConstant(E->getType(), 0),
                                    visit(E->getStepRecurrence(SE)),
                                    E->getLoop(), SCEV::FlagAnyWrap);
    return SE.getAddExpr(Start, AddRec);
  }

  const SCEV *visitUnknown(const SCEVUnknown *E) {
    if (auto *NewValue = VMap.lookup(E->getValue()))
      return SE.getUnknown(NewValue);
    return E;
  }
};

const SCEV *Scop::getRepresentingInvariantLoadSCEV(const SCEV *S) {
  return SCEVSensitiveParameterRewriter::rewrite(S, *SE, InvEquivClassVMap);
}

void Scop::addParams(std::vector<const SCEV *> NewParameters) {
  for (const SCEV *Parameter : NewParameters) {
    Parameter = extractConstantFactor(Parameter, *SE).second;

    // Normalize the SCEV to get the representing element for an invariant load.
    Parameter = getRepresentingInvariantLoadSCEV(Parameter);

    if (ParameterIds.find(Parameter) != ParameterIds.end())
      continue;

    int dimension = Parameters.size();

    Parameters.push_back(Parameter);
    ParameterIds[Parameter] = dimension;
  }
}

__isl_give isl_id *Scop::getIdForParam(const SCEV *Parameter) {
  // Normalize the SCEV to get the representing element for an invariant load.
  Parameter = getRepresentingInvariantLoadSCEV(Parameter);

  ParamIdType::const_iterator IdIter = ParameterIds.find(Parameter);

  if (IdIter == ParameterIds.end())
    return nullptr;

  std::string ParameterName;

  ParameterName = "p_" + utostr(IdIter->second);

  if (const SCEVUnknown *ValueParameter = dyn_cast<SCEVUnknown>(Parameter)) {
    Value *Val = ValueParameter->getValue();

    // If this parameter references a specific Value and this value has a name
    // we use this name as it is likely to be unique and more useful than just
    // a number.
    if (Val->hasName())
      ParameterName = Val->getName();
    else if (LoadInst *LI = dyn_cast<LoadInst>(Val)) {
      auto LoadOrigin = LI->getPointerOperand()->stripInBoundsOffsets();
      if (LoadOrigin->hasName()) {
        ParameterName += "_loaded_from_";
        ParameterName +=
            LI->getPointerOperand()->stripInBoundsOffsets()->getName();
      }
    }
  }

  return isl_id_alloc(getIslCtx(), ParameterName.c_str(),
                      const_cast<void *>((const void *)Parameter));
}

isl_set *Scop::addNonEmptyDomainConstraints(isl_set *C) const {
  isl_set *DomainContext = isl_union_set_params(getDomains());
  return isl_set_intersect_params(C, DomainContext);
}

void Scop::buildBoundaryContext() {
  if (IgnoreIntegerWrapping) {
    BoundaryContext = isl_set_universe(getParamSpace());
    return;
  }

  BoundaryContext = Affinator.getWrappingContext();

  // The isl_set_complement operation used to create the boundary context
  // can possibly become very expensive. We bound the compile time of
  // this operation by setting a compute out.
  //
  // TODO: We can probably get around using isl_set_complement and directly
  // AST generate BoundaryContext.
  long MaxOpsOld = isl_ctx_get_max_operations(getIslCtx());
  isl_ctx_reset_operations(getIslCtx());
  isl_ctx_set_max_operations(getIslCtx(), 300000);
  isl_options_set_on_error(getIslCtx(), ISL_ON_ERROR_CONTINUE);

  BoundaryContext = isl_set_complement(BoundaryContext);

  if (isl_ctx_last_error(getIslCtx()) == isl_error_quota) {
    isl_set_free(BoundaryContext);
    BoundaryContext = isl_set_empty(getParamSpace());
  }

  isl_options_set_on_error(getIslCtx(), ISL_ON_ERROR_ABORT);
  isl_ctx_reset_operations(getIslCtx());
  isl_ctx_set_max_operations(getIslCtx(), MaxOpsOld);
  BoundaryContext = isl_set_gist_params(BoundaryContext, getContext());
  trackAssumption(WRAPPING, BoundaryContext, DebugLoc());
}

void Scop::addUserAssumptions(AssumptionCache &AC) {
  auto *R = &getRegion();
  auto &F = *R->getEntry()->getParent();
  for (auto &Assumption : AC.assumptions()) {
    auto *CI = dyn_cast_or_null<CallInst>(Assumption);
    if (!CI || CI->getNumArgOperands() != 1)
      continue;
    if (!DT.dominates(CI->getParent(), R->getEntry()))
      continue;

    auto *Val = CI->getArgOperand(0);
    std::vector<const SCEV *> Params;
    if (!isAffineParamConstraint(Val, R, *SE, Params)) {
      emitOptimizationRemarkAnalysis(F.getContext(), DEBUG_TYPE, F,
                                     CI->getDebugLoc(),
                                     "Non-affine user assumption ignored.");
      continue;
    }

    addParams(Params);

    auto *L = LI.getLoopFor(CI->getParent());
    SmallVector<isl_set *, 2> ConditionSets;
    buildConditionSets(*this, Val, nullptr, L, Context, ConditionSets);
    assert(ConditionSets.size() == 2);
    isl_set_free(ConditionSets[1]);

    auto *AssumptionCtx = ConditionSets[0];
    emitOptimizationRemarkAnalysis(
        F.getContext(), DEBUG_TYPE, F, CI->getDebugLoc(),
        "Use user assumption: " + stringFromIslObj(AssumptionCtx));
    Context = isl_set_intersect(Context, AssumptionCtx);
  }
}

void Scop::addUserContext() {
  if (UserContextStr.empty())
    return;

  isl_set *UserContext = isl_set_read_from_str(IslCtx, UserContextStr.c_str());
  isl_space *Space = getParamSpace();
  if (isl_space_dim(Space, isl_dim_param) !=
      isl_set_dim(UserContext, isl_dim_param)) {
    auto SpaceStr = isl_space_to_str(Space);
    errs() << "Error: the context provided in -polly-context has not the same "
           << "number of dimensions than the computed context. Due to this "
           << "mismatch, the -polly-context option is ignored. Please provide "
           << "the context in the parameter space: " << SpaceStr << ".\n";
    free(SpaceStr);
    isl_set_free(UserContext);
    isl_space_free(Space);
    return;
  }

  for (unsigned i = 0; i < isl_space_dim(Space, isl_dim_param); i++) {
    auto NameContext = isl_set_get_dim_name(Context, isl_dim_param, i);
    auto NameUserContext = isl_set_get_dim_name(UserContext, isl_dim_param, i);

    if (strcmp(NameContext, NameUserContext) != 0) {
      auto SpaceStr = isl_space_to_str(Space);
      errs() << "Error: the name of dimension " << i
             << " provided in -polly-context "
             << "is '" << NameUserContext << "', but the name in the computed "
             << "context is '" << NameContext
             << "'. Due to this name mismatch, "
             << "the -polly-context option is ignored. Please provide "
             << "the context in the parameter space: " << SpaceStr << ".\n";
      free(SpaceStr);
      isl_set_free(UserContext);
      isl_space_free(Space);
      return;
    }

    UserContext =
        isl_set_set_dim_id(UserContext, isl_dim_param, i,
                           isl_space_get_dim_id(Space, isl_dim_param, i));
  }

  Context = isl_set_intersect(Context, UserContext);
  isl_space_free(Space);
}

void Scop::buildInvariantEquivalenceClasses() {
  DenseMap<const SCEV *, LoadInst *> EquivClasses;

  const InvariantLoadsSetTy &RIL = *SD.getRequiredInvariantLoads(&getRegion());
  for (LoadInst *LInst : RIL) {
    const SCEV *PointerSCEV = SE->getSCEV(LInst->getPointerOperand());

    LoadInst *&ClassRep = EquivClasses[PointerSCEV];
    if (ClassRep) {
      InvEquivClassVMap[LInst] = ClassRep;
      continue;
    }

    ClassRep = LInst;
    InvariantEquivClasses.emplace_back(PointerSCEV, MemoryAccessList(),
                                       nullptr);
  }
}

void Scop::buildContext() {
  isl_space *Space = isl_space_params_alloc(IslCtx, 0);
  Context = isl_set_universe(isl_space_copy(Space));
  AssumedContext = isl_set_universe(Space);
}

void Scop::addParameterBounds() {
  for (const auto &ParamID : ParameterIds) {
    int dim = ParamID.second;

    ConstantRange SRange = SE->getSignedRange(ParamID.first);

    Context = addRangeBoundsToSet(Context, SRange, dim, isl_dim_param);
  }
}

void Scop::realignParams() {
  // Add all parameters into a common model.
  isl_space *Space = isl_space_params_alloc(IslCtx, ParameterIds.size());

  for (const auto &ParamID : ParameterIds) {
    const SCEV *Parameter = ParamID.first;
    isl_id *id = getIdForParam(Parameter);
    Space = isl_space_set_dim_id(Space, isl_dim_param, ParamID.second, id);
  }

  // Align the parameters of all data structures to the model.
  Context = isl_set_align_params(Context, Space);

  for (ScopStmt &Stmt : *this)
    Stmt.realignParams();
}

static __isl_give isl_set *
simplifyAssumptionContext(__isl_take isl_set *AssumptionContext,
                          const Scop &S) {
  // If we modelt all blocks in the SCoP that have side effects we can simplify
  // the context with the constraints that are needed for anything to be
  // executed at all. However, if we have error blocks in the SCoP we already
  // assumed some parameter combinations cannot occure and removed them from the
  // domains, thus we cannot use the remaining domain to simplify the
  // assumptions.
  if (!S.hasErrorBlock()) {
    isl_set *DomainParameters = isl_union_set_params(S.getDomains());
    AssumptionContext =
        isl_set_gist_params(AssumptionContext, DomainParameters);
  }

  AssumptionContext = isl_set_gist_params(AssumptionContext, S.getContext());
  return AssumptionContext;
}

void Scop::simplifyContexts() {
  // The parameter constraints of the iteration domains give us a set of
  // constraints that need to hold for all cases where at least a single
  // statement iteration is executed in the whole scop. We now simplify the
  // assumed context under the assumption that such constraints hold and at
  // least a single statement iteration is executed. For cases where no
  // statement instances are executed, the assumptions we have taken about
  // the executed code do not matter and can be changed.
  //
  // WARNING: This only holds if the assumptions we have taken do not reduce
  //          the set of statement instances that are executed. Otherwise we
  //          may run into a case where the iteration domains suggest that
  //          for a certain set of parameter constraints no code is executed,
  //          but in the original program some computation would have been
  //          performed. In such a case, modifying the run-time conditions and
  //          possibly influencing the run-time check may cause certain scops
  //          to not be executed.
  //
  // Example:
  //
  //   When delinearizing the following code:
  //
  //     for (long i = 0; i < 100; i++)
  //       for (long j = 0; j < m; j++)
  //         A[i+p][j] = 1.0;
  //
  //   we assume that the condition m <= 0 or (m >= 1 and p >= 0) holds as
  //   otherwise we would access out of bound data. Now, knowing that code is
  //   only executed for the case m >= 0, it is sufficient to assume p >= 0.
  AssumedContext = simplifyAssumptionContext(AssumedContext, *this);
  BoundaryContext = simplifyAssumptionContext(BoundaryContext, *this);
}

/// @brief Add the minimal/maximal access in @p Set to @p User.
static isl_stat buildMinMaxAccess(__isl_take isl_set *Set, void *User) {
  Scop::MinMaxVectorTy *MinMaxAccesses = (Scop::MinMaxVectorTy *)User;
  isl_pw_multi_aff *MinPMA, *MaxPMA;
  isl_pw_aff *LastDimAff;
  isl_aff *OneAff;
  unsigned Pos;

  // Restrict the number of parameters involved in the access as the lexmin/
  // lexmax computation will take too long if this number is high.
  //
  // Experiments with a simple test case using an i7 4800MQ:
  //
  //  #Parameters involved | Time (in sec)
  //            6          |     0.01
  //            7          |     0.04
  //            8          |     0.12
  //            9          |     0.40
  //           10          |     1.54
  //           11          |     6.78
  //           12          |    30.38
  //
  if (isl_set_n_param(Set) > RunTimeChecksMaxParameters) {
    unsigned InvolvedParams = 0;
    for (unsigned u = 0, e = isl_set_n_param(Set); u < e; u++)
      if (isl_set_involves_dims(Set, isl_dim_param, u, 1))
        InvolvedParams++;

    if (InvolvedParams > RunTimeChecksMaxParameters) {
      isl_set_free(Set);
      return isl_stat_error;
    }
  }

  Set = isl_set_remove_divs(Set);

  MinPMA = isl_set_lexmin_pw_multi_aff(isl_set_copy(Set));
  MaxPMA = isl_set_lexmax_pw_multi_aff(isl_set_copy(Set));

  MinPMA = isl_pw_multi_aff_coalesce(MinPMA);
  MaxPMA = isl_pw_multi_aff_coalesce(MaxPMA);

  // Adjust the last dimension of the maximal access by one as we want to
  // enclose the accessed memory region by MinPMA and MaxPMA. The pointer
  // we test during code generation might now point after the end of the
  // allocated array but we will never dereference it anyway.
  assert(isl_pw_multi_aff_dim(MaxPMA, isl_dim_out) &&
         "Assumed at least one output dimension");
  Pos = isl_pw_multi_aff_dim(MaxPMA, isl_dim_out) - 1;
  LastDimAff = isl_pw_multi_aff_get_pw_aff(MaxPMA, Pos);
  OneAff = isl_aff_zero_on_domain(
      isl_local_space_from_space(isl_pw_aff_get_domain_space(LastDimAff)));
  OneAff = isl_aff_add_constant_si(OneAff, 1);
  LastDimAff = isl_pw_aff_add(LastDimAff, isl_pw_aff_from_aff(OneAff));
  MaxPMA = isl_pw_multi_aff_set_pw_aff(MaxPMA, Pos, LastDimAff);

  MinMaxAccesses->push_back(std::make_pair(MinPMA, MaxPMA));

  isl_set_free(Set);
  return isl_stat_ok;
}

static __isl_give isl_set *getAccessDomain(MemoryAccess *MA) {
  isl_set *Domain = MA->getStatement()->getDomain();
  Domain = isl_set_project_out(Domain, isl_dim_set, 0, isl_set_n_dim(Domain));
  return isl_set_reset_tuple_id(Domain);
}

/// @brief Wrapper function to calculate minimal/maximal accesses to each array.
static bool calculateMinMaxAccess(__isl_take isl_union_map *Accesses,
                                  __isl_take isl_union_set *Domains,
                                  Scop::MinMaxVectorTy &MinMaxAccesses) {

  Accesses = isl_union_map_intersect_domain(Accesses, Domains);
  isl_union_set *Locations = isl_union_map_range(Accesses);
  Locations = isl_union_set_coalesce(Locations);
  Locations = isl_union_set_detect_equalities(Locations);
  bool Valid = (0 == isl_union_set_foreach_set(Locations, buildMinMaxAccess,
                                               &MinMaxAccesses));
  isl_union_set_free(Locations);
  return Valid;
}

/// @brief Helper to treat non-affine regions and basic blocks the same.
///
///{

/// @brief Return the block that is the representing block for @p RN.
static inline BasicBlock *getRegionNodeBasicBlock(RegionNode *RN) {
  return RN->isSubRegion() ? RN->getNodeAs<Region>()->getEntry()
                           : RN->getNodeAs<BasicBlock>();
}

/// @brief Return the @p idx'th block that is executed after @p RN.
static inline BasicBlock *
getRegionNodeSuccessor(RegionNode *RN, TerminatorInst *TI, unsigned idx) {
  if (RN->isSubRegion()) {
    assert(idx == 0);
    return RN->getNodeAs<Region>()->getExit();
  }
  return TI->getSuccessor(idx);
}

/// @brief Return the smallest loop surrounding @p RN.
static inline Loop *getRegionNodeLoop(RegionNode *RN, LoopInfo &LI) {
  if (!RN->isSubRegion())
    return LI.getLoopFor(RN->getNodeAs<BasicBlock>());

  Region *NonAffineSubRegion = RN->getNodeAs<Region>();
  Loop *L = LI.getLoopFor(NonAffineSubRegion->getEntry());
  while (L && NonAffineSubRegion->contains(L))
    L = L->getParentLoop();
  return L;
}

static inline unsigned getNumBlocksInRegionNode(RegionNode *RN) {
  if (!RN->isSubRegion())
    return 1;

  Region *R = RN->getNodeAs<Region>();
  return std::distance(R->block_begin(), R->block_end());
}

static bool containsErrorBlock(RegionNode *RN, const Region &R, LoopInfo &LI,
                               const DominatorTree &DT) {
  if (!RN->isSubRegion())
    return isErrorBlock(*RN->getNodeAs<BasicBlock>(), R, LI, DT);
  for (BasicBlock *BB : RN->getNodeAs<Region>()->blocks())
    if (isErrorBlock(*BB, R, LI, DT))
      return true;
  return false;
}

///}

static inline __isl_give isl_set *addDomainDimId(__isl_take isl_set *Domain,
                                                 unsigned Dim, Loop *L) {
  Domain = isl_set_lower_bound_si(Domain, isl_dim_set, Dim, -1);
  isl_id *DimId =
      isl_id_alloc(isl_set_get_ctx(Domain), nullptr, static_cast<void *>(L));
  return isl_set_set_dim_id(Domain, isl_dim_set, Dim, DimId);
}

isl_set *Scop::getDomainConditions(ScopStmt *Stmt) {
  BasicBlock *BB = Stmt->isBlockStmt() ? Stmt->getBasicBlock()
                                       : Stmt->getRegion()->getEntry();
  return getDomainConditions(BB);
}

isl_set *Scop::getDomainConditions(BasicBlock *BB) {
  assert(DomainMap.count(BB) && "Requested BB did not have a domain");
  return isl_set_copy(DomainMap[BB]);
}

void Scop::removeErrorBlockDomains() {
  auto removeDomains = [this](BasicBlock *Start) {
    auto BBNode = DT.getNode(Start);
    for (auto ErrorChild : depth_first(BBNode)) {
      auto ErrorChildBlock = ErrorChild->getBlock();
      auto CurrentDomain = DomainMap[ErrorChildBlock];
      auto Empty = isl_set_empty(isl_set_get_space(CurrentDomain));
      DomainMap[ErrorChildBlock] = Empty;
      isl_set_free(CurrentDomain);
    }
  };

  SmallVector<Region *, 4> Todo = {&R};

  while (!Todo.empty()) {
    auto SubRegion = Todo.back();
    Todo.pop_back();

    if (!SD.isNonAffineSubRegion(SubRegion, &getRegion())) {
      for (auto &Child : *SubRegion)
        Todo.push_back(Child.get());
      continue;
    }
    if (containsErrorBlock(SubRegion->getNode(), getRegion(), LI, DT))
      removeDomains(SubRegion->getEntry());
  }

  for (auto BB : R.blocks())
    if (isErrorBlock(*BB, R, LI, DT))
      removeDomains(BB);
}

void Scop::buildDomains(Region *R) {

  bool IsOnlyNonAffineRegion = SD.isNonAffineSubRegion(R, R);
  auto *EntryBB = R->getEntry();
  auto *L = IsOnlyNonAffineRegion ? nullptr : LI.getLoopFor(EntryBB);
  int LD = getRelativeLoopDepth(L);
  auto *S = isl_set_universe(isl_space_set_alloc(getIslCtx(), 0, LD + 1));

  while (LD-- >= 0) {
    S = addDomainDimId(S, LD + 1, L);
    L = L->getParentLoop();
  }

  DomainMap[EntryBB] = S;

  if (IsOnlyNonAffineRegion)
    return;

  buildDomainsWithBranchConstraints(R);
  propagateDomainConstraints(R);

  // Error blocks and blocks dominated by them have been assumed to never be
  // executed. Representing them in the Scop does not add any value. In fact,
  // it is likely to cause issues during construction of the ScopStmts. The
  // contents of error blocks have not been verfied to be expressible and
  // will cause problems when building up a ScopStmt for them.
  // Furthermore, basic blocks dominated by error blocks may reference
  // instructions in the error block which, if the error block is not modeled,
  // can themselves not be constructed properly.
  removeErrorBlockDomains();
}

void Scop::buildDomainsWithBranchConstraints(Region *R) {
  auto &BoxedLoops = *SD.getBoxedLoops(&getRegion());

  // To create the domain for each block in R we iterate over all blocks and
  // subregions in R and propagate the conditions under which the current region
  // element is executed. To this end we iterate in reverse post order over R as
  // it ensures that we first visit all predecessors of a region node (either a
  // basic block or a subregion) before we visit the region node itself.
  // Initially, only the domain for the SCoP region entry block is set and from
  // there we propagate the current domain to all successors, however we add the
  // condition that the successor is actually executed next.
  // As we are only interested in non-loop carried constraints here we can
  // simply skip loop back edges.

  ReversePostOrderTraversal<Region *> RTraversal(R);
  for (auto *RN : RTraversal) {

    // Recurse for affine subregions but go on for basic blocks and non-affine
    // subregions.
    if (RN->isSubRegion()) {
      Region *SubRegion = RN->getNodeAs<Region>();
      if (!SD.isNonAffineSubRegion(SubRegion, &getRegion())) {
        buildDomainsWithBranchConstraints(SubRegion);
        continue;
      }
    }

    if (containsErrorBlock(RN, getRegion(), LI, DT))
      HasErrorBlock = true;

    BasicBlock *BB = getRegionNodeBasicBlock(RN);
    TerminatorInst *TI = BB->getTerminator();

    if (isa<UnreachableInst>(TI))
      continue;

    isl_set *Domain = DomainMap.lookup(BB);
    if (!Domain) {
      DEBUG(dbgs() << "\tSkip: " << BB->getName()
                   << ", it is only reachable from error blocks.\n");
      continue;
    }

    DEBUG(dbgs() << "\tVisit: " << BB->getName() << " : " << Domain << "\n");

    Loop *BBLoop = getRegionNodeLoop(RN, LI);
    int BBLoopDepth = getRelativeLoopDepth(BBLoop);

    // Build the condition sets for the successor nodes of the current region
    // node. If it is a non-affine subregion we will always execute the single
    // exit node, hence the single entry node domain is the condition set. For
    // basic blocks we use the helper function buildConditionSets.
    SmallVector<isl_set *, 8> ConditionSets;
    if (RN->isSubRegion())
      ConditionSets.push_back(isl_set_copy(Domain));
    else
      buildConditionSets(*this, TI, BBLoop, Domain, ConditionSets);

    // Now iterate over the successors and set their initial domain based on
    // their condition set. We skip back edges here and have to be careful when
    // we leave a loop not to keep constraints over a dimension that doesn't
    // exist anymore.
    assert(RN->isSubRegion() || TI->getNumSuccessors() == ConditionSets.size());
    for (unsigned u = 0, e = ConditionSets.size(); u < e; u++) {
      isl_set *CondSet = ConditionSets[u];
      BasicBlock *SuccBB = getRegionNodeSuccessor(RN, TI, u);

      // Skip back edges.
      if (DT.dominates(SuccBB, BB)) {
        isl_set_free(CondSet);
        continue;
      }

      // Do not adjust the number of dimensions if we enter a boxed loop or are
      // in a non-affine subregion or if the surrounding loop stays the same.
      Loop *SuccBBLoop = LI.getLoopFor(SuccBB);
      while (BoxedLoops.count(SuccBBLoop))
        SuccBBLoop = SuccBBLoop->getParentLoop();

      if (BBLoop != SuccBBLoop) {

        // Check if the edge to SuccBB is a loop entry or exit edge. If so
        // adjust the dimensionality accordingly. Lastly, if we leave a loop
        // and enter a new one we need to drop the old constraints.
        int SuccBBLoopDepth = getRelativeLoopDepth(SuccBBLoop);
        unsigned LoopDepthDiff = std::abs(BBLoopDepth - SuccBBLoopDepth);
        if (BBLoopDepth > SuccBBLoopDepth) {
          CondSet = isl_set_project_out(CondSet, isl_dim_set,
                                        isl_set_n_dim(CondSet) - LoopDepthDiff,
                                        LoopDepthDiff);
        } else if (SuccBBLoopDepth > BBLoopDepth) {
          assert(LoopDepthDiff == 1);
          CondSet = isl_set_add_dims(CondSet, isl_dim_set, 1);
          CondSet = addDomainDimId(CondSet, SuccBBLoopDepth, SuccBBLoop);
        } else if (BBLoopDepth >= 0) {
          assert(LoopDepthDiff <= 1);
          CondSet = isl_set_project_out(CondSet, isl_dim_set, BBLoopDepth, 1);
          CondSet = isl_set_add_dims(CondSet, isl_dim_set, 1);
          CondSet = addDomainDimId(CondSet, SuccBBLoopDepth, SuccBBLoop);
        }
      }

      // Set the domain for the successor or merge it with an existing domain in
      // case there are multiple paths (without loop back edges) to the
      // successor block.
      isl_set *&SuccDomain = DomainMap[SuccBB];
      if (!SuccDomain)
        SuccDomain = CondSet;
      else
        SuccDomain = isl_set_union(SuccDomain, CondSet);

      SuccDomain = isl_set_coalesce(SuccDomain);
      if (isl_set_n_basic_set(SuccDomain) > MaxConjunctsInDomain) {
        auto *Empty = isl_set_empty(isl_set_get_space(SuccDomain));
        isl_set_free(SuccDomain);
        SuccDomain = Empty;
        invalidate(ERROR_DOMAINCONJUNCTS, DebugLoc());
      }
      DEBUG(dbgs() << "\tSet SuccBB: " << SuccBB->getName() << " : "
                   << SuccDomain << "\n");
    }
  }
}

/// @brief Return the domain for @p BB wrt @p DomainMap.
///
/// This helper function will lookup @p BB in @p DomainMap but also handle the
/// case where @p BB is contained in a non-affine subregion using the region
/// tree obtained by @p RI.
static __isl_give isl_set *
getDomainForBlock(BasicBlock *BB, DenseMap<BasicBlock *, isl_set *> &DomainMap,
                  RegionInfo &RI) {
  auto DIt = DomainMap.find(BB);
  if (DIt != DomainMap.end())
    return isl_set_copy(DIt->getSecond());

  Region *R = RI.getRegionFor(BB);
  while (R->getEntry() == BB)
    R = R->getParent();
  return getDomainForBlock(R->getEntry(), DomainMap, RI);
}

void Scop::propagateDomainConstraints(Region *R) {
  // Iterate over the region R and propagate the domain constrains from the
  // predecessors to the current node. In contrast to the
  // buildDomainsWithBranchConstraints function, this one will pull the domain
  // information from the predecessors instead of pushing it to the successors.
  // Additionally, we assume the domains to be already present in the domain
  // map here. However, we iterate again in reverse post order so we know all
  // predecessors have been visited before a block or non-affine subregion is
  // visited.

  // The set of boxed loops (loops in non-affine subregions) for this SCoP.
  auto &BoxedLoops = *SD.getBoxedLoops(&getRegion());

  ReversePostOrderTraversal<Region *> RTraversal(R);
  for (auto *RN : RTraversal) {

    // Recurse for affine subregions but go on for basic blocks and non-affine
    // subregions.
    if (RN->isSubRegion()) {
      Region *SubRegion = RN->getNodeAs<Region>();
      if (!SD.isNonAffineSubRegion(SubRegion, &getRegion())) {
        propagateDomainConstraints(SubRegion);
        continue;
      }
    }

    // Get the domain for the current block and check if it was initialized or
    // not. The only way it was not is if this block is only reachable via error
    // blocks, thus will not be executed under the assumptions we make. Such
    // blocks have to be skipped as their predecessors might not have domains
    // either. It would not benefit us to compute the domain anyway, only the
    // domains of the error blocks that are reachable from non-error blocks
    // are needed to generate assumptions.
    BasicBlock *BB = getRegionNodeBasicBlock(RN);
    isl_set *&Domain = DomainMap[BB];
    if (!Domain) {
      DEBUG(dbgs() << "\tSkip: " << BB->getName()
                   << ", it is only reachable from error blocks.\n");
      DomainMap.erase(BB);
      continue;
    }
    DEBUG(dbgs() << "\tVisit: " << BB->getName() << " : " << Domain << "\n");

    Loop *BBLoop = getRegionNodeLoop(RN, LI);
    int BBLoopDepth = getRelativeLoopDepth(BBLoop);

    isl_set *PredDom = isl_set_empty(isl_set_get_space(Domain));
    for (auto *PredBB : predecessors(BB)) {

      // Skip backedges
      if (DT.dominates(BB, PredBB))
        continue;

      isl_set *PredBBDom = nullptr;

      // Handle the SCoP entry block with its outside predecessors.
      if (!getRegion().contains(PredBB))
        PredBBDom = isl_set_universe(isl_set_get_space(PredDom));

      if (!PredBBDom) {
        // Determine the loop depth of the predecessor and adjust its domain to
        // the domain of the current block. This can mean we have to:
        //  o) Drop a dimension if this block is the exit of a loop, not the
        //     header of a new loop and the predecessor was part of the loop.
        //  o) Add an unconstrainted new dimension if this block is the header
        //     of a loop and the predecessor is not part of it.
        //  o) Drop the information about the innermost loop dimension when the
        //     predecessor and the current block are surrounded by different
        //     loops in the same depth.
        PredBBDom = getDomainForBlock(PredBB, DomainMap, *R->getRegionInfo());
        Loop *PredBBLoop = LI.getLoopFor(PredBB);
        while (BoxedLoops.count(PredBBLoop))
          PredBBLoop = PredBBLoop->getParentLoop();

        int PredBBLoopDepth = getRelativeLoopDepth(PredBBLoop);
        unsigned LoopDepthDiff = std::abs(BBLoopDepth - PredBBLoopDepth);
        if (BBLoopDepth < PredBBLoopDepth)
          PredBBDom = isl_set_project_out(
              PredBBDom, isl_dim_set, isl_set_n_dim(PredBBDom) - LoopDepthDiff,
              LoopDepthDiff);
        else if (PredBBLoopDepth < BBLoopDepth) {
          assert(LoopDepthDiff == 1);
          PredBBDom = isl_set_add_dims(PredBBDom, isl_dim_set, 1);
        } else if (BBLoop != PredBBLoop && BBLoopDepth >= 0) {
          assert(LoopDepthDiff <= 1);
          PredBBDom = isl_set_drop_constraints_involving_dims(
              PredBBDom, isl_dim_set, BBLoopDepth, 1);
        }
      }

      PredDom = isl_set_union(PredDom, PredBBDom);
    }

    // Under the union of all predecessor conditions we can reach this block.
    Domain = isl_set_coalesce(isl_set_intersect(Domain, PredDom));

    if (BBLoop && BBLoop->getHeader() == BB && getRegion().contains(BBLoop))
      addLoopBoundsToHeaderDomain(BBLoop);

    // Add assumptions for error blocks.
    if (containsErrorBlock(RN, getRegion(), LI, DT)) {
      IsOptimized = true;
      isl_set *DomPar = isl_set_params(isl_set_copy(Domain));
      addAssumption(ERRORBLOCK, isl_set_complement(DomPar),
                    BB->getTerminator()->getDebugLoc());
    }
  }
}

/// @brief Create a map from SetSpace -> SetSpace where the dimensions @p Dim
///        is incremented by one and all other dimensions are equal, e.g.,
///             [i0, i1, i2, i3] -> [i0, i1, i2 + 1, i3]
///        if @p Dim is 2 and @p SetSpace has 4 dimensions.
static __isl_give isl_map *
createNextIterationMap(__isl_take isl_space *SetSpace, unsigned Dim) {
  auto *MapSpace = isl_space_map_from_set(SetSpace);
  auto *NextIterationMap = isl_map_universe(isl_space_copy(MapSpace));
  for (unsigned u = 0; u < isl_map_n_in(NextIterationMap); u++)
    if (u != Dim)
      NextIterationMap =
          isl_map_equate(NextIterationMap, isl_dim_in, u, isl_dim_out, u);
  auto *C = isl_constraint_alloc_equality(isl_local_space_from_space(MapSpace));
  C = isl_constraint_set_constant_si(C, 1);
  C = isl_constraint_set_coefficient_si(C, isl_dim_in, Dim, 1);
  C = isl_constraint_set_coefficient_si(C, isl_dim_out, Dim, -1);
  NextIterationMap = isl_map_add_constraint(NextIterationMap, C);
  return NextIterationMap;
}

void Scop::addLoopBoundsToHeaderDomain(Loop *L) {
  int LoopDepth = getRelativeLoopDepth(L);
  assert(LoopDepth >= 0 && "Loop in region should have at least depth one");

  BasicBlock *HeaderBB = L->getHeader();
  assert(DomainMap.count(HeaderBB));
  isl_set *&HeaderBBDom = DomainMap[HeaderBB];

  isl_map *NextIterationMap =
      createNextIterationMap(isl_set_get_space(HeaderBBDom), LoopDepth);

  isl_set *UnionBackedgeCondition =
      isl_set_empty(isl_set_get_space(HeaderBBDom));

  SmallVector<llvm::BasicBlock *, 4> LatchBlocks;
  L->getLoopLatches(LatchBlocks);

  for (BasicBlock *LatchBB : LatchBlocks) {

    // If the latch is only reachable via error statements we skip it.
    isl_set *LatchBBDom = DomainMap.lookup(LatchBB);
    if (!LatchBBDom)
      continue;

    isl_set *BackedgeCondition = nullptr;

    TerminatorInst *TI = LatchBB->getTerminator();
    BranchInst *BI = dyn_cast<BranchInst>(TI);
    if (BI && BI->isUnconditional())
      BackedgeCondition = isl_set_copy(LatchBBDom);
    else {
      SmallVector<isl_set *, 8> ConditionSets;
      int idx = BI->getSuccessor(0) != HeaderBB;
      buildConditionSets(*this, TI, L, LatchBBDom, ConditionSets);

      // Free the non back edge condition set as we do not need it.
      isl_set_free(ConditionSets[1 - idx]);

      BackedgeCondition = ConditionSets[idx];
    }

    int LatchLoopDepth = getRelativeLoopDepth(LI.getLoopFor(LatchBB));
    assert(LatchLoopDepth >= LoopDepth);
    BackedgeCondition =
        isl_set_project_out(BackedgeCondition, isl_dim_set, LoopDepth + 1,
                            LatchLoopDepth - LoopDepth);
    UnionBackedgeCondition =
        isl_set_union(UnionBackedgeCondition, BackedgeCondition);
  }

  isl_map *ForwardMap = isl_map_lex_le(isl_set_get_space(HeaderBBDom));
  for (int i = 0; i < LoopDepth; i++)
    ForwardMap = isl_map_equate(ForwardMap, isl_dim_in, i, isl_dim_out, i);

  isl_set *UnionBackedgeConditionComplement =
      isl_set_complement(UnionBackedgeCondition);
  UnionBackedgeConditionComplement = isl_set_lower_bound_si(
      UnionBackedgeConditionComplement, isl_dim_set, LoopDepth, 0);
  UnionBackedgeConditionComplement =
      isl_set_apply(UnionBackedgeConditionComplement, ForwardMap);
  HeaderBBDom = isl_set_subtract(HeaderBBDom, UnionBackedgeConditionComplement);
  HeaderBBDom = isl_set_apply(HeaderBBDom, NextIterationMap);

  auto Parts = partitionSetParts(HeaderBBDom, LoopDepth);
  HeaderBBDom = Parts.second;

  // Check if there is a <nsw> tagged AddRec for this loop and if so do not add
  // the bounded assumptions to the context as they are already implied by the
  // <nsw> tag.
  if (Affinator.hasNSWAddRecForLoop(L)) {
    isl_set_free(Parts.first);
    return;
  }

  isl_set *UnboundedCtx = isl_set_params(Parts.first);
  isl_set *BoundedCtx = isl_set_complement(UnboundedCtx);
  addAssumption(INFINITELOOP, BoundedCtx,
                HeaderBB->getTerminator()->getDebugLoc());
}

void Scop::buildAliasChecks(AliasAnalysis &AA) {
  if (!PollyUseRuntimeAliasChecks)
    return;

  if (buildAliasGroups(AA))
    return;

  // If a problem occurs while building the alias groups we need to delete
  // this SCoP and pretend it wasn't valid in the first place. To this end
  // we make the assumed context infeasible.
  invalidate(ALIASING, DebugLoc());

  DEBUG(dbgs() << "\n\nNOTE: Run time checks for " << getNameStr()
               << " could not be created as the number of parameters involved "
                  "is too high. The SCoP will be "
                  "dismissed.\nUse:\n\t--polly-rtc-max-parameters=X\nto adjust "
                  "the maximal number of parameters but be advised that the "
                  "compile time might increase exponentially.\n\n");
}

bool Scop::buildAliasGroups(AliasAnalysis &AA) {
  // To create sound alias checks we perform the following steps:
  //   o) Use the alias analysis and an alias set tracker to build alias sets
  //      for all memory accesses inside the SCoP.
  //   o) For each alias set we then map the aliasing pointers back to the
  //      memory accesses we know, thus obtain groups of memory accesses which
  //      might alias.
  //   o) We divide each group based on the domains of the minimal/maximal
  //      accesses. That means two minimal/maximal accesses are only in a group
  //      if their access domains intersect, otherwise they are in different
  //      ones.
  //   o) We partition each group into read only and non read only accesses.
  //   o) For each group with more than one base pointer we then compute minimal
  //      and maximal accesses to each array of a group in read only and non
  //      read only partitions separately.
  using AliasGroupTy = SmallVector<MemoryAccess *, 4>;

  AliasSetTracker AST(AA);

  DenseMap<Value *, MemoryAccess *> PtrToAcc;
  DenseSet<Value *> HasWriteAccess;
  for (ScopStmt &Stmt : *this) {

    // Skip statements with an empty domain as they will never be executed.
    isl_set *StmtDomain = Stmt.getDomain();
    bool StmtDomainEmpty = isl_set_is_empty(StmtDomain);
    isl_set_free(StmtDomain);
    if (StmtDomainEmpty)
      continue;

    for (MemoryAccess *MA : Stmt) {
      if (MA->isScalarKind())
        continue;
      if (!MA->isRead())
        HasWriteAccess.insert(MA->getBaseAddr());
      MemAccInst Acc(MA->getAccessInstruction());
      PtrToAcc[Acc.getPointerOperand()] = MA;
      AST.add(Acc);
    }
  }

  SmallVector<AliasGroupTy, 4> AliasGroups;
  for (AliasSet &AS : AST) {
    if (AS.isMustAlias() || AS.isForwardingAliasSet())
      continue;
    AliasGroupTy AG;
    for (auto PR : AS)
      AG.push_back(PtrToAcc[PR.getValue()]);
    assert(AG.size() > 1 &&
           "Alias groups should contain at least two accesses");
    AliasGroups.push_back(std::move(AG));
  }

  // Split the alias groups based on their domain.
  for (unsigned u = 0; u < AliasGroups.size(); u++) {
    AliasGroupTy NewAG;
    AliasGroupTy &AG = AliasGroups[u];
    AliasGroupTy::iterator AGI = AG.begin();
    isl_set *AGDomain = getAccessDomain(*AGI);
    while (AGI != AG.end()) {
      MemoryAccess *MA = *AGI;
      isl_set *MADomain = getAccessDomain(MA);
      if (isl_set_is_disjoint(AGDomain, MADomain)) {
        NewAG.push_back(MA);
        AGI = AG.erase(AGI);
        isl_set_free(MADomain);
      } else {
        AGDomain = isl_set_union(AGDomain, MADomain);
        AGI++;
      }
    }
    if (NewAG.size() > 1)
      AliasGroups.push_back(std::move(NewAG));
    isl_set_free(AGDomain);
  }

  auto &F = *getRegion().getEntry()->getParent();
  MapVector<const Value *, SmallPtrSet<MemoryAccess *, 8>> ReadOnlyPairs;
  SmallPtrSet<const Value *, 4> NonReadOnlyBaseValues;
  for (AliasGroupTy &AG : AliasGroups) {
    NonReadOnlyBaseValues.clear();
    ReadOnlyPairs.clear();

    if (AG.size() < 2) {
      AG.clear();
      continue;
    }

    for (auto II = AG.begin(); II != AG.end();) {
      emitOptimizationRemarkAnalysis(
          F.getContext(), DEBUG_TYPE, F,
          (*II)->getAccessInstruction()->getDebugLoc(),
          "Possibly aliasing pointer, use restrict keyword.");

      Value *BaseAddr = (*II)->getBaseAddr();
      if (HasWriteAccess.count(BaseAddr)) {
        NonReadOnlyBaseValues.insert(BaseAddr);
        II++;
      } else {
        ReadOnlyPairs[BaseAddr].insert(*II);
        II = AG.erase(II);
      }
    }

    // If we don't have read only pointers check if there are at least two
    // non read only pointers, otherwise clear the alias group.
    if (ReadOnlyPairs.empty() && NonReadOnlyBaseValues.size() <= 1) {
      AG.clear();
      continue;
    }

    // If we don't have non read only pointers clear the alias group.
    if (NonReadOnlyBaseValues.empty()) {
      AG.clear();
      continue;
    }

    // Calculate minimal and maximal accesses for non read only accesses.
    MinMaxAliasGroups.emplace_back();
    MinMaxVectorPairTy &pair = MinMaxAliasGroups.back();
    MinMaxVectorTy &MinMaxAccessesNonReadOnly = pair.first;
    MinMaxVectorTy &MinMaxAccessesReadOnly = pair.second;
    MinMaxAccessesNonReadOnly.reserve(AG.size());

    isl_union_map *Accesses = isl_union_map_empty(getParamSpace());

    // AG contains only non read only accesses.
    for (MemoryAccess *MA : AG)
      Accesses = isl_union_map_add_map(Accesses, MA->getAccessRelation());

    bool Valid = calculateMinMaxAccess(Accesses, getDomains(),
                                       MinMaxAccessesNonReadOnly);

    // Bail out if the number of values we need to compare is too large.
    // This is important as the number of comparisions grows quadratically with
    // the number of values we need to compare.
    if (!Valid || (MinMaxAccessesNonReadOnly.size() + !ReadOnlyPairs.empty() >
                   RunTimeChecksMaxArraysPerGroup))
      return false;

    // Calculate minimal and maximal accesses for read only accesses.
    MinMaxAccessesReadOnly.reserve(ReadOnlyPairs.size());
    Accesses = isl_union_map_empty(getParamSpace());

    for (const auto &ReadOnlyPair : ReadOnlyPairs)
      for (MemoryAccess *MA : ReadOnlyPair.second)
        Accesses = isl_union_map_add_map(Accesses, MA->getAccessRelation());

    Valid =
        calculateMinMaxAccess(Accesses, getDomains(), MinMaxAccessesReadOnly);

    if (!Valid)
      return false;
  }

  return true;
}

/// @brief Get the smallest loop that contains @p R but is not in @p R.
static Loop *getLoopSurroundingRegion(Region &R, LoopInfo &LI) {
  // Start with the smallest loop containing the entry and expand that
  // loop until it contains all blocks in the region. If there is a loop
  // containing all blocks in the region check if it is itself contained
  // and if so take the parent loop as it will be the smallest containing
  // the region but not contained by it.
  Loop *L = LI.getLoopFor(R.getEntry());
  while (L) {
    bool AllContained = true;
    for (auto *BB : R.blocks())
      AllContained &= L->contains(BB);
    if (AllContained)
      break;
    L = L->getParentLoop();
  }

  return L ? (R.contains(L) ? L->getParentLoop() : L) : nullptr;
}

static unsigned getMaxLoopDepthInRegion(const Region &R, LoopInfo &LI,
                                        ScopDetection &SD) {

  const ScopDetection::BoxedLoopsSetTy *BoxedLoops = SD.getBoxedLoops(&R);

  unsigned MinLD = INT_MAX, MaxLD = 0;
  for (BasicBlock *BB : R.blocks()) {
    if (Loop *L = LI.getLoopFor(BB)) {
      if (!R.contains(L))
        continue;
      if (BoxedLoops && BoxedLoops->count(L))
        continue;
      unsigned LD = L->getLoopDepth();
      MinLD = std::min(MinLD, LD);
      MaxLD = std::max(MaxLD, LD);
    }
  }

  // Handle the case that there is no loop in the SCoP first.
  if (MaxLD == 0)
    return 1;

  assert(MinLD >= 1 && "Minimal loop depth should be at least one");
  assert(MaxLD >= MinLD &&
         "Maximal loop depth was smaller than mininaml loop depth?");
  return MaxLD - MinLD + 1;
}

Scop::Scop(Region &R, AccFuncMapType &AccFuncMap, ScopDetection &SD,
           ScalarEvolution &ScalarEvolution, DominatorTree &DT, LoopInfo &LI,
           isl_ctx *Context, unsigned MaxLoopDepth)
    : LI(LI), DT(DT), SE(&ScalarEvolution), SD(SD), R(R),
      AccFuncMap(AccFuncMap), IsOptimized(false),
      HasSingleExitEdge(R.getExitingBlock()), HasErrorBlock(false),
      MaxLoopDepth(MaxLoopDepth), IslCtx(Context), Context(nullptr),
      Affinator(this), AssumedContext(nullptr), BoundaryContext(nullptr),
      Schedule(nullptr) {}

void Scop::init(AliasAnalysis &AA, AssumptionCache &AC) {
  buildContext();
  addUserAssumptions(AC);
  buildInvariantEquivalenceClasses();

  buildDomains(&R);

  // Remove empty and ignored statements.
  // Exit early in case there are no executable statements left in this scop.
  simplifySCoP(true);
  if (Stmts.empty())
    return;

  // The ScopStmts now have enough information to initialize themselves.
  for (ScopStmt &Stmt : Stmts)
    Stmt.init();

  buildSchedule();

  if (isl_set_is_empty(AssumedContext))
    return;

  updateAccessDimensionality();
  realignParams();
  addParameterBounds();
  addUserContext();
  buildBoundaryContext();
  simplifyContexts();
  buildAliasChecks(AA);

  hoistInvariantLoads();
  simplifySCoP(false);
}

Scop::~Scop() {
  isl_set_free(Context);
  isl_set_free(AssumedContext);
  isl_set_free(BoundaryContext);
  isl_schedule_free(Schedule);

  for (auto It : DomainMap)
    isl_set_free(It.second);

  // Free the alias groups
  for (MinMaxVectorPairTy &MinMaxAccessPair : MinMaxAliasGroups) {
    for (MinMaxAccessTy &MMA : MinMaxAccessPair.first) {
      isl_pw_multi_aff_free(MMA.first);
      isl_pw_multi_aff_free(MMA.second);
    }
    for (MinMaxAccessTy &MMA : MinMaxAccessPair.second) {
      isl_pw_multi_aff_free(MMA.first);
      isl_pw_multi_aff_free(MMA.second);
    }
  }

  for (const auto &IAClass : InvariantEquivClasses)
    isl_set_free(std::get<2>(IAClass));
}

void Scop::updateAccessDimensionality() {
  for (auto &Stmt : *this)
    for (auto &Access : Stmt)
      Access->updateDimensionality();
}

void Scop::simplifySCoP(bool RemoveIgnoredStmts) {
  for (auto StmtIt = Stmts.begin(), StmtEnd = Stmts.end(); StmtIt != StmtEnd;) {
    ScopStmt &Stmt = *StmtIt;
    RegionNode *RN = Stmt.isRegionStmt()
                         ? Stmt.getRegion()->getNode()
                         : getRegion().getBBNode(Stmt.getBasicBlock());

    bool RemoveStmt = StmtIt->isEmpty();
    if (!RemoveStmt)
      RemoveStmt = isl_set_is_empty(DomainMap[getRegionNodeBasicBlock(RN)]);
    if (!RemoveStmt)
      RemoveStmt = (RemoveIgnoredStmts && isIgnored(RN));

    // Remove read only statements only after invariant loop hoisting.
    if (!RemoveStmt && !RemoveIgnoredStmts) {
      bool OnlyRead = true;
      for (MemoryAccess *MA : Stmt) {
        if (MA->isRead())
          continue;

        OnlyRead = false;
        break;
      }

      RemoveStmt = OnlyRead;
    }

    if (RemoveStmt) {
      // Remove the statement because it is unnecessary.
      if (Stmt.isRegionStmt())
        for (BasicBlock *BB : Stmt.getRegion()->blocks())
          StmtMap.erase(BB);
      else
        StmtMap.erase(Stmt.getBasicBlock());

      StmtIt = Stmts.erase(StmtIt);
      continue;
    }

    StmtIt++;
  }
}

const InvariantEquivClassTy *Scop::lookupInvariantEquivClass(Value *Val) const {
  LoadInst *LInst = dyn_cast<LoadInst>(Val);
  if (!LInst)
    return nullptr;

  if (Value *Rep = InvEquivClassVMap.lookup(LInst))
    LInst = cast<LoadInst>(Rep);

  const SCEV *PointerSCEV = SE->getSCEV(LInst->getPointerOperand());
  for (auto &IAClass : InvariantEquivClasses)
    if (PointerSCEV == std::get<0>(IAClass))
      return &IAClass;

  return nullptr;
}

void Scop::addInvariantLoads(ScopStmt &Stmt, MemoryAccessList &InvMAs) {

  // Get the context under which the statement is executed.
  isl_set *DomainCtx = isl_set_params(Stmt.getDomain());
  DomainCtx = isl_set_remove_redundancies(DomainCtx);
  DomainCtx = isl_set_detect_equalities(DomainCtx);
  DomainCtx = isl_set_coalesce(DomainCtx);

  // Project out all parameters that relate to loads in the statement. Otherwise
  // we could have cyclic dependences on the constraints under which the
  // hoisted loads are executed and we could not determine an order in which to
  // pre-load them. This happens because not only lower bounds are part of the
  // domain but also upper bounds.
  for (MemoryAccess *MA : InvMAs) {
    Instruction *AccInst = MA->getAccessInstruction();
    if (SE->isSCEVable(AccInst->getType())) {
      SetVector<Value *> Values;
      for (const SCEV *Parameter : Parameters) {
        Values.clear();
        findValues(Parameter, Values);
        if (!Values.count(AccInst))
          continue;

        if (isl_id *ParamId = getIdForParam(Parameter)) {
          int Dim = isl_set_find_dim_by_id(DomainCtx, isl_dim_param, ParamId);
          DomainCtx = isl_set_eliminate(DomainCtx, isl_dim_param, Dim, 1);
          isl_id_free(ParamId);
        }
      }
    }
  }

  for (MemoryAccess *MA : InvMAs) {
    // Check for another invariant access that accesses the same location as
    // MA and if found consolidate them. Otherwise create a new equivalence
    // class at the end of InvariantEquivClasses.
    LoadInst *LInst = cast<LoadInst>(MA->getAccessInstruction());
    const SCEV *PointerSCEV = SE->getSCEV(LInst->getPointerOperand());

    bool Consolidated = false;
    for (auto &IAClass : InvariantEquivClasses) {
      if (PointerSCEV != std::get<0>(IAClass))
        continue;

      Consolidated = true;

      // Add MA to the list of accesses that are in this class.
      auto &MAs = std::get<1>(IAClass);
      MAs.push_front(MA);

      // Unify the execution context of the class and this statement.
      isl_set *&IAClassDomainCtx = std::get<2>(IAClass);
      if (IAClassDomainCtx)
        IAClassDomainCtx = isl_set_coalesce(
            isl_set_union(IAClassDomainCtx, isl_set_copy(DomainCtx)));
      else
        IAClassDomainCtx = isl_set_copy(DomainCtx);
      break;
    }

    if (Consolidated)
      continue;

    // If we did not consolidate MA, thus did not find an equivalence class
    // for it, we create a new one.
    InvariantEquivClasses.emplace_back(PointerSCEV, MemoryAccessList{MA},
                                       isl_set_copy(DomainCtx));
  }

  isl_set_free(DomainCtx);
}

bool Scop::isHoistableAccess(MemoryAccess *Access,
                             __isl_keep isl_union_map *Writes) {
  // TODO: Loads that are not loop carried, hence are in a statement with
  //       zero iterators, are by construction invariant, though we
  //       currently "hoist" them anyway. This is necessary because we allow
  //       them to be treated as parameters (e.g., in conditions) and our code
  //       generation would otherwise use the old value.

  auto &Stmt = *Access->getStatement();
  BasicBlock *BB =
      Stmt.isBlockStmt() ? Stmt.getBasicBlock() : Stmt.getRegion()->getEntry();

  if (Access->isScalarKind() || Access->isWrite() || !Access->isAffine())
    return false;

  // Skip accesses that have an invariant base pointer which is defined but
  // not loaded inside the SCoP. This can happened e.g., if a readnone call
  // returns a pointer that is used as a base address. However, as we want
  // to hoist indirect pointers, we allow the base pointer to be defined in
  // the region if it is also a memory access. Each ScopArrayInfo object
  // that has a base pointer origin has a base pointer that is loaded and
  // that it is invariant, thus it will be hoisted too. However, if there is
  // no base pointer origin we check that the base pointer is defined
  // outside the region.
  const ScopArrayInfo *SAI = Access->getScopArrayInfo();
  while (auto *BasePtrOriginSAI = SAI->getBasePtrOriginSAI())
    SAI = BasePtrOriginSAI;

  if (auto *BasePtrInst = dyn_cast<Instruction>(SAI->getBasePtr()))
    if (R.contains(BasePtrInst))
      return false;

  // Skip accesses in non-affine subregions as they might not be executed
  // under the same condition as the entry of the non-affine subregion.
  if (BB != Access->getAccessInstruction()->getParent())
    return false;

  isl_map *AccessRelation = Access->getAccessRelation();

  // Skip accesses that have an empty access relation. These can be caused
  // by multiple offsets with a type cast in-between that cause the overall
  // byte offset to be not divisible by the new types sizes.
  if (isl_map_is_empty(AccessRelation)) {
    isl_map_free(AccessRelation);
    return false;
  }

  if (isl_map_involves_dims(AccessRelation, isl_dim_in, 0,
                            Stmt.getNumIterators())) {
    isl_map_free(AccessRelation);
    return false;
  }

  AccessRelation = isl_map_intersect_domain(AccessRelation, Stmt.getDomain());
  isl_set *AccessRange = isl_map_range(AccessRelation);

  isl_union_map *Written = isl_union_map_intersect_range(
      isl_union_map_copy(Writes), isl_union_set_from_set(AccessRange));
  bool IsWritten = !isl_union_map_is_empty(Written);
  isl_union_map_free(Written);

  if (IsWritten)
    return false;

  return true;
}

void Scop::verifyInvariantLoads() {
  auto &RIL = *SD.getRequiredInvariantLoads(&getRegion());
  for (LoadInst *LI : RIL) {
    assert(LI && getRegion().contains(LI));
    ScopStmt *Stmt = getStmtForBasicBlock(LI->getParent());
    if (Stmt && Stmt->getArrayAccessOrNULLFor(LI)) {
      invalidate(INVARIANTLOAD, LI->getDebugLoc());
      return;
    }
  }
}

void Scop::hoistInvariantLoads() {
  isl_union_map *Writes = getWrites();
  for (ScopStmt &Stmt : *this) {

    MemoryAccessList InvariantAccesses;

    for (MemoryAccess *Access : Stmt)
      if (isHoistableAccess(Access, Writes))
        InvariantAccesses.push_front(Access);

    // We inserted invariant accesses always in the front but need them to be
    // sorted in a "natural order". The statements are already sorted in reverse
    // post order and that suffices for the accesses too. The reason we require
    // an order in the first place is the dependences between invariant loads
    // that can be caused by indirect loads.
    InvariantAccesses.reverse();

    // Transfer the memory access from the statement to the SCoP.
    Stmt.removeMemoryAccesses(InvariantAccesses);
    addInvariantLoads(Stmt, InvariantAccesses);
  }
  isl_union_map_free(Writes);

  verifyInvariantLoads();
}

const ScopArrayInfo *
Scop::getOrCreateScopArrayInfo(Value *BasePtr, Type *ElementType,
                               ArrayRef<const SCEV *> Sizes,
                               ScopArrayInfo::MemoryKind Kind) {
  auto &SAI = ScopArrayInfoMap[std::make_pair(BasePtr, Kind)];
  if (!SAI) {
    auto &DL = getRegion().getEntry()->getModule()->getDataLayout();
    SAI.reset(new ScopArrayInfo(BasePtr, ElementType, getIslCtx(), Sizes, Kind,
                                DL, this));
  } else {
    // In case of mismatching array sizes, we bail out by setting the run-time
    // context to false.
    if (!SAI->updateSizes(Sizes))
      invalidate(DELINEARIZATION, DebugLoc());
  }
  return SAI.get();
}

const ScopArrayInfo *Scop::getScopArrayInfo(Value *BasePtr,
                                            ScopArrayInfo::MemoryKind Kind) {
  auto *SAI = ScopArrayInfoMap[std::make_pair(BasePtr, Kind)].get();
  assert(SAI && "No ScopArrayInfo available for this base pointer");
  return SAI;
}

std::string Scop::getContextStr() const { return stringFromIslObj(Context); }
std::string Scop::getAssumedContextStr() const {
  return stringFromIslObj(AssumedContext);
}
std::string Scop::getBoundaryContextStr() const {
  return stringFromIslObj(BoundaryContext);
}

std::string Scop::getNameStr() const {
  std::string ExitName, EntryName;
  raw_string_ostream ExitStr(ExitName);
  raw_string_ostream EntryStr(EntryName);

  R.getEntry()->printAsOperand(EntryStr, false);
  EntryStr.str();

  if (R.getExit()) {
    R.getExit()->printAsOperand(ExitStr, false);
    ExitStr.str();
  } else
    ExitName = "FunctionExit";

  return EntryName + "---" + ExitName;
}

__isl_give isl_set *Scop::getContext() const { return isl_set_copy(Context); }
__isl_give isl_space *Scop::getParamSpace() const {
  return isl_set_get_space(Context);
}

__isl_give isl_set *Scop::getAssumedContext() const {
  return isl_set_copy(AssumedContext);
}

__isl_give isl_set *Scop::getRuntimeCheckContext() const {
  isl_set *RuntimeCheckContext = getAssumedContext();
  RuntimeCheckContext =
      isl_set_intersect(RuntimeCheckContext, getBoundaryContext());
  RuntimeCheckContext = simplifyAssumptionContext(RuntimeCheckContext, *this);
  return RuntimeCheckContext;
}

bool Scop::hasFeasibleRuntimeContext() const {
  isl_set *RuntimeCheckContext = getRuntimeCheckContext();
  RuntimeCheckContext = addNonEmptyDomainConstraints(RuntimeCheckContext);
  bool IsFeasible = !isl_set_is_empty(RuntimeCheckContext);
  isl_set_free(RuntimeCheckContext);
  return IsFeasible;
}

static std::string toString(AssumptionKind Kind) {
  switch (Kind) {
  case ALIASING:
    return "No-aliasing";
  case INBOUNDS:
    return "Inbounds";
  case WRAPPING:
    return "No-overflows";
  case ALIGNMENT:
    return "Alignment";
  case ERRORBLOCK:
    return "No-error";
  case INFINITELOOP:
    return "Finite loop";
  case INVARIANTLOAD:
    return "Invariant load";
  case DELINEARIZATION:
    return "Delinearization";
  case ERROR_DOMAINCONJUNCTS:
    return "Low number of domain conjuncts";
  }
  llvm_unreachable("Unknown AssumptionKind!");
}

void Scop::trackAssumption(AssumptionKind Kind, __isl_keep isl_set *Set,
                           DebugLoc Loc) {
  if (isl_set_is_subset(Context, Set))
    return;

  if (isl_set_is_subset(AssumedContext, Set))
    return;

  auto &F = *getRegion().getEntry()->getParent();
  std::string Msg = toString(Kind) + " assumption:\t" + stringFromIslObj(Set);
  emitOptimizationRemarkAnalysis(F.getContext(), DEBUG_TYPE, F, Loc, Msg);
}

void Scop::addAssumption(AssumptionKind Kind, __isl_take isl_set *Set,
                         DebugLoc Loc) {
  trackAssumption(Kind, Set, Loc);
  AssumedContext = isl_set_intersect(AssumedContext, Set);

  int NSets = isl_set_n_basic_set(AssumedContext);
  if (NSets >= MaxDisjunctsAssumed) {
    isl_space *Space = isl_set_get_space(AssumedContext);
    isl_set_free(AssumedContext);
    AssumedContext = isl_set_empty(Space);
  }

  AssumedContext = isl_set_coalesce(AssumedContext);
}

void Scop::invalidate(AssumptionKind Kind, DebugLoc Loc) {
  addAssumption(Kind, isl_set_empty(getParamSpace()), Loc);
}

__isl_give isl_set *Scop::getBoundaryContext() const {
  return isl_set_copy(BoundaryContext);
}

void Scop::printContext(raw_ostream &OS) const {
  OS << "Context:\n";

  if (!Context) {
    OS.indent(4) << "n/a\n\n";
    return;
  }

  OS.indent(4) << getContextStr() << "\n";

  OS.indent(4) << "Assumed Context:\n";
  if (!AssumedContext) {
    OS.indent(4) << "n/a\n\n";
    return;
  }

  OS.indent(4) << getAssumedContextStr() << "\n";

  OS.indent(4) << "Boundary Context:\n";
  if (!BoundaryContext) {
    OS.indent(4) << "n/a\n\n";
    return;
  }

  OS.indent(4) << getBoundaryContextStr() << "\n";

  for (const SCEV *Parameter : Parameters) {
    int Dim = ParameterIds.find(Parameter)->second;
    OS.indent(4) << "p" << Dim << ": " << *Parameter << "\n";
  }
}

void Scop::printAliasAssumptions(raw_ostream &OS) const {
  int noOfGroups = 0;
  for (const MinMaxVectorPairTy &Pair : MinMaxAliasGroups) {
    if (Pair.second.size() == 0)
      noOfGroups += 1;
    else
      noOfGroups += Pair.second.size();
  }

  OS.indent(4) << "Alias Groups (" << noOfGroups << "):\n";
  if (MinMaxAliasGroups.empty()) {
    OS.indent(8) << "n/a\n";
    return;
  }

  for (const MinMaxVectorPairTy &Pair : MinMaxAliasGroups) {

    // If the group has no read only accesses print the write accesses.
    if (Pair.second.empty()) {
      OS.indent(8) << "[[";
      for (const MinMaxAccessTy &MMANonReadOnly : Pair.first) {
        OS << " <" << MMANonReadOnly.first << ", " << MMANonReadOnly.second
           << ">";
      }
      OS << " ]]\n";
    }

    for (const MinMaxAccessTy &MMAReadOnly : Pair.second) {
      OS.indent(8) << "[[";
      OS << " <" << MMAReadOnly.first << ", " << MMAReadOnly.second << ">";
      for (const MinMaxAccessTy &MMANonReadOnly : Pair.first) {
        OS << " <" << MMANonReadOnly.first << ", " << MMANonReadOnly.second
           << ">";
      }
      OS << " ]]\n";
    }
  }
}

void Scop::printStatements(raw_ostream &OS) const {
  OS << "Statements {\n";

  for (const ScopStmt &Stmt : *this)
    OS.indent(4) << Stmt;

  OS.indent(4) << "}\n";
}

void Scop::printArrayInfo(raw_ostream &OS) const {
  OS << "Arrays {\n";

  for (auto &Array : arrays())
    Array.second->print(OS);

  OS.indent(4) << "}\n";

  OS.indent(4) << "Arrays (Bounds as pw_affs) {\n";

  for (auto &Array : arrays())
    Array.second->print(OS, /* SizeAsPwAff */ true);

  OS.indent(4) << "}\n";
}

void Scop::print(raw_ostream &OS) const {
  OS.indent(4) << "Function: " << getRegion().getEntry()->getParent()->getName()
               << "\n";
  OS.indent(4) << "Region: " << getNameStr() << "\n";
  OS.indent(4) << "Max Loop Depth:  " << getMaxLoopDepth() << "\n";
  OS.indent(4) << "Invariant Accesses: {\n";
  for (const auto &IAClass : InvariantEquivClasses) {
    const auto &MAs = std::get<1>(IAClass);
    if (MAs.empty()) {
      OS.indent(12) << "Class Pointer: " << *std::get<0>(IAClass) << "\n";
    } else {
      MAs.front()->print(OS);
      OS.indent(12) << "Execution Context: " << std::get<2>(IAClass) << "\n";
    }
  }
  OS.indent(4) << "}\n";
  printContext(OS.indent(4));
  printArrayInfo(OS.indent(4));
  printAliasAssumptions(OS);
  printStatements(OS.indent(4));
}

void Scop::dump() const { print(dbgs()); }

isl_ctx *Scop::getIslCtx() const { return IslCtx; }

__isl_give isl_pw_aff *Scop::getPwAff(const SCEV *E, BasicBlock *BB) {
  return Affinator.getPwAff(E, BB);
}

__isl_give isl_union_set *Scop::getDomains() const {
  isl_union_set *Domain = isl_union_set_empty(getParamSpace());

  for (const ScopStmt &Stmt : *this)
    Domain = isl_union_set_add_set(Domain, Stmt.getDomain());

  return Domain;
}

__isl_give isl_union_map *
Scop::getAccessesOfType(std::function<bool(MemoryAccess &)> Predicate) {
  isl_union_map *Accesses = isl_union_map_empty(getParamSpace());

  for (ScopStmt &Stmt : *this) {
    for (MemoryAccess *MA : Stmt) {
      if (!Predicate(*MA))
        continue;

      isl_set *Domain = Stmt.getDomain();
      isl_map *AccessDomain = MA->getAccessRelation();
      AccessDomain = isl_map_intersect_domain(AccessDomain, Domain);
      Accesses = isl_union_map_add_map(Accesses, AccessDomain);
    }
  }
  return isl_union_map_coalesce(Accesses);
}

__isl_give isl_union_map *Scop::getMustWrites() {
  return getAccessesOfType([](MemoryAccess &MA) { return MA.isMustWrite(); });
}

__isl_give isl_union_map *Scop::getMayWrites() {
  return getAccessesOfType([](MemoryAccess &MA) { return MA.isMayWrite(); });
}

__isl_give isl_union_map *Scop::getWrites() {
  return getAccessesOfType([](MemoryAccess &MA) { return MA.isWrite(); });
}

__isl_give isl_union_map *Scop::getReads() {
  return getAccessesOfType([](MemoryAccess &MA) { return MA.isRead(); });
}

__isl_give isl_union_map *Scop::getAccesses() {
  return getAccessesOfType([](MemoryAccess &MA) { return true; });
}

__isl_give isl_union_map *Scop::getSchedule() const {
  auto Tree = getScheduleTree();
  auto S = isl_schedule_get_map(Tree);
  isl_schedule_free(Tree);
  return S;
}

__isl_give isl_schedule *Scop::getScheduleTree() const {
  return isl_schedule_intersect_domain(isl_schedule_copy(Schedule),
                                       getDomains());
}

void Scop::setSchedule(__isl_take isl_union_map *NewSchedule) {
  auto *S = isl_schedule_from_domain(getDomains());
  S = isl_schedule_insert_partial_schedule(
      S, isl_multi_union_pw_aff_from_union_map(NewSchedule));
  isl_schedule_free(Schedule);
  Schedule = S;
}

void Scop::setScheduleTree(__isl_take isl_schedule *NewSchedule) {
  isl_schedule_free(Schedule);
  Schedule = NewSchedule;
}

bool Scop::restrictDomains(__isl_take isl_union_set *Domain) {
  bool Changed = false;
  for (ScopStmt &Stmt : *this) {
    isl_union_set *StmtDomain = isl_union_set_from_set(Stmt.getDomain());
    isl_union_set *NewStmtDomain = isl_union_set_intersect(
        isl_union_set_copy(StmtDomain), isl_union_set_copy(Domain));

    if (isl_union_set_is_subset(StmtDomain, NewStmtDomain)) {
      isl_union_set_free(StmtDomain);
      isl_union_set_free(NewStmtDomain);
      continue;
    }

    Changed = true;

    isl_union_set_free(StmtDomain);
    NewStmtDomain = isl_union_set_coalesce(NewStmtDomain);

    if (isl_union_set_is_empty(NewStmtDomain)) {
      Stmt.restrictDomain(isl_set_empty(Stmt.getDomainSpace()));
      isl_union_set_free(NewStmtDomain);
    } else
      Stmt.restrictDomain(isl_set_from_union_set(NewStmtDomain));
  }
  isl_union_set_free(Domain);
  return Changed;
}

ScalarEvolution *Scop::getSE() const { return SE; }

bool Scop::isIgnored(RegionNode *RN) {
  BasicBlock *BB = getRegionNodeBasicBlock(RN);
  ScopStmt *Stmt = getStmtForRegionNode(RN);

  // If there is no stmt, then it already has been removed.
  if (!Stmt)
    return true;

  // Check if there are accesses contained.
  if (Stmt->isEmpty())
    return true;

  // Check for reachability via non-error blocks.
  if (!DomainMap.count(BB))
    return true;

  // Check if error blocks are contained.
  if (containsErrorBlock(RN, getRegion(), LI, DT))
    return true;

  return false;
}

struct MapToDimensionDataTy {
  int N;
  isl_union_pw_multi_aff *Res;
};

// @brief Create a function that maps the elements of 'Set' to its N-th
//        dimension and add it to User->Res.
//
// @param Set        The input set.
// @param User->N    The dimension to map to.
// @param User->Res  The isl_union_pw_multi_aff to which to add the result.
//
// @returns   isl_stat_ok if no error occured, othewise isl_stat_error.
static isl_stat mapToDimension_AddSet(__isl_take isl_set *Set, void *User) {
  struct MapToDimensionDataTy *Data = (struct MapToDimensionDataTy *)User;
  int Dim;
  isl_space *Space;
  isl_pw_multi_aff *PMA;

  Dim = isl_set_dim(Set, isl_dim_set);
  Space = isl_set_get_space(Set);
  PMA = isl_pw_multi_aff_project_out_map(Space, isl_dim_set, Data->N,
                                         Dim - Data->N);
  if (Data->N > 1)
    PMA = isl_pw_multi_aff_drop_dims(PMA, isl_dim_out, 0, Data->N - 1);
  Data->Res = isl_union_pw_multi_aff_add_pw_multi_aff(Data->Res, PMA);

  isl_set_free(Set);

  return isl_stat_ok;
}

// @brief Create an isl_multi_union_aff that defines an identity mapping
//        from the elements of USet to their N-th dimension.
//
// # Example:
//
//            Domain: { A[i,j]; B[i,j,k] }
//                 N: 1
//
// Resulting Mapping: { {A[i,j] -> [(j)]; B[i,j,k] -> [(j)] }
//
// @param USet   A union set describing the elements for which to generate a
//               mapping.
// @param N      The dimension to map to.
// @returns      A mapping from USet to its N-th dimension.
static __isl_give isl_multi_union_pw_aff *
mapToDimension(__isl_take isl_union_set *USet, int N) {
  assert(N >= 0);
  assert(USet);
  assert(!isl_union_set_is_empty(USet));

  struct MapToDimensionDataTy Data;

  auto *Space = isl_union_set_get_space(USet);
  auto *PwAff = isl_union_pw_multi_aff_empty(Space);

  Data = {N, PwAff};

  auto Res = isl_union_set_foreach_set(USet, &mapToDimension_AddSet, &Data);

  (void)Res;

  assert(Res == isl_stat_ok);

  isl_union_set_free(USet);
  return isl_multi_union_pw_aff_from_union_pw_multi_aff(Data.Res);
}

void Scop::addScopStmt(BasicBlock *BB, Region *R) {
  if (BB) {
    Stmts.emplace_back(*this, *BB);
    auto Stmt = &Stmts.back();
    StmtMap[BB] = Stmt;
  } else {
    assert(R && "Either basic block or a region expected.");
    Stmts.emplace_back(*this, *R);
    auto Stmt = &Stmts.back();
    for (BasicBlock *BB : R->blocks())
      StmtMap[BB] = Stmt;
  }
}

void Scop::buildSchedule() {
  Loop *L = getLoopSurroundingRegion(getRegion(), LI);
  LoopStackTy LoopStack({LoopStackElementTy(L, nullptr, 0)});
  buildSchedule(getRegion().getNode(), LoopStack);
  assert(LoopStack.size() == 1 && LoopStack.back().L == L);
  Schedule = LoopStack[0].Schedule;
}

/// To generate a schedule for the elements in a Region we traverse the Region
/// in reverse-post-order and add the contained RegionNodes in traversal order
/// to the schedule of the loop that is currently at the top of the LoopStack.
/// For loop-free codes, this results in a correct sequential ordering.
///
/// Example:
///           bb1(0)
///         /     \.
///      bb2(1)   bb3(2)
///         \    /  \.
///          bb4(3)  bb5(4)
///             \   /
///              bb6(5)
///
/// Including loops requires additional processing. Whenever a loop header is
/// encountered, the corresponding loop is added to the @p LoopStack. Starting
/// from an empty schedule, we first process all RegionNodes that are within
/// this loop and complete the sequential schedule at this loop-level before
/// processing about any other nodes. To implement this
/// loop-nodes-first-processing, the reverse post-order traversal is
/// insufficient. Hence, we additionally check if the traversal yields
/// sub-regions or blocks that are outside the last loop on the @p LoopStack.
/// These region-nodes are then queue and only traverse after the all nodes
/// within the current loop have been processed.
void Scop::buildSchedule(Region *R, LoopStackTy &LoopStack) {
  Loop *OuterScopLoop = getLoopSurroundingRegion(getRegion(), LI);

  ReversePostOrderTraversal<Region *> RTraversal(R);
  std::deque<RegionNode *> WorkList(RTraversal.begin(), RTraversal.end());
  std::deque<RegionNode *> DelayList;
  bool LastRNWaiting = false;

  // Iterate over the region @p R in reverse post-order but queue
  // sub-regions/blocks iff they are not part of the last encountered but not
  // completely traversed loop. The variable LastRNWaiting is a flag to indicate
  // that we queued the last sub-region/block from the reverse post-order
  // iterator. If it is set we have to explore the next sub-region/block from
  // the iterator (if any) to guarantee progress. If it is not set we first try
  // the next queued sub-region/blocks.
  while (!WorkList.empty() || !DelayList.empty()) {
    RegionNode *RN;

    if ((LastRNWaiting && !WorkList.empty()) || DelayList.size() == 0) {
      RN = WorkList.front();
      WorkList.pop_front();
      LastRNWaiting = false;
    } else {
      RN = DelayList.front();
      DelayList.pop_front();
    }

    Loop *L = getRegionNodeLoop(RN, LI);
    if (!getRegion().contains(L))
      L = OuterScopLoop;

    Loop *LastLoop = LoopStack.back().L;
    if (LastLoop != L) {
      if (!LastLoop->contains(L)) {
        LastRNWaiting = true;
        DelayList.push_back(RN);
        continue;
      }
      LoopStack.push_back({L, nullptr, 0});
    }
    buildSchedule(RN, LoopStack);
  }

  return;
}

void Scop::buildSchedule(RegionNode *RN, LoopStackTy &LoopStack) {

  if (RN->isSubRegion()) {
    auto *LocalRegion = RN->getNodeAs<Region>();
    if (!SD.isNonAffineSubRegion(LocalRegion, &getRegion())) {
      buildSchedule(LocalRegion, LoopStack);
      return;
    }
  }

  auto &LoopData = LoopStack.back();
  LoopData.NumBlocksProcessed += getNumBlocksInRegionNode(RN);

  if (auto *Stmt = getStmtForRegionNode(RN)) {
    auto *UDomain = isl_union_set_from_set(Stmt->getDomain());
    auto *StmtSchedule = isl_schedule_from_domain(UDomain);
    LoopData.Schedule = combineInSequence(LoopData.Schedule, StmtSchedule);
  }

  // Check if we just processed the last node in this loop. If we did, finalize
  // the loop by:
  //
  //   - adding new schedule dimensions
  //   - folding the resulting schedule into the parent loop schedule
  //   - dropping the loop schedule from the LoopStack.
  //
  // Then continue to check surrounding loops, which might also have been
  // completed by this node.
  while (LoopData.L &&
         LoopData.NumBlocksProcessed == LoopData.L->getNumBlocks()) {
    auto Schedule = LoopData.Schedule;
    auto NumBlocksProcessed = LoopData.NumBlocksProcessed;

    LoopStack.pop_back();
    auto &NextLoopData = LoopStack.back();

    if (Schedule) {
      auto *Domain = isl_schedule_get_domain(Schedule);
      auto *MUPA = mapToDimension(Domain, LoopStack.size());
      Schedule = isl_schedule_insert_partial_schedule(Schedule, MUPA);
      NextLoopData.Schedule =
          combineInSequence(NextLoopData.Schedule, Schedule);
    }

    NextLoopData.NumBlocksProcessed += NumBlocksProcessed;
    LoopData = NextLoopData;
  }
}

ScopStmt *Scop::getStmtForBasicBlock(BasicBlock *BB) const {
  auto StmtMapIt = StmtMap.find(BB);
  if (StmtMapIt == StmtMap.end())
    return nullptr;
  return StmtMapIt->second;
}

ScopStmt *Scop::getStmtForRegionNode(RegionNode *RN) const {
  return getStmtForBasicBlock(getRegionNodeBasicBlock(RN));
}

int Scop::getRelativeLoopDepth(const Loop *L) const {
  Loop *OuterLoop =
      L ? R.outermostLoopInRegion(const_cast<Loop *>(L)) : nullptr;
  if (!OuterLoop)
    return -1;
  return L->getLoopDepth() - OuterLoop->getLoopDepth();
}

void ScopInfo::buildPHIAccesses(PHINode *PHI, Region &R,
                                Region *NonAffineSubRegion, bool IsExitBlock) {

  // PHI nodes that are in the exit block of the region, hence if IsExitBlock is
  // true, are not modeled as ordinary PHI nodes as they are not part of the
  // region. However, we model the operands in the predecessor blocks that are
  // part of the region as regular scalar accesses.

  // If we can synthesize a PHI we can skip it, however only if it is in
  // the region. If it is not it can only be in the exit block of the region.
  // In this case we model the operands but not the PHI itself.
  if (!IsExitBlock && canSynthesize(PHI, LI, SE, &R))
    return;

  // PHI nodes are modeled as if they had been demoted prior to the SCoP
  // detection. Hence, the PHI is a load of a new memory location in which the
  // incoming value was written at the end of the incoming basic block.
  bool OnlyNonAffineSubRegionOperands = true;
  for (unsigned u = 0; u < PHI->getNumIncomingValues(); u++) {
    Value *Op = PHI->getIncomingValue(u);
    BasicBlock *OpBB = PHI->getIncomingBlock(u);

    // Do not build scalar dependences inside a non-affine subregion.
    if (NonAffineSubRegion && NonAffineSubRegion->contains(OpBB))
      continue;

    OnlyNonAffineSubRegionOperands = false;

    if (!R.contains(OpBB))
      continue;

    Instruction *OpI = dyn_cast<Instruction>(Op);
    if (OpI) {
      BasicBlock *OpIBB = OpI->getParent();
      // As we pretend there is a use (or more precise a write) of OpI in OpBB
      // we have to insert a scalar dependence from the definition of OpI to
      // OpBB if the definition is not in OpBB.
      if (scop->getStmtForBasicBlock(OpIBB) !=
          scop->getStmtForBasicBlock(OpBB)) {
        ensureValueRead(OpI, OpBB);
        ensureValueWrite(OpI);
      }
    } else if (ModelReadOnlyScalars && !isa<Constant>(Op)) {
      ensureValueRead(Op, OpBB);
    }

    ensurePHIWrite(PHI, OpBB, Op, IsExitBlock);
  }

  if (!OnlyNonAffineSubRegionOperands && !IsExitBlock) {
    addPHIReadAccess(PHI);
  }
}

bool ScopInfo::buildScalarDependences(Instruction *Inst, Region *R,
                                      Region *NonAffineSubRegion) {
  bool canSynthesizeInst = canSynthesize(Inst, LI, SE, R);
  if (isIgnoredIntrinsic(Inst))
    return false;

  bool AnyCrossStmtUse = false;
  BasicBlock *ParentBB = Inst->getParent();

  for (User *U : Inst->users()) {
    Instruction *UI = dyn_cast<Instruction>(U);

    // Ignore the strange user
    if (UI == 0)
      continue;

    BasicBlock *UseParent = UI->getParent();

    // Ignore basic block local uses. A value that is defined in a scop, but
    // used in a PHI node in the same basic block does not count as basic block
    // local, as for such cases a control flow edge is passed between definition
    // and use.
    if (UseParent == ParentBB && !isa<PHINode>(UI))
      continue;

    // Uses by PHI nodes in the entry node count as external uses in case the
    // use is through an incoming block that is itself not contained in the
    // region.
    if (R->getEntry() == UseParent) {
      if (auto *PHI = dyn_cast<PHINode>(UI)) {
        bool ExternalUse = false;
        for (unsigned i = 0; i < PHI->getNumIncomingValues(); i++) {
          if (PHI->getIncomingValue(i) == Inst &&
              !R->contains(PHI->getIncomingBlock(i))) {
            ExternalUse = true;
            break;
          }
        }

        if (ExternalUse) {
          AnyCrossStmtUse = true;
          continue;
        }
      }
    }

    // Do not build scalar dependences inside a non-affine subregion.
    if (NonAffineSubRegion && NonAffineSubRegion->contains(UseParent))
      continue;

    // Check for PHI nodes in the region exit and skip them, if they will be
    // modeled as PHI nodes.
    //
    // PHI nodes in the region exit that have more than two incoming edges need
    // to be modeled as PHI-Nodes to correctly model the fact that depending on
    // the control flow a different value will be assigned to the PHI node. In
    // case this is the case, there is no need to create an additional normal
    // scalar dependence. Hence, bail out before we register an "out-of-region"
    // use for this definition.
    if (isa<PHINode>(UI) && UI->getParent() == R->getExit() &&
        !R->getExitingBlock())
      continue;

    // Check whether or not the use is in the SCoP.
    if (!R->contains(UseParent)) {
      AnyCrossStmtUse = true;
      continue;
    }

    // If the instruction can be synthesized and the user is in the region
    // we do not need to add scalar dependences.
    if (canSynthesizeInst)
      continue;

    // No need to translate these scalar dependences into polyhedral form,
    // because synthesizable scalars can be generated by the code generator.
    if (canSynthesize(UI, LI, SE, R))
      continue;

    // Skip PHI nodes in the region as they handle their operands on their own.
    if (isa<PHINode>(UI))
      continue;

    // Now U is used in another statement.
    AnyCrossStmtUse = true;

    // Do not build a read access that is not in the current SCoP
    // Use the def instruction as base address of the MemoryAccess, so that it
    // will become the name of the scalar access in the polyhedral form.
    ensureValueRead(Inst, UI->getParent());
  }

  if (ModelReadOnlyScalars && !isa<PHINode>(Inst)) {
    for (Value *Op : Inst->operands()) {
      if (canSynthesize(Op, LI, SE, R))
        continue;

      if (Instruction *OpInst = dyn_cast<Instruction>(Op))
        if (R->contains(OpInst))
          continue;

      if (isa<Constant>(Op))
        continue;

      ensureValueRead(Op, Inst->getParent());
    }
  }

  return AnyCrossStmtUse;
}

extern MapInsnToMemAcc InsnToMemAcc;

void ScopInfo::buildMemoryAccess(
    MemAccInst Inst, Loop *L, Region *R,
    const ScopDetection::BoxedLoopsSetTy *BoxedLoops,
    const InvariantLoadsSetTy &ScopRIL) {

  Value *Address = Inst.getPointerOperand();
  Value *Val = Inst.getValueOperand();
  Type *SizeType = Val->getType();
  unsigned Size = DL->getTypeAllocSize(SizeType);
  enum MemoryAccess::AccessType Type =
      Inst.isLoad() ? MemoryAccess::READ : MemoryAccess::MUST_WRITE;

  const SCEV *AccessFunction = SE->getSCEVAtScope(Address, L);
  const SCEVUnknown *BasePointer =
      dyn_cast<SCEVUnknown>(SE->getPointerBase(AccessFunction));

  assert(BasePointer && "Could not find base pointer");
  AccessFunction = SE->getMinusSCEV(AccessFunction, BasePointer);

  if (isa<GetElementPtrInst>(Address) || isa<BitCastInst>(Address)) {
    auto NewAddress = Address;
    if (auto *BitCast = dyn_cast<BitCastInst>(Address)) {
      auto Src = BitCast->getOperand(0);
      auto SrcTy = Src->getType();
      auto DstTy = BitCast->getType();
      if (SrcTy->getPrimitiveSizeInBits() == DstTy->getPrimitiveSizeInBits())
        NewAddress = Src;
    }

    if (auto *GEP = dyn_cast<GetElementPtrInst>(NewAddress)) {
      std::vector<const SCEV *> Subscripts;
      std::vector<int> Sizes;
      std::tie(Subscripts, Sizes) = getIndexExpressionsFromGEP(GEP, *SE);
      auto BasePtr = GEP->getOperand(0);

      std::vector<const SCEV *> SizesSCEV;

      bool AllAffineSubcripts = true;
      for (auto Subscript : Subscripts) {
        InvariantLoadsSetTy AccessILS;
        AllAffineSubcripts =
            isAffineExpr(R, Subscript, *SE, nullptr, &AccessILS);

        for (LoadInst *LInst : AccessILS)
          if (!ScopRIL.count(LInst))
            AllAffineSubcripts = false;

        if (!AllAffineSubcripts)
          break;
      }

      if (AllAffineSubcripts && Sizes.size() > 0) {
        for (auto V : Sizes)
          SizesSCEV.push_back(SE->getSCEV(ConstantInt::get(
              IntegerType::getInt64Ty(BasePtr->getContext()), V)));
        SizesSCEV.push_back(SE->getSCEV(ConstantInt::get(
            IntegerType::getInt64Ty(BasePtr->getContext()), Size)));

        addArrayAccess(Inst, Type, BasePointer->getValue(), Size, true,
                       Subscripts, SizesSCEV, Val);
        return;
      }
    }
  }

  auto AccItr = InsnToMemAcc.find(Inst);
  if (PollyDelinearize && AccItr != InsnToMemAcc.end()) {
    addArrayAccess(Inst, Type, BasePointer->getValue(), Size, true,
                   AccItr->second.DelinearizedSubscripts,
                   AccItr->second.Shape->DelinearizedSizes, Val);
    return;
  }

  // Check if the access depends on a loop contained in a non-affine subregion.
  bool isVariantInNonAffineLoop = false;
  if (BoxedLoops) {
    SetVector<const Loop *> Loops;
    findLoops(AccessFunction, Loops);
    for (const Loop *L : Loops)
      if (BoxedLoops->count(L))
        isVariantInNonAffineLoop = true;
  }

  InvariantLoadsSetTy AccessILS;
  bool IsAffine =
      !isVariantInNonAffineLoop &&
      isAffineExpr(R, AccessFunction, *SE, BasePointer->getValue(), &AccessILS);

  for (LoadInst *LInst : AccessILS)
    if (!ScopRIL.count(LInst))
      IsAffine = false;

  // FIXME: Size of the number of bytes of an array element, not the number of
  // elements as probably intended here.
  const SCEV *SizeSCEV =
      SE->getConstant(DL->getIntPtrType(Inst.getContext()), Size);

  if (!IsAffine && Type == MemoryAccess::MUST_WRITE)
    Type = MemoryAccess::MAY_WRITE;

  addArrayAccess(Inst, Type, BasePointer->getValue(), Size, IsAffine,
                 ArrayRef<const SCEV *>(AccessFunction),
                 ArrayRef<const SCEV *>(SizeSCEV), Val);
}

void ScopInfo::buildAccessFunctions(Region &R, Region &SR) {

  if (SD->isNonAffineSubRegion(&SR, &R)) {
    for (BasicBlock *BB : SR.blocks())
      buildAccessFunctions(R, *BB, &SR);
    return;
  }

  for (auto I = SR.element_begin(), E = SR.element_end(); I != E; ++I)
    if (I->isSubRegion())
      buildAccessFunctions(R, *I->getNodeAs<Region>());
    else
      buildAccessFunctions(R, *I->getNodeAs<BasicBlock>());
}

void ScopInfo::buildStmts(Region &R, Region &SR) {

  if (SD->isNonAffineSubRegion(&SR, &R)) {
    scop->addScopStmt(nullptr, &SR);
    return;
  }

  for (auto I = SR.element_begin(), E = SR.element_end(); I != E; ++I)
    if (I->isSubRegion())
      buildStmts(R, *I->getNodeAs<Region>());
    else
      scop->addScopStmt(I->getNodeAs<BasicBlock>(), nullptr);
}

void ScopInfo::buildAccessFunctions(Region &R, BasicBlock &BB,
                                    Region *NonAffineSubRegion,
                                    bool IsExitBlock) {
  // We do not build access functions for error blocks, as they may contain
  // instructions we can not model.
  if (isErrorBlock(BB, R, *LI, *DT) && !IsExitBlock)
    return;

  Loop *L = LI->getLoopFor(&BB);

  // The set of loops contained in non-affine subregions that are part of R.
  const ScopDetection::BoxedLoopsSetTy *BoxedLoops = SD->getBoxedLoops(&R);

  // The set of loads that are required to be invariant.
  auto &ScopRIL = *SD->getRequiredInvariantLoads(&R);

  for (BasicBlock::iterator I = BB.begin(), E = --BB.end(); I != E; ++I) {
    Instruction *Inst = &*I;

    PHINode *PHI = dyn_cast<PHINode>(Inst);
    if (PHI)
      buildPHIAccesses(PHI, R, NonAffineSubRegion, IsExitBlock);

    // For the exit block we stop modeling after the last PHI node.
    if (!PHI && IsExitBlock)
      break;

    // TODO: At this point we only know that elements of ScopRIL have to be
    //       invariant and will be hoisted for the SCoP to be processed. Though,
    //       there might be other invariant accesses that will be hoisted and
    //       that would allow to make a non-affine access affine.
    if (auto MemInst = MemAccInst::dyn_cast(Inst))
      buildMemoryAccess(MemInst, L, &R, BoxedLoops, ScopRIL);

    if (isIgnoredIntrinsic(Inst))
      continue;

    // Do not build scalar dependences for required invariant loads as we will
    // hoist them later on anyway or drop the SCoP if we cannot.
    if (ScopRIL.count(dyn_cast<LoadInst>(Inst)))
      continue;

    if (buildScalarDependences(Inst, &R, NonAffineSubRegion)) {
      if (!isa<StoreInst>(Inst))
        ensureValueWrite(Inst);
    }
  }
}

MemoryAccess *ScopInfo::addMemoryAccess(BasicBlock *BB, Instruction *Inst,
                                        MemoryAccess::AccessType Type,
                                        Value *BaseAddress, unsigned ElemBytes,
                                        bool Affine, Value *AccessValue,
                                        ArrayRef<const SCEV *> Subscripts,
                                        ArrayRef<const SCEV *> Sizes,
                                        ScopArrayInfo::MemoryKind Kind) {
  ScopStmt *Stmt = scop->getStmtForBasicBlock(BB);

  // Do not create a memory access for anything not in the SCoP. It would be
  // ignored anyway.
  if (!Stmt)
    return nullptr;

  AccFuncSetType &AccList = AccFuncMap[BB];
  Value *BaseAddr = BaseAddress;
  std::string BaseName = getIslCompatibleName("MemRef_", BaseAddr, "");

  bool isKnownMustAccess = false;

  // Accesses in single-basic block statements are always excuted.
  if (Stmt->isBlockStmt())
    isKnownMustAccess = true;

  if (Stmt->isRegionStmt()) {
    // Accesses that dominate the exit block of a non-affine region are always
    // executed. In non-affine regions there may exist MK_Values that do not
    // dominate the exit. MK_Values will always dominate the exit and MK_PHIs
    // only if there is at most one PHI_WRITE in the non-affine region.
    if (DT->dominates(BB, Stmt->getRegion()->getExit()))
      isKnownMustAccess = true;
  }

  // Non-affine PHI writes do not "happen" at a particular instruction, but
  // after exiting the statement. Therefore they are guaranteed execute and
  // overwrite the old value.
  if (Kind == ScopArrayInfo::MK_PHI || Kind == ScopArrayInfo::MK_ExitPHI)
    isKnownMustAccess = true;

  if (!isKnownMustAccess && Type == MemoryAccess::MUST_WRITE)
    Type = MemoryAccess::MAY_WRITE;

  AccList.emplace_back(Stmt, Inst, Type, BaseAddress, ElemBytes, Affine,
                       Subscripts, Sizes, AccessValue, Kind, BaseName);
  Stmt->addAccess(&AccList.back());
  return &AccList.back();
}

void ScopInfo::addArrayAccess(MemAccInst MemAccInst,
                              MemoryAccess::AccessType Type, Value *BaseAddress,
                              unsigned ElemBytes, bool IsAffine,
                              ArrayRef<const SCEV *> Subscripts,
                              ArrayRef<const SCEV *> Sizes,
                              Value *AccessValue) {
  assert(MemAccInst.isLoad() == (Type == MemoryAccess::READ));
  addMemoryAccess(MemAccInst.getParent(), MemAccInst, Type, BaseAddress,
                  ElemBytes, IsAffine, AccessValue, Subscripts, Sizes,
                  ScopArrayInfo::MK_Array);
}
void ScopInfo::ensureValueWrite(Instruction *Value) {
  ScopStmt *Stmt = scop->getStmtForBasicBlock(Value->getParent());

  // Value not defined within this SCoP.
  if (!Stmt)
    return;

  // Do not process further if the value is already written.
  if (Stmt->lookupValueWriteOf(Value))
    return;

  addMemoryAccess(Value->getParent(), Value, MemoryAccess::MUST_WRITE, Value, 1,
                  true, Value, ArrayRef<const SCEV *>(),
                  ArrayRef<const SCEV *>(), ScopArrayInfo::MK_Value);
}
void ScopInfo::ensureValueRead(Value *Value, BasicBlock *UserBB) {

  // If the instruction can be synthesized and the user is in the region we do
  // not need to add a value dependences.
  Region &ScopRegion = scop->getRegion();
  if (canSynthesize(Value, LI, SE, &ScopRegion))
    return;

  ScopStmt *UserStmt = scop->getStmtForBasicBlock(UserBB);

  // We do not model uses outside the scop.
  if (!UserStmt)
    return;

  // Do not create another MemoryAccess for reloading the value if one already
  // exists.
  if (UserStmt->lookupValueReadOf(Value))
    return;

  addMemoryAccess(UserBB, nullptr, MemoryAccess::READ, Value, 1, true, Value,
                  ArrayRef<const SCEV *>(), ArrayRef<const SCEV *>(),
                  ScopArrayInfo::MK_Value);
}
void ScopInfo::ensurePHIWrite(PHINode *PHI, BasicBlock *IncomingBlock,
                              Value *IncomingValue, bool IsExitBlock) {
  ScopStmt *IncomingStmt = scop->getStmtForBasicBlock(IncomingBlock);

  // Do not add more than one MemoryAccess per PHINode and ScopStmt.
  if (MemoryAccess *Acc = IncomingStmt->lookupPHIWriteOf(PHI)) {
    assert(Acc->getAccessInstruction() == PHI);
    Acc->addIncoming(IncomingBlock, IncomingValue);
    return;
  }

  MemoryAccess *Acc = addMemoryAccess(
      IncomingStmt->isBlockStmt() ? IncomingBlock
                                  : IncomingStmt->getRegion()->getEntry(),
      PHI, MemoryAccess::MUST_WRITE, PHI, 1, true, PHI,
      ArrayRef<const SCEV *>(), ArrayRef<const SCEV *>(),
      IsExitBlock ? ScopArrayInfo::MK_ExitPHI : ScopArrayInfo::MK_PHI);
  assert(Acc);
  Acc->addIncoming(IncomingBlock, IncomingValue);
}
void ScopInfo::addPHIReadAccess(PHINode *PHI) {
  addMemoryAccess(PHI->getParent(), PHI, MemoryAccess::READ, PHI, 1, true, PHI,
                  ArrayRef<const SCEV *>(), ArrayRef<const SCEV *>(),
                  ScopArrayInfo::MK_PHI);
}

void ScopInfo::buildScop(Region &R, AssumptionCache &AC) {
  unsigned MaxLoopDepth = getMaxLoopDepthInRegion(R, *LI, *SD);
  scop = new Scop(R, AccFuncMap, *SD, *SE, *DT, *LI, ctx, MaxLoopDepth);

  buildStmts(R, R);
  buildAccessFunctions(R, R);

  // In case the region does not have an exiting block we will later (during
  // code generation) split the exit block. This will move potential PHI nodes
  // from the current exit block into the new region exiting block. Hence, PHI
  // nodes that are at this point not part of the region will be.
  // To handle these PHI nodes later we will now model their operands as scalar
  // accesses. Note that we do not model anything in the exit block if we have
  // an exiting block in the region, as there will not be any splitting later.
  if (!R.getExitingBlock())
    buildAccessFunctions(R, *R.getExit(), nullptr, /* IsExitBlock */ true);

  scop->init(*AA, AC);
}

void ScopInfo::print(raw_ostream &OS, const Module *) const {
  if (!scop) {
    OS << "Invalid Scop!\n";
    return;
  }

  scop->print(OS);
}

void ScopInfo::clear() {
  AccFuncMap.clear();
  if (scop) {
    delete scop;
    scop = 0;
  }
}

//===----------------------------------------------------------------------===//
ScopInfo::ScopInfo() : RegionPass(ID), scop(0) {
  ctx = isl_ctx_alloc();
  isl_options_set_on_error(ctx, ISL_ON_ERROR_ABORT);
}

ScopInfo::~ScopInfo() {
  clear();
  isl_ctx_free(ctx);
}

void ScopInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<RegionInfoPass>();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequiredTransitive<ScalarEvolutionWrapperPass>();
  AU.addRequiredTransitive<ScopDetection>();
  AU.addRequired<AAResultsWrapperPass>();
  AU.addRequired<AssumptionCacheTracker>();
  AU.setPreservesAll();
}

bool ScopInfo::runOnRegion(Region *R, RGPassManager &RGM) {
  SD = &getAnalysis<ScopDetection>();

  if (!SD->isMaxRegionInScop(*R))
    return false;

  Function *F = R->getEntry()->getParent();
  SE = &getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  DL = &F->getParent()->getDataLayout();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(*F);

  DebugLoc Beg, End;
  getDebugLocations(R, Beg, End);
  std::string Msg = "SCoP begins here.";
  emitOptimizationRemarkAnalysis(F->getContext(), DEBUG_TYPE, *F, Beg, Msg);

  buildScop(*R, AC);

  DEBUG(scop->print(dbgs()));

  if (scop->isEmpty() || !scop->hasFeasibleRuntimeContext()) {
    Msg = "SCoP ends here but was dismissed.";
    delete scop;
    scop = nullptr;
  } else {
    Msg = "SCoP ends here.";
    ++ScopFound;
    if (scop->getMaxLoopDepth() > 0)
      ++RichScopFound;
  }

  emitOptimizationRemarkAnalysis(F->getContext(), DEBUG_TYPE, *F, End, Msg);

  return false;
}

char ScopInfo::ID = 0;

Pass *polly::createScopInfoPass() { return new ScopInfo(); }

INITIALIZE_PASS_BEGIN(ScopInfo, "polly-scops",
                      "Polly - Create polyhedral description of Scops", false,
                      false);
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass);
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker);
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass);
INITIALIZE_PASS_DEPENDENCY(RegionInfoPass);
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass);
INITIALIZE_PASS_DEPENDENCY(ScopDetection);
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass);
INITIALIZE_PASS_END(ScopInfo, "polly-scops",
                    "Polly - Create polyhedral description of Scops", false,
                    false)
