//===--------- ScopInfo.cpp ----------------------------------------------===//
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
#include "polly/ScopBuilder.h"
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
#include "llvm/Analysis/Loads.h"
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

STATISTIC(AssumptionsAliasing, "Number of aliasing assumptions taken.");
STATISTIC(AssumptionsInbounds, "Number of inbounds assumptions taken.");
STATISTIC(AssumptionsWrapping, "Number of wrapping assumptions taken.");
STATISTIC(AssumptionsUnsigned, "Number of unsigned assumptions taken.");
STATISTIC(AssumptionsComplexity, "Number of too complex SCoPs.");
STATISTIC(AssumptionsUnprofitable, "Number of unprofitable SCoPs.");
STATISTIC(AssumptionsErrorBlock, "Number of error block assumptions taken.");
STATISTIC(AssumptionsInfiniteLoop, "Number of bounded loop assumptions taken.");
STATISTIC(AssumptionsInvariantLoad,
          "Number of invariant loads assumptions taken.");
STATISTIC(AssumptionsDelinearization,
          "Number of delinearization assumptions taken.");

STATISTIC(NumLoopsInScop, "Number of loops in scops");
STATISTIC(NumScopsDepthOne, "Number of scops with maximal loop depth 1");
STATISTIC(NumScopsDepthTwo, "Number of scops with maximal loop depth 2");
STATISTIC(NumScopsDepthThree, "Number of scops with maximal loop depth 3");
STATISTIC(NumScopsDepthFour, "Number of scops with maximal loop depth 4");
STATISTIC(NumScopsDepthFive, "Number of scops with maximal loop depth 5");
STATISTIC(NumScopsDepthLarger,
          "Number of scops with maximal loop depth 6 and larger");
STATISTIC(MaxNumLoopsInScop, "Maximal number of loops in scops");

// The maximal number of basic sets we allow during domain construction to
// be created. More complex scops will result in very high compile time and
// are also unlikely to result in good code
static int const MaxDisjunctsInDomain = 20;

// The number of disjunct in the context after which we stop to add more
// disjuncts. This parameter is there to avoid exponential growth in the
// number of disjunct when adding non-convex sets to the context.
static int const MaxDisjunctsInContext = 4;

static cl::opt<bool> PollyRemarksMinimal(
    "polly-remarks-minimal",
    cl::desc("Do not emit remarks about assumptions that are known"),
    cl::Hidden, cl::ZeroOrMore, cl::init(false), cl::cat(PollyCategory));

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

static cl::opt<bool>
    IslOnErrorAbort("polly-on-isl-error-abort",
                    cl::desc("Abort if an isl error is encountered"),
                    cl::init(true), cl::cat(PollyCategory));

static cl::opt<bool> UnprofitableScalarAccs(
    "polly-unprofitable-scalar-accs",
    cl::desc("Count statements with scalar accesses as not optimizable"),
    cl::Hidden, cl::init(false), cl::cat(PollyCategory));

static cl::opt<bool> PollyPreciseInbounds(
    "polly-precise-inbounds",
    cl::desc("Take more precise inbounds assumptions (do not scale well)"),
    cl::Hidden, cl::init(false), cl::cat(PollyCategory));

static cl::opt<bool>
    PollyIgnoreInbounds("polly-ignore-inbounds",
                        cl::desc("Do not take inbounds assumptions at all"),
                        cl::Hidden, cl::init(false), cl::cat(PollyCategory));

static cl::opt<bool> PollyIgnoreParamBounds(
    "polly-ignore-parameter-bounds",
    cl::desc(
        "Do not add parameter bounds and do no gist simplify sets accordingly"),
    cl::Hidden, cl::init(false), cl::cat(PollyCategory));

static cl::opt<bool> PollyPreciseFoldAccesses(
    "polly-precise-fold-accesses",
    cl::desc("Fold memory accesses to model more possible delinearizations "
             "(does not scale well)"),
    cl::Hidden, cl::init(false), cl::cat(PollyCategory));
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
  isl_ctx *Ctx = isl_set_get_ctx(S);

  // The upper and lower bound for a parameter value is derived either from
  // the data type of the parameter or from the - possibly more restrictive -
  // range metadata.
  V = isl_valFromAPInt(Ctx, Range.getSignedMin(), true);
  S = isl_set_lower_bound_val(S, type, dim, V);
  V = isl_valFromAPInt(Ctx, Range.getSignedMax(), true);
  S = isl_set_upper_bound_val(S, type, dim, V);

  if (Range.isFullSet())
    return S;

  if (isl_set_n_basic_set(S) > MaxDisjunctsInContext)
    return S;

  // In case of signed wrapping, we can refine the set of valid values by
  // excluding the part not covered by the wrapping range.
  if (Range.isSignWrappedSet()) {
    V = isl_valFromAPInt(Ctx, Range.getLower(), true);
    isl_set *SLB = isl_set_lower_bound_val(isl_set_copy(S), type, dim, V);

    V = isl_valFromAPInt(Ctx, Range.getUpper(), true);
    V = isl_val_sub_ui(V, 1);
    isl_set *SUB = isl_set_upper_bound_val(S, type, dim, V);
    S = isl_set_union(SLB, SUB);
  }

  return S;
}

static const ScopArrayInfo *identifyBasePtrOriginSAI(Scop *S, Value *BasePtr) {
  LoadInst *BasePtrLI = dyn_cast<LoadInst>(BasePtr);
  if (!BasePtrLI)
    return nullptr;

  if (!S->contains(BasePtrLI))
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
                             MemoryKind::Array);
}

ScopArrayInfo::ScopArrayInfo(Value *BasePtr, Type *ElementType, isl_ctx *Ctx,
                             ArrayRef<const SCEV *> Sizes, MemoryKind Kind,
                             const DataLayout &DL, Scop *S,
                             const char *BaseName)
    : BasePtr(BasePtr), ElementType(ElementType), Kind(Kind), DL(DL), S(*S) {
  std::string BasePtrName =
      BaseName ? BaseName
               : getIslCompatibleName("MemRef_", BasePtr,
                                      Kind == MemoryKind::PHI ? "__phi" : "");
  Id = isl_id_alloc(Ctx, BasePtrName.c_str(), this);

  updateSizes(Sizes);

  if (!BasePtr || Kind != MemoryKind::Array) {
    BasePtrOriginSAI = nullptr;
    return;
  }

  BasePtrOriginSAI = identifyBasePtrOriginSAI(S, BasePtr);
  if (BasePtrOriginSAI)
    const_cast<ScopArrayInfo *>(BasePtrOriginSAI)->addDerivedSAI(this);
}

__isl_give isl_space *ScopArrayInfo::getSpace() const {
  auto *Space =
      isl_space_set_alloc(isl_id_get_ctx(Id), 0, getNumberOfDimensions());
  Space = isl_space_set_tuple_id(Space, isl_dim_set, isl_id_copy(Id));
  return Space;
}

bool ScopArrayInfo::isReadOnly() {
  isl_union_set *WriteSet = isl_union_map_range(S.getWrites());
  isl_space *Space = getSpace();
  WriteSet = isl_union_set_intersect(
      WriteSet, isl_union_set_from_set(isl_set_universe(Space)));

  bool IsReadOnly = isl_union_set_is_empty(WriteSet);
  isl_union_set_free(WriteSet);

  return IsReadOnly;
}

void ScopArrayInfo::updateElementType(Type *NewElementType) {
  if (NewElementType == ElementType)
    return;

  auto OldElementSize = DL.getTypeAllocSizeInBits(ElementType);
  auto NewElementSize = DL.getTypeAllocSizeInBits(NewElementType);

  if (NewElementSize == OldElementSize || NewElementSize == 0)
    return;

  if (NewElementSize % OldElementSize == 0 && NewElementSize < OldElementSize) {
    ElementType = NewElementType;
  } else {
    auto GCD = GreatestCommonDivisor64(NewElementSize, OldElementSize);
    ElementType = IntegerType::get(ElementType->getContext(), GCD);
  }
}

bool ScopArrayInfo::updateSizes(ArrayRef<const SCEV *> NewSizes,
                                bool CheckConsistency) {
  int SharedDims = std::min(NewSizes.size(), DimensionSizes.size());
  int ExtraDimsNew = NewSizes.size() - SharedDims;
  int ExtraDimsOld = DimensionSizes.size() - SharedDims;

  if (CheckConsistency) {
    for (int i = 0; i < SharedDims; i++) {
      auto *NewSize = NewSizes[i + ExtraDimsNew];
      auto *KnownSize = DimensionSizes[i + ExtraDimsOld];
      if (NewSize && KnownSize && NewSize != KnownSize)
        return false;
    }

    if (DimensionSizes.size() >= NewSizes.size())
      return true;
  }

  DimensionSizes.clear();
  DimensionSizes.insert(DimensionSizes.begin(), NewSizes.begin(),
                        NewSizes.end());
  for (isl_pw_aff *Size : DimensionSizesPw)
    isl_pw_aff_free(Size);
  DimensionSizesPw.clear();
  for (const SCEV *Expr : DimensionSizes) {
    if (!Expr) {
      DimensionSizesPw.push_back(nullptr);
      continue;
    }
    isl_pw_aff *Size = S.getPwAffOnly(Expr);
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

__isl_give isl_id *ScopArrayInfo::getBasePtrId() const {
  return isl_id_copy(Id);
}

void ScopArrayInfo::dump() const { print(errs()); }

void ScopArrayInfo::print(raw_ostream &OS, bool SizeAsPwAff) const {
  OS.indent(8) << *getElementType() << " " << getName();
  unsigned u = 0;
  if (getNumberOfDimensions() > 0 && !getDimensionSize(0)) {
    OS << "[*]";
    u++;
  }
  for (; u < getNumberOfDimensions(); u++) {
    OS << "[";

    if (SizeAsPwAff) {
      auto *Size = getDimensionSizePw(u);
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

const ScopArrayInfo *ScopArrayInfo::getFromId(__isl_take isl_id *Id) {
  void *User = isl_id_get_user(Id);
  const ScopArrayInfo *SAI = static_cast<ScopArrayInfo *>(User);
  isl_id_free(Id);
  return SAI;
}

void MemoryAccess::wrapConstantDimensions() {
  auto *SAI = getScopArrayInfo();
  auto *ArraySpace = SAI->getSpace();
  auto *Ctx = isl_space_get_ctx(ArraySpace);
  unsigned DimsArray = SAI->getNumberOfDimensions();

  auto *DivModAff = isl_multi_aff_identity(isl_space_map_from_domain_and_range(
      isl_space_copy(ArraySpace), isl_space_copy(ArraySpace)));
  auto *LArraySpace = isl_local_space_from_space(ArraySpace);

  // Begin with last dimension, to iteratively carry into higher dimensions.
  for (int i = DimsArray - 1; i > 0; i--) {
    auto *DimSize = SAI->getDimensionSize(i);
    auto *DimSizeCst = dyn_cast<SCEVConstant>(DimSize);

    // This transformation is not applicable to dimensions with dynamic size.
    if (!DimSizeCst)
      continue;

    // This transformation is not applicable to dimensions of size zero.
    if (DimSize->isZero())
      continue;

    auto *DimSizeVal = isl_valFromAPInt(Ctx, DimSizeCst->getAPInt(), false);
    auto *Var = isl_aff_var_on_domain(isl_local_space_copy(LArraySpace),
                                      isl_dim_set, i);
    auto *PrevVar = isl_aff_var_on_domain(isl_local_space_copy(LArraySpace),
                                          isl_dim_set, i - 1);

    // Compute: index % size
    // Modulo must apply in the divide of the previous iteration, if any.
    auto *Modulo = isl_aff_copy(Var);
    Modulo = isl_aff_mod_val(Modulo, isl_val_copy(DimSizeVal));
    Modulo = isl_aff_pullback_multi_aff(Modulo, isl_multi_aff_copy(DivModAff));

    // Compute: floor(index / size)
    auto *Divide = Var;
    Divide = isl_aff_div(
        Divide,
        isl_aff_val_on_domain(isl_local_space_copy(LArraySpace), DimSizeVal));
    Divide = isl_aff_floor(Divide);
    Divide = isl_aff_add(Divide, PrevVar);
    Divide = isl_aff_pullback_multi_aff(Divide, isl_multi_aff_copy(DivModAff));

    // Apply Modulo and Divide.
    DivModAff = isl_multi_aff_set_aff(DivModAff, i, Modulo);
    DivModAff = isl_multi_aff_set_aff(DivModAff, i - 1, Divide);
  }

  // Apply all modulo/divides on the accesses.
  AccessRelation =
      isl_map_apply_range(AccessRelation, isl_map_from_multi_aff(DivModAff));
  AccessRelation = isl_map_detect_equalities(AccessRelation);
  isl_local_space_free(LArraySpace);
}

void MemoryAccess::updateDimensionality() {
  auto *SAI = getScopArrayInfo();
  auto *ArraySpace = SAI->getSpace();
  auto *AccessSpace = isl_space_range(isl_map_get_space(AccessRelation));
  auto *Ctx = isl_space_get_ctx(AccessSpace);

  auto DimsArray = isl_space_dim(ArraySpace, isl_dim_set);
  auto DimsAccess = isl_space_dim(AccessSpace, isl_dim_set);
  auto DimsMissing = DimsArray - DimsAccess;

  auto *BB = getStatement()->getEntryBlock();
  auto &DL = BB->getModule()->getDataLayout();
  unsigned ArrayElemSize = SAI->getElemSizeInBytes();
  unsigned ElemBytes = DL.getTypeAllocSize(getElementType());

  auto *Map = isl_map_from_domain_and_range(
      isl_set_universe(AccessSpace),
      isl_set_universe(isl_space_copy(ArraySpace)));

  for (unsigned i = 0; i < DimsMissing; i++)
    Map = isl_map_fix_si(Map, isl_dim_out, i, 0);

  for (unsigned i = DimsMissing; i < DimsArray; i++)
    Map = isl_map_equate(Map, isl_dim_in, i - DimsMissing, isl_dim_out, i);

  AccessRelation = isl_map_apply_range(AccessRelation, Map);

  // For the non delinearized arrays, divide the access function of the last
  // subscript by the size of the elements in the array.
  //
  // A stride one array access in C expressed as A[i] is expressed in
  // LLVM-IR as something like A[i * elementsize]. This hides the fact that
  // two subsequent values of 'i' index two values that are stored next to
  // each other in memory. By this division we make this characteristic
  // obvious again. If the base pointer was accessed with offsets not divisible
  // by the accesses element size, we will have chosen a smaller ArrayElemSize
  // that divides the offsets of all accesses to this base pointer.
  if (DimsAccess == 1) {
    isl_val *V = isl_val_int_from_si(Ctx, ArrayElemSize);
    AccessRelation = isl_map_floordiv_val(AccessRelation, V);
  }

  // We currently do this only if we added at least one dimension, which means
  // some dimension's indices have not been specified, an indicator that some
  // index values have been added together.
  // TODO: Investigate general usefulness; Effect on unit tests is to make index
  // expressions more complicated.
  if (DimsMissing)
    wrapConstantDimensions();

  if (!isAffine())
    computeBoundsOnAccessRelation(ArrayElemSize);

  // Introduce multi-element accesses in case the type loaded by this memory
  // access is larger than the canonical element type of the array.
  //
  // An access ((float *)A)[i] to an array char *A is modeled as
  // {[i] -> A[o] : 4 i <= o <= 4 i + 3
  if (ElemBytes > ArrayElemSize) {
    assert(ElemBytes % ArrayElemSize == 0 &&
           "Loaded element size should be multiple of canonical element size");
    auto *Map = isl_map_from_domain_and_range(
        isl_set_universe(isl_space_copy(ArraySpace)),
        isl_set_universe(isl_space_copy(ArraySpace)));
    for (unsigned i = 0; i < DimsArray - 1; i++)
      Map = isl_map_equate(Map, isl_dim_in, i, isl_dim_out, i);

    isl_constraint *C;
    isl_local_space *LS;

    LS = isl_local_space_from_space(isl_map_get_space(Map));
    int Num = ElemBytes / getScopArrayInfo()->getElemSizeInBytes();

    C = isl_constraint_alloc_inequality(isl_local_space_copy(LS));
    C = isl_constraint_set_constant_val(C, isl_val_int_from_si(Ctx, Num - 1));
    C = isl_constraint_set_coefficient_si(C, isl_dim_in, DimsArray - 1, 1);
    C = isl_constraint_set_coefficient_si(C, isl_dim_out, DimsArray - 1, -1);
    Map = isl_map_add_constraint(Map, C);

    C = isl_constraint_alloc_inequality(LS);
    C = isl_constraint_set_coefficient_si(C, isl_dim_in, DimsArray - 1, -1);
    C = isl_constraint_set_coefficient_si(C, isl_dim_out, DimsArray - 1, 1);
    C = isl_constraint_set_constant_val(C, isl_val_int_from_si(Ctx, 0));
    Map = isl_map_add_constraint(Map, C);
    AccessRelation = isl_map_apply_range(AccessRelation, Map);
  }

  isl_space_free(ArraySpace);
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

/// Return the reduction type for a given binary operator.
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

MemoryAccess::~MemoryAccess() {
  isl_id_free(Id);
  isl_set_free(InvalidDomain);
  isl_map_free(AccessRelation);
  isl_map_free(NewAccessRelation);
}

const ScopArrayInfo *MemoryAccess::getOriginalScopArrayInfo() const {
  isl_id *ArrayId = getArrayId();
  void *User = isl_id_get_user(ArrayId);
  const ScopArrayInfo *SAI = static_cast<ScopArrayInfo *>(User);
  isl_id_free(ArrayId);
  return SAI;
}

const ScopArrayInfo *MemoryAccess::getLatestScopArrayInfo() const {
  isl_id *ArrayId = getLatestArrayId();
  void *User = isl_id_get_user(ArrayId);
  const ScopArrayInfo *SAI = static_cast<ScopArrayInfo *>(User);
  isl_id_free(ArrayId);
  return SAI;
}

__isl_give isl_id *MemoryAccess::getOriginalArrayId() const {
  return isl_map_get_tuple_id(AccessRelation, isl_dim_out);
}

__isl_give isl_id *MemoryAccess::getLatestArrayId() const {
  if (!hasNewAccessRelation())
    return getOriginalArrayId();
  return isl_map_get_tuple_id(NewAccessRelation, isl_dim_out);
}

__isl_give isl_map *MemoryAccess::getAddressFunction() const {
  return isl_map_lexmin(getAccessRelation());
}

__isl_give isl_pw_multi_aff *MemoryAccess::applyScheduleToAccessRelation(
    __isl_take isl_union_map *USchedule) const {
  isl_map *Schedule, *ScheduledAccRel;
  isl_union_set *UDomain;

  UDomain = isl_union_set_from_set(getStatement()->getDomain());
  USchedule = isl_union_map_intersect_domain(USchedule, UDomain);
  Schedule = isl_map_from_union_map(USchedule);
  ScheduledAccRel = isl_map_apply_domain(getAddressFunction(), Schedule);
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
  if (PollyIgnoreInbounds)
    return;
  auto *SAI = getScopArrayInfo();
  isl_space *Space = isl_space_range(getOriginalAccessRelationSpace());
  isl_set *Outside = isl_set_empty(isl_space_copy(Space));
  for (int i = 1, Size = isl_space_dim(Space, isl_dim_set); i < Size; ++i) {
    isl_local_space *LS = isl_local_space_from_space(isl_space_copy(Space));
    isl_pw_aff *Var =
        isl_pw_aff_var_on_domain(isl_local_space_copy(LS), isl_dim_set, i);
    isl_pw_aff *Zero = isl_pw_aff_zero_on_domain(LS);

    isl_set *DimOutside;

    DimOutside = isl_pw_aff_lt_set(isl_pw_aff_copy(Var), Zero);
    isl_pw_aff *SizeE = SAI->getDimensionSizePw(i);
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
  const auto &Loc = getAccessInstruction()
                        ? getAccessInstruction()->getDebugLoc()
                        : DebugLoc();
  if (!PollyPreciseInbounds)
    Outside = isl_set_gist(Outside, isl_set_params(Statement->getDomain()));
  Statement->getParent()->recordAssumption(INBOUNDS, Outside, Loc,
                                           AS_ASSUMPTION);
  isl_space_free(Space);
}

void MemoryAccess::buildMemIntrinsicAccessRelation() {
  assert(isMemoryIntrinsic());
  assert(Subscripts.size() == 2 && Sizes.size() == 1);

  auto *SubscriptPWA = getPwAff(Subscripts[0]);
  auto *SubscriptMap = isl_map_from_pw_aff(SubscriptPWA);

  isl_map *LengthMap;
  if (Subscripts[1] == nullptr) {
    LengthMap = isl_map_universe(isl_map_get_space(SubscriptMap));
  } else {
    auto *LengthPWA = getPwAff(Subscripts[1]);
    LengthMap = isl_map_from_pw_aff(LengthPWA);
    auto *RangeSpace = isl_space_range(isl_map_get_space(LengthMap));
    LengthMap = isl_map_apply_range(LengthMap, isl_map_lex_gt(RangeSpace));
  }
  LengthMap = isl_map_lower_bound_si(LengthMap, isl_dim_out, 0, 0);
  LengthMap = isl_map_align_params(LengthMap, isl_map_get_space(SubscriptMap));
  SubscriptMap =
      isl_map_align_params(SubscriptMap, isl_map_get_space(LengthMap));
  LengthMap = isl_map_sum(LengthMap, SubscriptMap);
  AccessRelation = isl_map_set_tuple_id(LengthMap, isl_dim_in,
                                        getStatement()->getDomainId());
}

void MemoryAccess::computeBoundsOnAccessRelation(unsigned ElementSize) {
  ScalarEvolution *SE = Statement->getParent()->getSE();

  auto MAI = MemAccInst(getAccessInstruction());
  if (isa<MemIntrinsic>(MAI))
    return;

  Value *Ptr = MAI.getPointerOperand();
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

  if (Range.isWrappedSet())
    return;

  bool isWrapping = Range.isSignWrappedSet();

  unsigned BW = Range.getBitWidth();
  const auto One = APInt(BW, 1);
  const auto LB = isWrapping ? Range.getLower() : Range.getSignedMin();
  const auto UB = isWrapping ? (Range.getUpper() - One) : Range.getSignedMax();

  auto Min = LB.sdiv(APInt(BW, ElementSize));
  auto Max = UB.sdiv(APInt(BW, ElementSize)) + One;

  assert(Min.sle(Max) && "Minimum expected to be less or equal than max");

  isl_set *AccessRange = isl_map_range(isl_map_copy(AccessRelation));
  AccessRange =
      addRangeBoundsToSet(AccessRange, ConstantRange(Min, Max), 0, isl_dim_set);
  AccessRelation = isl_map_intersect_range(AccessRelation, AccessRange);
}

void MemoryAccess::foldAccessRelation() {
  if (Sizes.size() < 2 || isa<SCEVConstant>(Sizes[1]))
    return;

  int Size = Subscripts.size();

  isl_map *OldAccessRelation = isl_map_copy(AccessRelation);

  for (int i = Size - 2; i >= 0; --i) {
    isl_space *Space;
    isl_map *MapOne, *MapTwo;
    isl_pw_aff *DimSize = getPwAff(Sizes[i + 1]);

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

  isl_id *BaseAddrId = getScopArrayInfo()->getBasePtrId();
  auto Space = Statement->getDomainSpace();
  AccessRelation = isl_map_set_tuple_id(
      AccessRelation, isl_dim_in, isl_space_get_tuple_id(Space, isl_dim_set));
  AccessRelation =
      isl_map_set_tuple_id(AccessRelation, isl_dim_out, BaseAddrId);
  AccessRelation = isl_map_gist_domain(AccessRelation, Statement->getDomain());

  // Access dimension folding might in certain cases increase the number of
  // disjuncts in the memory access, which can possibly complicate the generated
  // run-time checks and can lead to costly compilation.
  if (!PollyPreciseFoldAccesses && isl_map_n_basic_map(AccessRelation) >
                                       isl_map_n_basic_map(OldAccessRelation)) {
    isl_map_free(AccessRelation);
    AccessRelation = OldAccessRelation;
  } else {
    isl_map_free(OldAccessRelation);
  }

  isl_space_free(Space);
}

/// Check if @p Expr is divisible by @p Size.
static bool isDivisible(const SCEV *Expr, unsigned Size, ScalarEvolution &SE) {
  assert(Size != 0);
  if (Size == 1)
    return true;

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

  // Initialize the invalid domain which describes all iterations for which the
  // access relation is not modeled correctly.
  auto *StmtInvalidDomain = getStatement()->getInvalidDomain();
  InvalidDomain = isl_set_empty(isl_set_get_space(StmtInvalidDomain));
  isl_set_free(StmtInvalidDomain);

  isl_ctx *Ctx = isl_id_get_ctx(Id);
  isl_id *BaseAddrId = SAI->getBasePtrId();

  if (getAccessInstruction() && isa<MemIntrinsic>(getAccessInstruction())) {
    buildMemIntrinsicAccessRelation();
    AccessRelation =
        isl_map_set_tuple_id(AccessRelation, isl_dim_out, BaseAddrId);
    return;
  }

  if (!isAffine()) {
    // We overapproximate non-affine accesses with a possible access to the
    // whole array. For read accesses it does not make a difference, if an
    // access must or may happen. However, for write accesses it is important to
    // differentiate between writes that must happen and writes that may happen.
    if (!AccessRelation)
      AccessRelation = isl_map_from_basic_map(createBasicAccessMap(Statement));

    AccessRelation =
        isl_map_set_tuple_id(AccessRelation, isl_dim_out, BaseAddrId);
    return;
  }

  isl_space *Space = isl_space_alloc(Ctx, 0, Statement->getNumIterators(), 0);
  AccessRelation = isl_map_universe(Space);

  for (int i = 0, Size = Subscripts.size(); i < Size; ++i) {
    isl_pw_aff *Affine = getPwAff(Subscripts[i]);
    isl_map *SubscriptMap = isl_map_from_pw_aff(Affine);
    AccessRelation = isl_map_flat_range_product(AccessRelation, SubscriptMap);
  }

  Space = Statement->getDomainSpace();
  AccessRelation = isl_map_set_tuple_id(
      AccessRelation, isl_dim_in, isl_space_get_tuple_id(Space, isl_dim_set));
  AccessRelation =
      isl_map_set_tuple_id(AccessRelation, isl_dim_out, BaseAddrId);

  AccessRelation = isl_map_gist_domain(AccessRelation, Statement->getDomain());
  isl_space_free(Space);
}

MemoryAccess::MemoryAccess(ScopStmt *Stmt, Instruction *AccessInst,
                           AccessType AccType, Value *BaseAddress,
                           Type *ElementType, bool Affine,
                           ArrayRef<const SCEV *> Subscripts,
                           ArrayRef<const SCEV *> Sizes, Value *AccessValue,
                           MemoryKind Kind, StringRef BaseName)
    : Kind(Kind), AccType(AccType), RedType(RT_NONE), Statement(Stmt),
      InvalidDomain(nullptr), BaseAddr(BaseAddress), BaseName(BaseName),
      ElementType(ElementType), Sizes(Sizes.begin(), Sizes.end()),
      AccessInstruction(AccessInst), AccessValue(AccessValue), IsAffine(Affine),
      Subscripts(Subscripts.begin(), Subscripts.end()), AccessRelation(nullptr),
      NewAccessRelation(nullptr) {
  static const std::string TypeStrings[] = {"", "_Read", "_Write", "_MayWrite"};
  const std::string Access = TypeStrings[AccType] + utostr(Stmt->size()) + "_";

  std::string IdName =
      getIslCompatibleName(Stmt->getBaseName(), Access, BaseName);
  Id = isl_id_alloc(Stmt->getParent()->getIslCtx(), IdName.c_str(), this);
}

MemoryAccess::MemoryAccess(ScopStmt *Stmt, AccessType AccType,
                           __isl_take isl_map *AccRel)
    : Kind(MemoryKind::Array), AccType(AccType), RedType(RT_NONE),
      Statement(Stmt), InvalidDomain(nullptr), AccessInstruction(nullptr),
      IsAffine(true), AccessRelation(nullptr), NewAccessRelation(AccRel) {
  auto *ArrayInfoId = isl_map_get_tuple_id(NewAccessRelation, isl_dim_out);
  auto *SAI = ScopArrayInfo::getFromId(ArrayInfoId);
  Sizes.push_back(nullptr);
  for (unsigned i = 1; i < SAI->getNumberOfDimensions(); i++)
    Sizes.push_back(SAI->getDimensionSize(i));
  ElementType = SAI->getElementType();
  BaseAddr = SAI->getBasePtr();
  BaseName = SAI->getName();
  static const std::string TypeStrings[] = {"", "_Read", "_Write", "_MayWrite"};
  const std::string Access = TypeStrings[AccType] + utostr(Stmt->size()) + "_";

  std::string IdName =
      getIslCompatibleName(Stmt->getBaseName(), Access, BaseName);
  Id = isl_id_alloc(Stmt->getParent()->getIslCtx(), IdName.c_str(), this);
}

void MemoryAccess::realignParams() {
  auto *Ctx = Statement->getParent()->getContext();
  InvalidDomain = isl_set_gist_params(InvalidDomain, isl_set_copy(Ctx));
  AccessRelation = isl_map_gist_params(AccessRelation, Ctx);
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

__isl_give isl_pw_aff *MemoryAccess::getPwAff(const SCEV *E) {
  auto *Stmt = getStatement();
  PWACtx PWAC = Stmt->getParent()->getPwAff(E, Stmt->getEntryBlock());
  isl_set *StmtDom = isl_set_reset_tuple_id(getStatement()->getDomain());
  isl_set *NewInvalidDom = isl_set_intersect(StmtDom, PWAC.second);
  InvalidDomain = isl_set_union(InvalidDomain, NewInvalidDom);
  return PWAC.first;
}

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
static isl_map *getEqualAndLarger(__isl_take isl_space *setDomain) {
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

bool MemoryAccess::isStrideZero(__isl_take const isl_map *Schedule) const {
  return isStrideX(Schedule, 0);
}

bool MemoryAccess::isStrideOne(__isl_take const isl_map *Schedule) const {
  return isStrideX(Schedule, 1);
}

void MemoryAccess::setAccessRelation(__isl_take isl_map *NewAccess) {
  isl_map_free(AccessRelation);
  AccessRelation = NewAccess;
}

void MemoryAccess::setNewAccessRelation(__isl_take isl_map *NewAccess) {
  assert(NewAccess);

#ifndef NDEBUG
  // Check domain space compatibility.
  auto *NewSpace = isl_map_get_space(NewAccess);
  auto *NewDomainSpace = isl_space_domain(isl_space_copy(NewSpace));
  auto *OriginalDomainSpace = getStatement()->getDomainSpace();
  assert(isl_space_has_equal_tuples(OriginalDomainSpace, NewDomainSpace));
  isl_space_free(NewDomainSpace);
  isl_space_free(OriginalDomainSpace);

  // Check whether there is an access for every statement instance.
  auto *StmtDomain = getStatement()->getDomain();
  StmtDomain = isl_set_intersect_params(
      StmtDomain, getStatement()->getParent()->getContext());
  auto *NewDomain = isl_map_domain(isl_map_copy(NewAccess));
  assert(isl_set_is_subset(StmtDomain, NewDomain) &&
         "Partial accesses not supported");
  isl_set_free(NewDomain);
  isl_set_free(StmtDomain);

  auto *NewAccessSpace = isl_space_range(NewSpace);
  assert(isl_space_has_tuple_id(NewAccessSpace, isl_dim_set) &&
         "Must specify the array that is accessed");
  auto *NewArrayId = isl_space_get_tuple_id(NewAccessSpace, isl_dim_set);
  auto *SAI = static_cast<ScopArrayInfo *>(isl_id_get_user(NewArrayId));
  assert(SAI && "Must set a ScopArrayInfo");

  if (SAI->isArrayKind() && SAI->getBasePtrOriginSAI()) {
    InvariantEquivClassTy *EqClass =
        getStatement()->getParent()->lookupInvariantEquivClass(
            SAI->getBasePtr());
    assert(EqClass &&
           "Access functions to indirect arrays must have an invariant and "
           "hoisted base pointer");
  }

  // Check whether access dimensions correspond to number of dimensions of the
  // accesses array.
  auto Dims = SAI->getNumberOfDimensions();
  assert(isl_space_dim(NewAccessSpace, isl_dim_set) == Dims &&
         "Access dims must match array dims");
  isl_space_free(NewAccessSpace);
  isl_id_free(NewArrayId);
#endif

  isl_map_free(NewAccessRelation);
  NewAccessRelation = NewAccess;
}

//===----------------------------------------------------------------------===//

__isl_give isl_map *ScopStmt::getSchedule() const {
  isl_set *Domain = getDomain();
  if (isl_set_is_empty(Domain)) {
    isl_set_free(Domain);
    return isl_map_from_aff(
        isl_aff_zero_on_domain(isl_local_space_from_space(getDomainSpace())));
  }
  auto *Schedule = getParent()->getSchedule();
  if (!Schedule) {
    isl_set_free(Domain);
    return nullptr;
  }
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

__isl_give isl_pw_aff *ScopStmt::getPwAff(const SCEV *E, bool NonNegative) {
  PWACtx PWAC = getParent()->getPwAff(E, getEntryBlock(), NonNegative);
  InvalidDomain = isl_set_union(InvalidDomain, PWAC.second);
  return PWAC.first;
}

void ScopStmt::restrictDomain(__isl_take isl_set *NewDomain) {
  assert(isl_set_is_subset(NewDomain, Domain) &&
         "New domain is not a subset of old domain!");
  isl_set_free(Domain);
  Domain = NewDomain;
}

void ScopStmt::buildAccessRelations() {
  Scop &S = *getParent();
  for (MemoryAccess *Access : MemAccs) {
    Type *ElementType = Access->getElementType();

    MemoryKind Ty;
    if (Access->isPHIKind())
      Ty = MemoryKind::PHI;
    else if (Access->isExitPHIKind())
      Ty = MemoryKind::ExitPHI;
    else if (Access->isValueKind())
      Ty = MemoryKind::Value;
    else
      Ty = MemoryKind::Array;

    auto *SAI = S.getOrCreateScopArrayInfo(Access->getOriginalBaseAddr(),
                                           ElementType, Access->Sizes, Ty);
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
    assert(Parent.getStmtFor(AccessVal) == this);
    assert(!ValueWrites.lookup(AccessVal));

    ValueWrites[AccessVal] = Access;
  } else if (Access->isValueKind() && Access->isRead()) {
    Value *AccessVal = Access->getAccessValue();
    assert(!ValueReads.lookup(AccessVal));

    ValueReads[AccessVal] = Access;
  } else if (Access->isAnyPHIKind() && Access->isWrite()) {
    PHINode *PHI = cast<PHINode>(Access->getAccessValue());
    assert(!PHIWrites.lookup(PHI));

    PHIWrites[PHI] = Access;
  }

  MemAccs.push_back(Access);
}

void ScopStmt::realignParams() {
  for (MemoryAccess *MA : *this)
    MA->realignParams();

  auto *Ctx = Parent.getContext();
  InvalidDomain = isl_set_gist_params(InvalidDomain, isl_set_copy(Ctx));
  Domain = isl_set_gist_params(Domain, Ctx);
}

/// Add @p BSet to the set @p User if @p BSet is bounded.
static isl_stat collectBoundedParts(__isl_take isl_basic_set *BSet,
                                    void *User) {
  isl_set **BoundedParts = static_cast<isl_set **>(User);
  if (isl_basic_set_is_bounded(BSet))
    *BoundedParts = isl_set_union(*BoundedParts, isl_set_from_basic_set(BSet));
  else
    isl_basic_set_free(BSet);
  return isl_stat_ok;
}

/// Return the bounded parts of @p S.
static __isl_give isl_set *collectBoundedParts(__isl_take isl_set *S) {
  isl_set *BoundedParts = isl_set_empty(isl_set_get_space(S));
  isl_set_foreach_basic_set(S, collectBoundedParts, &BoundedParts);
  isl_set_free(S);
  return BoundedParts;
}

/// Compute the (un)bounded parts of @p S wrt. to dimension @p Dim.
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

/// Set the dimension Ids from @p From in @p To.
static __isl_give isl_set *setDimensionIds(__isl_keep isl_set *From,
                                           __isl_take isl_set *To) {
  for (unsigned u = 0, e = isl_set_n_dim(From); u < e; u++) {
    isl_id *DimId = isl_set_get_dim_id(From, isl_dim_set, u);
    To = isl_set_set_dim_id(To, isl_dim_set, u, DimId);
  }
  return To;
}

/// Create the conditions under which @p L @p Pred @p R is true.
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

/// Create the conditions under which @p L @p Pred @p R is true.
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

/// Build the conditions sets for the switch @p SI in the @p Domain.
///
/// This will fill @p ConditionSets with the conditions under which control
/// will be moved from @p SI to its successors. Hence, @p ConditionSets will
/// have as many elements as @p SI has successors.
static bool
buildConditionSets(ScopStmt &Stmt, SwitchInst *SI, Loop *L,
                   __isl_keep isl_set *Domain,
                   SmallVectorImpl<__isl_give isl_set *> &ConditionSets) {

  Value *Condition = getConditionFromTerminator(SI);
  assert(Condition && "No condition for switch");

  Scop &S = *Stmt.getParent();
  ScalarEvolution &SE = *S.getSE();
  isl_pw_aff *LHS, *RHS;
  LHS = Stmt.getPwAff(SE.getSCEVAtScope(Condition, L));

  unsigned NumSuccessors = SI->getNumSuccessors();
  ConditionSets.resize(NumSuccessors);
  for (auto &Case : SI->cases()) {
    unsigned Idx = Case.getSuccessorIndex();
    ConstantInt *CaseValue = Case.getCaseValue();

    RHS = Stmt.getPwAff(SE.getSCEV(CaseValue));
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

  isl_pw_aff_free(LHS);

  return true;
}

/// Build the conditions sets for the branch condition @p Condition in
/// the @p Domain.
///
/// This will fill @p ConditionSets with the conditions under which control
/// will be moved from @p TI to its successors. Hence, @p ConditionSets will
/// have as many elements as @p TI has successors. If @p TI is nullptr the
/// context under which @p Condition is true/false will be returned as the
/// new elements of @p ConditionSets.
static bool
buildConditionSets(ScopStmt &Stmt, Value *Condition, TerminatorInst *TI,
                   Loop *L, __isl_keep isl_set *Domain,
                   SmallVectorImpl<__isl_give isl_set *> &ConditionSets) {

  Scop &S = *Stmt.getParent();
  isl_set *ConsequenceCondSet = nullptr;
  if (auto *CCond = dyn_cast<ConstantInt>(Condition)) {
    if (CCond->isZero())
      ConsequenceCondSet = isl_set_empty(isl_set_get_space(Domain));
    else
      ConsequenceCondSet = isl_set_universe(isl_set_get_space(Domain));
  } else if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(Condition)) {
    auto Opcode = BinOp->getOpcode();
    assert(Opcode == Instruction::And || Opcode == Instruction::Or);

    bool Valid = buildConditionSets(Stmt, BinOp->getOperand(0), TI, L, Domain,
                                    ConditionSets) &&
                 buildConditionSets(Stmt, BinOp->getOperand(1), TI, L, Domain,
                                    ConditionSets);
    if (!Valid) {
      while (!ConditionSets.empty())
        isl_set_free(ConditionSets.pop_back_val());
      return false;
    }

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
    isl_pw_aff *LHS, *RHS;
    // For unsigned comparisons we assumed the signed bit of neither operand
    // to be set. The comparison is equal to a signed comparison under this
    // assumption.
    bool NonNeg = ICond->isUnsigned();
    LHS = Stmt.getPwAff(SE.getSCEVAtScope(ICond->getOperand(0), L), NonNeg);
    RHS = Stmt.getPwAff(SE.getSCEVAtScope(ICond->getOperand(1), L), NonNeg);
    ConsequenceCondSet =
        buildConditionSet(ICond->getPredicate(), LHS, RHS, Domain);
  }

  // If no terminator was given we are only looking for parameter constraints
  // under which @p Condition is true/false.
  if (!TI)
    ConsequenceCondSet = isl_set_params(ConsequenceCondSet);
  assert(ConsequenceCondSet);
  ConsequenceCondSet = isl_set_coalesce(
      isl_set_intersect(ConsequenceCondSet, isl_set_copy(Domain)));

  isl_set *AlternativeCondSet = nullptr;
  bool TooComplex =
      isl_set_n_basic_set(ConsequenceCondSet) >= MaxDisjunctsInDomain;

  if (!TooComplex) {
    AlternativeCondSet = isl_set_subtract(isl_set_copy(Domain),
                                          isl_set_copy(ConsequenceCondSet));
    TooComplex =
        isl_set_n_basic_set(AlternativeCondSet) >= MaxDisjunctsInDomain;
  }

  if (TooComplex) {
    S.invalidate(COMPLEXITY, TI ? TI->getDebugLoc() : DebugLoc());
    isl_set_free(AlternativeCondSet);
    isl_set_free(ConsequenceCondSet);
    return false;
  }

  ConditionSets.push_back(ConsequenceCondSet);
  ConditionSets.push_back(isl_set_coalesce(AlternativeCondSet));

  return true;
}

/// Build the conditions sets for the terminator @p TI in the @p Domain.
///
/// This will fill @p ConditionSets with the conditions under which control
/// will be moved from @p TI to its successors. Hence, @p ConditionSets will
/// have as many elements as @p TI has successors.
static bool
buildConditionSets(ScopStmt &Stmt, TerminatorInst *TI, Loop *L,
                   __isl_keep isl_set *Domain,
                   SmallVectorImpl<__isl_give isl_set *> &ConditionSets) {

  if (SwitchInst *SI = dyn_cast<SwitchInst>(TI))
    return buildConditionSets(Stmt, SI, L, Domain, ConditionSets);

  assert(isa<BranchInst>(TI) && "Terminator was neither branch nor switch.");

  if (TI->getNumSuccessors() == 1) {
    ConditionSets.push_back(isl_set_copy(Domain));
    return true;
  }

  Value *Condition = getConditionFromTerminator(TI);
  assert(Condition && "No condition for Terminator");

  return buildConditionSets(Stmt, Condition, TI, L, Domain, ConditionSets);
}

void ScopStmt::buildDomain() {
  isl_id *Id = isl_id_alloc(getIslCtx(), getBaseName(), this);

  Domain = getParent()->getDomainConditions(this);
  Domain = isl_set_set_tuple_id(Domain, Id);
}

void ScopStmt::collectSurroundingLoops() {
  for (unsigned u = 0, e = isl_set_n_dim(Domain); u < e; u++) {
    isl_id *DimId = isl_set_get_dim_id(Domain, isl_dim_set, u);
    NestLoops.push_back(static_cast<Loop *>(isl_id_get_user(DimId)));
    isl_id_free(DimId);
  }
}

ScopStmt::ScopStmt(Scop &parent, Region &R, Loop *SurroundingLoop)
    : Parent(parent), InvalidDomain(nullptr), Domain(nullptr), BB(nullptr),
      R(&R), Build(nullptr), SurroundingLoop(SurroundingLoop) {

  BaseName = getIslCompatibleName("Stmt_", R.getNameStr(), "");
}

ScopStmt::ScopStmt(Scop &parent, BasicBlock &bb, Loop *SurroundingLoop)
    : Parent(parent), InvalidDomain(nullptr), Domain(nullptr), BB(&bb),
      R(nullptr), Build(nullptr), SurroundingLoop(SurroundingLoop) {

  BaseName = getIslCompatibleName("Stmt_", &bb, "");
}

ScopStmt::ScopStmt(Scop &parent, __isl_take isl_map *SourceRel,
                   __isl_take isl_map *TargetRel, __isl_take isl_set *NewDomain)
    : Parent(parent), InvalidDomain(nullptr), Domain(NewDomain), BB(nullptr),
      R(nullptr), Build(nullptr) {
  BaseName = getIslCompatibleName("CopyStmt_", "",
                                  std::to_string(parent.getCopyStmtsNum()));
  auto *Id = isl_id_alloc(getIslCtx(), getBaseName(), this);
  Domain = isl_set_set_tuple_id(Domain, isl_id_copy(Id));
  TargetRel = isl_map_set_tuple_id(TargetRel, isl_dim_in, Id);
  auto *Access =
      new MemoryAccess(this, MemoryAccess::AccessType::MUST_WRITE, TargetRel);
  parent.addAccessFunction(Access);
  addAccess(Access);
  SourceRel = isl_map_set_tuple_id(SourceRel, isl_dim_in, isl_id_copy(Id));
  Access = new MemoryAccess(this, MemoryAccess::AccessType::READ, SourceRel);
  parent.addAccessFunction(Access);
  addAccess(Access);
}

void ScopStmt::init(LoopInfo &LI) {
  assert(!Domain && "init must be called only once");

  buildDomain();
  collectSurroundingLoops();
  buildAccessRelations();

  if (DetectReductions)
    checkForReductions();
}

/// Collect loads which might form a reduction chain with @p StoreMA.
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

/// Check for reductions in this ScopStmt.
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

      if (isl_set_has_equal_space(AllAccs, Accs)) {
        isl_set *OverlapAccs = isl_set_intersect(Accs, isl_set_copy(AllAccs));
        Valid = Valid && isl_set_is_empty(OverlapAccs);
        isl_set_free(OverlapAccs);
      } else {
        isl_set_free(Accs);
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
  if (!S)
    return "";
  auto Str = stringFromIslObj(S);
  isl_map_free(S);
  return Str;
}

void ScopStmt::setInvalidDomain(__isl_take isl_set *ID) {
  isl_set_free(InvalidDomain);
  InvalidDomain = ID;
}

BasicBlock *ScopStmt::getEntryBlock() const {
  if (isBlockStmt())
    return getBasicBlock();
  return getRegion()->getEntry();
}

unsigned ScopStmt::getNumIterators() const { return NestLoops.size(); }

const char *ScopStmt::getBaseName() const { return BaseName.c_str(); }

Loop *ScopStmt::getLoopForDimension(unsigned Dimension) const {
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

ScopStmt::~ScopStmt() {
  isl_set_free(Domain);
  isl_set_free(InvalidDomain);
}

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

void ScopStmt::removeMemoryAccess(MemoryAccess *MA) {
  // Remove the memory accesses from this statement together with all scalar
  // accesses that were caused by it. MemoryKind::Value READs have no access
  // instruction, hence would not be removed by this function. However, it is
  // only used for invariant LoadInst accesses, its arguments are always affine,
  // hence synthesizable, and therefore there are no MemoryKind::Value READ
  // accesses to be removed.
  auto Predicate = [&](MemoryAccess *Acc) {
    return Acc->getAccessInstruction() == MA->getAccessInstruction();
  };
  MemAccs.erase(std::remove_if(MemAccs.begin(), MemAccs.end(), Predicate),
                MemAccs.end());
  InstructionToAccess.erase(MA->getAccessInstruction());
}

void ScopStmt::removeSingleMemoryAccess(MemoryAccess *MA) {
  auto MAIt = std::find(MemAccs.begin(), MemAccs.end(), MA);
  assert(MAIt != MemAccs.end());
  MemAccs.erase(MAIt);

  auto It = InstructionToAccess.find(MA->getAccessInstruction());
  if (It != InstructionToAccess.end()) {
    It->second.remove(MA);
    if (It->second.empty())
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

/// Remap parameter values but keep AddRecs valid wrt. invariant loads.
struct SCEVSensitiveParameterRewriter
    : public SCEVRewriteVisitor<SCEVSensitiveParameterRewriter> {
  ValueToValueMap &VMap;

public:
  SCEVSensitiveParameterRewriter(ValueToValueMap &VMap, ScalarEvolution &SE)
      : SCEVRewriteVisitor(SE), VMap(VMap) {}

  static const SCEV *rewrite(const SCEV *E, ScalarEvolution &SE,
                             ValueToValueMap &VMap) {
    SCEVSensitiveParameterRewriter SSPR(VMap, SE);
    return SSPR.visit(E);
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

void Scop::createParameterId(const SCEV *Parameter) {
  assert(Parameters.count(Parameter));
  assert(!ParameterIds.count(Parameter));

  std::string ParameterName = "p_" + std::to_string(getNumParams() - 1);

  if (const SCEVUnknown *ValueParameter = dyn_cast<SCEVUnknown>(Parameter)) {
    Value *Val = ValueParameter->getValue();

    // If this parameter references a specific Value and this value has a name
    // we use this name as it is likely to be unique and more useful than just
    // a number.
    if (Val->hasName())
      ParameterName = Val->getName();
    else if (LoadInst *LI = dyn_cast<LoadInst>(Val)) {
      auto *LoadOrigin = LI->getPointerOperand()->stripInBoundsOffsets();
      if (LoadOrigin->hasName()) {
        ParameterName += "_loaded_from_";
        ParameterName +=
            LI->getPointerOperand()->stripInBoundsOffsets()->getName();
      }
    }
  }

  ParameterName = getIslCompatibleName("", ParameterName, "");

  auto *Id = isl_id_alloc(getIslCtx(), ParameterName.c_str(),
                          const_cast<void *>((const void *)Parameter));
  ParameterIds[Parameter] = Id;
}

void Scop::addParams(const ParameterSetTy &NewParameters) {
  for (const SCEV *Parameter : NewParameters) {
    // Normalize the SCEV to get the representing element for an invariant load.
    Parameter = extractConstantFactor(Parameter, *SE).second;
    Parameter = getRepresentingInvariantLoadSCEV(Parameter);

    if (Parameters.insert(Parameter))
      createParameterId(Parameter);
  }
}

__isl_give isl_id *Scop::getIdForParam(const SCEV *Parameter) {
  // Normalize the SCEV to get the representing element for an invariant load.
  Parameter = getRepresentingInvariantLoadSCEV(Parameter);
  return isl_id_copy(ParameterIds.lookup(Parameter));
}

__isl_give isl_set *
Scop::addNonEmptyDomainConstraints(__isl_take isl_set *C) const {
  isl_set *DomainContext = isl_union_set_params(getDomains());
  return isl_set_intersect_params(C, DomainContext);
}

bool Scop::isDominatedBy(const DominatorTree &DT, BasicBlock *BB) const {
  return DT.dominates(BB, getEntry());
}

void Scop::addUserAssumptions(AssumptionCache &AC, DominatorTree &DT,
                              LoopInfo &LI) {
  auto &F = getFunction();
  for (auto &Assumption : AC.assumptions()) {
    auto *CI = dyn_cast_or_null<CallInst>(Assumption);
    if (!CI || CI->getNumArgOperands() != 1)
      continue;

    bool InScop = contains(CI);
    if (!InScop && !isDominatedBy(DT, CI->getParent()))
      continue;

    auto *L = LI.getLoopFor(CI->getParent());
    auto *Val = CI->getArgOperand(0);
    ParameterSetTy DetectedParams;
    if (!isAffineConstraint(Val, &R, L, *SE, DetectedParams)) {
      emitOptimizationRemarkAnalysis(F.getContext(), DEBUG_TYPE, F,
                                     CI->getDebugLoc(),
                                     "Non-affine user assumption ignored.");
      continue;
    }

    // Collect all newly introduced parameters.
    ParameterSetTy NewParams;
    for (auto *Param : DetectedParams) {
      Param = extractConstantFactor(Param, *SE).second;
      Param = getRepresentingInvariantLoadSCEV(Param);
      if (Parameters.count(Param))
        continue;
      NewParams.insert(Param);
    }

    SmallVector<isl_set *, 2> ConditionSets;
    auto *TI = InScop ? CI->getParent()->getTerminator() : nullptr;
    auto &Stmt = InScop ? *getStmtFor(CI->getParent()) : *Stmts.begin();
    auto *Dom = InScop ? getDomainConditions(&Stmt) : isl_set_copy(Context);
    bool Valid = buildConditionSets(Stmt, Val, TI, L, Dom, ConditionSets);
    isl_set_free(Dom);

    if (!Valid)
      continue;

    isl_set *AssumptionCtx = nullptr;
    if (InScop) {
      AssumptionCtx = isl_set_complement(isl_set_params(ConditionSets[1]));
      isl_set_free(ConditionSets[0]);
    } else {
      AssumptionCtx = isl_set_complement(ConditionSets[1]);
      AssumptionCtx = isl_set_intersect(AssumptionCtx, ConditionSets[0]);
    }

    // Project out newly introduced parameters as they are not otherwise useful.
    if (!NewParams.empty()) {
      for (unsigned u = 0; u < isl_set_n_param(AssumptionCtx); u++) {
        auto *Id = isl_set_get_dim_id(AssumptionCtx, isl_dim_param, u);
        auto *Param = static_cast<const SCEV *>(isl_id_get_user(Id));
        isl_id_free(Id);

        if (!NewParams.count(Param))
          continue;

        AssumptionCtx =
            isl_set_project_out(AssumptionCtx, isl_dim_param, u--, 1);
      }
    }

    emitOptimizationRemarkAnalysis(
        F.getContext(), DEBUG_TYPE, F, CI->getDebugLoc(),
        "Use user assumption: " + stringFromIslObj(AssumptionCtx));
    Context = isl_set_intersect(Context, AssumptionCtx);
  }
}

void Scop::addUserContext() {
  if (UserContextStr.empty())
    return;

  isl_set *UserContext =
      isl_set_read_from_str(getIslCtx(), UserContextStr.c_str());
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
    auto *NameContext = isl_set_get_dim_name(Context, isl_dim_param, i);
    auto *NameUserContext = isl_set_get_dim_name(UserContext, isl_dim_param, i);

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
  DenseMap<std::pair<const SCEV *, Type *>, LoadInst *> EquivClasses;

  const InvariantLoadsSetTy &RIL = getRequiredInvariantLoads();
  for (LoadInst *LInst : RIL) {
    const SCEV *PointerSCEV = SE->getSCEV(LInst->getPointerOperand());

    Type *Ty = LInst->getType();
    LoadInst *&ClassRep = EquivClasses[std::make_pair(PointerSCEV, Ty)];
    if (ClassRep) {
      InvEquivClassVMap[LInst] = ClassRep;
      continue;
    }

    ClassRep = LInst;
    InvariantEquivClasses.emplace_back(
        InvariantEquivClassTy{PointerSCEV, MemoryAccessList(), nullptr, Ty});
  }
}

void Scop::buildContext() {
  isl_space *Space = isl_space_params_alloc(getIslCtx(), 0);
  Context = isl_set_universe(isl_space_copy(Space));
  InvalidContext = isl_set_empty(isl_space_copy(Space));
  AssumedContext = isl_set_universe(Space);
}

void Scop::addParameterBounds() {
  unsigned PDim = 0;
  for (auto *Parameter : Parameters) {
    ConstantRange SRange = SE->getSignedRange(Parameter);
    Context = addRangeBoundsToSet(Context, SRange, PDim++, isl_dim_param);
  }
}

void Scop::realignParams() {
  if (PollyIgnoreParamBounds)
    return;

  // Add all parameters into a common model.
  isl_space *Space = isl_space_params_alloc(getIslCtx(), ParameterIds.size());

  unsigned PDim = 0;
  for (const auto *Parameter : Parameters) {
    isl_id *id = getIdForParam(Parameter);
    Space = isl_space_set_dim_id(Space, isl_dim_param, PDim++, id);
  }

  // Align the parameters of all data structures to the model.
  Context = isl_set_align_params(Context, Space);

  // As all parameters are known add bounds to them.
  addParameterBounds();

  for (ScopStmt &Stmt : *this)
    Stmt.realignParams();

  // Simplify the schedule according to the context too.
  Schedule = isl_schedule_gist_domain_params(Schedule, getContext());
}

static __isl_give isl_set *
simplifyAssumptionContext(__isl_take isl_set *AssumptionContext,
                          const Scop &S) {
  // If we have modeled all blocks in the SCoP that have side effects we can
  // simplify the context with the constraints that are needed for anything to
  // be executed at all. However, if we have error blocks in the SCoP we already
  // assumed some parameter combinations cannot occur and removed them from the
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
  InvalidContext = isl_set_align_params(InvalidContext, getParamSpace());
}

/// Add the minimal/maximal access in @p Set to @p User.
static isl_stat buildMinMaxAccess(__isl_take isl_set *Set, void *User) {
  Scop::MinMaxVectorTy *MinMaxAccesses = (Scop::MinMaxVectorTy *)User;
  isl_pw_multi_aff *MinPMA, *MaxPMA;
  isl_pw_aff *LastDimAff;
  isl_aff *OneAff;
  unsigned Pos;

  Set = isl_set_remove_divs(Set);

  if (isl_set_n_basic_set(Set) >= MaxDisjunctsInDomain) {
    isl_set_free(Set);
    return isl_stat_error;
  }

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

/// Wrapper function to calculate minimal/maximal accesses to each array.
static bool calculateMinMaxAccess(Scop::AliasGroupTy AliasGroup, Scop &S,
                                  Scop::MinMaxVectorTy &MinMaxAccesses) {

  MinMaxAccesses.reserve(AliasGroup.size());

  isl_union_set *Domains = S.getDomains();
  isl_union_map *Accesses = isl_union_map_empty(S.getParamSpace());

  for (MemoryAccess *MA : AliasGroup)
    Accesses = isl_union_map_add_map(Accesses, MA->getAccessRelation());

  Accesses = isl_union_map_intersect_domain(Accesses, Domains);
  isl_union_set *Locations = isl_union_map_range(Accesses);
  Locations = isl_union_set_coalesce(Locations);
  Locations = isl_union_set_detect_equalities(Locations);
  bool Valid = (0 == isl_union_set_foreach_set(Locations, buildMinMaxAccess,
                                               &MinMaxAccesses));
  isl_union_set_free(Locations);
  return Valid;
}

/// Helper to treat non-affine regions and basic blocks the same.
///
///{

/// Return the block that is the representing block for @p RN.
static inline BasicBlock *getRegionNodeBasicBlock(RegionNode *RN) {
  return RN->isSubRegion() ? RN->getNodeAs<Region>()->getEntry()
                           : RN->getNodeAs<BasicBlock>();
}

/// Return the @p idx'th block that is executed after @p RN.
static inline BasicBlock *
getRegionNodeSuccessor(RegionNode *RN, TerminatorInst *TI, unsigned idx) {
  if (RN->isSubRegion()) {
    assert(idx == 0);
    return RN->getNodeAs<Region>()->getExit();
  }
  return TI->getSuccessor(idx);
}

/// Return the smallest loop surrounding @p RN.
static inline Loop *getRegionNodeLoop(RegionNode *RN, LoopInfo &LI) {
  if (!RN->isSubRegion()) {
    BasicBlock *BB = RN->getNodeAs<BasicBlock>();
    Loop *L = LI.getLoopFor(BB);

    // Unreachable statements are not considered to belong to a LLVM loop, as
    // they are not part of an actual loop in the control flow graph.
    // Nevertheless, we handle certain unreachable statements that are common
    // when modeling run-time bounds checks as being part of the loop to be
    // able to model them and to later eliminate the run-time bounds checks.
    //
    // Specifically, for basic blocks that terminate in an unreachable and
    // where the immeditate predecessor is part of a loop, we assume these
    // basic blocks belong to the loop the predecessor belongs to. This
    // allows us to model the following code.
    //
    // for (i = 0; i < N; i++) {
    //   if (i > 1024)
    //     abort();            <- this abort might be translated to an
    //                            unreachable
    //
    //   A[i] = ...
    // }
    if (!L && isa<UnreachableInst>(BB->getTerminator()) && BB->getPrevNode())
      L = LI.getLoopFor(BB->getPrevNode());
    return L;
  }

  Region *NonAffineSubRegion = RN->getNodeAs<Region>();
  Loop *L = LI.getLoopFor(NonAffineSubRegion->getEntry());
  while (L && NonAffineSubRegion->contains(L))
    L = L->getParentLoop();
  return L;
}

/// Get the number of blocks in @p L.
///
/// The number of blocks in a loop are the number of basic blocks actually
/// belonging to the loop, as well as all single basic blocks that the loop
/// exits to and which terminate in an unreachable instruction. We do not
/// allow such basic blocks in the exit of a scop, hence they belong to the
/// scop and represent run-time conditions which we want to model and
/// subsequently speculate away.
///
/// @see getRegionNodeLoop for additional details.
long getNumBlocksInLoop(Loop *L) {
  long NumBlocks = L->getNumBlocks();
  SmallVector<llvm::BasicBlock *, 4> ExitBlocks;
  L->getExitBlocks(ExitBlocks);

  for (auto ExitBlock : ExitBlocks) {
    if (isa<UnreachableInst>(ExitBlock->getTerminator()))
      NumBlocks++;
  }
  return NumBlocks;
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

__isl_give isl_set *Scop::getDomainConditions(const ScopStmt *Stmt) const {
  return getDomainConditions(Stmt->getEntryBlock());
}

__isl_give isl_set *Scop::getDomainConditions(BasicBlock *BB) const {
  auto DIt = DomainMap.find(BB);
  if (DIt != DomainMap.end())
    return isl_set_copy(DIt->getSecond());

  auto &RI = *R.getRegionInfo();
  auto *BBR = RI.getRegionFor(BB);
  while (BBR->getEntry() == BB)
    BBR = BBR->getParent();
  return getDomainConditions(BBR->getEntry());
}

bool Scop::buildDomains(Region *R, DominatorTree &DT, LoopInfo &LI) {

  bool IsOnlyNonAffineRegion = isNonAffineSubRegion(R);
  auto *EntryBB = R->getEntry();
  auto *L = IsOnlyNonAffineRegion ? nullptr : LI.getLoopFor(EntryBB);
  int LD = getRelativeLoopDepth(L);
  auto *S = isl_set_universe(isl_space_set_alloc(getIslCtx(), 0, LD + 1));

  while (LD-- >= 0) {
    S = addDomainDimId(S, LD + 1, L);
    L = L->getParentLoop();
  }

  // Initialize the invalid domain.
  auto *EntryStmt = getStmtFor(EntryBB);
  EntryStmt->setInvalidDomain(isl_set_empty(isl_set_get_space(S)));

  DomainMap[EntryBB] = S;

  if (IsOnlyNonAffineRegion)
    return !containsErrorBlock(R->getNode(), *R, LI, DT);

  if (!buildDomainsWithBranchConstraints(R, DT, LI))
    return false;

  if (!propagateDomainConstraints(R, DT, LI))
    return false;

  // Error blocks and blocks dominated by them have been assumed to never be
  // executed. Representing them in the Scop does not add any value. In fact,
  // it is likely to cause issues during construction of the ScopStmts. The
  // contents of error blocks have not been verified to be expressible and
  // will cause problems when building up a ScopStmt for them.
  // Furthermore, basic blocks dominated by error blocks may reference
  // instructions in the error block which, if the error block is not modeled,
  // can themselves not be constructed properly. To this end we will replace
  // the domains of error blocks and those only reachable via error blocks
  // with an empty set. Additionally, we will record for each block under which
  // parameter combination it would be reached via an error block in its
  // InvalidDomain. This information is needed during load hoisting.
  if (!propagateInvalidStmtDomains(R, DT, LI))
    return false;

  return true;
}

/// Adjust the dimensions of @p Dom that was constructed for @p OldL
///        to be compatible to domains constructed for loop @p NewL.
///
/// This function assumes @p NewL and @p OldL are equal or there is a CFG
/// edge from @p OldL to @p NewL.
static __isl_give isl_set *adjustDomainDimensions(Scop &S,
                                                  __isl_take isl_set *Dom,
                                                  Loop *OldL, Loop *NewL) {

  // If the loops are the same there is nothing to do.
  if (NewL == OldL)
    return Dom;

  int OldDepth = S.getRelativeLoopDepth(OldL);
  int NewDepth = S.getRelativeLoopDepth(NewL);
  // If both loops are non-affine loops there is nothing to do.
  if (OldDepth == -1 && NewDepth == -1)
    return Dom;

  // Distinguish three cases:
  //   1) The depth is the same but the loops are not.
  //      => One loop was left one was entered.
  //   2) The depth increased from OldL to NewL.
  //      => One loop was entered, none was left.
  //   3) The depth decreased from OldL to NewL.
  //      => Loops were left were difference of the depths defines how many.
  if (OldDepth == NewDepth) {
    assert(OldL->getParentLoop() == NewL->getParentLoop());
    Dom = isl_set_project_out(Dom, isl_dim_set, NewDepth, 1);
    Dom = isl_set_add_dims(Dom, isl_dim_set, 1);
    Dom = addDomainDimId(Dom, NewDepth, NewL);
  } else if (OldDepth < NewDepth) {
    assert(OldDepth + 1 == NewDepth);
    auto &R = S.getRegion();
    (void)R;
    assert(NewL->getParentLoop() == OldL ||
           ((!OldL || !R.contains(OldL)) && R.contains(NewL)));
    Dom = isl_set_add_dims(Dom, isl_dim_set, 1);
    Dom = addDomainDimId(Dom, NewDepth, NewL);
  } else {
    assert(OldDepth > NewDepth);
    int Diff = OldDepth - NewDepth;
    int NumDim = isl_set_n_dim(Dom);
    assert(NumDim >= Diff);
    Dom = isl_set_project_out(Dom, isl_dim_set, NumDim - Diff, Diff);
  }

  return Dom;
}

bool Scop::propagateInvalidStmtDomains(Region *R, DominatorTree &DT,
                                       LoopInfo &LI) {
  ReversePostOrderTraversal<Region *> RTraversal(R);
  for (auto *RN : RTraversal) {

    // Recurse for affine subregions but go on for basic blocks and non-affine
    // subregions.
    if (RN->isSubRegion()) {
      Region *SubRegion = RN->getNodeAs<Region>();
      if (!isNonAffineSubRegion(SubRegion)) {
        propagateInvalidStmtDomains(SubRegion, DT, LI);
        continue;
      }
    }

    bool ContainsErrorBlock = containsErrorBlock(RN, getRegion(), LI, DT);
    BasicBlock *BB = getRegionNodeBasicBlock(RN);
    ScopStmt *Stmt = getStmtFor(BB);
    isl_set *&Domain = DomainMap[BB];
    assert(Domain && "Cannot propagate a nullptr");

    auto *InvalidDomain = Stmt->getInvalidDomain();
    bool IsInvalidBlock =
        ContainsErrorBlock || isl_set_is_subset(Domain, InvalidDomain);

    if (!IsInvalidBlock) {
      InvalidDomain = isl_set_intersect(InvalidDomain, isl_set_copy(Domain));
    } else {
      isl_set_free(InvalidDomain);
      InvalidDomain = Domain;
      isl_set *DomPar = isl_set_params(isl_set_copy(Domain));
      recordAssumption(ERRORBLOCK, DomPar, BB->getTerminator()->getDebugLoc(),
                       AS_RESTRICTION);
      Domain = nullptr;
    }

    if (isl_set_is_empty(InvalidDomain)) {
      Stmt->setInvalidDomain(InvalidDomain);
      continue;
    }

    auto *BBLoop = getRegionNodeLoop(RN, LI);
    auto *TI = BB->getTerminator();
    unsigned NumSuccs = RN->isSubRegion() ? 1 : TI->getNumSuccessors();
    for (unsigned u = 0; u < NumSuccs; u++) {
      auto *SuccBB = getRegionNodeSuccessor(RN, TI, u);
      auto *SuccStmt = getStmtFor(SuccBB);

      // Skip successors outside the SCoP.
      if (!SuccStmt)
        continue;

      // Skip backedges.
      if (DT.dominates(SuccBB, BB))
        continue;

      auto *SuccBBLoop = SuccStmt->getSurroundingLoop();
      auto *AdjustedInvalidDomain = adjustDomainDimensions(
          *this, isl_set_copy(InvalidDomain), BBLoop, SuccBBLoop);
      auto *SuccInvalidDomain = SuccStmt->getInvalidDomain();
      SuccInvalidDomain =
          isl_set_union(SuccInvalidDomain, AdjustedInvalidDomain);
      SuccInvalidDomain = isl_set_coalesce(SuccInvalidDomain);
      unsigned NumConjucts = isl_set_n_basic_set(SuccInvalidDomain);
      SuccStmt->setInvalidDomain(SuccInvalidDomain);

      // Check if the maximal number of domain disjunctions was reached.
      // In case this happens we will bail.
      if (NumConjucts < MaxDisjunctsInDomain)
        continue;

      isl_set_free(InvalidDomain);
      invalidate(COMPLEXITY, TI->getDebugLoc());
      return false;
    }

    Stmt->setInvalidDomain(InvalidDomain);
  }

  return true;
}

void Scop::propagateDomainConstraintsToRegionExit(
    BasicBlock *BB, Loop *BBLoop,
    SmallPtrSetImpl<BasicBlock *> &FinishedExitBlocks, LoopInfo &LI) {

  // Check if the block @p BB is the entry of a region. If so we propagate it's
  // domain to the exit block of the region. Otherwise we are done.
  auto *RI = R.getRegionInfo();
  auto *BBReg = RI ? RI->getRegionFor(BB) : nullptr;
  auto *ExitBB = BBReg ? BBReg->getExit() : nullptr;
  if (!BBReg || BBReg->getEntry() != BB || !contains(ExitBB))
    return;

  // Do not propagate the domain if there is a loop backedge inside the region
  // that would prevent the exit block from being executed.
  auto *L = BBLoop;
  while (L && contains(L)) {
    SmallVector<BasicBlock *, 4> LatchBBs;
    BBLoop->getLoopLatches(LatchBBs);
    for (auto *LatchBB : LatchBBs)
      if (BB != LatchBB && BBReg->contains(LatchBB))
        return;
    L = L->getParentLoop();
  }

  auto *Domain = DomainMap[BB];
  assert(Domain && "Cannot propagate a nullptr");

  auto *ExitStmt = getStmtFor(ExitBB);
  auto *ExitBBLoop = ExitStmt->getSurroundingLoop();

  // Since the dimensions of @p BB and @p ExitBB might be different we have to
  // adjust the domain before we can propagate it.
  auto *AdjustedDomain =
      adjustDomainDimensions(*this, isl_set_copy(Domain), BBLoop, ExitBBLoop);
  auto *&ExitDomain = DomainMap[ExitBB];

  // If the exit domain is not yet created we set it otherwise we "add" the
  // current domain.
  ExitDomain =
      ExitDomain ? isl_set_union(AdjustedDomain, ExitDomain) : AdjustedDomain;

  // Initialize the invalid domain.
  ExitStmt->setInvalidDomain(isl_set_empty(isl_set_get_space(ExitDomain)));

  FinishedExitBlocks.insert(ExitBB);
}

bool Scop::buildDomainsWithBranchConstraints(Region *R, DominatorTree &DT,
                                             LoopInfo &LI) {
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

  SmallPtrSet<BasicBlock *, 8> FinishedExitBlocks;
  ReversePostOrderTraversal<Region *> RTraversal(R);
  for (auto *RN : RTraversal) {

    // Recurse for affine subregions but go on for basic blocks and non-affine
    // subregions.
    if (RN->isSubRegion()) {
      Region *SubRegion = RN->getNodeAs<Region>();
      if (!isNonAffineSubRegion(SubRegion)) {
        if (!buildDomainsWithBranchConstraints(SubRegion, DT, LI))
          return false;
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
    if (!Domain)
      continue;
    MaxLoopDepth = std::max(MaxLoopDepth, isl_set_n_dim(Domain));

    auto *BBLoop = getRegionNodeLoop(RN, LI);
    // Propagate the domain from BB directly to blocks that have a superset
    // domain, at the moment only region exit nodes of regions that start in BB.
    propagateDomainConstraintsToRegionExit(BB, BBLoop, FinishedExitBlocks, LI);

    // If all successors of BB have been set a domain through the propagation
    // above we do not need to build condition sets but can just skip this
    // block. However, it is important to note that this is a local property
    // with regards to the region @p R. To this end FinishedExitBlocks is a
    // local variable.
    auto IsFinishedRegionExit = [&FinishedExitBlocks](BasicBlock *SuccBB) {
      return FinishedExitBlocks.count(SuccBB);
    };
    if (std::all_of(succ_begin(BB), succ_end(BB), IsFinishedRegionExit))
      continue;

    // Build the condition sets for the successor nodes of the current region
    // node. If it is a non-affine subregion we will always execute the single
    // exit node, hence the single entry node domain is the condition set. For
    // basic blocks we use the helper function buildConditionSets.
    SmallVector<isl_set *, 8> ConditionSets;
    if (RN->isSubRegion())
      ConditionSets.push_back(isl_set_copy(Domain));
    else if (!buildConditionSets(*getStmtFor(BB), TI, BBLoop, Domain,
                                 ConditionSets))
      return false;

    // Now iterate over the successors and set their initial domain based on
    // their condition set. We skip back edges here and have to be careful when
    // we leave a loop not to keep constraints over a dimension that doesn't
    // exist anymore.
    assert(RN->isSubRegion() || TI->getNumSuccessors() == ConditionSets.size());
    for (unsigned u = 0, e = ConditionSets.size(); u < e; u++) {
      isl_set *CondSet = ConditionSets[u];
      BasicBlock *SuccBB = getRegionNodeSuccessor(RN, TI, u);

      auto *SuccStmt = getStmtFor(SuccBB);
      // Skip blocks outside the region.
      if (!SuccStmt) {
        isl_set_free(CondSet);
        continue;
      }

      // If we propagate the domain of some block to "SuccBB" we do not have to
      // adjust the domain.
      if (FinishedExitBlocks.count(SuccBB)) {
        isl_set_free(CondSet);
        continue;
      }

      // Skip back edges.
      if (DT.dominates(SuccBB, BB)) {
        isl_set_free(CondSet);
        continue;
      }

      auto *SuccBBLoop = SuccStmt->getSurroundingLoop();
      CondSet = adjustDomainDimensions(*this, CondSet, BBLoop, SuccBBLoop);

      // Set the domain for the successor or merge it with an existing domain in
      // case there are multiple paths (without loop back edges) to the
      // successor block.
      isl_set *&SuccDomain = DomainMap[SuccBB];

      if (SuccDomain) {
        SuccDomain = isl_set_coalesce(isl_set_union(SuccDomain, CondSet));
      } else {
        // Initialize the invalid domain.
        SuccStmt->setInvalidDomain(isl_set_empty(isl_set_get_space(CondSet)));
        SuccDomain = CondSet;
      }

      // Check if the maximal number of domain disjunctions was reached.
      // In case this happens we will clean up and bail.
      if (isl_set_n_basic_set(SuccDomain) < MaxDisjunctsInDomain)
        continue;

      invalidate(COMPLEXITY, DebugLoc());
      while (++u < ConditionSets.size())
        isl_set_free(ConditionSets[u]);
      return false;
    }
  }

  return true;
}

__isl_give isl_set *
Scop::getPredecessorDomainConstraints(BasicBlock *BB,
                                      __isl_keep isl_set *Domain,
                                      DominatorTree &DT, LoopInfo &LI) {
  // If @p BB is the ScopEntry we are done
  if (R.getEntry() == BB)
    return isl_set_universe(isl_set_get_space(Domain));

  // The region info of this function.
  auto &RI = *R.getRegionInfo();

  auto *BBLoop = getStmtFor(BB)->getSurroundingLoop();

  // A domain to collect all predecessor domains, thus all conditions under
  // which the block is executed. To this end we start with the empty domain.
  isl_set *PredDom = isl_set_empty(isl_set_get_space(Domain));

  // Set of regions of which the entry block domain has been propagated to BB.
  // all predecessors inside any of the regions can be skipped.
  SmallSet<Region *, 8> PropagatedRegions;

  for (auto *PredBB : predecessors(BB)) {
    // Skip backedges.
    if (DT.dominates(BB, PredBB))
      continue;

    // If the predecessor is in a region we used for propagation we can skip it.
    auto PredBBInRegion = [PredBB](Region *PR) { return PR->contains(PredBB); };
    if (std::any_of(PropagatedRegions.begin(), PropagatedRegions.end(),
                    PredBBInRegion)) {
      continue;
    }

    // Check if there is a valid region we can use for propagation, thus look
    // for a region that contains the predecessor and has @p BB as exit block.
    auto *PredR = RI.getRegionFor(PredBB);
    while (PredR->getExit() != BB && !PredR->contains(BB))
      PredR->getParent();

    // If a valid region for propagation was found use the entry of that region
    // for propagation, otherwise the PredBB directly.
    if (PredR->getExit() == BB) {
      PredBB = PredR->getEntry();
      PropagatedRegions.insert(PredR);
    }

    auto *PredBBDom = getDomainConditions(PredBB);
    auto *PredBBLoop = getStmtFor(PredBB)->getSurroundingLoop();
    PredBBDom = adjustDomainDimensions(*this, PredBBDom, PredBBLoop, BBLoop);

    PredDom = isl_set_union(PredDom, PredBBDom);
  }

  return PredDom;
}

bool Scop::propagateDomainConstraints(Region *R, DominatorTree &DT,
                                      LoopInfo &LI) {
  // Iterate over the region R and propagate the domain constrains from the
  // predecessors to the current node. In contrast to the
  // buildDomainsWithBranchConstraints function, this one will pull the domain
  // information from the predecessors instead of pushing it to the successors.
  // Additionally, we assume the domains to be already present in the domain
  // map here. However, we iterate again in reverse post order so we know all
  // predecessors have been visited before a block or non-affine subregion is
  // visited.

  ReversePostOrderTraversal<Region *> RTraversal(R);
  for (auto *RN : RTraversal) {

    // Recurse for affine subregions but go on for basic blocks and non-affine
    // subregions.
    if (RN->isSubRegion()) {
      Region *SubRegion = RN->getNodeAs<Region>();
      if (!isNonAffineSubRegion(SubRegion)) {
        if (!propagateDomainConstraints(SubRegion, DT, LI))
          return false;
        continue;
      }
    }

    BasicBlock *BB = getRegionNodeBasicBlock(RN);
    isl_set *&Domain = DomainMap[BB];
    assert(Domain);

    // Under the union of all predecessor conditions we can reach this block.
    auto *PredDom = getPredecessorDomainConstraints(BB, Domain, DT, LI);
    Domain = isl_set_coalesce(isl_set_intersect(Domain, PredDom));
    Domain = isl_set_align_params(Domain, getParamSpace());

    Loop *BBLoop = getRegionNodeLoop(RN, LI);
    if (BBLoop && BBLoop->getHeader() == BB && contains(BBLoop))
      if (!addLoopBoundsToHeaderDomain(BBLoop, LI))
        return false;
  }

  return true;
}

/// Create a map to map from a given iteration to a subsequent iteration.
///
/// This map maps from SetSpace -> SetSpace where the dimensions @p Dim
/// is incremented by one and all other dimensions are equal, e.g.,
///             [i0, i1, i2, i3] -> [i0, i1, i2 + 1, i3]
///
/// if @p Dim is 2 and @p SetSpace has 4 dimensions.
static __isl_give isl_map *
createNextIterationMap(__isl_take isl_space *SetSpace, unsigned Dim) {
  auto *MapSpace = isl_space_map_from_set(SetSpace);
  auto *NextIterationMap = isl_map_universe(isl_space_copy(MapSpace));
  for (unsigned u = 0; u < isl_map_dim(NextIterationMap, isl_dim_in); u++)
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

bool Scop::addLoopBoundsToHeaderDomain(Loop *L, LoopInfo &LI) {
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
    assert(BI && "Only branch instructions allowed in loop latches");

    if (BI->isUnconditional())
      BackedgeCondition = isl_set_copy(LatchBBDom);
    else {
      SmallVector<isl_set *, 8> ConditionSets;
      int idx = BI->getSuccessor(0) != HeaderBB;
      if (!buildConditionSets(*getStmtFor(LatchBB), TI, L, LatchBBDom,
                              ConditionSets)) {
        isl_map_free(NextIterationMap);
        isl_set_free(UnionBackedgeCondition);
        return false;
      }

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
    return true;
  }

  isl_set *UnboundedCtx = isl_set_params(Parts.first);
  recordAssumption(INFINITELOOP, UnboundedCtx,
                   HeaderBB->getTerminator()->getDebugLoc(), AS_RESTRICTION);
  return true;
}

MemoryAccess *Scop::lookupBasePtrAccess(MemoryAccess *MA) {
  Value *PointerBase = MA->getOriginalBaseAddr();

  auto *PointerBaseInst = dyn_cast<Instruction>(PointerBase);
  if (!PointerBaseInst)
    return nullptr;

  auto *BasePtrStmt = getStmtFor(PointerBaseInst);
  if (!BasePtrStmt)
    return nullptr;

  return BasePtrStmt->getArrayAccessOrNULLFor(PointerBaseInst);
}

bool Scop::hasNonHoistableBasePtrInScop(MemoryAccess *MA,
                                        __isl_keep isl_union_map *Writes) {
  if (auto *BasePtrMA = lookupBasePtrAccess(MA)) {
    auto *NHCtx = getNonHoistableCtx(BasePtrMA, Writes);
    bool Hoistable = NHCtx != nullptr;
    isl_set_free(NHCtx);
    return !Hoistable;
  }

  Value *BaseAddr = MA->getOriginalBaseAddr();
  if (auto *BasePtrInst = dyn_cast<Instruction>(BaseAddr))
    if (!isa<LoadInst>(BasePtrInst))
      return contains(BasePtrInst);

  return false;
}

bool Scop::buildAliasChecks(AliasAnalysis &AA) {
  if (!PollyUseRuntimeAliasChecks)
    return true;

  if (buildAliasGroups(AA)) {
    // Aliasing assumptions do not go through addAssumption but we still want to
    // collect statistics so we do it here explicitly.
    if (MinMaxAliasGroups.size())
      AssumptionsAliasing++;
    return true;
  }

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
  return false;
}

std::tuple<Scop::AliasGroupVectorTy, DenseSet<const ScopArrayInfo *>>
Scop::buildAliasGroupsForAccesses(AliasAnalysis &AA) {
  AliasSetTracker AST(AA);

  DenseMap<Value *, MemoryAccess *> PtrToAcc;
  DenseSet<const ScopArrayInfo *> HasWriteAccess;
  for (ScopStmt &Stmt : *this) {

    isl_set *StmtDomain = Stmt.getDomain();
    bool StmtDomainEmpty = isl_set_is_empty(StmtDomain);
    isl_set_free(StmtDomain);

    // Statements with an empty domain will never be executed.
    if (StmtDomainEmpty)
      continue;

    for (MemoryAccess *MA : Stmt) {
      if (MA->isScalarKind())
        continue;
      if (!MA->isRead())
        HasWriteAccess.insert(MA->getScopArrayInfo());
      MemAccInst Acc(MA->getAccessInstruction());
      if (MA->isRead() && isa<MemTransferInst>(Acc))
        PtrToAcc[cast<MemTransferInst>(Acc)->getRawSource()] = MA;
      else
        PtrToAcc[Acc.getPointerOperand()] = MA;
      AST.add(Acc);
    }
  }

  AliasGroupVectorTy AliasGroups;
  for (AliasSet &AS : AST) {
    if (AS.isMustAlias() || AS.isForwardingAliasSet())
      continue;
    AliasGroupTy AG;
    for (auto &PR : AS)
      AG.push_back(PtrToAcc[PR.getValue()]);
    if (AG.size() < 2)
      continue;
    AliasGroups.push_back(std::move(AG));
  }

  return std::make_tuple(AliasGroups, HasWriteAccess);
}

void Scop::splitAliasGroupsByDomain(AliasGroupVectorTy &AliasGroups) {
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
}

bool Scop::buildAliasGroups(AliasAnalysis &AA) {
  // To create sound alias checks we perform the following steps:
  //   o) We partition each group into read only and non read only accesses.
  //   o) For each group with more than one base pointer we then compute minimal
  //      and maximal accesses to each array of a group in read only and non
  //      read only partitions separately.
  AliasGroupVectorTy AliasGroups;
  DenseSet<const ScopArrayInfo *> HasWriteAccess;

  std::tie(AliasGroups, HasWriteAccess) = buildAliasGroupsForAccesses(AA);

  splitAliasGroupsByDomain(AliasGroups);

  for (AliasGroupTy &AG : AliasGroups) {
    bool Valid = buildAliasGroup(AG, HasWriteAccess);
    if (!Valid)
      return false;
  }

  return true;
}

bool Scop::buildAliasGroup(Scop::AliasGroupTy &AliasGroup,
                           DenseSet<const ScopArrayInfo *> HasWriteAccess) {
  AliasGroupTy ReadOnlyAccesses;
  AliasGroupTy ReadWriteAccesses;
  SmallPtrSet<const ScopArrayInfo *, 4> ReadWriteArrays;
  SmallPtrSet<const ScopArrayInfo *, 4> ReadOnlyArrays;

  auto &F = getFunction();

  if (AliasGroup.size() < 2)
    return true;

  for (MemoryAccess *Access : AliasGroup) {
    emitOptimizationRemarkAnalysis(
        F.getContext(), DEBUG_TYPE, F,
        Access->getAccessInstruction()->getDebugLoc(),
        "Possibly aliasing pointer, use restrict keyword.");

    const ScopArrayInfo *Array = Access->getScopArrayInfo();
    if (HasWriteAccess.count(Array)) {
      ReadWriteArrays.insert(Array);
      ReadWriteAccesses.push_back(Access);
    } else {
      ReadOnlyArrays.insert(Array);
      ReadOnlyAccesses.push_back(Access);
    }
  }

  // If there are no read-only pointers, and less than two read-write pointers,
  // no alias check is needed.
  if (ReadOnlyAccesses.empty() && ReadWriteArrays.size() <= 1)
    return true;

  // If there is no read-write pointer, no alias check is needed.
  if (ReadWriteArrays.empty())
    return true;

  // For non-affine accesses, no alias check can be generated as we cannot
  // compute a sufficiently tight lower and upper bound: bail out.
  for (MemoryAccess *MA : AliasGroup) {
    if (!MA->isAffine()) {
      invalidate(ALIASING, MA->getAccessInstruction()->getDebugLoc());
      return false;
    }
  }

  // Ensure that for all memory accesses for which we generate alias checks,
  // their base pointers are available.
  for (MemoryAccess *MA : AliasGroup) {
    if (MemoryAccess *BasePtrMA = lookupBasePtrAccess(MA))
      addRequiredInvariantLoad(
          cast<LoadInst>(BasePtrMA->getAccessInstruction()));
  }

  MinMaxAliasGroups.emplace_back();
  MinMaxVectorPairTy &pair = MinMaxAliasGroups.back();
  MinMaxVectorTy &MinMaxAccessesReadWrite = pair.first;
  MinMaxVectorTy &MinMaxAccessesReadOnly = pair.second;

  bool Valid;

  Valid =
      calculateMinMaxAccess(ReadWriteAccesses, *this, MinMaxAccessesReadWrite);

  if (!Valid)
    return false;

  // Bail out if the number of values we need to compare is too large.
  // This is important as the number of comparisons grows quadratically with
  // the number of values we need to compare.
  if (MinMaxAccessesReadWrite.size() + ReadOnlyArrays.size() >
      RunTimeChecksMaxArraysPerGroup)
    return false;

  Valid =
      calculateMinMaxAccess(ReadOnlyAccesses, *this, MinMaxAccessesReadOnly);

  if (!Valid)
    return false;

  return true;
}

/// Get the smallest loop that contains @p S but is not in @p S.
static Loop *getLoopSurroundingScop(Scop &S, LoopInfo &LI) {
  // Start with the smallest loop containing the entry and expand that
  // loop until it contains all blocks in the region. If there is a loop
  // containing all blocks in the region check if it is itself contained
  // and if so take the parent loop as it will be the smallest containing
  // the region but not contained by it.
  Loop *L = LI.getLoopFor(S.getEntry());
  while (L) {
    bool AllContained = true;
    for (auto *BB : S.blocks())
      AllContained &= L->contains(BB);
    if (AllContained)
      break;
    L = L->getParentLoop();
  }

  return L ? (S.contains(L) ? L->getParentLoop() : L) : nullptr;
}

Scop::Scop(Region &R, ScalarEvolution &ScalarEvolution, LoopInfo &LI,
           ScopDetection::DetectionContext &DC)
    : SE(&ScalarEvolution), R(R), IsOptimized(false),
      HasSingleExitEdge(R.getExitingBlock()), HasErrorBlock(false),
      MaxLoopDepth(0), CopyStmtsNum(0), DC(DC),
      IslCtx(isl_ctx_alloc(), isl_ctx_free), Context(nullptr),
      Affinator(this, LI), AssumedContext(nullptr), InvalidContext(nullptr),
      Schedule(nullptr) {
  if (IslOnErrorAbort)
    isl_options_set_on_error(getIslCtx(), ISL_ON_ERROR_ABORT);
  buildContext();
}

void Scop::foldSizeConstantsToRight() {
  isl_union_set *Accessed = isl_union_map_range(getAccesses());

  for (auto Array : arrays()) {
    if (Array->getNumberOfDimensions() <= 1)
      continue;

    isl_space *Space = Array->getSpace();

    Space = isl_space_align_params(Space, isl_union_set_get_space(Accessed));

    if (!isl_union_set_contains(Accessed, Space)) {
      isl_space_free(Space);
      continue;
    }

    isl_set *Elements = isl_union_set_extract_set(Accessed, Space);

    isl_map *Transform =
        isl_map_universe(isl_space_map_from_set(Array->getSpace()));

    std::vector<int> Int;

    int Dims = isl_set_dim(Elements, isl_dim_set);
    for (int i = 0; i < Dims; i++) {
      isl_set *DimOnly =
          isl_set_project_out(isl_set_copy(Elements), isl_dim_set, 0, i);
      DimOnly = isl_set_project_out(DimOnly, isl_dim_set, 1, Dims - i - 1);
      DimOnly = isl_set_lower_bound_si(DimOnly, isl_dim_set, 0, 0);

      isl_basic_set *DimHull = isl_set_affine_hull(DimOnly);

      if (i == Dims - 1) {
        Int.push_back(1);
        Transform = isl_map_equate(Transform, isl_dim_in, i, isl_dim_out, i);
        isl_basic_set_free(DimHull);
        continue;
      }

      if (isl_basic_set_dim(DimHull, isl_dim_div) == 1) {
        isl_aff *Diff = isl_basic_set_get_div(DimHull, 0);
        isl_val *Val = isl_aff_get_denominator_val(Diff);
        isl_aff_free(Diff);

        int ValInt = 1;

        if (isl_val_is_int(Val))
          ValInt = isl_val_get_num_si(Val);
        isl_val_free(Val);

        Int.push_back(ValInt);

        isl_constraint *C = isl_constraint_alloc_equality(
            isl_local_space_from_space(isl_map_get_space(Transform)));
        C = isl_constraint_set_coefficient_si(C, isl_dim_out, i, ValInt);
        C = isl_constraint_set_coefficient_si(C, isl_dim_in, i, -1);
        Transform = isl_map_add_constraint(Transform, C);
        isl_basic_set_free(DimHull);
        continue;
      }

      isl_basic_set *ZeroSet = isl_basic_set_copy(DimHull);
      ZeroSet = isl_basic_set_fix_si(ZeroSet, isl_dim_set, 0, 0);

      int ValInt = 1;
      if (isl_basic_set_is_equal(ZeroSet, DimHull)) {
        ValInt = 0;
      }

      Int.push_back(ValInt);
      Transform = isl_map_equate(Transform, isl_dim_in, i, isl_dim_out, i);
      isl_basic_set_free(DimHull);
      isl_basic_set_free(ZeroSet);
    }

    isl_set *MappedElements = isl_map_domain(isl_map_copy(Transform));

    if (!isl_set_is_subset(Elements, MappedElements)) {
      isl_set_free(Elements);
      isl_set_free(MappedElements);
      isl_map_free(Transform);
      continue;
    }

    isl_set_free(MappedElements);

    bool CanFold = true;

    if (Int[0] <= 1)
      CanFold = false;

    unsigned NumDims = Array->getNumberOfDimensions();
    for (unsigned i = 1; i < NumDims - 1; i++)
      if (Int[0] != Int[i] && Int[i])
        CanFold = false;

    if (!CanFold) {
      isl_set_free(Elements);
      isl_map_free(Transform);
      continue;
    }

    for (auto &Access : AccessFunctions)
      if (Access->getScopArrayInfo() == Array)
        Access->setAccessRelation(isl_map_apply_range(
            Access->getAccessRelation(), isl_map_copy(Transform)));

    isl_map_free(Transform);

    std::vector<const SCEV *> Sizes;
    for (unsigned i = 0; i < NumDims; i++) {
      auto Size = Array->getDimensionSize(i);

      if (i == NumDims - 1)
        Size = SE->getMulExpr(Size, SE->getConstant(Size->getType(), Int[0]));
      Sizes.push_back(Size);
    }

    Array->updateSizes(Sizes, false /* CheckConsistency */);

    isl_set_free(Elements);
  }
  isl_union_set_free(Accessed);
  return;
}

void Scop::finalizeAccesses() {
  updateAccessDimensionality();
  foldSizeConstantsToRight();
  foldAccessRelations();
  assumeNoOutOfBounds();
}

void Scop::init(AliasAnalysis &AA, AssumptionCache &AC, DominatorTree &DT,
                LoopInfo &LI) {
  buildInvariantEquivalenceClasses();

  if (!buildDomains(&R, DT, LI))
    return;

  addUserAssumptions(AC, DT, LI);

  // Remove empty statements.
  // Exit early in case there are no executable statements left in this scop.
  simplifySCoP(false);
  if (Stmts.empty())
    return;

  // The ScopStmts now have enough information to initialize themselves.
  for (ScopStmt &Stmt : Stmts)
    Stmt.init(LI);

  // Check early for a feasible runtime context.
  if (!hasFeasibleRuntimeContext())
    return;

  // Check early for profitability. Afterwards it cannot change anymore,
  // only the runtime context could become infeasible.
  if (!isProfitable(UnprofitableScalarAccs)) {
    invalidate(PROFITABLE, DebugLoc());
    return;
  }

  buildSchedule(LI);

  finalizeAccesses();

  realignParams();
  addUserContext();

  // After the context was fully constructed, thus all our knowledge about
  // the parameters is in there, we add all recorded assumptions to the
  // assumed/invalid context.
  addRecordedAssumptions();

  simplifyContexts();
  if (!buildAliasChecks(AA))
    return;

  hoistInvariantLoads();
  verifyInvariantLoads();
  simplifySCoP(true);

  // Check late for a feasible runtime context because profitability did not
  // change.
  if (!hasFeasibleRuntimeContext())
    return;
}

Scop::~Scop() {
  isl_set_free(Context);
  isl_set_free(AssumedContext);
  isl_set_free(InvalidContext);
  isl_schedule_free(Schedule);

  for (auto &It : ParameterIds)
    isl_id_free(It.second);

  for (auto It : DomainMap)
    isl_set_free(It.second);

  for (auto &AS : RecordedAssumptions)
    isl_set_free(AS.Set);

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
    isl_set_free(IAClass.ExecutionContext);

  // Explicitly release all Scop objects and the underlying isl objects before
  // we release the isl context.
  Stmts.clear();
  ScopArrayInfoSet.clear();
  ScopArrayInfoMap.clear();
  ScopArrayNameMap.clear();
  AccessFunctions.clear();
}

void Scop::updateAccessDimensionality() {
  // Check all array accesses for each base pointer and find a (virtual) element
  // size for the base pointer that divides all access functions.
  for (ScopStmt &Stmt : *this)
    for (MemoryAccess *Access : Stmt) {
      if (!Access->isArrayKind())
        continue;
      ScopArrayInfo *Array =
          const_cast<ScopArrayInfo *>(Access->getScopArrayInfo());

      if (Array->getNumberOfDimensions() != 1)
        continue;
      unsigned DivisibleSize = Array->getElemSizeInBytes();
      const SCEV *Subscript = Access->getSubscript(0);
      while (!isDivisible(Subscript, DivisibleSize, *SE))
        DivisibleSize /= 2;
      auto *Ty = IntegerType::get(SE->getContext(), DivisibleSize * 8);
      Array->updateElementType(Ty);
    }

  for (auto &Stmt : *this)
    for (auto &Access : Stmt)
      Access->updateDimensionality();
}

void Scop::foldAccessRelations() {
  for (auto &Stmt : *this)
    for (auto &Access : Stmt)
      Access->foldAccessRelation();
}

void Scop::assumeNoOutOfBounds() {
  for (auto &Stmt : *this)
    for (auto &Access : Stmt)
      Access->assumeNoOutOfBound();
}

void Scop::simplifySCoP(bool AfterHoisting) {
  for (auto StmtIt = Stmts.begin(), StmtEnd = Stmts.end(); StmtIt != StmtEnd;) {
    ScopStmt &Stmt = *StmtIt;

    bool RemoveStmt = Stmt.isEmpty();
    if (!RemoveStmt)
      RemoveStmt = !DomainMap[Stmt.getEntryBlock()];

    // Remove read only statements only after invariant loop hoisting.
    if (!RemoveStmt && AfterHoisting) {
      bool OnlyRead = true;
      for (MemoryAccess *MA : Stmt) {
        if (MA->isRead())
          continue;

        OnlyRead = false;
        break;
      }

      RemoveStmt = OnlyRead;
    }

    if (!RemoveStmt) {
      StmtIt++;
      continue;
    }

    // Remove the statement because it is unnecessary.
    if (Stmt.isRegionStmt())
      for (BasicBlock *BB : Stmt.getRegion()->blocks())
        StmtMap.erase(BB);
    else
      StmtMap.erase(Stmt.getBasicBlock());

    StmtIt = Stmts.erase(StmtIt);
  }
}

InvariantEquivClassTy *Scop::lookupInvariantEquivClass(Value *Val) {
  LoadInst *LInst = dyn_cast<LoadInst>(Val);
  if (!LInst)
    return nullptr;

  if (Value *Rep = InvEquivClassVMap.lookup(LInst))
    LInst = cast<LoadInst>(Rep);

  Type *Ty = LInst->getType();
  const SCEV *PointerSCEV = SE->getSCEV(LInst->getPointerOperand());
  for (auto &IAClass : InvariantEquivClasses) {
    if (PointerSCEV != IAClass.IdentifyingPointer || Ty != IAClass.AccessType)
      continue;

    auto &MAs = IAClass.InvariantAccesses;
    for (auto *MA : MAs)
      if (MA->getAccessInstruction() == Val)
        return &IAClass;
  }

  return nullptr;
}

/// Check if @p MA can always be hoisted without execution context.
static bool canAlwaysBeHoisted(MemoryAccess *MA, bool StmtInvalidCtxIsEmpty,
                               bool MAInvalidCtxIsEmpty,
                               bool NonHoistableCtxIsEmpty) {
  LoadInst *LInst = cast<LoadInst>(MA->getAccessInstruction());
  const DataLayout &DL = LInst->getParent()->getModule()->getDataLayout();
  // TODO: We can provide more information for better but more expensive
  //       results.
  if (!isDereferenceableAndAlignedPointer(LInst->getPointerOperand(),
                                          LInst->getAlignment(), DL))
    return false;

  // If the location might be overwritten we do not hoist it unconditionally.
  //
  // TODO: This is probably to conservative.
  if (!NonHoistableCtxIsEmpty)
    return false;

  // If a dereferencable load is in a statement that is modeled precisely we can
  // hoist it.
  if (StmtInvalidCtxIsEmpty && MAInvalidCtxIsEmpty)
    return true;

  // Even if the statement is not modeled precisely we can hoist the load if it
  // does not involve any parameters that might have been specialized by the
  // statement domain.
  for (unsigned u = 0, e = MA->getNumSubscripts(); u < e; u++)
    if (!isa<SCEVConstant>(MA->getSubscript(u)))
      return false;
  return true;
}

void Scop::addInvariantLoads(ScopStmt &Stmt, InvariantAccessesTy &InvMAs) {

  if (InvMAs.empty())
    return;

  auto *StmtInvalidCtx = Stmt.getInvalidContext();
  bool StmtInvalidCtxIsEmpty = isl_set_is_empty(StmtInvalidCtx);

  // Get the context under which the statement is executed but remove the error
  // context under which this statement is reached.
  isl_set *DomainCtx = isl_set_params(Stmt.getDomain());
  DomainCtx = isl_set_subtract(DomainCtx, StmtInvalidCtx);

  if (isl_set_n_basic_set(DomainCtx) >= MaxDisjunctsInDomain) {
    auto *AccInst = InvMAs.front().MA->getAccessInstruction();
    invalidate(COMPLEXITY, AccInst->getDebugLoc());
    isl_set_free(DomainCtx);
    for (auto &InvMA : InvMAs)
      isl_set_free(InvMA.NonHoistableCtx);
    return;
  }

  // Project out all parameters that relate to loads in the statement. Otherwise
  // we could have cyclic dependences on the constraints under which the
  // hoisted loads are executed and we could not determine an order in which to
  // pre-load them. This happens because not only lower bounds are part of the
  // domain but also upper bounds.
  for (auto &InvMA : InvMAs) {
    auto *MA = InvMA.MA;
    Instruction *AccInst = MA->getAccessInstruction();
    if (SE->isSCEVable(AccInst->getType())) {
      SetVector<Value *> Values;
      for (const SCEV *Parameter : Parameters) {
        Values.clear();
        findValues(Parameter, *SE, Values);
        if (!Values.count(AccInst))
          continue;

        if (isl_id *ParamId = getIdForParam(Parameter)) {
          int Dim = isl_set_find_dim_by_id(DomainCtx, isl_dim_param, ParamId);
          if (Dim >= 0)
            DomainCtx = isl_set_eliminate(DomainCtx, isl_dim_param, Dim, 1);
          isl_id_free(ParamId);
        }
      }
    }
  }

  for (auto &InvMA : InvMAs) {
    auto *MA = InvMA.MA;
    auto *NHCtx = InvMA.NonHoistableCtx;

    // Check for another invariant access that accesses the same location as
    // MA and if found consolidate them. Otherwise create a new equivalence
    // class at the end of InvariantEquivClasses.
    LoadInst *LInst = cast<LoadInst>(MA->getAccessInstruction());
    Type *Ty = LInst->getType();
    const SCEV *PointerSCEV = SE->getSCEV(LInst->getPointerOperand());

    auto *MAInvalidCtx = MA->getInvalidContext();
    bool NonHoistableCtxIsEmpty = isl_set_is_empty(NHCtx);
    bool MAInvalidCtxIsEmpty = isl_set_is_empty(MAInvalidCtx);

    isl_set *MACtx;
    // Check if we know that this pointer can be speculatively accessed.
    if (canAlwaysBeHoisted(MA, StmtInvalidCtxIsEmpty, MAInvalidCtxIsEmpty,
                           NonHoistableCtxIsEmpty)) {
      MACtx = isl_set_universe(isl_set_get_space(DomainCtx));
      isl_set_free(MAInvalidCtx);
      isl_set_free(NHCtx);
    } else {
      MACtx = isl_set_copy(DomainCtx);
      MACtx = isl_set_subtract(MACtx, isl_set_union(MAInvalidCtx, NHCtx));
      MACtx = isl_set_gist_params(MACtx, getContext());
    }

    bool Consolidated = false;
    for (auto &IAClass : InvariantEquivClasses) {
      if (PointerSCEV != IAClass.IdentifyingPointer || Ty != IAClass.AccessType)
        continue;

      // If the pointer and the type is equal check if the access function wrt.
      // to the domain is equal too. It can happen that the domain fixes
      // parameter values and these can be different for distinct part of the
      // SCoP. If this happens we cannot consolidate the loads but need to
      // create a new invariant load equivalence class.
      auto &MAs = IAClass.InvariantAccesses;
      if (!MAs.empty()) {
        auto *LastMA = MAs.front();

        auto *AR = isl_map_range(MA->getAccessRelation());
        auto *LastAR = isl_map_range(LastMA->getAccessRelation());
        bool SameAR = isl_set_is_equal(AR, LastAR);
        isl_set_free(AR);
        isl_set_free(LastAR);

        if (!SameAR)
          continue;
      }

      // Add MA to the list of accesses that are in this class.
      MAs.push_front(MA);

      Consolidated = true;

      // Unify the execution context of the class and this statement.
      isl_set *&IAClassDomainCtx = IAClass.ExecutionContext;
      if (IAClassDomainCtx)
        IAClassDomainCtx =
            isl_set_coalesce(isl_set_union(IAClassDomainCtx, MACtx));
      else
        IAClassDomainCtx = MACtx;
      break;
    }

    if (Consolidated)
      continue;

    // If we did not consolidate MA, thus did not find an equivalence class
    // for it, we create a new one.
    InvariantEquivClasses.emplace_back(
        InvariantEquivClassTy{PointerSCEV, MemoryAccessList{MA}, MACtx, Ty});
  }

  isl_set_free(DomainCtx);
}

__isl_give isl_set *Scop::getNonHoistableCtx(MemoryAccess *Access,
                                             __isl_keep isl_union_map *Writes) {
  // TODO: Loads that are not loop carried, hence are in a statement with
  //       zero iterators, are by construction invariant, though we
  //       currently "hoist" them anyway. This is necessary because we allow
  //       them to be treated as parameters (e.g., in conditions) and our code
  //       generation would otherwise use the old value.

  auto &Stmt = *Access->getStatement();
  BasicBlock *BB = Stmt.getEntryBlock();

  if (Access->isScalarKind() || Access->isWrite() || !Access->isAffine() ||
      Access->isMemoryIntrinsic())
    return nullptr;

  // Skip accesses that have an invariant base pointer which is defined but
  // not loaded inside the SCoP. This can happened e.g., if a readnone call
  // returns a pointer that is used as a base address. However, as we want
  // to hoist indirect pointers, we allow the base pointer to be defined in
  // the region if it is also a memory access. Each ScopArrayInfo object
  // that has a base pointer origin has a base pointer that is loaded and
  // that it is invariant, thus it will be hoisted too. However, if there is
  // no base pointer origin we check that the base pointer is defined
  // outside the region.
  auto *LI = cast<LoadInst>(Access->getAccessInstruction());
  if (hasNonHoistableBasePtrInScop(Access, Writes))
    return nullptr;

  auto &DL = getFunction().getParent()->getDataLayout();
  if (isSafeToLoadUnconditionally(LI->getPointerOperand(), LI->getAlignment(),
                                  DL))
    return isl_set_empty(getParamSpace());

  // Skip accesses in non-affine subregions as they might not be executed
  // under the same condition as the entry of the non-affine subregion.
  if (BB != LI->getParent())
    return nullptr;

  isl_map *AccessRelation = Access->getAccessRelation();
  assert(!isl_map_is_empty(AccessRelation));

  if (isl_map_involves_dims(AccessRelation, isl_dim_in, 0,
                            Stmt.getNumIterators())) {
    isl_map_free(AccessRelation);
    return nullptr;
  }

  AccessRelation = isl_map_intersect_domain(AccessRelation, Stmt.getDomain());
  isl_set *AccessRange = isl_map_range(AccessRelation);

  isl_union_map *Written = isl_union_map_intersect_range(
      isl_union_map_copy(Writes), isl_union_set_from_set(AccessRange));
  auto *WrittenCtx = isl_union_map_params(Written);
  bool IsWritten = !isl_set_is_empty(WrittenCtx);

  if (!IsWritten)
    return WrittenCtx;

  WrittenCtx = isl_set_remove_divs(WrittenCtx);
  bool TooComplex = isl_set_n_basic_set(WrittenCtx) >= MaxDisjunctsInDomain;
  if (TooComplex || !isRequiredInvariantLoad(LI)) {
    isl_set_free(WrittenCtx);
    return nullptr;
  }

  addAssumption(INVARIANTLOAD, isl_set_copy(WrittenCtx), LI->getDebugLoc(),
                AS_RESTRICTION);
  return WrittenCtx;
}

void Scop::verifyInvariantLoads() {
  auto &RIL = getRequiredInvariantLoads();
  for (LoadInst *LI : RIL) {
    assert(LI && contains(LI));
    ScopStmt *Stmt = getStmtFor(LI);
    if (Stmt && Stmt->getArrayAccessOrNULLFor(LI)) {
      invalidate(INVARIANTLOAD, LI->getDebugLoc());
      return;
    }
  }
}

void Scop::hoistInvariantLoads() {
  if (!PollyInvariantLoadHoisting)
    return;

  isl_union_map *Writes = getWrites();
  for (ScopStmt &Stmt : *this) {
    InvariantAccessesTy InvariantAccesses;

    for (MemoryAccess *Access : Stmt)
      if (auto *NHCtx = getNonHoistableCtx(Access, Writes))
        InvariantAccesses.push_back({Access, NHCtx});

    // Transfer the memory access from the statement to the SCoP.
    for (auto InvMA : InvariantAccesses)
      Stmt.removeMemoryAccess(InvMA.MA);
    addInvariantLoads(Stmt, InvariantAccesses);
  }
  isl_union_map_free(Writes);
}

const ScopArrayInfo *
Scop::getOrCreateScopArrayInfo(Value *BasePtr, Type *ElementType,
                               ArrayRef<const SCEV *> Sizes, MemoryKind Kind,
                               const char *BaseName) {
  assert((BasePtr || BaseName) &&
         "BasePtr and BaseName can not be nullptr at the same time.");
  assert(!(BasePtr && BaseName) && "BaseName is redundant.");
  auto &SAI = BasePtr ? ScopArrayInfoMap[std::make_pair(BasePtr, Kind)]
                      : ScopArrayNameMap[BaseName];
  if (!SAI) {
    auto &DL = getFunction().getParent()->getDataLayout();
    SAI.reset(new ScopArrayInfo(BasePtr, ElementType, getIslCtx(), Sizes, Kind,
                                DL, this, BaseName));
    ScopArrayInfoSet.insert(SAI.get());
  } else {
    SAI->updateElementType(ElementType);
    // In case of mismatching array sizes, we bail out by setting the run-time
    // context to false.
    if (!SAI->updateSizes(Sizes))
      invalidate(DELINEARIZATION, DebugLoc());
  }
  return SAI.get();
}

const ScopArrayInfo *
Scop::createScopArrayInfo(Type *ElementType, const std::string &BaseName,
                          const std::vector<unsigned> &Sizes) {
  auto *DimSizeType = Type::getInt64Ty(getSE()->getContext());
  std::vector<const SCEV *> SCEVSizes;

  for (auto size : Sizes)
    if (size)
      SCEVSizes.push_back(getSE()->getConstant(DimSizeType, size, false));
    else
      SCEVSizes.push_back(nullptr);

  auto *SAI = getOrCreateScopArrayInfo(nullptr, ElementType, SCEVSizes,
                                       MemoryKind::Array, BaseName.c_str());
  return SAI;
}

const ScopArrayInfo *Scop::getScopArrayInfo(Value *BasePtr, MemoryKind Kind) {
  auto *SAI = ScopArrayInfoMap[std::make_pair(BasePtr, Kind)].get();
  assert(SAI && "No ScopArrayInfo available for this base pointer");
  return SAI;
}

std::string Scop::getContextStr() const { return stringFromIslObj(Context); }

std::string Scop::getAssumedContextStr() const {
  assert(AssumedContext && "Assumed context not yet built");
  return stringFromIslObj(AssumedContext);
}

std::string Scop::getInvalidContextStr() const {
  return stringFromIslObj(InvalidContext);
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
  assert(AssumedContext && "Assumed context not yet built");
  return isl_set_copy(AssumedContext);
}

bool Scop::isProfitable(bool ScalarsAreUnprofitable) const {
  if (PollyProcessUnprofitable)
    return true;

  if (isEmpty())
    return false;

  unsigned OptimizableStmtsOrLoops = 0;
  for (auto &Stmt : *this) {
    if (Stmt.getNumIterators() == 0)
      continue;

    bool ContainsArrayAccs = false;
    bool ContainsScalarAccs = false;
    for (auto *MA : Stmt) {
      if (MA->isRead())
        continue;
      ContainsArrayAccs |= MA->isLatestArrayKind();
      ContainsScalarAccs |= MA->isLatestScalarKind();
    }

    if (!ScalarsAreUnprofitable || (ContainsArrayAccs && !ContainsScalarAccs))
      OptimizableStmtsOrLoops += Stmt.getNumIterators();
  }

  return OptimizableStmtsOrLoops > 1;
}

bool Scop::hasFeasibleRuntimeContext() const {
  auto *PositiveContext = getAssumedContext();
  auto *NegativeContext = getInvalidContext();
  PositiveContext = addNonEmptyDomainConstraints(PositiveContext);
  bool IsFeasible = !(isl_set_is_empty(PositiveContext) ||
                      isl_set_is_subset(PositiveContext, NegativeContext));
  isl_set_free(PositiveContext);
  if (!IsFeasible) {
    isl_set_free(NegativeContext);
    return false;
  }

  auto *DomainContext = isl_union_set_params(getDomains());
  IsFeasible = !isl_set_is_subset(DomainContext, NegativeContext);
  IsFeasible &= !isl_set_is_subset(Context, NegativeContext);
  isl_set_free(NegativeContext);
  isl_set_free(DomainContext);

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
  case UNSIGNED:
    return "Signed-unsigned";
  case COMPLEXITY:
    return "Low complexity";
  case PROFITABLE:
    return "Profitable";
  case ERRORBLOCK:
    return "No-error";
  case INFINITELOOP:
    return "Finite loop";
  case INVARIANTLOAD:
    return "Invariant load";
  case DELINEARIZATION:
    return "Delinearization";
  }
  llvm_unreachable("Unknown AssumptionKind!");
}

bool Scop::isEffectiveAssumption(__isl_keep isl_set *Set, AssumptionSign Sign) {
  if (Sign == AS_ASSUMPTION) {
    if (isl_set_is_subset(Context, Set))
      return false;

    if (isl_set_is_subset(AssumedContext, Set))
      return false;
  } else {
    if (isl_set_is_disjoint(Set, Context))
      return false;

    if (isl_set_is_subset(Set, InvalidContext))
      return false;
  }
  return true;
}

bool Scop::trackAssumption(AssumptionKind Kind, __isl_keep isl_set *Set,
                           DebugLoc Loc, AssumptionSign Sign) {
  if (PollyRemarksMinimal && !isEffectiveAssumption(Set, Sign))
    return false;

  // Do never emit trivial assumptions as they only clutter the output.
  if (!PollyRemarksMinimal) {
    isl_set *Univ = nullptr;
    if (Sign == AS_ASSUMPTION)
      Univ = isl_set_universe(isl_set_get_space(Set));

    bool IsTrivial = (Sign == AS_RESTRICTION && isl_set_is_empty(Set)) ||
                     (Sign == AS_ASSUMPTION && isl_set_is_equal(Univ, Set));
    isl_set_free(Univ);

    if (IsTrivial)
      return false;
  }

  switch (Kind) {
  case ALIASING:
    AssumptionsAliasing++;
    break;
  case INBOUNDS:
    AssumptionsInbounds++;
    break;
  case WRAPPING:
    AssumptionsWrapping++;
    break;
  case UNSIGNED:
    AssumptionsUnsigned++;
    break;
  case COMPLEXITY:
    AssumptionsComplexity++;
    break;
  case PROFITABLE:
    AssumptionsUnprofitable++;
    break;
  case ERRORBLOCK:
    AssumptionsErrorBlock++;
    break;
  case INFINITELOOP:
    AssumptionsInfiniteLoop++;
    break;
  case INVARIANTLOAD:
    AssumptionsInvariantLoad++;
    break;
  case DELINEARIZATION:
    AssumptionsDelinearization++;
    break;
  }

  auto &F = getFunction();
  auto Suffix = Sign == AS_ASSUMPTION ? " assumption:\t" : " restriction:\t";
  std::string Msg = toString(Kind) + Suffix + stringFromIslObj(Set);
  emitOptimizationRemarkAnalysis(F.getContext(), DEBUG_TYPE, F, Loc, Msg);
  return true;
}

void Scop::addAssumption(AssumptionKind Kind, __isl_take isl_set *Set,
                         DebugLoc Loc, AssumptionSign Sign) {
  // Simplify the assumptions/restrictions first.
  Set = isl_set_gist_params(Set, getContext());

  if (!trackAssumption(Kind, Set, Loc, Sign)) {
    isl_set_free(Set);
    return;
  }

  if (Sign == AS_ASSUMPTION) {
    AssumedContext = isl_set_intersect(AssumedContext, Set);
    AssumedContext = isl_set_coalesce(AssumedContext);
  } else {
    InvalidContext = isl_set_union(InvalidContext, Set);
    InvalidContext = isl_set_coalesce(InvalidContext);
  }
}

void Scop::recordAssumption(AssumptionKind Kind, __isl_take isl_set *Set,
                            DebugLoc Loc, AssumptionSign Sign, BasicBlock *BB) {
  assert((isl_set_is_params(Set) || BB) &&
         "Assumptions without a basic block must be parameter sets");
  RecordedAssumptions.push_back({Kind, Sign, Set, Loc, BB});
}

void Scop::addRecordedAssumptions() {
  while (!RecordedAssumptions.empty()) {
    const Assumption &AS = RecordedAssumptions.pop_back_val();

    if (!AS.BB) {
      addAssumption(AS.Kind, AS.Set, AS.Loc, AS.Sign);
      continue;
    }

    // If the domain was deleted the assumptions are void.
    isl_set *Dom = getDomainConditions(AS.BB);
    if (!Dom) {
      isl_set_free(AS.Set);
      continue;
    }

    // If a basic block was given use its domain to simplify the assumption.
    // In case of restrictions we know they only have to hold on the domain,
    // thus we can intersect them with the domain of the block. However, for
    // assumptions the domain has to imply them, thus:
    //                     _              _____
    //   Dom => S   <==>   A v B   <==>   A - B
    //
    // To avoid the complement we will register A - B as a restriction not an
    // assumption.
    isl_set *S = AS.Set;
    if (AS.Sign == AS_RESTRICTION)
      S = isl_set_params(isl_set_intersect(S, Dom));
    else /* (AS.Sign == AS_ASSUMPTION) */
      S = isl_set_params(isl_set_subtract(Dom, S));

    addAssumption(AS.Kind, S, AS.Loc, AS_RESTRICTION);
  }
}

void Scop::invalidate(AssumptionKind Kind, DebugLoc Loc) {
  addAssumption(Kind, isl_set_empty(getParamSpace()), Loc, AS_ASSUMPTION);
}

__isl_give isl_set *Scop::getInvalidContext() const {
  return isl_set_copy(InvalidContext);
}

void Scop::printContext(raw_ostream &OS) const {
  OS << "Context:\n";
  OS.indent(4) << Context << "\n";

  OS.indent(4) << "Assumed Context:\n";
  OS.indent(4) << AssumedContext << "\n";

  OS.indent(4) << "Invalid Context:\n";
  OS.indent(4) << InvalidContext << "\n";

  unsigned Dim = 0;
  for (const SCEV *Parameter : Parameters)
    OS.indent(4) << "p" << Dim++ << ": " << *Parameter << "\n";
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
    Array->print(OS);

  OS.indent(4) << "}\n";

  OS.indent(4) << "Arrays (Bounds as pw_affs) {\n";

  for (auto &Array : arrays())
    Array->print(OS, /* SizeAsPwAff */ true);

  OS.indent(4) << "}\n";
}

void Scop::print(raw_ostream &OS) const {
  OS.indent(4) << "Function: " << getFunction().getName() << "\n";
  OS.indent(4) << "Region: " << getNameStr() << "\n";
  OS.indent(4) << "Max Loop Depth:  " << getMaxLoopDepth() << "\n";
  OS.indent(4) << "Invariant Accesses: {\n";
  for (const auto &IAClass : InvariantEquivClasses) {
    const auto &MAs = IAClass.InvariantAccesses;
    if (MAs.empty()) {
      OS.indent(12) << "Class Pointer: " << *IAClass.IdentifyingPointer << "\n";
    } else {
      MAs.front()->print(OS);
      OS.indent(12) << "Execution Context: " << IAClass.ExecutionContext
                    << "\n";
    }
  }
  OS.indent(4) << "}\n";
  printContext(OS.indent(4));
  printArrayInfo(OS.indent(4));
  printAliasAssumptions(OS);
  printStatements(OS.indent(4));
}

void Scop::dump() const { print(dbgs()); }

isl_ctx *Scop::getIslCtx() const { return IslCtx.get(); }

__isl_give PWACtx Scop::getPwAff(const SCEV *E, BasicBlock *BB,
                                 bool NonNegative) {
  // First try to use the SCEVAffinator to generate a piecewise defined
  // affine function from @p E in the context of @p BB. If that tasks becomes to
  // complex the affinator might return a nullptr. In such a case we invalidate
  // the SCoP and return a dummy value. This way we do not need to add error
  // handling code to all users of this function.
  auto PWAC = Affinator.getPwAff(E, BB);
  if (PWAC.first) {
    // TODO: We could use a heuristic and either use:
    //         SCEVAffinator::takeNonNegativeAssumption
    //       or
    //         SCEVAffinator::interpretAsUnsigned
    //       to deal with unsigned or "NonNegative" SCEVs.
    if (NonNegative)
      Affinator.takeNonNegativeAssumption(PWAC);
    return PWAC;
  }

  auto DL = BB ? BB->getTerminator()->getDebugLoc() : DebugLoc();
  invalidate(COMPLEXITY, DL);
  return Affinator.getPwAff(SE->getZero(E->getType()), BB);
}

__isl_give isl_union_set *Scop::getDomains() const {
  isl_space *EmptySpace = isl_space_params_alloc(getIslCtx(), 0);
  isl_union_set *Domain = isl_union_set_empty(EmptySpace);

  for (const ScopStmt &Stmt : *this)
    Domain = isl_union_set_add_set(Domain, Stmt.getDomain());

  return Domain;
}

__isl_give isl_pw_aff *Scop::getPwAffOnly(const SCEV *E, BasicBlock *BB) {
  PWACtx PWAC = getPwAff(E, BB);
  isl_set_free(PWAC.second);
  return PWAC.first;
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

// Check whether @p Node is an extension node.
//
// @return true if @p Node is an extension node.
isl_bool isNotExtNode(__isl_keep isl_schedule_node *Node, void *User) {
  if (isl_schedule_node_get_type(Node) == isl_schedule_node_extension)
    return isl_bool_error;
  else
    return isl_bool_true;
}

bool Scop::containsExtensionNode(__isl_keep isl_schedule *Schedule) {
  return isl_schedule_foreach_schedule_node_top_down(Schedule, isNotExtNode,
                                                     nullptr) == isl_stat_error;
}

__isl_give isl_union_map *Scop::getSchedule() const {
  auto *Tree = getScheduleTree();
  if (containsExtensionNode(Tree)) {
    isl_schedule_free(Tree);
    return nullptr;
  }
  auto *S = isl_schedule_get_map(Tree);
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

struct MapToDimensionDataTy {
  int N;
  isl_union_pw_multi_aff *Res;
};

// Create a function that maps the elements of 'Set' to its N-th dimension and
// add it to User->Res.
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

// Create an isl_multi_union_aff that defines an identity mapping from the
// elements of USet to their N-th dimension.
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

void Scop::addScopStmt(BasicBlock *BB, Loop *SurroundingLoop) {
  assert(BB && "Unexpected nullptr!");
  Stmts.emplace_back(*this, *BB, SurroundingLoop);
  auto *Stmt = &Stmts.back();
  StmtMap[BB] = Stmt;
}

void Scop::addScopStmt(Region *R, Loop *SurroundingLoop) {
  assert(R && "Unexpected nullptr!");
  Stmts.emplace_back(*this, *R, SurroundingLoop);
  auto *Stmt = &Stmts.back();
  for (BasicBlock *BB : R->blocks())
    StmtMap[BB] = Stmt;
}

ScopStmt *Scop::addScopStmt(__isl_take isl_map *SourceRel,
                            __isl_take isl_map *TargetRel,
                            __isl_take isl_set *Domain) {
#ifndef NDEBUG
  isl_set *SourceDomain = isl_map_domain(isl_map_copy(SourceRel));
  isl_set *TargetDomain = isl_map_domain(isl_map_copy(TargetRel));
  assert(isl_set_is_subset(Domain, TargetDomain) &&
         "Target access not defined for complete statement domain");
  assert(isl_set_is_subset(Domain, SourceDomain) &&
         "Source access not defined for complete statement domain");
  isl_set_free(SourceDomain);
  isl_set_free(TargetDomain);
#endif
  Stmts.emplace_back(*this, SourceRel, TargetRel, Domain);
  CopyStmtsNum++;
  return &(Stmts.back());
}

void Scop::buildSchedule(LoopInfo &LI) {
  Loop *L = getLoopSurroundingScop(*this, LI);
  LoopStackTy LoopStack({LoopStackElementTy(L, nullptr, 0)});
  buildSchedule(getRegion().getNode(), LoopStack, LI);
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
void Scop::buildSchedule(Region *R, LoopStackTy &LoopStack, LoopInfo &LI) {
  Loop *OuterScopLoop = getLoopSurroundingScop(*this, LI);

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
    if (!contains(L))
      L = OuterScopLoop;

    Loop *LastLoop = LoopStack.back().L;
    if (LastLoop != L) {
      if (LastLoop && !LastLoop->contains(L)) {
        LastRNWaiting = true;
        DelayList.push_back(RN);
        continue;
      }
      LoopStack.push_back({L, nullptr, 0});
    }
    buildSchedule(RN, LoopStack, LI);
  }

  return;
}

void Scop::buildSchedule(RegionNode *RN, LoopStackTy &LoopStack, LoopInfo &LI) {

  if (RN->isSubRegion()) {
    auto *LocalRegion = RN->getNodeAs<Region>();
    if (!isNonAffineSubRegion(LocalRegion)) {
      buildSchedule(LocalRegion, LoopStack, LI);
      return;
    }
  }

  auto &LoopData = LoopStack.back();
  LoopData.NumBlocksProcessed += getNumBlocksInRegionNode(RN);

  if (auto *Stmt = getStmtFor(RN)) {
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
         LoopData.NumBlocksProcessed == getNumBlocksInLoop(LoopData.L)) {
    auto *Schedule = LoopData.Schedule;
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

ScopStmt *Scop::getStmtFor(BasicBlock *BB) const {
  auto StmtMapIt = StmtMap.find(BB);
  if (StmtMapIt == StmtMap.end())
    return nullptr;
  return StmtMapIt->second;
}

ScopStmt *Scop::getStmtFor(RegionNode *RN) const {
  if (RN->isSubRegion())
    return getStmtFor(RN->getNodeAs<Region>());
  return getStmtFor(RN->getNodeAs<BasicBlock>());
}

ScopStmt *Scop::getStmtFor(Region *R) const {
  ScopStmt *Stmt = getStmtFor(R->getEntry());
  assert(!Stmt || Stmt->getRegion() == R);
  return Stmt;
}

int Scop::getRelativeLoopDepth(const Loop *L) const {
  Loop *OuterLoop =
      L ? R.outermostLoopInRegion(const_cast<Loop *>(L)) : nullptr;
  if (!OuterLoop)
    return -1;
  return L->getLoopDepth() - OuterLoop->getLoopDepth();
}

ScopArrayInfo *Scop::getArrayInfoByName(const std::string BaseName) {
  for (auto &SAI : arrays()) {
    if (SAI->getName() == BaseName)
      return SAI;
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
void ScopInfoRegionPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<RegionInfoPass>();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequiredTransitive<ScalarEvolutionWrapperPass>();
  AU.addRequiredTransitive<ScopDetection>();
  AU.addRequired<AAResultsWrapperPass>();
  AU.addRequired<AssumptionCacheTracker>();
  AU.setPreservesAll();
}

void updateLoopCountStatistic(ScopDetection::LoopStats Stats) {
  NumLoopsInScop += Stats.NumLoops;
  MaxNumLoopsInScop =
      std::max(MaxNumLoopsInScop.getValue(), (unsigned)Stats.NumLoops);

  if (Stats.MaxDepth == 1)
    NumScopsDepthOne++;
  else if (Stats.MaxDepth == 2)
    NumScopsDepthTwo++;
  else if (Stats.MaxDepth == 3)
    NumScopsDepthThree++;
  else if (Stats.MaxDepth == 4)
    NumScopsDepthFour++;
  else if (Stats.MaxDepth == 5)
    NumScopsDepthFive++;
  else
    NumScopsDepthLarger++;
}

bool ScopInfoRegionPass::runOnRegion(Region *R, RGPassManager &RGM) {
  auto &SD = getAnalysis<ScopDetection>();

  if (!SD.isMaxRegionInScop(*R))
    return false;

  Function *F = R->getEntry()->getParent();
  auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
  auto const &DL = F->getParent()->getDataLayout();
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(*F);

  ScopBuilder SB(R, AC, AA, DL, DT, LI, SD, SE);
  S = SB.getScop(); // take ownership of scop object

  if (S) {
    ScopDetection::LoopStats Stats =
        ScopDetection::countBeneficialLoops(&S->getRegion(), SE, LI, 0);
    updateLoopCountStatistic(Stats);
  }

  return false;
}

void ScopInfoRegionPass::print(raw_ostream &OS, const Module *) const {
  if (S)
    S->print(OS);
  else
    OS << "Invalid Scop!\n";
}

char ScopInfoRegionPass::ID = 0;

Pass *polly::createScopInfoRegionPassPass() { return new ScopInfoRegionPass(); }

INITIALIZE_PASS_BEGIN(ScopInfoRegionPass, "polly-scops",
                      "Polly - Create polyhedral description of Scops", false,
                      false);
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass);
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker);
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass);
INITIALIZE_PASS_DEPENDENCY(RegionInfoPass);
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass);
INITIALIZE_PASS_DEPENDENCY(ScopDetection);
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass);
INITIALIZE_PASS_END(ScopInfoRegionPass, "polly-scops",
                    "Polly - Create polyhedral description of Scops", false,
                    false)

//===----------------------------------------------------------------------===//
void ScopInfoWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<RegionInfoPass>();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequiredTransitive<ScalarEvolutionWrapperPass>();
  AU.addRequiredTransitive<ScopDetection>();
  AU.addRequired<AAResultsWrapperPass>();
  AU.addRequired<AssumptionCacheTracker>();
  AU.setPreservesAll();
}

bool ScopInfoWrapperPass::runOnFunction(Function &F) {
  auto &SD = getAnalysis<ScopDetection>();

  auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
  auto const &DL = F.getParent()->getDataLayout();
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);

  /// Create polyhedral descripton of scops for all the valid regions of a
  /// function.
  for (auto &It : SD) {
    Region *R = const_cast<Region *>(It);
    if (!SD.isMaxRegionInScop(*R))
      continue;

    ScopBuilder SB(R, AC, AA, DL, DT, LI, SD, SE);
    std::unique_ptr<Scop> S = SB.getScop();
    if (!S)
      continue;
    bool Inserted =
        RegionToScopMap.insert(std::make_pair(R, std::move(S))).second;
    assert(Inserted && "Building Scop for the same region twice!");
    (void)Inserted;
  }
  return false;
}

void ScopInfoWrapperPass::print(raw_ostream &OS, const Module *) const {
  for (auto &It : RegionToScopMap) {
    if (It.second)
      It.second->print(OS);
    else
      OS << "Invalid Scop!\n";
  }
}

char ScopInfoWrapperPass::ID = 0;

Pass *polly::createScopInfoWrapperPassPass() {
  return new ScopInfoWrapperPass();
}

INITIALIZE_PASS_BEGIN(
    ScopInfoWrapperPass, "polly-function-scops",
    "Polly - Create polyhedral description of all Scops of a function", false,
    false);
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass);
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker);
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass);
INITIALIZE_PASS_DEPENDENCY(RegionInfoPass);
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass);
INITIALIZE_PASS_DEPENDENCY(ScopDetection);
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass);
INITIALIZE_PASS_END(
    ScopInfoWrapperPass, "polly-function-scops",
    "Polly - Create polyhedral description of all Scops of a function", false,
    false)
