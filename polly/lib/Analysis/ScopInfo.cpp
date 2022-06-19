//===- ScopInfo.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "polly/ScopDetection.h"
#include "polly/Support/GICHelper.h"
#include "polly/Support/ISLOStream.h"
#include "polly/Support/ISLTools.h"
#include "polly/Support/SCEVAffinator.h"
#include "polly/Support/SCEVValidator.h"
#include "polly/Support/ScopHelper.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/RegionIterator.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "isl/aff.h"
#include "isl/local_space.h"
#include "isl/map.h"
#include "isl/options.h"
#include "isl/set.h"
#include <cassert>

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

STATISTIC(NumScops, "Number of feasible SCoPs after ScopInfo");
STATISTIC(NumLoopsInScop, "Number of loops in scops");
STATISTIC(NumBoxedLoops, "Number of boxed loops in SCoPs after ScopInfo");
STATISTIC(NumAffineLoops, "Number of affine loops in SCoPs after ScopInfo");

STATISTIC(NumScopsDepthZero, "Number of scops with maximal loop depth 0");
STATISTIC(NumScopsDepthOne, "Number of scops with maximal loop depth 1");
STATISTIC(NumScopsDepthTwo, "Number of scops with maximal loop depth 2");
STATISTIC(NumScopsDepthThree, "Number of scops with maximal loop depth 3");
STATISTIC(NumScopsDepthFour, "Number of scops with maximal loop depth 4");
STATISTIC(NumScopsDepthFive, "Number of scops with maximal loop depth 5");
STATISTIC(NumScopsDepthLarger,
          "Number of scops with maximal loop depth 6 and larger");
STATISTIC(MaxNumLoopsInScop, "Maximal number of loops in scops");

STATISTIC(NumValueWrites, "Number of scalar value writes after ScopInfo");
STATISTIC(
    NumValueWritesInLoops,
    "Number of scalar value writes nested in affine loops after ScopInfo");
STATISTIC(NumPHIWrites, "Number of scalar phi writes after ScopInfo");
STATISTIC(NumPHIWritesInLoops,
          "Number of scalar phi writes nested in affine loops after ScopInfo");
STATISTIC(NumSingletonWrites, "Number of singleton writes after ScopInfo");
STATISTIC(NumSingletonWritesInLoops,
          "Number of singleton writes nested in affine loops after ScopInfo");

unsigned const polly::MaxDisjunctsInDomain = 20;

// The number of disjunct in the context after which we stop to add more
// disjuncts. This parameter is there to avoid exponential growth in the
// number of disjunct when adding non-convex sets to the context.
static int const MaxDisjunctsInContext = 4;

// Be a bit more generous for the defined behavior context which is used less
// often.
static int const MaxDisjunktsInDefinedBehaviourContext = 8;

static cl::opt<bool> PollyRemarksMinimal(
    "polly-remarks-minimal",
    cl::desc("Do not emit remarks about assumptions that are known"),
    cl::Hidden, cl::cat(PollyCategory));

static cl::opt<bool>
    IslOnErrorAbort("polly-on-isl-error-abort",
                    cl::desc("Abort if an isl error is encountered"),
                    cl::init(true), cl::cat(PollyCategory));

static cl::opt<bool> PollyPreciseInbounds(
    "polly-precise-inbounds",
    cl::desc("Take more precise inbounds assumptions (do not scale well)"),
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

bool polly::UseInstructionNames;

static cl::opt<bool, true> XUseInstructionNames(
    "polly-use-llvm-names",
    cl::desc("Use LLVM-IR names when deriving statement names"),
    cl::location(UseInstructionNames), cl::Hidden, cl::cat(PollyCategory));

static cl::opt<bool> PollyPrintInstructions(
    "polly-print-instructions", cl::desc("Output instructions per ScopStmt"),
    cl::Hidden, cl::Optional, cl::init(false), cl::cat(PollyCategory));

static cl::list<std::string> IslArgs("polly-isl-arg",
                                     cl::value_desc("argument"),
                                     cl::desc("Option passed to ISL"),
                                     cl::cat(PollyCategory));

//===----------------------------------------------------------------------===//

static isl::set addRangeBoundsToSet(isl::set S, const ConstantRange &Range,
                                    int dim, isl::dim type) {
  isl::val V;
  isl::ctx Ctx = S.ctx();

  // The upper and lower bound for a parameter value is derived either from
  // the data type of the parameter or from the - possibly more restrictive -
  // range metadata.
  V = valFromAPInt(Ctx.get(), Range.getSignedMin(), true);
  S = S.lower_bound_val(type, dim, V);
  V = valFromAPInt(Ctx.get(), Range.getSignedMax(), true);
  S = S.upper_bound_val(type, dim, V);

  if (Range.isFullSet())
    return S;

  if (S.n_basic_set().release() > MaxDisjunctsInContext)
    return S;

  // In case of signed wrapping, we can refine the set of valid values by
  // excluding the part not covered by the wrapping range.
  if (Range.isSignWrappedSet()) {
    V = valFromAPInt(Ctx.get(), Range.getLower(), true);
    isl::set SLB = S.lower_bound_val(type, dim, V);

    V = valFromAPInt(Ctx.get(), Range.getUpper(), true);
    V = V.sub(1);
    isl::set SUB = S.upper_bound_val(type, dim, V);
    S = SLB.unite(SUB);
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

ScopArrayInfo::ScopArrayInfo(Value *BasePtr, Type *ElementType, isl::ctx Ctx,
                             ArrayRef<const SCEV *> Sizes, MemoryKind Kind,
                             const DataLayout &DL, Scop *S,
                             const char *BaseName)
    : BasePtr(BasePtr), ElementType(ElementType), Kind(Kind), DL(DL), S(*S) {
  std::string BasePtrName =
      BaseName ? BaseName
               : getIslCompatibleName("MemRef", BasePtr, S->getNextArrayIdx(),
                                      Kind == MemoryKind::PHI ? "__phi" : "",
                                      UseInstructionNames);
  Id = isl::id::alloc(Ctx, BasePtrName, this);

  updateSizes(Sizes);

  if (!BasePtr || Kind != MemoryKind::Array) {
    BasePtrOriginSAI = nullptr;
    return;
  }

  BasePtrOriginSAI = identifyBasePtrOriginSAI(S, BasePtr);
  if (BasePtrOriginSAI)
    const_cast<ScopArrayInfo *>(BasePtrOriginSAI)->addDerivedSAI(this);
}

ScopArrayInfo::~ScopArrayInfo() = default;

isl::space ScopArrayInfo::getSpace() const {
  auto Space = isl::space(Id.ctx(), 0, getNumberOfDimensions());
  Space = Space.set_tuple_id(isl::dim::set, Id);
  return Space;
}

bool ScopArrayInfo::isReadOnly() {
  isl::union_set WriteSet = S.getWrites().range();
  isl::space Space = getSpace();
  WriteSet = WriteSet.extract_set(Space);

  return bool(WriteSet.is_empty());
}

bool ScopArrayInfo::isCompatibleWith(const ScopArrayInfo *Array) const {
  if (Array->getElementType() != getElementType())
    return false;

  if (Array->getNumberOfDimensions() != getNumberOfDimensions())
    return false;

  for (unsigned i = 0; i < getNumberOfDimensions(); i++)
    if (Array->getDimensionSize(i) != getDimensionSize(i))
      return false;

  return true;
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
  DimensionSizesPw.clear();
  for (const SCEV *Expr : DimensionSizes) {
    if (!Expr) {
      DimensionSizesPw.push_back(isl::pw_aff());
      continue;
    }
    isl::pw_aff Size = S.getPwAffOnly(Expr);
    DimensionSizesPw.push_back(Size);
  }
  return true;
}

std::string ScopArrayInfo::getName() const { return Id.get_name(); }

int ScopArrayInfo::getElemSizeInBytes() const {
  return DL.getTypeAllocSize(ElementType);
}

isl::id ScopArrayInfo::getBasePtrId() const { return Id; }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void ScopArrayInfo::dump() const { print(errs()); }
#endif

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
      isl::pw_aff Size = getDimensionSizePw(u);
      OS << " " << Size << " ";
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
ScopArrayInfo::getFromAccessFunction(isl::pw_multi_aff PMA) {
  isl::id Id = PMA.get_tuple_id(isl::dim::out);
  assert(!Id.is_null() && "Output dimension didn't have an ID");
  return getFromId(Id);
}

const ScopArrayInfo *ScopArrayInfo::getFromId(isl::id Id) {
  void *User = Id.get_user();
  const ScopArrayInfo *SAI = static_cast<ScopArrayInfo *>(User);
  return SAI;
}

void MemoryAccess::wrapConstantDimensions() {
  auto *SAI = getScopArrayInfo();
  isl::space ArraySpace = SAI->getSpace();
  isl::ctx Ctx = ArraySpace.ctx();
  unsigned DimsArray = SAI->getNumberOfDimensions();

  isl::multi_aff DivModAff = isl::multi_aff::identity(
      ArraySpace.map_from_domain_and_range(ArraySpace));
  isl::local_space LArraySpace = isl::local_space(ArraySpace);

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

    isl::val DimSizeVal =
        valFromAPInt(Ctx.get(), DimSizeCst->getAPInt(), false);
    isl::aff Var = isl::aff::var_on_domain(LArraySpace, isl::dim::set, i);
    isl::aff PrevVar =
        isl::aff::var_on_domain(LArraySpace, isl::dim::set, i - 1);

    // Compute: index % size
    // Modulo must apply in the divide of the previous iteration, if any.
    isl::aff Modulo = Var.mod(DimSizeVal);
    Modulo = Modulo.pullback(DivModAff);

    // Compute: floor(index / size)
    isl::aff Divide = Var.div(isl::aff(LArraySpace, DimSizeVal));
    Divide = Divide.floor();
    Divide = Divide.add(PrevVar);
    Divide = Divide.pullback(DivModAff);

    // Apply Modulo and Divide.
    DivModAff = DivModAff.set_aff(i, Modulo);
    DivModAff = DivModAff.set_aff(i - 1, Divide);
  }

  // Apply all modulo/divides on the accesses.
  isl::map Relation = AccessRelation;
  Relation = Relation.apply_range(isl::map::from_multi_aff(DivModAff));
  Relation = Relation.detect_equalities();
  AccessRelation = Relation;
}

void MemoryAccess::updateDimensionality() {
  auto *SAI = getScopArrayInfo();
  isl::space ArraySpace = SAI->getSpace();
  isl::space AccessSpace = AccessRelation.get_space().range();
  isl::ctx Ctx = ArraySpace.ctx();

  unsigned DimsArray = unsignedFromIslSize(ArraySpace.dim(isl::dim::set));
  unsigned DimsAccess = unsignedFromIslSize(AccessSpace.dim(isl::dim::set));
  assert(DimsArray >= DimsAccess);
  unsigned DimsMissing = DimsArray - DimsAccess;

  auto *BB = getStatement()->getEntryBlock();
  auto &DL = BB->getModule()->getDataLayout();
  unsigned ArrayElemSize = SAI->getElemSizeInBytes();
  unsigned ElemBytes = DL.getTypeAllocSize(getElementType());

  isl::map Map = isl::map::from_domain_and_range(
      isl::set::universe(AccessSpace), isl::set::universe(ArraySpace));

  for (auto i : seq<unsigned>(0, DimsMissing))
    Map = Map.fix_si(isl::dim::out, i, 0);

  for (auto i : seq<unsigned>(DimsMissing, DimsArray))
    Map = Map.equate(isl::dim::in, i - DimsMissing, isl::dim::out, i);

  AccessRelation = AccessRelation.apply_range(Map);

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
    isl::val V = isl::val(Ctx, ArrayElemSize);
    AccessRelation = AccessRelation.floordiv_val(V);
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
    assert(DimsArray >= 1);
    isl::map Map = isl::map::from_domain_and_range(
        isl::set::universe(ArraySpace), isl::set::universe(ArraySpace));
    for (auto i : seq<unsigned>(0, DimsArray - 1))
      Map = Map.equate(isl::dim::in, i, isl::dim::out, i);

    isl::constraint C;
    isl::local_space LS;

    LS = isl::local_space(Map.get_space());
    int Num = ElemBytes / getScopArrayInfo()->getElemSizeInBytes();

    C = isl::constraint::alloc_inequality(LS);
    C = C.set_constant_val(isl::val(Ctx, Num - 1));
    C = C.set_coefficient_si(isl::dim::in, DimsArray - 1, 1);
    C = C.set_coefficient_si(isl::dim::out, DimsArray - 1, -1);
    Map = Map.add_constraint(C);

    C = isl::constraint::alloc_inequality(LS);
    C = C.set_coefficient_si(isl::dim::in, DimsArray - 1, -1);
    C = C.set_coefficient_si(isl::dim::out, DimsArray - 1, 1);
    C = C.set_constant_val(isl::val(Ctx, 0));
    Map = Map.add_constraint(C);
    AccessRelation = AccessRelation.apply_range(Map);
  }
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
}

const ScopArrayInfo *MemoryAccess::getOriginalScopArrayInfo() const {
  isl::id ArrayId = getArrayId();
  void *User = ArrayId.get_user();
  const ScopArrayInfo *SAI = static_cast<ScopArrayInfo *>(User);
  return SAI;
}

const ScopArrayInfo *MemoryAccess::getLatestScopArrayInfo() const {
  isl::id ArrayId = getLatestArrayId();
  void *User = ArrayId.get_user();
  const ScopArrayInfo *SAI = static_cast<ScopArrayInfo *>(User);
  return SAI;
}

isl::id MemoryAccess::getOriginalArrayId() const {
  return AccessRelation.get_tuple_id(isl::dim::out);
}

isl::id MemoryAccess::getLatestArrayId() const {
  if (!hasNewAccessRelation())
    return getOriginalArrayId();
  return NewAccessRelation.get_tuple_id(isl::dim::out);
}

isl::map MemoryAccess::getAddressFunction() const {
  return getAccessRelation().lexmin();
}

isl::pw_multi_aff
MemoryAccess::applyScheduleToAccessRelation(isl::union_map USchedule) const {
  isl::map Schedule, ScheduledAccRel;
  isl::union_set UDomain;

  UDomain = getStatement()->getDomain();
  USchedule = USchedule.intersect_domain(UDomain);
  Schedule = isl::map::from_union_map(USchedule);
  ScheduledAccRel = getAddressFunction().apply_domain(Schedule);
  return isl::pw_multi_aff::from_map(ScheduledAccRel);
}

isl::map MemoryAccess::getOriginalAccessRelation() const {
  return AccessRelation;
}

std::string MemoryAccess::getOriginalAccessRelationStr() const {
  return stringFromIslObj(AccessRelation);
}

isl::space MemoryAccess::getOriginalAccessRelationSpace() const {
  return AccessRelation.get_space();
}

isl::map MemoryAccess::getNewAccessRelation() const {
  return NewAccessRelation;
}

std::string MemoryAccess::getNewAccessRelationStr() const {
  return stringFromIslObj(NewAccessRelation);
}

std::string MemoryAccess::getAccessRelationStr() const {
  return stringFromIslObj(getAccessRelation());
}

isl::basic_map MemoryAccess::createBasicAccessMap(ScopStmt *Statement) {
  isl::space Space = isl::space(Statement->getIslCtx(), 0, 1);
  Space = Space.align_params(Statement->getDomainSpace());

  return isl::basic_map::from_domain_and_range(
      isl::basic_set::universe(Statement->getDomainSpace()),
      isl::basic_set::universe(Space));
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
isl::set MemoryAccess::assumeNoOutOfBound() {
  auto *SAI = getScopArrayInfo();
  isl::space Space = getOriginalAccessRelationSpace().range();
  isl::set Outside = isl::set::empty(Space);
  for (int i = 1, Size = Space.dim(isl::dim::set).release(); i < Size; ++i) {
    isl::local_space LS(Space);
    isl::pw_aff Var = isl::pw_aff::var_on_domain(LS, isl::dim::set, i);
    isl::pw_aff Zero = isl::pw_aff(LS);

    isl::set DimOutside = Var.lt_set(Zero);
    isl::pw_aff SizeE = SAI->getDimensionSizePw(i);
    SizeE = SizeE.add_dims(isl::dim::in, Space.dim(isl::dim::set).release());
    SizeE = SizeE.set_tuple_id(isl::dim::in, Space.get_tuple_id(isl::dim::set));
    DimOutside = DimOutside.unite(SizeE.le_set(Var));

    Outside = Outside.unite(DimOutside);
  }

  Outside = Outside.apply(getAccessRelation().reverse());
  Outside = Outside.intersect(Statement->getDomain());
  Outside = Outside.params();

  // Remove divs to avoid the construction of overly complicated assumptions.
  // Doing so increases the set of parameter combinations that are assumed to
  // not appear. This is always save, but may make the resulting run-time check
  // bail out more often than strictly necessary.
  Outside = Outside.remove_divs();
  Outside = Outside.complement();

  if (!PollyPreciseInbounds)
    Outside = Outside.gist_params(Statement->getDomain().params());
  return Outside;
}

void MemoryAccess::buildMemIntrinsicAccessRelation() {
  assert(isMemoryIntrinsic());
  assert(Subscripts.size() == 2 && Sizes.size() == 1);

  isl::pw_aff SubscriptPWA = getPwAff(Subscripts[0]);
  isl::map SubscriptMap = isl::map::from_pw_aff(SubscriptPWA);

  isl::map LengthMap;
  if (Subscripts[1] == nullptr) {
    LengthMap = isl::map::universe(SubscriptMap.get_space());
  } else {
    isl::pw_aff LengthPWA = getPwAff(Subscripts[1]);
    LengthMap = isl::map::from_pw_aff(LengthPWA);
    isl::space RangeSpace = LengthMap.get_space().range();
    LengthMap = LengthMap.apply_range(isl::map::lex_gt(RangeSpace));
  }
  LengthMap = LengthMap.lower_bound_si(isl::dim::out, 0, 0);
  LengthMap = LengthMap.align_params(SubscriptMap.get_space());
  SubscriptMap = SubscriptMap.align_params(LengthMap.get_space());
  LengthMap = LengthMap.sum(SubscriptMap);
  AccessRelation =
      LengthMap.set_tuple_id(isl::dim::in, getStatement()->getDomainId());
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

  if (Range.isUpperWrapped() || Range.isSignWrappedSet())
    return;

  bool isWrapping = Range.isSignWrappedSet();

  unsigned BW = Range.getBitWidth();
  const auto One = APInt(BW, 1);
  const auto LB = isWrapping ? Range.getLower() : Range.getSignedMin();
  const auto UB = isWrapping ? (Range.getUpper() - One) : Range.getSignedMax();

  auto Min = LB.sdiv(APInt(BW, ElementSize));
  auto Max = UB.sdiv(APInt(BW, ElementSize)) + One;

  assert(Min.sle(Max) && "Minimum expected to be less or equal than max");

  isl::map Relation = AccessRelation;
  isl::set AccessRange = Relation.range();
  AccessRange = addRangeBoundsToSet(AccessRange, ConstantRange(Min, Max), 0,
                                    isl::dim::set);
  AccessRelation = Relation.intersect_range(AccessRange);
}

void MemoryAccess::foldAccessRelation() {
  if (Sizes.size() < 2 || isa<SCEVConstant>(Sizes[1]))
    return;

  int Size = Subscripts.size();

  isl::map NewAccessRelation = AccessRelation;

  for (int i = Size - 2; i >= 0; --i) {
    isl::space Space;
    isl::map MapOne, MapTwo;
    isl::pw_aff DimSize = getPwAff(Sizes[i + 1]);

    isl::space SpaceSize = DimSize.get_space();
    isl::id ParamId = SpaceSize.get_dim_id(isl::dim::param, 0);

    Space = AccessRelation.get_space();
    Space = Space.range().map_from_set();
    Space = Space.align_params(SpaceSize);

    int ParamLocation = Space.find_dim_by_id(isl::dim::param, ParamId);

    MapOne = isl::map::universe(Space);
    for (int j = 0; j < Size; ++j)
      MapOne = MapOne.equate(isl::dim::in, j, isl::dim::out, j);
    MapOne = MapOne.lower_bound_si(isl::dim::in, i + 1, 0);

    MapTwo = isl::map::universe(Space);
    for (int j = 0; j < Size; ++j)
      if (j < i || j > i + 1)
        MapTwo = MapTwo.equate(isl::dim::in, j, isl::dim::out, j);

    isl::local_space LS(Space);
    isl::constraint C;
    C = isl::constraint::alloc_equality(LS);
    C = C.set_constant_si(-1);
    C = C.set_coefficient_si(isl::dim::in, i, 1);
    C = C.set_coefficient_si(isl::dim::out, i, -1);
    MapTwo = MapTwo.add_constraint(C);
    C = isl::constraint::alloc_equality(LS);
    C = C.set_coefficient_si(isl::dim::in, i + 1, 1);
    C = C.set_coefficient_si(isl::dim::out, i + 1, -1);
    C = C.set_coefficient_si(isl::dim::param, ParamLocation, 1);
    MapTwo = MapTwo.add_constraint(C);
    MapTwo = MapTwo.upper_bound_si(isl::dim::in, i + 1, -1);

    MapOne = MapOne.unite(MapTwo);
    NewAccessRelation = NewAccessRelation.apply_range(MapOne);
  }

  isl::id BaseAddrId = getScopArrayInfo()->getBasePtrId();
  isl::space Space = Statement->getDomainSpace();
  NewAccessRelation = NewAccessRelation.set_tuple_id(
      isl::dim::in, Space.get_tuple_id(isl::dim::set));
  NewAccessRelation = NewAccessRelation.set_tuple_id(isl::dim::out, BaseAddrId);
  NewAccessRelation = NewAccessRelation.gist_domain(Statement->getDomain());

  // Access dimension folding might in certain cases increase the number of
  // disjuncts in the memory access, which can possibly complicate the generated
  // run-time checks and can lead to costly compilation.
  if (!PollyPreciseFoldAccesses && NewAccessRelation.n_basic_map().release() >
                                       AccessRelation.n_basic_map().release()) {
  } else {
    AccessRelation = NewAccessRelation;
  }
}

void MemoryAccess::buildAccessRelation(const ScopArrayInfo *SAI) {
  assert(AccessRelation.is_null() && "AccessRelation already built");

  // Initialize the invalid domain which describes all iterations for which the
  // access relation is not modeled correctly.
  isl::set StmtInvalidDomain = getStatement()->getInvalidDomain();
  InvalidDomain = isl::set::empty(StmtInvalidDomain.get_space());

  isl::ctx Ctx = Id.ctx();
  isl::id BaseAddrId = SAI->getBasePtrId();

  if (getAccessInstruction() && isa<MemIntrinsic>(getAccessInstruction())) {
    buildMemIntrinsicAccessRelation();
    AccessRelation = AccessRelation.set_tuple_id(isl::dim::out, BaseAddrId);
    return;
  }

  if (!isAffine()) {
    // We overapproximate non-affine accesses with a possible access to the
    // whole array. For read accesses it does not make a difference, if an
    // access must or may happen. However, for write accesses it is important to
    // differentiate between writes that must happen and writes that may happen.
    if (AccessRelation.is_null())
      AccessRelation = createBasicAccessMap(Statement);

    AccessRelation = AccessRelation.set_tuple_id(isl::dim::out, BaseAddrId);
    return;
  }

  isl::space Space = isl::space(Ctx, 0, Statement->getNumIterators(), 0);
  AccessRelation = isl::map::universe(Space);

  for (int i = 0, Size = Subscripts.size(); i < Size; ++i) {
    isl::pw_aff Affine = getPwAff(Subscripts[i]);
    isl::map SubscriptMap = isl::map::from_pw_aff(Affine);
    AccessRelation = AccessRelation.flat_range_product(SubscriptMap);
  }

  Space = Statement->getDomainSpace();
  AccessRelation = AccessRelation.set_tuple_id(
      isl::dim::in, Space.get_tuple_id(isl::dim::set));
  AccessRelation = AccessRelation.set_tuple_id(isl::dim::out, BaseAddrId);

  AccessRelation = AccessRelation.gist_domain(Statement->getDomain());
}

MemoryAccess::MemoryAccess(ScopStmt *Stmt, Instruction *AccessInst,
                           AccessType AccType, Value *BaseAddress,
                           Type *ElementType, bool Affine,
                           ArrayRef<const SCEV *> Subscripts,
                           ArrayRef<const SCEV *> Sizes, Value *AccessValue,
                           MemoryKind Kind)
    : Kind(Kind), AccType(AccType), Statement(Stmt), InvalidDomain(),
      BaseAddr(BaseAddress), ElementType(ElementType),
      Sizes(Sizes.begin(), Sizes.end()), AccessInstruction(AccessInst),
      AccessValue(AccessValue), IsAffine(Affine),
      Subscripts(Subscripts.begin(), Subscripts.end()), AccessRelation(),
      NewAccessRelation() {
  static const std::string TypeStrings[] = {"", "_Read", "_Write", "_MayWrite"};
  const std::string Access = TypeStrings[AccType] + utostr(Stmt->size());

  std::string IdName = Stmt->getBaseName() + Access;
  Id = isl::id::alloc(Stmt->getParent()->getIslCtx(), IdName, this);
}

MemoryAccess::MemoryAccess(ScopStmt *Stmt, AccessType AccType, isl::map AccRel)
    : Kind(MemoryKind::Array), AccType(AccType), Statement(Stmt),
      InvalidDomain(), AccessRelation(), NewAccessRelation(AccRel) {
  isl::id ArrayInfoId = NewAccessRelation.get_tuple_id(isl::dim::out);
  auto *SAI = ScopArrayInfo::getFromId(ArrayInfoId);
  Sizes.push_back(nullptr);
  for (unsigned i = 1; i < SAI->getNumberOfDimensions(); i++)
    Sizes.push_back(SAI->getDimensionSize(i));
  ElementType = SAI->getElementType();
  BaseAddr = SAI->getBasePtr();
  static const std::string TypeStrings[] = {"", "_Read", "_Write", "_MayWrite"};
  const std::string Access = TypeStrings[AccType] + utostr(Stmt->size());

  std::string IdName = Stmt->getBaseName() + Access;
  Id = isl::id::alloc(Stmt->getParent()->getIslCtx(), IdName, this);
}

MemoryAccess::~MemoryAccess() = default;

void MemoryAccess::realignParams() {
  isl::set Ctx = Statement->getParent()->getContext();
  InvalidDomain = InvalidDomain.gist_params(Ctx);
  AccessRelation = AccessRelation.gist_params(Ctx);

  // Predictable parameter order is required for JSON imports. Ensure alignment
  // by explicitly calling align_params.
  isl::space CtxSpace = Ctx.get_space();
  InvalidDomain = InvalidDomain.align_params(CtxSpace);
  AccessRelation = AccessRelation.align_params(CtxSpace);
}

const std::string MemoryAccess::getReductionOperatorStr() const {
  return MemoryAccess::getReductionOperatorStr(getReductionType());
}

isl::id MemoryAccess::getId() const { return Id; }

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

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void MemoryAccess::dump() const { print(errs()); }
#endif

isl::pw_aff MemoryAccess::getPwAff(const SCEV *E) {
  auto *Stmt = getStatement();
  PWACtx PWAC = Stmt->getParent()->getPwAff(E, Stmt->getEntryBlock());
  isl::set StmtDom = getStatement()->getDomain();
  StmtDom = StmtDom.reset_tuple_id();
  isl::set NewInvalidDom = StmtDom.intersect(PWAC.second);
  InvalidDomain = InvalidDomain.unite(NewInvalidDom);
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
static isl::map getEqualAndLarger(isl::space SetDomain) {
  isl::space Space = SetDomain.map_from_set();
  isl::map Map = isl::map::universe(Space);
  unsigned lastDimension = Map.domain_tuple_dim().release() - 1;

  // Set all but the last dimension to be equal for the input and output
  //
  //   input[i0, i1, ..., iX] -> output[o0, o1, ..., oX]
  //     : i0 = o0, i1 = o1, ..., i(X-1) = o(X-1)
  for (unsigned i = 0; i < lastDimension; ++i)
    Map = Map.equate(isl::dim::in, i, isl::dim::out, i);

  // Set the last dimension of the input to be strict smaller than the
  // last dimension of the output.
  //
  //   input[?,?,?,...,iX] -> output[?,?,?,...,oX] : iX < oX
  Map = Map.order_lt(isl::dim::in, lastDimension, isl::dim::out, lastDimension);
  return Map;
}

isl::set MemoryAccess::getStride(isl::map Schedule) const {
  isl::map AccessRelation = getAccessRelation();
  isl::space Space = Schedule.get_space().range();
  isl::map NextScatt = getEqualAndLarger(Space);

  Schedule = Schedule.reverse();
  NextScatt = NextScatt.lexmin();

  NextScatt = NextScatt.apply_range(Schedule);
  NextScatt = NextScatt.apply_range(AccessRelation);
  NextScatt = NextScatt.apply_domain(Schedule);
  NextScatt = NextScatt.apply_domain(AccessRelation);

  isl::set Deltas = NextScatt.deltas();
  return Deltas;
}

bool MemoryAccess::isStrideX(isl::map Schedule, int StrideWidth) const {
  isl::set Stride, StrideX;
  bool IsStrideX;

  Stride = getStride(Schedule);
  StrideX = isl::set::universe(Stride.get_space());
  int Size = unsignedFromIslSize(StrideX.tuple_dim());
  for (auto i : seq<int>(0, Size - 1))
    StrideX = StrideX.fix_si(isl::dim::set, i, 0);
  StrideX = StrideX.fix_si(isl::dim::set, Size - 1, StrideWidth);
  IsStrideX = Stride.is_subset(StrideX);

  return IsStrideX;
}

bool MemoryAccess::isStrideZero(isl::map Schedule) const {
  return isStrideX(Schedule, 0);
}

bool MemoryAccess::isStrideOne(isl::map Schedule) const {
  return isStrideX(Schedule, 1);
}

void MemoryAccess::setAccessRelation(isl::map NewAccess) {
  AccessRelation = NewAccess;
}

void MemoryAccess::setNewAccessRelation(isl::map NewAccess) {
  assert(!NewAccess.is_null());

#ifndef NDEBUG
  // Check domain space compatibility.
  isl::space NewSpace = NewAccess.get_space();
  isl::space NewDomainSpace = NewSpace.domain();
  isl::space OriginalDomainSpace = getStatement()->getDomainSpace();
  assert(OriginalDomainSpace.has_equal_tuples(NewDomainSpace));

  // Reads must be executed unconditionally. Writes might be executed in a
  // subdomain only.
  if (isRead()) {
    // Check whether there is an access for every statement instance.
    isl::set StmtDomain = getStatement()->getDomain();
    isl::set DefinedContext =
        getStatement()->getParent()->getBestKnownDefinedBehaviorContext();
    StmtDomain = StmtDomain.intersect_params(DefinedContext);
    isl::set NewDomain = NewAccess.domain();
    assert(!StmtDomain.is_subset(NewDomain).is_false() &&
           "Partial READ accesses not supported");
  }

  isl::space NewAccessSpace = NewAccess.get_space();
  assert(NewAccessSpace.has_tuple_id(isl::dim::set) &&
         "Must specify the array that is accessed");
  isl::id NewArrayId = NewAccessSpace.get_tuple_id(isl::dim::set);
  auto *SAI = static_cast<ScopArrayInfo *>(NewArrayId.get_user());
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
  unsigned Dims = SAI->getNumberOfDimensions();
  unsigned SpaceSize = unsignedFromIslSize(NewAccessSpace.dim(isl::dim::set));
  assert(SpaceSize == Dims && "Access dims must match array dims");
#endif

  NewAccess = NewAccess.gist_params(getStatement()->getParent()->getContext());
  NewAccess = NewAccess.gist_domain(getStatement()->getDomain());
  NewAccessRelation = NewAccess;
}

bool MemoryAccess::isLatestPartialAccess() const {
  isl::set StmtDom = getStatement()->getDomain();
  isl::set AccDom = getLatestAccessRelation().domain();

  return !StmtDom.is_subset(AccDom);
}

//===----------------------------------------------------------------------===//

isl::map ScopStmt::getSchedule() const {
  isl::set Domain = getDomain();
  if (Domain.is_empty())
    return isl::map::from_aff(isl::aff(isl::local_space(getDomainSpace())));
  auto Schedule = getParent()->getSchedule();
  if (Schedule.is_null())
    return {};
  Schedule = Schedule.intersect_domain(isl::union_set(Domain));
  if (Schedule.is_empty())
    return isl::map::from_aff(isl::aff(isl::local_space(getDomainSpace())));
  isl::map M = M.from_union_map(Schedule);
  M = M.coalesce();
  M = M.gist_domain(Domain);
  M = M.coalesce();
  return M;
}

void ScopStmt::restrictDomain(isl::set NewDomain) {
  assert(NewDomain.is_subset(Domain) &&
         "New domain is not a subset of old domain!");
  Domain = NewDomain;
}

void ScopStmt::addAccess(MemoryAccess *Access, bool Prepend) {
  Instruction *AccessInst = Access->getAccessInstruction();

  if (Access->isArrayKind()) {
    MemoryAccessList &MAL = InstructionToAccess[AccessInst];
    MAL.emplace_front(Access);
  } else if (Access->isValueKind() && Access->isWrite()) {
    Instruction *AccessVal = cast<Instruction>(Access->getAccessValue());
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
  } else if (Access->isAnyPHIKind() && Access->isRead()) {
    PHINode *PHI = cast<PHINode>(Access->getAccessValue());
    assert(!PHIReads.lookup(PHI));

    PHIReads[PHI] = Access;
  }

  if (Prepend) {
    MemAccs.insert(MemAccs.begin(), Access);
    return;
  }
  MemAccs.push_back(Access);
}

void ScopStmt::realignParams() {
  for (MemoryAccess *MA : *this)
    MA->realignParams();

  simplify(InvalidDomain);
  simplify(Domain);

  isl::set Ctx = Parent.getContext();
  InvalidDomain = InvalidDomain.gist_params(Ctx);
  Domain = Domain.gist_params(Ctx);

  // Predictable parameter order is required for JSON imports. Ensure alignment
  // by explicitly calling align_params.
  isl::space CtxSpace = Ctx.get_space();
  InvalidDomain = InvalidDomain.align_params(CtxSpace);
  Domain = Domain.align_params(CtxSpace);
}

ScopStmt::ScopStmt(Scop &parent, Region &R, StringRef Name,
                   Loop *SurroundingLoop,
                   std::vector<Instruction *> EntryBlockInstructions)
    : Parent(parent), InvalidDomain(), Domain(), R(&R), Build(), BaseName(Name),
      SurroundingLoop(SurroundingLoop), Instructions(EntryBlockInstructions) {}

ScopStmt::ScopStmt(Scop &parent, BasicBlock &bb, StringRef Name,
                   Loop *SurroundingLoop,
                   std::vector<Instruction *> Instructions)
    : Parent(parent), InvalidDomain(), Domain(), BB(&bb), Build(),
      BaseName(Name), SurroundingLoop(SurroundingLoop),
      Instructions(Instructions) {}

ScopStmt::ScopStmt(Scop &parent, isl::map SourceRel, isl::map TargetRel,
                   isl::set NewDomain)
    : Parent(parent), InvalidDomain(), Domain(NewDomain), Build() {
  BaseName = getIslCompatibleName("CopyStmt_", "",
                                  std::to_string(parent.getCopyStmtsNum()));
  isl::id Id = isl::id::alloc(getIslCtx(), getBaseName(), this);
  Domain = Domain.set_tuple_id(Id);
  TargetRel = TargetRel.set_tuple_id(isl::dim::in, Id);
  auto *Access =
      new MemoryAccess(this, MemoryAccess::AccessType::MUST_WRITE, TargetRel);
  parent.addAccessFunction(Access);
  addAccess(Access);
  SourceRel = SourceRel.set_tuple_id(isl::dim::in, Id);
  Access = new MemoryAccess(this, MemoryAccess::AccessType::READ, SourceRel);
  parent.addAccessFunction(Access);
  addAccess(Access);
}

ScopStmt::~ScopStmt() = default;

std::string ScopStmt::getDomainStr() const { return stringFromIslObj(Domain); }

std::string ScopStmt::getScheduleStr() const {
  return stringFromIslObj(getSchedule());
}

void ScopStmt::setInvalidDomain(isl::set ID) { InvalidDomain = ID; }

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

isl::ctx ScopStmt::getIslCtx() const { return Parent.getIslCtx(); }

isl::set ScopStmt::getDomain() const { return Domain; }

isl::space ScopStmt::getDomainSpace() const { return Domain.get_space(); }

isl::id ScopStmt::getDomainId() const { return Domain.get_tuple_id(); }

void ScopStmt::printInstructions(raw_ostream &OS) const {
  OS << "Instructions {\n";

  for (Instruction *Inst : Instructions)
    OS.indent(16) << *Inst << "\n";

  OS.indent(12) << "}\n";
}

void ScopStmt::print(raw_ostream &OS, bool PrintInstructions) const {
  OS << "\t" << getBaseName() << "\n";
  OS.indent(12) << "Domain :=\n";

  if (!Domain.is_null()) {
    OS.indent(16) << getDomainStr() << ";\n";
  } else
    OS.indent(16) << "n/a\n";

  OS.indent(12) << "Schedule :=\n";

  if (!Domain.is_null()) {
    OS.indent(16) << getScheduleStr() << ";\n";
  } else
    OS.indent(16) << "n/a\n";

  for (MemoryAccess *Access : MemAccs)
    Access->print(OS);

  if (PrintInstructions)
    printInstructions(OS.indent(12));
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void ScopStmt::dump() const { print(dbgs(), true); }
#endif

void ScopStmt::removeAccessData(MemoryAccess *MA) {
  if (MA->isRead() && MA->isOriginalValueKind()) {
    bool Found = ValueReads.erase(MA->getAccessValue());
    (void)Found;
    assert(Found && "Expected access data not found");
  }
  if (MA->isWrite() && MA->isOriginalValueKind()) {
    bool Found = ValueWrites.erase(cast<Instruction>(MA->getAccessValue()));
    (void)Found;
    assert(Found && "Expected access data not found");
  }
  if (MA->isWrite() && MA->isOriginalAnyPHIKind()) {
    bool Found = PHIWrites.erase(cast<PHINode>(MA->getAccessInstruction()));
    (void)Found;
    assert(Found && "Expected access data not found");
  }
  if (MA->isRead() && MA->isOriginalAnyPHIKind()) {
    bool Found = PHIReads.erase(cast<PHINode>(MA->getAccessInstruction()));
    (void)Found;
    assert(Found && "Expected access data not found");
  }
}

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
  for (auto *MA : MemAccs) {
    if (Predicate(MA)) {
      removeAccessData(MA);
      Parent.removeAccessData(MA);
    }
  }
  llvm::erase_if(MemAccs, Predicate);
  InstructionToAccess.erase(MA->getAccessInstruction());
}

void ScopStmt::removeSingleMemoryAccess(MemoryAccess *MA, bool AfterHoisting) {
  if (AfterHoisting) {
    auto MAIt = std::find(MemAccs.begin(), MemAccs.end(), MA);
    assert(MAIt != MemAccs.end());
    MemAccs.erase(MAIt);

    removeAccessData(MA);
    Parent.removeAccessData(MA);
  }

  auto It = InstructionToAccess.find(MA->getAccessInstruction());
  if (It != InstructionToAccess.end()) {
    It->second.remove(MA);
    if (It->second.empty())
      InstructionToAccess.erase(MA->getAccessInstruction());
  }
}

MemoryAccess *ScopStmt::ensureValueRead(Value *V) {
  MemoryAccess *Access = lookupInputAccessOf(V);
  if (Access)
    return Access;

  ScopArrayInfo *SAI =
      Parent.getOrCreateScopArrayInfo(V, V->getType(), {}, MemoryKind::Value);
  Access = new MemoryAccess(this, nullptr, MemoryAccess::READ, V, V->getType(),
                            true, {}, {}, V, MemoryKind::Value);
  Parent.addAccessFunction(Access);
  Access->buildAccessRelation(SAI);
  addAccess(Access);
  Parent.addAccessData(Access);
  return Access;
}

raw_ostream &polly::operator<<(raw_ostream &OS, const ScopStmt &S) {
  S.print(OS, PollyPrintInstructions);
  return OS;
}

//===----------------------------------------------------------------------===//
/// Scop class implement

void Scop::setContext(isl::set NewContext) {
  Context = NewContext.align_params(Context.get_space());
}

namespace {

/// Remap parameter values but keep AddRecs valid wrt. invariant loads.
class SCEVSensitiveParameterRewriter final
    : public SCEVRewriteVisitor<SCEVSensitiveParameterRewriter> {
  const ValueToValueMap &VMap;

public:
  SCEVSensitiveParameterRewriter(const ValueToValueMap &VMap,
                                 ScalarEvolution &SE)
      : SCEVRewriteVisitor(SE), VMap(VMap) {}

  static const SCEV *rewrite(const SCEV *E, ScalarEvolution &SE,
                             const ValueToValueMap &VMap) {
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

/// Check whether we should remap a SCEV expression.
class SCEVFindInsideScop : public SCEVTraversal<SCEVFindInsideScop> {
  const ValueToValueMap &VMap;
  bool FoundInside = false;
  const Scop *S;

public:
  SCEVFindInsideScop(const ValueToValueMap &VMap, ScalarEvolution &SE,
                     const Scop *S)
      : SCEVTraversal(*this), VMap(VMap), S(S) {}

  static bool hasVariant(const SCEV *E, ScalarEvolution &SE,
                         const ValueToValueMap &VMap, const Scop *S) {
    SCEVFindInsideScop SFIS(VMap, SE, S);
    SFIS.visitAll(E);
    return SFIS.FoundInside;
  }

  bool follow(const SCEV *E) {
    if (auto *AddRec = dyn_cast<SCEVAddRecExpr>(E)) {
      FoundInside |= S->getRegion().contains(AddRec->getLoop());
    } else if (auto *Unknown = dyn_cast<SCEVUnknown>(E)) {
      if (Instruction *I = dyn_cast<Instruction>(Unknown->getValue()))
        FoundInside |= S->getRegion().contains(I) && !VMap.count(I);
    }
    return !FoundInside;
  }

  bool isDone() { return FoundInside; }
};
} // end anonymous namespace

const SCEV *Scop::getRepresentingInvariantLoadSCEV(const SCEV *E) const {
  // Check whether it makes sense to rewrite the SCEV.  (ScalarEvolution
  // doesn't like addition between an AddRec and an expression that
  // doesn't have a dominance relationship with it.)
  if (SCEVFindInsideScop::hasVariant(E, *SE, InvEquivClassVMap, this))
    return E;

  // Rewrite SCEV.
  return SCEVSensitiveParameterRewriter::rewrite(E, *SE, InvEquivClassVMap);
}

void Scop::createParameterId(const SCEV *Parameter) {
  assert(Parameters.count(Parameter));
  assert(!ParameterIds.count(Parameter));

  std::string ParameterName = "p_" + std::to_string(getNumParams() - 1);

  if (const SCEVUnknown *ValueParameter = dyn_cast<SCEVUnknown>(Parameter)) {
    Value *Val = ValueParameter->getValue();

    if (UseInstructionNames) {
      // If this parameter references a specific Value and this value has a name
      // we use this name as it is likely to be unique and more useful than just
      // a number.
      if (Val->hasName())
        ParameterName = Val->getName().str();
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
  }

  isl::id Id = isl::id::alloc(getIslCtx(), ParameterName,
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

isl::id Scop::getIdForParam(const SCEV *Parameter) const {
  // Normalize the SCEV to get the representing element for an invariant load.
  Parameter = getRepresentingInvariantLoadSCEV(Parameter);
  return ParameterIds.lookup(Parameter);
}

bool Scop::isDominatedBy(const DominatorTree &DT, BasicBlock *BB) const {
  return DT.dominates(BB, getEntry());
}

void Scop::buildContext() {
  isl::space Space = isl::space::params_alloc(getIslCtx(), 0);
  Context = isl::set::universe(Space);
  InvalidContext = isl::set::empty(Space);
  AssumedContext = isl::set::universe(Space);
  DefinedBehaviorContext = isl::set::universe(Space);
}

void Scop::addParameterBounds() {
  unsigned PDim = 0;
  for (auto *Parameter : Parameters) {
    ConstantRange SRange = SE->getSignedRange(Parameter);
    Context = addRangeBoundsToSet(Context, SRange, PDim++, isl::dim::param);
  }
  intersectDefinedBehavior(Context, AS_ASSUMPTION);
}

void Scop::realignParams() {
  if (PollyIgnoreParamBounds)
    return;

  // Add all parameters into a common model.
  isl::space Space = getFullParamSpace();

  // Align the parameters of all data structures to the model.
  Context = Context.align_params(Space);
  AssumedContext = AssumedContext.align_params(Space);
  InvalidContext = InvalidContext.align_params(Space);

  // As all parameters are known add bounds to them.
  addParameterBounds();

  for (ScopStmt &Stmt : *this)
    Stmt.realignParams();
  // Simplify the schedule according to the context too.
  Schedule = Schedule.gist_domain_params(getContext());

  // Predictable parameter order is required for JSON imports. Ensure alignment
  // by explicitly calling align_params.
  Schedule = Schedule.align_params(Space);
}

static isl::set simplifyAssumptionContext(isl::set AssumptionContext,
                                          const Scop &S) {
  // If we have modeled all blocks in the SCoP that have side effects we can
  // simplify the context with the constraints that are needed for anything to
  // be executed at all. However, if we have error blocks in the SCoP we already
  // assumed some parameter combinations cannot occur and removed them from the
  // domains, thus we cannot use the remaining domain to simplify the
  // assumptions.
  if (!S.hasErrorBlock()) {
    auto DomainParameters = S.getDomains().params();
    AssumptionContext = AssumptionContext.gist_params(DomainParameters);
  }

  AssumptionContext = AssumptionContext.gist_params(S.getContext());
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
  InvalidContext = InvalidContext.align_params(getParamSpace());
  simplify(DefinedBehaviorContext);
  DefinedBehaviorContext = DefinedBehaviorContext.align_params(getParamSpace());
}

isl::set Scop::getDomainConditions(const ScopStmt *Stmt) const {
  return getDomainConditions(Stmt->getEntryBlock());
}

isl::set Scop::getDomainConditions(BasicBlock *BB) const {
  auto DIt = DomainMap.find(BB);
  if (DIt != DomainMap.end())
    return DIt->getSecond();

  auto &RI = *R.getRegionInfo();
  auto *BBR = RI.getRegionFor(BB);
  while (BBR->getEntry() == BB)
    BBR = BBR->getParent();
  return getDomainConditions(BBR->getEntry());
}

Scop::Scop(Region &R, ScalarEvolution &ScalarEvolution, LoopInfo &LI,
           DominatorTree &DT, ScopDetection::DetectionContext &DC,
           OptimizationRemarkEmitter &ORE, int ID)
    : IslCtx(isl_ctx_alloc(), isl_ctx_free), SE(&ScalarEvolution), DT(&DT),
      R(R), name(None), HasSingleExitEdge(R.getExitingBlock()), DC(DC),
      ORE(ORE), Affinator(this, LI), ID(ID) {

  // Options defaults that are different from ISL's.
  isl_options_set_schedule_serialize_sccs(IslCtx.get(), true);

  SmallVector<char *, 8> IslArgv;
  IslArgv.reserve(1 + IslArgs.size());

  // Substitute for program name.
  IslArgv.push_back(const_cast<char *>("-polly-isl-arg"));

  for (std::string &Arg : IslArgs)
    IslArgv.push_back(const_cast<char *>(Arg.c_str()));

  // Abort if unknown argument is passed.
  // Note that "-V" (print isl version) will always call exit(0), so we cannot
  // avoid ISL aborting the program at this point.
  unsigned IslParseFlags = ISL_ARG_ALL;

  isl_ctx_parse_options(IslCtx.get(), IslArgv.size(), IslArgv.data(),
                        IslParseFlags);

  if (IslOnErrorAbort)
    isl_options_set_on_error(getIslCtx().get(), ISL_ON_ERROR_ABORT);
  buildContext();
}

Scop::~Scop() = default;

void Scop::removeFromStmtMap(ScopStmt &Stmt) {
  for (Instruction *Inst : Stmt.getInstructions())
    InstStmtMap.erase(Inst);

  if (Stmt.isRegionStmt()) {
    for (BasicBlock *BB : Stmt.getRegion()->blocks()) {
      StmtMap.erase(BB);
      // Skip entry basic block, as its instructions are already deleted as
      // part of the statement's instruction list.
      if (BB == Stmt.getEntryBlock())
        continue;
      for (Instruction &Inst : *BB)
        InstStmtMap.erase(&Inst);
    }
  } else {
    auto StmtMapIt = StmtMap.find(Stmt.getBasicBlock());
    if (StmtMapIt != StmtMap.end())
      StmtMapIt->second.erase(std::remove(StmtMapIt->second.begin(),
                                          StmtMapIt->second.end(), &Stmt),
                              StmtMapIt->second.end());
    for (Instruction *Inst : Stmt.getInstructions())
      InstStmtMap.erase(Inst);
  }
}

void Scop::removeStmts(function_ref<bool(ScopStmt &)> ShouldDelete,
                       bool AfterHoisting) {
  for (auto StmtIt = Stmts.begin(), StmtEnd = Stmts.end(); StmtIt != StmtEnd;) {
    if (!ShouldDelete(*StmtIt)) {
      StmtIt++;
      continue;
    }

    // Start with removing all of the statement's accesses including erasing it
    // from all maps that are pointing to them.
    // Make a temporary copy because removing MAs invalidates the iterator.
    SmallVector<MemoryAccess *, 16> MAList(StmtIt->begin(), StmtIt->end());
    for (MemoryAccess *MA : MAList)
      StmtIt->removeSingleMemoryAccess(MA, AfterHoisting);

    removeFromStmtMap(*StmtIt);
    StmtIt = Stmts.erase(StmtIt);
  }
}

void Scop::removeStmtNotInDomainMap() {
  removeStmts([this](ScopStmt &Stmt) -> bool {
    isl::set Domain = DomainMap.lookup(Stmt.getEntryBlock());
    if (Domain.is_null())
      return true;
    return Domain.is_empty();
  });
}

void Scop::simplifySCoP(bool AfterHoisting) {
  removeStmts(
      [AfterHoisting](ScopStmt &Stmt) -> bool {
        // Never delete statements that contain calls to debug functions.
        if (hasDebugCall(&Stmt))
          return false;

        bool RemoveStmt = Stmt.isEmpty();

        // Remove read only statements only after invariant load hoisting.
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
        return RemoveStmt;
      },
      AfterHoisting);
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

ScopArrayInfo *Scop::getOrCreateScopArrayInfo(Value *BasePtr, Type *ElementType,
                                              ArrayRef<const SCEV *> Sizes,
                                              MemoryKind Kind,
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

ScopArrayInfo *Scop::createScopArrayInfo(Type *ElementType,
                                         const std::string &BaseName,
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

ScopArrayInfo *Scop::getScopArrayInfoOrNull(Value *BasePtr, MemoryKind Kind) {
  auto *SAI = ScopArrayInfoMap[std::make_pair(BasePtr, Kind)].get();
  return SAI;
}

ScopArrayInfo *Scop::getScopArrayInfo(Value *BasePtr, MemoryKind Kind) {
  auto *SAI = getScopArrayInfoOrNull(BasePtr, Kind);
  assert(SAI && "No ScopArrayInfo available for this base pointer");
  return SAI;
}

std::string Scop::getContextStr() const {
  return stringFromIslObj(getContext());
}

std::string Scop::getAssumedContextStr() const {
  assert(!AssumedContext.is_null() && "Assumed context not yet built");
  return stringFromIslObj(AssumedContext);
}

std::string Scop::getInvalidContextStr() const {
  return stringFromIslObj(InvalidContext);
}

std::string Scop::getNameStr() const {
  std::string ExitName, EntryName;
  std::tie(EntryName, ExitName) = getEntryExitStr();
  return EntryName + "---" + ExitName;
}

std::pair<std::string, std::string> Scop::getEntryExitStr() const {
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

  return std::make_pair(EntryName, ExitName);
}

isl::set Scop::getContext() const { return Context; }

isl::space Scop::getParamSpace() const { return getContext().get_space(); }

isl::space Scop::getFullParamSpace() const {

  isl::space Space = isl::space::params_alloc(getIslCtx(), ParameterIds.size());

  unsigned PDim = 0;
  for (const SCEV *Parameter : Parameters) {
    isl::id Id = getIdForParam(Parameter);
    Space = Space.set_dim_id(isl::dim::param, PDim++, Id);
  }

  return Space;
}

isl::set Scop::getAssumedContext() const {
  assert(!AssumedContext.is_null() && "Assumed context not yet built");
  return AssumedContext;
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
  if (Stmts.empty())
    return false;

  isl::set PositiveContext = getAssumedContext();
  isl::set NegativeContext = getInvalidContext();
  PositiveContext = PositiveContext.intersect_params(Context);
  PositiveContext = PositiveContext.intersect_params(getDomains().params());
  return PositiveContext.is_empty().is_false() &&
         PositiveContext.is_subset(NegativeContext).is_false();
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

bool Scop::isEffectiveAssumption(isl::set Set, AssumptionSign Sign) {
  if (Sign == AS_ASSUMPTION) {
    if (Context.is_subset(Set))
      return false;

    if (AssumedContext.is_subset(Set))
      return false;
  } else {
    if (Set.is_disjoint(Context))
      return false;

    if (Set.is_subset(InvalidContext))
      return false;
  }
  return true;
}

bool Scop::trackAssumption(AssumptionKind Kind, isl::set Set, DebugLoc Loc,
                           AssumptionSign Sign, BasicBlock *BB) {
  if (PollyRemarksMinimal && !isEffectiveAssumption(Set, Sign))
    return false;

  // Do never emit trivial assumptions as they only clutter the output.
  if (!PollyRemarksMinimal) {
    isl::set Univ;
    if (Sign == AS_ASSUMPTION)
      Univ = isl::set::universe(Set.get_space());

    bool IsTrivial = (Sign == AS_RESTRICTION && Set.is_empty()) ||
                     (Sign == AS_ASSUMPTION && Univ.is_equal(Set));

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

  auto Suffix = Sign == AS_ASSUMPTION ? " assumption:\t" : " restriction:\t";
  std::string Msg = toString(Kind) + Suffix + stringFromIslObj(Set);
  if (BB)
    ORE.emit(OptimizationRemarkAnalysis(DEBUG_TYPE, "AssumpRestrict", Loc, BB)
             << Msg);
  else
    ORE.emit(OptimizationRemarkAnalysis(DEBUG_TYPE, "AssumpRestrict", Loc,
                                        R.getEntry())
             << Msg);
  return true;
}

void Scop::addAssumption(AssumptionKind Kind, isl::set Set, DebugLoc Loc,
                         AssumptionSign Sign, BasicBlock *BB,
                         bool RequiresRTC) {
  // Simplify the assumptions/restrictions first.
  Set = Set.gist_params(getContext());
  intersectDefinedBehavior(Set, Sign);

  if (!RequiresRTC)
    return;

  if (!trackAssumption(Kind, Set, Loc, Sign, BB))
    return;

  if (Sign == AS_ASSUMPTION)
    AssumedContext = AssumedContext.intersect(Set).coalesce();
  else
    InvalidContext = InvalidContext.unite(Set).coalesce();
}

void Scop::intersectDefinedBehavior(isl::set Set, AssumptionSign Sign) {
  if (DefinedBehaviorContext.is_null())
    return;

  if (Sign == AS_ASSUMPTION)
    DefinedBehaviorContext = DefinedBehaviorContext.intersect(Set);
  else
    DefinedBehaviorContext = DefinedBehaviorContext.subtract(Set);

  // Limit the complexity of the context. If complexity is exceeded, simplify
  // the set and check again.
  if (DefinedBehaviorContext.n_basic_set().release() >
      MaxDisjunktsInDefinedBehaviourContext) {
    simplify(DefinedBehaviorContext);
    if (DefinedBehaviorContext.n_basic_set().release() >
        MaxDisjunktsInDefinedBehaviourContext)
      DefinedBehaviorContext = {};
  }
}

void Scop::invalidate(AssumptionKind Kind, DebugLoc Loc, BasicBlock *BB) {
  LLVM_DEBUG(dbgs() << "Invalidate SCoP because of reason " << Kind << "\n");
  addAssumption(Kind, isl::set::empty(getParamSpace()), Loc, AS_ASSUMPTION, BB);
}

isl::set Scop::getInvalidContext() const { return InvalidContext; }

void Scop::printContext(raw_ostream &OS) const {
  OS << "Context:\n";
  OS.indent(4) << Context << "\n";

  OS.indent(4) << "Assumed Context:\n";
  OS.indent(4) << AssumedContext << "\n";

  OS.indent(4) << "Invalid Context:\n";
  OS.indent(4) << InvalidContext << "\n";

  OS.indent(4) << "Defined Behavior Context:\n";
  if (!DefinedBehaviorContext.is_null())
    OS.indent(4) << DefinedBehaviorContext << "\n";
  else
    OS.indent(4) << "<unavailable>\n";

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

void Scop::printStatements(raw_ostream &OS, bool PrintInstructions) const {
  OS << "Statements {\n";

  for (const ScopStmt &Stmt : *this) {
    OS.indent(4);
    Stmt.print(OS, PrintInstructions);
  }

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

void Scop::print(raw_ostream &OS, bool PrintInstructions) const {
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
  printStatements(OS.indent(4), PrintInstructions);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void Scop::dump() const { print(dbgs(), true); }
#endif

isl::ctx Scop::getIslCtx() const { return IslCtx.get(); }

__isl_give PWACtx Scop::getPwAff(const SCEV *E, BasicBlock *BB,
                                 bool NonNegative,
                                 RecordedAssumptionsTy *RecordedAssumptions) {
  // First try to use the SCEVAffinator to generate a piecewise defined
  // affine function from @p E in the context of @p BB. If that tasks becomes to
  // complex the affinator might return a nullptr. In such a case we invalidate
  // the SCoP and return a dummy value. This way we do not need to add error
  // handling code to all users of this function.
  auto PWAC = Affinator.getPwAff(E, BB, RecordedAssumptions);
  if (!PWAC.first.is_null()) {
    // TODO: We could use a heuristic and either use:
    //         SCEVAffinator::takeNonNegativeAssumption
    //       or
    //         SCEVAffinator::interpretAsUnsigned
    //       to deal with unsigned or "NonNegative" SCEVs.
    if (NonNegative)
      Affinator.takeNonNegativeAssumption(PWAC, RecordedAssumptions);
    return PWAC;
  }

  auto DL = BB ? BB->getTerminator()->getDebugLoc() : DebugLoc();
  invalidate(COMPLEXITY, DL, BB);
  return Affinator.getPwAff(SE->getZero(E->getType()), BB, RecordedAssumptions);
}

isl::union_set Scop::getDomains() const {
  isl_space *EmptySpace = isl_space_params_alloc(getIslCtx().get(), 0);
  isl_union_set *Domain = isl_union_set_empty(EmptySpace);

  for (const ScopStmt &Stmt : *this)
    Domain = isl_union_set_add_set(Domain, Stmt.getDomain().release());

  return isl::manage(Domain);
}

isl::pw_aff Scop::getPwAffOnly(const SCEV *E, BasicBlock *BB,
                               RecordedAssumptionsTy *RecordedAssumptions) {
  PWACtx PWAC = getPwAff(E, BB, RecordedAssumptions);
  return PWAC.first;
}

isl::union_map
Scop::getAccessesOfType(std::function<bool(MemoryAccess &)> Predicate) {
  isl::union_map Accesses = isl::union_map::empty(getIslCtx());

  for (ScopStmt &Stmt : *this) {
    for (MemoryAccess *MA : Stmt) {
      if (!Predicate(*MA))
        continue;

      isl::set Domain = Stmt.getDomain();
      isl::map AccessDomain = MA->getAccessRelation();
      AccessDomain = AccessDomain.intersect_domain(Domain);
      Accesses = Accesses.unite(AccessDomain);
    }
  }

  return Accesses.coalesce();
}

isl::union_map Scop::getMustWrites() {
  return getAccessesOfType([](MemoryAccess &MA) { return MA.isMustWrite(); });
}

isl::union_map Scop::getMayWrites() {
  return getAccessesOfType([](MemoryAccess &MA) { return MA.isMayWrite(); });
}

isl::union_map Scop::getWrites() {
  return getAccessesOfType([](MemoryAccess &MA) { return MA.isWrite(); });
}

isl::union_map Scop::getReads() {
  return getAccessesOfType([](MemoryAccess &MA) { return MA.isRead(); });
}

isl::union_map Scop::getAccesses() {
  return getAccessesOfType([](MemoryAccess &MA) { return true; });
}

isl::union_map Scop::getAccesses(ScopArrayInfo *Array) {
  return getAccessesOfType(
      [Array](MemoryAccess &MA) { return MA.getScopArrayInfo() == Array; });
}

isl::union_map Scop::getSchedule() const {
  auto Tree = getScheduleTree();
  return Tree.get_map();
}

isl::schedule Scop::getScheduleTree() const {
  return Schedule.intersect_domain(getDomains());
}

void Scop::setSchedule(isl::union_map NewSchedule) {
  auto S = isl::schedule::from_domain(getDomains());
  Schedule = S.insert_partial_schedule(
      isl::multi_union_pw_aff::from_union_map(NewSchedule));
  ScheduleModified = true;
}

void Scop::setScheduleTree(isl::schedule NewSchedule) {
  Schedule = NewSchedule;
  ScheduleModified = true;
}

bool Scop::restrictDomains(isl::union_set Domain) {
  bool Changed = false;
  for (ScopStmt &Stmt : *this) {
    isl::union_set StmtDomain = isl::union_set(Stmt.getDomain());
    isl::union_set NewStmtDomain = StmtDomain.intersect(Domain);

    if (StmtDomain.is_subset(NewStmtDomain))
      continue;

    Changed = true;

    NewStmtDomain = NewStmtDomain.coalesce();

    if (NewStmtDomain.is_empty())
      Stmt.restrictDomain(isl::set::empty(Stmt.getDomainSpace()));
    else
      Stmt.restrictDomain(isl::set(NewStmtDomain));
  }
  return Changed;
}

ScalarEvolution *Scop::getSE() const { return SE; }

void Scop::addScopStmt(BasicBlock *BB, StringRef Name, Loop *SurroundingLoop,
                       std::vector<Instruction *> Instructions) {
  assert(BB && "Unexpected nullptr!");
  Stmts.emplace_back(*this, *BB, Name, SurroundingLoop, Instructions);
  auto *Stmt = &Stmts.back();
  StmtMap[BB].push_back(Stmt);
  for (Instruction *Inst : Instructions) {
    assert(!InstStmtMap.count(Inst) &&
           "Unexpected statement corresponding to the instruction.");
    InstStmtMap[Inst] = Stmt;
  }
}

void Scop::addScopStmt(Region *R, StringRef Name, Loop *SurroundingLoop,
                       std::vector<Instruction *> Instructions) {
  assert(R && "Unexpected nullptr!");
  Stmts.emplace_back(*this, *R, Name, SurroundingLoop, Instructions);
  auto *Stmt = &Stmts.back();

  for (Instruction *Inst : Instructions) {
    assert(!InstStmtMap.count(Inst) &&
           "Unexpected statement corresponding to the instruction.");
    InstStmtMap[Inst] = Stmt;
  }

  for (BasicBlock *BB : R->blocks()) {
    StmtMap[BB].push_back(Stmt);
    if (BB == R->getEntry())
      continue;
    for (Instruction &Inst : *BB) {
      assert(!InstStmtMap.count(&Inst) &&
             "Unexpected statement corresponding to the instruction.");
      InstStmtMap[&Inst] = Stmt;
    }
  }
}

ScopStmt *Scop::addScopStmt(isl::map SourceRel, isl::map TargetRel,
                            isl::set Domain) {
#ifndef NDEBUG
  isl::set SourceDomain = SourceRel.domain();
  isl::set TargetDomain = TargetRel.domain();
  assert(Domain.is_subset(TargetDomain) &&
         "Target access not defined for complete statement domain");
  assert(Domain.is_subset(SourceDomain) &&
         "Source access not defined for complete statement domain");
#endif
  Stmts.emplace_back(*this, SourceRel, TargetRel, Domain);
  CopyStmtsNum++;
  return &(Stmts.back());
}

ArrayRef<ScopStmt *> Scop::getStmtListFor(BasicBlock *BB) const {
  auto StmtMapIt = StmtMap.find(BB);
  if (StmtMapIt == StmtMap.end())
    return {};
  return StmtMapIt->second;
}

ScopStmt *Scop::getIncomingStmtFor(const Use &U) const {
  auto *PHI = cast<PHINode>(U.getUser());
  BasicBlock *IncomingBB = PHI->getIncomingBlock(U);

  // If the value is a non-synthesizable from the incoming block, use the
  // statement that contains it as user statement.
  if (auto *IncomingInst = dyn_cast<Instruction>(U.get())) {
    if (IncomingInst->getParent() == IncomingBB) {
      if (ScopStmt *IncomingStmt = getStmtFor(IncomingInst))
        return IncomingStmt;
    }
  }

  // Otherwise, use the epilogue/last statement.
  return getLastStmtFor(IncomingBB);
}

ScopStmt *Scop::getLastStmtFor(BasicBlock *BB) const {
  ArrayRef<ScopStmt *> StmtList = getStmtListFor(BB);
  if (!StmtList.empty())
    return StmtList.back();
  return nullptr;
}

ArrayRef<ScopStmt *> Scop::getStmtListFor(RegionNode *RN) const {
  if (RN->isSubRegion())
    return getStmtListFor(RN->getNodeAs<Region>());
  return getStmtListFor(RN->getNodeAs<BasicBlock>());
}

ArrayRef<ScopStmt *> Scop::getStmtListFor(Region *R) const {
  return getStmtListFor(R->getEntry());
}

int Scop::getRelativeLoopDepth(const Loop *L) const {
  if (!L || !R.contains(L))
    return -1;
  // outermostLoopInRegion always returns nullptr for top level regions
  if (R.isTopLevelRegion()) {
    // LoopInfo's depths start at 1, we start at 0
    return L->getLoopDepth() - 1;
  } else {
    Loop *OuterLoop = R.outermostLoopInRegion(const_cast<Loop *>(L));
    assert(OuterLoop);
    return L->getLoopDepth() - OuterLoop->getLoopDepth();
  }
}

ScopArrayInfo *Scop::getArrayInfoByName(const std::string BaseName) {
  for (auto &SAI : arrays()) {
    if (SAI->getName() == BaseName)
      return SAI;
  }
  return nullptr;
}

void Scop::addAccessData(MemoryAccess *Access) {
  const ScopArrayInfo *SAI = Access->getOriginalScopArrayInfo();
  assert(SAI && "can only use after access relations have been constructed");

  if (Access->isOriginalValueKind() && Access->isRead())
    ValueUseAccs[SAI].push_back(Access);
  else if (Access->isOriginalAnyPHIKind() && Access->isWrite())
    PHIIncomingAccs[SAI].push_back(Access);
}

void Scop::removeAccessData(MemoryAccess *Access) {
  if (Access->isOriginalValueKind() && Access->isWrite()) {
    ValueDefAccs.erase(Access->getAccessValue());
  } else if (Access->isOriginalValueKind() && Access->isRead()) {
    auto &Uses = ValueUseAccs[Access->getScopArrayInfo()];
    auto NewEnd = std::remove(Uses.begin(), Uses.end(), Access);
    Uses.erase(NewEnd, Uses.end());
  } else if (Access->isOriginalPHIKind() && Access->isRead()) {
    PHINode *PHI = cast<PHINode>(Access->getAccessInstruction());
    PHIReadAccs.erase(PHI);
  } else if (Access->isOriginalAnyPHIKind() && Access->isWrite()) {
    auto &Incomings = PHIIncomingAccs[Access->getScopArrayInfo()];
    auto NewEnd = std::remove(Incomings.begin(), Incomings.end(), Access);
    Incomings.erase(NewEnd, Incomings.end());
  }
}

MemoryAccess *Scop::getValueDef(const ScopArrayInfo *SAI) const {
  assert(SAI->isValueKind());

  Instruction *Val = dyn_cast<Instruction>(SAI->getBasePtr());
  if (!Val)
    return nullptr;

  return ValueDefAccs.lookup(Val);
}

ArrayRef<MemoryAccess *> Scop::getValueUses(const ScopArrayInfo *SAI) const {
  assert(SAI->isValueKind());
  auto It = ValueUseAccs.find(SAI);
  if (It == ValueUseAccs.end())
    return {};
  return It->second;
}

MemoryAccess *Scop::getPHIRead(const ScopArrayInfo *SAI) const {
  assert(SAI->isPHIKind() || SAI->isExitPHIKind());

  if (SAI->isExitPHIKind())
    return nullptr;

  PHINode *PHI = cast<PHINode>(SAI->getBasePtr());
  return PHIReadAccs.lookup(PHI);
}

ArrayRef<MemoryAccess *> Scop::getPHIIncomings(const ScopArrayInfo *SAI) const {
  assert(SAI->isPHIKind() || SAI->isExitPHIKind());
  auto It = PHIIncomingAccs.find(SAI);
  if (It == PHIIncomingAccs.end())
    return {};
  return It->second;
}

bool Scop::isEscaping(Instruction *Inst) {
  assert(contains(Inst) && "The concept of escaping makes only sense for "
                           "values defined inside the SCoP");

  for (Use &Use : Inst->uses()) {
    BasicBlock *UserBB = getUseBlock(Use);
    if (!contains(UserBB))
      return true;

    // When the SCoP region exit needs to be simplified, PHIs in the region exit
    // move to a new basic block such that its incoming blocks are not in the
    // SCoP anymore.
    if (hasSingleExitEdge() && isa<PHINode>(Use.getUser()) &&
        isExit(cast<PHINode>(Use.getUser())->getParent()))
      return true;
  }
  return false;
}

void Scop::incrementNumberOfAliasingAssumptions(unsigned step) {
  AssumptionsAliasing += step;
}

Scop::ScopStatistics Scop::getStatistics() const {
  ScopStatistics Result;
#if !defined(NDEBUG) || defined(LLVM_ENABLE_STATS)
  auto LoopStat = ScopDetection::countBeneficialLoops(&R, *SE, *getLI(), 0);

  int NumTotalLoops = LoopStat.NumLoops;
  Result.NumBoxedLoops = getBoxedLoops().size();
  Result.NumAffineLoops = NumTotalLoops - Result.NumBoxedLoops;

  for (const ScopStmt &Stmt : *this) {
    isl::set Domain = Stmt.getDomain().intersect_params(getContext());
    bool IsInLoop = Stmt.getNumIterators() >= 1;
    for (MemoryAccess *MA : Stmt) {
      if (!MA->isWrite())
        continue;

      if (MA->isLatestValueKind()) {
        Result.NumValueWrites += 1;
        if (IsInLoop)
          Result.NumValueWritesInLoops += 1;
      }

      if (MA->isLatestAnyPHIKind()) {
        Result.NumPHIWrites += 1;
        if (IsInLoop)
          Result.NumPHIWritesInLoops += 1;
      }

      isl::set AccSet =
          MA->getAccessRelation().intersect_domain(Domain).range();
      if (AccSet.is_singleton()) {
        Result.NumSingletonWrites += 1;
        if (IsInLoop)
          Result.NumSingletonWritesInLoops += 1;
      }
    }
  }
#endif
  return Result;
}

raw_ostream &polly::operator<<(raw_ostream &OS, const Scop &scop) {
  scop.print(OS, PollyPrintInstructions);
  return OS;
}

//===----------------------------------------------------------------------===//
void ScopInfoRegionPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<RegionInfoPass>();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequiredTransitive<ScalarEvolutionWrapperPass>();
  AU.addRequiredTransitive<ScopDetectionWrapperPass>();
  AU.addRequired<AAResultsWrapperPass>();
  AU.addRequired<AssumptionCacheTracker>();
  AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
  AU.setPreservesAll();
}

void updateLoopCountStatistic(ScopDetection::LoopStats Stats,
                              Scop::ScopStatistics ScopStats) {
  assert(Stats.NumLoops == ScopStats.NumAffineLoops + ScopStats.NumBoxedLoops);

  NumScops++;
  NumLoopsInScop += Stats.NumLoops;
  MaxNumLoopsInScop =
      std::max(MaxNumLoopsInScop.getValue(), (unsigned)Stats.NumLoops);

  if (Stats.MaxDepth == 0)
    NumScopsDepthZero++;
  else if (Stats.MaxDepth == 1)
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

  NumAffineLoops += ScopStats.NumAffineLoops;
  NumBoxedLoops += ScopStats.NumBoxedLoops;

  NumValueWrites += ScopStats.NumValueWrites;
  NumValueWritesInLoops += ScopStats.NumValueWritesInLoops;
  NumPHIWrites += ScopStats.NumPHIWrites;
  NumPHIWritesInLoops += ScopStats.NumPHIWritesInLoops;
  NumSingletonWrites += ScopStats.NumSingletonWrites;
  NumSingletonWritesInLoops += ScopStats.NumSingletonWritesInLoops;
}

bool ScopInfoRegionPass::runOnRegion(Region *R, RGPassManager &RGM) {
  auto &SD = getAnalysis<ScopDetectionWrapperPass>().getSD();

  if (!SD.isMaxRegionInScop(*R))
    return false;

  Function *F = R->getEntry()->getParent();
  auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
  auto const &DL = F->getParent()->getDataLayout();
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(*F);
  auto &ORE = getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();

  ScopBuilder SB(R, AC, AA, DL, DT, LI, SD, SE, ORE);
  S = SB.getScop(); // take ownership of scop object

#if !defined(NDEBUG) || defined(LLVM_ENABLE_STATS)
  if (S) {
    ScopDetection::LoopStats Stats =
        ScopDetection::countBeneficialLoops(&S->getRegion(), SE, LI, 0);
    updateLoopCountStatistic(Stats, S->getStatistics());
  }
#endif

  return false;
}

void ScopInfoRegionPass::print(raw_ostream &OS, const Module *) const {
  if (S)
    S->print(OS, PollyPrintInstructions);
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
INITIALIZE_PASS_DEPENDENCY(ScopDetectionWrapperPass);
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass);
INITIALIZE_PASS_END(ScopInfoRegionPass, "polly-scops",
                    "Polly - Create polyhedral description of Scops", false,
                    false)

//===----------------------------------------------------------------------===//

namespace {

/// Print result from ScopInfoRegionPass.
class ScopInfoPrinterLegacyRegionPass final : public RegionPass {
public:
  static char ID;

  ScopInfoPrinterLegacyRegionPass() : ScopInfoPrinterLegacyRegionPass(outs()) {}

  explicit ScopInfoPrinterLegacyRegionPass(llvm::raw_ostream &OS)
      : RegionPass(ID), OS(OS) {}

  bool runOnRegion(Region *R, RGPassManager &RGM) override {
    ScopInfoRegionPass &P = getAnalysis<ScopInfoRegionPass>();

    OS << "Printing analysis '" << P.getPassName() << "' for region: '"
       << R->getNameStr() << "' in function '"
       << R->getEntry()->getParent()->getName() << "':\n";
    P.print(OS);

    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    RegionPass::getAnalysisUsage(AU);
    AU.addRequired<ScopInfoRegionPass>();
    AU.setPreservesAll();
  }

private:
  llvm::raw_ostream &OS;
};

char ScopInfoPrinterLegacyRegionPass::ID = 0;
} // namespace

Pass *polly::createScopInfoPrinterLegacyRegionPass(raw_ostream &OS) {
  return new ScopInfoPrinterLegacyRegionPass(OS);
}

INITIALIZE_PASS_BEGIN(ScopInfoPrinterLegacyRegionPass, "polly-print-scops",
                      "Polly - Print polyhedral description of Scops", false,
                      false);
INITIALIZE_PASS_DEPENDENCY(ScopInfoRegionPass);
INITIALIZE_PASS_END(ScopInfoPrinterLegacyRegionPass, "polly-print-scops",
                    "Polly - Print polyhedral description of Scops", false,
                    false)

//===----------------------------------------------------------------------===//

ScopInfo::ScopInfo(const DataLayout &DL, ScopDetection &SD, ScalarEvolution &SE,
                   LoopInfo &LI, AliasAnalysis &AA, DominatorTree &DT,
                   AssumptionCache &AC, OptimizationRemarkEmitter &ORE)
    : DL(DL), SD(SD), SE(SE), LI(LI), AA(AA), DT(DT), AC(AC), ORE(ORE) {
  recompute();
}

void ScopInfo::recompute() {
  RegionToScopMap.clear();
  /// Create polyhedral description of scops for all the valid regions of a
  /// function.
  for (auto &It : SD) {
    Region *R = const_cast<Region *>(It);
    if (!SD.isMaxRegionInScop(*R))
      continue;

    ScopBuilder SB(R, AC, AA, DL, DT, LI, SD, SE, ORE);
    std::unique_ptr<Scop> S = SB.getScop();
    if (!S)
      continue;
#if !defined(NDEBUG) || defined(LLVM_ENABLE_STATS)
    ScopDetection::LoopStats Stats =
        ScopDetection::countBeneficialLoops(&S->getRegion(), SE, LI, 0);
    updateLoopCountStatistic(Stats, S->getStatistics());
#endif
    bool Inserted = RegionToScopMap.insert({R, std::move(S)}).second;
    assert(Inserted && "Building Scop for the same region twice!");
    (void)Inserted;
  }
}

bool ScopInfo::invalidate(Function &F, const PreservedAnalyses &PA,
                          FunctionAnalysisManager::Invalidator &Inv) {
  // Check whether the analysis, all analyses on functions have been preserved
  // or anything we're holding references to is being invalidated
  auto PAC = PA.getChecker<ScopInfoAnalysis>();
  return !(PAC.preserved() || PAC.preservedSet<AllAnalysesOn<Function>>()) ||
         Inv.invalidate<ScopAnalysis>(F, PA) ||
         Inv.invalidate<ScalarEvolutionAnalysis>(F, PA) ||
         Inv.invalidate<LoopAnalysis>(F, PA) ||
         Inv.invalidate<AAManager>(F, PA) ||
         Inv.invalidate<DominatorTreeAnalysis>(F, PA) ||
         Inv.invalidate<AssumptionAnalysis>(F, PA);
}

AnalysisKey ScopInfoAnalysis::Key;

ScopInfoAnalysis::Result ScopInfoAnalysis::run(Function &F,
                                               FunctionAnalysisManager &FAM) {
  auto &SD = FAM.getResult<ScopAnalysis>(F);
  auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
  auto &LI = FAM.getResult<LoopAnalysis>(F);
  auto &AA = FAM.getResult<AAManager>(F);
  auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  auto &AC = FAM.getResult<AssumptionAnalysis>(F);
  auto &DL = F.getParent()->getDataLayout();
  auto &ORE = FAM.getResult<OptimizationRemarkEmitterAnalysis>(F);
  return {DL, SD, SE, LI, AA, DT, AC, ORE};
}

PreservedAnalyses ScopInfoPrinterPass::run(Function &F,
                                           FunctionAnalysisManager &FAM) {
  auto &SI = FAM.getResult<ScopInfoAnalysis>(F);
  // Since the legacy PM processes Scops in bottom up, we print them in reverse
  // order here to keep the output persistent
  for (auto &It : reverse(SI)) {
    if (It.second)
      It.second->print(Stream, PollyPrintInstructions);
    else
      Stream << "Invalid Scop!\n";
  }
  return PreservedAnalyses::all();
}

void ScopInfoWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<RegionInfoPass>();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequiredTransitive<ScalarEvolutionWrapperPass>();
  AU.addRequiredTransitive<ScopDetectionWrapperPass>();
  AU.addRequired<AAResultsWrapperPass>();
  AU.addRequired<AssumptionCacheTracker>();
  AU.addRequired<OptimizationRemarkEmitterWrapperPass>();
  AU.setPreservesAll();
}

bool ScopInfoWrapperPass::runOnFunction(Function &F) {
  auto &SD = getAnalysis<ScopDetectionWrapperPass>().getSD();
  auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
  auto const &DL = F.getParent()->getDataLayout();
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  auto &ORE = getAnalysis<OptimizationRemarkEmitterWrapperPass>().getORE();

  Result.reset(new ScopInfo{DL, SD, SE, LI, AA, DT, AC, ORE});
  return false;
}

void ScopInfoWrapperPass::print(raw_ostream &OS, const Module *) const {
  for (auto &It : *Result) {
    if (It.second)
      It.second->print(OS, PollyPrintInstructions);
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
INITIALIZE_PASS_DEPENDENCY(ScopDetectionWrapperPass);
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass);
INITIALIZE_PASS_END(
    ScopInfoWrapperPass, "polly-function-scops",
    "Polly - Create polyhedral description of all Scops of a function", false,
    false)

//===----------------------------------------------------------------------===//

namespace {
/// Print result from ScopInfoWrapperPass.
class ScopInfoPrinterLegacyFunctionPass final : public FunctionPass {
public:
  static char ID;

  ScopInfoPrinterLegacyFunctionPass()
      : ScopInfoPrinterLegacyFunctionPass(outs()) {}
  explicit ScopInfoPrinterLegacyFunctionPass(llvm::raw_ostream &OS)
      : FunctionPass(ID), OS(OS) {}

  bool runOnFunction(Function &F) override {
    ScopInfoWrapperPass &P = getAnalysis<ScopInfoWrapperPass>();

    OS << "Printing analysis '" << P.getPassName() << "' for function '"
       << F.getName() << "':\n";
    P.print(OS);

    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    FunctionPass::getAnalysisUsage(AU);
    AU.addRequired<ScopInfoWrapperPass>();
    AU.setPreservesAll();
  }

private:
  llvm::raw_ostream &OS;
};

char ScopInfoPrinterLegacyFunctionPass::ID = 0;
} // namespace

Pass *polly::createScopInfoPrinterLegacyFunctionPass(raw_ostream &OS) {
  return new ScopInfoPrinterLegacyFunctionPass(OS);
}

INITIALIZE_PASS_BEGIN(
    ScopInfoPrinterLegacyFunctionPass, "polly-print-function-scops",
    "Polly - Print polyhedral description of all Scops of a function", false,
    false);
INITIALIZE_PASS_DEPENDENCY(ScopInfoWrapperPass);
INITIALIZE_PASS_END(
    ScopInfoPrinterLegacyFunctionPass, "polly-print-function-scops",
    "Polly - Print polyhedral description of all Scops of a function", false,
    false)
