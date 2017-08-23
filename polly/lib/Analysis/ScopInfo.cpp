//===- ScopInfo.cpp -------------------------------------------------------===//
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
#include "polly/ScopDetection.h"
#include "polly/Support/GICHelper.h"
#include "polly/Support/ISLOStream.h"
#include "polly/Support/SCEVAffinator.h"
#include "polly/Support/SCEVValidator.h"
#include "polly/Support/ScopHelper.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationDiagnosticInfo.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/RegionIterator.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
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
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iterator>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
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

STATISTIC(NumScops, "Number of feasible SCoPs after ScopInfo");
STATISTIC(NumLoopsInScop, "Number of loops in scops");
STATISTIC(NumBoxedLoops, "Number of boxed loops in SCoPs after ScopInfo");
STATISTIC(NumAffineLoops, "Number of affine loops in SCoPs after ScopInfo");

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

// The maximal number of basic sets we allow during domain construction to
// be created. More complex scops will result in very high compile time and
// are also unlikely to result in good code
static int const MaxDisjunctsInDomain = 20;

// The number of disjunct in the context after which we stop to add more
// disjuncts. This parameter is there to avoid exponential growth in the
// number of disjunct when adding non-convex sets to the context.
static int const MaxDisjunctsInContext = 4;

// The maximal number of dimensions we allow during invariant load construction.
// More complex access ranges will result in very high compile time and are also
// unlikely to result in good code. This value is very high and should only
// trigger for corner cases (e.g., the "dct_luma" function in h264, SPEC2006).
static int const MaxDimensionsInAccessRange = 9;

static cl::opt<int>
    OptComputeOut("polly-analysis-computeout",
                  cl::desc("Bound the scop analysis by a maximal amount of "
                           "computational steps (0 means no bound)"),
                  cl::Hidden, cl::init(800000), cl::ZeroOrMore,
                  cl::cat(PollyCategory));

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

static cl::opt<int> RunTimeChecksMaxAccessDisjuncts(
    "polly-rtc-max-array-disjuncts",
    cl::desc("The maximal number of disjunts allowed in memory accesses to "
             "to build RTCs."),
    cl::Hidden, cl::ZeroOrMore, cl::init(8), cl::cat(PollyCategory));

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

static cl::opt<bool> PollyAllowDereferenceOfAllFunctionParams(
    "polly-allow-dereference-of-all-function-parameters",
    cl::desc(
        "Treat all parameters to functions that are pointers as dereferencible."
        " This is useful for invariant load hoisting, since we can generate"
        " less runtime checks. This is only valid if all pointers to functions"
        " are always initialized, so that Polly can choose to hoist"
        " their loads. "),
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
    cl::location(UseInstructionNames), cl::Hidden, cl::init(false),
    cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool> PollyPrintInstructions(
    "polly-print-instructions", cl::desc("Output instructions per ScopStmt"),
    cl::Hidden, cl::Optional, cl::init(false), cl::cat(PollyCategory));

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

static isl::set addRangeBoundsToSet(isl::set S, const ConstantRange &Range,
                                    int dim, isl::dim type) {
  isl::val V;
  isl::ctx Ctx = S.get_ctx();

  // The upper and lower bound for a parameter value is derived either from
  // the data type of the parameter or from the - possibly more restrictive -
  // range metadata.
  V = valFromAPInt(Ctx.get(), Range.getSignedMin(), true);
  S = S.lower_bound_val(type, dim, V);
  V = valFromAPInt(Ctx.get(), Range.getSignedMax(), true);
  S = S.upper_bound_val(type, dim, V);

  if (Range.isFullSet())
    return S;

  if (isl_set_n_basic_set(S.get()) > MaxDisjunctsInContext)
    return S;

  // In case of signed wrapping, we can refine the set of valid values by
  // excluding the part not covered by the wrapping range.
  if (Range.isSignWrappedSet()) {
    V = valFromAPInt(Ctx.get(), Range.getLower(), true);
    isl::set SLB = S.lower_bound_val(type, dim, V);

    V = valFromAPInt(Ctx.get(), Range.getUpper(), true);
    V = V.sub_ui(1);
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
  auto Space = isl::space(Id.get_ctx(), 0, getNumberOfDimensions());
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

/// Make the ScopArrayInfo model a Fortran Array
void ScopArrayInfo::applyAndSetFAD(Value *FAD) {
  assert(FAD && "got invalid Fortran array descriptor");
  if (this->FAD) {
    assert(this->FAD == FAD &&
           "receiving different array descriptors for same array");
    return;
  }

  assert(DimensionSizesPw.size() > 0 && !DimensionSizesPw[0]);
  assert(!this->FAD);
  this->FAD = FAD;

  isl::space Space(S.getIslCtx(), 1, 0);

  std::string param_name = getName();
  param_name += "_fortranarr_size";
  isl::id IdPwAff = isl::id::alloc(S.getIslCtx(), param_name, this);

  Space = Space.set_dim_id(isl::dim::param, 0, IdPwAff);
  isl::pw_aff PwAff =
      isl::aff::var_on_domain(isl::local_space(Space), isl::dim::param, 0);

  DimensionSizesPw[0] = PwAff;
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
      DimensionSizesPw.push_back(nullptr);
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
  // If this is a Fortran array, then we can print the outermost dimension
  // as a isl_pw_aff even though there is no SCEV information.
  bool IsOutermostSizeKnown = SizeAsPwAff && FAD;

  if (!IsOutermostSizeKnown && getNumberOfDimensions() > 0 &&
      !getDimensionSize(0)) {
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
  isl::ctx Ctx = ArraySpace.get_ctx();
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
  isl::ctx Ctx = ArraySpace.get_ctx();

  auto DimsArray = ArraySpace.dim(isl::dim::set);
  auto DimsAccess = AccessSpace.dim(isl::dim::set);
  auto DimsMissing = DimsArray - DimsAccess;

  auto *BB = getStatement()->getEntryBlock();
  auto &DL = BB->getModule()->getDataLayout();
  unsigned ArrayElemSize = SAI->getElemSizeInBytes();
  unsigned ElemBytes = DL.getTypeAllocSize(getElementType());

  isl::map Map = isl::map::from_domain_and_range(
      isl::set::universe(AccessSpace), isl::set::universe(ArraySpace));

  for (unsigned i = 0; i < DimsMissing; i++)
    Map = Map.fix_si(isl::dim::out, i, 0);

  for (unsigned i = DimsMissing; i < DimsArray; i++)
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
    isl::map Map = isl::map::from_domain_and_range(
        isl::set::universe(ArraySpace), isl::set::universe(ArraySpace));
    for (unsigned i = 0; i < DimsArray - 1; i++)
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
  return stringFromIslObj(AccessRelation.get());
}

isl::space MemoryAccess::getOriginalAccessRelationSpace() const {
  return AccessRelation.get_space();
}

isl::map MemoryAccess::getNewAccessRelation() const {
  return NewAccessRelation;
}

std::string MemoryAccess::getNewAccessRelationStr() const {
  return stringFromIslObj(NewAccessRelation.get());
}

std::string MemoryAccess::getAccessRelationStr() const {
  return getAccessRelation().to_str();
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
void MemoryAccess::assumeNoOutOfBound() {
  if (PollyIgnoreInbounds)
    return;
  auto *SAI = getScopArrayInfo();
  isl::space Space = getOriginalAccessRelationSpace().range();
  isl::set Outside = isl::set::empty(Space);
  for (int i = 1, Size = Space.dim(isl::dim::set); i < Size; ++i) {
    isl::local_space LS(Space);
    isl::pw_aff Var = isl::pw_aff::var_on_domain(LS, isl::dim::set, i);
    isl::pw_aff Zero = isl::pw_aff(LS);

    isl::set DimOutside = Var.lt_set(Zero);
    isl::pw_aff SizeE = SAI->getDimensionSizePw(i);
    SizeE = SizeE.add_dims(isl::dim::in, Space.dim(isl::dim::set));
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
  const auto &Loc = getAccessInstruction()
                        ? getAccessInstruction()->getDebugLoc()
                        : DebugLoc();
  if (!PollyPreciseInbounds)
    Outside = Outside.gist_params(Statement->getDomain().params());
  Statement->getParent()->recordAssumption(INBOUNDS, Outside.release(), Loc,
                                           AS_ASSUMPTION);
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

  if (Range.isWrappedSet() || Range.isSignWrappedSet())
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
    isl::id ParamId =
        give(isl_space_get_dim_id(SpaceSize.get(), isl_dim_param, 0));

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
  if (!PollyPreciseFoldAccesses &&
      isl_map_n_basic_map(NewAccessRelation.get()) >
          isl_map_n_basic_map(AccessRelation.get())) {
  } else {
    AccessRelation = NewAccessRelation;
  }
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
  // to be divisible.
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
  assert(AccessRelation.is_null() && "AccessRelation already built");

  // Initialize the invalid domain which describes all iterations for which the
  // access relation is not modeled correctly.
  isl::set StmtInvalidDomain = getStatement()->getInvalidDomain();
  InvalidDomain = isl::set::empty(StmtInvalidDomain.get_space());

  isl::ctx Ctx = Id.get_ctx();
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
    : Kind(Kind), AccType(AccType), Statement(Stmt), InvalidDomain(nullptr),
      BaseAddr(BaseAddress), ElementType(ElementType),
      Sizes(Sizes.begin(), Sizes.end()), AccessInstruction(AccessInst),
      AccessValue(AccessValue), IsAffine(Affine),
      Subscripts(Subscripts.begin(), Subscripts.end()), AccessRelation(nullptr),
      NewAccessRelation(nullptr), FAD(nullptr) {
  static const std::string TypeStrings[] = {"", "_Read", "_Write", "_MayWrite"};
  const std::string Access = TypeStrings[AccType] + utostr(Stmt->size());

  std::string IdName = Stmt->getBaseName() + Access;
  Id = isl::id::alloc(Stmt->getParent()->getIslCtx(), IdName, this);
}

MemoryAccess::MemoryAccess(ScopStmt *Stmt, AccessType AccType, isl::map AccRel)
    : Kind(MemoryKind::Array), AccType(AccType), Statement(Stmt),
      InvalidDomain(nullptr), AccessRelation(nullptr),
      NewAccessRelation(AccRel), FAD(nullptr) {
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

void MemoryAccess::setFortranArrayDescriptor(Value *FAD) { this->FAD = FAD; }

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

  if (FAD) {
    OS << "[Fortran array descriptor: " << FAD->getName();
    OS << "] ";
  };

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
  isl::set NewInvalidDom = StmtDom.intersect(isl::manage(PWAC.second));
  InvalidDomain = InvalidDomain.unite(NewInvalidDom);
  return isl::manage(PWAC.first);
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
  unsigned lastDimension = Map.dim(isl::dim::in) - 1;

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
  for (unsigned i = 0; i < StrideX.dim(isl::dim::set) - 1; i++)
    StrideX = StrideX.fix_si(isl::dim::set, i, 0);
  StrideX = StrideX.fix_si(isl::dim::set, StrideX.dim(isl::dim::set) - 1,
                           StrideWidth);
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
  assert(NewAccess);

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
    StmtDomain =
        StmtDomain.intersect_params(getStatement()->getParent()->getContext());
    isl::set NewDomain = NewAccess.domain();
    assert(StmtDomain.is_subset(NewDomain) &&
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
  auto Dims = SAI->getNumberOfDimensions();
  assert(NewAccessSpace.dim(isl::dim::set) == Dims &&
         "Access dims must match array dims");
#endif

  NewAccess = NewAccess.gist_domain(getStatement()->getDomain());
  NewAccessRelation = NewAccess;
}

bool MemoryAccess::isLatestPartialAccess() const {
  isl::set StmtDom = getStatement()->getDomain();
  isl::set AccDom = getLatestAccessRelation().domain();

  return isl_set_is_subset(StmtDom.keep(), AccDom.keep()) == isl_bool_false;
}

//===----------------------------------------------------------------------===//

isl::map ScopStmt::getSchedule() const {
  isl::set Domain = getDomain();
  if (Domain.is_empty())
    return isl::map::from_aff(isl::aff(isl::local_space(getDomainSpace())));
  auto Schedule = getParent()->getSchedule();
  if (!Schedule)
    return nullptr;
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
    S.addAccessData(Access);
  }
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

  isl::set Ctx = Parent.getContext();
  InvalidDomain = InvalidDomain.gist_params(Ctx);
  Domain = Domain.gist_params(Ctx);
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

/// Compute the isl representation for the SCEV @p E in this BB.
///
/// @param S                The Scop in which @p BB resides in.
/// @param BB               The BB for which isl representation is to be
/// computed.
/// @param InvalidDomainMap A map of BB to their invalid domains.
/// @param E                The SCEV that should be translated.
/// @param NonNegative      Flag to indicate the @p E has to be non-negative.
///
/// Note that this function will also adjust the invalid context accordingly.

__isl_give isl_pw_aff *
getPwAff(Scop &S, BasicBlock *BB,
         DenseMap<BasicBlock *, isl::set> &InvalidDomainMap, const SCEV *E,
         bool NonNegative = false) {
  PWACtx PWAC = S.getPwAff(E, BB, NonNegative);
  InvalidDomainMap[BB] = InvalidDomainMap[BB].unite(isl::manage(PWAC.second));
  return PWAC.first;
}

/// Build the conditions sets for the switch @p SI in the @p Domain.
///
/// This will fill @p ConditionSets with the conditions under which control
/// will be moved from @p SI to its successors. Hence, @p ConditionSets will
/// have as many elements as @p SI has successors.
static bool
buildConditionSets(Scop &S, BasicBlock *BB, SwitchInst *SI, Loop *L,
                   __isl_keep isl_set *Domain,
                   DenseMap<BasicBlock *, isl::set> &InvalidDomainMap,
                   SmallVectorImpl<__isl_give isl_set *> &ConditionSets) {
  Value *Condition = getConditionFromTerminator(SI);
  assert(Condition && "No condition for switch");

  ScalarEvolution &SE = *S.getSE();
  isl_pw_aff *LHS, *RHS;
  LHS = getPwAff(S, BB, InvalidDomainMap, SE.getSCEVAtScope(Condition, L));

  unsigned NumSuccessors = SI->getNumSuccessors();
  ConditionSets.resize(NumSuccessors);
  for (auto &Case : SI->cases()) {
    unsigned Idx = Case.getSuccessorIndex();
    ConstantInt *CaseValue = Case.getCaseValue();

    RHS = getPwAff(S, BB, InvalidDomainMap, SE.getSCEV(CaseValue));
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

/// Build condition sets for unsigned ICmpInst(s).
/// Special handling is required for unsigned operands to ensure that if
/// MSB (aka the Sign bit) is set for an operands in an unsigned ICmpInst
/// it should wrap around.
///
/// @param IsStrictUpperBound holds information on the predicate relation
/// between TestVal and UpperBound, i.e,
/// TestVal < UpperBound  OR  TestVal <= UpperBound
static __isl_give isl_set *
buildUnsignedConditionSets(Scop &S, BasicBlock *BB, Value *Condition,
                           __isl_keep isl_set *Domain, const SCEV *SCEV_TestVal,
                           const SCEV *SCEV_UpperBound,
                           DenseMap<BasicBlock *, isl::set> &InvalidDomainMap,
                           bool IsStrictUpperBound) {
  // Do not take NonNeg assumption on TestVal
  // as it might have MSB (Sign bit) set.
  isl_pw_aff *TestVal = getPwAff(S, BB, InvalidDomainMap, SCEV_TestVal, false);
  // Take NonNeg assumption on UpperBound.
  isl_pw_aff *UpperBound =
      getPwAff(S, BB, InvalidDomainMap, SCEV_UpperBound, true);

  // 0 <= TestVal
  isl_set *First =
      isl_pw_aff_le_set(isl_pw_aff_zero_on_domain(isl_local_space_from_space(
                            isl_pw_aff_get_domain_space(TestVal))),
                        isl_pw_aff_copy(TestVal));

  isl_set *Second;
  if (IsStrictUpperBound)
    // TestVal < UpperBound
    Second = isl_pw_aff_lt_set(TestVal, UpperBound);
  else
    // TestVal <= UpperBound
    Second = isl_pw_aff_le_set(TestVal, UpperBound);

  isl_set *ConsequenceCondSet = isl_set_intersect(First, Second);
  ConsequenceCondSet = setDimensionIds(Domain, ConsequenceCondSet);
  return ConsequenceCondSet;
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
buildConditionSets(Scop &S, BasicBlock *BB, Value *Condition,
                   TerminatorInst *TI, Loop *L, __isl_keep isl_set *Domain,
                   DenseMap<BasicBlock *, isl::set> &InvalidDomainMap,
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

    bool Valid = buildConditionSets(S, BB, BinOp->getOperand(0), TI, L, Domain,
                                    InvalidDomainMap, ConditionSets) &&
                 buildConditionSets(S, BB, BinOp->getOperand(1), TI, L, Domain,
                                    InvalidDomainMap, ConditionSets);
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
    const SCEV *LeftOperand = SE.getSCEVAtScope(ICond->getOperand(0), L),
               *RightOperand = SE.getSCEVAtScope(ICond->getOperand(1), L);

    switch (ICond->getPredicate()) {
    case ICmpInst::ICMP_ULT:
      ConsequenceCondSet =
          buildUnsignedConditionSets(S, BB, Condition, Domain, LeftOperand,
                                     RightOperand, InvalidDomainMap, true);
      break;
    case ICmpInst::ICMP_ULE:
      ConsequenceCondSet =
          buildUnsignedConditionSets(S, BB, Condition, Domain, LeftOperand,
                                     RightOperand, InvalidDomainMap, false);
      break;
    case ICmpInst::ICMP_UGT:
      ConsequenceCondSet =
          buildUnsignedConditionSets(S, BB, Condition, Domain, RightOperand,
                                     LeftOperand, InvalidDomainMap, true);
      break;
    case ICmpInst::ICMP_UGE:
      ConsequenceCondSet =
          buildUnsignedConditionSets(S, BB, Condition, Domain, RightOperand,
                                     LeftOperand, InvalidDomainMap, false);
      break;
    default:
      LHS = getPwAff(S, BB, InvalidDomainMap, LeftOperand, NonNeg);
      RHS = getPwAff(S, BB, InvalidDomainMap, RightOperand, NonNeg);
      ConsequenceCondSet =
          buildConditionSet(ICond->getPredicate(), LHS, RHS, Domain);
      break;
    }
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
    S.invalidate(COMPLEXITY, TI ? TI->getDebugLoc() : DebugLoc(),
                 TI ? TI->getParent() : nullptr /* BasicBlock */);
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
buildConditionSets(Scop &S, BasicBlock *BB, TerminatorInst *TI, Loop *L,
                   __isl_keep isl_set *Domain,
                   DenseMap<BasicBlock *, isl::set> &InvalidDomainMap,
                   SmallVectorImpl<__isl_give isl_set *> &ConditionSets) {
  if (SwitchInst *SI = dyn_cast<SwitchInst>(TI))
    return buildConditionSets(S, BB, SI, L, Domain, InvalidDomainMap,
                              ConditionSets);

  assert(isa<BranchInst>(TI) && "Terminator was neither branch nor switch.");

  if (TI->getNumSuccessors() == 1) {
    ConditionSets.push_back(isl_set_copy(Domain));
    return true;
  }

  Value *Condition = getConditionFromTerminator(TI);
  assert(Condition && "No condition for Terminator");

  return buildConditionSets(S, BB, Condition, TI, L, Domain, InvalidDomainMap,
                            ConditionSets);
}

void ScopStmt::buildDomain() {
  isl::id Id = isl::id::alloc(getIslCtx(), getBaseName(), this);

  Domain = getParent()->getDomainConditions(this);
  Domain = Domain.set_tuple_id(Id);
}

void ScopStmt::collectSurroundingLoops() {
  for (unsigned u = 0, e = Domain.dim(isl::dim::set); u < e; u++) {
    isl::id DimId = Domain.get_dim_id(isl::dim::set, u);
    NestLoops.push_back(static_cast<Loop *>(DimId.get_user()));
  }
}

ScopStmt::ScopStmt(Scop &parent, Region &R, Loop *SurroundingLoop)
    : Parent(parent), InvalidDomain(nullptr), Domain(nullptr), R(&R),
      Build(nullptr), SurroundingLoop(SurroundingLoop) {
  BaseName = getIslCompatibleName(
      "Stmt", R.getNameStr(), parent.getNextStmtIdx(), "", UseInstructionNames);
}

ScopStmt::ScopStmt(Scop &parent, BasicBlock &bb, Loop *SurroundingLoop,
                   std::vector<Instruction *> Instructions)
    : Parent(parent), InvalidDomain(nullptr), Domain(nullptr), BB(&bb),
      Build(nullptr), SurroundingLoop(SurroundingLoop),
      Instructions(Instructions) {
  BaseName = getIslCompatibleName("Stmt", &bb, parent.getNextStmtIdx(), "",
                                  UseInstructionNames);
}

ScopStmt::ScopStmt(Scop &parent, isl::map SourceRel, isl::map TargetRel,
                   isl::set NewDomain)
    : Parent(parent), InvalidDomain(nullptr), Domain(NewDomain),
      Build(nullptr) {
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
    isl::map LoadAccs = CandidatePair.first->getAccessRelation();
    isl::map StoreAccs = CandidatePair.second->getAccessRelation();

    // Skip those with obviously unequal base addresses.
    if (!LoadAccs.has_equal_space(StoreAccs)) {
      continue;
    }

    // And check if the remaining for overlap with other memory accesses.
    isl::map AllAccsRel = LoadAccs.unite(StoreAccs);
    AllAccsRel = AllAccsRel.intersect_domain(getDomain());
    isl::set AllAccs = AllAccsRel.range();

    for (MemoryAccess *MA : MemAccs) {
      if (MA == CandidatePair.first || MA == CandidatePair.second)
        continue;

      isl::map AccRel = MA->getAccessRelation().intersect_domain(getDomain());
      isl::set Accs = AccRel.range();

      if (AllAccs.has_equal_space(Accs)) {
        isl::set OverlapAccs = Accs.intersect(AllAccs);
        Valid = Valid && OverlapAccs.is_empty();
      }
    }

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

std::string ScopStmt::getDomainStr() const { return Domain.to_str(); }

std::string ScopStmt::getScheduleStr() const {
  auto *S = getSchedule().release();
  if (!S)
    return {};
  auto Str = stringFromIslObj(S);
  isl_map_free(S);
  return Str;
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

isl_ctx *ScopStmt::getIslCtx() const { return Parent.getIslCtx(); }

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

  if (PrintInstructions && isBlockStmt())
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
  MemAccs.erase(std::remove_if(MemAccs.begin(), MemAccs.end(), Predicate),
                MemAccs.end());
  InstructionToAccess.erase(MA->getAccessInstruction());
}

void ScopStmt::removeSingleMemoryAccess(MemoryAccess *MA) {
  auto MAIt = std::find(MemAccs.begin(), MemAccs.end(), MA);
  assert(MAIt != MemAccs.end());
  MemAccs.erase(MAIt);

  removeAccessData(MA);
  Parent.removeAccessData(MA);

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

void Scop::setContext(__isl_take isl_set *NewContext) {
  NewContext = isl_set_align_params(NewContext, isl_set_get_space(Context));
  isl_set_free(Context);
  Context = NewContext;
}

namespace {

/// Remap parameter values but keep AddRecs valid wrt. invariant loads.
struct SCEVSensitiveParameterRewriter
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
struct SCEVFindInsideScop : public SCEVTraversal<SCEVFindInsideScop> {
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

// This table of function names is used to translate parameter names in more
// human-readable names. This makes it easier to interpret Polly analysis
// results.
StringMap<std::string> KnownNames = {
    {"_Z13get_global_idj", "global_id"},
    {"_Z12get_local_idj", "local_id"},
    {"_Z15get_global_sizej", "global_size"},
    {"_Z14get_local_sizej", "local_size"},
    {"_Z12get_work_dimv", "work_dim"},
    {"_Z17get_global_offsetj", "global_offset"},
    {"_Z12get_group_idj", "group_id"},
    {"_Z14get_num_groupsj", "num_groups"},
};

static std::string getCallParamName(CallInst *Call) {
  std::string Result;
  raw_string_ostream OS(Result);
  std::string Name = Call->getCalledFunction()->getName();

  auto Iterator = KnownNames.find(Name);
  if (Iterator != KnownNames.end())
    Name = "__" + Iterator->getValue();
  OS << Name;
  for (auto &Operand : Call->arg_operands()) {
    ConstantInt *Op = cast<ConstantInt>(&Operand);
    OS << "_" << Op->getValue();
  }
  OS.flush();
  return Result;
}

void Scop::createParameterId(const SCEV *Parameter) {
  assert(Parameters.count(Parameter));
  assert(!ParameterIds.count(Parameter));

  std::string ParameterName = "p_" + std::to_string(getNumParams() - 1);

  if (const SCEVUnknown *ValueParameter = dyn_cast<SCEVUnknown>(Parameter)) {
    Value *Val = ValueParameter->getValue();
    CallInst *Call = dyn_cast<CallInst>(Val);

    if (Call && isConstCall(Call)) {
      ParameterName = getCallParamName(Call);
    } else if (UseInstructionNames) {
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

isl::set Scop::addNonEmptyDomainConstraints(isl::set C) const {
  isl_set *DomainContext = isl_union_set_params(getDomains().release());
  return isl::manage(isl_set_intersect_params(C.release(), DomainContext));
}

bool Scop::isDominatedBy(const DominatorTree &DT, BasicBlock *BB) const {
  return DT.dominates(BB, getEntry());
}

void Scop::addUserAssumptions(
    AssumptionCache &AC, DominatorTree &DT, LoopInfo &LI,
    DenseMap<BasicBlock *, isl::set> &InvalidDomainMap) {
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
      ORE.emit(
          OptimizationRemarkAnalysis(DEBUG_TYPE, "IgnoreUserAssumption", CI)
          << "Non-affine user assumption ignored.");
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
    BasicBlock *BB = InScop ? CI->getParent() : getRegion().getEntry();
    auto *Dom = InScop ? DomainMap[BB].copy() : isl_set_copy(Context);
    assert(Dom && "Cannot propagate a nullptr.");
    bool Valid = buildConditionSets(*this, BB, Val, TI, L, Dom,
                                    InvalidDomainMap, ConditionSets);
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
    ORE.emit(OptimizationRemarkAnalysis(DEBUG_TYPE, "UserAssumption", CI)
             << "Use user assumption: " << stringFromIslObj(AssumptionCtx));
    Context = isl_set_intersect(Context, AssumptionCtx);
  }
}

void Scop::addUserContext() {
  if (UserContextStr.empty())
    return;

  isl_set *UserContext =
      isl_set_read_from_str(getIslCtx(), UserContextStr.c_str());
  isl_space *Space = getParamSpace().release();
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
    Context =
        addRangeBoundsToSet(give(Context), SRange, PDim++, isl::dim::param)
            .release();
  }
}

static std::vector<isl::id> getFortranArrayIds(Scop::array_range Arrays) {
  std::vector<isl::id> OutermostSizeIds;
  for (auto Array : Arrays) {
    // To check if an array is a Fortran array, we check if it has a isl_pw_aff
    // for its outermost dimension. Fortran arrays will have this since the
    // outermost dimension size can be picked up from their runtime description.
    // TODO: actually need to check if it has a FAD, but for now this works.
    if (Array->getNumberOfDimensions() > 0) {
      isl::pw_aff PwAff = Array->getDimensionSizePw(0);
      if (!PwAff)
        continue;

      isl::id Id =
          isl::manage(isl_pw_aff_get_dim_id(PwAff.get(), isl_dim_param, 0));
      assert(!Id.is_null() &&
             "Invalid Id for PwAff expression in Fortran array");
      Id.dump();
      OutermostSizeIds.push_back(Id);
    }
  }
  return OutermostSizeIds;
}

// The FORTRAN array size parameters are known to be non-negative.
static isl_set *boundFortranArrayParams(__isl_give isl_set *Context,
                                        Scop::array_range Arrays) {
  std::vector<isl::id> OutermostSizeIds;
  OutermostSizeIds = getFortranArrayIds(Arrays);

  for (isl::id Id : OutermostSizeIds) {
    int dim = isl_set_find_dim_by_id(Context, isl_dim_param, Id.get());
    Context = isl_set_lower_bound_si(Context, isl_dim_param, dim, 0);
  }

  return Context;
}

void Scop::realignParams() {
  if (PollyIgnoreParamBounds)
    return;

  // Add all parameters into a common model.
  isl::space Space = getFullParamSpace();

  // Align the parameters of all data structures to the model.
  Context = isl_set_align_params(Context, Space.copy());

  // Bound the size of the fortran array dimensions.
  Context = boundFortranArrayParams(Context, arrays());

  // As all parameters are known add bounds to them.
  addParameterBounds();

  for (ScopStmt &Stmt : *this)
    Stmt.realignParams();
  // Simplify the schedule according to the context too.
  Schedule = isl_schedule_gist_domain_params(Schedule, getContext().release());
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
    isl_set *DomainParameters = isl_union_set_params(S.getDomains().release());
    AssumptionContext =
        isl_set_gist_params(AssumptionContext, DomainParameters);
  }

  AssumptionContext =
      isl_set_gist_params(AssumptionContext, S.getContext().release());
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
  InvalidContext =
      isl_set_align_params(InvalidContext, getParamSpace().release());
}

/// Add the minimal/maximal access in @p Set to @p User.
static isl::stat
buildMinMaxAccess(isl::set Set, Scop::MinMaxVectorTy &MinMaxAccesses, Scop &S) {
  isl::pw_multi_aff MinPMA, MaxPMA;
  isl::pw_aff LastDimAff;
  isl::aff OneAff;
  unsigned Pos;
  isl::ctx Ctx = Set.get_ctx();

  Set = Set.remove_divs();

  if (isl_set_n_basic_set(Set.get()) >= MaxDisjunctsInDomain)
    return isl::stat::error;

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
  if (isl_set_n_param(Set.get()) > RunTimeChecksMaxParameters) {
    unsigned InvolvedParams = 0;
    for (unsigned u = 0, e = isl_set_n_param(Set.get()); u < e; u++)
      if (Set.involves_dims(isl::dim::param, u, 1))
        InvolvedParams++;

    if (InvolvedParams > RunTimeChecksMaxParameters)
      return isl::stat::error;
  }

  if (isl_set_n_basic_set(Set.get()) > RunTimeChecksMaxAccessDisjuncts)
    return isl::stat::error;

  MinPMA = Set.lexmin_pw_multi_aff();
  MaxPMA = Set.lexmax_pw_multi_aff();

  if (isl_ctx_last_error(Ctx.get()) == isl_error_quota)
    return isl::stat::error;

  MinPMA = MinPMA.coalesce();
  MaxPMA = MaxPMA.coalesce();

  // Adjust the last dimension of the maximal access by one as we want to
  // enclose the accessed memory region by MinPMA and MaxPMA. The pointer
  // we test during code generation might now point after the end of the
  // allocated array but we will never dereference it anyway.
  assert(MaxPMA.dim(isl::dim::out) && "Assumed at least one output dimension");
  Pos = MaxPMA.dim(isl::dim::out) - 1;
  LastDimAff = MaxPMA.get_pw_aff(Pos);
  OneAff = isl::aff(isl::local_space(LastDimAff.get_domain_space()));
  OneAff = OneAff.add_constant_si(1);
  LastDimAff = LastDimAff.add(OneAff);
  MaxPMA = MaxPMA.set_pw_aff(Pos, LastDimAff);

  MinMaxAccesses.push_back(std::make_pair(MinPMA.copy(), MaxPMA.copy()));

  return isl::stat::ok;
}

static __isl_give isl_set *getAccessDomain(MemoryAccess *MA) {
  isl_set *Domain = MA->getStatement()->getDomain().release();
  Domain = isl_set_project_out(Domain, isl_dim_set, 0, isl_set_n_dim(Domain));
  return isl_set_reset_tuple_id(Domain);
}

/// Wrapper function to calculate minimal/maximal accesses to each array.
static bool calculateMinMaxAccess(Scop::AliasGroupTy AliasGroup, Scop &S,
                                  Scop::MinMaxVectorTy &MinMaxAccesses) {
  MinMaxAccesses.reserve(AliasGroup.size());

  isl::union_set Domains = S.getDomains();
  isl::union_map Accesses = isl::union_map::empty(S.getParamSpace());

  for (MemoryAccess *MA : AliasGroup)
    Accesses = Accesses.add_map(give(MA->getAccessRelation().release()));

  Accesses = Accesses.intersect_domain(Domains);
  isl::union_set Locations = Accesses.range();
  Locations = Locations.coalesce();
  Locations = Locations.detect_equalities();

  auto Lambda = [&MinMaxAccesses, &S](isl::set Set) -> isl::stat {
    return buildMinMaxAccess(Set, MinMaxAccesses, S);
  };
  return Locations.foreach_set(Lambda) == isl::stat::ok;
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
    // where the immediate predecessor is part of a loop, we assume these
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
unsigned getNumBlocksInLoop(Loop *L) {
  unsigned NumBlocks = L->getNumBlocks();
  SmallVector<BasicBlock *, 4> ExitBlocks;
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

bool Scop::buildDomains(Region *R, DominatorTree &DT, LoopInfo &LI,
                        DenseMap<BasicBlock *, isl::set> &InvalidDomainMap) {
  bool IsOnlyNonAffineRegion = isNonAffineSubRegion(R);
  auto *EntryBB = R->getEntry();
  auto *L = IsOnlyNonAffineRegion ? nullptr : LI.getLoopFor(EntryBB);
  int LD = getRelativeLoopDepth(L);
  auto *S = isl_set_universe(isl_space_set_alloc(getIslCtx(), 0, LD + 1));

  while (LD-- >= 0) {
    S = addDomainDimId(S, LD + 1, L);
    L = L->getParentLoop();
  }

  InvalidDomainMap[EntryBB] = isl::manage(isl_set_empty(isl_set_get_space(S)));
  DomainMap[EntryBB] = isl::manage(S);

  if (IsOnlyNonAffineRegion)
    return !containsErrorBlock(R->getNode(), *R, LI, DT);

  if (!buildDomainsWithBranchConstraints(R, DT, LI, InvalidDomainMap))
    return false;

  if (!propagateDomainConstraints(R, DT, LI, InvalidDomainMap))
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
  if (!propagateInvalidStmtDomains(R, DT, LI, InvalidDomainMap))
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

bool Scop::propagateInvalidStmtDomains(
    Region *R, DominatorTree &DT, LoopInfo &LI,
    DenseMap<BasicBlock *, isl::set> &InvalidDomainMap) {
  ReversePostOrderTraversal<Region *> RTraversal(R);
  for (auto *RN : RTraversal) {

    // Recurse for affine subregions but go on for basic blocks and non-affine
    // subregions.
    if (RN->isSubRegion()) {
      Region *SubRegion = RN->getNodeAs<Region>();
      if (!isNonAffineSubRegion(SubRegion)) {
        propagateInvalidStmtDomains(SubRegion, DT, LI, InvalidDomainMap);
        continue;
      }
    }

    bool ContainsErrorBlock = containsErrorBlock(RN, getRegion(), LI, DT);
    BasicBlock *BB = getRegionNodeBasicBlock(RN);
    isl::set &Domain = DomainMap[BB];
    assert(Domain && "Cannot propagate a nullptr");

    isl::set InvalidDomain = InvalidDomainMap[BB];

    bool IsInvalidBlock = ContainsErrorBlock || Domain.is_subset(InvalidDomain);

    if (!IsInvalidBlock) {
      InvalidDomain = InvalidDomain.intersect(Domain);
    } else {
      InvalidDomain = Domain;
      isl::set DomPar = Domain.params();
      recordAssumption(ERRORBLOCK, DomPar.release(),
                       BB->getTerminator()->getDebugLoc(), AS_RESTRICTION);
      Domain = nullptr;
    }

    if (InvalidDomain.is_empty()) {
      InvalidDomainMap[BB] = InvalidDomain;
      continue;
    }

    auto *BBLoop = getRegionNodeLoop(RN, LI);
    auto *TI = BB->getTerminator();
    unsigned NumSuccs = RN->isSubRegion() ? 1 : TI->getNumSuccessors();
    for (unsigned u = 0; u < NumSuccs; u++) {
      auto *SuccBB = getRegionNodeSuccessor(RN, TI, u);

      // Skip successors outside the SCoP.
      if (!contains(SuccBB))
        continue;

      // Skip backedges.
      if (DT.dominates(SuccBB, BB))
        continue;

      Loop *SuccBBLoop = getFirstNonBoxedLoopFor(SuccBB, LI, getBoxedLoops());

      auto *AdjustedInvalidDomain = adjustDomainDimensions(
          *this, InvalidDomain.copy(), BBLoop, SuccBBLoop);

      auto *SuccInvalidDomain = InvalidDomainMap[SuccBB].copy();
      SuccInvalidDomain =
          isl_set_union(SuccInvalidDomain, AdjustedInvalidDomain);
      SuccInvalidDomain = isl_set_coalesce(SuccInvalidDomain);
      unsigned NumConjucts = isl_set_n_basic_set(SuccInvalidDomain);

      InvalidDomainMap[SuccBB] = isl::manage(SuccInvalidDomain);

      // Check if the maximal number of domain disjunctions was reached.
      // In case this happens we will bail.
      if (NumConjucts < MaxDisjunctsInDomain)
        continue;

      InvalidDomainMap.erase(BB);
      invalidate(COMPLEXITY, TI->getDebugLoc(), TI->getParent());
      return false;
    }

    InvalidDomainMap[BB] = InvalidDomain;
  }

  return true;
}

void Scop::propagateDomainConstraintsToRegionExit(
    BasicBlock *BB, Loop *BBLoop,
    SmallPtrSetImpl<BasicBlock *> &FinishedExitBlocks, LoopInfo &LI,
    DenseMap<BasicBlock *, isl::set> &InvalidDomainMap) {
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

  isl::set Domain = DomainMap[BB];
  assert(Domain && "Cannot propagate a nullptr");

  Loop *ExitBBLoop = getFirstNonBoxedLoopFor(ExitBB, LI, getBoxedLoops());

  // Since the dimensions of @p BB and @p ExitBB might be different we have to
  // adjust the domain before we can propagate it.
  isl::set AdjustedDomain = isl::manage(
      adjustDomainDimensions(*this, Domain.copy(), BBLoop, ExitBBLoop));
  isl::set &ExitDomain = DomainMap[ExitBB];

  // If the exit domain is not yet created we set it otherwise we "add" the
  // current domain.
  ExitDomain = ExitDomain ? AdjustedDomain.unite(ExitDomain) : AdjustedDomain;

  // Initialize the invalid domain.
  InvalidDomainMap[ExitBB] = ExitDomain.empty(ExitDomain.get_space());

  FinishedExitBlocks.insert(ExitBB);
}

bool Scop::buildDomainsWithBranchConstraints(
    Region *R, DominatorTree &DT, LoopInfo &LI,
    DenseMap<BasicBlock *, isl::set> &InvalidDomainMap) {
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
        if (!buildDomainsWithBranchConstraints(SubRegion, DT, LI,
                                               InvalidDomainMap))
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

    isl::set Domain = DomainMap.lookup(BB);
    if (!Domain)
      continue;
    MaxLoopDepth = std::max(MaxLoopDepth, isl_set_n_dim(Domain.get()));

    auto *BBLoop = getRegionNodeLoop(RN, LI);
    // Propagate the domain from BB directly to blocks that have a superset
    // domain, at the moment only region exit nodes of regions that start in BB.
    propagateDomainConstraintsToRegionExit(BB, BBLoop, FinishedExitBlocks, LI,
                                           InvalidDomainMap);

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
      ConditionSets.push_back(Domain.copy());
    else if (!buildConditionSets(*this, BB, TI, BBLoop, Domain.get(),
                                 InvalidDomainMap, ConditionSets))
      return false;

    // Now iterate over the successors and set their initial domain based on
    // their condition set. We skip back edges here and have to be careful when
    // we leave a loop not to keep constraints over a dimension that doesn't
    // exist anymore.
    assert(RN->isSubRegion() || TI->getNumSuccessors() == ConditionSets.size());
    for (unsigned u = 0, e = ConditionSets.size(); u < e; u++) {
      isl::set CondSet = isl::manage(ConditionSets[u]);
      BasicBlock *SuccBB = getRegionNodeSuccessor(RN, TI, u);

      // Skip blocks outside the region.
      if (!contains(SuccBB))
        continue;

      // If we propagate the domain of some block to "SuccBB" we do not have to
      // adjust the domain.
      if (FinishedExitBlocks.count(SuccBB))
        continue;

      // Skip back edges.
      if (DT.dominates(SuccBB, BB))
        continue;

      Loop *SuccBBLoop = getFirstNonBoxedLoopFor(SuccBB, LI, getBoxedLoops());

      CondSet = isl::manage(
          adjustDomainDimensions(*this, CondSet.copy(), BBLoop, SuccBBLoop));

      // Set the domain for the successor or merge it with an existing domain in
      // case there are multiple paths (without loop back edges) to the
      // successor block.
      isl::set &SuccDomain = DomainMap[SuccBB];

      if (SuccDomain) {
        SuccDomain = SuccDomain.unite(CondSet).coalesce();
      } else {
        // Initialize the invalid domain.
        InvalidDomainMap[SuccBB] = CondSet.empty(CondSet.get_space());
        SuccDomain = CondSet;
      }

      SuccDomain = SuccDomain.detect_equalities();

      // Check if the maximal number of domain disjunctions was reached.
      // In case this happens we will clean up and bail.
      if (isl_set_n_basic_set(SuccDomain.get()) < MaxDisjunctsInDomain)
        continue;

      invalidate(COMPLEXITY, DebugLoc());
      while (++u < ConditionSets.size())
        isl_set_free(ConditionSets[u]);
      return false;
    }
  }

  return true;
}

isl::set Scop::getPredecessorDomainConstraints(BasicBlock *BB, isl::set Domain,
                                               DominatorTree &DT,
                                               LoopInfo &LI) {
  // If @p BB is the ScopEntry we are done
  if (R.getEntry() == BB)
    return isl::set::universe(Domain.get_space());

  // The region info of this function.
  auto &RI = *R.getRegionInfo();

  Loop *BBLoop = getFirstNonBoxedLoopFor(BB, LI, getBoxedLoops());

  // A domain to collect all predecessor domains, thus all conditions under
  // which the block is executed. To this end we start with the empty domain.
  isl::set PredDom = isl::set::empty(Domain.get_space());

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

    auto *PredBBDom = getDomainConditions(PredBB).release();
    Loop *PredBBLoop = getFirstNonBoxedLoopFor(PredBB, LI, getBoxedLoops());

    PredBBDom = adjustDomainDimensions(*this, PredBBDom, PredBBLoop, BBLoop);

    PredDom = PredDom.unite(isl::manage(PredBBDom));
  }

  return PredDom;
}

bool Scop::propagateDomainConstraints(
    Region *R, DominatorTree &DT, LoopInfo &LI,
    DenseMap<BasicBlock *, isl::set> &InvalidDomainMap) {
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
        if (!propagateDomainConstraints(SubRegion, DT, LI, InvalidDomainMap))
          return false;
        continue;
      }
    }

    BasicBlock *BB = getRegionNodeBasicBlock(RN);
    isl::set &Domain = DomainMap[BB];
    assert(Domain);

    // Under the union of all predecessor conditions we can reach this block.
    isl::set PredDom = getPredecessorDomainConstraints(BB, Domain, DT, LI);
    Domain = Domain.intersect(PredDom).coalesce();
    Domain = Domain.align_params(getParamSpace());

    Loop *BBLoop = getRegionNodeLoop(RN, LI);
    if (BBLoop && BBLoop->getHeader() == BB && contains(BBLoop))
      if (!addLoopBoundsToHeaderDomain(BBLoop, LI, InvalidDomainMap))
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

bool Scop::addLoopBoundsToHeaderDomain(
    Loop *L, LoopInfo &LI, DenseMap<BasicBlock *, isl::set> &InvalidDomainMap) {
  int LoopDepth = getRelativeLoopDepth(L);
  assert(LoopDepth >= 0 && "Loop in region should have at least depth one");

  BasicBlock *HeaderBB = L->getHeader();
  assert(DomainMap.count(HeaderBB));
  isl::set &HeaderBBDom = DomainMap[HeaderBB];

  isl::map NextIterationMap = isl::manage(
      createNextIterationMap(HeaderBBDom.get_space().release(), LoopDepth));

  isl::set UnionBackedgeCondition = HeaderBBDom.empty(HeaderBBDom.get_space());

  SmallVector<BasicBlock *, 4> LatchBlocks;
  L->getLoopLatches(LatchBlocks);

  for (BasicBlock *LatchBB : LatchBlocks) {
    // If the latch is only reachable via error statements we skip it.
    isl::set LatchBBDom = DomainMap.lookup(LatchBB);
    if (!LatchBBDom)
      continue;

    isl::set BackedgeCondition = nullptr;

    TerminatorInst *TI = LatchBB->getTerminator();
    BranchInst *BI = dyn_cast<BranchInst>(TI);
    assert(BI && "Only branch instructions allowed in loop latches");

    if (BI->isUnconditional())
      BackedgeCondition = LatchBBDom;
    else {
      SmallVector<isl_set *, 8> ConditionSets;
      int idx = BI->getSuccessor(0) != HeaderBB;
      if (!buildConditionSets(*this, LatchBB, TI, L, LatchBBDom.get(),
                              InvalidDomainMap, ConditionSets))
        return false;

      // Free the non back edge condition set as we do not need it.
      isl_set_free(ConditionSets[1 - idx]);

      BackedgeCondition = isl::manage(ConditionSets[idx]);
    }

    int LatchLoopDepth = getRelativeLoopDepth(LI.getLoopFor(LatchBB));
    assert(LatchLoopDepth >= LoopDepth);
    BackedgeCondition = BackedgeCondition.project_out(
        isl::dim::set, LoopDepth + 1, LatchLoopDepth - LoopDepth);
    UnionBackedgeCondition = UnionBackedgeCondition.unite(BackedgeCondition);
  }

  isl::map ForwardMap = ForwardMap.lex_le(HeaderBBDom.get_space());
  for (int i = 0; i < LoopDepth; i++)
    ForwardMap = ForwardMap.equate(isl::dim::in, i, isl::dim::out, i);

  isl::set UnionBackedgeConditionComplement =
      UnionBackedgeCondition.complement();
  UnionBackedgeConditionComplement =
      UnionBackedgeConditionComplement.lower_bound_si(isl::dim::set, LoopDepth,
                                                      0);
  UnionBackedgeConditionComplement =
      UnionBackedgeConditionComplement.apply(ForwardMap);
  HeaderBBDom = HeaderBBDom.subtract(UnionBackedgeConditionComplement);
  HeaderBBDom = HeaderBBDom.apply(NextIterationMap);

  auto Parts = partitionSetParts(HeaderBBDom.copy(), LoopDepth);
  HeaderBBDom = isl::manage(Parts.second);

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
                                        isl::union_map Writes) {
  if (auto *BasePtrMA = lookupBasePtrAccess(MA)) {
    return getNonHoistableCtx(BasePtrMA, Writes).is_null();
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

    isl_set *StmtDomain = Stmt.getDomain().release();
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
    if (!hasFeasibleRuntimeContext())
      return false;

    {
      IslMaxOperationsGuard MaxOpGuard(getIslCtx(), OptComputeOut);
      bool Valid = buildAliasGroup(AG, HasWriteAccess);
      if (!Valid)
        return false;
    }
    if (isl_ctx_last_error(getIslCtx()) == isl_error_quota) {
      invalidate(COMPLEXITY, DebugLoc());
      return false;
    }
  }

  return true;
}

bool Scop::buildAliasGroup(Scop::AliasGroupTy &AliasGroup,
                           DenseSet<const ScopArrayInfo *> HasWriteAccess) {
  AliasGroupTy ReadOnlyAccesses;
  AliasGroupTy ReadWriteAccesses;
  SmallPtrSet<const ScopArrayInfo *, 4> ReadWriteArrays;
  SmallPtrSet<const ScopArrayInfo *, 4> ReadOnlyArrays;

  if (AliasGroup.size() < 2)
    return true;

  for (MemoryAccess *Access : AliasGroup) {
    ORE.emit(OptimizationRemarkAnalysis(DEBUG_TYPE, "PossibleAlias",
                                        Access->getAccessInstruction())
             << "Possibly aliasing pointer, use restrict keyword.");
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
      invalidate(ALIASING, MA->getAccessInstruction()->getDebugLoc(),
                 MA->getAccessInstruction()->getParent());
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

int Scop::NextScopID = 0;

std::string Scop::CurrentFunc;

int Scop::getNextID(std::string ParentFunc) {
  if (ParentFunc != CurrentFunc) {
    CurrentFunc = ParentFunc;
    NextScopID = 0;
  }
  return NextScopID++;
}

Scop::Scop(Region &R, ScalarEvolution &ScalarEvolution, LoopInfo &LI,
           ScopDetection::DetectionContext &DC, OptimizationRemarkEmitter &ORE)
    : SE(&ScalarEvolution), R(R), name(R.getNameStr()),
      HasSingleExitEdge(R.getExitingBlock()), DC(DC), ORE(ORE),
      IslCtx(isl_ctx_alloc(), isl_ctx_free), Affinator(this, LI),
      ID(getNextID((*R.getEntry()->getParent()).getName().str())) {
  if (IslOnErrorAbort)
    isl_options_set_on_error(getIslCtx(), ISL_ON_ERROR_ABORT);
  buildContext();
}

Scop::~Scop() {
  isl_set_free(Context);
  isl_set_free(AssumedContext);
  isl_set_free(InvalidContext);
  isl_schedule_free(Schedule);

  ParameterIds.clear();

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

void Scop::foldSizeConstantsToRight() {
  isl_union_set *Accessed = isl_union_map_range(getAccesses().release());

  for (auto Array : arrays()) {
    if (Array->getNumberOfDimensions() <= 1)
      continue;

    isl_space *Space = Array->getSpace().release();

    Space = isl_space_align_params(Space, isl_union_set_get_space(Accessed));

    if (!isl_union_set_contains(Accessed, Space)) {
      isl_space_free(Space);
      continue;
    }

    isl_set *Elements = isl_union_set_extract_set(Accessed, Space);

    isl_map *Transform =
        isl_map_universe(isl_space_map_from_set(Array->getSpace().release()));

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
        Access->setAccessRelation(Access->getAccessRelation().apply_range(
            isl::manage(isl_map_copy(Transform))));

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
}

void Scop::markFortranArrays() {
  for (ScopStmt &Stmt : Stmts) {
    for (MemoryAccess *MemAcc : Stmt) {
      Value *FAD = MemAcc->getFortranArrayDescriptor();
      if (!FAD)
        continue;

      // TODO: const_cast-ing to edit
      ScopArrayInfo *SAI =
          const_cast<ScopArrayInfo *>(MemAcc->getLatestScopArrayInfo());
      assert(SAI && "memory access into a Fortran array does not "
                    "have an associated ScopArrayInfo");
      SAI->applyAndSetFAD(FAD);
    }
  }
}

void Scop::finalizeAccesses() {
  updateAccessDimensionality();
  foldSizeConstantsToRight();
  foldAccessRelations();
  assumeNoOutOfBounds();
  markFortranArrays();
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

void Scop::removeFromStmtMap(ScopStmt &Stmt) {
  if (Stmt.isRegionStmt())
    for (BasicBlock *BB : Stmt.getRegion()->blocks()) {
      StmtMap.erase(BB);
      for (Instruction &Inst : *BB)
        InstStmtMap.erase(&Inst);
    }
  else {
    StmtMap.erase(Stmt.getBasicBlock());
    for (Instruction *Inst : Stmt.getInstructions())
      InstStmtMap.erase(Inst);
  }
}

void Scop::removeStmts(std::function<bool(ScopStmt &)> ShouldDelete) {
  for (auto StmtIt = Stmts.begin(), StmtEnd = Stmts.end(); StmtIt != StmtEnd;) {
    if (!ShouldDelete(*StmtIt)) {
      StmtIt++;
      continue;
    }

    removeFromStmtMap(*StmtIt);
    StmtIt = Stmts.erase(StmtIt);
  }
}

void Scop::removeStmtNotInDomainMap() {
  auto ShouldDelete = [this](ScopStmt &Stmt) -> bool {
    return !this->DomainMap.lookup(Stmt.getEntryBlock());
  };
  removeStmts(ShouldDelete);
}

void Scop::simplifySCoP(bool AfterHoisting) {
  auto ShouldDelete = [AfterHoisting](ScopStmt &Stmt) -> bool {
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
  };

  removeStmts(ShouldDelete);
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

bool isAParameter(llvm::Value *maybeParam, const Function &F) {
  for (const llvm::Argument &Arg : F.args())
    if (&Arg == maybeParam)
      return true;

  return false;
}

bool Scop::canAlwaysBeHoisted(MemoryAccess *MA, bool StmtInvalidCtxIsEmpty,
                              bool MAInvalidCtxIsEmpty,
                              bool NonHoistableCtxIsEmpty) {
  LoadInst *LInst = cast<LoadInst>(MA->getAccessInstruction());
  const DataLayout &DL = LInst->getParent()->getModule()->getDataLayout();
  if (PollyAllowDereferenceOfAllFunctionParams &&
      isAParameter(LInst->getPointerOperand(), getFunction()))
    return true;

  // TODO: We can provide more information for better but more expensive
  //       results.
  if (!isDereferenceableAndAlignedPointer(LInst->getPointerOperand(),
                                          LInst->getAlignment(), DL))
    return false;

  // If the location might be overwritten we do not hoist it unconditionally.
  //
  // TODO: This is probably too conservative.
  if (!NonHoistableCtxIsEmpty)
    return false;

  // If a dereferenceable load is in a statement that is modeled precisely we
  // can hoist it.
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

  isl::set StmtInvalidCtx = Stmt.getInvalidContext();
  bool StmtInvalidCtxIsEmpty = StmtInvalidCtx.is_empty();

  // Get the context under which the statement is executed but remove the error
  // context under which this statement is reached.
  isl::set DomainCtx = Stmt.getDomain().params();
  DomainCtx = DomainCtx.subtract(StmtInvalidCtx);

  if (isl_set_n_basic_set(DomainCtx.get()) >= MaxDisjunctsInDomain) {
    auto *AccInst = InvMAs.front().MA->getAccessInstruction();
    invalidate(COMPLEXITY, AccInst->getDebugLoc(), AccInst->getParent());
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

        if (isl::id ParamId = getIdForParam(Parameter)) {
          int Dim = DomainCtx.find_dim_by_id(isl::dim::param, ParamId);
          if (Dim >= 0)
            DomainCtx = DomainCtx.eliminate(isl::dim::param, Dim, 1);
        }
      }
    }
  }

  for (auto &InvMA : InvMAs) {
    auto *MA = InvMA.MA;
    isl::set NHCtx = InvMA.NonHoistableCtx;

    // Check for another invariant access that accesses the same location as
    // MA and if found consolidate them. Otherwise create a new equivalence
    // class at the end of InvariantEquivClasses.
    LoadInst *LInst = cast<LoadInst>(MA->getAccessInstruction());
    Type *Ty = LInst->getType();
    const SCEV *PointerSCEV = SE->getSCEV(LInst->getPointerOperand());

    isl::set MAInvalidCtx = MA->getInvalidContext();
    bool NonHoistableCtxIsEmpty = NHCtx.is_empty();
    bool MAInvalidCtxIsEmpty = MAInvalidCtx.is_empty();

    isl::set MACtx;
    // Check if we know that this pointer can be speculatively accessed.
    if (canAlwaysBeHoisted(MA, StmtInvalidCtxIsEmpty, MAInvalidCtxIsEmpty,
                           NonHoistableCtxIsEmpty)) {
      MACtx = isl::set::universe(DomainCtx.get_space());
    } else {
      MACtx = DomainCtx;
      MACtx = MACtx.subtract(MAInvalidCtx.unite(NHCtx));
      MACtx = MACtx.gist_params(getContext());
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

        isl::set AR = MA->getAccessRelation().range();
        isl::set LastAR = LastMA->getAccessRelation().range();
        bool SameAR = AR.is_equal(LastAR);

        if (!SameAR)
          continue;
      }

      // Add MA to the list of accesses that are in this class.
      MAs.push_front(MA);

      Consolidated = true;

      // Unify the execution context of the class and this statement.
      isl::set IAClassDomainCtx = isl::manage(IAClass.ExecutionContext);
      if (IAClassDomainCtx)
        IAClassDomainCtx = IAClassDomainCtx.unite(MACtx).coalesce();
      else
        IAClassDomainCtx = MACtx;
      IAClass.ExecutionContext = IAClassDomainCtx.release();
      break;
    }

    if (Consolidated)
      continue;

    // If we did not consolidate MA, thus did not find an equivalence class
    // for it, we create a new one.
    InvariantEquivClasses.emplace_back(InvariantEquivClassTy{
        PointerSCEV, MemoryAccessList{MA}, MACtx.release(), Ty});
  }
}

/// Check if an access range is too complex.
///
/// An access range is too complex, if it contains either many disjuncts or
/// very complex expressions. As a simple heuristic, we assume if a set to
/// be too complex if the sum of existentially quantified dimensions and
/// set dimensions is larger than a threshold. This reliably detects both
/// sets with many disjuncts as well as sets with many divisions as they
/// arise in h264.
///
/// @param AccessRange The range to check for complexity.
///
/// @returns True if the access range is too complex.
static bool isAccessRangeTooComplex(isl::set AccessRange) {
  unsigned NumTotalDims = 0;

  auto CountDimensions = [&NumTotalDims](isl::basic_set BSet) -> isl::stat {
    NumTotalDims += BSet.dim(isl::dim::div);
    NumTotalDims += BSet.dim(isl::dim::set);
    return isl::stat::ok;
  };

  AccessRange.foreach_basic_set(CountDimensions);

  if (NumTotalDims > MaxDimensionsInAccessRange)
    return true;

  return false;
}

isl::set Scop::getNonHoistableCtx(MemoryAccess *Access, isl::union_map Writes) {
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

  isl::map AccessRelation = give(Access->getAccessRelation().release());
  assert(!AccessRelation.is_empty());

  if (AccessRelation.involves_dims(isl::dim::in, 0, Stmt.getNumIterators()))
    return nullptr;

  AccessRelation = AccessRelation.intersect_domain(Stmt.getDomain());
  isl::set SafeToLoad;

  auto &DL = getFunction().getParent()->getDataLayout();
  if (isSafeToLoadUnconditionally(LI->getPointerOperand(), LI->getAlignment(),
                                  DL)) {
    SafeToLoad = isl::set::universe(AccessRelation.get_space().range());
  } else if (BB != LI->getParent()) {
    // Skip accesses in non-affine subregions as they might not be executed
    // under the same condition as the entry of the non-affine subregion.
    return nullptr;
  } else {
    SafeToLoad = AccessRelation.range();
  }

  if (isAccessRangeTooComplex(AccessRelation.range()))
    return nullptr;

  isl::union_map Written = Writes.intersect_range(SafeToLoad);
  isl::set WrittenCtx = Written.params();
  bool IsWritten = !WrittenCtx.is_empty();

  if (!IsWritten)
    return WrittenCtx;

  WrittenCtx = WrittenCtx.remove_divs();
  bool TooComplex =
      isl_set_n_basic_set(WrittenCtx.get()) >= MaxDisjunctsInDomain;
  if (TooComplex || !isRequiredInvariantLoad(LI))
    return nullptr;

  addAssumption(INVARIANTLOAD, WrittenCtx.copy(), LI->getDebugLoc(),
                AS_RESTRICTION, LI->getParent());
  return WrittenCtx;
}

void Scop::verifyInvariantLoads() {
  auto &RIL = getRequiredInvariantLoads();
  for (LoadInst *LI : RIL) {
    assert(LI && contains(LI));
    // If there exists a statement in the scop which has a memory access for
    // @p LI, then mark this scop as infeasible for optimization.
    for (ScopStmt &Stmt : Stmts)
      if (Stmt.getArrayAccessOrNULLFor(LI)) {
        invalidate(INVARIANTLOAD, LI->getDebugLoc(), LI->getParent());
        return;
      }
  }
}

void Scop::hoistInvariantLoads() {
  if (!PollyInvariantLoadHoisting)
    return;

  isl::union_map Writes = getWrites();
  for (ScopStmt &Stmt : *this) {
    InvariantAccessesTy InvariantAccesses;

    for (MemoryAccess *Access : Stmt)
      if (isl::set NHCtx = getNonHoistableCtx(Access, Writes))
        InvariantAccesses.push_back({Access, NHCtx});

    // Transfer the memory access from the statement to the SCoP.
    for (auto InvMA : InvariantAccesses)
      Stmt.removeMemoryAccess(InvMA.MA);
    addInvariantLoads(Stmt, InvariantAccesses);
  }
}

/// Find the canonical scop array info object for a set of invariant load
/// hoisted loads. The canonical array is the one that corresponds to the
/// first load in the list of accesses which is used as base pointer of a
/// scop array.
static const ScopArrayInfo *findCanonicalArray(Scop *S,
                                               MemoryAccessList &Accesses) {
  for (MemoryAccess *Access : Accesses) {
    const ScopArrayInfo *CanonicalArray = S->getScopArrayInfoOrNull(
        Access->getAccessInstruction(), MemoryKind::Array);
    if (CanonicalArray)
      return CanonicalArray;
  }
  return nullptr;
}

/// Check if @p Array severs as base array in an invariant load.
static bool isUsedForIndirectHoistedLoad(Scop *S, const ScopArrayInfo *Array) {
  for (InvariantEquivClassTy &EqClass2 : S->getInvariantAccesses())
    for (MemoryAccess *Access2 : EqClass2.InvariantAccesses)
      if (Access2->getScopArrayInfo() == Array)
        return true;
  return false;
}

/// Replace the base pointer arrays in all memory accesses referencing @p Old,
/// with a reference to @p New.
static void replaceBasePtrArrays(Scop *S, const ScopArrayInfo *Old,
                                 const ScopArrayInfo *New) {
  for (ScopStmt &Stmt : *S)
    for (MemoryAccess *Access : Stmt) {
      if (Access->getLatestScopArrayInfo() != Old)
        continue;

      isl::id Id = New->getBasePtrId();
      isl::map Map = Access->getAccessRelation();
      Map = Map.set_tuple_id(isl::dim::out, Id);
      Access->setAccessRelation(Map);
    }
}

void Scop::canonicalizeDynamicBasePtrs() {
  for (InvariantEquivClassTy &EqClass : InvariantEquivClasses) {
    MemoryAccessList &BasePtrAccesses = EqClass.InvariantAccesses;

    const ScopArrayInfo *CanonicalBasePtrSAI =
        findCanonicalArray(this, BasePtrAccesses);

    if (!CanonicalBasePtrSAI)
      continue;

    for (MemoryAccess *BasePtrAccess : BasePtrAccesses) {
      const ScopArrayInfo *BasePtrSAI = getScopArrayInfoOrNull(
          BasePtrAccess->getAccessInstruction(), MemoryKind::Array);
      if (!BasePtrSAI || BasePtrSAI == CanonicalBasePtrSAI ||
          !BasePtrSAI->isCompatibleWith(CanonicalBasePtrSAI))
        continue;

      // we currently do not canonicalize arrays where some accesses are
      // hoisted as invariant loads. If we would, we need to update the access
      // function of the invariant loads as well. However, as this is not a
      // very common situation, we leave this for now to avoid further
      // complexity increases.
      if (isUsedForIndirectHoistedLoad(this, BasePtrSAI))
        continue;

      replaceBasePtrArrays(this, BasePtrSAI, CanonicalBasePtrSAI);
    }
  }
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

const ScopArrayInfo *Scop::getScopArrayInfoOrNull(Value *BasePtr,
                                                  MemoryKind Kind) {
  auto *SAI = ScopArrayInfoMap[std::make_pair(BasePtr, Kind)].get();
  return SAI;
}

const ScopArrayInfo *Scop::getScopArrayInfo(Value *BasePtr, MemoryKind Kind) {
  auto *SAI = getScopArrayInfoOrNull(BasePtr, Kind);
  assert(SAI && "No ScopArrayInfo available for this base pointer");
  return SAI;
}

std::string Scop::getContextStr() const { return getContext().to_str(); }

std::string Scop::getAssumedContextStr() const {
  assert(AssumedContext && "Assumed context not yet built");
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

isl::set Scop::getContext() const { return isl::manage(isl_set_copy(Context)); }
isl::space Scop::getParamSpace() const { return getContext().get_space(); }

isl::space Scop::getFullParamSpace() const {
  std::vector<isl::id> FortranIDs;
  FortranIDs = getFortranArrayIds(arrays());

  isl::space Space = isl::space::params_alloc(
      getIslCtx(), ParameterIds.size() + FortranIDs.size());

  unsigned PDim = 0;
  for (const SCEV *Parameter : Parameters) {
    isl::id Id = getIdForParam(Parameter);
    Space = Space.set_dim_id(isl::dim::param, PDim++, Id);
  }

  for (isl::id Id : FortranIDs)
    Space = Space.set_dim_id(isl::dim::param, PDim++, Id);

  return Space;
}

isl::set Scop::getAssumedContext() const {
  assert(AssumedContext && "Assumed context not yet built");
  return isl::manage(isl_set_copy(AssumedContext));
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
  auto *PositiveContext = getAssumedContext().release();
  auto *NegativeContext = getInvalidContext().release();
  PositiveContext =
      addNonEmptyDomainConstraints(isl::manage(PositiveContext)).release();
  bool IsFeasible = !(isl_set_is_empty(PositiveContext) ||
                      isl_set_is_subset(PositiveContext, NegativeContext));
  isl_set_free(PositiveContext);
  if (!IsFeasible) {
    isl_set_free(NegativeContext);
    return false;
  }

  auto *DomainContext = isl_union_set_params(getDomains().release());
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
                           DebugLoc Loc, AssumptionSign Sign, BasicBlock *BB) {
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

void Scop::addAssumption(AssumptionKind Kind, __isl_take isl_set *Set,
                         DebugLoc Loc, AssumptionSign Sign, BasicBlock *BB) {
  // Simplify the assumptions/restrictions first.
  Set = isl_set_gist_params(Set, getContext().release());

  if (!trackAssumption(Kind, Set, Loc, Sign, BB)) {
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
      addAssumption(AS.Kind, AS.Set, AS.Loc, AS.Sign, nullptr /* BasicBlock */);
      continue;
    }

    // If the domain was deleted the assumptions are void.
    isl_set *Dom = getDomainConditions(AS.BB).release();
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

    addAssumption(AS.Kind, S, AS.Loc, AS_RESTRICTION, AS.BB);
  }
}

void Scop::invalidate(AssumptionKind Kind, DebugLoc Loc, BasicBlock *BB) {
  addAssumption(Kind, isl_set_empty(getParamSpace().release()), Loc,
                AS_ASSUMPTION, BB);
}

isl::set Scop::getInvalidContext() const {
  return isl::manage(isl_set_copy(InvalidContext));
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
  invalidate(COMPLEXITY, DL, BB);
  return Affinator.getPwAff(SE->getZero(E->getType()), BB);
}

isl::union_set Scop::getDomains() const {
  isl_space *EmptySpace = isl_space_params_alloc(getIslCtx(), 0);
  isl_union_set *Domain = isl_union_set_empty(EmptySpace);

  for (const ScopStmt &Stmt : *this)
    Domain = isl_union_set_add_set(Domain, Stmt.getDomain().release());

  return isl::manage(Domain);
}

isl::pw_aff Scop::getPwAffOnly(const SCEV *E, BasicBlock *BB) {
  PWACtx PWAC = getPwAff(E, BB);
  isl_set_free(PWAC.second);
  return isl::manage(PWAC.first);
}

isl::union_map
Scop::getAccessesOfType(std::function<bool(MemoryAccess &)> Predicate) {
  isl::union_map Accesses = isl::union_map::empty(getParamSpace());

  for (ScopStmt &Stmt : *this) {
    for (MemoryAccess *MA : Stmt) {
      if (!Predicate(*MA))
        continue;

      isl::set Domain = Stmt.getDomain();
      isl::map AccessDomain = MA->getAccessRelation();
      AccessDomain = AccessDomain.intersect_domain(Domain);
      Accesses = Accesses.add_map(AccessDomain);
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

isl::union_map Scop::getSchedule() const {
  auto *Tree = getScheduleTree().release();
  if (containsExtensionNode(Tree)) {
    isl_schedule_free(Tree);
    return nullptr;
  }
  auto *S = isl_schedule_get_map(Tree);
  isl_schedule_free(Tree);
  return isl::manage(S);
}

isl::schedule Scop::getScheduleTree() const {
  return isl::manage(isl_schedule_intersect_domain(isl_schedule_copy(Schedule),
                                                   getDomains().release()));
}

void Scop::setSchedule(__isl_take isl_union_map *NewSchedule) {
  auto *S = isl_schedule_from_domain(getDomains().release());
  S = isl_schedule_insert_partial_schedule(
      S, isl_multi_union_pw_aff_from_union_map(NewSchedule));
  isl_schedule_free(Schedule);
  Schedule = S;
}

void Scop::setScheduleTree(__isl_take isl_schedule *NewSchedule) {
  isl_schedule_free(Schedule);
  Schedule = NewSchedule;
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
static isl::multi_union_pw_aff mapToDimension(isl::union_set USet, int N) {
  assert(N >= 0);
  assert(USet);
  assert(!USet.is_empty());

  auto Result = isl::union_pw_multi_aff::empty(USet.get_space());

  auto Lambda = [&Result, N](isl::set S) -> isl::stat {
    int Dim = S.dim(isl::dim::set);
    auto PMA = isl::pw_multi_aff::project_out_map(S.get_space(), isl::dim::set,
                                                  N, Dim - N);
    if (N > 1)
      PMA = PMA.drop_dims(isl::dim::out, 0, N - 1);

    Result = Result.add_pw_multi_aff(PMA);
    return isl::stat::ok;
  };

  isl::stat Res = USet.foreach_set(Lambda);
  (void)Res;

  assert(Res == isl::stat::ok);

  return isl::multi_union_pw_aff(isl::union_pw_multi_aff(Result));
}

void Scop::addScopStmt(BasicBlock *BB, Loop *SurroundingLoop,
                       std::vector<Instruction *> Instructions) {
  assert(BB && "Unexpected nullptr!");
  Stmts.emplace_back(*this, *BB, SurroundingLoop, Instructions);
  auto *Stmt = &Stmts.back();
  StmtMap[BB].push_back(Stmt);
  for (Instruction *Inst : Instructions) {
    assert(!InstStmtMap.count(Inst) &&
           "Unexpected statement corresponding to the instruction.");
    InstStmtMap[Inst] = Stmt;
  }
}

void Scop::addScopStmt(Region *R, Loop *SurroundingLoop) {
  assert(R && "Unexpected nullptr!");
  Stmts.emplace_back(*this, *R, SurroundingLoop);
  auto *Stmt = &Stmts.back();
  for (BasicBlock *BB : R->blocks()) {
    StmtMap[BB].push_back(Stmt);
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

    if ((LastRNWaiting && !WorkList.empty()) || DelayList.empty()) {
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

  for (auto *Stmt : getStmtListFor(RN)) {
    auto *UDomain = isl_union_set_from_set(Stmt->getDomain().release());
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
      isl::union_set Domain = give(isl_schedule_get_domain(Schedule));
      isl::multi_union_pw_aff MUPA = mapToDimension(Domain, LoopStack.size());
      Schedule = isl_schedule_insert_partial_schedule(Schedule, MUPA.release());
      NextLoopData.Schedule =
          combineInSequence(NextLoopData.Schedule, Schedule);
    }

    NextLoopData.NumBlocksProcessed += NumBlocksProcessed;
    LoopData = NextLoopData;
  }
}

ArrayRef<ScopStmt *> Scop::getStmtListFor(BasicBlock *BB) const {
  auto StmtMapIt = StmtMap.find(BB);
  if (StmtMapIt == StmtMap.end())
    return {};
  assert(StmtMapIt->second.size() == 1 &&
         "Each statement corresponds to exactly one BB.");
  return StmtMapIt->second;
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
  if (Access->isOriginalValueKind() && Access->isRead()) {
    auto &Uses = ValueUseAccs[Access->getScopArrayInfo()];
    std::remove(Uses.begin(), Uses.end(), Access);
  } else if (Access->isOriginalAnyPHIKind() && Access->isWrite()) {
    auto &Incomings = PHIIncomingAccs[Access->getScopArrayInfo()];
    std::remove(Incomings.begin(), Incomings.end(), Access);
  }
}

MemoryAccess *Scop::getValueDef(const ScopArrayInfo *SAI) const {
  assert(SAI->isValueKind());

  Instruction *Val = dyn_cast<Instruction>(SAI->getBasePtr());
  if (!Val)
    return nullptr;

  ScopStmt *Stmt = getStmtFor(Val);
  if (!Stmt)
    return nullptr;

  return Stmt->lookupValueWriteOf(Val);
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
  ScopStmt *Stmt = getStmtFor(PHI);
  assert(Stmt && "PHINode must be within the SCoP");

  return Stmt->lookupPHIReadOf(PHI);
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
  AU.setPreservesAll();
}

void updateLoopCountStatistic(ScopDetection::LoopStats Stats,
                              Scop::ScopStatistics ScopStats) {
  assert(Stats.NumLoops == ScopStats.NumAffineLoops + ScopStats.NumBoxedLoops);

  NumScops++;
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

  ScopBuilder SB(R, AC, AA, DL, DT, LI, SD, SE);
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
ScopInfo::ScopInfo(const DataLayout &DL, ScopDetection &SD, ScalarEvolution &SE,
                   LoopInfo &LI, AliasAnalysis &AA, DominatorTree &DT,
                   AssumptionCache &AC)
    : DL(DL), SD(SD), SE(SE), LI(LI), AA(AA), DT(DT), AC(AC) {
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

    ScopBuilder SB(R, AC, AA, DL, DT, LI, SD, SE);
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
  return {DL, SD, SE, LI, AA, DT, AC};
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

  Result.reset(new ScopInfo{DL, SD, SE, LI, AA, DT, AC});
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
