//===- MatmulOptimizer.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "polly/MatmulOptimizer.h"
#include "polly/DependenceInfo.h"
#include "polly/Options.h"
#include "polly/ScheduleTreeTransform.h"
#include "polly/ScopInfo.h"
#include "polly/ScopPass.h"
#include "polly/Simplify.h"
#include "polly/Support/ISLTools.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"
#include "isl/ctx.h"
#include "isl/schedule_node.h"
#include "isl/schedule_type.h"
#include "isl/union_map.h"
#include "isl/union_set.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#define DEBUG_TYPE "polly-opt-isl"

using namespace llvm;
using namespace polly;

namespace llvm {
class Value;
}

static cl::opt<int> LatencyVectorFma(
    "polly-target-latency-vector-fma",
    cl::desc("The minimal number of cycles between issuing two "
             "dependent consecutive vector fused multiply-add "
             "instructions."),
    cl::Hidden, cl::init(8), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> ThroughputVectorFma(
    "polly-target-throughput-vector-fma",
    cl::desc("A throughput of the processor floating-point arithmetic units "
             "expressed in the number of vector fused multiply-add "
             "instructions per clock cycle."),
    cl::Hidden, cl::init(1), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> FirstCacheLevelSize(
    "polly-target-1st-cache-level-size",
    cl::desc("The size of the first cache level specified in bytes."),
    cl::Hidden, cl::init(-1), cl::cat(PollyCategory));

static cl::opt<int> FirstCacheLevelDefaultSize(
    "polly-target-1st-cache-level-default-size",
    cl::desc("The default size of the first cache level specified in bytes"
             " (if not enough were provided by the TargetTransformInfo)."),
    cl::Hidden, cl::init(32768), cl::cat(PollyCategory));

static cl::opt<int> SecondCacheLevelSize(
    "polly-target-2nd-cache-level-size",
    cl::desc("The size of the second level specified in bytes."), cl::Hidden,
    cl::init(-1), cl::cat(PollyCategory));

static cl::opt<int> SecondCacheLevelDefaultSize(
    "polly-target-2nd-cache-level-default-size",
    cl::desc("The default size of the second cache level specified in bytes"
             " (if not enough were provided by the TargetTransformInfo)."),
    cl::Hidden, cl::init(262144), cl::cat(PollyCategory));

// This option, along with --polly-target-2nd-cache-level-associativity,
// --polly-target-1st-cache-level-size, and --polly-target-2st-cache-level-size
// represent the parameters of the target cache, which do not have typical
// values that can be used by default. However, to apply the pattern matching
// optimizations, we use the values of the parameters of Intel Core i7-3820
// SandyBridge in case the parameters are not specified or not provided by the
// TargetTransformInfo.
static cl::opt<int> FirstCacheLevelAssociativity(
    "polly-target-1st-cache-level-associativity",
    cl::desc("The associativity of the first cache level."), cl::Hidden,
    cl::init(-1), cl::cat(PollyCategory));

static cl::opt<int> FirstCacheLevelDefaultAssociativity(
    "polly-target-1st-cache-level-default-associativity",
    cl::desc("The default associativity of the first cache level"
             " (if not enough were provided by the TargetTransformInfo)."),
    cl::Hidden, cl::init(8), cl::cat(PollyCategory));

static cl::opt<int> SecondCacheLevelAssociativity(
    "polly-target-2nd-cache-level-associativity",
    cl::desc("The associativity of the second cache level."), cl::Hidden,
    cl::init(-1), cl::cat(PollyCategory));

static cl::opt<int> SecondCacheLevelDefaultAssociativity(
    "polly-target-2nd-cache-level-default-associativity",
    cl::desc("The default associativity of the second cache level"
             " (if not enough were provided by the TargetTransformInfo)."),
    cl::Hidden, cl::init(8), cl::cat(PollyCategory));

static cl::opt<int> VectorRegisterBitwidth(
    "polly-target-vector-register-bitwidth",
    cl::desc("The size in bits of a vector register (if not set, this "
             "information is taken from LLVM's target information."),
    cl::Hidden, cl::init(-1), cl::cat(PollyCategory));

static cl::opt<int> PollyPatternMatchingNcQuotient(
    "polly-pattern-matching-nc-quotient",
    cl::desc("Quotient that is obtained by dividing Nc, the parameter of the"
             "macro-kernel, by Nr, the parameter of the micro-kernel"),
    cl::Hidden, cl::init(256), cl::cat(PollyCategory));

namespace {
/// Parameters of the micro kernel.
///
/// Parameters, which determine sizes of rank-1 (i.e., outer product) update
/// used in the optimized matrix multiplication.
struct MicroKernelParamsTy {
  int Mr;
  int Nr;
};

/// Parameters of the macro kernel.
///
/// Parameters, which determine sizes of blocks of partitioned matrices
/// used in the optimized matrix multiplication.
struct MacroKernelParamsTy {
  int Mc;
  int Nc;
  int Kc;
};

/// Parameters of the matrix multiplication operands.
///
/// Parameters, which describe access relations that represent operands of the
/// matrix multiplication.
struct MatMulInfoTy {
  MemoryAccess *A = nullptr;
  MemoryAccess *B = nullptr;
  MemoryAccess *ReadFromC = nullptr;
  MemoryAccess *WriteToC = nullptr;
  int i = -1;
  int j = -1;
  int k = -1;
};

/// Create an isl::union_set, which describes the option of the form
/// [isolate[] -> unroll[x]].
///
/// @param Ctx An isl::ctx, which is used to create the isl::union_set.
static isl::union_set getUnrollIsolatedSetOptions(isl::ctx Ctx) {
  isl::space Space = isl::space(Ctx, 0, 0, 1);
  isl::map UnrollIsolatedSetOption = isl::map::universe(Space);
  isl::id DimInId = isl::id::alloc(Ctx, "isolate", nullptr);
  isl::id DimOutId = isl::id::alloc(Ctx, "unroll", nullptr);
  UnrollIsolatedSetOption =
      UnrollIsolatedSetOption.set_tuple_id(isl::dim::in, DimInId);
  UnrollIsolatedSetOption =
      UnrollIsolatedSetOption.set_tuple_id(isl::dim::out, DimOutId);
  return UnrollIsolatedSetOption.wrap();
}

/// Permute the two dimensions of the isl map.
///
/// Permute @p DstPos and @p SrcPos dimensions of the isl map @p Map that
/// have type @p DimType.
///
/// @param Map     The isl map to be modified.
/// @param DimType The type of the dimensions.
/// @param DstPos  The first dimension.
/// @param SrcPos  The second dimension.
/// @return        The modified map.
static isl::map permuteDimensions(isl::map Map, isl::dim DimType,
                                  unsigned DstPos, unsigned SrcPos) {
  assert(DstPos < unsignedFromIslSize(Map.dim(DimType)) &&
         SrcPos < unsignedFromIslSize(Map.dim(DimType)));
  if (DstPos == SrcPos)
    return Map;
  isl::id DimId;
  if (Map.has_tuple_id(DimType))
    DimId = Map.get_tuple_id(DimType);
  auto FreeDim = DimType == isl::dim::in ? isl::dim::out : isl::dim::in;
  isl::id FreeDimId;
  if (Map.has_tuple_id(FreeDim))
    FreeDimId = Map.get_tuple_id(FreeDim);
  auto MaxDim = std::max(DstPos, SrcPos);
  auto MinDim = std::min(DstPos, SrcPos);
  Map = Map.move_dims(FreeDim, 0, DimType, MaxDim, 1);
  Map = Map.move_dims(FreeDim, 0, DimType, MinDim, 1);
  Map = Map.move_dims(DimType, MinDim, FreeDim, 1, 1);
  Map = Map.move_dims(DimType, MaxDim, FreeDim, 0, 1);
  if (!DimId.is_null())
    Map = Map.set_tuple_id(DimType, DimId);
  if (!FreeDimId.is_null())
    Map = Map.set_tuple_id(FreeDim, FreeDimId);
  return Map;
}

/// Check the form of the access relation.
///
/// Check that the access relation @p AccMap has the form M[i][j], where i
/// is a @p FirstPos and j is a @p SecondPos.
///
/// @param AccMap    The access relation to be checked.
/// @param FirstPos  The index of the input dimension that is mapped to
///                  the first output dimension.
/// @param SecondPos The index of the input dimension that is mapped to the
///                  second output dimension.
/// @return          True in case @p AccMap has the expected form and false,
///                  otherwise.
static bool isMatMulOperandAcc(isl::set Domain, isl::map AccMap, int &FirstPos,
                               int &SecondPos) {
  isl::space Space = AccMap.get_space();
  isl::map Universe = isl::map::universe(Space);

  if (unsignedFromIslSize(Space.dim(isl::dim::out)) != 2)
    return false;

  // MatMul has the form:
  // for (i = 0; i < N; i++)
  //   for (j = 0; j < M; j++)
  //     for (k = 0; k < P; k++)
  //       C[i, j] += A[i, k] * B[k, j]
  //
  // Permutation of three outer loops: 3! = 6 possibilities.
  int FirstDims[] = {0, 0, 1, 1, 2, 2};
  int SecondDims[] = {1, 2, 2, 0, 0, 1};
  for (int i = 0; i < 6; i += 1) {
    auto PossibleMatMul =
        Universe.equate(isl::dim::in, FirstDims[i], isl::dim::out, 0)
            .equate(isl::dim::in, SecondDims[i], isl::dim::out, 1);

    AccMap = AccMap.intersect_domain(Domain);
    PossibleMatMul = PossibleMatMul.intersect_domain(Domain);

    // If AccMap spans entire domain (Non-partial write),
    // compute FirstPos and SecondPos.
    // If AccMap != PossibleMatMul here (the two maps have been gisted at
    // this point), it means that the writes are not complete, or in other
    // words, it is a Partial write and Partial writes must be rejected.
    if (AccMap.is_equal(PossibleMatMul)) {
      if (FirstPos != -1 && FirstPos != FirstDims[i])
        continue;
      FirstPos = FirstDims[i];
      if (SecondPos != -1 && SecondPos != SecondDims[i])
        continue;
      SecondPos = SecondDims[i];
      return true;
    }
  }

  return false;
}

/// Does the memory access represent a non-scalar operand of the matrix
/// multiplication.
///
/// Check that the memory access @p MemAccess is the read access to a non-scalar
/// operand of the matrix multiplication or its result.
///
/// @param MemAccess The memory access to be checked.
/// @param MMI       Parameters of the matrix multiplication operands.
/// @return          True in case the memory access represents the read access
///                  to a non-scalar operand of the matrix multiplication and
///                  false, otherwise.
static bool isMatMulNonScalarReadAccess(MemoryAccess *MemAccess,
                                        MatMulInfoTy &MMI) {
  if (!MemAccess->isLatestArrayKind() || !MemAccess->isRead())
    return false;
  auto AccMap = MemAccess->getLatestAccessRelation();
  isl::set StmtDomain = MemAccess->getStatement()->getDomain();
  if (isMatMulOperandAcc(StmtDomain, AccMap, MMI.i, MMI.j) && !MMI.ReadFromC) {
    MMI.ReadFromC = MemAccess;
    return true;
  }
  if (isMatMulOperandAcc(StmtDomain, AccMap, MMI.i, MMI.k) && !MMI.A) {
    MMI.A = MemAccess;
    return true;
  }
  if (isMatMulOperandAcc(StmtDomain, AccMap, MMI.k, MMI.j) && !MMI.B) {
    MMI.B = MemAccess;
    return true;
  }
  return false;
}

/// Check accesses to operands of the matrix multiplication.
///
/// Check that accesses of the SCoP statement, which corresponds to
/// the partial schedule @p PartialSchedule, are scalar in terms of loops
/// containing the matrix multiplication, in case they do not represent
/// accesses to the non-scalar operands of the matrix multiplication or
/// its result.
///
/// @param  PartialSchedule The partial schedule of the SCoP statement.
/// @param  MMI             Parameters of the matrix multiplication operands.
/// @return                 True in case the corresponding SCoP statement
///                         represents matrix multiplication and false,
///                         otherwise.
static bool containsOnlyMatrMultAcc(isl::map PartialSchedule,
                                    MatMulInfoTy &MMI) {
  auto InputDimId = PartialSchedule.get_tuple_id(isl::dim::in);
  auto *Stmt = static_cast<ScopStmt *>(InputDimId.get_user());
  unsigned OutDimNum = unsignedFromIslSize(PartialSchedule.range_tuple_dim());
  assert(OutDimNum > 2 && "In case of the matrix multiplication the loop nest "
                          "and, consequently, the corresponding scheduling "
                          "functions have at least three dimensions.");
  auto MapI =
      permuteDimensions(PartialSchedule, isl::dim::out, MMI.i, OutDimNum - 1);
  auto MapJ =
      permuteDimensions(PartialSchedule, isl::dim::out, MMI.j, OutDimNum - 1);
  auto MapK =
      permuteDimensions(PartialSchedule, isl::dim::out, MMI.k, OutDimNum - 1);

  auto Accesses = getAccessesInOrder(*Stmt);
  for (auto *MemA = Accesses.begin(); MemA != Accesses.end() - 1; MemA++) {
    auto *MemAccessPtr = *MemA;
    if (MemAccessPtr->isLatestArrayKind() && MemAccessPtr != MMI.WriteToC &&
        !isMatMulNonScalarReadAccess(MemAccessPtr, MMI) &&
        !(MemAccessPtr->isStrideZero(MapI) &&
          MemAccessPtr->isStrideZero(MapJ) && MemAccessPtr->isStrideZero(MapK)))
      return false;
  }
  return true;
}

/// Check for dependencies corresponding to the matrix multiplication.
///
/// Check that there is only true dependence of the form
/// S(..., k, ...) -> S(..., k + 1, …), where S is the SCoP statement
/// represented by @p Schedule and k is @p Pos. Such a dependence corresponds
/// to the dependency produced by the matrix multiplication.
///
/// @param  Schedule The schedule of the SCoP statement.
/// @param  D The SCoP dependencies.
/// @param  Pos The parameter to describe an acceptable true dependence.
///             In case it has a negative value, try to determine its
///             acceptable value.
/// @return True in case dependencies correspond to the matrix multiplication
///         and false, otherwise.
static bool containsOnlyMatMulDep(isl::map Schedule, const Dependences *D,
                                  int &Pos) {
  isl::union_map Dep = D->getDependences(Dependences::TYPE_RAW);
  isl::union_map Red = D->getDependences(Dependences::TYPE_RED);
  if (!Red.is_null())
    Dep = Dep.unite(Red);
  auto DomainSpace = Schedule.get_space().domain();
  auto Space = DomainSpace.map_from_domain_and_range(DomainSpace);
  auto Deltas = Dep.extract_map(Space).deltas();
  int DeltasDimNum = unsignedFromIslSize(Deltas.dim(isl::dim::set));
  for (int i = 0; i < DeltasDimNum; i++) {
    auto Val = Deltas.plain_get_val_if_fixed(isl::dim::set, i);
    Pos = Pos < 0 && Val.is_one() ? i : Pos;
    if (Val.is_nan() || !(Val.is_zero() || (i == Pos && Val.is_one())))
      return false;
  }
  if (DeltasDimNum == 0 || Pos < 0)
    return false;
  return true;
}

/// Check if the SCoP statement could probably be optimized with analytical
/// modeling.
///
/// containsMatrMult tries to determine whether the following conditions
/// are true:
/// 1. The last memory access modeling an array, MA1, represents writing to
///    memory and has the form S(..., i1, ..., i2, ...) -> M(i1, i2) or
///    S(..., i2, ..., i1, ...) -> M(i1, i2), where S is the SCoP statement
///    under consideration.
/// 2. There is only one loop-carried true dependency, and it has the
///    form S(..., i3, ...) -> S(..., i3 + 1, ...), and there are no
///    loop-carried or anti dependencies.
/// 3. SCoP contains three access relations, MA2, MA3, and MA4 that represent
///    reading from memory and have the form S(..., i3, ...) -> M(i1, i3),
///    S(..., i3, ...) -> M(i3, i2), S(...) -> M(i1, i2), respectively,
///    and all memory accesses of the SCoP that are different from MA1, MA2,
///    MA3, and MA4 have stride 0, if the innermost loop is exchanged with any
///    of loops i1, i2 and i3.
///
/// @param PartialSchedule The PartialSchedule that contains a SCoP statement
///        to check.
/// @D     The SCoP dependencies.
/// @MMI   Parameters of the matrix multiplication operands.
static bool containsMatrMult(isl::map PartialSchedule, const Dependences *D,
                             MatMulInfoTy &MMI) {
  auto InputDimsId = PartialSchedule.get_tuple_id(isl::dim::in);
  auto *Stmt = static_cast<ScopStmt *>(InputDimsId.get_user());
  if (Stmt->size() <= 1)
    return false;

  auto Accesses = getAccessesInOrder(*Stmt);
  for (auto *MemA = Accesses.end() - 1; MemA != Accesses.begin(); MemA--) {
    auto *MemAccessPtr = *MemA;
    if (!MemAccessPtr->isLatestArrayKind())
      continue;
    if (!MemAccessPtr->isWrite())
      return false;
    auto AccMap = MemAccessPtr->getLatestAccessRelation();
    if (!isMatMulOperandAcc(Stmt->getDomain(), AccMap, MMI.i, MMI.j))
      return false;
    MMI.WriteToC = MemAccessPtr;
    break;
  }

  if (!containsOnlyMatMulDep(PartialSchedule, D, MMI.k))
    return false;

  if (!MMI.WriteToC || !containsOnlyMatrMultAcc(PartialSchedule, MMI))
    return false;

  if (!MMI.A || !MMI.B || !MMI.ReadFromC)
    return false;
  return true;
}

/// Permute two dimensions of the band node.
///
/// Permute FirstDim and SecondDim dimensions of the Node.
///
/// @param Node The band node to be modified.
/// @param FirstDim The first dimension to be permuted.
/// @param SecondDim The second dimension to be permuted.
static isl::schedule_node permuteBandNodeDimensions(isl::schedule_node Node,
                                                    unsigned FirstDim,
                                                    unsigned SecondDim) {
  assert(isl_schedule_node_get_type(Node.get()) == isl_schedule_node_band &&
         (unsigned)isl_schedule_node_band_n_member(Node.get()) >
             std::max(FirstDim, SecondDim));
  auto PartialSchedule =
      isl::manage(isl_schedule_node_band_get_partial_schedule(Node.get()));
  auto PartialScheduleFirstDim = PartialSchedule.at(FirstDim);
  auto PartialScheduleSecondDim = PartialSchedule.at(SecondDim);
  PartialSchedule =
      PartialSchedule.set_union_pw_aff(SecondDim, PartialScheduleFirstDim);
  PartialSchedule =
      PartialSchedule.set_union_pw_aff(FirstDim, PartialScheduleSecondDim);
  Node = isl::manage(isl_schedule_node_delete(Node.release()));
  return Node.insert_partial_schedule(PartialSchedule);
}

static isl::schedule_node
createMicroKernel(isl::schedule_node Node,
                  MicroKernelParamsTy MicroKernelParams) {
  Node = applyRegisterTiling(Node, {MicroKernelParams.Mr, MicroKernelParams.Nr},
                             1);
  Node = Node.parent().parent();
  return permuteBandNodeDimensions(Node, 0, 1).child(0).child(0);
}

/// Create the BLIS macro-kernel.
///
/// We create the BLIS macro-kernel by applying a combination of tiling
/// of dimensions of the band node and interchanging of two innermost
/// modified dimensions. The values of of MacroKernelParams's fields are used
/// as tile sizes.
///
/// @param Node The schedule node to be modified.
/// @param MacroKernelParams Parameters of the macro kernel
///                          to be used as tile sizes.
static isl::schedule_node
createMacroKernel(isl::schedule_node Node,
                  MacroKernelParamsTy MacroKernelParams) {
  assert(isl_schedule_node_get_type(Node.get()) == isl_schedule_node_band);
  if (MacroKernelParams.Mc == 1 && MacroKernelParams.Nc == 1 &&
      MacroKernelParams.Kc == 1)
    return Node;
  int DimOutNum = isl_schedule_node_band_n_member(Node.get());
  std::vector<int> TileSizes(DimOutNum, 1);
  TileSizes[DimOutNum - 3] = MacroKernelParams.Mc;
  TileSizes[DimOutNum - 2] = MacroKernelParams.Nc;
  TileSizes[DimOutNum - 1] = MacroKernelParams.Kc;
  Node = tileNode(Node, "1st level tiling", TileSizes, 1);
  Node = Node.parent().parent();
  Node = permuteBandNodeDimensions(Node, DimOutNum - 2, DimOutNum - 1);
  Node = permuteBandNodeDimensions(Node, DimOutNum - 3, DimOutNum - 1);

  // Mark the outermost loop as parallelizable.
  Node = Node.as<isl::schedule_node_band>().member_set_coincident(0, true);

  return Node.child(0).child(0);
}

/// Get the size of the widest type of the matrix multiplication operands
/// in bytes, including alignment padding.
///
/// @param MMI Parameters of the matrix multiplication operands.
/// @return The size of the widest type of the matrix multiplication operands
///         in bytes, including alignment padding.
static uint64_t getMatMulAlignTypeSize(MatMulInfoTy MMI) {
  auto *S = MMI.A->getStatement()->getParent();
  auto &DL = S->getFunction().getParent()->getDataLayout();
  auto ElementSizeA = DL.getTypeAllocSize(MMI.A->getElementType());
  auto ElementSizeB = DL.getTypeAllocSize(MMI.B->getElementType());
  auto ElementSizeC = DL.getTypeAllocSize(MMI.WriteToC->getElementType());
  return std::max({ElementSizeA, ElementSizeB, ElementSizeC});
}

/// Get the size of the widest type of the matrix multiplication operands
/// in bits.
///
/// @param MMI Parameters of the matrix multiplication operands.
/// @return The size of the widest type of the matrix multiplication operands
///         in bits.
static uint64_t getMatMulTypeSize(MatMulInfoTy MMI) {
  auto *S = MMI.A->getStatement()->getParent();
  auto &DL = S->getFunction().getParent()->getDataLayout();
  auto ElementSizeA = DL.getTypeSizeInBits(MMI.A->getElementType());
  auto ElementSizeB = DL.getTypeSizeInBits(MMI.B->getElementType());
  auto ElementSizeC = DL.getTypeSizeInBits(MMI.WriteToC->getElementType());
  return std::max({ElementSizeA, ElementSizeB, ElementSizeC});
}

/// Get parameters of the BLIS micro kernel.
///
/// We choose the Mr and Nr parameters of the micro kernel to be large enough
/// such that no stalls caused by the combination of latencies and dependencies
/// are introduced during the updates of the resulting matrix of the matrix
/// multiplication. However, they should also be as small as possible to
/// release more registers for entries of multiplied matrices.
///
/// @param TTI Target Transform Info.
/// @param MMI Parameters of the matrix multiplication operands.
/// @return The structure of type MicroKernelParamsTy.
/// @see MicroKernelParamsTy
static MicroKernelParamsTy getMicroKernelParams(const TargetTransformInfo *TTI,
                                                MatMulInfoTy MMI) {
  assert(TTI && "The target transform info should be provided.");

  // Nvec - Number of double-precision floating-point numbers that can be hold
  // by a vector register. Use 2 by default.
  long RegisterBitwidth = VectorRegisterBitwidth;

  if (RegisterBitwidth == -1)
    RegisterBitwidth =
        TTI->getRegisterBitWidth(TargetTransformInfo::RGK_FixedWidthVector);
  auto ElementSize = getMatMulTypeSize(MMI);
  assert(ElementSize > 0 && "The element size of the matrix multiplication "
                            "operands should be greater than zero.");
  auto Nvec = RegisterBitwidth / ElementSize;
  if (Nvec == 0)
    Nvec = 2;
  int Nr = ceil(sqrt((double)(Nvec * LatencyVectorFma * ThroughputVectorFma)) /
                Nvec) *
           Nvec;
  int Mr = ceil((double)(Nvec * LatencyVectorFma * ThroughputVectorFma / Nr));
  return {Mr, Nr};
}

/// Determine parameters of the target cache.
///
/// @param TTI Target Transform Info.
static void getTargetCacheParameters(const llvm::TargetTransformInfo *TTI) {
  auto L1DCache = llvm::TargetTransformInfo::CacheLevel::L1D;
  auto L2DCache = llvm::TargetTransformInfo::CacheLevel::L2D;
  if (FirstCacheLevelSize == -1) {
    if (TTI->getCacheSize(L1DCache).hasValue())
      FirstCacheLevelSize = TTI->getCacheSize(L1DCache).getValue();
    else
      FirstCacheLevelSize = static_cast<int>(FirstCacheLevelDefaultSize);
  }
  if (SecondCacheLevelSize == -1) {
    if (TTI->getCacheSize(L2DCache).hasValue())
      SecondCacheLevelSize = TTI->getCacheSize(L2DCache).getValue();
    else
      SecondCacheLevelSize = static_cast<int>(SecondCacheLevelDefaultSize);
  }
  if (FirstCacheLevelAssociativity == -1) {
    if (TTI->getCacheAssociativity(L1DCache).hasValue())
      FirstCacheLevelAssociativity =
          TTI->getCacheAssociativity(L1DCache).getValue();
    else
      FirstCacheLevelAssociativity =
          static_cast<int>(FirstCacheLevelDefaultAssociativity);
  }
  if (SecondCacheLevelAssociativity == -1) {
    if (TTI->getCacheAssociativity(L2DCache).hasValue())
      SecondCacheLevelAssociativity =
          TTI->getCacheAssociativity(L2DCache).getValue();
    else
      SecondCacheLevelAssociativity =
          static_cast<int>(SecondCacheLevelDefaultAssociativity);
  }
}

/// Get parameters of the BLIS macro kernel.
///
/// During the computation of matrix multiplication, blocks of partitioned
/// matrices are mapped to different layers of the memory hierarchy.
/// To optimize data reuse, blocks should be ideally kept in cache between
/// iterations. Since parameters of the macro kernel determine sizes of these
/// blocks, there are upper and lower bounds on these parameters.
///
/// @param TTI Target Transform Info.
/// @param MicroKernelParams Parameters of the micro-kernel
///                          to be taken into account.
/// @param MMI Parameters of the matrix multiplication operands.
/// @return The structure of type MacroKernelParamsTy.
/// @see MacroKernelParamsTy
/// @see MicroKernelParamsTy
static MacroKernelParamsTy
getMacroKernelParams(const llvm::TargetTransformInfo *TTI,
                     const MicroKernelParamsTy &MicroKernelParams,
                     MatMulInfoTy MMI) {
  getTargetCacheParameters(TTI);
  // According to www.cs.utexas.edu/users/flame/pubs/TOMS-BLIS-Analytical.pdf,
  // it requires information about the first two levels of a cache to determine
  // all the parameters of a macro-kernel. It also checks that an associativity
  // degree of a cache level is greater than two. Otherwise, another algorithm
  // for determination of the parameters should be used.
  if (!(MicroKernelParams.Mr > 0 && MicroKernelParams.Nr > 0 &&
        FirstCacheLevelSize > 0 && SecondCacheLevelSize > 0 &&
        FirstCacheLevelAssociativity > 2 && SecondCacheLevelAssociativity > 2))
    return {1, 1, 1};
  // The quotient should be greater than zero.
  if (PollyPatternMatchingNcQuotient <= 0)
    return {1, 1, 1};
  int Car = floor(
      (FirstCacheLevelAssociativity - 1) /
      (1 + static_cast<double>(MicroKernelParams.Nr) / MicroKernelParams.Mr));

  // Car can be computed to be zero since it is floor to int.
  // On Mac OS, division by 0 does not raise a signal. This causes negative
  // tile sizes to be computed. Prevent division by Cac==0 by early returning
  // if this happens.
  if (Car == 0)
    return {1, 1, 1};

  auto ElementSize = getMatMulAlignTypeSize(MMI);
  assert(ElementSize > 0 && "The element size of the matrix multiplication "
                            "operands should be greater than zero.");
  int Kc = (Car * FirstCacheLevelSize) /
           (MicroKernelParams.Mr * FirstCacheLevelAssociativity * ElementSize);
  double Cac =
      static_cast<double>(Kc * ElementSize * SecondCacheLevelAssociativity) /
      SecondCacheLevelSize;
  int Mc = floor((SecondCacheLevelAssociativity - 2) / Cac);
  int Nc = PollyPatternMatchingNcQuotient * MicroKernelParams.Nr;

  assert(Mc > 0 && Nc > 0 && Kc > 0 &&
         "Matrix block sizes should be  greater than zero");
  return {Mc, Nc, Kc};
}

/// Create an access relation that is specific to
///        the matrix multiplication pattern.
///
/// Create an access relation of the following form:
/// [O0, O1, O2, O3, O4, O5, O6, O7, O8] -> [OI, O5, OJ]
/// where I is @p FirstDim, J is @p SecondDim.
///
/// It can be used, for example, to create relations that helps to consequently
/// access elements of operands of a matrix multiplication after creation of
/// the BLIS micro and macro kernels.
///
/// @see ScheduleTreeOptimizer::createMicroKernel
/// @see ScheduleTreeOptimizer::createMacroKernel
///
/// Subsequently, the described access relation is applied to the range of
/// @p MapOldIndVar, that is used to map original induction variables to
/// the ones, which are produced by schedule transformations. It helps to
/// define relations using a new space and, at the same time, keep them
/// in the original one.
///
/// @param MapOldIndVar The relation, which maps original induction variables
///                     to the ones, which are produced by schedule
///                     transformations.
/// @param FirstDim, SecondDim The input dimensions that are used to define
///        the specified access relation.
/// @return The specified access relation.
static isl::map getMatMulAccRel(isl::map MapOldIndVar, unsigned FirstDim,
                                unsigned SecondDim) {
  auto AccessRelSpace = isl::space(MapOldIndVar.ctx(), 0, 9, 3);
  auto AccessRel = isl::map::universe(AccessRelSpace);
  AccessRel = AccessRel.equate(isl::dim::in, FirstDim, isl::dim::out, 0);
  AccessRel = AccessRel.equate(isl::dim::in, 5, isl::dim::out, 1);
  AccessRel = AccessRel.equate(isl::dim::in, SecondDim, isl::dim::out, 2);
  return MapOldIndVar.apply_range(AccessRel);
}

static isl::schedule_node createExtensionNode(isl::schedule_node Node,
                                              isl::map ExtensionMap) {
  auto Extension = isl::union_map(ExtensionMap);
  auto NewNode = isl::schedule_node::from_extension(Extension);
  return Node.graft_before(NewNode);
}

static isl::schedule_node optimizePackedB(isl::schedule_node Node,
                                          ScopStmt *Stmt, isl::map MapOldIndVar,
                                          MicroKernelParamsTy MicroParams,
                                          MacroKernelParamsTy MacroParams,
                                          MatMulInfoTy &MMI) {
  Scop *S = Stmt->getParent();
  isl::set Domain = Stmt->getDomain();

  // Create packed array.
  unsigned FirstDimSize = MacroParams.Nc / MicroParams.Nr;
  unsigned SecondDimSize = MacroParams.Kc;
  unsigned ThirdDimSize = MicroParams.Nr;
  ScopArrayInfo *PackedB =
      S->createScopArrayInfo(MMI.B->getElementType(), "Packed_B",
                             {FirstDimSize, SecondDimSize, ThirdDimSize});

  // Compute the access relation for copying from B to PackedB.
  isl::map AccRelB = MMI.B->getLatestAccessRelation();
  isl::map AccRelPackedB = getMatMulAccRel(MapOldIndVar, 3, 7);
  AccRelPackedB =
      AccRelPackedB.set_tuple_id(isl::dim::out, PackedB->getBasePtrId());

  // Create the copy statement and redirect access.
  ScopStmt *CopyStmt = S->addScopStmt(AccRelB, AccRelPackedB, Domain);
  MMI.B->setNewAccessRelation(AccRelPackedB);

  unsigned Dim = unsignedFromIslSize(MapOldIndVar.range_tuple_dim());
  assert(Dim >= 2);
  // Insert into the schedule tree.
  isl::map ExtMap = MapOldIndVar.project_out(isl::dim::out, 2, Dim - 2);
  ExtMap = ExtMap.reverse();
  ExtMap = ExtMap.fix_si(isl::dim::out, MMI.i, 0);
  ExtMap = ExtMap.intersect_range(Domain);
  ExtMap = ExtMap.set_tuple_id(isl::dim::out, CopyStmt->getDomainId());
  return createExtensionNode(Node, ExtMap);
}

static isl::schedule_node optimizePackedA(isl::schedule_node Node, ScopStmt *,
                                          isl::map MapOldIndVar,
                                          MicroKernelParamsTy MicroParams,
                                          MacroKernelParamsTy MacroParams,
                                          MatMulInfoTy &MMI) {
  isl::id InputDimsId = MapOldIndVar.get_tuple_id(isl::dim::in);
  ScopStmt *Stmt = static_cast<ScopStmt *>(InputDimsId.get_user());
  isl::set Domain = Stmt->getDomain();
  isl::id DomainId = Domain.get_tuple_id();

  // Create the packed array.
  unsigned FirstDimSize = MacroParams.Mc / MicroParams.Mr;
  unsigned SecondDimSize = MacroParams.Kc;
  unsigned ThirdDimSize = MicroParams.Mr;
  ScopArrayInfo *PackedA = Stmt->getParent()->createScopArrayInfo(
      MMI.A->getElementType(), "Packed_A",
      {FirstDimSize, SecondDimSize, ThirdDimSize});

  // Compute the access relation for copying from A to PackedA.
  isl::map AccRelA = MMI.A->getLatestAccessRelation();
  isl::map AccRelPackedA = getMatMulAccRel(MapOldIndVar, 4, 6);
  AccRelPackedA =
      AccRelPackedA.set_tuple_id(isl::dim::out, PackedA->getBasePtrId());
  // { MemrefA[] -> PackedA[] }
  isl::map PackedATranslator = AccRelPackedA.apply_domain(AccRelA);

  // Compute the domain for the copy statement.
  // Construct the copy statement domain out of the 3 outermost scatter
  // dimensions (to match the 3 band nodes surrounding the extension node) and
  // the array elements to copy (one statement instance per array element).
  // { Scatter[] }
  isl::set ScatterDomain = MapOldIndVar.intersect_domain(Domain).range();
  // { Scatter[] -> OutermostScatter[] }
  isl::map OuterDomainMap =
      makeIdentityMap(ScatterDomain, true).project_out(isl::dim::out, 3, 6);
  // { Scatter[] -> MemrefA[] }
  isl::map CopyFrom = MapOldIndVar.reverse().apply_range(AccRelA);
  // { Scatter[] -> CopyStmt[] }
  isl::map DomainTranslator = OuterDomainMap.range_product(CopyFrom);
  // { CopyStmt[] }
  isl::set CopyDomain = DomainTranslator.range();

  // Translate the access relations to the new domain.
  // { CopyStmt[] -> MemrefA[] }
  CopyFrom = CopyFrom.apply_domain(DomainTranslator);
  // { CopyStmt[] -> PackedA[] }
  isl::map CopyTo = CopyFrom.apply_range(PackedATranslator);

  // Create the copy statement and redirect access.
  ScopStmt *CopyStmt =
      Stmt->getParent()->addScopStmt(CopyFrom, CopyTo, CopyDomain);
  MMI.A->setNewAccessRelation(AccRelPackedA);

  // Insert into the schedule tree.
  // { Scatter[] -> CopyStmt[] }
  isl::map ExtScatterCopy = makeIdentityMap(CopyStmt->getDomain(), true);
  ExtScatterCopy = ExtScatterCopy.project_out(isl::dim::in, 3, 2);
  return createExtensionNode(Node, ExtScatterCopy);
}

/// Apply the packing transformation.
///
/// The packing transformation can be described as a data-layout
/// transformation that requires to introduce a new array, copy data
/// to the array, and change memory access locations to reference the array.
/// It can be used to ensure that elements of the new array are read in-stride
/// access, aligned to cache lines boundaries, and preloaded into certain cache
/// levels.
///
/// As an example let us consider the packing of the array A that would help
/// to read its elements with in-stride access. An access to the array A
/// is represented by an access relation that has the form
/// S[i, j, k] -> A[i, k]. The scheduling function of the SCoP statement S has
/// the form S[i,j, k] -> [floor((j mod Nc) / Nr), floor((i mod Mc) / Mr),
/// k mod Kc, j mod Nr, i mod Mr].
///
/// To ensure that elements of the array A are read in-stride access, we add
/// a new array Packed_A[Mc/Mr][Kc][Mr] to the SCoP, using
/// Scop::createScopArrayInfo, change the access relation
/// S[i, j, k] -> A[i, k] to
/// S[i, j, k] -> Packed_A[floor((i mod Mc) / Mr), k mod Kc, i mod Mr], using
/// MemoryAccess::setNewAccessRelation, and copy the data to the array, using
/// the copy statement created by Scop::addScopStmt.
///
/// @param Node The schedule node to be optimized.
/// @param MapOldIndVar The relation, which maps original induction variables
///                     to the ones, which are produced by schedule
///                     transformations.
/// @param MicroParams, MacroParams Parameters of the BLIS kernel
///                                 to be taken into account.
/// @param MMI Parameters of the matrix multiplication operands.
/// @return The optimized schedule node.
static isl::schedule_node
optimizeDataLayoutMatrMulPattern(isl::schedule_node Node, isl::map MapOldIndVar,
                                 MicroKernelParamsTy MicroParams,
                                 MacroKernelParamsTy MacroParams,
                                 MatMulInfoTy &MMI) {
  isl::id InputDimsId = MapOldIndVar.get_tuple_id(isl::dim::in);
  ScopStmt *Stmt = static_cast<ScopStmt *>(InputDimsId.get_user());

  Node = Node.parent().parent().parent().parent().parent().parent();
  Node = isl::manage(isl_schedule_node_band_split(Node.release(), 2));

  Node = Node.child(0);
  Node =
      optimizePackedB(Node, Stmt, MapOldIndVar, MicroParams, MacroParams, MMI);

  Node = Node.child(0);
  Node =
      optimizePackedA(Node, Stmt, MapOldIndVar, MicroParams, MacroParams, MMI);

  return Node.child(0).child(0).child(0).child(0).child(0);
}

/// Get a relation mapping induction variables produced by schedule
/// transformations to the original ones.
///
/// @param Node The schedule node produced as the result of creation
///        of the BLIS kernels.
/// @param MicroKernelParams, MacroKernelParams Parameters of the BLIS kernel
///                                             to be taken into account.
/// @return  The relation mapping original induction variables to the ones
///          produced by schedule transformation.
/// @see ScheduleTreeOptimizer::createMicroKernel
/// @see ScheduleTreeOptimizer::createMacroKernel
/// @see getMacroKernelParams
static isl::map
getInductionVariablesSubstitution(isl::schedule_node Node,
                                  MicroKernelParamsTy MicroKernelParams,
                                  MacroKernelParamsTy MacroKernelParams) {
  auto Child = Node.child(0);
  auto UnMapOldIndVar = Child.get_prefix_schedule_union_map();
  auto MapOldIndVar = isl::map::from_union_map(UnMapOldIndVar);
  unsigned Dim = unsignedFromIslSize(MapOldIndVar.range_tuple_dim());
  if (Dim > 9u)
    return MapOldIndVar.project_out(isl::dim::out, 0, Dim - 9);
  return MapOldIndVar;
}

/// Isolate a set of partial tile prefixes and unroll the isolated part.
///
/// The set should ensure that it contains only partial tile prefixes that have
/// exactly Mr x Nr iterations of the two innermost loops produced by
/// the optimization of the matrix multiplication. Mr and Nr are parameters of
/// the micro-kernel.
///
/// In case of parametric bounds, this helps to auto-vectorize the unrolled
/// innermost loops, using the SLP vectorizer.
///
/// @param Node              The schedule node to be modified.
/// @param MicroKernelParams Parameters of the micro-kernel
///                          to be taken into account.
/// @return The modified isl_schedule_node.
static isl::schedule_node
isolateAndUnrollMatMulInnerLoops(isl::schedule_node Node,
                                 MicroKernelParamsTy MicroKernelParams) {
  isl::schedule_node Child = Node.child(0);
  isl::union_map UnMapOldIndVar = Child.get_prefix_schedule_relation();
  isl::set Prefix = isl::map::from_union_map(UnMapOldIndVar).range();
  unsigned Dims = unsignedFromIslSize(Prefix.tuple_dim());
  assert(Dims >= 1);
  Prefix = Prefix.project_out(isl::dim::set, Dims - 1, 1);
  Prefix = getPartialTilePrefixes(Prefix, MicroKernelParams.Nr);
  Prefix = getPartialTilePrefixes(Prefix, MicroKernelParams.Mr);

  isl::union_set IsolateOption =
      getIsolateOptions(Prefix.add_dims(isl::dim::set, 3), 3);
  isl::ctx Ctx = Node.ctx();
  auto Options = IsolateOption.unite(getDimOptions(Ctx, "unroll"));
  Options = Options.unite(getUnrollIsolatedSetOptions(Ctx));
  Node = Node.as<isl::schedule_node_band>().set_ast_build_options(Options);
  Node = Node.parent().parent().parent();
  IsolateOption = getIsolateOptions(Prefix, 3);
  Options = IsolateOption.unite(getDimOptions(Ctx, "separate"));
  Node = Node.as<isl::schedule_node_band>().set_ast_build_options(Options);
  Node = Node.child(0).child(0).child(0);
  return Node;
}

/// Insert "Loop Vectorizer Disabled" mark node.
///
/// @param Node The child of the mark node to be inserted.
/// @return The modified isl_schedule_node.
static isl::schedule_node markLoopVectorizerDisabled(isl::schedule_node Node) {
  auto Id = isl::id::alloc(Node.ctx(), "Loop Vectorizer Disabled", nullptr);
  return Node.insert_mark(Id).child(0);
}

/// Restore the initial ordering of dimensions of the band node
///
/// In case the band node represents all the dimensions of the iteration
/// domain, recreate the band node to restore the initial ordering of the
/// dimensions.
///
/// @param Node The band node to be modified.
/// @return The modified schedule node.
static isl::schedule_node
getBandNodeWithOriginDimOrder(isl::schedule_node Node) {
  assert(isl_schedule_node_get_type(Node.get()) == isl_schedule_node_band);
  if (isl_schedule_node_get_type(Node.child(0).get()) != isl_schedule_node_leaf)
    return Node;
  auto Domain = Node.get_universe_domain();
  assert(isl_union_set_n_set(Domain.get()) == 1);
  if (Node.get_schedule_depth().release() != 0 ||
      (unsignedFromIslSize(isl::set(Domain).tuple_dim()) !=
       unsignedFromIslSize(Node.as<isl::schedule_node_band>().n_member())))
    return Node;
  Node = isl::manage(isl_schedule_node_delete(Node.copy()));
  auto PartialSchedulePwAff = Domain.identity_union_pw_multi_aff();
  auto PartialScheduleMultiPwAff =
      isl::multi_union_pw_aff(PartialSchedulePwAff);
  PartialScheduleMultiPwAff =
      PartialScheduleMultiPwAff.reset_tuple_id(isl::dim::set);
  return Node.insert_partial_schedule(PartialScheduleMultiPwAff);
}

static isl::schedule_node optimizeMatMulPattern(isl::schedule_node Node,
                                                const TargetTransformInfo *TTI,
                                                MatMulInfoTy &MMI) {
  assert(TTI && "The target transform info should be provided.");
  int DimOutNum = isl_schedule_node_band_n_member(Node.get());
  assert(DimOutNum > 2 && "In case of the matrix multiplication the loop nest "
                          "and, consequently, the corresponding scheduling "
                          "functions have at least three dimensions.");
  Node = getBandNodeWithOriginDimOrder(Node);
  Node = permuteBandNodeDimensions(Node, MMI.i, DimOutNum - 3);
  int NewJ = MMI.j == DimOutNum - 3 ? MMI.i : MMI.j;
  int NewK = MMI.k == DimOutNum - 3 ? MMI.i : MMI.k;
  Node = permuteBandNodeDimensions(Node, NewJ, DimOutNum - 2);
  NewK = NewK == DimOutNum - 2 ? NewJ : NewK;
  Node = permuteBandNodeDimensions(Node, NewK, DimOutNum - 1);
  auto MicroKernelParams = getMicroKernelParams(TTI, MMI);
  auto MacroKernelParams = getMacroKernelParams(TTI, MicroKernelParams, MMI);
  Node = createMacroKernel(Node, MacroKernelParams);
  Node = createMicroKernel(Node, MicroKernelParams);
  if (MacroKernelParams.Mc == 1 || MacroKernelParams.Nc == 1 ||
      MacroKernelParams.Kc == 1)
    return Node;
  auto MapOldIndVar = getInductionVariablesSubstitution(Node, MicroKernelParams,
                                                        MacroKernelParams);
  if (MapOldIndVar.is_null())
    return Node;
  Node = markLoopVectorizerDisabled(Node.parent()).child(0);
  Node = isolateAndUnrollMatMulInnerLoops(Node, MicroKernelParams);
  return optimizeDataLayoutMatrMulPattern(Node, MapOldIndVar, MicroKernelParams,
                                          MacroKernelParams, MMI);
}

/// Check if this node contains a partial schedule that could
///        probably be optimized with analytical modeling.
///
/// isMatrMultPattern tries to determine whether the following conditions
/// are true:
/// 1. the partial schedule contains only one statement.
/// 2. there are exactly three input dimensions.
/// 3. all memory accesses of the statement will have stride 0 or 1, if we
///    interchange loops (switch the variable used in the inner loop to
///    the outer loop).
/// 4. all memory accesses of the statement except from the last one, are
///    read memory access and the last one is write memory access.
/// 5. all subscripts of the last memory access of the statement don't
///    contain the variable used in the inner loop.
/// If this is the case, we could try to use an approach that is similar to
/// the one used to get close-to-peak performance of matrix multiplications.
///
/// @param Node The node to check.
/// @param D    The SCoP dependencies.
/// @param MMI  Parameters of the matrix multiplication operands.
static bool isMatrMultPattern(isl::schedule_node Node, const Dependences *D,
                              MatMulInfoTy &MMI) {
  auto PartialSchedule = isl::manage(
      isl_schedule_node_band_get_partial_schedule_union_map(Node.get()));
  Node = Node.child(0);
  auto LeafType = isl_schedule_node_get_type(Node.get());
  Node = Node.parent();
  if (LeafType != isl_schedule_node_leaf ||
      isl_schedule_node_band_n_member(Node.get()) < 3 ||
      Node.get_schedule_depth().release() != 0 ||
      isl_union_map_n_map(PartialSchedule.get()) != 1)
    return false;
  auto NewPartialSchedule = isl::map::from_union_map(PartialSchedule);
  if (containsMatrMult(NewPartialSchedule, D, MMI))
    return true;
  return false;
}

} // namespace

isl::schedule_node
polly::tryOptimizeMatMulPattern(isl::schedule_node Node,
                                const llvm::TargetTransformInfo *TTI,
                                const Dependences *D) {
  MatMulInfoTy MMI;
  if (isMatrMultPattern(Node, D, MMI)) {
    LLVM_DEBUG(dbgs() << "The matrix multiplication pattern was detected\n");
    return optimizeMatMulPattern(Node, TTI, MMI);
  }
  return {};
}
