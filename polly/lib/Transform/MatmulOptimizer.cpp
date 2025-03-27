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
#include "polly/Support/GICHelper.h"
#include "polly/Support/ISLTools.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SetOperations.h"
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

#include "polly/Support/PollyDebug.h"
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
    cl::Hidden, cl::init(8), cl::cat(PollyCategory));

static cl::opt<int> ThroughputVectorFma(
    "polly-target-throughput-vector-fma",
    cl::desc("A throughput of the processor floating-point arithmetic units "
             "expressed in the number of vector fused multiply-add "
             "instructions per clock cycle."),
    cl::Hidden, cl::init(1), cl::cat(PollyCategory));

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

static cl::opt<bool>
    PMBasedTCOpts("polly-tc-opt",
                  cl::desc("Perform optimizations of tensor contractions based "
                           "on pattern matching"),
                  cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<bool>
    PMBasedMMMOpts("polly-matmul-opt",
                   cl::desc("Perform optimizations of matrix multiplications "
                            "based on pattern matching"),
                   cl::init(true), cl::ZeroOrMore, cl::cat(PollyCategory));

static cl::opt<int> OptComputeOut(
    "polly-tc-dependences-computeout",
    cl::desc("Bound the dependence analysis by a maximal amount of "
             "computational steps (0 means no bound)"),
    cl::Hidden, cl::init(500000), cl::ZeroOrMore, cl::cat(PollyCategory));

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

/// Parameters of the tensor contraction operands.
///
/// A general d-dimensional tensor T ∈ R ^ Nu0 x ... x Nud−1 can be defined
/// as the set of scalar elements indexed by the set of indices u0 ... ud,
///
/// T ≡ {Anu0...nud−1 ∈ R | (u0,...,ud−1) ∈ Nu0 x ... x Nud−1}.
///
/// Let A, B, and C be dA, dB, and dC-dimensional tensors, respectively.
/// Let the free and the contracted indices of the tensor A be grouped into
/// two bundles I = i0...ir−1 and P = p0...pt−1, respectively. Similarly,
/// the free and the contracted indices of B are grouped into bundles
/// J = j0..js−1 and P and the free indices of C are grouped into
/// bundles I and J.
///
/// Tensor contraction (TC) of tensors A, B into tensor C can be represented as
/// C(shuffle(I,J))=∑α·A(shuffle(I,P))·B(shuffle(P,J))+β·C(shuffle(I,J)),
/// where ∑ is a summation over all contracted indices of P,
/// α, β ∈ R, Npi is the length of the tensor dimension that corresponds
/// to the index pi, A(shuffle(I, P)), B(shuffle(P, J)), C(shuffle(I, J)) are
/// accesses to tensors A, B, C, respectively,
/// shuffle(I, J), shuffle(I, P), and shuffle(P, J) are permutations of
/// the enclosed indices.
///
/// Multiplication of C(shuffle(I,J)) by β can be moved into a different SCoP
/// statement by loop distribution, which is done by the isl scheduler.
//  If β is not equal to one, the optimization of TC of Polly requires
/// such a transformation.
///
/// TCInfoTy contains parameters, which describe access relations that represent
/// operands of the tensor contraction.
struct TCInfoTy {
  /// @{
  /// Memory accesses that represent reading from tensors, which are operands of
  /// the tensor contraction.
  MemoryAccess *A = nullptr;
  MemoryAccess *B = nullptr;
  /// @}

  /// @{
  /// Memory accesses that represent reading from and writing into the tensor,
  /// which contains the result of the tensor contraction.
  MemoryAccess *ReadFromC = nullptr;
  MemoryAccess *WriteToC = nullptr;
  /// @}

  /// @{
  /// Input dimensions of the schedule space, which represent free
  /// indices of tensors.
  SmallDenseSet<int> I;
  SmallDenseSet<int> J;
  /// @}

  /// Input dimension of the schedule space, which represents contracted
  /// indices of tensors.
  SmallDenseSet<int> P;

  /// @{
  /// Sizes of tensor dimensions for corresponding input dimensions of
  /// the schedule space. The size of the tensor dimension can be larger than
  /// the size of the corresponding input dimension of the schedule space.
  /// This does not correspond to a tensor contraction. However, such a pattern
  /// will be optimized by the transformation.
  SmallVector<int> DimensionSizes;
  SmallVector<int> ADimensions;
  SmallVector<int> BDimensions;
  SmallVector<int> CDimensions;
  /// @}

  /// @{
  /// Permutations of indices of I, J, and P, which describe operands of
  /// the tensor contraction and its result.
  SmallVector<int> OrderedI;
  SmallVector<int> OrderedJ;
  SmallVector<int> OrderedP;
  /// @}
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
/// modified dimensions. The values of MacroKernelParams's fields are used
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

  return Node.child(0).child(0);
}

/// Get the size of the widest type of the matrix multiplication operands
/// in bytes, including alignment padding.
///
/// @param MMI Parameters of the matrix multiplication operands.
/// @return The size of the widest type of the matrix multiplication operands
///         in bytes, including alignment padding.
static uint64_t getMatMulAlignTypeSize(const MatMulInfoTy &MMI) {
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
static uint64_t getMatMulTypeSize(const MatMulInfoTy &MMI) {
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
                                                const MatMulInfoTy &MMI) {
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
    if (TTI->getCacheSize(L1DCache))
      FirstCacheLevelSize = TTI->getCacheSize(L1DCache).value();
    else
      FirstCacheLevelSize = static_cast<int>(FirstCacheLevelDefaultSize);
  }
  if (SecondCacheLevelSize == -1) {
    if (TTI->getCacheSize(L2DCache))
      SecondCacheLevelSize = TTI->getCacheSize(L2DCache).value();
    else
      SecondCacheLevelSize = static_cast<int>(SecondCacheLevelDefaultSize);
  }
  if (FirstCacheLevelAssociativity == -1) {
    if (TTI->getCacheAssociativity(L1DCache))
      FirstCacheLevelAssociativity =
          TTI->getCacheAssociativity(L1DCache).value();
    else
      FirstCacheLevelAssociativity =
          static_cast<int>(FirstCacheLevelDefaultAssociativity);
  }
  if (SecondCacheLevelAssociativity == -1) {
    if (TTI->getCacheAssociativity(L2DCache))
      SecondCacheLevelAssociativity =
          TTI->getCacheAssociativity(L2DCache).value();
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
                     const MatMulInfoTy &MMI) {
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
  if (isl_schedule_node_band_n_member(Node.get()) < 3 ||
      Node.get_schedule_depth().release() != 0 ||
      isl_union_map_n_map(PartialSchedule.get()) != 1)
    return false;
  auto NewPartialSchedule = isl::map::from_union_map(PartialSchedule);
  if (containsMatrMult(NewPartialSchedule, D, MMI))
    return true;
  return false;
}

/// Get the dimension size.
///
/// Return the size of the dimension @p Pos, which is obtained from @p SAI.
/// Return -1 in the case of the first dimension of a multi-dimensional array,
/// since the ScopArrayInfo class does not carry size information.
///
/// @param SAI The information about the array.
/// @param Pos The position of the dimension.
/// @return The size of the dimension.
static int getDimSize(const ScopArrayInfo *SAI, unsigned Pos) {
  if (Pos == 0)
    return -1;
  const llvm::SCEV *SCEVDimSize = SAI->getDimensionSize(Pos);
  assert(SCEVDimSize);
  auto *ConstantDimSize = dyn_cast<const SCEVConstant>(SCEVDimSize);
  assert(ConstantDimSize);
  auto *IntDimSize = dyn_cast<ConstantInt>(ConstantDimSize->getValue());
  assert(IntDimSize);
  return IntDimSize->getSExtValue();
}

/// Check whether the access relation has the specified form.
///
/// Check that the access relation @p AccMap has the form T[I0, …, In], where
/// indexes I0, …, In are specified by @p Dimensions.
///
/// @param Domain     The domain of the access relation.
/// @param AccMap     The access relation to be checked.
/// @param Dimensions The permutation of the subset of the input dimensions.
/// @return True if @p AccMap has the expected form and false,
///         otherwise.
static bool isCorrectAccessMap(isl::set Domain, isl::map AccMap,
                               ArrayRef<int> Dimensions) {
  isl::space Space = AccMap.get_space();
  if (unsignedFromIslSize(Space.dim(isl::dim::out)) != Dimensions.size())
    return false;

  // Create an access relation of the following form:
  // [I0, …, Im] -> [Il, …, In], where indexes
  // Il, …, In are specified by @p Dimensions.
  isl::map PossibleTensor = isl::map::universe(Space);
  unsigned DimInSize = unsignedFromIslSize(Space.dim(isl::dim::in));
  for (unsigned i = 0; i < Dimensions.size(); i++) {
    const int InPos = Dimensions[i];
    if ((InPos >= static_cast<int>(DimInSize)) || (InPos < 0))
      return false;
    PossibleTensor =
        PossibleTensor.equate(isl::dim::in, InPos, isl::dim::out, i);
  }

  AccMap = AccMap.intersect_domain(Domain);
  PossibleTensor = PossibleTensor.intersect_domain(Domain);

  // If AccMap != PossibleTensor here (the two maps have been gisted at
  // this point), it means that the writes are not complete, or in other
  // words, it is a Partial write and Partial writes must be rejected.
  return AccMap.is_equal(PossibleTensor);
}

/// Check whether the access represents the tensor contraction operand.
///
/// Check that the access relation @p AccMap has the form T[i1, …, in].
/// Obtained indexes i1, …, in, their sizes and their permutation are stored
/// into @p IndexSet, @p DimensionSizes, and @p Dimensions, respectively.
///
/// @param Domain         The domain of the access relation.
/// @param AccMap         The access relation to be checked.
/// @param IndexSet       The subset of the input dimensions.
/// @param DimensionSizes Sizes of the input dimensions of @p Dimensions.
/// @param Dimensions     The permutation of the subset of the input dimensions.
/// @return True if @p AccMap has the expected form and false,
///         otherwise.
static bool isTCOperandAcc(isl::set Domain, isl::map AccMap,
                           SmallDenseSet<int> &IndexSet,
                           SmallVectorImpl<int> &DimensionSizes,
                           SmallVectorImpl<int> &Dimensions) {
  isl::id Id = AccMap.get_tuple_id(isl::dim::out);
  const ScopArrayInfo *SAI = ScopArrayInfo::getFromId(Id);
  assert(SAI && "AccMap should represent memory access");

  // Fix values of output dimensions with respect to their positions.
  // In the case of the tensor contraction, values of output dimensions are
  // fixed and form a permutation of a subset of values of input dimensions.
  //
  // For example, in the case of Stmt[i][j][k] -> A[k][i], which represents
  // the operand of the tensor contraction, we get the following map by fixing
  // the output dimensions Stmt[1][j][0] -> A[0][1].
  //
  // We store the permutation of the subset of the input dimensions {2, 0} into
  // @p Dimensions.
  //
  // The obtained permutation and the isCorrectAccessMap function are used to
  // check whether the access relation @p AccMap represents the tensor
  // contraction operand. For example, in the case of
  // Stmt[i][j][k] -> A[i-1][j+1], we get Stmt[1][0][k] -> A[0][1] and,
  // consequently, {1, 0}, which is rejected by isCorrectAccessMap,
  // since it corresponds to Stmt[i][j][k] -> A[j][i].
  isl::map CheckMap = isl::manage(AccMap.copy());
  unsigned OutDimNum = unsignedFromIslSize(CheckMap.dim(isl::dim::out));
  for (unsigned i = 0; i < OutDimNum; i++)
    CheckMap = CheckMap.fix_si(isl::dim::out, i, i);

  // Try to obtain the permutation and sizes of corresponding input dimensions.
  Dimensions.assign(OutDimNum, -1);
  for (unsigned i : rangeIslSize(0, CheckMap.dim(isl::dim::in))) {
    isl::val Val = getConstant(CheckMap, isl::dim::in, i);
    if (!Val.is_int())
      continue;
    int OutPos = -1;
    llvm::APInt ValAPInt = APIntFromVal(Val);
    if (ValAPInt.isSignedIntN(32))
      OutPos = ValAPInt.getSExtValue();
    if ((OutPos < 0) || (OutPos >= static_cast<int>(OutDimNum)) ||
        IndexSet.count(i))
      return false;
    IndexSet.insert(i);
    Dimensions[OutPos] = i;
    if (DimensionSizes[i] <= 0)
      DimensionSizes[i] = getDimSize(SAI, OutPos);
  }

  return isCorrectAccessMap(Domain, AccMap, Dimensions);
}

/// Find the intersection of two sets.
///
/// Find the intersection of the set @p A and the set @p B.
///
/// @param A, B Sets to intersect.
/// @return The set intersection.
static SmallDenseSet<int> intersect(const SmallDenseSet<int> &A,
                                    const SmallDenseSet<int> &B) {
  SmallDenseSet<int> Intersection = A;
  set_intersect(Intersection, B);
  return Intersection;
}

/// Check whether the set is a superset.
///
/// Check that the set @p A is a superset of @p B.
///
/// @param A, B Sets to be checked.
/// @return True if the set A is a superset of B.
static bool isSuperset(const SmallDenseSet<int> &A,
                       const SmallDenseSet<int> &B) {
  return intersect(A, B).size() == B.size();
}

/// Find the union of two sets.
///
/// Find the union of the set @p A and the set @p B.
///
/// @param A, B Sets to unite.
/// @return The set union.
static SmallDenseSet<int> unite(const SmallDenseSet<int> &A,
                                const SmallDenseSet<int> &B) {
  SmallDenseSet<int> Union = A;
  set_union(Union, B);
  return Union;
}

/// Determine the access that writes to the tensor, which contains
/// the result of the tensor contraction.
///
/// @param Domain        The domain of the statement.
/// @param Stmt          The statement, which writes to memory.
/// @param TCI           The information about the tensor contraction.
/// @param IandJIndexSet The set, which contains free indexes of tensors.
/// @return The determined MemoryAccess, or nullptr if there is no necessary
///         access within the SCoP.
static MemoryAccess *getWriteAccess(isl::set Domain, ScopStmt *Stmt,
                                    TCInfoTy &TCI,
                                    SmallDenseSet<int> &IandJIndexSet) {
  TCI.WriteToC = nullptr;
  SmallVector<MemoryAccess *, 32> Accesses = getAccessesInOrder(*Stmt);
  for (MemoryAccess *MemA : reverse(Accesses)) {
    // A TC-like does not contain write scalar memory accesses
    if (!MemA->isLatestArrayKind())
      return nullptr;
    // The last memory access should be a write memory access.
    if (!MemA->isWrite())
      return nullptr;

    isl::map AccMap = MemA->getLatestAccessRelation();
    if (!isTCOperandAcc(Domain, AccMap, IandJIndexSet, TCI.DimensionSizes,
                        TCI.CDimensions))
      return nullptr;

    return MemA;
  }
  return nullptr;
}

/// Determine an access, which reads elements of an operand of the tensor
/// contraction
///
/// @param MemAccessPtr  The access, which reads elements of the tensor.
/// @param IndexSet      The set, which contains indexes of the tensors.
/// @param IandJIndexSet The set, which contains free indexes of tensors.
/// @param Dimensions    The permutation of the subset of the input dimensions.
/// @param TCI           The information about the tensor contraction.
/// @return True if the memory access @p MemAccessPtr corresponds
///         to the tensor contraction.
static bool setReadAccess(MemoryAccess *MemAccessPtr,
                          const SmallDenseSet<int> &IndexSet,
                          const SmallDenseSet<int> &IandJIndexSet,
                          ArrayRef<int> Dimensions, TCInfoTy &TCI) {
  if (!TCI.A) {
    // Probably IndexSet is a union of I and P sets.
    if (!isSuperset(IndexSet, TCI.P))
      return false;

    // Obtain the set I.
    TCI.I = set_difference(IndexSet, TCI.P);
    if (!isSuperset(IandJIndexSet, TCI.I))
      return false;

    // Obtain the set J.
    TCI.J = set_difference(IandJIndexSet, TCI.I);

    // Set the first operand of the tensor contraction.
    TCI.A = MemAccessPtr;
    llvm::replace(TCI.ADimensions, TCI.ADimensions.begin(),
                  TCI.ADimensions.end(), Dimensions.begin(), Dimensions.end());
    return true;
  }

  if (!TCI.B) {
    // IndexSet should be a union of J and P sets.
    if (unite(TCI.P, TCI.J) != IndexSet)
      return false;

    // Set the second operand of the tensor contraction.
    TCI.B = MemAccessPtr;
    llvm::replace(TCI.BDimensions, TCI.BDimensions.begin(),
                  TCI.BDimensions.end(), Dimensions.begin(), Dimensions.end());
    return true;
  }

  return false;
}

/// Check that all memory accesses of the statement, except from the last
/// one, are read memory accesses, which read elements of operands of the tensor
/// contraction and its result.
///
/// @param Domain        The domain of the statement.
/// @param Stmt          The statement, which writes to memory.
/// @param TCI           The information about the tensor contraction.
/// @param IandJIndexSet The set, which contains free indexes of tensors.
/// @return True if all read memory accesses of the statement @p Stmt correspond
///         to the tensor contraction.
static bool setReadAccesses(isl::set Domain, ScopStmt *Stmt, TCInfoTy &TCI,
                            SmallDenseSet<int> &IandJIndexSet) {
  TCI.A = nullptr;
  TCI.B = nullptr;
  TCI.ReadFromC = nullptr;
  SmallVector<MemoryAccess *, 32> Accesses = getAccessesInOrder(*Stmt);
  for (auto *MemA = Accesses.begin(); *MemA != TCI.WriteToC; MemA++) {
    MemoryAccess *MemAccessPtr = *MemA;

    // All memory accesses, except from the last one, should be read memory
    // accesses.
    if (MemAccessPtr->isWrite())
      return false;

    isl::map AccMap = MemAccessPtr->getLatestAccessRelation();

    if (!MemAccessPtr->isLatestArrayKind()) {
      // Check whether the scalar read memory access is not partial.
      if (!Domain.is_subset(AccMap.domain()))
        return false;
      continue;
      return false;
    }

    // There is only one memory access, which reads elements of the result of
    // the tensor contraction.
    if (AccMap.is_equal(TCI.WriteToC->getLatestAccessRelation())) {
      if (TCI.ReadFromC)
        return false;
      TCI.ReadFromC = MemAccessPtr;
      continue;
    }

    SmallVector<int> Dimensions;
    SmallDenseSet<int> IndexSet;
    if (!isTCOperandAcc(Domain, AccMap, IndexSet, TCI.DimensionSizes,
                        Dimensions))
      return false;

    if (!setReadAccess(MemAccessPtr, IndexSet, IandJIndexSet, Dimensions, TCI))
      return false;
  }

  // Check that there are read memory accesses, which read elements of operands
  // of the tensor contraction and its result.
  return TCI.ReadFromC && TCI.A && TCI.B;
}

/// Check accesses to operands of the tensor contraction.
///
/// Check that accesses of the SCoP statement, which corresponds to
/// the partial schedule @p PartialSchedule, represent accesses
/// to the non-scalar operands of the tensor contraction.
///
/// @param  Domain          The domain of the SCoP statement.
/// @param  PartialSchedule The partial schedule of the SCoP statement.
/// @param  TCI             Parameters of the tensor contraction operands.
/// @return                 True if the corresponding SCoP statement
///                         represents tensor contraction and false,
///                         otherwise.
static bool containsOnlyTCAcc(isl::set Domain, isl::map PartialSchedule,
                              TCInfoTy &TCI) {
  isl::id InputDimsId = PartialSchedule.get_tuple_id(isl::dim::in);
  ScopStmt *Stmt = static_cast<ScopStmt *>(InputDimsId.get_user());

  // In region statements, the order of memory accesses execution is not
  // predictable at compile-time.
  if ((Stmt->size() <= 1) || Stmt->isRegionStmt())
    return false;

  unsigned DimNum = unsignedFromIslSize(PartialSchedule.dim(isl::dim::in));
  TCI.DimensionSizes.resize(DimNum);
  SmallDenseSet<int> IandJIndexSet;

  TCI.WriteToC = getWriteAccess(Domain, Stmt, TCI, IandJIndexSet);
  if (!TCI.WriteToC)
    return false;

  if (intersect(IandJIndexSet, TCI.P).size() != 0)
    return false;

  if (!setReadAccesses(Domain, Stmt, TCI, IandJIndexSet))
    return false;

  return true;
}

/// Check that dependency corresponds to the tensor contraction carried over
/// loop dimension @p Dim.
///
/// Check that the dependency has the form
/// S(..., ki, max(k(i + 1)), ..., max(kn), ...) ->
/// S(..., ki + 1, min(k(i + 1)), ..., min(kn), ...), where S is the SCoP
/// statement. For this purpose, we analyze the set @p DepDelta, which
/// represents the differences between image elements and domain elements of
/// the corresponding map.
///
/// @param  DepDelta    The set contains the differences between image elements
///                     and corresponding domain elements of the map, which
///                     represents the dependency.
/// @param  Dim         The position of the index ki.
/// @param  BoundDeltas In the case of indexes of ki, the difference between
///                     image elements and corresponding domain elements
///                     corresponds to the difference between lexicographic
///                     minimum and lexicographic maximum of the corresponding
///                     dimension of the domain of the statement.
/// @param  IndexSet    Obtained indexes ki, which describe the dependency.
/// @return True if dependencies correspond to the tensor contraction
///         and false, otherwise.
static bool isReductionCarriedOverDim(isl::set DepDelta, unsigned Dim,
                                      isl::pw_multi_aff BoundDeltas,
                                      const SmallDenseSet<int> &IndexSet) {
  isl::space Space = DepDelta.get_space();
  isl::set Superset = isl::set::universe(Space);
  for (unsigned i = 0; i < Dim; i += 1)
    Superset = Superset.fix_si(isl::dim::set, i, 0);
  Superset = Superset.fix_si(isl::dim::set, Dim, 1);

  // Check that the difference between the image element and the domain element
  // is equal to one in the case of the index ki. Image elements and
  // corresponding domain elements should be equal in the case of positions,
  // which are lower than the specified position.
  if (!DepDelta.is_subset(Superset))
    return false;

  // Compute a set, which is used to analyze how values of
  // the domain are related to the map that describes the dependency.
  isl_pw_multi_aff *DepDeltaPW = isl_pw_multi_aff_from_set(DepDelta.copy());
  BoundDeltas = BoundDeltas.add(isl::manage(DepDeltaPW));
  isl_set *ComplementRawSet = isl_set_from_pw_multi_aff(BoundDeltas.release());
  isl::set Complement = isl::manage(ComplementRawSet);

  for (unsigned i : rangeIslSize(Dim + 1, DepDelta.dim(isl::dim::set))) {
    if (!IndexSet.count(i)) {
      // Check the difference between the image element and the domain element
      // in the case of indexes, which do not describe the dependency.
      if (DepDelta.plain_get_val_if_fixed(isl::dim::set, i).is_zero())
        continue;
      return false;
    }

    // In the case of other indexes, which describe the dependency,
    // the difference between the image element and the domain element
    // should be equal to the difference between lexicographic minimum and
    // lexicographic maximum of the domain of the statement.
    if (!Complement.plain_get_val_if_fixed(isl::dim::set, i).is_zero())
      return false;
  }

  return true;
}

/// Check whether dependencies are over the complete domain.
///
/// In the case of the tensor contraction RAW, WAW, WAR dependencies
/// have the form
/// S(..., ki, max(k(i + 1)), ..., max(kn), ...) ->
/// S(..., ki + 1, min(k(i + 1)), ..., min(kn), ...), where S is the SCoP
/// statement. Consequently, the domain of the dependencies
/// can be described as
/// Domain / Domain ∩ S(…, max(kn),…) ∩ S(…, max(k(i + 1)),…),
/// where Domain is the domain of the statement S.
///
/// For example, in the case of the following tensor contraction,
/// corresponding domains will have the following form.
///
/// An example of the tensor contraction:
/// for (i = 0; i < 1024; i++)
///   for (j = 0; j < 1024; j++)
///     for (l = 0; l < 64; ++l)
///       for (w = 0; w < 64; ++w)
///         C[i][j] += A[i][l][w] * B[w][j][l];
///
/// The domain of the statement:
/// { S[i0, i1, i2, i3] : i0 >= 0 and i0 <= 1023 and
///                       i1 >= 0 and i1 <= 1023 and
///                       i2 >= 0 and i2 <= 63 and
///                       i3 >= 0 and i3 <= 63 }
///
/// The domain of the dependencies:
/// { S[i0, i1, i2, i3] : (i0 >= 0 and i0 <= 1023 and
///                        i1 >= 0 and i1 <= 1023 and
///                        i2 >= 0 and i2 <= 63 and
///                        i3 >= 0 and i3 <= 62) or
///                       (i3 = 63 and i0 >= 0 and i0 <= 1023 and
///                        i1 >= 0 and i1 <= 1023 and
///                        i2 >= 0 and i2 <= 62) }
///
/// @param  Domain       The domain of the statement.
/// @param  DepsForStmt  RAW and RED dependencies for the statement.
/// @param  UpperBound   The lexicographic maximum of the elements in
///                      the @p Domain.
/// @param  IndexSet     Obtained indexes ki, which describe the dependencies.
/// @return True if dependencies are over the complete domain
///         and false, otherwise.
static bool areDepsOverCompleteDomain(isl::set Domain, isl::map DepsForStmt,
                                      isl::pw_multi_aff UpperBound,
                                      SmallDenseSet<int> &IndexSet) {
  isl_set *UpperBoundRawSet = isl_set_from_pw_multi_aff(UpperBound.copy());
  isl::set UpperBoundSet = isl::manage(UpperBoundRawSet);

  isl::set DomainRed = isl::manage(Domain.copy());
  for (const auto It : IndexSet) {
    isl::val FixedVal = UpperBoundSet.plain_get_val_if_fixed(isl::dim::set, It);
    if (FixedVal.is_nan())
      return false;
    DomainRed = isl::manage(
        isl_set_fix_val(DomainRed.copy(), isl_dim_set, It, FixedVal.release()));
  }
  return DepsForStmt.domain().intersect(Domain).is_equal(
      Domain.subtract(DomainRed));
}

/// Check that dependencies correspond to the tensor contraction.
///
/// Check that there are only true dependencies of the form
/// S(..., ki, max(k(i + 1)), ..., max(kn), ...) ->
/// S(..., ki + 1, min(k(i + 1)), ..., min(kn), ...), where S is the SCoP
/// statement represented by @p Schedule. Such dependencies are produced by
/// the tensor contraction. Obtained indexes ki are stored into @p IndexSet.
///
/// The form of anti and output dependencies is specified implicitly by
/// the form the SCoP statement, which is checked by subsequent analysis.
///
/// @param  Schedule The schedule of the SCoP statement.
/// @param  D        The SCoP dependencies.
/// @param  Domain   The domain of the statement.
/// @param  IndexSet Obtained indexes ki, which describe the dependencies.
/// @return True if dependencies correspond to the tensor contraction
///         and false, otherwise.
static bool containsOnlyTcDeps(isl::map Schedule, const Dependences *D,
                               SmallDenseSet<int> &IndexSet, isl::set Domain) {
  IslMaxOperationsGuard MaxOpGuard(Schedule.ctx().get(), OptComputeOut);

  isl::union_map Dep =
      D->getDependences(Dependences::TYPE_RAW | Dependences::TYPE_RED);

  isl::space DomainSpace = Schedule.get_space().domain();
  isl::space Space = DomainSpace.map_from_domain_and_range(DomainSpace);
  isl::map DepsForStmt = Dep.extract_map(Space);
  isl::set DepDeltas = DepsForStmt.deltas();
  isl::size DeltasDimNum = DepDeltas.dim(isl::dim::set);
  isl::pw_multi_aff LowerBound = Domain.lexmin_pw_multi_aff();
  isl::pw_multi_aff UpperBound = Domain.lexmax_pw_multi_aff();
  isl::pw_multi_aff BoundDeltas = UpperBound.sub(LowerBound);

  for (int i : reverse(rangeIslSize(0, DeltasDimNum))) {
    // In the case of the tensor contraction, the difference between image
    // elements and domain elements lies on a hyperplane where a dimension
    // has the fixed value one.
    isl::set Intersection = DepDeltas.fix_si(isl::dim::set, i, 1);
    if (Intersection.is_empty())
      continue;

    if (!isReductionCarriedOverDim(Intersection, i, BoundDeltas, IndexSet))
      return false;

    IndexSet.insert(i);
    DepDeltas = DepDeltas.subtract(Intersection);
  }

  // In the case of the tensor contraction, all dependencies should have
  // the previously described form.
  if ((unsignedFromIslSize(DeltasDimNum) == 0) || !DepDeltas.is_empty())
    return false;

  return areDepsOverCompleteDomain(Domain, DepsForStmt, UpperBound, IndexSet);
}

/// Check if the SCoP statement could probably be optimized with analytical
/// modeling.
///
/// containsTCInfoTy tries to determine whether the following conditions
/// are true:
///
/// 1. The last memory access modeling an array, MA1, represents writing to
///    memory and has the form S(..., I, ..., J, ...) -> M(shuffle(I, J)),
///    where S is the SCoP statement under consideration and shuffle(I, J)
///    is a permutation of indexes of sets I and J.
/// 2. There are only true dependencies of the form
///    S(..., ki, max(k(i + 1)), ..., max(kn), ...) ->
///    S(..., ki + 1, min(k(i + 1)), ..., min(kn), ...), where S is the SCoP
///    statement represented by @p Schedule and ki are indexes of the set P.
/// 3. SCoP contains an arbitrary number of reads from constants and only three
///    access relations, MA2, MA3, and MA4 that represent reading from memory
///    and have the form
///    S(..., I, ..., P, ...) -> M(shuffle(I, P)),
///    S(..., P, ..., J, ...) -> M(shuffle(J, P)),
///    S(...) -> M(shuffle(I, J)), respectively.
///
/// @param  PartialSchedule The PartialSchedule that contains a SCoP statement
///                         to check.
/// @param  D               The SCoP dependencies.
/// @param  TCI             Parameters of the tensor contraction operands.
/// @param  Domain          The domain of the statement.
/// @return True if dependencies and memory accesses correspond to the tensor
///              contraction and false, otherwise.
static bool containsTCInfoTy(isl::map PartialSchedule, const Dependences *D,
                             TCInfoTy &TCI, isl::set Domain) {
  if (!containsOnlyTcDeps(PartialSchedule, D, TCI.P, Domain))
    return false;

  // TODO: handle cases of scalar multiplication if needed.
  if (TCI.P.size() == 0)
    return false;

  if (!containsOnlyTCAcc(Domain, PartialSchedule, TCI))
    return false;

  // TODO: handle cases of GEMV if needed.
  if ((TCI.I.size() == 0) || (TCI.J.size() == 0))
    return false;

  return true;
}

/// Check if this node contains a partial schedule that could
/// probably be optimized with analytical modeling.
///
/// isTCPattern is used to determine whether the SCoP represents a TC-like
/// kernel [1], which is a perfectly nested set of loops, with a data usage
/// pattern that is similar to that produced by the tensor contraction.
///
/// A TC-like kernel can be defined as follows:
///
/// 1. It satisfies the requirements of the polyhedral model.
/// 2. Without loss of generality, it contains three nonempty bundles of
///    one-dimensional for-loops with induction variables that are grouped into
///    bundles I = i0...i(r-1), J = j0..j(s-1), and P = p0...p(t-1), and they
///    are incremented by one.
/// 3. The innermost loop body can be represented as a statement of the form
///    C(shuffle(I, J)) = E(A(shuffle(I, P)), B(shuffle(P, J)),
///    C(shuffle(I, J))), where A(shuffle(I, P)), B(shuffle(P, J)),
///    C(shuffle(I, J)) are accesses to tensors A, B, C, respectively,
///    shuffle(I, J), shuffle(I, P), and shuffle(P, J) are permutations of the
///    enclosed indices, and E is an expression that contains reads from
///    the tensors A, B, C, and an arbitrary number of reads from constants
///    with respect to bundles I, J, and P.
///
/// TC can be considered as a particular case of a TC-like kernel.
///
/// The order of loops with indexes from P should be preserved. Otherwise,
/// isTCPattern should check if a commutative operation is used.
///
/// isTCPattern performs the following steps to check whether the SCoP
/// corresponds to a definition of a TC-like kernel:
///
/// 1. Checks that the node is the innermost band node.
/// 2. Checks that the partial schedule contains only one statement.
/// 3. Check that all ancestors of the node contain all band nodes for
///    the statement and only mark nodes interleave such band nodes. This
///    corresponds to a straightforward implementation of TC.
/// 4. Analyses the dependencies to determine contraction dimensions.
/// 5. Check that the last memory access modeling an array, represents writing
///    to the result of the TC-like kernel.
/// 6. Check that SCoP contains only three access relations that represent
///    reading of the operands of the TC-like kernel and an arbitrary number of
///    reads from constants.
///
/// [1] - Gareev R., Grosser T., Kruse M. High-Performance Generalized Tensor
///       Operations: A Compiler-Oriented Approach // ACM Transactions
///       Architecture and Code Optimization (TACO). 2018.
///       Vol. 15, no. 3. P. 34:1–34:27. DOI: 10.1145/3235029.
///
/// If this is the case, we could logically represent tensors as matrices and
/// apply algorithms, which are used to get close-to-peak performance of
/// matrix multiplications in manually tuned BLAS libraries (e.g., BLIS).
///
/// @param Node The node to check.
/// @param D    The SCoP dependencies.
/// @param TCI  Parameters of the tensor contraction operands.
static bool isTCPattern(isl::schedule_node Node, const Dependences *D,
                        TCInfoTy &TCI) {
  Node = Node.child(0);
  isl::union_map PartialSchedule = Node.get_prefix_schedule_union_map();
  isl::union_set Domain = Node.domain();
  Node = Node.parent();

  // The partial schedule should contain only one statement.
  // TODO: This constraint should not be intrinsic to the algorithm.
  if (isl_union_set_n_set(Domain.get()) != 1)
    return false;

  isl_schedule_node_type NodeType = isl_schedule_node_get_type(Node.get());

  // Check that all ancestors of the node contain all band nodes for
  // the statement, which represents the TC-like kernel, and only mark nodes
  // interleave such band nodes. This corresponds to a straightforward
  // implementation of TC with/without DeLICM applied.
  //
  // For example, this covers the matrix multiplication pattern after a full
  // run of -polly-optree and -polly-delicm, where the write access is not
  // through the original memory access, but through a PHI node that was
  // delicmed. Subsequently, such band nodes will be replaced by a single band
  // node.
  //
  // The corresponding schedule can be the following, where Stmt_for_body8
  // contains the matrix multiplication:
  //
  // domain: "{ Stmt_for_body8[i0, i1, i2]  : 0 <= i0 <= 1599 and
  //                                          0 <= i1 <= 1799 and
  //                                          0 <= i2 <= 2199;
  //            Stmt_for_body3[i0, i1] :      0 <= i0 <= 1599 and
  //                                          0 <= i1 <= 1799;
  //            Stmt_for_body3_last[i0, i1] : 0 <= i0 <= 1599 and
  //                                          0 <= i1 <= 1799 }"
  // child:
  //  sequence:
  //  - filter: "{ Stmt_for_body3[i0, i1] }"
  //    child:
  //      schedule: "[{ Stmt_for_body3[i0, i1] -> [(i0)] },
  //                  { Stmt_for_body3[i0, i1] -> [(i1)] }]"
  //      permutable: 1
  //      coincident: [ 1, 1 ]
  //  - filter: "{ Stmt_for_body3_last[i0, i1] }"
  //    child:
  //      schedule: "[{ Stmt_for_body3_last[i0, i1] -> [(i0)] },
  //                  { Stmt_for_body3_last[i0, i1] -> [(i1)] }]"
  //      permutable: 1
  //      coincident: [ 1, 1 ]
  //  - filter: "{ Stmt_for_body8[i0, i1, i2] }"
  //    child:
  //      schedule: "[{ Stmt_for_body8[i0, i1, i2] -> [(i0)] },
  //                  { Stmt_for_body8[i0, i1, i2] -> [(i1)] },
  //                  { Stmt_for_body8[i0, i1, i2] -> [(i2)] }]"
  //      permutable: 1
  //      coincident: [ 1, 1, 0 ]
  //
  while (NodeType != isl_schedule_node_domain) {
    if (NodeType == isl_schedule_node_filter) {
      if (!Node.parent().isa<isl::schedule_node_sequence>() ||
          !Node.parent().parent().isa<isl::schedule_node_domain>())
        return false;
      break;
    }

    if ((NodeType != isl_schedule_node_band) &&
        (NodeType != isl_schedule_node_mark))
      return false;

    Node = Node.parent();
    NodeType = isl_schedule_node_get_type(Node.get());
  }

  isl::map PartialScheduleMap = isl::map::from_union_map(PartialSchedule);
  if (containsTCInfoTy(PartialScheduleMap, D, TCI, isl::set(Domain)))
    return true;

  return false;
}

} // namespace

isl::schedule_node
polly::tryOptimizeMatMulPattern(isl::schedule_node Node,
                                const llvm::TargetTransformInfo *TTI,
                                const Dependences *D) {
  TCInfoTy TCI;
  if (PMBasedTCOpts && isTCPattern(Node, D, TCI))
    POLLY_DEBUG(dbgs() << "The tensor contraction pattern was detected\n");
  MatMulInfoTy MMI;
  if (PMBasedMMMOpts && isMatrMultPattern(Node, D, MMI)) {
    POLLY_DEBUG(dbgs() << "The matrix multiplication pattern was detected\n");
    return optimizeMatMulPattern(Node, TTI, MMI);
  }
  return {};
}
