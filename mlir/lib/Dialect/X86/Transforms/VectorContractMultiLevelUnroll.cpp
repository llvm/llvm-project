//===- VectorContractMultiLevelUnroll.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/Dialect/X86/Transforms.h"
#include "mlir/Dialect/X86/Utils/X86Utils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#define DEBUG_TYPE "x86-vector-contract-multi-level-unroll"

using namespace mlir;

using llvm::all_of;
using llvm::count;

// Discardable attribute to control vector unroll patterns.
static constexpr auto nativeShapeAttrName = "x86_vcmlu_native_shape";

namespace {
// Shared state for the pattern match and IR transformation.
struct MLUCandidate {
  // The CPU feature flag to target with the transformation, in the spelling
  // used by llvm/include/llvm/TargetParser/X86TargetParser.def, e.g.
  // "amx-bf16". Determines native shapes and supported data types for operands
  // and the accumulator.
  StringRef target;

  // Anchor operation.
  vector::ContractionOp contract;
  bool isInVnniLayout = false;

  // Shape of the original contraction.
  VectorType accType;
  VectorType lhsType;
  VectorType rhsType;

  // Shape of the register-tiled contraction. We'll insert a loop nest to cover
  // the original shape.
  VectorType accRegTileType;
  VectorType lhsRegTileType;
  VectorType rhsRegTileType;

  // Shape of the native contraction (for the X86 nanokernel patterns). We'll
  // unroll the register-tiled contraction to this shape.
  VectorType accNativeType;
  VectorType lhsNativeType;
  VectorType rhsNativeType;
  SmallVector<int64_t> contractNativeShape;

  // Bookkeeping for the accumulation loop and operands.
  scf::ForOp accLoop;
  Value accInitVal;

  bool accIsZeroInit = false;
  vector::TransferReadOp lhsRead;
  vector::TransferReadOp rhsRead;

  // Memory backing the accumulator.
  Value accMemIn;
  Value accMemOut;

  // Reconstructed accumulation loop with native-shaped iter_args.
  scf::ForOp slicedAccLoop;

  // Keep track of operations that should be deleted after transformation
  // (depending on which ops could be integrated into the new loop nest).
  SmallVector<Operation *> opsToDelete;
};
} // anonymous namespace

//----------------------------------------------------------------------------//
// Utilities                                                                  //
//----------------------------------------------------------------------------//

// Return true if the argument is an all-zero vector.
static bool isZeroVector(Value v) {
  auto constOp = v.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return false;
  auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
  if (!denseAttr || !denseAttr.isSplat())
    return false;

  return isZeroIntegerOrFloat(denseAttr.getSplatValue<Attribute>());
}

// Return true if given vector.transfer_read/write has a default permutation
// map, no mask and no out-of-bounds check.
static bool isSimpleTransferOp(VectorTransferOpInterface op) {
  return op.getPermutationMap().isMinorIdentity() && !op.getMask() &&
         !op.hasOutOfBoundsDim();
}

// Create a rank-reducing subview of a memref that matches the target vector
// type, e.g. with targetType = vector<64x32xbf16>:
//
//   memref.subview %A[%block_m, %block_k, 0, 0] [1, 1, 64, 32] [1, 1, 1, 1]
//     : memref<?x?x64x32xbf16> to
//       memref<64x32xbf16, strided<[32, 1], offset: ?>>
static Value makeRankReducingSubview(OpBuilder &b, Location loc, Value base,
                                     ValueRange indices,
                                     VectorType targetType) {
  auto *ctx = b.getContext();

  auto baseType = dyn_cast<MemRefType>(base.getType());
  assert(baseType && "base must be a MemRefType");

  int64_t baseRank = baseType.getRank();
  int64_t targetRank = targetType.getRank();
  assert(targetRank <= baseRank && "target shape's rank > memref's rank");

  ArrayRef<int64_t> targetShape = targetType.getShape();

  SmallVector<int64_t> sizesInt(baseType.getRank(), 1);
  copy(targetShape, sizesInt.begin() + baseRank - targetRank);
  SmallVector<int64_t> stridesInt(baseRank, 1);

  auto offsets = getAsOpFoldResult(indices);
  auto sizes = getAsIndexOpFoldResult(ctx, sizesInt);
  auto strides = getAsIndexOpFoldResult(ctx, stridesInt);

  MemRefType rankReducedType = memref::SubViewOp::inferRankReducedResultType(
      targetShape, baseType, offsets, sizes, strides);
  return memref::SubViewOp::create(b, loc, rankReducedType, base, offsets,
                                   sizes, strides);
}

// Slice a 2D vector into smaller tiles of the given native type.
static SmallVector<Value> slice(OpBuilder &b, Location loc, Value value,
                                VectorType nativeType) {
  VectorType regTileType = cast<VectorType>(value.getType());
  ArrayRef<int64_t> regTileShape = regTileType.getShape();
  ArrayRef<int64_t> nativeShape = nativeType.getShape();

  assert(regTileShape.size() == 2 && nativeShape.size() == 2 &&
         "Only 2D shapes are supported");

  int64_t regTileM = regTileShape[0];
  int64_t regTileN = regTileShape[1];
  int64_t nativeM = nativeShape[0];
  int64_t nativeN = nativeShape[1];

  SmallVector<Value> result;
  result.reserve((regTileM / nativeM) * (regTileN / nativeN));

  for (int64_t m = 0; m < regTileM; m += nativeM)
    for (int64_t n = 0; n < regTileN; n += nativeN)
      result.push_back(vector::ExtractStridedSliceOp::create(
          b, loc, value, ArrayRef<int64_t>{m, n}, nativeShape,
          ArrayRef<int64_t>{1, 1}));

  return result;
}

// Splice smaller tiles back into a 2D vector of the given register tile type.
static Value splice(OpBuilder &b, Location loc, ValueRange values,
                    VectorType regTileType) {
  auto types = values.getTypes();
  assert(!types.empty() &&
         count(types, types.front()) == static_cast<int64_t>(types.size()) &&
         "All values must have the same type");

  VectorType nativeType = cast<VectorType>(types.front());

  ArrayRef<int64_t> nativeShape = nativeType.getShape();
  ArrayRef<int64_t> regTileShape = regTileType.getShape();

  assert(nativeShape.size() == 2 && regTileShape.size() == 2 &&
         "Only 2D shapes are supported");

  int64_t nativeM = nativeShape[0];
  int64_t nativeN = nativeShape[1];
  int64_t regTileM = regTileShape[0];
  int64_t regTileN = regTileShape[1];

  assert((regTileM / nativeM) * (regTileN / nativeN) ==
             static_cast<int64_t>(values.size()) &&
         "The number of values must match the number of slices");

  Value result = ub::PoisonOp::create(b, loc, regTileType);

  auto valueIt = values.begin();
  for (int64_t m = 0; m < regTileM; m += nativeM)
    for (int64_t n = 0; n < regTileN; n += nativeN) {
      result = vector::InsertStridedSliceOp::create(b, loc, *valueIt, result,
                                                    ArrayRef<int64_t>{m, n},
                                                    ArrayRef<int64_t>{1, 1});
      ++valueIt;
    }

  return result;
}

static LogicalResult hasNativeShapeAttr(Operation *op) {
  return success(op->hasAttr(nativeShapeAttrName));
}

static std::optional<SmallVector<int64_t>>
getFromNativeShapeAttr(Operation *op) {
  auto attr = op->getAttrOfType<DenseI64ArrayAttr>(nativeShapeAttrName);
  if (!attr)
    return std::nullopt;
  return SmallVector<int64_t>(attr.asArrayRef());
}

static void setNativeShapeAttr(VectorTransferOpInterface op,
                               VectorType nativeType) {
  if (op.getVectorType() != nativeType)
    op->setAttr(
        nativeShapeAttrName,
        DenseI64ArrayAttr::get(nativeType.getContext(), nativeType.getShape()));
}

//----------------------------------------------------------------------------//
// Pattern match and target shape computation                                 //
//----------------------------------------------------------------------------//

// Establish whether the candidate contraction is a canonical matrix
// multiplication on flat or VNNI-packed vectors. If successful, the discovered
// operand and accumulator types are recorded in the candidate struct.
//
// Example (flat):
//   vector.contract {
//     kind = #vector.kind<add>, indexing_maps = [
//       affine_map<(dm, dn, dk) -> (dm, dk)>,
//       affine_map<(dm, dn, dk) -> (dk, dn)>,
//       affine_map<(dm, dn, dk) -> (dm, dn)>],
//     iterator_types = ["parallel", "parallel", "reduction"],
//   } %lhs, %rhs, %acc : !lhsType, !rhsType into !accType
//
// Example (VNNI-packed):
//   vector.contract {
//     kind = #vector.kind<add>, indexing_maps = [
//       affine_map<(dm, dn, dk, dvnni) -> (dm, dk, dvnni)>,
//       affine_map<(dm, dn, dk, dvnni) -> (dk, dn, dvnni)>,
//       affine_map<(dm, dn, dk, dvnni) -> (dm, dn)>],
//     iterator_types = ["parallel", "parallel", "reduction", "reduction"],
//   } %lhs, %rhs, %acc : !lhsType, !rhsType into !accType
static LogicalResult matchCanonicalMatmul(MLUCandidate &candidate,
                                          PatternRewriter &rewriter) {
  vector::ContractionOp contract = candidate.contract;
  if (contract.getKind() != vector::CombiningKind::ADD)
    return rewriter.notifyMatchFailure(contract,
                                       "not using ADD combining kind");

  VectorType accType = dyn_cast<VectorType>(contract.getAccType());
  if (!accType || accType.getRank() != 2)
    return rewriter.notifyMatchFailure(contract,
                                       "accumulator is not a 2D vector");

  VectorType lhsType = contract.getLhsType();
  VectorType rhsType = contract.getRhsType();
  if (lhsType.getElementType() != rhsType.getElementType())
    return rewriter.notifyMatchFailure(
        contract, "LHS and RHS element types do not match");

  auto iteratorTypes = contract.getIteratorTypesArray();
  auto indexingMaps = contract.getIndexingMapsArray();
  assert(indexingMaps.size() == 3 && "expected 3 indexing maps");
  bool isInVnniLayout = x86::isInVnniLayout(contract, indexingMaps);

  auto par = vector::IteratorType::parallel;
  auto red = vector::IteratorType::reduction;
  auto m = rewriter.getAffineDimExpr(0);
  auto n = rewriter.getAffineDimExpr(1);
  auto k = rewriter.getAffineDimExpr(2);
  auto vnni = rewriter.getAffineDimExpr(3);

  if (!isInVnniLayout) {
    if (!equal(indexingMaps[0].getResults(), ArrayRef{m, k}) ||
        !equal(indexingMaps[1].getResults(), ArrayRef{k, n}) ||
        !equal(indexingMaps[2].getResults(), ArrayRef{m, n}))
      return rewriter.notifyMatchFailure(contract,
                                         "indexing maps are not canonical");

    if (!equal(iteratorTypes, ArrayRef{par, par, red}))
      return rewriter.notifyMatchFailure(contract,
                                         "iterator types are not canonical");
  } else {
    if (!equal(indexingMaps[0].getResults(), ArrayRef{m, k, vnni}) ||
        !equal(indexingMaps[1].getResults(), ArrayRef{k, n, vnni}) ||
        !equal(indexingMaps[2].getResults(), ArrayRef{m, n}))
      return rewriter.notifyMatchFailure(
          contract, "indexing maps are not canonical (VNNI)");

    if (!equal(iteratorTypes, ArrayRef{par, par, red, red}))
      return rewriter.notifyMatchFailure(
          contract, "iterator types are not canonical (VNNI)");
  }

  candidate.isInVnniLayout = isInVnniLayout;
  candidate.accType = accType;
  candidate.lhsType = lhsType;
  candidate.rhsType = rhsType;

  return success();
}

// Check compatibility of datatypes with the given target ISA extension, and try
// to determine a suitable tiling strategy for it. If successful, the strategy
// is encoded in the *RegTileType and *NativeType fields of the candidate
// struct.
static LogicalResult matchShapesAndTypes(MLUCandidate &candidate,
                                         PatternRewriter &rewriter) {
  vector::ContractionOp contract = candidate.contract;
  StringRef target = candidate.target;

  Type accElemType = candidate.accType.getElementType();
  Type inpElemType = candidate.lhsType.getElementType();

  ArrayRef<int64_t> accShape = candidate.accType.getShape();
  ArrayRef<int64_t> lhsShape = candidate.lhsType.getShape();

  int64_t origM = accShape[0];
  int64_t origN = accShape[1];
  int64_t origK = lhsShape[1];
  int64_t origVnni = candidate.isInVnniLayout ? lhsShape[2] : -1;

  int64_t regTileM, regTileN, regTileK;
  int64_t nativeM, nativeN, nativeK, vnni;

  if (target.starts_with("amx")) {
    if (!((target == "amx-bf16" && inpElemType.isBF16() &&
           accElemType.isF32()) ||
          (target == "amx-int8" && inpElemType.isInteger(8) &&
           accElemType.isInteger(32))))
      return rewriter.notifyMatchFailure(
          contract, "unsupported combination of input and accumulator types");

    nativeM = nativeN = 16;
    vnni = 32 / inpElemType.getIntOrFloatBitWidth();

    if (candidate.isInVnniLayout) {
      nativeK = 16;
      if (origM % nativeM != 0 || origN % nativeN != 0)
        return rewriter.notifyMatchFailure(
            contract, "vector shape cannot be cleanly unrolled");
      if (origK != nativeK || origVnni != vnni)
        return rewriter.notifyMatchFailure(
            contract, "K dimension or VNNI factor mismatch");
      if (origM == nativeM && origN == nativeN)
        return rewriter.notifyMatchFailure(contract,
                                           "already in native tile size");
      // Prefer 2x2 tiling if shapes allow.
      regTileM = origM % (2 * nativeM) == 0 ? 2 * nativeM : nativeM;
      regTileN = origN % (2 * nativeN) == 0 ? 2 * nativeN : nativeN;
      regTileK = nativeK;
    } else {
      // Online packing requires a 2x2 tile register layout.
      nativeK = 16 * vnni;
      regTileM = 2 * nativeM;
      regTileN = 2 * nativeN;
      regTileK = nativeK;
      if (origM % regTileM != 0 || origN % regTileN != 0)
        return rewriter.notifyMatchFailure(
            contract, "vector shape cannot be cleanly unrolled");
      if (origK != nativeK)
        return rewriter.notifyMatchFailure(contract, "K dimension mismatch");
    }
  } else {
    return rewriter.notifyMatchFailure(contract, "unsupported target");
  }

  candidate.accRegTileType = VectorType::get({regTileM, regTileN}, accElemType);
  candidate.accNativeType = VectorType::get({nativeM, nativeN}, accElemType);

  auto getOperandType = [&](ArrayRef<int64_t> shape) {
    if (candidate.isInVnniLayout) {
      SmallVector<int64_t, 3> newShape(shape.begin(), shape.end());
      newShape.push_back(vnni);
      return VectorType::get(newShape, inpElemType);
    }
    return VectorType::get(shape, inpElemType);
  };

  candidate.lhsRegTileType = getOperandType({regTileM, regTileK});
  candidate.rhsRegTileType = getOperandType({regTileK, regTileN});

  candidate.lhsNativeType = getOperandType({nativeM, nativeK});
  candidate.rhsNativeType = getOperandType({nativeK, nativeN});

  candidate.contractNativeShape = {nativeM, nativeN, nativeK};
  if (candidate.isInVnniLayout)
    candidate.contractNativeShape.push_back(vnni);

  return success();
}

// Check constraints on the accumulation loop. We're looking for an index-based
// for-loop with a step of 1 or K, a single accumulator block argument. The body
// shall only contain the transfer_reads for the operands, the contraction and
// the yield operation. This restriction is in place because the pattern
// rebuilds the accumulation loop from scratch with the known operations.
//
// For the operand reads, we require that they have a minor identity permutation
// map, impose no bounds checks and don't carry a mask. In addition, they shall
// use the induction variable as exactly one of their indices. This covers plain
// 2D memrefs as well as block-based layouts.
//
// If all these conditions are met, the loop op, initial value of the
// accumulator and the operand reads are recorded in the candidate struct.
//
// Example:
//   !accType = vector<64x128xf32>
//   !lhsType = vector<64x32xbf16>
//   !rhsType = vector<32x128xbf16>
//   ...
//   %res = scf.for %k = %k_start to %k_end step %c1 iter_args(%acc = %c)
//            -> (!accType) {
//     %a = vector.transfer_read %A[%m, %k, %c0, %c0], %c0_0
//            {in_bounds = [true, true]} : memref<?x?x64x32xbf16>, !lhsType
//     %b = vector.transfer_read %B[%n, %k, %c0, %c0], %c0_0
//            {in_bounds = [true, true]} : memref<?x?x32x128xbf16>, !rhsType
//     %d = vector.contract { ... } %a, %b, %acc
//            : !lhsType, !rhsType into !accType
//     scf.yield %d : !accType
//   }
static LogicalResult matchAccumulationLoop(MLUCandidate &candidate,
                                           PatternRewriter &rewriter) {
  auto &contract = candidate.contract;

  auto accLoop = dyn_cast_if_present<scf::ForOp>(contract->getParentOp());
  if (!accLoop || !accLoop.getInductionVar().getType().isIndex())
    return rewriter.notifyMatchFailure(contract,
                                       "is not in an index-based for-loop");

  if (auto step = getConstantIntValue(accLoop.getStep());
      !step || (*step != 1 && *step != candidate.lhsType.getShape()[1]))
    return rewriter.notifyMatchFailure(contract,
                                       "accumulation loop step is not 1 or K");

  BlockArgument accIterArg = dyn_cast<BlockArgument>(contract.getAcc());
  if (!accIterArg || accIterArg.getOwner() != accLoop.getBody())
    return rewriter.notifyMatchFailure(contract,
                                       "accumulator is not a block argument");

  if (accLoop.getNumRegionIterArgs() != 1 ||
      accLoop.getBody()->getOperations().size() != 4)
    return rewriter.notifyMatchFailure(
        contract, "accumulation loop has additional iter_args or operations");

  if (!contract->getResult(0).hasOneUse())
    return rewriter.notifyMatchFailure(contract,
                                       "result does not have exactly one use");

  Operation *user = *contract->getResult(0).getUsers().begin();
  if (!isa<scf::YieldOp>(user))
    return rewriter.notifyMatchFailure(contract,
                                       "accumulator is not loop-carried");

  Value accInitVal = accLoop.getTiedLoopInit(accIterArg)->get();
  bool accIsZeroInit = isZeroVector(accInitVal);

  auto lhsRead = contract.getLhs().getDefiningOp<vector::TransferReadOp>();
  auto rhsRead = contract.getRhs().getDefiningOp<vector::TransferReadOp>();
  if (!lhsRead || !rhsRead)
    return rewriter.notifyMatchFailure(
        contract, "LHS/RHS are not vector.transfer_read ops");

  auto checkTransferOp = [&](vector::TransferReadOp transferOp) {
    if (!isSimpleTransferOp(transferOp))
      return rewriter.notifyMatchFailure(contract, [&](Diagnostic &diag) {
        diag << "transfer op has non-identity permutation, mask, or "
                "out-of-bounds check: "
             << *transferOp;
      });

    if (count(transferOp.getIndices(), accLoop.getInductionVar()) != 1)
      return rewriter.notifyMatchFailure(
          contract, "transfer op does not use accumulator index");

    return success();
  };

  if (failed(checkTransferOp(lhsRead)) || failed(checkTransferOp(rhsRead)))
    return failure();

  candidate.accLoop = accLoop;
  candidate.accInitVal = accInitVal;
  candidate.accIsZeroInit = accIsZeroInit;
  candidate.lhsRead = lhsRead;
  candidate.rhsRead = rhsRead;

  return success();
}

//----------------------------------------------------------------------------//
// Transformation
//----------------------------------------------------------------------------//

// Ensure that the initial accumulator value can be sliced into register-tile
// shaped vectors. Unless it is zero-initialized, we need it to be backed
// by memory so that the slice can be loaded via a vector.transfer_read inside
// the loop nest.
//
// If the initial value is produced by a simple vector.transfer_read, we can
// simply derive a subview from its base and indices. Otherwise, we fall back to
// allocating a new buffer on stack and store the initial value into it.
static void makeAccInitializationBackedByMemory(MLUCandidate &candidate,
                                                PatternRewriter &rewriter) {
  if (candidate.accIsZeroInit)
    return;

  rewriter.setInsertionPoint(candidate.accLoop);
  auto loc = candidate.accLoop.getLoc();

  if (auto accRead =
          candidate.accInitVal.getDefiningOp<vector::TransferReadOp>();
      accRead && isSimpleTransferOp(accRead)) {
    candidate.accMemIn =
        makeRankReducingSubview(rewriter, loc, accRead.getBase(),
                                accRead.getIndices(), accRead.getVectorType());
    return;
  }

  auto accBufferType = MemRefType::get(candidate.accType.getShape(),
                                       candidate.accType.getElementType());
  candidate.accMemIn = memref::AllocaOp::create(rewriter, loc, accBufferType);

  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  SmallVector<Value, 2> zeroIndices(candidate.accType.getRank(), c0);
  vector::TransferWriteOp::create(rewriter, loc, candidate.accInitVal,
                                  candidate.accMemIn, zeroIndices);
}

// Ensure that slices of the result of the accumulator loop can be stored, so
// again we need it to be backed by memory.
//
// If the only user of the result is a simple vector.transfer_write, and its
// base and indices are defined before the accumulator loop, we can derive a
// subview. The transfer_write will be deleted after building the loop nest,
// which at that point incorporates its effect.
//
// Otherwise, we fall back to buffering the result and rematerializing it after
// the loop. If we already had a stack allocation for the initial value, we
// reuse it.
static void makeAccResultBackedByMemory(MLUCandidate &candidate,
                                        PatternRewriter &rewriter) {
  rewriter.setInsertionPoint(candidate.accLoop);
  auto loc = candidate.accLoop.getLoc();

  auto accRes = candidate.accLoop.getResult(0);
  vector::TransferWriteOp accWrite;
  if (accRes.hasOneUse())
    accWrite = dyn_cast<vector::TransferWriteOp>(*accRes.getUsers().begin());

  Block *accLoopBlock = candidate.accLoop->getBlock();
  if (accWrite && isSimpleTransferOp(accWrite) &&
      accWrite->getBlock() == accLoopBlock) {
    auto isDefinedBeforeAccLoop = [&](Value v) {
      Operation *defOp = v.getDefiningOp();
      // No need for full dominance check, as we already know that accLoop and
      // accWrite are in the same block.
      return !defOp || defOp->getBlock() != accLoopBlock ||
             defOp->isBeforeInBlock(candidate.accLoop);
    };
    if (isDefinedBeforeAccLoop(accWrite.getBase()) &&
        all_of(accWrite.getIndices(), isDefinedBeforeAccLoop)) {
      candidate.accMemOut = makeRankReducingSubview(
          rewriter, loc, accWrite.getBase(), accWrite.getIndices(),
          accWrite.getVectorType());
      candidate.opsToDelete.push_back(accWrite);
      candidate.opsToDelete.push_back(candidate.accLoop);
      return;
    }
  }

  if (!candidate.accIsZeroInit &&
      isa<memref::AllocaOp>(candidate.accMemIn.getDefiningOp())) {
    candidate.accMemOut = candidate.accMemIn;
  } else {
    auto accBufferType = MemRefType::get(candidate.accType.getShape(),
                                         candidate.accType.getElementType());
    candidate.accMemOut =
        memref::AllocaOp::create(rewriter, loc, accBufferType);
  }

  rewriter.setInsertionPointAfter(candidate.accLoop);
  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  SmallVector<Value, 2> zeroIndices(candidate.accType.getRank(), c0);
  auto accRemat = vector::TransferReadOp::create(
      rewriter, loc, candidate.accType, candidate.accMemOut, zeroIndices,
      /*padding=*/std::nullopt);

  rewriter.replaceAllUsesWith(candidate.accLoop.getResult(0), accRemat);
  candidate.opsToDelete.push_back(candidate.accLoop);
}

static void buildMNLoopBody(MLUCandidate &candidate, OpBuilder &b, Location loc,
                            Value ivM, Value ivN);
static SmallVector<Value> buildKLoopBody(MLUCandidate &candidate, OpBuilder &b,
                                         Location loc, Value ivM, Value ivN,
                                         Value ivK, ValueRange iterArgs);

// Create the loop nest for iterating over the M and N dimensions of the
// accumulator. Construction continues in buildMNLoopBody.
static void buildLoopNestForCandidate(MLUCandidate &candidate,
                                      PatternRewriter &rewriter) {
  rewriter.setInsertionPoint(candidate.accLoop);
  auto loc = candidate.accLoop.getLoc();

  ArrayRef<int64_t> accShape = candidate.accType.getShape();
  ArrayRef<int64_t> accRegTileShape = candidate.accRegTileType.getShape();

  auto makeIndexConst = [&](int64_t v) {
    return arith::ConstantIndexOp::create(rewriter, loc, v);
  };

  Value c0 = makeIndexConst(0);
  Value ubM = makeIndexConst(accShape[0]);
  Value stepM = makeIndexConst(accRegTileShape[0]);
  Value ubN = makeIndexConst(accShape[1]);
  Value stepN = makeIndexConst(accRegTileShape[1]);

  scf::ForOp::create(
      rewriter, loc, c0, ubM, stepM, {},
      [&](OpBuilder &b, Location loc, Value ivM, ValueRange iterArgs) {
        scf::ForOp::create(
            b, loc, c0, ubN, stepN, {},
            [&](OpBuilder &b, Location loc, Value ivN, ValueRange iterArgs) {
              buildMNLoopBody(candidate, b, loc, ivM, ivN);
              scf::YieldOp::create(b, loc);
            });
        scf::YieldOp::create(b, loc);
      });
}

// Create the body for M-N-loop nest according to the following scheme:
//
// 1. Load a accRegTileType-shaped slice of the accumulator from memory (or make
//    a zero constant).
// 2. Split the initial value into accNativeType-shaped slices.
// 3. Create the K-loop with the initial value slices as loop-carried arguments,
//    i.e. the iter_args have accNativeType.
// 4. -> buildKLoopBody
// 5. Splice the final value slices back into an accRegTileType-shaped value and
//    write it back to memory.
//
// The accRegTileType-shaped operations are annotated with an attribute in order
// to unroll them with the vector-dialect's unroll patterns.
static void buildMNLoopBody(MLUCandidate &candidate, OpBuilder &b, Location loc,
                            Value ivM, Value ivN) {
  ArrayRef<int64_t> accRegTileShape = candidate.accRegTileType.getShape();

  Value c0 = arith::ConstantIndexOp::create(b, loc, 0);

  SmallVector<OpFoldResult, 2> accViewOffsets = {ivM, ivN};
  SmallVector<OpFoldResult, 2> accViewSizes = {
      b.getI64IntegerAttr(accRegTileShape[0]),
      b.getI64IntegerAttr(accRegTileShape[1])};
  SmallVector<OpFoldResult, 2> accViewStrides = {b.getI64IntegerAttr(1),
                                                 b.getI64IntegerAttr(1)};

  Operation *accInit;
  if (candidate.accIsZeroInit) {
    accInit = arith::ConstantOp::create(
        b, loc, b.getZeroAttr(candidate.accRegTileType));
  } else {
    Value accViewIn =
        memref::SubViewOp::create(b, loc, candidate.accMemIn, accViewOffsets,
                                  accViewSizes, accViewStrides);
    auto accInitRead = vector::TransferReadOp::create(
        b, loc, candidate.accRegTileType, accViewIn, ValueRange{c0, c0},
        /*padding=*/std::nullopt);
    setNativeShapeAttr(accInitRead, candidate.accNativeType);
    accInit = accInitRead;
  }

  SmallVector<Value> accInitSliced =
      slice(b, loc, accInit->getResult(0), candidate.accNativeType);
  candidate.slicedAccLoop = scf::ForOp::create(
      b, loc, candidate.accLoop.getLowerBound(),
      candidate.accLoop.getUpperBound(), candidate.accLoop.getStep(),
      accInitSliced,
      [&](OpBuilder &b, Location loc, Value ivK, ValueRange iterArgs) {
        SmallVector<Value> newIterArgs =
            buildKLoopBody(candidate, b, loc, ivM, ivN, ivK, iterArgs);
        scf::YieldOp::create(b, loc, newIterArgs);
      });

  Value accSpliced = splice(b, loc, candidate.slicedAccLoop.getResults(),
                            candidate.accRegTileType);
  Value accViewOut =
      memref::SubViewOp::create(b, loc, candidate.accMemOut, accViewOffsets,
                                accViewSizes, accViewStrides);
  auto accWrite = vector::TransferWriteOp::create(
      b, loc, accSpliced, accViewOut, ValueRange{c0, c0});

  setNativeShapeAttr(accWrite, candidate.accNativeType);
}

// Rebuild the accumulation loop with [lhs|rhs]RegTileType-shaped operations,
// taking the offsets from the M-N-loop nests into account. The accumulator
// iter_args are spliced into a accRegTileType-shaped value, and the contraction
// result is split into accNativeType-shaped yield operands. Again, the
// operations are annotated with an attribute for unrolling, after which the
// splice/split logic will fold away completely.
static SmallVector<Value> buildKLoopBody(MLUCandidate &candidate, OpBuilder &b,
                                         Location loc, Value ivM, Value ivN,
                                         Value ivK, ValueRange iterArgs) {
  Value accSpliced = splice(b, loc, iterArgs, candidate.accRegTileType);

  SmallVector<Value, 4> lhsOffsets = candidate.lhsRead.getIndices();
  SmallVector<Value, 4> rhsOffsets = candidate.rhsRead.getIndices();
  Value accLoopIV = candidate.accLoop.getInductionVar();
  replace(lhsOffsets, accLoopIV, ivK);
  replace(rhsOffsets, accLoopIV, ivK);
  unsigned mPos = lhsOffsets.size() - 2 - candidate.isInVnniLayout;
  unsigned nPos = rhsOffsets.size() - 1 - candidate.isInVnniLayout;
  lhsOffsets[mPos] = arith::AddIOp::create(b, loc, lhsOffsets[mPos], ivM);
  rhsOffsets[nPos] = arith::AddIOp::create(b, loc, rhsOffsets[nPos], ivN);
  Value lhsView = makeRankReducingSubview(b, loc, candidate.lhsRead.getBase(),
                                          lhsOffsets, candidate.lhsRegTileType);
  Value rhsView = makeRankReducingSubview(b, loc, candidate.rhsRead.getBase(),
                                          rhsOffsets, candidate.rhsRegTileType);

  SmallVector<Value, 3> zeroIndices(2 + candidate.isInVnniLayout,
                                    arith::ConstantIndexOp::create(b, loc, 0));
  auto lhsRead = vector::TransferReadOp::create(
      b, candidate.lhsRead.getLoc(), candidate.lhsRegTileType, lhsView,
      zeroIndices, /*padding=*/std::nullopt);
  auto rhsRead = vector::TransferReadOp::create(
      b, candidate.rhsRead.getLoc(), candidate.rhsRegTileType, rhsView,
      zeroIndices, /*padding=*/std::nullopt);
  auto contract = vector::ContractionOp::create(
      b, candidate.contract.getLoc(), lhsRead, rhsRead, accSpliced,
      candidate.contract.getIndexingMaps(),
      candidate.contract.getIteratorTypes());

  setNativeShapeAttr(lhsRead, candidate.lhsNativeType);
  setNativeShapeAttr(rhsRead, candidate.rhsNativeType);
  contract->setAttr(nativeShapeAttrName,
                    b.getDenseI64ArrayAttr(candidate.contractNativeShape));

  SmallVector<Value> accSliced =
      slice(b, loc, contract, candidate.accNativeType);
  return accSliced;
}

//----------------------------------------------------------------------------//
// Pattern implementation                                                     //
//----------------------------------------------------------------------------//

namespace {
// Prepares a contraction inside an accumulation loop for the nanokernel
// lowerings.
//
// The pattern determines a register tile size for the contraction that can be
// fully unrolled to a target-specific native shape without exceeding the
// target's vector registers or AMX tile registers. It then introduces a
// M-N-loop nest to iterate over such register tiles, and applies the vector
// dialect's unroll patterns to produce native-shaped operations that the
// nanokernel patterns can match.
//
// Currently, the match is deliberately strict (see the helpers above):
// The contraction must be a canonical matrix multiplication, have
// target-compatible types and shapes that can be tiled and unrolled cleanly,
// and be embedded in an accumulation loop containing only the operand reads and
// itself. This lets the rewrite regenerate the loop from scratch instead of
// transforming it in place.
struct VectorContractMultiLevelUnroll
    : public OpRewritePattern<vector::ContractionOp> {
  VectorContractMultiLevelUnroll(MLIRContext *context, StringRef target,
                                 PatternBenefit benefit = 1)
      : OpRewritePattern<vector::ContractionOp>(context, benefit),
        target(target.str()) {}

  LogicalResult matchAndRewrite(vector::ContractionOp contract,
                                PatternRewriter &rewriter) const override {
    // Bail out early if this op is the result of a previous application of the
    // pattern.
    if (contract->hasAttr(nativeShapeAttrName))
      return failure();

    MLUCandidate candidate;
    candidate.target = target;
    candidate.contract = contract;
    if (failed(matchCanonicalMatmul(candidate, rewriter)) ||
        failed(matchShapesAndTypes(candidate, rewriter)) ||
        failed(matchAccumulationLoop(candidate, rewriter)))
      return failure();

    makeAccInitializationBackedByMemory(candidate, rewriter);
    makeAccResultBackedByMemory(candidate, rewriter);
    buildLoopNestForCandidate(candidate, rewriter);
    moveLoopInvariantCode(candidate.slicedAccLoop);
    for (auto *op : candidate.opsToDelete)
      rewriter.eraseOp(op);

    return success();
  }

private:
  std::string target;
};
} // namespace

void x86::populateVectorContractMultiLevelUnrollPatterns(
    RewritePatternSet &patterns, StringRef target) {
  patterns.add<VectorContractMultiLevelUnroll>(patterns.getContext(), target);
  vector::UnrollVectorOptions options;
  options.setFilterConstraint(hasNativeShapeAttr);
  options.setNativeShapeFn(getFromNativeShapeAttr);
  vector::populateVectorUnrollPatterns(patterns, options);
}
