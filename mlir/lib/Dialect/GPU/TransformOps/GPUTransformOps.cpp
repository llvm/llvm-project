//===- GPUTransformOps.cpp - Implementation of GPU transform ops ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/Utils.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InterleavedRange.h"
#include "llvm/Support/LogicalResult.h"
#include <type_traits>

using namespace mlir;
using namespace mlir::gpu;
using namespace mlir::transform;
using namespace mlir::transform::gpu;

#define DEBUG_TYPE "gpu-transforms"
#define DEBUG_TYPE_ALIAS "gpu-transforms-alias"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define DBGS_ALIAS() (llvm::dbgs() << '[' << DEBUG_TYPE_ALIAS << "] ")

//===----------------------------------------------------------------------===//
// Apply...ConversionPatternsOp
//===----------------------------------------------------------------------===//

void transform::ApplyGPUToNVVMConversionPatternsOp::populatePatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  auto &llvmTypeConverter = static_cast<LLVMTypeConverter &>(typeConverter);
  // NVVM uses alloca in the default address space to represent private
  // memory allocations, so drop private annotations. NVVM uses address
  // space 3 for shared memory. NVVM uses the default address space to
  // represent global memory.
  // Used in populateGpuToNVVMConversionPatternsso attaching here for now.
  // TODO: We should have a single to_nvvm_type_converter.
  populateGpuMemorySpaceAttributeConversions(
      llvmTypeConverter, [](AddressSpace space) -> unsigned {
        switch (space) {
        case AddressSpace::Global:
          return static_cast<unsigned>(
              NVVM::NVVMMemorySpace::kGlobalMemorySpace);
        case AddressSpace::Workgroup:
          return static_cast<unsigned>(
              NVVM::NVVMMemorySpace::kSharedMemorySpace);
        case AddressSpace::Private:
          return 0;
        }
        llvm_unreachable("unknown address space enum value");
        return 0;
      });
  // Used in GPUToNVVM/WmmaOpsToNvvm.cpp so attaching here for now.
  // TODO: We should have a single to_nvvm_type_converter.
  llvmTypeConverter.addConversion(
      [&](MMAMatrixType type) -> Type { return convertMMAToLLVMType(type); });
  // Set higher benefit, so patterns will run before generic LLVM lowering.
  populateGpuToNVVMConversionPatterns(llvmTypeConverter, patterns,
                                      getBenefit());
}

LogicalResult
transform::ApplyGPUToNVVMConversionPatternsOp::verifyTypeConverter(
    transform::TypeConverterBuilderOpInterface builder) {
  if (builder.getTypeConverterType() != "LLVMTypeConverter")
    return emitOpError("expected LLVMTypeConverter");
  return success();
}

void transform::ApplyGPUWwmaToNVVMConversionPatternsOp::populatePatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  auto &llvmTypeConverter = static_cast<LLVMTypeConverter &>(typeConverter);
  populateGpuWMMAToNVVMConversionPatterns(llvmTypeConverter, patterns);
}

LogicalResult
transform::ApplyGPUWwmaToNVVMConversionPatternsOp::verifyTypeConverter(
    transform::TypeConverterBuilderOpInterface builder) {
  if (builder.getTypeConverterType() != "LLVMTypeConverter")
    return emitOpError("expected LLVMTypeConverter");
  return success();
}

void transform::ApplyGPUSubgroupReduceToNVVMConversionPatternsOp::
    populatePatterns(TypeConverter &typeConverter,
                     RewritePatternSet &patterns) {
  auto &llvmTypeConverter = static_cast<LLVMTypeConverter &>(typeConverter);
  populateGpuSubgroupReduceOpLoweringPattern(llvmTypeConverter, patterns);
}

LogicalResult transform::ApplyGPUSubgroupReduceToNVVMConversionPatternsOp::
    verifyTypeConverter(transform::TypeConverterBuilderOpInterface builder) {
  if (builder.getTypeConverterType() != "LLVMTypeConverter")
    return emitOpError("expected LLVMTypeConverter");
  return success();
}

void transform::ApplyGPUToROCDLConversionPatternsOp::populatePatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  auto &llvmTypeConverter = static_cast<LLVMTypeConverter &>(typeConverter);
  populateGpuMemorySpaceAttributeConversions(
      llvmTypeConverter, [](AddressSpace space) {
        switch (space) {
        case AddressSpace::Global:
          return ROCDL::ROCDLDialect::kGlobalMemoryAddressSpace;
        case AddressSpace::Workgroup:
          return ROCDL::ROCDLDialect::kSharedMemoryAddressSpace;
        case AddressSpace::Private:
          return ROCDL::ROCDLDialect::kPrivateMemoryAddressSpace;
        }
        llvm_unreachable("unknown address space enum value");
      });
  FailureOr<amdgpu::Chipset> maybeChipset =
      amdgpu::Chipset::parse(getChipset());
  assert(llvm::succeeded(maybeChipset) && "expected valid chipset");
  populateGpuToROCDLConversionPatterns(
      llvmTypeConverter, patterns, mlir::gpu::amd::Runtime::HIP, *maybeChipset);
}

LogicalResult
transform::ApplyGPUToROCDLConversionPatternsOp::verifyTypeConverter(
    transform::TypeConverterBuilderOpInterface builder) {
  FailureOr<amdgpu::Chipset> maybeChipset =
      amdgpu::Chipset::parse(getChipset());
  if (failed(maybeChipset)) {
    return emitOpError("Invalid chipset name: " + getChipset());
  }
  if (builder.getTypeConverterType() != "LLVMTypeConverter")
    return emitOpError("expected LLVMTypeConverter");
  return success();
}

//===----------------------------------------------------------------------===//
// Apply...PatternsOp
//===----------------------------------------------------------------------===//s

void ApplyGPURewritePatternsOp::populatePatterns(RewritePatternSet &patterns) {
  populateGpuRewritePatterns(patterns);
}

void transform::ApplyGPUPromoteShuffleToAMDGPUPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  populateGpuPromoteShuffleToAMDGPUPatterns(patterns);
}

//===----------------------------------------------------------------------===//
// ApplyUnrollVectorsSubgroupMmaOp
//===----------------------------------------------------------------------===//

/// Pick an unrolling order that will allow tensorcore operation to reuse LHS
/// register.
static std::optional<SmallVector<int64_t>>
gpuMmaUnrollOrder(vector::ContractionOp contract) {
  SmallVector<int64_t> order;
  // First make reduction the outer dimensions.
  for (auto [index, iter] : llvm::enumerate(contract.getIteratorTypes())) {
    if (vector::isReductionIterator(iter)) {
      order.push_back(index);
    }
  }

  llvm::SmallDenseSet<int64_t> dims;
  for (AffineExpr expr : contract.getIndexingMapsArray()[0].getResults()) {
    dims.insert(cast<AffineDimExpr>(expr).getPosition());
  }
  // Then parallel dimensions that are part of Lhs as we want to re-use Lhs.
  for (auto [index, iter] : llvm::enumerate(contract.getIteratorTypes())) {
    if (vector::isParallelIterator(iter) && dims.count(index)) {
      order.push_back(index);
    }
  }
  // Then the remaining parallel loops.
  for (auto [index, iter] : llvm::enumerate(contract.getIteratorTypes())) {
    if (vector::isParallelIterator(iter) && !dims.count(index)) {
      order.push_back(index);
    }
  }
  return order;
}

/// Returns the target vector size for the target operation based on the native
/// vector size specified with `m`, `n`, and `k`.
static std::optional<SmallVector<int64_t>>
getSubgroupMmaNativeVectorSize(Operation *op, int64_t m, int64_t n, int64_t k) {
  if (auto contract = dyn_cast<vector::ContractionOp>(op)) {
    int64_t contractRank = contract.getIteratorTypes().size();
    if (contractRank < 3)
      return std::nullopt;
    SmallVector<int64_t> nativeSize(contractRank - 3, 1);
    nativeSize.append({m, n, k});
    return nativeSize;
  }
  if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
    int64_t writeRank = writeOp.getVectorType().getRank();
    if (writeRank < 2)
      return std::nullopt;
    SmallVector<int64_t> nativeSize(writeRank - 2, 1);
    nativeSize.append({m, n});
    return nativeSize;
  }
  if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
    // Transfer read ops may need different shapes based on how they are being
    // used. For simplicity just match the shape used by the extract strided op.
    VectorType sliceType;
    for (Operation *users : op->getUsers()) {
      auto extract = dyn_cast<vector::ExtractStridedSliceOp>(users);
      if (!extract)
        return std::nullopt;
      auto vecType = cast<VectorType>(extract.getResult().getType());
      if (sliceType && sliceType != vecType)
        return std::nullopt;
      sliceType = vecType;
    }
    return llvm::to_vector(sliceType.getShape());
  }
  if ((OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1)) {
    if (auto vecType = dyn_cast<VectorType>(op->getResultTypes()[0])) {
      // TODO: The condition for unrolling elementwise should be restricted
      // only to operations that need unrolling (connected to the contract).
      if (vecType.getRank() < 2)
        return std::nullopt;

      // First check whether there is a slice to infer the shape from. This is
      // required for cases where the accumulator type differs from the input
      // types, in which case we will see an `arith.ext_` between the contract
      // and transfer_read which needs to be unrolled.
      VectorType sliceType;
      for (Operation *users : op->getUsers()) {
        auto extract = dyn_cast<vector::ExtractStridedSliceOp>(users);
        if (!extract)
          return std::nullopt;
        auto vecType = cast<VectorType>(extract.getResult().getType());
        if (sliceType && sliceType != vecType)
          return std::nullopt;
        sliceType = vecType;
      }
      if (sliceType)
        return llvm::to_vector(sliceType.getShape());

      // Else unroll for trailing elementwise.
      SmallVector<int64_t> nativeSize(vecType.getRank() - 2, 1);
      // Map elementwise ops to the output shape.
      nativeSize.append({m, n});
      return nativeSize;
    }
  }
  return std::nullopt;
}

void transform::ApplyUnrollVectorsSubgroupMmaOp::populatePatterns(
    RewritePatternSet &patterns) {
  auto unrollOrder = [](Operation *op) -> std::optional<SmallVector<int64_t>> {
    auto contract = dyn_cast<vector::ContractionOp>(op);
    if (!contract)
      return std::nullopt;
    return gpuMmaUnrollOrder(contract);
  };

  int64_t m = getM();
  int64_t n = getN();
  int64_t k = getK();
  auto nativeShapeFn =
      [m, n, k](Operation *op) -> std::optional<SmallVector<int64_t>> {
    return getSubgroupMmaNativeVectorSize(op, m, n, k);
  };
  vector::populateVectorUnrollPatterns(
      patterns, vector::UnrollVectorOptions()
                    .setNativeShapeFn(nativeShapeFn)
                    .setUnrollTraversalOrderFn(unrollOrder));
}

//===----------------------------------------------------------------------===//
// EliminateBarriersOp
//===----------------------------------------------------------------------===//

void EliminateBarriersOp::populatePatterns(RewritePatternSet &patterns) {
  populateGpuEliminateBarriersPatterns(patterns);
}

//===----------------------------------------------------------------------===//
// Block and thread mapping utilities.
//===----------------------------------------------------------------------===//

namespace {
/// Local types used for mapping verification.
struct MappingKind {};
struct BlockMappingKind : MappingKind {};
struct ThreadMappingKind : MappingKind {};
} // namespace

static DiagnosedSilenceableFailure
definiteFailureHelper(std::optional<TransformOpInterface> transformOp,
                      Operation *target, const Twine &message) {
  if (transformOp.has_value())
    return transformOp->emitDefiniteFailure() << message;
  return emitDefiniteFailure(target, message);
}

/// Check if given mapping attributes are one of the desired attributes
template <typename MappingKindType>
static DiagnosedSilenceableFailure
checkMappingAttributeTypes(std::optional<TransformOpInterface> transformOp,
                           scf::ForallOp forallOp) {
  if (!forallOp.getMapping().has_value()) {
    return definiteFailureHelper(transformOp, forallOp,
                                 "scf.forall op requires a mapping attribute");
  }

  bool hasBlockMapping = llvm::any_of(forallOp.getMapping().value(),
                                      llvm::IsaPred<GPUBlockMappingAttr>);
  bool hasWarpgroupMapping = llvm::any_of(
      forallOp.getMapping().value(), llvm::IsaPred<GPUWarpgroupMappingAttr>);
  bool hasWarpMapping = llvm::any_of(forallOp.getMapping().value(),
                                     llvm::IsaPred<GPUWarpMappingAttr>);
  bool hasThreadMapping = llvm::any_of(forallOp.getMapping().value(),
                                       llvm::IsaPred<GPUThreadMappingAttr>);
  bool hasLaneMapping = llvm::any_of(forallOp.getMapping().value(),
                                     llvm::IsaPred<GPULaneMappingAttr>);
  int64_t countMappingTypes = 0;
  countMappingTypes += hasBlockMapping ? 1 : 0;
  countMappingTypes += hasWarpgroupMapping ? 1 : 0;
  countMappingTypes += hasWarpMapping ? 1 : 0;
  countMappingTypes += hasThreadMapping ? 1 : 0;
  countMappingTypes += hasLaneMapping ? 1 : 0;
  if (countMappingTypes > 1) {
    return definiteFailureHelper(
        transformOp, forallOp,
        "cannot mix different mapping types, use nesting");
  }
  if (std::is_same<MappingKindType, BlockMappingKind>::value &&
      !hasBlockMapping) {
    return definiteFailureHelper(
        transformOp, forallOp,
        "scf.forall op requires a mapping attribute of kind 'block'");
  }
  if (std::is_same<MappingKindType, ThreadMappingKind>::value &&
      !hasLaneMapping && !hasThreadMapping && !hasWarpMapping &&
      !hasWarpgroupMapping) {
    return definiteFailureHelper(transformOp, forallOp,
                                 "scf.forall op requires a mapping attribute "
                                 "of kind 'thread' or 'warp'");
  }

  DenseSet<Attribute> seen;
  for (Attribute map : forallOp.getMapping()->getValue()) {
    if (seen.contains(map)) {
      return definiteFailureHelper(
          transformOp, forallOp,
          "duplicate attribute, cannot map different loops "
          "to the same mapping id");
    }
    seen.insert(map);
  }

  auto isLinear = [](DeviceMappingAttrInterface attr) {
    return attr.isLinearMapping();
  };
  if (llvm::any_of(forallOp.getDeviceMappingAttrs(), isLinear) &&
      !llvm::all_of(forallOp.getDeviceMappingAttrs(), isLinear)) {
    return definiteFailureHelper(
        transformOp, forallOp,
        "cannot mix linear and non-linear mapping modes");
  }

  FailureOr<DeviceMaskingAttrInterface> maybeMaskingAttr =
      forallOp.getDeviceMaskingAttr();
  if (succeeded(maybeMaskingAttr) && *maybeMaskingAttr &&
      !forallOp.usesLinearMapping()) {
    return definiteFailureHelper(
        transformOp, forallOp,
        "device masking is only available in linear mapping mode");
  }

  return DiagnosedSilenceableFailure::success();
}

template <typename MappingKindType>
static DiagnosedSilenceableFailure
verifyGpuMapping(std::optional<TransformOpInterface> transformOp,
                 scf::ForallOp forallOp) {
  // Check the types of the mapping attributes match.
  DiagnosedSilenceableFailure typeRes =
      checkMappingAttributeTypes<MappingKindType>(transformOp, forallOp);
  if (!typeRes.succeeded())
    return typeRes;

  // Perform other non-types verifications.
  if (!forallOp.isNormalized())
    return definiteFailureHelper(transformOp, forallOp,
                                 "unsupported non-normalized loops");
  if (forallOp.getNumResults() > 0)
    return definiteFailureHelper(transformOp, forallOp,
                                 "only bufferized scf.forall can be mapped");
  bool useLinearMapping = forallOp.usesLinearMapping();
  // TODO: This would be more natural with support for Optional<EnumParameter>
  // in GPUDeviceMappingAttr.
  int64_t maxNumMappingsSupported =
      useLinearMapping ? (getMaxEnumValForMappingId() -
                          static_cast<uint64_t>(MappingId::DimZ))
                       : 3;
  if (forallOp.getRank() > maxNumMappingsSupported) {
    return definiteFailureHelper(transformOp, forallOp,
                                 "scf.forall with rank > ")
           << maxNumMappingsSupported
           << " does not lower for the specified mapping attribute type";
  }
  auto numParallelIterations =
      getConstantIntValues(forallOp.getMixedUpperBound());
  if (!forallOp.isNormalized() || !numParallelIterations.has_value()) {
    return definiteFailureHelper(
        transformOp, forallOp,
        "requires statically sized, normalized forall op");
  }
  return DiagnosedSilenceableFailure::success();
}

/// Struct to return the result of the rewrite of a forall operation.
struct ForallRewriteResult {
  SmallVector<int64_t> mappingSizes;
  SmallVector<Value> mappingIds;
};

/// Helper to replace ids of dimensions known to be 1 by 0 to simplify the IR.
template <typename OpTy, typename OperationOrBlock>
static void
replaceUnitMappingIdsHelper(RewriterBase &rewriter, Location loc,
                            OperationOrBlock *parent, Value replacement,
                            ArrayRef<int64_t> availableMappingSizes) {
  parent->walk([&](OpTy idOp) {
    if (availableMappingSizes[static_cast<int64_t>(idOp.getDimension())] == 1)
      rewriter.replaceAllUsesWith(idOp.getResult(), replacement);
  });
}

static DiagnosedSilenceableFailure rewriteOneForallCommonImpl(
    RewriterBase &rewriter, std::optional<TransformOpInterface> transformOp,
    scf::ForallOp forallOp, ArrayRef<int64_t> availableMappingSizes,
    ForallRewriteResult &result, const GpuIdBuilder &gpuIdBuilder) {
  LDBG("--start rewriteOneForallCommonImpl");

  // Step 1. Complete the mapping to a full mapping (with 1s) if necessary.
  auto numParallelIterations =
      getConstantIntValues(forallOp.getMixedUpperBound());
  assert(forallOp.isNormalized() && numParallelIterations.has_value() &&
         "requires statically sized, normalized forall op");
  SmallVector<int64_t> tmpMappingSizes = numParallelIterations.value();
  SmallVector<DeviceMappingAttrInterface> forallMappingAttrsVec =
      forallOp.getDeviceMappingAttrs();
  SetVector<Attribute> forallMappingAttrs;
  forallMappingAttrs.insert_range(forallMappingAttrsVec);
  auto comparator = [](Attribute a, Attribute b) -> bool {
    return cast<DeviceMappingAttrInterface>(a).getMappingId() <
           cast<DeviceMappingAttrInterface>(b).getMappingId();
  };

  // Step 1.b. In the linear case, compute the max mapping to avoid needlessly
  // mapping all dimensions. In the 3-D mapping case we need to map all
  // dimensions.
  DeviceMappingAttrInterface maxMapping = cast<DeviceMappingAttrInterface>(
      *llvm::max_element(forallMappingAttrs, comparator));
  DeviceMappingAttrInterface maxLinearMapping;
  if (maxMapping.isLinearMapping())
    maxLinearMapping = maxMapping;
  for (auto attr : gpuIdBuilder.mappingAttributes) {
    // If attr overflows, just skip.
    if (maxLinearMapping && comparator(maxLinearMapping, attr))
      continue;
    // Try to insert. If element was already present, just continue.
    if (!forallMappingAttrs.insert(attr))
      continue;
    // Otherwise, we have a new insertion without a size -> use size 1.
    tmpMappingSizes.push_back(1);
  }
  LDBG("----tmpMappingSizes extracted from scf.forall op: "
       << llvm::interleaved(tmpMappingSizes));

  // Step 2. sort the values by the corresponding DeviceMappingAttrInterface.
  SmallVector<int64_t> forallMappingSizes = getValuesSortedByKey(
      forallMappingAttrs.getArrayRef(), tmpMappingSizes, comparator);
  LDBG("----forallMappingSizes: " << llvm::interleaved(forallMappingSizes));
  LDBG("----forallMappingAttrs: " << llvm::interleaved(forallMappingAttrs));

  // Step 3. Generate the mappingIdOps using the provided generator.
  Location loc = forallOp.getLoc();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forallOp);
  SmallVector<int64_t> originalBasis(availableMappingSizes);
  bool originalBasisWasProvided = !originalBasis.empty();
  if (!originalBasisWasProvided) {
    LDBG("----originalBasis was not provided, deriving it and there will be no "
         "predication");
    originalBasis = forallMappingSizes;
    while (originalBasis.size() < 3)
      originalBasis.push_back(1);
  } else {
    LDBG("----originalBasis was provided, using it, there will be predication");
  }
  LLVM_DEBUG(
      llvm::interleaveComma(originalBasis, DBGS() << "------originalBasis: ");
      llvm::dbgs() << "\n");

  IdBuilderResult builderResult =
      gpuIdBuilder.idBuilder(rewriter, loc, forallMappingSizes, originalBasis);
  if (!builderResult.errorMsg.empty())
    return definiteFailureHelper(transformOp, forallOp, builderResult.errorMsg);

  LLVM_DEBUG(DBGS() << builderResult);

  // Step 4. Map the induction variables to the mappingIdOps, this may involve
  // a permutation.
  SmallVector<Value> mappingIdOps = builderResult.mappingIdOps;
  IRMapping bvm;
  for (auto [iv, dim] : llvm::zip_equal(
           forallOp.getInductionVars(),
           forallMappingAttrs.getArrayRef().take_front(forallOp.getRank()))) {
    auto mappingAttr = cast<DeviceMappingAttrInterface>(dim);
    Value peIdOp = mappingIdOps[mappingAttr.getRelativeIndex()];
    LDBG("----map: " << iv << " to " << peIdOp);
    bvm.map(iv, peIdOp);
  }

  // Step 5. If the originalBasis is already known, create conditionals to
  // predicate the region. Otherwise, the current forall determines the
  // originalBasis and no predication occurs.
  Value predicate;
  if (originalBasisWasProvided) {
    for (Value tmpPredicate : builderResult.predicateOps) {
      predicate = predicate ? arith::AndIOp::create(rewriter, loc, predicate,
                                                    tmpPredicate)
                            : tmpPredicate;
    }
  }

  // Step 6. Move the body of forallOp.
  // Erase the terminator first, it will not be used.
  rewriter.eraseOp(forallOp.getTerminator());
  Block *targetBlock;
  Block::iterator insertionPoint;
  if (predicate) {
    // Step 6.a. If predicated, move at the beginning.
    auto ifOp = scf::IfOp::create(rewriter, loc, predicate,
                                  /*withElseRegion=*/false);
    targetBlock = ifOp.thenBlock();
    insertionPoint = ifOp.thenBlock()->begin();
  } else {
    // Step 6.b. Otherwise, move inline just at the rewriter insertion
    // point.
    targetBlock = forallOp->getBlock();
    insertionPoint = rewriter.getInsertionPoint();
  }
  Block &sourceBlock = forallOp.getRegion().front();
  targetBlock->getOperations().splice(insertionPoint,
                                      sourceBlock.getOperations());

  // Step 7. RAUW indices.
  for (Value loopIndex : forallOp.getInductionVars()) {
    Value threadIdx = bvm.lookup(loopIndex);
    rewriter.replaceAllUsesWith(loopIndex, threadIdx);
  }

  // Step 8. Erase old op.
  rewriter.eraseOp(forallOp);

  LDBG("----result forallMappingSizes: "
       << llvm::interleaved(forallMappingSizes));
  LDBG("----result mappingIdOps: " << llvm::interleaved(mappingIdOps));

  result = ForallRewriteResult{forallMappingSizes, mappingIdOps};
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// MapForallToBlocks
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure mlir::transform::gpu::mapForallToBlocksImpl(
    RewriterBase &rewriter, TransformOpInterface transformOp,
    scf::ForallOp forallOp, SmallVectorImpl<int64_t> &gridDims,
    const GpuIdBuilder &gpuIdBuilder) {
  LDBG("Start mapForallToBlocksImpl");

  {
    // GPU-specific verifications. There is no better place to anchor
    // those right now: the ForallOp is target-independent and the transform
    // op does not apply to individual ForallOp.
    DiagnosedSilenceableFailure diag =
        verifyGpuMapping<BlockMappingKind>(transformOp, forallOp);
    if (!diag.succeeded())
      return diag;
  }

  Location loc = forallOp.getLoc();
  Block *parentBlock = forallOp->getBlock();
  Value zero;
  {
    // Create an early zero index value for replacements and immediately reset
    // the insertion point.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(parentBlock);
    zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  }

  ForallRewriteResult rewriteResult;
  DiagnosedSilenceableFailure diag = rewriteOneForallCommonImpl(
      rewriter, transformOp, forallOp,
      /*availableMappingSizes=*/gridDims, rewriteResult, gpuIdBuilder);

  // Return if anything goes wrong, use silenceable failure as a match
  // failure.
  if (!diag.succeeded())
    return diag;

  // If gridDims was not provided already, set it from the return.
  if (gridDims.empty()) {
    gridDims = rewriteResult.mappingSizes;
    while (gridDims.size() < 3)
      gridDims.push_back(1);
  }
  assert(gridDims.size() == 3 && "Need 3-D gridDims");

  // Replace ids of dimensions known to be 1 by 0 to simplify the IR.
  // Here, the result of mapping determines the available mapping sizes.
  replaceUnitMappingIdsHelper<BlockDimOp>(rewriter, loc, parentBlock, zero,
                                          rewriteResult.mappingSizes);

  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
mlir::transform::gpu::findTopLevelForallOp(Operation *target,
                                           scf::ForallOp &topLevelForallOp,
                                           TransformOpInterface transformOp) {
  auto walkResult = target->walk([&](scf::ForallOp forallOp) {
    if (forallOp->getParentOfType<scf::ForallOp>())
      return WalkResult::advance();
    if (topLevelForallOp)
      // TODO: Handle multiple forall if they are independent.
      return WalkResult::interrupt();
    topLevelForallOp = forallOp;
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted() || !topLevelForallOp)
    return transformOp.emitSilenceableError()
           << "could not find a unique topLevel scf.forall";
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure transform::MapForallToBlocks::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    ApplyToEachResultList &results, transform::TransformState &state) {
  LaunchOp gpuLaunch = dyn_cast<LaunchOp>(target);
  auto transformOp = cast<TransformOpInterface>(getOperation());

  if (!getGenerateGpuLaunch() && !gpuLaunch) {
    DiagnosedSilenceableFailure diag =
        emitSilenceableError()
        << "Given target is not gpu.launch, set `generate_gpu_launch` "
           "attribute";
    diag.attachNote(target->getLoc()) << "when applied to this payload op";
    return diag;
  }

  scf::ForallOp topLevelForallOp;
  DiagnosedSilenceableFailure diag = mlir::transform::gpu::findTopLevelForallOp(
      target, topLevelForallOp, transformOp);
  if (!diag.succeeded()) {
    diag.attachNote(target->getLoc()) << "when applied to this payload op";
    return diag;
  }
  assert(topLevelForallOp && "expect an scf.forall");

  SmallVector<int64_t> gridDims{getGridDims()};
  if (!getGenerateGpuLaunch() && gridDims.size() != 3)
    return transformOp.emitDefiniteFailure("transform require size-3 mapping");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(topLevelForallOp);

  // Generate gpu launch here and move the forall inside
  if (getGenerateGpuLaunch()) {
    DiagnosedSilenceableFailure diag =
        createGpuLaunch(rewriter, target->getLoc(), transformOp, gpuLaunch);
    if (!diag.succeeded())
      return diag;

    rewriter.setInsertionPointToStart(&gpuLaunch.getBody().front());
    Operation *newForallOp = rewriter.clone(*topLevelForallOp);
    rewriter.eraseOp(topLevelForallOp);
    topLevelForallOp = cast<scf::ForallOp>(newForallOp);
  }

  // The BlockIdBuilder adapts to whatever is thrown at it.
  bool useLinearMapping = false;
  if (topLevelForallOp.getMapping())
    useLinearMapping = topLevelForallOp.usesLinearMapping();

  FailureOr<DeviceMaskingAttrInterface> maybeMaskingAttr =
      topLevelForallOp.getDeviceMaskingAttr();
  assert(succeeded(maybeMaskingAttr) && "unexpected failed maybeMaskingAttr");
  assert((!*maybeMaskingAttr || useLinearMapping) &&
         "masking requires linear mapping");

  GpuBlockIdBuilder gpuBlockIdBuilder(getContext(), useLinearMapping,
                                      *maybeMaskingAttr);

  diag = mlir::transform::gpu::mapForallToBlocksImpl(
      rewriter, transformOp, topLevelForallOp, gridDims, gpuBlockIdBuilder);
  if (!diag.succeeded())
    return diag;

  // Set the GPU launch configuration for the grid dims late, this is
  // subject to IR inspection.
  diag = alterGpuLaunch(rewriter, gpuLaunch,
                        cast<TransformOpInterface>(getOperation()), gridDims[0],
                        gridDims[1], gridDims[2]);

  results.push_back(gpuLaunch);
  return diag;
}

LogicalResult transform::MapForallToBlocks::verify() {
  if (!getGridDims().empty() && getGridDims().size() != 3) {
    return emitOpError() << "transform requires empty or size-3 grid_dims";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MapNestedForallToThreads
//===----------------------------------------------------------------------===//

static DiagnosedSilenceableFailure checkMappingSpec(
    std::optional<TransformOpInterface> transformOp, scf::ForallOp forallOp,
    ArrayRef<int64_t> numParallelIterations, ArrayRef<int64_t> blockOrGridSizes,
    int factor, bool useLinearMapping = false) {
  if (!useLinearMapping && blockOrGridSizes.front() % factor != 0) {
    auto diag = definiteFailureHelper(
        transformOp, forallOp,
        Twine("3-D mapping: size of threadIdx.x must be a multiple of ") +
            Twine(factor));
    return diag;
  }
  if (computeProduct(numParallelIterations) * factor >
      computeProduct(blockOrGridSizes)) {
    auto diag = definiteFailureHelper(
        transformOp, forallOp,
        Twine("the number of required parallel resources (blocks or "
              "threads) ") +
            Twine(computeProduct(numParallelIterations) * factor) +
            " overflows the number of available resources " +
            Twine(computeProduct(blockOrGridSizes)));
    return diag;
  }
  return DiagnosedSilenceableFailure::success();
}

static DiagnosedSilenceableFailure
getThreadIdBuilder(std::optional<TransformOpInterface> transformOp,
                   scf::ForallOp forallOp, ArrayRef<int64_t> blockSizes,
                   int64_t warpSize, GpuIdBuilder &gpuIdBuilder) {
  DeviceMappingAttrInterface mappingAttr =
      forallOp.getDeviceMappingAttrs().front();
  bool useLinearMapping = mappingAttr.isLinearMapping();

  // Sanity checks that may result in runtime verification errors.
  auto numParallelIterations =
      getConstantIntValues((forallOp.getMixedUpperBound()));
  if (!forallOp.isNormalized() || !numParallelIterations.has_value()) {
    return definiteFailureHelper(
        transformOp, forallOp,
        "requires statically sized, normalized forall op");
  }
  int64_t factor = 1;
  if (isa<GPUWarpgroupMappingAttr>(mappingAttr)) {
    factor = GpuWarpgroupIdBuilder::kNumWarpsPerGroup * warpSize;
  } else if (isa<GPUWarpMappingAttr>(mappingAttr)) {
    factor = warpSize;
  }
  DiagnosedSilenceableFailure diag =
      checkMappingSpec(transformOp, forallOp, numParallelIterations.value(),
                       blockSizes, factor, useLinearMapping);
  if (!diag.succeeded())
    return diag;

  FailureOr<DeviceMaskingAttrInterface> maybeMaskingAttr =
      forallOp.getDeviceMaskingAttr();
  assert(succeeded(maybeMaskingAttr) && "unexpected failed maybeMaskingAttr");
  assert((!*maybeMaskingAttr || useLinearMapping) &&
         "masking requires linear mapping");

  // Start mapping.
  MLIRContext *ctx = forallOp.getContext();
  gpuIdBuilder =
      TypeSwitch<DeviceMappingAttrInterface, GpuIdBuilder>(mappingAttr)
          .Case([&](GPUWarpgroupMappingAttr) {
            return GpuWarpgroupIdBuilder(ctx, warpSize, useLinearMapping,
                                         *maybeMaskingAttr);
          })
          .Case([&](GPUWarpMappingAttr) {
            return GpuWarpIdBuilder(ctx, warpSize, useLinearMapping,
                                    *maybeMaskingAttr);
          })
          .Case([&](GPUThreadMappingAttr) {
            return GpuThreadIdBuilder(ctx, useLinearMapping, *maybeMaskingAttr);
          })
          .Case([&](GPULaneMappingAttr) {
            return GpuLaneIdBuilder(ctx, warpSize, useLinearMapping,
                                    *maybeMaskingAttr);
          })
          .Default([&](DeviceMappingAttrInterface) -> GpuIdBuilder {
            llvm_unreachable("unknown mapping attribute");
          });
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::transform::gpu::mapOneForallToThreadsImpl(
    RewriterBase &rewriter, std::optional<TransformOpInterface> transformOp,
    scf::ForallOp forallOp, ArrayRef<int64_t> blockSizes, int64_t warpSize,
    bool syncAfterDistribute) {

  {
    // GPU-specific verifications. There is no better place to anchor
    // those right now: the ForallOp is target-independent and the transform
    // op does not apply to individual ForallOp.
    DiagnosedSilenceableFailure diag =
        verifyGpuMapping<ThreadMappingKind>(transformOp, forallOp);
    if (!diag.succeeded())
      return diag;
  }

  GpuIdBuilder gpuIdBuilder;
  {
    // Try to construct the id builder, if it fails, return.
    DiagnosedSilenceableFailure diag = getThreadIdBuilder(
        transformOp, forallOp, blockSizes, warpSize, gpuIdBuilder);
    if (!diag.succeeded())
      return diag;
  }

  Location loc = forallOp.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  // Insert after to allow for syncthreads after `forall` is erased.
  rewriter.setInsertionPointAfter(forallOp);
  ForallRewriteResult rewriteResult;
  DiagnosedSilenceableFailure diag = rewriteOneForallCommonImpl(
      rewriter, transformOp, forallOp, blockSizes, rewriteResult, gpuIdBuilder);
  if (!diag.succeeded())
    return diag;
  // Add a syncthreads if needed. TODO: warpsync
  if (syncAfterDistribute)
    BarrierOp::create(rewriter, loc);

  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::transform::gpu::mapNestedForallToThreadsImpl(
    RewriterBase &rewriter, std::optional<TransformOpInterface> transformOp,
    Operation *target, ArrayRef<int64_t> blockDims, int64_t warpSize,
    bool syncAfterDistribute) {
  LDBG("Start mapNestedForallToThreadsImpl");
  if (blockDims.size() != 3) {
    return definiteFailureHelper(transformOp, target,
                                 "requires size-3 thread mapping");
  }

  // Create an early zero index value for replacements.
  Location loc = target->getLoc();
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  DiagnosedSilenceableFailure diag = DiagnosedSilenceableFailure::success();
  WalkResult walkResult = target->walk([&](scf::ForallOp forallOp) {
    diag = mlir::transform::gpu::mapOneForallToThreadsImpl(
        rewriter, transformOp, forallOp, blockDims, warpSize,
        syncAfterDistribute);
    if (diag.isDefiniteFailure())
      return WalkResult::interrupt();
    if (diag.succeeded())
      return WalkResult::skip();
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return diag;

  // Replace ids of dimensions known to be 1 by 0 to simplify the IR.
  // Here, the result of mapping determines the available mapping sizes.
  replaceUnitMappingIdsHelper<ThreadIdOp>(rewriter, loc, target, zero,
                                          blockDims);

  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure transform::MapNestedForallToThreads::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    ApplyToEachResultList &results, TransformState &state) {
  LaunchOp gpuLaunch = dyn_cast<LaunchOp>(target);
  auto transformOp = cast<TransformOpInterface>(getOperation());

  // Basic high-level verifications.
  if (!gpuLaunch)
    return emitSilenceableError() << "Given target is not a gpu.launch";

  // Mapping to block ids.
  SmallVector<int64_t> blockDims{getBlockDims()};
  DiagnosedSilenceableFailure diag =
      checkGpuLimits(transformOp, std::nullopt, std::nullopt, std::nullopt,
                     blockDims[0], blockDims[1], blockDims[2]);
  if (diag.isSilenceableFailure()) {
    diag.attachNote(getLoc()) << getBlockDimsAttrName() << " is too large";
    return diag;
  }

  // Set the GPU launch configuration for the block dims early, this is not
  // subject to IR inspection.
  diag = alterGpuLaunch(rewriter, gpuLaunch, transformOp, std::nullopt,
                        std::nullopt, std::nullopt, blockDims[0], blockDims[1],
                        blockDims[2]);

  rewriter.setInsertionPointToStart(&gpuLaunch.getBody().front());
  diag =
      mapNestedForallToThreadsImpl(rewriter, transformOp, gpuLaunch, blockDims,
                                   getWarpSize(), getSyncAfterDistribute());

  results.push_back(gpuLaunch.getOperation());
  return diag;
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
/// Registers new ops and declares PDL as dependent dialect since the
/// additional ops are using PDL types for operands and results.
class GPUTransformDialectExtension
    : public transform::TransformDialectExtension<
          GPUTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPUTransformDialectExtension)

  GPUTransformDialectExtension() {
    declareGeneratedDialect<GPUDialect>();
    declareGeneratedDialect<amdgpu::AMDGPUDialect>();
    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<scf::SCFDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.cpp.inc"

void mlir::gpu::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<GPUTransformDialectExtension>();
}
