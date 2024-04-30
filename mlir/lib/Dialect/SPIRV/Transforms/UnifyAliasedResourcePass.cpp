//===- UnifyAliasedResourcePass.cpp - Pass to Unify Aliased Resources -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that unifies access of multiple aliased resources
// into access of one single resource.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/Transforms/Passes.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <iterator>

namespace mlir {
namespace spirv {
#define GEN_PASS_DEF_SPIRVUNIFYALIASEDRESOURCEPASS
#include "mlir/Dialect/SPIRV/Transforms/Passes.h.inc"
} // namespace spirv
} // namespace mlir

#define DEBUG_TYPE "spirv-unify-aliased-resource"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

using Descriptor = std::pair<uint32_t, uint32_t>; // (set #, binding #)
using AliasedResourceMap =
    DenseMap<Descriptor, SmallVector<spirv::GlobalVariableOp>>;

/// Collects all aliased resources in the given SPIR-V `moduleOp`.
static AliasedResourceMap collectAliasedResources(spirv::ModuleOp moduleOp) {
  AliasedResourceMap aliasedResources;
  moduleOp->walk([&aliasedResources](spirv::GlobalVariableOp varOp) {
    if (varOp->getAttrOfType<UnitAttr>("aliased")) {
      std::optional<uint32_t> set = varOp.getDescriptorSet();
      std::optional<uint32_t> binding = varOp.getBinding();
      if (set && binding)
        aliasedResources[{*set, *binding}].push_back(varOp);
    }
  });
  return aliasedResources;
}

/// Returns the element type if the given `type` is a runtime array resource:
/// `!spirv.ptr<!spirv.struct<!spirv.rtarray<...>>>`. Returns null type
/// otherwise.
static Type getRuntimeArrayElementType(Type type) {
  auto ptrType = dyn_cast<spirv::PointerType>(type);
  if (!ptrType)
    return {};

  auto structType = dyn_cast<spirv::StructType>(ptrType.getPointeeType());
  if (!structType || structType.getNumElements() != 1)
    return {};

  auto rtArrayType =
      dyn_cast<spirv::RuntimeArrayType>(structType.getElementType(0));
  if (!rtArrayType)
    return {};

  return rtArrayType.getElementType();
}

/// Given a list of resource element `types`, returns the index of the canonical
/// resource that all resources should be unified into. Returns std::nullopt if
/// unable to unify.
static std::optional<int>
deduceCanonicalResource(ArrayRef<spirv::SPIRVType> types) {
  // scalarNumBits: contains all resources' scalar types' bit counts.
  // vectorNumBits: only contains resources whose element types are vectors.
  // vectorIndices: each vector's original index in `types`.
  SmallVector<int> scalarNumBits, vectorNumBits, vectorIndices;
  scalarNumBits.reserve(types.size());
  vectorNumBits.reserve(types.size());
  vectorIndices.reserve(types.size());

  for (const auto &indexedTypes : llvm::enumerate(types)) {
    spirv::SPIRVType type = indexedTypes.value();
    assert(type.isScalarOrVector());
    if (auto vectorType = dyn_cast<VectorType>(type)) {
      if (vectorType.getNumElements() % 2 != 0)
        return std::nullopt; // Odd-sized vector has special layout
                             // requirements.

      std::optional<int64_t> numBytes = type.getSizeInBytes();
      if (!numBytes)
        return std::nullopt;

      scalarNumBits.push_back(
          vectorType.getElementType().getIntOrFloatBitWidth());
      vectorNumBits.push_back(*numBytes * 8);
      vectorIndices.push_back(indexedTypes.index());
    } else {
      scalarNumBits.push_back(type.getIntOrFloatBitWidth());
    }
  }

  if (!vectorNumBits.empty()) {
    // Choose the *vector* with the smallest bitwidth as the canonical resource,
    // so that we can still keep vectorized load/store and avoid partial updates
    // to large vectors.
    auto *minVal = llvm::min_element(vectorNumBits);
    // Make sure that the canonical resource's bitwidth is divisible by others.
    // With out this, we cannot properly adjust the index later.
    if (llvm::any_of(vectorNumBits,
                     [&](int bits) { return bits % *minVal != 0; }))
      return std::nullopt;

    // Require all scalar type bit counts to be a multiple of the chosen
    // vector's primitive type to avoid reading/writing subcomponents.
    int index = vectorIndices[std::distance(vectorNumBits.begin(), minVal)];
    int baseNumBits = scalarNumBits[index];
    if (llvm::any_of(scalarNumBits,
                     [&](int bits) { return bits % baseNumBits != 0; }))
      return std::nullopt;

    return index;
  }

  // All element types are scalars. Then choose the smallest bitwidth as the
  // cannonical resource to avoid subcomponent load/store.
  auto *minVal = llvm::min_element(scalarNumBits);
  if (llvm::any_of(scalarNumBits,
                   [minVal](int64_t bit) { return bit % *minVal != 0; }))
    return std::nullopt;
  return std::distance(scalarNumBits.begin(), minVal);
}

static bool areSameBitwidthScalarType(Type a, Type b) {
  return a.isIntOrFloat() && b.isIntOrFloat() &&
         a.getIntOrFloatBitWidth() == b.getIntOrFloatBitWidth();
}

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//

namespace {
/// A class for analyzing aliased resources.
///
/// Resources are expected to be spirv.GlobalVarible that has a descriptor set
/// and binding number. Such resources are of the type
/// `!spirv.ptr<!spirv.struct<...>>` per Vulkan requirements.
///
/// Right now, we only support the case that there is a single runtime array
/// inside the struct.
class ResourceAliasAnalysis {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ResourceAliasAnalysis)

  explicit ResourceAliasAnalysis(Operation *);

  /// Returns true if the given `op` can be rewritten to use a canonical
  /// resource.
  bool shouldUnify(Operation *op) const;

  /// Returns all descriptors and their corresponding aliased resources.
  const AliasedResourceMap &getResourceMap() const { return resourceMap; }

  /// Returns the canonical resource for the given descriptor/variable.
  spirv::GlobalVariableOp
  getCanonicalResource(const Descriptor &descriptor) const;
  spirv::GlobalVariableOp
  getCanonicalResource(spirv::GlobalVariableOp varOp) const;

  /// Returns the element type for the given variable.
  spirv::SPIRVType getElementType(spirv::GlobalVariableOp varOp) const;

private:
  /// Given the descriptor and aliased resources bound to it, analyze whether we
  /// can unify them and record if so.
  void recordIfUnifiable(const Descriptor &descriptor,
                         ArrayRef<spirv::GlobalVariableOp> resources);

  /// Mapping from a descriptor to all aliased resources bound to it.
  AliasedResourceMap resourceMap;

  /// Mapping from a descriptor to the chosen canonical resource.
  DenseMap<Descriptor, spirv::GlobalVariableOp> canonicalResourceMap;

  /// Mapping from an aliased resource to its descriptor.
  DenseMap<spirv::GlobalVariableOp, Descriptor> descriptorMap;

  /// Mapping from an aliased resource to its element (scalar/vector) type.
  DenseMap<spirv::GlobalVariableOp, spirv::SPIRVType> elementTypeMap;
};
} // namespace

ResourceAliasAnalysis::ResourceAliasAnalysis(Operation *root) {
  // Collect all aliased resources first and put them into different sets
  // according to the descriptor.
  AliasedResourceMap aliasedResources =
      collectAliasedResources(cast<spirv::ModuleOp>(root));

  // For each resource set, analyze whether we can unify; if so, try to identify
  // a canonical resource, whose element type has the largest bitwidth.
  for (const auto &descriptorResource : aliasedResources) {
    recordIfUnifiable(descriptorResource.first, descriptorResource.second);
  }
}

bool ResourceAliasAnalysis::shouldUnify(Operation *op) const {
  if (!op)
    return false;

  if (auto varOp = dyn_cast<spirv::GlobalVariableOp>(op)) {
    auto canonicalOp = getCanonicalResource(varOp);
    return canonicalOp && varOp != canonicalOp;
  }
  if (auto addressOp = dyn_cast<spirv::AddressOfOp>(op)) {
    auto moduleOp = addressOp->getParentOfType<spirv::ModuleOp>();
    auto *varOp =
        SymbolTable::lookupSymbolIn(moduleOp, addressOp.getVariable());
    return shouldUnify(varOp);
  }

  if (auto acOp = dyn_cast<spirv::AccessChainOp>(op))
    return shouldUnify(acOp.getBasePtr().getDefiningOp());
  if (auto loadOp = dyn_cast<spirv::LoadOp>(op))
    return shouldUnify(loadOp.getPtr().getDefiningOp());
  if (auto storeOp = dyn_cast<spirv::StoreOp>(op))
    return shouldUnify(storeOp.getPtr().getDefiningOp());

  return false;
}

spirv::GlobalVariableOp ResourceAliasAnalysis::getCanonicalResource(
    const Descriptor &descriptor) const {
  auto varIt = canonicalResourceMap.find(descriptor);
  if (varIt == canonicalResourceMap.end())
    return {};
  return varIt->second;
}

spirv::GlobalVariableOp ResourceAliasAnalysis::getCanonicalResource(
    spirv::GlobalVariableOp varOp) const {
  auto descriptorIt = descriptorMap.find(varOp);
  if (descriptorIt == descriptorMap.end())
    return {};
  return getCanonicalResource(descriptorIt->second);
}

spirv::SPIRVType
ResourceAliasAnalysis::getElementType(spirv::GlobalVariableOp varOp) const {
  auto it = elementTypeMap.find(varOp);
  if (it == elementTypeMap.end())
    return {};
  return it->second;
}

void ResourceAliasAnalysis::recordIfUnifiable(
    const Descriptor &descriptor, ArrayRef<spirv::GlobalVariableOp> resources) {
  // Collect the element types for all resources in the current set.
  SmallVector<spirv::SPIRVType> elementTypes;
  for (spirv::GlobalVariableOp resource : resources) {
    Type elementType = getRuntimeArrayElementType(resource.getType());
    if (!elementType)
      return; // Unexpected resource variable type.

    auto type = cast<spirv::SPIRVType>(elementType);
    if (!type.isScalarOrVector())
      return; // Unexpected resource element type.

    elementTypes.push_back(type);
  }

  std::optional<int> index = deduceCanonicalResource(elementTypes);
  if (!index)
    return;

  // Update internal data structures for later use.
  resourceMap[descriptor].assign(resources.begin(), resources.end());
  canonicalResourceMap[descriptor] = resources[*index];
  for (const auto &resource : llvm::enumerate(resources)) {
    descriptorMap[resource.value()] = descriptor;
    elementTypeMap[resource.value()] = elementTypes[resource.index()];
  }
}

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

template <typename OpTy>
class ConvertAliasResource : public OpConversionPattern<OpTy> {
public:
  ConvertAliasResource(const ResourceAliasAnalysis &analysis,
                       MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<OpTy>(context, benefit), analysis(analysis) {}

protected:
  const ResourceAliasAnalysis &analysis;
};

struct ConvertVariable : public ConvertAliasResource<spirv::GlobalVariableOp> {
  using ConvertAliasResource::ConvertAliasResource;

  LogicalResult
  matchAndRewrite(spirv::GlobalVariableOp varOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Just remove the aliased resource. Users will be rewritten to use the
    // canonical one.
    rewriter.eraseOp(varOp);
    return success();
  }
};

struct ConvertAddressOf : public ConvertAliasResource<spirv::AddressOfOp> {
  using ConvertAliasResource::ConvertAliasResource;

  LogicalResult
  matchAndRewrite(spirv::AddressOfOp addressOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Rewrite the AddressOf op to get the address of the canoncical resource.
    auto moduleOp = addressOp->getParentOfType<spirv::ModuleOp>();
    auto srcVarOp = cast<spirv::GlobalVariableOp>(
        SymbolTable::lookupSymbolIn(moduleOp, addressOp.getVariable()));
    auto dstVarOp = analysis.getCanonicalResource(srcVarOp);
    rewriter.replaceOpWithNewOp<spirv::AddressOfOp>(addressOp, dstVarOp);
    return success();
  }
};

struct ConvertAccessChain : public ConvertAliasResource<spirv::AccessChainOp> {
  using ConvertAliasResource::ConvertAliasResource;

  LogicalResult
  matchAndRewrite(spirv::AccessChainOp acOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto addressOp = acOp.getBasePtr().getDefiningOp<spirv::AddressOfOp>();
    if (!addressOp)
      return rewriter.notifyMatchFailure(acOp, "base ptr not addressof op");

    auto moduleOp = acOp->getParentOfType<spirv::ModuleOp>();
    auto srcVarOp = cast<spirv::GlobalVariableOp>(
        SymbolTable::lookupSymbolIn(moduleOp, addressOp.getVariable()));
    auto dstVarOp = analysis.getCanonicalResource(srcVarOp);

    spirv::SPIRVType srcElemType = analysis.getElementType(srcVarOp);
    spirv::SPIRVType dstElemType = analysis.getElementType(dstVarOp);

    if (srcElemType == dstElemType ||
        areSameBitwidthScalarType(srcElemType, dstElemType)) {
      // We have the same bitwidth for source and destination element types.
      // Thie indices keep the same.
      rewriter.replaceOpWithNewOp<spirv::AccessChainOp>(
          acOp, adaptor.getBasePtr(), adaptor.getIndices());
      return success();
    }

    Location loc = acOp.getLoc();

    if (srcElemType.isIntOrFloat() && isa<VectorType>(dstElemType)) {
      // The source indices are for a buffer with scalar element types. Rewrite
      // them into a buffer with vector element types. We need to scale the last
      // index for the vector as a whole, then add one level of index for inside
      // the vector.
      int srcNumBytes = *srcElemType.getSizeInBytes();
      int dstNumBytes = *dstElemType.getSizeInBytes();
      assert(dstNumBytes >= srcNumBytes && dstNumBytes % srcNumBytes == 0);

      auto indices = llvm::to_vector<4>(acOp.getIndices());
      Value oldIndex = indices.back();
      Type indexType = oldIndex.getType();

      int ratio = dstNumBytes / srcNumBytes;
      auto ratioValue = rewriter.create<spirv::ConstantOp>(
          loc, indexType, rewriter.getIntegerAttr(indexType, ratio));

      indices.back() =
          rewriter.create<spirv::SDivOp>(loc, indexType, oldIndex, ratioValue);
      indices.push_back(
          rewriter.create<spirv::SModOp>(loc, indexType, oldIndex, ratioValue));

      rewriter.replaceOpWithNewOp<spirv::AccessChainOp>(
          acOp, adaptor.getBasePtr(), indices);
      return success();
    }

    if ((srcElemType.isIntOrFloat() && dstElemType.isIntOrFloat()) ||
        (isa<VectorType>(srcElemType) && isa<VectorType>(dstElemType))) {
      // The source indices are for a buffer with larger bitwidth scalar/vector
      // element types. Rewrite them into a buffer with smaller bitwidth element
      // types. We only need to scale the last index.
      int srcNumBytes = *srcElemType.getSizeInBytes();
      int dstNumBytes = *dstElemType.getSizeInBytes();
      assert(srcNumBytes >= dstNumBytes && srcNumBytes % dstNumBytes == 0);

      auto indices = llvm::to_vector<4>(acOp.getIndices());
      Value oldIndex = indices.back();
      Type indexType = oldIndex.getType();

      int ratio = srcNumBytes / dstNumBytes;
      auto ratioValue = rewriter.create<spirv::ConstantOp>(
          loc, indexType, rewriter.getIntegerAttr(indexType, ratio));

      indices.back() =
          rewriter.create<spirv::IMulOp>(loc, indexType, oldIndex, ratioValue);

      rewriter.replaceOpWithNewOp<spirv::AccessChainOp>(
          acOp, adaptor.getBasePtr(), indices);
      return success();
    }

    return rewriter.notifyMatchFailure(
        acOp, "unsupported src/dst types for spirv.AccessChain");
  }
};

struct ConvertLoad : public ConvertAliasResource<spirv::LoadOp> {
  using ConvertAliasResource::ConvertAliasResource;

  LogicalResult
  matchAndRewrite(spirv::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcPtrType = cast<spirv::PointerType>(loadOp.getPtr().getType());
    auto srcElemType = cast<spirv::SPIRVType>(srcPtrType.getPointeeType());
    auto dstPtrType = cast<spirv::PointerType>(adaptor.getPtr().getType());
    auto dstElemType = cast<spirv::SPIRVType>(dstPtrType.getPointeeType());

    Location loc = loadOp.getLoc();
    auto newLoadOp = rewriter.create<spirv::LoadOp>(loc, adaptor.getPtr());
    if (srcElemType == dstElemType) {
      rewriter.replaceOp(loadOp, newLoadOp->getResults());
      return success();
    }

    if (areSameBitwidthScalarType(srcElemType, dstElemType)) {
      auto castOp = rewriter.create<spirv::BitcastOp>(loc, srcElemType,
                                                      newLoadOp.getValue());
      rewriter.replaceOp(loadOp, castOp->getResults());

      return success();
    }

    if ((srcElemType.isIntOrFloat() && dstElemType.isIntOrFloat()) ||
        (isa<VectorType>(srcElemType) && isa<VectorType>(dstElemType))) {
      // The source and destination have scalar types of different bitwidths, or
      // vector types of different component counts. For such cases, we load
      // multiple smaller bitwidth values and construct a larger bitwidth one.

      int srcNumBytes = *srcElemType.getSizeInBytes();
      int dstNumBytes = *dstElemType.getSizeInBytes();
      assert(srcNumBytes > dstNumBytes && srcNumBytes % dstNumBytes == 0);
      int ratio = srcNumBytes / dstNumBytes;
      if (ratio > 4)
        return rewriter.notifyMatchFailure(loadOp, "more than 4 components");

      SmallVector<Value> components;
      components.reserve(ratio);
      components.push_back(newLoadOp);

      auto acOp = adaptor.getPtr().getDefiningOp<spirv::AccessChainOp>();
      if (!acOp)
        return rewriter.notifyMatchFailure(loadOp, "ptr not spirv.AccessChain");

      auto i32Type = rewriter.getI32Type();
      Value oneValue = spirv::ConstantOp::getOne(i32Type, loc, rewriter);
      auto indices = llvm::to_vector<4>(acOp.getIndices());
      for (int i = 1; i < ratio; ++i) {
        // Load all subsequent components belonging to this element.
        indices.back() = rewriter.create<spirv::IAddOp>(
            loc, i32Type, indices.back(), oneValue);
        auto componentAcOp = rewriter.create<spirv::AccessChainOp>(
            loc, acOp.getBasePtr(), indices);
        // Assuming little endian, this reads lower-ordered bits of the number
        // to lower-numbered components of the vector.
        components.push_back(
            rewriter.create<spirv::LoadOp>(loc, componentAcOp));
      }

      // Create a vector of the components and then cast back to the larger
      // bitwidth element type. For spirv.bitcast, the lower-numbered components
      // of the vector map to lower-ordered bits of the larger bitwidth element
      // type.

      Type vectorType = srcElemType;
      if (!isa<VectorType>(srcElemType))
        vectorType = VectorType::get({ratio}, dstElemType);

      // If both the source and destination are vector types, we need to make
      // sure the scalar type is the same for composite construction later.
      if (auto srcElemVecType = dyn_cast<VectorType>(srcElemType))
        if (auto dstElemVecType = dyn_cast<VectorType>(dstElemType)) {
          if (srcElemVecType.getElementType() !=
              dstElemVecType.getElementType()) {
            int64_t count =
                dstNumBytes / (srcElemVecType.getElementTypeBitWidth() / 8);

            // Make sure not to create 1-element vectors, which are illegal in
            // SPIR-V.
            Type castType = srcElemVecType.getElementType();
            if (count > 1)
              castType = VectorType::get({count}, castType);

            for (Value &c : components)
              c = rewriter.create<spirv::BitcastOp>(loc, castType, c);
          }
        }
      Value vectorValue = rewriter.create<spirv::CompositeConstructOp>(
          loc, vectorType, components);

      if (!isa<VectorType>(srcElemType))
        vectorValue =
            rewriter.create<spirv::BitcastOp>(loc, srcElemType, vectorValue);
      rewriter.replaceOp(loadOp, vectorValue);
      return success();
    }

    return rewriter.notifyMatchFailure(
        loadOp, "unsupported src/dst types for spirv.Load");
  }
};

struct ConvertStore : public ConvertAliasResource<spirv::StoreOp> {
  using ConvertAliasResource::ConvertAliasResource;

  LogicalResult
  matchAndRewrite(spirv::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcElemType =
        cast<spirv::PointerType>(storeOp.getPtr().getType()).getPointeeType();
    auto dstElemType =
        cast<spirv::PointerType>(adaptor.getPtr().getType()).getPointeeType();
    if (!srcElemType.isIntOrFloat() || !dstElemType.isIntOrFloat())
      return rewriter.notifyMatchFailure(storeOp, "not scalar type");
    if (!areSameBitwidthScalarType(srcElemType, dstElemType))
      return rewriter.notifyMatchFailure(storeOp, "different bitwidth");

    Location loc = storeOp.getLoc();
    Value value = adaptor.getValue();
    if (srcElemType != dstElemType)
      value = rewriter.create<spirv::BitcastOp>(loc, dstElemType, value);
    rewriter.replaceOpWithNewOp<spirv::StoreOp>(storeOp, adaptor.getPtr(),
                                                value, storeOp->getAttrs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {
class UnifyAliasedResourcePass final
    : public spirv::impl::SPIRVUnifyAliasedResourcePassBase<
          UnifyAliasedResourcePass> {
public:
  explicit UnifyAliasedResourcePass(spirv::GetTargetEnvFn getTargetEnv)
      : getTargetEnvFn(std::move(getTargetEnv)) {}

  void runOnOperation() override;

private:
  spirv::GetTargetEnvFn getTargetEnvFn;
};

void UnifyAliasedResourcePass::runOnOperation() {
  spirv::ModuleOp moduleOp = getOperation();
  MLIRContext *context = &getContext();

  if (getTargetEnvFn) {
    // This pass is only needed for targeting WebGPU, Metal, or layering
    // Vulkan on Metal via MoltenVK, where we need to translate SPIR-V into
    // WGSL or MSL. The translation has limitations.
    spirv::TargetEnvAttr targetEnv = getTargetEnvFn(moduleOp);
    spirv::ClientAPI clientAPI = targetEnv.getClientAPI();
    bool isVulkanOnAppleDevices =
        clientAPI == spirv::ClientAPI::Vulkan &&
        targetEnv.getVendorID() == spirv::Vendor::Apple;
    if (clientAPI != spirv::ClientAPI::WebGPU &&
        clientAPI != spirv::ClientAPI::Metal && !isVulkanOnAppleDevices)
      return;
  }

  // Analyze aliased resources first.
  ResourceAliasAnalysis &analysis = getAnalysis<ResourceAliasAnalysis>();

  ConversionTarget target(*context);
  target.addDynamicallyLegalOp<spirv::GlobalVariableOp, spirv::AddressOfOp,
                               spirv::AccessChainOp, spirv::LoadOp,
                               spirv::StoreOp>(
      [&analysis](Operation *op) { return !analysis.shouldUnify(op); });
  target.addLegalDialect<spirv::SPIRVDialect>();

  // Run patterns to rewrite usages of non-canonical resources.
  RewritePatternSet patterns(context);
  patterns.add<ConvertVariable, ConvertAddressOf, ConvertAccessChain,
               ConvertLoad, ConvertStore>(analysis, context);
  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
    return signalPassFailure();

  // Drop aliased attribute if we only have one single bound resource for a
  // descriptor. We need to re-collect the map here given in the above the
  // conversion is best effort; certain sets may not be converted.
  AliasedResourceMap resourceMap =
      collectAliasedResources(cast<spirv::ModuleOp>(moduleOp));
  for (const auto &dr : resourceMap) {
    const auto &resources = dr.second;
    if (resources.size() == 1)
      resources.front()->removeAttr("aliased");
  }
}
} // namespace

std::unique_ptr<mlir::OperationPass<spirv::ModuleOp>>
spirv::createUnifyAliasedResourcePass(spirv::GetTargetEnvFn getTargetEnv) {
  return std::make_unique<UnifyAliasedResourcePass>(std::move(getTargetEnv));
}
