//===- SPIRVLowering.cpp - Standard to SPIR-V dialect conversion--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities used to lower to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Dialect/SPIRV/LayoutUtils.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Debug.h"

#include <functional>

#define DEBUG_TYPE "mlir-spirv-lowering"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

Type SPIRVTypeConverter::getIndexType(MLIRContext *context) {
  // Convert to 32-bit integers for now. Might need a way to control this in
  // future.
  // TODO(ravishankarm): It is probably better to make it 64-bit integers. To
  // this some support is needed in SPIR-V dialect for Conversion
  // instructions. The Vulkan spec requires the builtins like
  // GlobalInvocationID, etc. to be 32-bit (unsigned) integers which should be
  // SExtended to 64-bit for index computations.
  return IntegerType::get(32, context);
}

// TODO(ravishankarm): This is a utility function that should probably be
// exposed by the SPIR-V dialect. Keeping it local till the use case arises.
static Optional<int64_t> getTypeNumBytes(Type t) {
  if (spirv::SPIRVDialect::isValidScalarType(t)) {
    auto bitWidth = t.getIntOrFloatBitWidth();
    // According to the SPIR-V spec:
    // "There is no physical size or bit pattern defined for values with boolean
    // type. If they are stored (in conjunction with OpVariable), they can only
    // be used with logical addressing operations, not physical, and only with
    // non-externally visible shader Storage Classes: Workgroup, CrossWorkgroup,
    // Private, Function, Input, and Output."
    if (bitWidth == 1) {
      return llvm::None;
    }
    return bitWidth / 8;
  } else if (auto memRefType = t.dyn_cast<MemRefType>()) {
    // TODO: Layout should also be controlled by the ABI attributes. For now
    // using the layout from MemRef.
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    if (!memRefType.hasStaticShape() ||
        failed(getStridesAndOffset(memRefType, strides, offset))) {
      return llvm::None;
    }
    // To get the size of the memref object in memory, the total size is the
    // max(stride * dimension-size) computed for all dimensions times the size
    // of the element.
    auto elementSize = getTypeNumBytes(memRefType.getElementType());
    if (!elementSize) {
      return llvm::None;
    }
    if (memRefType.getRank() == 0) {
      return elementSize;
    }
    auto dims = memRefType.getShape();
    if (llvm::is_contained(dims, ShapedType::kDynamicSize) ||
        offset == MemRefType::getDynamicStrideOrOffset() ||
        llvm::is_contained(strides, MemRefType::getDynamicStrideOrOffset())) {
      return llvm::None;
    }
    int64_t memrefSize = -1;
    for (auto shape : enumerate(dims)) {
      memrefSize = std::max(memrefSize, shape.value() * strides[shape.index()]);
    }
    return (offset + memrefSize) * elementSize.getValue();
  } else if (auto tensorType = t.dyn_cast<TensorType>()) {
    if (!tensorType.hasStaticShape()) {
      return llvm::None;
    }
    auto elementSize = getTypeNumBytes(tensorType.getElementType());
    if (!elementSize) {
      return llvm::None;
    }
    int64_t size = elementSize.getValue();
    for (auto shape : tensorType.getShape()) {
      size *= shape;
    }
    return size;
  }
  // TODO: Add size computation for other types.
  return llvm::None;
}

SPIRVTypeConverter::SPIRVTypeConverter() {
  addConversion([](Type type) -> Optional<Type> {
    // If the type is already valid in SPIR-V, directly return.
    return spirv::SPIRVDialect::isValidType(type) ? type : Optional<Type>();
  });
  addConversion([](IndexType indexType) {
    return SPIRVTypeConverter::getIndexType(indexType.getContext());
  });
  addConversion([this](MemRefType memRefType) -> Type {
    // TODO(ravishankarm): For now only support default memory space. The memory
    // space description is not set is stone within MLIR, i.e. it depends on the
    // context it is being used. To map this to SPIR-V storage classes, we
    // should rely on the ABI attributes, and not on the memory space. This is
    // still evolving, and needs to be revisited when there is more clarity.
    if (memRefType.getMemorySpace())
      return Type();

    auto elementType = convertType(memRefType.getElementType());
    if (!elementType)
      return Type();

    auto elementSize = getTypeNumBytes(elementType);
    if (!elementSize)
      return Type();

    // TODO(ravishankarm) : Handle dynamic shapes.
    if (memRefType.hasStaticShape()) {
      auto arraySize = getTypeNumBytes(memRefType);
      if (!arraySize)
        return Type();

      auto arrayType = spirv::ArrayType::get(
          elementType, arraySize.getValue() / elementSize.getValue(),
          elementSize.getValue());
      auto structType = spirv::StructType::get(arrayType, 0);
      // For now initialize the storage class to StorageBuffer. This will be
      // updated later based on whats passed in w.r.t to the ABI attributes.
      return spirv::PointerType::get(structType,
                                     spirv::StorageClass::StorageBuffer);
    }
    return Type();
  });
  addConversion([this](TensorType tensorType) -> Type {
    // TODO(ravishankarm) : Handle dynamic shapes.
    if (!tensorType.hasStaticShape())
      return Type();

    auto elementType = convertType(tensorType.getElementType());
    if (!elementType)
      return Type();

    auto elementSize = getTypeNumBytes(elementType);
    if (!elementSize)
      return Type();

    auto tensorSize = getTypeNumBytes(tensorType);
    if (!tensorSize)
      return Type();

    return spirv::ArrayType::get(elementType,
                                 tensorSize.getValue() / elementSize.getValue(),
                                 elementSize.getValue());
  });
}

//===----------------------------------------------------------------------===//
// FuncOp Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {
/// A pattern for rewriting function signature to convert arguments of functions
/// to be of valid SPIR-V types.
class FuncOpConversion final : public SPIRVOpLowering<FuncOp> {
public:
  using SPIRVOpLowering<FuncOp>::SPIRVOpLowering;

  PatternMatchResult
  matchAndRewrite(FuncOp funcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

PatternMatchResult
FuncOpConversion::matchAndRewrite(FuncOp funcOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const {
  auto fnType = funcOp.getType();
  // TODO(antiagainst): support converting functions with one result.
  if (fnType.getNumResults())
    return matchFailure();

  TypeConverter::SignatureConversion signatureConverter(fnType.getNumInputs());
  for (auto argType : enumerate(funcOp.getType().getInputs())) {
    auto convertedType = typeConverter.convertType(argType.value());
    if (!convertedType)
      return matchFailure();
    signatureConverter.addInputs(argType.index(), convertedType);
  }

  // Create the converted spv.func op.
  auto newFuncOp = rewriter.create<spirv::FuncOp>(
      funcOp.getLoc(), funcOp.getName(),
      rewriter.getFunctionType(signatureConverter.getConvertedTypes(),
                               llvm::None));

  // Copy over all attributes other than the function name and type.
  for (const auto &namedAttr : funcOp.getAttrs()) {
    if (!namedAttr.first.is(impl::getTypeAttrName()) &&
        !namedAttr.first.is(SymbolTable::getSymbolAttrName()))
      newFuncOp.setAttr(namedAttr.first, namedAttr.second);
  }

  rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                              newFuncOp.end());
  rewriter.applySignatureConversion(&newFuncOp.getBody(), signatureConverter);
  rewriter.eraseOp(funcOp);
  return matchSuccess();
}

void mlir::populateBuiltinFuncToSPIRVPatterns(
    MLIRContext *context, SPIRVTypeConverter &typeConverter,
    OwningRewritePatternList &patterns) {
  patterns.insert<FuncOpConversion>(context, typeConverter);
}

//===----------------------------------------------------------------------===//
// Builtin Variables
//===----------------------------------------------------------------------===//

static spirv::GlobalVariableOp getBuiltinVariable(Block &body,
                                                  spirv::BuiltIn builtin) {
  // Look through all global variables in the given `body` block and check if
  // there is a spv.globalVariable that has the same `builtin` attribute.
  for (auto varOp : body.getOps<spirv::GlobalVariableOp>()) {
    if (auto builtinAttr = varOp.getAttrOfType<StringAttr>(
            spirv::SPIRVDialect::getAttributeName(
                spirv::Decoration::BuiltIn))) {
      auto varBuiltIn = spirv::symbolizeBuiltIn(builtinAttr.getValue());
      if (varBuiltIn && varBuiltIn.getValue() == builtin) {
        return varOp;
      }
    }
  }
  return nullptr;
}

/// Gets name of global variable for a builtin.
static std::string getBuiltinVarName(spirv::BuiltIn builtin) {
  return std::string("__builtin_var_") + stringifyBuiltIn(builtin).str() + "__";
}

/// Gets or inserts a global variable for a builtin within `body` block.
static spirv::GlobalVariableOp
getOrInsertBuiltinVariable(Block &body, Location loc, spirv::BuiltIn builtin,
                           OpBuilder &builder) {
  if (auto varOp = getBuiltinVariable(body, builtin))
    return varOp;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&body);

  spirv::GlobalVariableOp newVarOp;
  switch (builtin) {
  case spirv::BuiltIn::NumWorkgroups:
  case spirv::BuiltIn::WorkgroupSize:
  case spirv::BuiltIn::WorkgroupId:
  case spirv::BuiltIn::LocalInvocationId:
  case spirv::BuiltIn::GlobalInvocationId: {
    auto ptrType = spirv::PointerType::get(
        VectorType::get({3}, builder.getIntegerType(32)),
        spirv::StorageClass::Input);
    std::string name = getBuiltinVarName(builtin);
    newVarOp =
        builder.create<spirv::GlobalVariableOp>(loc, ptrType, name, builtin);
    break;
  }
  default:
    emitError(loc, "unimplemented builtin variable generation for ")
        << stringifyBuiltIn(builtin);
  }
  return newVarOp;
}

Value mlir::spirv::getBuiltinVariableValue(Operation *op,
                                           spirv::BuiltIn builtin,
                                           OpBuilder &builder) {
  Operation *parent = SymbolTable::getNearestSymbolTable(op->getParentOp());
  if (!parent) {
    op->emitError("expected operation to be within a module-like op");
    return nullptr;
  }

  spirv::GlobalVariableOp varOp = getOrInsertBuiltinVariable(
      *parent->getRegion(0).begin(), op->getLoc(), builtin, builder);
  Value ptr = builder.create<spirv::AddressOfOp>(op->getLoc(), varOp);
  return builder.create<spirv::LoadOp>(op->getLoc(), ptr);
}

//===----------------------------------------------------------------------===//
// Index calculation
//===----------------------------------------------------------------------===//

spirv::AccessChainOp mlir::spirv::getElementPtr(
    SPIRVTypeConverter &typeConverter, MemRefType baseType, Value basePtr,
    ArrayRef<Value> indices, Location loc, OpBuilder &builder) {
  // Get base and offset of the MemRefType and verify they are static.
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  if (failed(getStridesAndOffset(baseType, strides, offset)) ||
      llvm::is_contained(strides, MemRefType::getDynamicStrideOrOffset())) {
    return nullptr;
  }

  auto indexType = typeConverter.getIndexType(builder.getContext());

  Value ptrLoc = nullptr;
  assert(indices.size() == strides.size() &&
         "must provide indices for all dimensions");
  for (auto index : enumerate(indices)) {
    Value strideVal = builder.create<spirv::ConstantOp>(
        loc, indexType, IntegerAttr::get(indexType, strides[index.index()]));
    Value update = builder.create<spirv::IMulOp>(loc, strideVal, index.value());
    ptrLoc =
        (ptrLoc ? builder.create<spirv::IAddOp>(loc, ptrLoc, update).getResult()
                : update);
  }
  SmallVector<Value, 2> linearizedIndices;
  // Add a '0' at the start to index into the struct.
  auto zero = spirv::ConstantOp::getZero(indexType, loc, &builder);
  linearizedIndices.push_back(zero);
  // If it is a zero-rank memref type, extract the element directly.
  if (!ptrLoc) {
    ptrLoc = zero;
  }
  linearizedIndices.push_back(ptrLoc);
  return builder.create<spirv::AccessChainOp>(loc, basePtr, linearizedIndices);
}

//===----------------------------------------------------------------------===//
// Set ABI attributes for lowering entry functions.
//===----------------------------------------------------------------------===//

LogicalResult
mlir::spirv::setABIAttrs(spirv::FuncOp funcOp,
                         spirv::EntryPointABIAttr entryPointInfo,
                         ArrayRef<spirv::InterfaceVarABIAttr> argABIInfo) {
  // Set the attributes for argument and the function.
  StringRef argABIAttrName = spirv::getInterfaceVarABIAttrName();
  for (auto argIndex : llvm::seq<unsigned>(0, funcOp.getNumArguments())) {
    funcOp.setArgAttr(argIndex, argABIAttrName, argABIInfo[argIndex]);
  }
  funcOp.setAttr(spirv::getEntryPointABIAttrName(), entryPointInfo);
  return success();
}

//===----------------------------------------------------------------------===//
// SPIR-V ConversionTarget
//===----------------------------------------------------------------------===//

std::unique_ptr<spirv::SPIRVConversionTarget>
spirv::SPIRVConversionTarget::get(spirv::TargetEnvAttr targetEnv,
                                  MLIRContext *context) {
  std::unique_ptr<SPIRVConversionTarget> target(
      // std::make_unique does not work here because the constructor is private.
      new SPIRVConversionTarget(targetEnv, context));
  SPIRVConversionTarget *targetPtr = target.get();
  target->addDynamicallyLegalDialect<SPIRVDialect>(
      Optional<ConversionTarget::DynamicLegalityCallbackFn>(
          // We need to capture the raw pointer here because it is stable:
          // target will be destroyed once this function is returned.
          [targetPtr](Operation *op) { return targetPtr->isLegalOp(op); }));
  return target;
}

spirv::SPIRVConversionTarget::SPIRVConversionTarget(
    spirv::TargetEnvAttr targetEnv, MLIRContext *context)
    : ConversionTarget(*context), givenVersion(targetEnv.getVersion()) {
  for (spirv::Extension ext : targetEnv.getExtensions())
    givenExtensions.insert(ext);

  // Add extensions implied by the current version.
  for (spirv::Extension ext : spirv::getImpliedExtensions(givenVersion))
    givenExtensions.insert(ext);

  for (spirv::Capability cap : targetEnv.getCapabilities()) {
    givenCapabilities.insert(cap);

    // Add capabilities implied by the current capability.
    for (spirv::Capability c : spirv::getRecursiveImpliedCapabilities(cap))
      givenCapabilities.insert(c);
  }
}

bool spirv::SPIRVConversionTarget::isLegalOp(Operation *op) {
  // Make sure this op is available at the given version. Ops not implementing
  // QueryMinVersionInterface/QueryMaxVersionInterface are available to all
  // SPIR-V versions.
  if (auto minVersion = dyn_cast<spirv::QueryMinVersionInterface>(op))
    if (minVersion.getMinVersion() > givenVersion) {
      LLVM_DEBUG(llvm::dbgs()
                 << op->getName() << " illegal: requiring min version "
                 << spirv::stringifyVersion(minVersion.getMinVersion())
                 << "\n");
      return false;
    }
  if (auto maxVersion = dyn_cast<spirv::QueryMaxVersionInterface>(op))
    if (maxVersion.getMaxVersion() < givenVersion) {
      LLVM_DEBUG(llvm::dbgs()
                 << op->getName() << " illegal: requiring max version "
                 << spirv::stringifyVersion(maxVersion.getMaxVersion())
                 << "\n");
      return false;
    }

  // Make sure this op's required extensions are allowed to use. For each op,
  // we return a vector of vector for its extension requirements following
  // ((Extension::A OR Extension::B) AND (Extension::C OR Extension::D))
  // convention. Ops not implementing QueryExtensionInterface do not require
  // extensions to be available.
  if (auto extensions = dyn_cast<spirv::QueryExtensionInterface>(op)) {
    auto exts = extensions.getExtensions();
    for (const auto &ors : exts)
      if (llvm::all_of(ors, [this](spirv::Extension ext) {
            return this->givenExtensions.count(ext) == 0;
          })) {
        LLVM_DEBUG(llvm::dbgs() << op->getName()
                                << " illegal: missing required extension\n");
        return false;
      }
  }

  // Make sure this op's required extensions are allowed to use. For each op,
  // we return a vector of vector for its capability requirements following
  // ((Capability::A OR Extension::B) AND (Capability::C OR Capability::D))
  // convention. Ops not implementing QueryExtensionInterface do not require
  // extensions to be available.
  if (auto capabilities = dyn_cast<spirv::QueryCapabilityInterface>(op)) {
    auto caps = capabilities.getCapabilities();
    for (const auto &ors : caps)
      if (llvm::all_of(ors, [this](spirv::Capability cap) {
            return this->givenCapabilities.count(cap) == 0;
          })) {
        LLVM_DEBUG(llvm::dbgs() << op->getName()
                                << " illegal: missing required capability\n");
        return false;
      }
  }

  return true;
}
