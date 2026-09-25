//===- ACCDataRuntime.cpp - Emit OpenACC data runtime arguments -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Builds the mapping argument arrays used by OpenACC data runtime entry points.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/OpenACCToLLVM/ACCDataRuntime.h"
#include "mlir/Conversion/OpenACCToLLVM/ACCToLLVMUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsCG.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsType.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include <algorithm>

#define DEBUG_TYPE "acc-data-runtime"

using namespace mlir;
using namespace mlir::acc;

static Value remap(Value value, ConversionPatternRewriter &rewriter) {
  if (!value)
    return {};
  if (Value converted = rewriter.getRemappedValue(value))
    return converted;
  return value;
}

static Value constantI64(Location loc, int64_t value,
                         ConversionPatternRewriter &rewriter) {
  return LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(), value);
}

static Value getPointer(Location loc, Value value,
                        ConversionPatternRewriter &rewriter) {
  Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
  if (!value)
    return LLVM::ZeroOp::create(rewriter, loc, ptrTy);
  if (value.getType() == ptrTy)
    return value;

  if (isa<PointerLikeType>(value.getType()))
    return castPointerLikeTypeIfNeeded(rewriter, loc, value, ptrTy);

  Value one = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(), 1);
  Value storage =
      LLVM::AllocaOp::create(rewriter, loc, ptrTy, value.getType(), one);
  LLVM::StoreOp::create(rewriter, loc, value, storage);
  return storage;
}

/// Returns the address of a slot holding \p value, so that the runtime reads
/// the value from memory rather than receiving it directly.
static Value getIndirectPointer(Location loc, Value value,
                                ConversionPatternRewriter &rewriter) {
  Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
  Value one = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(), 1);
  Value storage =
      LLVM::AllocaOp::create(rewriter, loc, ptrTy, value.getType(), one);
  LLVM::StoreOp::create(rewriter, loc, value, storage);
  return storage;
}

Value mlir::createACCDataArray(Location loc, Type elementType,
                               ArrayRef<Value> values, RewriterBase &rewriter) {
  Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
  Type i32Ty = rewriter.getI32Type();
  Value count = LLVM::ConstantOp::create(rewriter, loc, i32Ty, values.size());
  Value array =
      LLVM::AllocaOp::create(rewriter, loc, ptrTy, elementType, count);
  for (auto [index, value] : llvm::enumerate(values)) {
    Value indexValue = LLVM::ConstantOp::create(rewriter, loc, i32Ty, index);
    Value element = LLVM::GEPOp::create(rewriter, loc, ptrTy, elementType,
                                        array, ValueRange{indexValue});
    LLVM::StoreOp::create(rewriter, loc, value, element);
  }
  return array;
}

/// The address of the mapped object, as the runtime expects to receive it.
/// Privatized storage only ever exists on the device, so there is no host
/// address to give. A device address names storage the runtime must not look
/// up, and an object mapped by value has no address at all, so both are handed
/// over in a slot the runtime reads them from.
static Value getMappedObjectPointer(Location loc, Value operand,
                                    Value converted, MapFlags mapFlags,
                                    ConversionPatternRewriter &rewriter) {
  Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
  if (isa<PrivateType>(operand.getType()))
    return LLVM::ZeroOp::create(rewriter, loc, ptrTy);
  if (converted && converted.getType() == ptrTy &&
      bitEnumContainsAny(mapFlags, MapFlags::devptr))
    return getIndirectPointer(loc, converted, rewriter);
  return getPointer(loc, converted, rewriter);
}

static std::optional<MapFlags> computePackedMapFlags(Operation *mapOp) {
  if (auto mapInfo = dyn_cast<MapInfoOp>(mapOp))
    return mapInfo.getMapFlags();
  if (isa<ACC_DATA_ENTRY_OPS>(mapOp))
    return computeDataClauseMapFlags(mapOp, hasAttachPoint(mapOp));
  return std::nullopt;
}

static Value getMapSize(Operation *mapOp, Value converted, MapFlags mapFlags,
                        OpenACCSupport &accSupport,
                        ConversionPatternRewriter &rewriter) {
  Location loc = mapOp->getLoc();
  ModuleOp module = mapOp->getParentOfType<ModuleOp>();
  // An object mapped by value is passed as the argument itself, so its size is
  // the size of what is passed and not of the object it was read from. The two
  // differ when the argument holds one field of a decomposed aggregate.
  if (bitEnumContainsAny(mapFlags, MapFlags::literal) && converted) {
    if (auto sizeAndAlignment = acc::getTypeSizeAndAlignment(
            converted.getType(), module, &accSupport))
      return constantI64(loc, sizeAndAlignment->first.getFixedValue(),
                         rewriter);
  }
  if (Value size = acc::getMapSize(mapOp))
    return castToI64(loc, remap(size, rewriter), rewriter);

  Value var = acc::getVar(mapOp);
  if (!var)
    return constantI64(loc, 0, rewriter);
  Type varType = acc::getVarType(mapOp);
  if (!varType)
    varType = var.getType();
  // A memref is handed over with a descriptor stating its extents and element
  // size, which is what states the size of such a mapping.
  if (isa<MemRefType>(var.getType()) || isa<MemRefType>(varType))
    return constantI64(loc, 0, rewriter);

  // A data clause operation that was not turned into an `acc.map_info` has no
  // size of its own, so it is sized the way that operation would have been.
  // A size that is not statically known is left to the runtime, as an unsized
  // mapping is.
  int64_t size =
      acc::computeMapInfoSizeBytes(var, varType, acc::getDataDescKind(mapOp),
                                   acc::getBounds(mapOp), &accSupport);
  return constantI64(loc, std::max<int64_t>(size, 0), rewriter);
}

static Value getElementSize(Operation *mapOp, OpenACCSupport &accSupport,
                            ConversionPatternRewriter &rewriter) {
  Location loc = mapOp->getLoc();
  ModuleOp module = mapOp->getParentOfType<ModuleOp>();
  if (std::optional<int64_t> size = acc::getMapElementSize(mapOp))
    return constantI64(loc, *size, rewriter);

  Type type = acc::getVarType(mapOp);
  if (auto shaped = dyn_cast<ShapedType>(type))
    type = shaped.getElementType();
  if (!type)
    return constantI64(loc, 0, rewriter);
  if (auto sizeAndAlignment =
          acc::getTypeSizeAndAlignment(type, module, &accSupport))
    return constantI64(loc, sizeAndAlignment->first.getFixedValue(), rewriter);
  return constantI64(loc, 0, rewriter);
}

static Value getBoundValue(Value value, Location loc,
                           ConversionPatternRewriter &rewriter) {
  if (!value)
    return constantI64(loc, 0, rewriter);
  return castToI64(loc, remap(value, rewriter), rewriter);
}

Value mlir::createACCDataDescriptor(Location loc, Value baseDescriptor,
                                    Type baseDescriptorType, uint32_t version,
                                    ValueRange bounds, Value elementSize,
                                    ConversionPatternRewriter &rewriter) {
  MLIRContext *context = rewriter.getContext();
  Type i8Ty = rewriter.getI8Type();
  Type i32Ty = rewriter.getI32Type();
  Type i64Ty = rewriter.getI64Type();
  Type ptrTy = LLVM::LLVMPointerType::get(context);

  Value versionValue = LLVM::ConstantOp::create(
      rewriter, loc, i32Ty,
      version | static_cast<uint32_t>(
                    getDataDescriptorKind(DataDescriptor::AccDataDescOpenACC)));
  baseDescriptor = LLVM::InsertValueOp::create(
      rewriter, loc, baseDescriptorType, baseDescriptor, versionValue,
      ArrayRef<int64_t>{0});

  Type descriptorType = getDataDescriptorType(
      context, DataDescriptor::AccDataDescOpenACC, baseDescriptorType);
  Value descriptor = LLVM::ZeroOp::create(rewriter, loc, descriptorType);
  Value rank = LLVM::ConstantOp::create(rewriter, loc, i8Ty, bounds.size());

  SmallVector<Value> lowerBounds;
  SmallVector<Value> upperBounds;
  SmallVector<Value> extents;
  SmallVector<Value> strides;
  SmallVector<Value> starts;
  for (Value boundValue : bounds) {
    auto bound = cast<DataBoundsOp>(boundValue.getDefiningOp());
    lowerBounds.push_back(
        getBoundValue(bound.getLowerbound(), bound.getLoc(), rewriter));
    upperBounds.push_back(
        getBoundValue(bound.getUpperbound(), bound.getLoc(), rewriter));
    Value sourceExtent =
        bound.getSourceExtent() ? bound.getSourceExtent() : bound.getExtent();
    extents.push_back(getBoundValue(sourceExtent, bound.getLoc(), rewriter));
    Value stride = getBoundValue(bound.getStride(), bound.getLoc(), rewriter);
    if (!bound.getStrideInBytes())
      stride =
          LLVM::MulOp::create(rewriter, bound.getLoc(), stride, elementSize);
    strides.push_back(stride);
    starts.push_back(
        getBoundValue(bound.getStartIdx(), bound.getLoc(), rewriter));
  }

  Value lowerBoundsArray =
      createACCDataArray(loc, i64Ty, lowerBounds, rewriter);
  Value upperBoundsArray =
      createACCDataArray(loc, i64Ty, upperBounds, rewriter);
  Value extentsArray = createACCDataArray(loc, i64Ty, extents, rewriter);
  Value stridesArray = createACCDataArray(loc, i64Ty, strides, rewriter);
  Value startsArray = createACCDataArray(loc, i64Ty, starts, rewriter);

  using Field = AccDataDescOpenACCField;
  auto insert = [&](Value value, Field field) {
    descriptor = LLVM::InsertValueOp::create(
        rewriter, loc, descriptorType, descriptor, value,
        ArrayRef<int64_t>{getDataDescriptorFieldIndex(field)});
  };
  insert(baseDescriptor, Field::Base);
  insert(rank, Field::Rank);
  insert(elementSize, Field::ElementSize);
  insert(lowerBoundsArray, Field::LowerBounds);
  insert(upperBoundsArray, Field::UpperBounds);
  insert(extentsArray, Field::Extents);
  insert(stridesArray, Field::StridesInBytes);
  insert(startsArray, Field::StartIndices);

  Value one = LLVM::ConstantOp::create(rewriter, loc, i32Ty, 1);
  Value storage =
      LLVM::AllocaOp::create(rewriter, loc, ptrTy, descriptorType, one);
  LLVM::StoreOp::create(rewriter, loc, descriptor, storage);
  return storage;
}

Value mlir::createACCMemRefDescriptorWrapperArg(
    Location loc, MemRefType memrefType, Value convertedMemref,
    ValueRange bounds, Value elementSize, ConversionPatternRewriter &rewriter) {
  MLIRContext *context = rewriter.getContext();
  Type i8Ty = rewriter.getI8Type();
  Type i32Ty = rewriter.getI32Type();
  Type ptrTy = LLVM::LLVMPointerType::get(context);
  using Field = AccDataDescMemRefField;
  Type descriptorType =
      getDataDescriptorType(context, DataDescriptor::AccDataDescMemRef);
  auto version = static_cast<uint32_t>(
      getDataDescriptorKind(DataDescriptor::AccDataDescMemRef));
  Value descriptor = LLVM::ZeroOp::create(rewriter, loc, descriptorType);
  Value rank =
      LLVM::ConstantOp::create(rewriter, loc, i8Ty, memrefType.getRank());
  auto insert = [&](Value value, Field field) {
    descriptor = LLVM::InsertValueOp::create(
        rewriter, loc, descriptorType, descriptor, value,
        ArrayRef<int64_t>{getDataDescriptorFieldIndex(field)});
  };
  insert(rank, Field::Rank);
  insert(elementSize, Field::ElementSize);
  insert(getPointer(loc, convertedMemref, rewriter), Field::MemRefDescriptor);
  if (!bounds.empty())
    return createACCDataDescriptor(loc, descriptor, descriptorType, version,
                                   bounds, elementSize, rewriter);

  insert(LLVM::ConstantOp::create(rewriter, loc, i32Ty, version),
         Field::Version);
  Value one = LLVM::ConstantOp::create(rewriter, loc, i32Ty, 1);
  Value storage =
      LLVM::AllocaOp::create(rewriter, loc, ptrTy, descriptorType, one);
  LLVM::StoreOp::create(rewriter, loc, descriptor, storage);
  return storage;
}

Value mlir::createACCArgumentDescriptor(Operation *mapOp,
                                        Value convertedOperand,
                                        OpenACCSupport &accSupport,
                                        ConversionPatternRewriter &rewriter) {
  Location loc = mapOp->getLoc();
  MLIRContext *context = rewriter.getContext();
  Type i32Ty = rewriter.getI32Type();
  Type ptrTy = LLVM::LLVMPointerType::get(context);
  SmallVector<Value> bounds = acc::getBounds(mapOp);
  DataDescKind descKind = acc::getDataDescKind(mapOp);
  // Only a descriptor stating bounds reads the element size, so it is
  // materialized where it is used, leaving no constant behind for a mapping
  // that needs no descriptor at all.
  auto elementSize = [&] {
    return getElementSize(mapOp, accSupport, rewriter);
  };

  Value var = acc::getVar(mapOp);
  if (var && isa<MemRefType>(var.getType())) {
    auto memrefType = cast<MemRefType>(var.getType());
    return createACCMemRefDescriptorWrapperArg(
        loc, memrefType, convertedOperand, bounds, elementSize(), rewriter);
  }

  if (bitEnumContainsAny(descKind, DataDescKind::cfi)) {
    using Field = AccDataDescCFIField;
    Type descriptorType =
        getDataDescriptorType(context, DataDescriptor::AccDataDescCFI);
    auto version = static_cast<uint32_t>(
        getDataDescriptorKind(DataDescriptor::AccDataDescCFI));
    Value descriptor = LLVM::ZeroOp::create(rewriter, loc, descriptorType);
    Value descriptorStorage = remap(acc::getDesc(mapOp), rewriter);
    if (!descriptorStorage)
      descriptorStorage = convertedOperand;
    descriptor = LLVM::InsertValueOp::create(
        rewriter, loc, descriptorType, descriptor,
        getPointer(loc, descriptorStorage, rewriter),
        ArrayRef<int64_t>{getDataDescriptorFieldIndex(Field::CFIDescriptor)});
    if (!bounds.empty())
      return createACCDataDescriptor(loc, descriptor, descriptorType, version,
                                     bounds, elementSize(), rewriter);

    Value versionValue =
        LLVM::ConstantOp::create(rewriter, loc, i32Ty, version);
    descriptor = LLVM::InsertValueOp::create(
        rewriter, loc, descriptorType, descriptor, versionValue,
        ArrayRef<int64_t>{getDataDescriptorFieldIndex(Field::Version)});
    Value one = LLVM::ConstantOp::create(rewriter, loc, i32Ty, 1);
    Value storage =
        LLVM::AllocaOp::create(rewriter, loc, ptrTy, descriptorType, one);
    LLVM::StoreOp::create(rewriter, loc, descriptor, storage);
    return storage;
  }

  if (!bounds.empty()) {
    Type descriptorType =
        getDataDescriptorType(context, DataDescriptor::AccDataDescGeneric);
    auto version = static_cast<uint32_t>(
        getDataDescriptorKind(DataDescriptor::AccDataDescGeneric));
    Value descriptor = LLVM::ZeroOp::create(rewriter, loc, descriptorType);
    return createACCDataDescriptor(loc, descriptor, descriptorType, version,
                                   bounds, elementSize(), rewriter);
  }
  return LLVM::ZeroOp::create(rewriter, loc, ptrTy);
}

SmallVector<Value> mlir::ACCDataRuntimeArgs::getCallArgs() const {
  return {ident,    flags,    deviceType, argNum,     argBasePtrs, argPtrs,
          argSizes, argTypes, argNames,   argMappers, argDescs};
}

LogicalResult mlir::emitACCDataRuntimeArgs(
    Location loc, ValueRange mappingOperands, ValueRange convertedOperands,
    ConversionPatternRewriter &rewriter, Region &globalSymbolRegion,
    acc::OpenACCSupport &accSupport, const acc::ACCRuntimeCallConfig &config,
    ACCDataRuntimeArgs &runtimeArgs, ACCDataCallKind callKind,
    SymbolTable *symbolTable) {
  if (mappingOperands.size() != convertedOperands.size())
    return failure();

  Type i32Ty = rewriter.getI32Type();
  Type i64Ty = rewriter.getI64Type();
  Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());

  runtimeArgs.ident =
      createIdent(loc, getParentFunctionName(mappingOperands), rewriter,
                  globalSymbolRegion, config, symbolTable);
  runtimeArgs.flags = constantI64(loc, 0, rewriter);
  runtimeArgs.deviceType = constantI64(
      loc, config.getDeviceTypeRuntimeValue(DeviceType::None), rewriter);
  runtimeArgs.argMappers = LLVM::ZeroOp::create(rewriter, loc, ptrTy);

  // An aggregate passed by value is decomposed into one argument per field by
  // the target ABI, so the runtime is told about one object per field as well.
  // Each of them keeps the map entry of the aggregate it came from.
  SmallVector<std::pair<Value, Value>> mappedObjects;
  for (auto [operand, converted] :
       llvm::zip_equal(mappingOperands, convertedOperands)) {
    auto structType = dyn_cast<LLVM::LLVMStructType>(converted.getType());
    if (structType && !structType.isIdentified() &&
        structType.getBody().size() > 1) {
      for (unsigned field = 0, fields = structType.getBody().size();
           field != fields; ++field)
        mappedObjects.emplace_back(
            operand,
            LLVM::ExtractValueOp::create(rewriter, loc, converted, field));
      continue;
    }
    mappedObjects.emplace_back(operand, converted);
  }

  runtimeArgs.argNum =
      LLVM::ConstantOp::create(rewriter, loc, i32Ty, mappedObjects.size());

  SmallVector<Value> bases;
  SmallVector<Value> pointers;
  SmallVector<Value> sizes;
  SmallVector<Value> types;
  SmallVector<Value> names;
  SmallVector<Value> descriptors;
  for (auto [operand, converted] : mappedObjects) {
    Operation *mapOp = operand.getDefiningOp();
    if (!mapOp || !isa<MapInfoOp, ACC_DATA_ENTRY_OPS>(mapOp))
      return failure();
    std::optional<MapFlags> mapFlags = computePackedMapFlags(mapOp);
    if (!mapFlags)
      return failure();
    *mapFlags = config.postProcessMapFlags(mapOp, *mapFlags);
    LLVM_DEBUG(llvm::dbgs() << "mapping " << *mapOp << "\n  as "
                            << config.formatMapFlags(*mapFlags) << "\n");

    Location mapLoc = mapOp->getLoc();
    pointers.push_back(getMappedObjectPointer(mapLoc, operand, converted,
                                              *mapFlags, rewriter));

    Value base;
    if (Value attach = acc::getVarPtrPtr(mapOp))
      base = remap(attach, rewriter);
    else if (bitEnumContainsAny(*mapFlags, MapFlags::ptr_and_obj))
      base = remap(acc::getDesc(mapOp), rewriter);
    // An object that only lives on the device has no host address to state as
    // its base.
    if (bitEnumContainsAny(*mapFlags, MapFlags::device_resident))
      base = {};
    bases.push_back(getPointer(mapLoc, base, rewriter));

    // A local object that lives on the device is torn down when the region
    // that declared it ends, which the runtime only does for a mapping that
    // states it is deleted.
    if (callKind == ACCDataCallKind::DataExit &&
        bitEnumContainsAny(*mapFlags, MapFlags::device_resident)) {
      Value var = acc::getVar(mapOp);
      if (var &&
          !isa_and_nonnull<AddressOfGlobalOpInterface>(var.getDefiningOp()))
        *mapFlags = *mapFlags | MapFlags::delete_;
    }

    sizes.push_back(
        getMapSize(mapOp, converted, *mapFlags, accSupport, rewriter));
    types.push_back(constantI64(
        mapLoc, config.getMapFlagsRuntimeValue(*mapFlags), rewriter));

    // The runtime resolves an object that lives on the device against the
    // symbols of the binary by this name, so it is the name the object is
    // emitted under rather than the one the source spells.
    VariableNameConfig nameConfig;
    nameConfig.preferDemangledName = false;
    std::string name = accSupport.getVariableName(operand, nameConfig);
    if (name.empty()) {
      names.push_back(LLVM::ZeroOp::create(rewriter, mapLoc, ptrTy));
    } else {
      names.push_back(getOrCreateGlobalString(
          mapLoc, rewriter, getInternalGlobalName("var_name", name), name,
          globalSymbolRegion, symbolTable));
    }
    descriptors.push_back(
        createACCArgumentDescriptor(mapOp, converted, accSupport, rewriter));
  }

  runtimeArgs.argBasePtrs = createACCDataArray(loc, ptrTy, bases, rewriter);
  runtimeArgs.argPtrs = createACCDataArray(loc, ptrTy, pointers, rewriter);
  runtimeArgs.argSizes = createACCDataArray(loc, i64Ty, sizes, rewriter);
  runtimeArgs.argTypes = createACCDataArray(loc, i64Ty, types, rewriter);
  runtimeArgs.argNames = createACCDataArray(loc, ptrTy, names, rewriter);
  runtimeArgs.argDescs = createACCDataArray(loc, ptrTy, descriptors, rewriter);
  return success();
}
