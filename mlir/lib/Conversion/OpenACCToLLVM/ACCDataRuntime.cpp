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
#include "mlir/Dialect/OpenACC/OpenACCUtils.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsCG.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsType.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

#define DEBUG_TYPE "acc-data-runtime"

using namespace mlir;
using namespace mlir::acc;

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

void mlir::createACCMappedObjectArrays(Location loc,
                                       ArrayRef<ACCMappedObjectArgs> objects,
                                       ACCDataRuntimeArgs &runtimeArgs,
                                       RewriterBase &rewriter) {
  Type i32Ty = rewriter.getI32Type();
  Type i64Ty = rewriter.getI64Type();
  Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
  auto createArray = [&](Type elementType, Value ACCMappedObjectArgs::*field) {
    SmallVector<Value> values =
        llvm::map_to_vector(objects, [&](const ACCMappedObjectArgs &object) {
          return object.*field;
        });
    return createACCDataArray(loc, elementType, values, rewriter);
  };
  // The data-call emitter states the count before the object values.
  if (!runtimeArgs.argNum)
    runtimeArgs.argNum =
        LLVM::ConstantOp::create(rewriter, loc, i32Ty, objects.size());
  runtimeArgs.argBasePtrs = createArray(ptrTy, &ACCMappedObjectArgs::base);
  runtimeArgs.argPtrs = createArray(ptrTy, &ACCMappedObjectArgs::pointer);
  runtimeArgs.argSizes = createArray(i64Ty, &ACCMappedObjectArgs::size);
  runtimeArgs.argTypes = createArray(i64Ty, &ACCMappedObjectArgs::type);
  runtimeArgs.argNames = createArray(ptrTy, &ACCMappedObjectArgs::name);
  if (!runtimeArgs.argMappers)
    runtimeArgs.argMappers = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
  runtimeArgs.argDescs = createArray(ptrTy, &ACCMappedObjectArgs::descriptor);
}

/// Privatized storage only ever exists on the device, so there is no host
/// address to give. A device address names storage the runtime must not look
/// up, and an object mapped by value has no address at all, so both are handed
/// over in a slot the runtime reads them from. A pointer that no clause maps
/// but the device can already access is a device address as well.
Value mlir::createACCMappedObjectPointer(Location loc, Value operand,
                                         Value converted, MapFlags mapFlags,
                                         ConversionPatternRewriter &rewriter) {
  Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
  if (isa<PrivateType>(operand.getType()))
    return LLVM::ZeroOp::create(rewriter, loc, ptrTy);
  if (converted && converted.getType() == ptrTy &&
      (bitEnumContainsAny(mapFlags, MapFlags::devptr) ||
       (!isDataEntryOp(operand.getDefiningOp()) &&
        isDeviceAccessibleValue(operand))))
    return getIndirectPointer(loc, converted, rewriter);
  return getPointer(loc, converted, rewriter);
}

Value mlir::createACCMappedObjectBase(Location loc, Operation *mapOp,
                                      MapFlags mapFlags,
                                      ConversionPatternRewriter &rewriter) {
  Value base;
  if (mapOp) {
    if (Value attach = acc::getVarPtrPtr(mapOp))
      base = remapOrSelf(attach, rewriter);
    else if (bitEnumContainsAny(mapFlags, MapFlags::ptr_and_obj))
      base = remapOrSelf(acc::getDesc(mapOp), rewriter);
  }
  return getPointer(loc, base, rewriter);
}

Value mlir::createACCMappedObjectName(Location loc, Value operand,
                                      OpenACCSupport &accSupport,
                                      Region &globalSymbolRegion,
                                      SymbolTable *symbolTable,
                                      ConversionPatternRewriter &rewriter) {
  // The runtime resolves an object that lives on the device against the
  // symbols of the binary by this name, so it is the name the object is
  // emitted under rather than the one the source spells.
  VariableNameConfig nameConfig;
  nameConfig.preferDemangledName = false;
  std::string name = accSupport.getVariableName(operand, nameConfig);
  if (name.empty())
    return LLVM::ZeroOp::create(
        rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext()));
  return getOrCreateGlobalString(loc, rewriter,
                                 getInternalGlobalName("var_name", name), name,
                                 globalSymbolRegion, symbolTable);
}

static std::optional<MapFlags> computePackedMapFlags(Operation *mapOp) {
  if (auto mapInfo = dyn_cast<MapInfoOp>(mapOp))
    return mapInfo.getMapFlags();
  if (isa<ACC_DATA_ENTRY_OPS>(mapOp))
    return computeDataClauseMapFlags(mapOp, hasAttachPoint(mapOp));
  return std::nullopt;
}

FailureOr<Value>
mlir::createACCMappedObjectSize(Location loc, Value operand, Value converted,
                                MapFlags mapFlags, OpenACCSupport &accSupport,
                                ConversionPatternRewriter &rewriter) {
  Operation *scope = rewriter.getInsertionBlock()->getParentOp();
  auto module = dyn_cast<ModuleOp>(scope);
  if (!module)
    module = scope->getParentOfType<ModuleOp>();
  // An object mapped by value is passed as the argument itself, so its size is
  // the size of what is passed and not of the object it was read from. The two
  // differ when the argument holds one field of a decomposed aggregate.
  if (bitEnumContainsAny(mapFlags, MapFlags::literal) && converted) {
    auto sizeAndAlignment =
        acc::getTypeSizeAndAlignment(converted.getType(), module, &accSupport);
    if (!sizeAndAlignment) {
      std::string message;
      llvm::raw_string_ostream os(message);
      os << "unhandled type size computation for type: " << converted.getType();
      (void)accSupport.emitNYI(loc, message);
      return failure();
    }
    return constantI64(loc, sizeAndAlignment->first.getFixedValue(), rewriter);
  }
  Operation *mapOp = operand.getDefiningOp();
  if (!isDataEntryOp(mapOp))
    return constantI64(loc, 0, rewriter);
  if (Value size = acc::getMapSize(mapOp))
    return castToI64(loc, remapOrSelf(size, rewriter), rewriter);

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
  return castToI64(loc, remapOrSelf(value, rewriter), rewriter);
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
    Value descriptorStorage = remapOrSelf(acc::getDesc(mapOp), rewriter);
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
    ACCDataRuntimeArgs &runtimeArgs, const ACCDataCallOptions &options,
    SymbolTable *symbolTable) {
  if (mappingOperands.size() != convertedOperands.size())
    return failure();

  runtimeArgs.ident =
      createIdent(loc, getParentFunctionName(mappingOperands), rewriter,
                  globalSymbolRegion, config, symbolTable);
  runtimeArgs.flags = constantI64(loc, options.flags, rewriter);
  runtimeArgs.deviceType = constantI64(
      loc, config.getDeviceTypeRuntimeValue(DeviceType::None), rewriter);
  runtimeArgs.argMappers = LLVM::ZeroOp::create(
      rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext()));

  // An aggregate passed by value is decomposed into one argument per field by
  // the target ABI, so the runtime is told about one object per field as well.
  // Each of them keeps the map entry of the aggregate it came from. The device
  // address of an aggregate is returned in the slot of its first field.
  struct MappedObject {
    Value operand;
    Value converted;
    bool firstField;
  };
  SmallVector<MappedObject> mappedObjects;
  for (auto [operand, converted] :
       llvm::zip_equal(mappingOperands, convertedOperands)) {
    auto structType = dyn_cast<LLVM::LLVMStructType>(converted.getType());
    if (structType && !structType.isIdentified() &&
        structType.getBody().size() > 1) {
      for (unsigned field = 0, fields = structType.getBody().size();
           field != fields; ++field)
        mappedObjects.push_back(
            {operand,
             LLVM::ExtractValueOp::create(rewriter, loc, converted, field),
             field == 0});
      continue;
    }
    mappedObjects.push_back({operand, converted, true});
  }

  // The count precedes the values of the objects, which is where a decomposed
  // aggregate states how many fields it became.
  runtimeArgs.argNum = LLVM::ConstantOp::create(
      rewriter, loc, rewriter.getI32Type(), mappedObjects.size());

  constexpr unsigned noSlot = ~0u;
  runtimeArgs.returnAddressSlots.assign(options.returnAddressOperands.size(),
                                        noSlot);
  SmallVector<ACCMappedObjectArgs> objects;
  for (auto [slot, object] : llvm::enumerate(mappedObjects)) {
    auto [operand, converted, firstField] = object;
    Operation *mapOp = operand.getDefiningOp();
    if (!isDataEntryOp(mapOp))
      return failure();
    std::optional<MapFlags> mapFlags = computePackedMapFlags(mapOp);
    if (!mapFlags)
      return failure();
    *mapFlags = config.postProcessMapFlags(mapOp, *mapFlags);
    LLVM_DEBUG(llvm::dbgs() << "mapping " << *mapOp << "\n  as "
                            << config.formatMapFlags(*mapFlags) << "\n");

    Location mapLoc = mapOp->getLoc();
    ACCMappedObjectArgs args;
    args.pointer = createACCMappedObjectPointer(mapLoc, operand, converted,
                                                *mapFlags, rewriter);
    // An object that only lives on the device has no host address to state as
    // its base.
    args.base =
        bitEnumContainsAny(*mapFlags, MapFlags::device_resident)
            ? getPointer(mapLoc, Value(), rewriter)
            : createACCMappedObjectBase(mapLoc, mapOp, *mapFlags, rewriter);

    // A local object that lives on the device is torn down when the region
    // that declared it ends, which the runtime only does for a mapping that
    // states it is deleted.
    if (options.callKind == ACCDataCallKind::DataExit &&
        bitEnumContainsAny(*mapFlags, MapFlags::device_resident)) {
      Value var = acc::getVar(mapOp);
      if (var &&
          !isa_and_nonnull<AddressOfGlobalOpInterface>(var.getDefiningOp()))
        *mapFlags = *mapFlags | MapFlags::delete_;
    }

    if (firstField) {
      for (auto [index, returned] :
           llvm::enumerate(options.returnAddressOperands)) {
        if (returned != operand)
          continue;
        *mapFlags = *mapFlags | MapFlags::return_param;
        runtimeArgs.returnAddressSlots[index] = slot;
      }
    }
    int64_t mapType = config.getMapFlagsRuntimeValue(*mapFlags);

    FailureOr<Value> size = createACCMappedObjectSize(
        mapLoc, operand, converted, *mapFlags, accSupport, rewriter);
    if (failed(size))
      return failure();
    args.size = *size;
    args.type = constantI64(mapLoc, mapType, rewriter);
    args.name = createACCMappedObjectName(
        mapLoc, operand, accSupport, globalSymbolRegion, symbolTable, rewriter);
    args.descriptor =
        createACCArgumentDescriptor(mapOp, converted, accSupport, rewriter);
    objects.push_back(args);
  }
  if (llvm::is_contained(runtimeArgs.returnAddressSlots, noSlot))
    return failure();

  createACCMappedObjectArrays(loc, objects, runtimeArgs, rewriter);
  return success();
}

FailureOr<SmallVector<Value>> mlir::emitACCDataRuntimeCall(
    Location loc, RuntimeFunction fn, ValueRange mappingOperands,
    ValueRange convertedOperands, ConversionPatternRewriter &rewriter,
    OpenACCSupport &accSupport, Region &globalSymbolRegion,
    SymbolTable &symbolTable, const ACCRuntimeCallConfig &config,
    ArrayRef<Value> trailingArguments, const ACCDataCallOptions &options) {
  SmallVector<Value> returnedAddresses;
  if (mappingOperands.empty())
    return returnedAddresses;

  ACCDataRuntimeArgs runtimeArgs;
  if (failed(emitACCDataRuntimeArgs(
          loc, mappingOperands, convertedOperands, rewriter, globalSymbolRegion,
          accSupport, config, runtimeArgs, options, &symbolTable)))
    return rewriter.notifyMatchFailure(
        mappingOperands.front().getDefiningOp(),
        "unsupported OpenACC data-clause operand");

  SmallVector<Value> arguments = runtimeArgs.getCallArgs();
  arguments.append(trailingArguments.begin(), trailingArguments.end());
  if (failed(createRuntimeCall(loc, rewriter, globalSymbolRegion, symbolTable,
                               fn, config, arguments)))
    return failure();

  Type i32Ty = rewriter.getI32Type();
  Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
  for (unsigned slot : runtimeArgs.returnAddressSlots) {
    Value index = LLVM::ConstantOp::create(rewriter, loc, i32Ty, slot);
    Value element =
        LLVM::GEPOp::create(rewriter, loc, ptrTy, ptrTy,
                            runtimeArgs.argBasePtrs, ValueRange{index});
    returnedAddresses.push_back(
        LLVM::LoadOp::create(rewriter, loc, ptrTy, element));
  }
  return returnedAddresses;
}
