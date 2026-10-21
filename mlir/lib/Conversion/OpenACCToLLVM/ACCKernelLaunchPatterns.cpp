//===- ACCKernelLaunchPatterns.cpp - Kernel launch to LLVM ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lower acc.kernel_environment and the gpu.launch_func operations in its body
// together. The environment establishes one implicit data region. Its entry
// call returns the device addresses that every contained kernel launch uses.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/OpenACCToLLVM/ACCDataRuntime.h"
#include "mlir/Conversion/OpenACCToLLVM/ACCToLLVM.h"
#include "mlir/Conversion/OpenACCToLLVM/ACCToLLVMUtils.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsCG.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsType.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <array>

using namespace mlir;
using namespace mlir::acc;

static bool isDevicePtrMapping(Value operand) {
  Operation *mapOp = operand.getDefiningOp();
  if (!mapOp)
    return false;
  if (std::optional<MapFlags> flags = getMapFlags(mapOp))
    return bitEnumContainsAny(*flags, MapFlags::devptr);
  if (std::optional<DataClause> clause = getDataClause(mapOp))
    return *clause == DataClause::acc_deviceptr;
  return false;
}

static bool isDataEntryInsideEnvironment(KernelEnvironmentOp environment,
                                         Value operand) {
  Operation *mapOp = operand.getDefiningOp();
  return isDataEntryOp(mapOp) && environment->isAncestor(mapOp);
}

/// The flags of a data clause operation that maps a kernel argument. A kernel
/// argument only looks up the mapping its clause established, so the clauses
/// that establish or tear down something else contribute no flag for it.
static MapFlags getKernelDataClauseMapFlags(Operation *entryOp) {
  bool ptrAndObj = hasAttachPoint(entryOp);
  MapFlags ptrAndObjFlag = ptrAndObj ? MapFlags::ptr_and_obj : MapFlags::none;
  std::optional<DataClause> clause = getDataClause(entryOp);
  if (clause == DataClause::acc_declare_device_resident)
    return MapFlags::present | ptrAndObjFlag;
  if (clause == DataClause::acc_attach || clause == DataClause::acc_detach)
    return ptrAndObjFlag;
  return bitEnumClear(computeDataClauseMapFlags(entryOp, ptrAndObj),
                      MapFlags::device_resident);
}

/// The flags a kernel argument is passed with. An operand the environment
/// already mapped is passed as the device address its entry call returned. An
/// operand that no clause maps is passed by value.
static MapFlags getKernelArgMapFlags(Value operand, bool alreadyMapped,
                                     const ACCRuntimeCallConfig &config) {
  Operation *mapOp = operand.getDefiningOp();
  if (!isDataEntryOp(mapOp))
    return mapOp ? config.postProcessMapFlags(mapOp, MapFlags::literal)
                 : MapFlags::literal;

  MapFlags flags;
  if (alreadyMapped)
    flags = MapFlags::devptr | MapFlags::literal;
  else if (auto mapInfo = dyn_cast<MapInfoOp>(mapOp))
    flags = mapInfo.getMapFlags();
  else
    flags = getKernelDataClauseMapFlags(mapOp);
  if (getImplicitFlag(mapOp))
    flags = flags | MapFlags::implicit;
  return config.postProcessMapFlags(mapOp, flags);
}

/// The descriptor of a kernel argument. The device address the environment
/// returned is passed as is, since the runtime would map an address that
/// carries a descriptor again.
static FailureOr<Value>
getKernelArgDescriptor(Location loc, Value operand, Value converted,
                       bool alreadyMapped, OpenACCSupport &accSupport,
                       ConversionPatternRewriter &rewriter) {
  Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
  Operation *mapOp = operand.getDefiningOp();
  if (alreadyMapped)
    return Value(LLVM::ZeroOp::create(rewriter, loc, ptrTy));
  if (isDataEntryOp(mapOp))
    return createACCArgumentDescriptor(mapOp, converted, accSupport, rewriter);
  if (auto memrefType = dyn_cast<MemRefType>(operand.getType())) {
    ModuleOp module = rewriter.getInsertionBlock()
                          ->getParentOp()
                          ->getParentOfType<ModuleOp>();
    auto sizeAndAlignment = acc::getTypeSizeAndAlignment(
        memrefType.getElementType(), module, &accSupport);
    if (!sizeAndAlignment) {
      std::string message;
      llvm::raw_string_ostream os(message);
      os << "unhandled type size computation for type: "
         << memrefType.getElementType();
      (void)accSupport.emitNYI(loc, message);
      return failure();
    }
    return createACCMemRefDescriptorWrapperArg(
        loc, memrefType, converted, /*bounds=*/{},
        constantI64(loc, sizeAndAlignment->first.getFixedValue(), rewriter),
        rewriter);
  }
  return Value(LLVM::ZeroOp::create(rewriter, loc, ptrTy));
}

/// Emits the argument arrays of the kernel launched by \p launch, one entry per
/// kernel operand. Each operand is one parameter of the kernel.
static FailureOr<ACCDataRuntimeArgs> emitKernelRuntimeArgs(
    gpu::LaunchFuncOp launch, gpu::LaunchFuncOpAdaptor adaptor,
    ValueRange alreadyMappedOperands, ConversionPatternRewriter &rewriter,
    OpenACCSupport &accSupport, Region &globalSymbolRegion,
    SymbolTable &symbolTable, const ACCRuntimeCallConfig &config) {
  Location loc = launch.getLoc();
  SmallVector<ACCMappedObjectArgs> objects;
  for (auto [operand, converted] : llvm::zip_equal(
           launch.getKernelOperands(), adaptor.getKernelOperands())) {
    bool alreadyMapped = llvm::is_contained(alreadyMappedOperands, operand);
    MapFlags mapFlags = getKernelArgMapFlags(operand, alreadyMapped, config);
    Operation *mapOp = operand.getDefiningOp();

    ACCMappedObjectArgs args;
    // A kernel argument states the base of the mapping it looks up, which for
    // an object the declare directive made resident is that of its entry.
    args.base = createACCMappedObjectBase(
        loc, isDataEntryOp(mapOp) ? mapOp : nullptr, mapFlags, rewriter);
    args.pointer = createACCMappedObjectPointer(loc, operand, converted,
                                                mapFlags, rewriter);
    FailureOr<Value> size = createACCMappedObjectSize(
        loc, operand, converted, mapFlags, accSupport, rewriter);
    if (failed(size))
      return failure();
    args.size = *size;
    args.type =
        constantI64(loc, config.getMapFlagsRuntimeValue(mapFlags), rewriter);
    args.name = createACCMappedObjectName(
        loc, operand, accSupport, globalSymbolRegion, &symbolTable, rewriter);
    FailureOr<Value> descriptor = getKernelArgDescriptor(
        loc, operand, converted, alreadyMapped, accSupport, rewriter);
    if (failed(descriptor))
      return failure();
    args.descriptor = *descriptor;
    objects.push_back(args);
  }

  ACCDataRuntimeArgs runtimeArgs;
  createACCMappedObjectArrays(loc, objects, runtimeArgs, rewriter);
  return runtimeArgs;
}

static FailureOr<Value> createKernelArgumentStruct(
    gpu::LaunchFuncOp launch, gpu::LaunchFuncOpAdaptor adaptor,
    ValueRange alreadyMappedOperands, ConversionPatternRewriter &rewriter,
    OpenACCSupport &accSupport, Region &globalSymbolRegion,
    SymbolTable &symbolTable, const ACCRuntimeCallConfig &config) {
  FailureOr<ACCDataRuntimeArgs> emitted = emitKernelRuntimeArgs(
      launch, adaptor, alreadyMappedOperands, rewriter, accSupport,
      globalSymbolRegion, symbolTable, config);
  if (failed(emitted))
    return failure();
  ACCDataRuntimeArgs runtimeArgs = *emitted;

  Location loc = launch.getLoc();
  MLIRContext *context = rewriter.getContext();
  Type i32Ty = rewriter.getI32Type();
  Type ptrTy = LLVM::LLVMPointerType::get(context);
  Type argsType = getKernelArgumentType(context);
  Value args = LLVM::ZeroOp::create(rewriter, loc, argsType);
  auto insert = [&](Value value, KernelArgumentField field) {
    args = LLVM::InsertValueOp::create(
        rewriter, loc, argsType, args, value,
        ArrayRef<int64_t>{getKernelArgumentFieldIndex(field)});
  };
  insert(runtimeArgs.argNum, KernelArgumentField::ArgNum);
  insert(runtimeArgs.argBasePtrs, KernelArgumentField::ArgBasePtrs);
  insert(runtimeArgs.argPtrs, KernelArgumentField::ArgPtrs);
  insert(runtimeArgs.argSizes, KernelArgumentField::ArgSizes);
  insert(runtimeArgs.argTypes, KernelArgumentField::ArgTypes);
  insert(runtimeArgs.argNames, KernelArgumentField::ArgNames);
  insert(runtimeArgs.argMappers, KernelArgumentField::ArgMappers);
  insert(runtimeArgs.argDescs, KernelArgumentField::ArgDescs);

  // The vector length is the first block dimension. The worker count is the
  // product of the remaining ones.
  Value blockY = castToI64(loc, adaptor.getBlockSizeY(), rewriter);
  Value blockZ = castToI64(loc, adaptor.getBlockSizeZ(), rewriter);
  insert(LLVM::MulOp::create(rewriter, loc, blockY, blockZ),
         KernelArgumentField::NumWorkers);
  insert(castToI64(loc, adaptor.getBlockSizeX(), rewriter),
         KernelArgumentField::VectorLength);
  Value dynamicSharedMemorySize =
      adaptor.getDynamicSharedMemorySize()
          ? castToI64(loc, adaptor.getDynamicSharedMemorySize(), rewriter)
          : constantI64(loc, 0, rewriter);
  insert(dynamicSharedMemorySize, KernelArgumentField::SmemSize);

  // The grid sizes are the entries of the array in the NumGangs field.
  int64_t numGangsField =
      getKernelArgumentFieldIndex(KernelArgumentField::NumGangs);
  std::array<Value, 3> gridSizes = {
      adaptor.getGridSizeX(), adaptor.getGridSizeY(), adaptor.getGridSizeZ()};
  for (auto [dimension, gridSize] : llvm::enumerate(gridSizes))
    args = LLVM::InsertValueOp::create(
        rewriter, loc, argsType, args, castToI64(loc, gridSize, rewriter),
        ArrayRef<int64_t>{numGangsField, static_cast<int64_t>(dimension)});

  Value one = LLVM::ConstantOp::create(rewriter, loc, i32Ty, 1);
  Value argsPtr = LLVM::AllocaOp::create(rewriter, loc, ptrTy, argsType, one);
  LLVM::StoreOp::create(rewriter, loc, args, argsPtr);
  return argsPtr;
}

static FailureOr<Value> getKernelPointer(gpu::LaunchFuncOp launch,
                                         ConversionPatternRewriter &rewriter,
                                         Region &globalSymbolRegion,
                                         SymbolTable &symbolTable) {
  Location loc = launch.getLoc();
  Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
  StringRef kernelName = launch.getKernel().getLeafReference().getValue();
  if (auto function = symbolTable.lookup<LLVM::LLVMFuncOp>(kernelName))
    return LLVM::AddressOfOp::create(rewriter, loc, ptrTy, function.getName())
        .getResult();
  if (auto function = symbolTable.lookup<func::FuncOp>(kernelName))
    return LLVM::AddressOfOp::create(rewriter, loc, ptrTy, function.getName())
        .getResult();

  auto gpuFunction = SymbolTable::lookupNearestSymbolFrom<gpu::GPUFuncOp>(
      launch, launch.getKernel());
  if (!gpuFunction)
    return failure();

  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&globalSymbolRegion.front());
    auto functionType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(rewriter.getContext()), {});
    auto declaration =
        LLVM::LLVMFuncOp::create(rewriter, loc, kernelName, functionType);
    symbolTable.insert(declaration);
  }
  return LLVM::AddressOfOp::create(rewriter, loc, ptrTy, kernelName)
      .getResult();
}

static void declareLibc(SymbolTable &symbolTable, Region &globalSymbolRegion,
                        Location loc, StringRef name, Type result,
                        ArrayRef<Type> arguments) {
  if (symbolTable.lookup<LLVM::LLVMFuncOp>(name))
    return;
  OpBuilder builder = OpBuilder::atBlockEnd(&globalSymbolRegion.front());
  auto functionType = LLVM::LLVMFunctionType::get(result, arguments);
  symbolTable.insert(
      LLVM::LLVMFuncOp::create(builder, loc, name, functionType));
}

/// The kernel entry point reports a status other than success when the kernel
/// could not run on the device. The program is then meant to run the region on
/// the host, which is not implemented, so the failure is reported and the
/// program stops.
static void emitHostFallbackNotImplemented(Location loc, Value status,
                                           ConversionPatternRewriter &rewriter,
                                           Region &globalSymbolRegion,
                                           SymbolTable &symbolTable) {
  MLIRContext *context = rewriter.getContext();
  Type i32Ty = rewriter.getI32Type();
  Type ptrTy = LLVM::LLVMPointerType::get(context);
  Type voidTy = LLVM::LLVMVoidType::get(context);

  Block *preBlock = rewriter.getInsertionBlock();
  Block *continueBlock =
      rewriter.splitBlock(preBlock, rewriter.getInsertionPoint());
  Block *errorBlock =
      rewriter.createBlock(preBlock->getParent(), continueBlock->getIterator());
  rewriter.setInsertionPointToEnd(preBlock);
  Value success = LLVM::ConstantOp::create(
      rewriter, loc, i32Ty, static_cast<int32_t>(RuntimeResult::Success));
  Value failed = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::ne,
                                      status, success);
  LLVM::CondBrOp::create(rewriter, loc, failed, errorBlock, ValueRange{},
                         continueBlock, ValueRange{});

  // TODO: Runtime API should provide a mechanism to report this error.
  rewriter.setInsertionPointToStart(errorBlock);
  declareLibc(symbolTable, globalSymbolRegion, loc, "puts", i32Ty, {ptrTy});
  declareLibc(symbolTable, globalSymbolRegion, loc, "fflush", i32Ty, {ptrTy});
  declareLibc(symbolTable, globalSymbolRegion, loc, "abort", voidTy, {});
  Value message = getOrCreateGlobalString(
      loc, rewriter, getInternalGlobalName("message", "host_fallback_nyi"),
      "host fallback execution is not implemented\n", globalSymbolRegion,
      &symbolTable);
  LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                       SymbolRefAttr::get(context, "puts"),
                       ValueRange{message});
  Value nullFile = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
  LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                       SymbolRefAttr::get(context, "fflush"),
                       ValueRange{nullFile});
  LLVM::CallOp::create(rewriter, loc, TypeRange{},
                       SymbolRefAttr::get(context, "abort"), ValueRange{});
  LLVM::UnreachableOp::create(rewriter, loc);
  rewriter.setInsertionPointToStart(continueBlock);
}

static LogicalResult
emitKernelCall(gpu::LaunchFuncOp launch, gpu::LaunchFuncOpAdaptor adaptor,
               Value asyncQueue, ValueRange alreadyMappedOperands,
               ConversionPatternRewriter &rewriter, OpenACCSupport &accSupport,
               Region &globalSymbolRegion, SymbolTable &symbolTable,
               const ACCRuntimeCallConfig &config) {
  Location loc = launch.getLoc();
  FailureOr<Value> kernelPointer =
      getKernelPointer(launch, rewriter, globalSymbolRegion, symbolTable);
  if (failed(kernelPointer)) {
    (void)accSupport.emitNYI(loc, "undefined kernel");
    return failure();
  }
  FailureOr<Value> arguments = createKernelArgumentStruct(
      launch, adaptor, alreadyMappedOperands, rewriter, accSupport,
      globalSymbolRegion, symbolTable, config);
  if (failed(arguments))
    return failure();

  StringRef kernelName = launch.getKernel().getLeafReference().getValue();
  Value kernelNamePtr = getOrCreateGlobalString(
      loc, rewriter, getInternalGlobalName("kernel_name", kernelName),
      kernelName, globalSymbolRegion, &symbolTable);
  Value ident = createIdent(loc, getParentFunctionName(launch), rewriter,
                            globalSymbolRegion, config, &symbolTable);
  // The kernel call uses the explicit data-region flag.
  Value flags = constantI64(loc, static_cast<int64_t>(ACCDataRegionFlags::None),
                            rewriter);
  // The launch is made for the default device type, as the data calls of the
  // environment are.
  Value deviceType = constantI64(
      loc, config.getDeviceTypeRuntimeValue(DeviceType::None), rewriter);
  Value binaryDescriptor = config.createKernelBinaryDescriptor(loc, rewriter);
  FailureOr<LLVM::CallOp> call =
      createRuntimeCall(loc, rewriter, globalSymbolRegion, symbolTable,
                        RuntimeFunction::ACCRTL_tgt_acc_kernel, config,
                        {ident, *kernelPointer, flags, deviceType, *arguments,
                         asyncQueue, kernelNamePtr, binaryDescriptor});
  if (failed(call))
    return failure();
  emitHostFallbackNotImplemented(loc, call->getResult(), rewriter,
                                 globalSymbolRegion, symbolTable);
  return success();
}

namespace {
struct KernelEnvironmentLowering
    : public ConvertOpToLLVMPattern<KernelEnvironmentOp> {
  KernelEnvironmentLowering(const LLVMTypeConverter &converter,
                            OpenACCSupport &accSupport,
                            Region &globalSymbolRegion,
                            SymbolTable &symbolTable,
                            const ACCRuntimeCallConfig &config,
                            DeviceType clauseDeviceType,
                            PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<KernelEnvironmentOp>(converter, benefit),
        accSupport(accSupport), globalSymbolRegion(globalSymbolRegion),
        symbolTable(symbolTable), config(config),
        clauseDeviceType(clauseDeviceType) {}

  LogicalResult
  matchAndRewrite(KernelEnvironmentOp environment, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange mappingOperands = environment.getDataClauseOperands();
    ValueRange convertedMappingOperands = adaptor.getDataClauseOperands();
    Location beginLoc = mappingOperands.empty()
                            ? environment.getLoc()
                            : mappingOperands.front().getLoc();
    Location endLoc = getMappingExitLoc(mappingOperands).value_or(beginLoc);
    Value asyncQueue =
        getAsyncQueue(environment, clauseDeviceType, rewriter, config);
    if (failed(emitWaitClause(environment, clauseDeviceType, asyncQueue,
                              rewriter, accSupport, globalSymbolRegion,
                              symbolTable, config)))
      return failure();

    SmallVector<Value> alreadyMappedOperands;
    for (Value operand : mappingOperands)
      if (!isDevicePtrMapping(operand))
        alreadyMappedOperands.push_back(operand);

    ACCDataCallOptions beginOptions;
    beginOptions.flags = static_cast<int64_t>(ACCDataRegionFlags::Implicit);
    beginOptions.returnAddressOperands = alreadyMappedOperands;
    FailureOr<SmallVector<Value>> devicePointers = emitACCDataRuntimeCall(
        beginLoc, RuntimeFunction::ACCRTL_tgt_acc_data_begin, mappingOperands,
        convertedMappingOperands, rewriter, accSupport, globalSymbolRegion,
        symbolTable, config, {asyncQueue}, beginOptions);
    if (failed(devicePointers))
      return failure();

    llvm::DenseMap<Value, Value> deviceAddress;
    for (auto [operand, pointer] :
         llvm::zip_equal(alreadyMappedOperands, *devicePointers))
      deviceAddress[operand] = pointer;

    SmallVector<gpu::LaunchFuncOp> launches;
    environment.walk(
        [&](gpu::LaunchFuncOp launch) { launches.push_back(launch); });
    for (gpu::LaunchFuncOp launch : launches) {
      SmallVector<Value> converted;
      converted.reserve(launch->getNumOperands());
      for (Value operand : launch->getOperands()) {
        Value value;
        // A data entry in the region is mapped by this kernel, so it is passed
        // as the variable of its clause.
        if (isDataEntryInsideEnvironment(environment, operand))
          value = remapOrSelf(getVar(operand.getDefiningOp()), rewriter);
        else
          value = deviceAddress.lookup(operand);
        if (!value)
          value = rewriter.getRemappedValue(operand);
        if (!value)
          return rewriter.notifyMatchFailure(launch,
                                             "unconverted kernel operand");
        converted.push_back(value);
      }

      gpu::LaunchFuncOpAdaptor launchAdaptor(converted, launch);
      Value queue = launchAdaptor.getAsyncObject();
      queue = queue ? castToI64(launch.getLoc(), queue, rewriter) : asyncQueue;
      rewriter.setInsertionPoint(launch);
      if (failed(emitKernelCall(launch, launchAdaptor, queue,
                                alreadyMappedOperands, rewriter, accSupport,
                                globalSymbolRegion, symbolTable, config)))
        return failure();
      rewriter.eraseOp(launch);
    }

    rewriter.setInsertionPointAfter(environment);
    ACCDataCallOptions endOptions;
    endOptions.callKind = ACCDataCallKind::DataExit;
    endOptions.flags = static_cast<int64_t>(ACCDataRegionFlags::Implicit);
    if (failed(emitACCDataRuntimeCall(
            endLoc, RuntimeFunction::ACCRTL_tgt_acc_data_end, mappingOperands,
            convertedMappingOperands, rewriter, accSupport, globalSymbolRegion,
            symbolTable, config, {asyncQueue}, endOptions)))
      return failure();

    spliceConstructRegion(environment, environment.getRegion(), rewriter);
    rewriter.eraseOp(environment);
    return success();
  }

  OpenACCSupport &accSupport;
  Region &globalSymbolRegion;
  SymbolTable &symbolTable;
  ACCRuntimeCallConfig config;
  DeviceType clauseDeviceType;
};
} // namespace

void mlir::configureACCKernelLaunchConversionLegality(
    ConversionTarget &target) {
  target.addIllegalOp<KernelEnvironmentOp>();
}

void mlir::populateACCKernelLaunchPatterns(LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns,
                                           OpenACCSupport &accSupport,
                                           Region &globalSymbolRegion,
                                           SymbolTable &symbolTable,
                                           const ACCRuntimeCallConfig &config,
                                           DeviceType clauseDeviceType) {
  patterns.add<KernelEnvironmentLowering>(converter, accSupport,
                                          globalSymbolRegion, symbolTable,
                                          config, clauseDeviceType);
}
