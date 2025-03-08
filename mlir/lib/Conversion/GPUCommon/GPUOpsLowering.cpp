//===- GPUOpsLowering.cpp - GPU FuncOp / ReturnOp lowering ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GPUOpsLowering.h"

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;

LLVM::LLVMFuncOp mlir::getOrDefineFunction(gpu::GPUModuleOp moduleOp,
                                           Location loc, OpBuilder &b,
                                           StringRef name,
                                           LLVM::LLVMFunctionType type) {
  LLVM::LLVMFuncOp ret;
  if (!(ret = moduleOp.template lookupSymbol<LLVM::LLVMFuncOp>(name))) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(moduleOp.getBody());
    ret = b.create<LLVM::LLVMFuncOp>(loc, name, type, LLVM::Linkage::External);
  }
  return ret;
}

static SmallString<16> getUniqueSymbolName(gpu::GPUModuleOp moduleOp,
                                           StringRef prefix) {
  // Get a unique global name.
  unsigned stringNumber = 0;
  SmallString<16> stringConstName;
  do {
    stringConstName.clear();
    (prefix + Twine(stringNumber++)).toStringRef(stringConstName);
  } while (moduleOp.lookupSymbol(stringConstName));
  return stringConstName;
}

LLVM::GlobalOp
mlir::getOrCreateStringConstant(OpBuilder &b, Location loc,
                                gpu::GPUModuleOp moduleOp, Type llvmI8,
                                StringRef namePrefix, StringRef str,
                                uint64_t alignment, unsigned addrSpace) {
  llvm::SmallString<20> nullTermStr(str);
  nullTermStr.push_back('\0'); // Null terminate for C
  auto globalType =
      LLVM::LLVMArrayType::get(llvmI8, nullTermStr.size_in_bytes());
  StringAttr attr = b.getStringAttr(nullTermStr);

  // Try to find existing global.
  for (auto globalOp : moduleOp.getOps<LLVM::GlobalOp>())
    if (globalOp.getGlobalType() == globalType && globalOp.getConstant() &&
        globalOp.getValueAttr() == attr &&
        globalOp.getAlignment().value_or(0) == alignment &&
        globalOp.getAddrSpace() == addrSpace)
      return globalOp;

  // Not found: create new global.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(moduleOp.getBody());
  SmallString<16> name = getUniqueSymbolName(moduleOp, namePrefix);
  return b.create<LLVM::GlobalOp>(loc, globalType,
                                  /*isConstant=*/true, LLVM::Linkage::Internal,
                                  name, attr, alignment, addrSpace);
}

LogicalResult
GPUFuncOpLowering::matchAndRewrite(gpu::GPUFuncOp gpuFuncOp, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {

  return failure();
}

LogicalResult GPUPrintfOpToHIPLowering::matchAndRewrite(
    gpu::PrintfOp gpuPrintfOp, gpu::PrintfOpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = gpuPrintfOp->getLoc();

  mlir::Type llvmI8 = typeConverter->convertType(rewriter.getI8Type());
  auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  mlir::Type llvmI32 = typeConverter->convertType(rewriter.getI32Type());
  mlir::Type llvmI64 = typeConverter->convertType(rewriter.getI64Type());
  // Note: this is the GPUModule op, not the ModuleOp that surrounds it
  // This ensures that global constants and declarations are placed within
  // the device code, not the host code
  auto moduleOp = gpuPrintfOp->getParentOfType<gpu::GPUModuleOp>();

  auto ocklBegin =
      getOrDefineFunction(moduleOp, loc, rewriter, "__ockl_printf_begin",
                          LLVM::LLVMFunctionType::get(llvmI64, {llvmI64}));
  LLVM::LLVMFuncOp ocklAppendArgs;
  if (!adaptor.getArgs().empty()) {
    ocklAppendArgs = getOrDefineFunction(
        moduleOp, loc, rewriter, "__ockl_printf_append_args",
        LLVM::LLVMFunctionType::get(
            llvmI64, {llvmI64, /*numArgs*/ llvmI32, llvmI64, llvmI64, llvmI64,
                      llvmI64, llvmI64, llvmI64, llvmI64, /*isLast*/ llvmI32}));
  }
  auto ocklAppendStringN = getOrDefineFunction(
      moduleOp, loc, rewriter, "__ockl_printf_append_string_n",
      LLVM::LLVMFunctionType::get(
          llvmI64,
          {llvmI64, ptrType, /*length (bytes)*/ llvmI64, /*isLast*/ llvmI32}));

  /// Start the printf hostcall
  Value zeroI64 = rewriter.create<LLVM::ConstantOp>(loc, llvmI64, 0);
  auto printfBeginCall = rewriter.create<LLVM::CallOp>(loc, ocklBegin, zeroI64);
  Value printfDesc = printfBeginCall.getResult();

  // Create the global op or find an existing one.
  LLVM::GlobalOp global = getOrCreateStringConstant(
      rewriter, loc, moduleOp, llvmI8, "printfFormat_", adaptor.getFormat());

  // Get a pointer to the format string's first element and pass it to printf()
  Value globalPtr = rewriter.create<LLVM::AddressOfOp>(
      loc,
      LLVM::LLVMPointerType::get(rewriter.getContext(), global.getAddrSpace()),
      global.getSymNameAttr());
  Value stringStart =
      rewriter.create<LLVM::GEPOp>(loc, ptrType, global.getGlobalType(),
                                   globalPtr, ArrayRef<LLVM::GEPArg>{0, 0});
  Value stringLen = rewriter.create<LLVM::ConstantOp>(
      loc, llvmI64, cast<StringAttr>(global.getValueAttr()).size());

  Value oneI32 = rewriter.create<LLVM::ConstantOp>(loc, llvmI32, 1);
  Value zeroI32 = rewriter.create<LLVM::ConstantOp>(loc, llvmI32, 0);

  auto appendFormatCall = rewriter.create<LLVM::CallOp>(
      loc, ocklAppendStringN,
      ValueRange{printfDesc, stringStart, stringLen,
                 adaptor.getArgs().empty() ? oneI32 : zeroI32});
  printfDesc = appendFormatCall.getResult();

  // __ockl_printf_append_args takes 7 values per append call
  constexpr size_t argsPerAppend = 7;
  size_t nArgs = adaptor.getArgs().size();
  for (size_t group = 0; group < nArgs; group += argsPerAppend) {
    size_t bound = std::min(group + argsPerAppend, nArgs);
    size_t numArgsThisCall = bound - group;

    SmallVector<mlir::Value, 2 + argsPerAppend + 1> arguments;
    arguments.push_back(printfDesc);
    arguments.push_back(
        rewriter.create<LLVM::ConstantOp>(loc, llvmI32, numArgsThisCall));
    for (size_t i = group; i < bound; ++i) {
      Value arg = adaptor.getArgs()[i];
      if (auto floatType = dyn_cast<FloatType>(arg.getType())) {
        if (!floatType.isF64())
          arg = rewriter.create<LLVM::FPExtOp>(
              loc, typeConverter->convertType(rewriter.getF64Type()), arg);
        arg = rewriter.create<LLVM::BitcastOp>(loc, llvmI64, arg);
      }
      if (arg.getType().getIntOrFloatBitWidth() != 64)
        arg = rewriter.create<LLVM::ZExtOp>(loc, llvmI64, arg);

      arguments.push_back(arg);
    }
    // Pad out to 7 arguments since the hostcall always needs 7
    for (size_t extra = numArgsThisCall; extra < argsPerAppend; ++extra) {
      arguments.push_back(zeroI64);
    }

    auto isLast = (bound == nArgs) ? oneI32 : zeroI32;
    arguments.push_back(isLast);
    auto call = rewriter.create<LLVM::CallOp>(loc, ocklAppendArgs, arguments);
    printfDesc = call.getResult();
  }
  rewriter.eraseOp(gpuPrintfOp);
  return success();
}

LogicalResult GPUPrintfOpToLLVMCallLowering::matchAndRewrite(
    gpu::PrintfOp gpuPrintfOp, gpu::PrintfOpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = gpuPrintfOp->getLoc();

  mlir::Type llvmI8 = typeConverter->convertType(rewriter.getIntegerType(8));
  mlir::Type ptrType =
      LLVM::LLVMPointerType::get(rewriter.getContext(), addressSpace);

  // Note: this is the GPUModule op, not the ModuleOp that surrounds it
  // This ensures that global constants and declarations are placed within
  // the device code, not the host code
  auto moduleOp = gpuPrintfOp->getParentOfType<gpu::GPUModuleOp>();

  auto printfType =
      LLVM::LLVMFunctionType::get(rewriter.getI32Type(), {ptrType},
                                  /*isVarArg=*/true);
  LLVM::LLVMFuncOp printfDecl =
      getOrDefineFunction(moduleOp, loc, rewriter, "printf", printfType);

  // Create the global op or find an existing one.
  LLVM::GlobalOp global = getOrCreateStringConstant(
      rewriter, loc, moduleOp, llvmI8, "printfFormat_", adaptor.getFormat(),
      /*alignment=*/0, addressSpace);

  // Get a pointer to the format string's first element
  Value globalPtr = rewriter.create<LLVM::AddressOfOp>(
      loc,
      LLVM::LLVMPointerType::get(rewriter.getContext(), global.getAddrSpace()),
      global.getSymNameAttr());
  Value stringStart =
      rewriter.create<LLVM::GEPOp>(loc, ptrType, global.getGlobalType(),
                                   globalPtr, ArrayRef<LLVM::GEPArg>{0, 0});

  // Construct arguments and function call
  auto argsRange = adaptor.getArgs();
  SmallVector<Value, 4> printfArgs;
  printfArgs.reserve(argsRange.size() + 1);
  printfArgs.push_back(stringStart);
  printfArgs.append(argsRange.begin(), argsRange.end());

  rewriter.create<LLVM::CallOp>(loc, printfDecl, printfArgs);
  rewriter.eraseOp(gpuPrintfOp);
  return success();
}

LogicalResult GPUPrintfOpToVPrintfLowering::matchAndRewrite(
    gpu::PrintfOp gpuPrintfOp, gpu::PrintfOpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = gpuPrintfOp->getLoc();

  mlir::Type llvmI8 = typeConverter->convertType(rewriter.getIntegerType(8));
  mlir::Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

  // Note: this is the GPUModule op, not the ModuleOp that surrounds it
  // This ensures that global constants and declarations are placed within
  // the device code, not the host code
  auto moduleOp = gpuPrintfOp->getParentOfType<gpu::GPUModuleOp>();

  auto vprintfType =
      LLVM::LLVMFunctionType::get(rewriter.getI32Type(), {ptrType, ptrType});
  LLVM::LLVMFuncOp vprintfDecl =
      getOrDefineFunction(moduleOp, loc, rewriter, "vprintf", vprintfType);

  // Create the global op or find an existing one.
  LLVM::GlobalOp global = getOrCreateStringConstant(
      rewriter, loc, moduleOp, llvmI8, "printfFormat_", adaptor.getFormat());

  // Get a pointer to the format string's first element
  Value globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, global);
  Value stringStart =
      rewriter.create<LLVM::GEPOp>(loc, ptrType, global.getGlobalType(),
                                   globalPtr, ArrayRef<LLVM::GEPArg>{0, 0});
  SmallVector<Type> types;
  SmallVector<Value> args;
  // Promote and pack the arguments into a stack allocation.
  for (Value arg : adaptor.getArgs()) {
    Type type = arg.getType();
    Value promotedArg = arg;
    assert(type.isIntOrFloat());
    if (isa<FloatType>(type)) {
      type = rewriter.getF64Type();
      promotedArg = rewriter.create<LLVM::FPExtOp>(loc, type, arg);
    }
    types.push_back(type);
    args.push_back(promotedArg);
  }
  Type structType =
      LLVM::LLVMStructType::getLiteral(gpuPrintfOp.getContext(), types);
  Value one = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(),
                                                rewriter.getIndexAttr(1));
  Value tempAlloc =
      rewriter.create<LLVM::AllocaOp>(loc, ptrType, structType, one,
                                      /*alignment=*/0);
  for (auto [index, arg] : llvm::enumerate(args)) {
    Value ptr = rewriter.create<LLVM::GEPOp>(
        loc, ptrType, structType, tempAlloc,
        ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(index)});
    rewriter.create<LLVM::StoreOp>(loc, arg, ptr);
  }
  std::array<Value, 2> printfArgs = {stringStart, tempAlloc};

  rewriter.create<LLVM::CallOp>(loc, vprintfDecl, printfArgs);
  rewriter.eraseOp(gpuPrintfOp);
  return success();
}

/// Helper for impl::scalarizeVectorOp. Scalarizes vectors to elements.
/// Used either directly (for ops on 1D vectors) or as the callback passed to
/// detail::handleMultidimensionalVectors (for ops on higher-rank vectors).
static Value scalarizeVectorOpHelper(Operation *op, ValueRange operands,
                                     Type llvm1DVectorTy,
                                     ConversionPatternRewriter &rewriter,
                                     const LLVMTypeConverter &converter) {
  TypeRange operandTypes(operands);
  VectorType vectorType = cast<VectorType>(llvm1DVectorTy);
  Location loc = op->getLoc();
  Value result = rewriter.create<LLVM::PoisonOp>(loc, vectorType);
  Type indexType = converter.convertType(rewriter.getIndexType());
  StringAttr name = op->getName().getIdentifier();
  Type elementType = vectorType.getElementType();

  for (int64_t i = 0; i < vectorType.getNumElements(); ++i) {
    Value index = rewriter.create<LLVM::ConstantOp>(loc, indexType, i);
    auto extractElement = [&](Value operand) -> Value {
      if (!isa<VectorType>(operand.getType()))
        return operand;
      return rewriter.create<LLVM::ExtractElementOp>(loc, operand, index);
    };
    auto scalarOperands = llvm::map_to_vector(operands, extractElement);
    Operation *scalarOp =
        rewriter.create(loc, name, scalarOperands, elementType, op->getAttrs());
    result = rewriter.create<LLVM::InsertElementOp>(
        loc, result, scalarOp->getResult(0), index);
  }
  return result;
}

/// Unrolls op to array/vector elements.
LogicalResult impl::scalarizeVectorOp(Operation *op, ValueRange operands,
                                      ConversionPatternRewriter &rewriter,
                                      const LLVMTypeConverter &converter) {
  TypeRange operandTypes(operands);
  if (llvm::any_of(operandTypes, llvm::IsaPred<VectorType>)) {
    VectorType vectorType =
        cast<VectorType>(converter.convertType(op->getResultTypes()[0]));
    rewriter.replaceOp(op, scalarizeVectorOpHelper(op, operands, vectorType,
                                                   rewriter, converter));
    return success();
  }

  if (llvm::any_of(operandTypes, llvm::IsaPred<LLVM::LLVMArrayType>)) {
    return LLVM::detail::handleMultidimensionalVectors(
        op, operands, converter,
        [&](Type llvm1DVectorTy, ValueRange operands) -> Value {
          return scalarizeVectorOpHelper(op, operands, llvm1DVectorTy, rewriter,
                                         converter);
        },
        rewriter);
  }

  return rewriter.notifyMatchFailure(op, "no llvm.array or vector to unroll");
}

static IntegerAttr wrapNumericMemorySpace(MLIRContext *ctx, unsigned space) {
  return IntegerAttr::get(IntegerType::get(ctx, 64), space);
}

/// Generates a symbol with 0-sized array type for dynamic shared memory usage,
/// or uses existing symbol.
LLVM::GlobalOp getDynamicSharedMemorySymbol(
    ConversionPatternRewriter &rewriter, gpu::GPUModuleOp moduleOp,
    gpu::DynamicSharedMemoryOp op, const LLVMTypeConverter *typeConverter,
    MemRefType memrefType, unsigned alignmentBit) {
  uint64_t alignmentByte = alignmentBit / memrefType.getElementTypeBitWidth();

  FailureOr<unsigned> addressSpace =
      typeConverter->getMemRefAddressSpace(memrefType);
  if (failed(addressSpace)) {
    op->emitError() << "conversion of memref memory space "
                    << memrefType.getMemorySpace()
                    << " to integer address space "
                       "failed. Consider adding memory space conversions.";
  }

  // Step 1. Collect symbol names of LLVM::GlobalOp Ops. Also if any of
  // LLVM::GlobalOp is suitable for shared memory, return it.
  llvm::StringSet<> existingGlobalNames;
  for (auto globalOp : moduleOp.getBody()->getOps<LLVM::GlobalOp>()) {
    existingGlobalNames.insert(globalOp.getSymName());
    if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(globalOp.getType())) {
      if (globalOp.getAddrSpace() == addressSpace.value() &&
          arrayType.getNumElements() == 0 &&
          globalOp.getAlignment().value_or(0) == alignmentByte) {
        return globalOp;
      }
    }
  }

  // Step 2. Find a unique symbol name
  unsigned uniquingCounter = 0;
  SmallString<128> symName = SymbolTable::generateSymbolName<128>(
      "__dynamic_shmem_",
      [&](StringRef candidate) {
        return existingGlobalNames.contains(candidate);
      },
      uniquingCounter);

  // Step 3. Generate a global op
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  auto zeroSizedArrayType = LLVM::LLVMArrayType::get(
      typeConverter->convertType(memrefType.getElementType()), 0);

  return rewriter.create<LLVM::GlobalOp>(
      op->getLoc(), zeroSizedArrayType, /*isConstant=*/false,
      LLVM::Linkage::Internal, symName, /*value=*/Attribute(), alignmentByte,
      addressSpace.value());
}

LogicalResult GPUDynamicSharedMemoryOpLowering::matchAndRewrite(
    gpu::DynamicSharedMemoryOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  MemRefType memrefType = op.getResultMemref().getType();
  Type elementType = typeConverter->convertType(memrefType.getElementType());

  // Step 1: Generate a memref<0xi8> type
  MemRefLayoutAttrInterface layout = {};
  auto memrefType0sz =
      MemRefType::get({0}, elementType, layout, memrefType.getMemorySpace());

  // Step 2: Generate a global symbol or existing for the dynamic shared
  // memory with memref<0xi8> type
  auto moduleOp = op->getParentOfType<gpu::GPUModuleOp>();
  LLVM::GlobalOp shmemOp = getDynamicSharedMemorySymbol(
      rewriter, moduleOp, op, getTypeConverter(), memrefType0sz, alignmentBit);

  // Step 3. Get address of the global symbol
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  auto basePtr = rewriter.create<LLVM::AddressOfOp>(loc, shmemOp);
  Type baseType = basePtr->getResultTypes().front();

  // Step 4. Generate GEP using offsets
  SmallVector<LLVM::GEPArg> gepArgs = {0};
  Value shmemPtr = rewriter.create<LLVM::GEPOp>(loc, baseType, elementType,
                                                basePtr, gepArgs);
  // Step 5. Create a memref descriptor
  SmallVector<Value> shape, strides;
  Value sizeBytes;
  getMemRefDescriptorSizes(loc, memrefType0sz, {}, rewriter, shape, strides,
                           sizeBytes);
  auto memRefDescriptor = this->createMemRefDescriptor(
      loc, memrefType0sz, shmemPtr, shmemPtr, shape, strides, rewriter);

  // Step 5. Replace the op with memref descriptor
  rewriter.replaceOp(op, {memRefDescriptor});
  return success();
}

LogicalResult GPUReturnOpLowering::matchAndRewrite(
    gpu::ReturnOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  unsigned numArguments = op.getNumOperands();
  SmallVector<Value, 4> updatedOperands;

  bool useBarePtrCallConv = getTypeConverter()->getOptions().useBarePtrCallConv;
  if (useBarePtrCallConv) {
    // For the bare-ptr calling convention, extract the aligned pointer to
    // be returned from the memref descriptor.
    for (auto it : llvm::zip(op->getOperands(), adaptor.getOperands())) {
      Type oldTy = std::get<0>(it).getType();
      Value newOperand = std::get<1>(it);
      if (isa<MemRefType>(oldTy) && getTypeConverter()->canConvertToBarePtr(
                                        cast<BaseMemRefType>(oldTy))) {
        MemRefDescriptor memrefDesc(newOperand);
        newOperand = memrefDesc.allocatedPtr(rewriter, loc);
      } else if (isa<UnrankedMemRefType>(oldTy)) {
        // Unranked memref is not supported in the bare pointer calling
        // convention.
        return failure();
      }
      updatedOperands.push_back(newOperand);
    }
  } else {
    updatedOperands = llvm::to_vector<4>(adaptor.getOperands());
    (void)copyUnrankedDescriptors(rewriter, loc, op.getOperands().getTypes(),
                                  updatedOperands,
                                  /*toDynamic=*/true);
  }

  // If ReturnOp has 0 or 1 operand, create it and return immediately.
  if (numArguments <= 1) {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(
        op, TypeRange(), updatedOperands, op->getAttrs());
    return success();
  }

  // Otherwise, we need to pack the arguments into an LLVM struct type before
  // returning.
  auto packedType = getTypeConverter()->packFunctionResults(
      op.getOperandTypes(), useBarePtrCallConv);
  if (!packedType) {
    return rewriter.notifyMatchFailure(op, "could not convert result types");
  }

  Value packed = rewriter.create<LLVM::PoisonOp>(loc, packedType);
  for (auto [idx, operand] : llvm::enumerate(updatedOperands)) {
    packed = rewriter.create<LLVM::InsertValueOp>(loc, packed, operand, idx);
  }
  rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(), packed,
                                              op->getAttrs());
  return success();
}

void mlir::populateGpuMemorySpaceAttributeConversions(
    TypeConverter &typeConverter, const MemorySpaceMapping &mapping) {
  typeConverter.addTypeAttributeConversion(
      [mapping](BaseMemRefType type, gpu::AddressSpaceAttr memorySpaceAttr) {
        gpu::AddressSpace memorySpace = memorySpaceAttr.getValue();
        unsigned addressSpace = mapping(memorySpace);
        return wrapNumericMemorySpace(memorySpaceAttr.getContext(),
                                      addressSpace);
      });
}
