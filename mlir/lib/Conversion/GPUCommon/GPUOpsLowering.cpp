//===- GPUOpsLowering.cpp - GPU FuncOp / ReturnOp lowering ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GPUOpsLowering.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;

LogicalResult
GPUFuncOpLowering::matchAndRewrite(gpu::GPUFuncOp gpuFuncOp, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  Location loc = gpuFuncOp.getLoc();

  SmallVector<LLVM::GlobalOp, 3> workgroupBuffers;
  workgroupBuffers.reserve(gpuFuncOp.getNumWorkgroupAttributions());
  for (const auto &en : llvm::enumerate(gpuFuncOp.getWorkgroupAttributions())) {
    Value attribution = en.value();

    auto type = attribution.getType().dyn_cast<MemRefType>();
    assert(type && type.hasStaticShape() && "unexpected type in attribution");

    uint64_t numElements = type.getNumElements();

    auto elementType =
        typeConverter->convertType(type.getElementType()).template cast<Type>();
    auto arrayType = LLVM::LLVMArrayType::get(elementType, numElements);
    std::string name = std::string(
        llvm::formatv("__wg_{0}_{1}", gpuFuncOp.getName(), en.index()));
    auto globalOp = rewriter.create<LLVM::GlobalOp>(
        gpuFuncOp.getLoc(), arrayType, /*isConstant=*/false,
        LLVM::Linkage::Internal, name, /*value=*/Attribute(),
        /*alignment=*/0, gpu::GPUDialect::getWorkgroupAddressSpace());
    workgroupBuffers.push_back(globalOp);
  }

  // Rewrite the original GPU function to an LLVM function.
  auto funcType = typeConverter->convertType(gpuFuncOp.getFunctionType())
                      .template cast<LLVM::LLVMPointerType>()
                      .getElementType();

  // Remap proper input types.
  TypeConverter::SignatureConversion signatureConversion(
      gpuFuncOp.front().getNumArguments());
  getTypeConverter()->convertFunctionSignature(
      gpuFuncOp.getFunctionType(), /*isVariadic=*/false, signatureConversion);

  // Create the new function operation. Only copy those attributes that are
  // not specific to function modeling.
  SmallVector<NamedAttribute, 4> attributes;
  for (const auto &attr : gpuFuncOp->getAttrs()) {
    if (attr.getName() == SymbolTable::getSymbolAttrName() ||
        attr.getName() == FunctionOpInterface::getTypeAttrName() ||
        attr.getName() == gpu::GPUFuncOp::getNumWorkgroupAttributionsAttrName())
      continue;
    attributes.push_back(attr);
  }
  // Add a dialect specific kernel attribute in addition to GPU kernel
  // attribute. The former is necessary for further translation while the
  // latter is expected by gpu.launch_func.
  if (gpuFuncOp.isKernel())
    attributes.emplace_back(kernelAttributeName, rewriter.getUnitAttr());
  auto llvmFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
      gpuFuncOp.getLoc(), gpuFuncOp.getName(), funcType,
      LLVM::Linkage::External, /*dsoLocal*/ false, attributes);

  {
    // Insert operations that correspond to converted workgroup and private
    // memory attributions to the body of the function. This must operate on
    // the original function, before the body region is inlined in the new
    // function to maintain the relation between block arguments and the
    // parent operation that assigns their semantics.
    OpBuilder::InsertionGuard guard(rewriter);

    // Rewrite workgroup memory attributions to addresses of global buffers.
    rewriter.setInsertionPointToStart(&gpuFuncOp.front());
    unsigned numProperArguments = gpuFuncOp.getNumArguments();
    auto i32Type = IntegerType::get(rewriter.getContext(), 32);

    Value zero = nullptr;
    if (!workgroupBuffers.empty())
      zero = rewriter.create<LLVM::ConstantOp>(loc, i32Type,
                                               rewriter.getI32IntegerAttr(0));
    for (const auto &en : llvm::enumerate(workgroupBuffers)) {
      LLVM::GlobalOp global = en.value();
      Value address = rewriter.create<LLVM::AddressOfOp>(loc, global);
      auto elementType =
          global.getType().cast<LLVM::LLVMArrayType>().getElementType();
      Value memory = rewriter.create<LLVM::GEPOp>(
          loc, LLVM::LLVMPointerType::get(elementType, global.getAddrSpace()),
          address, ArrayRef<Value>{zero, zero});

      // Build a memref descriptor pointing to the buffer to plug with the
      // existing memref infrastructure. This may use more registers than
      // otherwise necessary given that memref sizes are fixed, but we can try
      // and canonicalize that away later.
      Value attribution = gpuFuncOp.getWorkgroupAttributions()[en.index()];
      auto type = attribution.getType().cast<MemRefType>();
      auto descr = MemRefDescriptor::fromStaticShape(
          rewriter, loc, *getTypeConverter(), type, memory);
      signatureConversion.remapInput(numProperArguments + en.index(), descr);
    }

    // Rewrite private memory attributions to alloca'ed buffers.
    unsigned numWorkgroupAttributions = gpuFuncOp.getNumWorkgroupAttributions();
    auto int64Ty = IntegerType::get(rewriter.getContext(), 64);
    for (const auto &en : llvm::enumerate(gpuFuncOp.getPrivateAttributions())) {
      Value attribution = en.value();
      auto type = attribution.getType().cast<MemRefType>();
      assert(type && type.hasStaticShape() && "unexpected type in attribution");

      // Explicitly drop memory space when lowering private memory
      // attributions since NVVM models it as `alloca`s in the default
      // memory space and does not support `alloca`s with addrspace(5).
      auto ptrType = LLVM::LLVMPointerType::get(
          typeConverter->convertType(type.getElementType())
              .template cast<Type>(),
          allocaAddrSpace);
      Value numElements = rewriter.create<LLVM::ConstantOp>(
          gpuFuncOp.getLoc(), int64Ty,
          rewriter.getI64IntegerAttr(type.getNumElements()));
      Value allocated = rewriter.create<LLVM::AllocaOp>(
          gpuFuncOp.getLoc(), ptrType, numElements, /*alignment=*/0);
      auto descr = MemRefDescriptor::fromStaticShape(
          rewriter, loc, *getTypeConverter(), type, allocated);
      signatureConversion.remapInput(
          numProperArguments + numWorkgroupAttributions + en.index(), descr);
    }
  }

  // Move the region to the new function, update the entry block signature.
  rewriter.inlineRegionBefore(gpuFuncOp.getBody(), llvmFuncOp.getBody(),
                              llvmFuncOp.end());
  if (failed(rewriter.convertRegionTypes(&llvmFuncOp.getBody(), *typeConverter,
                                         &signatureConversion)))
    return failure();

  rewriter.eraseOp(gpuFuncOp);
  return success();
}

static const char formatStringPrefix[] = "printfFormat_";

template <typename T>
static LLVM::LLVMFuncOp getOrDefineFunction(T &moduleOp, const Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            StringRef name,
                                            LLVM::LLVMFunctionType type) {
  LLVM::LLVMFuncOp ret;
  if (!(ret = moduleOp.template lookupSymbol<LLVM::LLVMFuncOp>(name))) {
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    ret = rewriter.create<LLVM::LLVMFuncOp>(loc, name, type,
                                            LLVM::Linkage::External);
  }
  return ret;
}

LogicalResult GPUPrintfOpToHIPLowering::matchAndRewrite(
    gpu::PrintfOp gpuPrintfOp, gpu::PrintfOpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = gpuPrintfOp->getLoc();

  mlir::Type llvmI8 = typeConverter->convertType(rewriter.getI8Type());
  mlir::Type i8Ptr = LLVM::LLVMPointerType::get(llvmI8);
  mlir::Type llvmIndex = typeConverter->convertType(rewriter.getIndexType());
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
  if (!adaptor.args().empty()) {
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
          {llvmI64, i8Ptr, /*length (bytes)*/ llvmI64, /*isLast*/ llvmI32}));

  /// Start the printf hostcall
  Value zeroI64 = rewriter.create<LLVM::ConstantOp>(
      loc, llvmI64, rewriter.getI64IntegerAttr(0));
  auto printfBeginCall = rewriter.create<LLVM::CallOp>(loc, ocklBegin, zeroI64);
  Value printfDesc = printfBeginCall.getResult(0);

  // Create a global constant for the format string
  unsigned stringNumber = 0;
  SmallString<16> stringConstName;
  do {
    stringConstName.clear();
    (formatStringPrefix + Twine(stringNumber++)).toStringRef(stringConstName);
  } while (moduleOp.lookupSymbol(stringConstName));

  llvm::SmallString<20> formatString(adaptor.format());
  formatString.push_back('\0'); // Null terminate for C
  size_t formatStringSize = formatString.size_in_bytes();

  auto globalType = LLVM::LLVMArrayType::get(llvmI8, formatStringSize);
  LLVM::GlobalOp global;
  {
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    global = rewriter.create<LLVM::GlobalOp>(
        loc, globalType,
        /*isConstant=*/true, LLVM::Linkage::Internal, stringConstName,
        rewriter.getStringAttr(formatString));
  }

  // Get a pointer to the format string's first element and pass it to printf()
  Value globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, global);
  Value zero = rewriter.create<LLVM::ConstantOp>(
      loc, llvmIndex, rewriter.getIntegerAttr(llvmIndex, 0));
  Value stringStart = rewriter.create<LLVM::GEPOp>(
      loc, i8Ptr, globalPtr, mlir::ValueRange({zero, zero}));
  Value stringLen = rewriter.create<LLVM::ConstantOp>(
      loc, llvmI64, rewriter.getI64IntegerAttr(formatStringSize));

  Value oneI32 = rewriter.create<LLVM::ConstantOp>(
      loc, llvmI32, rewriter.getI32IntegerAttr(1));
  Value zeroI32 = rewriter.create<LLVM::ConstantOp>(
      loc, llvmI32, rewriter.getI32IntegerAttr(0));

  auto appendFormatCall = rewriter.create<LLVM::CallOp>(
      loc, ocklAppendStringN,
      ValueRange{printfDesc, stringStart, stringLen,
                 adaptor.args().empty() ? oneI32 : zeroI32});
  printfDesc = appendFormatCall.getResult(0);

  // __ockl_printf_append_args takes 7 values per append call
  constexpr size_t argsPerAppend = 7;
  size_t nArgs = adaptor.args().size();
  for (size_t group = 0; group < nArgs; group += argsPerAppend) {
    size_t bound = std::min(group + argsPerAppend, nArgs);
    size_t numArgsThisCall = bound - group;

    SmallVector<mlir::Value, 2 + argsPerAppend + 1> arguments;
    arguments.push_back(printfDesc);
    arguments.push_back(rewriter.create<LLVM::ConstantOp>(
        loc, llvmI32, rewriter.getI32IntegerAttr(numArgsThisCall)));
    for (size_t i = group; i < bound; ++i) {
      Value arg = adaptor.args()[i];
      if (auto floatType = arg.getType().dyn_cast<FloatType>()) {
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
    printfDesc = call.getResult(0);
  }
  rewriter.eraseOp(gpuPrintfOp);
  return success();
}

LogicalResult GPUPrintfOpToLLVMCallLowering::matchAndRewrite(
    gpu::PrintfOp gpuPrintfOp, gpu::PrintfOpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = gpuPrintfOp->getLoc();

  mlir::Type llvmI8 = typeConverter->convertType(rewriter.getIntegerType(8));
  mlir::Type i8Ptr = LLVM::LLVMPointerType::get(llvmI8, addressSpace);
  mlir::Type llvmIndex = typeConverter->convertType(rewriter.getIndexType());

  // Note: this is the GPUModule op, not the ModuleOp that surrounds it
  // This ensures that global constants and declarations are placed within
  // the device code, not the host code
  auto moduleOp = gpuPrintfOp->getParentOfType<gpu::GPUModuleOp>();

  auto printfType = LLVM::LLVMFunctionType::get(rewriter.getI32Type(), {i8Ptr},
                                                /*isVarArg=*/true);
  LLVM::LLVMFuncOp printfDecl =
      getOrDefineFunction(moduleOp, loc, rewriter, "printf", printfType);

  // Create a global constant for the format string
  unsigned stringNumber = 0;
  SmallString<16> stringConstName;
  do {
    stringConstName.clear();
    (formatStringPrefix + Twine(stringNumber++)).toStringRef(stringConstName);
  } while (moduleOp.lookupSymbol(stringConstName));

  llvm::SmallString<20> formatString(adaptor.format());
  formatString.push_back('\0'); // Null terminate for C
  auto globalType =
      LLVM::LLVMArrayType::get(llvmI8, formatString.size_in_bytes());
  LLVM::GlobalOp global;
  {
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    global = rewriter.create<LLVM::GlobalOp>(
        loc, globalType,
        /*isConstant=*/true, LLVM::Linkage::Internal, stringConstName,
        rewriter.getStringAttr(formatString), /*allignment=*/0, addressSpace);
  }

  // Get a pointer to the format string's first element
  Value globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, global);
  Value zero = rewriter.create<LLVM::ConstantOp>(
      loc, llvmIndex, rewriter.getIntegerAttr(llvmIndex, 0));
  Value stringStart = rewriter.create<LLVM::GEPOp>(
      loc, i8Ptr, globalPtr, mlir::ValueRange({zero, zero}));

  // Construct arguments and function call
  auto argsRange = adaptor.args();
  SmallVector<Value, 4> printfArgs;
  printfArgs.reserve(argsRange.size() + 1);
  printfArgs.push_back(stringStart);
  printfArgs.append(argsRange.begin(), argsRange.end());

  rewriter.create<LLVM::CallOp>(loc, printfDecl, printfArgs);
  rewriter.eraseOp(gpuPrintfOp);
  return success();
}
