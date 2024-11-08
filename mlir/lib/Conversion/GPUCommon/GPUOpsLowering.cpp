//===- GPUOpsLowering.cpp - GPU FuncOp / ReturnOp lowering ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GPUOpsLowering.h"

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;

LogicalResult
GPUFuncOpLowering::matchAndRewrite(gpu::GPUFuncOp gpuFuncOp, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
  Location loc = gpuFuncOp.getLoc();

  SmallVector<LLVM::GlobalOp, 3> workgroupBuffers;
  if (encodeWorkgroupAttributionsAsArguments) {
    // Append an `llvm.ptr` argument to the function signature to encode
    // workgroup attributions.

    ArrayRef<BlockArgument> workgroupAttributions =
        gpuFuncOp.getWorkgroupAttributions();
    size_t numAttributions = workgroupAttributions.size();

    // Insert all arguments at the end.
    unsigned index = gpuFuncOp.getNumArguments();
    SmallVector<unsigned> argIndices(numAttributions, index);

    // New arguments will simply be `llvm.ptr` with the correct address space
    Type workgroupPtrType =
        rewriter.getType<LLVM::LLVMPointerType>(workgroupAddrSpace);
    SmallVector<Type> argTypes(numAttributions, workgroupPtrType);

    // Attributes: noalias, llvm.mlir.workgroup_attribution(<size>, <type>)
    std::array attrs{
        rewriter.getNamedAttr(LLVM::LLVMDialect::getNoAliasAttrName(),
                              rewriter.getUnitAttr()),
        rewriter.getNamedAttr(
            getDialect().getWorkgroupAttributionAttrHelper().getName(),
            rewriter.getUnitAttr()),
    };
    SmallVector<DictionaryAttr> argAttrs;
    for (BlockArgument attribution : workgroupAttributions) {
      auto attributionType = cast<MemRefType>(attribution.getType());
      IntegerAttr numElements =
          rewriter.getI64IntegerAttr(attributionType.getNumElements());
      Type llvmElementType =
          getTypeConverter()->convertType(attributionType.getElementType());
      if (!llvmElementType)
        return failure();
      TypeAttr type = TypeAttr::get(llvmElementType);
      attrs.back().setValue(
          rewriter.getAttr<LLVM::WorkgroupAttributionAttr>(numElements, type));
      argAttrs.push_back(rewriter.getDictionaryAttr(attrs));
    }

    // Location match function location
    SmallVector<Location> argLocs(numAttributions, gpuFuncOp.getLoc());

    // Perform signature modification
    rewriter.modifyOpInPlace(
        gpuFuncOp, [gpuFuncOp, &argIndices, &argTypes, &argAttrs, &argLocs]() {
          static_cast<FunctionOpInterface>(gpuFuncOp).insertArguments(
              argIndices, argTypes, argAttrs, argLocs);
        });
  } else {
    workgroupBuffers.reserve(gpuFuncOp.getNumWorkgroupAttributions());
    for (auto [idx, attribution] :
         llvm::enumerate(gpuFuncOp.getWorkgroupAttributions())) {
      auto type = dyn_cast<MemRefType>(attribution.getType());
      assert(type && type.hasStaticShape() && "unexpected type in attribution");

      uint64_t numElements = type.getNumElements();

      auto elementType =
          cast<Type>(typeConverter->convertType(type.getElementType()));
      auto arrayType = LLVM::LLVMArrayType::get(elementType, numElements);
      std::string name =
          std::string(llvm::formatv("__wg_{0}_{1}", gpuFuncOp.getName(), idx));
      uint64_t alignment = 0;
      if (auto alignAttr = dyn_cast_or_null<IntegerAttr>(
              gpuFuncOp.getWorkgroupAttributionAttr(
                  idx, LLVM::LLVMDialect::getAlignAttrName())))
        alignment = alignAttr.getInt();
      auto globalOp = rewriter.create<LLVM::GlobalOp>(
          gpuFuncOp.getLoc(), arrayType, /*isConstant=*/false,
          LLVM::Linkage::Internal, name, /*value=*/Attribute(), alignment,
          workgroupAddrSpace);
      workgroupBuffers.push_back(globalOp);
    }
  }

  // Remap proper input types.
  TypeConverter::SignatureConversion signatureConversion(
      gpuFuncOp.front().getNumArguments());

  Type funcType = getTypeConverter()->convertFunctionSignature(
      gpuFuncOp.getFunctionType(), /*isVariadic=*/false,
      getTypeConverter()->getOptions().useBarePtrCallConv, signatureConversion);
  if (!funcType) {
    return rewriter.notifyMatchFailure(gpuFuncOp, [&](Diagnostic &diag) {
      diag << "failed to convert function signature type for: "
           << gpuFuncOp.getFunctionType();
    });
  }

  // Create the new function operation. Only copy those attributes that are
  // not specific to function modeling.
  SmallVector<NamedAttribute, 4> attributes;
  ArrayAttr argAttrs;
  for (const auto &attr : gpuFuncOp->getAttrs()) {
    if (attr.getName() == SymbolTable::getSymbolAttrName() ||
        attr.getName() == gpuFuncOp.getFunctionTypeAttrName() ||
        attr.getName() ==
            gpu::GPUFuncOp::getNumWorkgroupAttributionsAttrName() ||
        attr.getName() == gpuFuncOp.getWorkgroupAttribAttrsAttrName() ||
        attr.getName() == gpuFuncOp.getPrivateAttribAttrsAttrName() ||
        attr.getName() == gpuFuncOp.getKnownBlockSizeAttrName() ||
        attr.getName() == gpuFuncOp.getKnownGridSizeAttrName())
      continue;
    if (attr.getName() == gpuFuncOp.getArgAttrsAttrName()) {
      argAttrs = gpuFuncOp.getArgAttrsAttr();
      continue;
    }
    attributes.push_back(attr);
  }

  DenseI32ArrayAttr knownBlockSize = gpuFuncOp.getKnownBlockSizeAttr();
  DenseI32ArrayAttr knownGridSize = gpuFuncOp.getKnownGridSizeAttr();
  // Ensure we don't lose information if the function is lowered before its
  // surrounding context.
  auto *gpuDialect = cast<gpu::GPUDialect>(gpuFuncOp->getDialect());
  if (knownBlockSize)
    attributes.emplace_back(gpuDialect->getKnownBlockSizeAttrHelper().getName(),
                            knownBlockSize);
  if (knownGridSize)
    attributes.emplace_back(gpuDialect->getKnownGridSizeAttrHelper().getName(),
                            knownGridSize);

  // Add a dialect specific kernel attribute in addition to GPU kernel
  // attribute. The former is necessary for further translation while the
  // latter is expected by gpu.launch_func.
  if (gpuFuncOp.isKernel()) {
    if (kernelAttributeName)
      attributes.emplace_back(kernelAttributeName, rewriter.getUnitAttr());
    // Set the dialect-specific block size attribute if there is one.
    if (kernelBlockSizeAttributeName && knownBlockSize) {
      attributes.emplace_back(kernelBlockSizeAttributeName, knownBlockSize);
    }
  }
  LLVM::CConv callingConvention = gpuFuncOp.isKernel()
                                      ? kernelCallingConvention
                                      : nonKernelCallingConvention;
  auto llvmFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
      gpuFuncOp.getLoc(), gpuFuncOp.getName(), funcType,
      LLVM::Linkage::External, /*dsoLocal=*/false, callingConvention,
      /*comdat=*/nullptr, attributes);

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

    if (encodeWorkgroupAttributionsAsArguments) {
      // Build a MemRefDescriptor with each of the arguments added above.

      unsigned numAttributions = gpuFuncOp.getNumWorkgroupAttributions();
      assert(numProperArguments >= numAttributions &&
             "Expecting attributions to be encoded as arguments already");

      // Arguments encoding workgroup attributions will be in positions
      // [numProperArguments, numProperArguments+numAttributions)
      ArrayRef<BlockArgument> attributionArguments =
          gpuFuncOp.getArguments().slice(numProperArguments - numAttributions,
                                         numAttributions);
      for (auto [idx, vals] : llvm::enumerate(llvm::zip_equal(
               gpuFuncOp.getWorkgroupAttributions(), attributionArguments))) {
        auto [attribution, arg] = vals;
        auto type = cast<MemRefType>(attribution.getType());

        // Arguments are of llvm.ptr type and attributions are of memref type:
        // we need to wrap them in memref descriptors.
        Value descr = MemRefDescriptor::fromStaticShape(
            rewriter, loc, *getTypeConverter(), type, arg);

        // And remap the arguments
        signatureConversion.remapInput(numProperArguments + idx, descr);
      }
    } else {
      for (const auto [idx, global] : llvm::enumerate(workgroupBuffers)) {
        auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(),
                                                  global.getAddrSpace());
        Value address = rewriter.create<LLVM::AddressOfOp>(
            loc, ptrType, global.getSymNameAttr());
        Value memory =
            rewriter.create<LLVM::GEPOp>(loc, ptrType, global.getType(),
                                         address, ArrayRef<LLVM::GEPArg>{0, 0});

        // Build a memref descriptor pointing to the buffer to plug with the
        // existing memref infrastructure. This may use more registers than
        // otherwise necessary given that memref sizes are fixed, but we can try
        // and canonicalize that away later.
        Value attribution = gpuFuncOp.getWorkgroupAttributions()[idx];
        auto type = cast<MemRefType>(attribution.getType());
        auto descr = MemRefDescriptor::fromStaticShape(
            rewriter, loc, *getTypeConverter(), type, memory);
        signatureConversion.remapInput(numProperArguments + idx, descr);
      }
    }

    // Rewrite private memory attributions to alloca'ed buffers.
    unsigned numWorkgroupAttributions = gpuFuncOp.getNumWorkgroupAttributions();
    auto int64Ty = IntegerType::get(rewriter.getContext(), 64);
    for (const auto [idx, attribution] :
         llvm::enumerate(gpuFuncOp.getPrivateAttributions())) {
      auto type = cast<MemRefType>(attribution.getType());
      assert(type && type.hasStaticShape() && "unexpected type in attribution");

      // Explicitly drop memory space when lowering private memory
      // attributions since NVVM models it as `alloca`s in the default
      // memory space and does not support `alloca`s with addrspace(5).
      Type elementType = typeConverter->convertType(type.getElementType());
      auto ptrType =
          LLVM::LLVMPointerType::get(rewriter.getContext(), allocaAddrSpace);
      Value numElements = rewriter.create<LLVM::ConstantOp>(
          gpuFuncOp.getLoc(), int64Ty, type.getNumElements());
      uint64_t alignment = 0;
      if (auto alignAttr =
              dyn_cast_or_null<IntegerAttr>(gpuFuncOp.getPrivateAttributionAttr(
                  idx, LLVM::LLVMDialect::getAlignAttrName())))
        alignment = alignAttr.getInt();
      Value allocated = rewriter.create<LLVM::AllocaOp>(
          gpuFuncOp.getLoc(), ptrType, elementType, numElements, alignment);
      auto descr = MemRefDescriptor::fromStaticShape(
          rewriter, loc, *getTypeConverter(), type, allocated);
      signatureConversion.remapInput(
          numProperArguments + numWorkgroupAttributions + idx, descr);
    }
  }

  // Move the region to the new function, update the entry block signature.
  rewriter.inlineRegionBefore(gpuFuncOp.getBody(), llvmFuncOp.getBody(),
                              llvmFuncOp.end());
  if (failed(rewriter.convertRegionTypes(&llvmFuncOp.getBody(), *typeConverter,
                                         &signatureConversion)))
    return failure();

  // Get memref type from function arguments and set the noalias to
  // pointer arguments.
  for (const auto [idx, argTy] :
       llvm::enumerate(gpuFuncOp.getArgumentTypes())) {
    auto remapping = signatureConversion.getInputMapping(idx);
    NamedAttrList argAttr =
        argAttrs ? cast<DictionaryAttr>(argAttrs[idx]) : NamedAttrList();
    auto copyAttribute = [&](StringRef attrName) {
      Attribute attr = argAttr.erase(attrName);
      if (!attr)
        return;
      for (size_t i = 0, e = remapping->size; i < e; ++i)
        llvmFuncOp.setArgAttr(remapping->inputNo + i, attrName, attr);
    };
    auto copyPointerAttribute = [&](StringRef attrName) {
      Attribute attr = argAttr.erase(attrName);

      if (!attr)
        return;
      if (remapping->size > 1 &&
          attrName == LLVM::LLVMDialect::getNoAliasAttrName()) {
        emitWarning(llvmFuncOp.getLoc(),
                    "Cannot copy noalias with non-bare pointers.\n");
        return;
      }
      for (size_t i = 0, e = remapping->size; i < e; ++i) {
        if (isa<LLVM::LLVMPointerType>(
                llvmFuncOp.getArgument(remapping->inputNo + i).getType())) {
          llvmFuncOp.setArgAttr(remapping->inputNo + i, attrName, attr);
        }
      }
    };

    if (argAttr.empty())
      continue;

    copyAttribute(LLVM::LLVMDialect::getReturnedAttrName());
    copyAttribute(LLVM::LLVMDialect::getNoUndefAttrName());
    copyAttribute(LLVM::LLVMDialect::getInRegAttrName());
    bool lowersToPointer = false;
    for (size_t i = 0, e = remapping->size; i < e; ++i) {
      lowersToPointer |= isa<LLVM::LLVMPointerType>(
          llvmFuncOp.getArgument(remapping->inputNo + i).getType());
    }

    if (lowersToPointer) {
      copyPointerAttribute(LLVM::LLVMDialect::getNoAliasAttrName());
      copyPointerAttribute(LLVM::LLVMDialect::getNoCaptureAttrName());
      copyPointerAttribute(LLVM::LLVMDialect::getNoFreeAttrName());
      copyPointerAttribute(LLVM::LLVMDialect::getAlignAttrName());
      copyPointerAttribute(LLVM::LLVMDialect::getReadonlyAttrName());
      copyPointerAttribute(LLVM::LLVMDialect::getWriteOnlyAttrName());
      copyPointerAttribute(LLVM::LLVMDialect::getReadnoneAttrName());
      copyPointerAttribute(LLVM::LLVMDialect::getNonNullAttrName());
      copyPointerAttribute(LLVM::LLVMDialect::getDereferenceableAttrName());
      copyPointerAttribute(
          LLVM::LLVMDialect::getDereferenceableOrNullAttrName());
      copyPointerAttribute(
          LLVM::LLVMDialect::WorkgroupAttributionAttrHelper::getNameStr());
    }
  }
  rewriter.eraseOp(gpuFuncOp);
  return success();
}

static SmallString<16> getUniqueFormatGlobalName(gpu::GPUModuleOp moduleOp) {
  const char formatStringPrefix[] = "printfFormat_";
  // Get a unique global name.
  unsigned stringNumber = 0;
  SmallString<16> stringConstName;
  do {
    stringConstName.clear();
    (formatStringPrefix + Twine(stringNumber++)).toStringRef(stringConstName);
  } while (moduleOp.lookupSymbol(stringConstName));
  return stringConstName;
}

/// Create an global that contains the given format string. If a global with
/// the same format string exists already in the module, return that global.
static LLVM::GlobalOp getOrCreateFormatStringConstant(
    OpBuilder &b, Location loc, gpu::GPUModuleOp moduleOp, Type llvmI8,
    StringRef str, uint64_t alignment = 0, unsigned addrSpace = 0) {
  llvm::SmallString<20> formatString(str);
  formatString.push_back('\0'); // Null terminate for C
  auto globalType =
      LLVM::LLVMArrayType::get(llvmI8, formatString.size_in_bytes());
  StringAttr attr = b.getStringAttr(formatString);

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
  SmallString<16> name = getUniqueFormatGlobalName(moduleOp);
  return b.create<LLVM::GlobalOp>(loc, globalType,
                                  /*isConstant=*/true, LLVM::Linkage::Internal,
                                  name, attr, alignment, addrSpace);
}

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
  LLVM::GlobalOp global = getOrCreateFormatStringConstant(
      rewriter, loc, moduleOp, llvmI8, adaptor.getFormat());

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
  LLVM::GlobalOp global = getOrCreateFormatStringConstant(
      rewriter, loc, moduleOp, llvmI8, adaptor.getFormat(), /*alignment=*/0,
      addressSpace);

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
  LLVM::GlobalOp global = getOrCreateFormatStringConstant(
      rewriter, loc, moduleOp, llvmI8, adaptor.getFormat());

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

/// Unrolls op if it's operating on vectors.
LogicalResult impl::scalarizeVectorOp(Operation *op, ValueRange operands,
                                      ConversionPatternRewriter &rewriter,
                                      const LLVMTypeConverter &converter) {
  TypeRange operandTypes(operands);
  if (llvm::none_of(operandTypes, llvm::IsaPred<VectorType>)) {
    return rewriter.notifyMatchFailure(op, "expected vector operand");
  }
  if (op->getNumRegions() != 0 || op->getNumSuccessors() != 0)
    return rewriter.notifyMatchFailure(op, "expected no region/successor");
  if (op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op, "expected single result");
  VectorType vectorType = dyn_cast<VectorType>(op->getResult(0).getType());
  if (!vectorType)
    return rewriter.notifyMatchFailure(op, "expected vector result");

  Location loc = op->getLoc();
  Value result = rewriter.create<LLVM::UndefOp>(loc, vectorType);
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

  rewriter.replaceOp(op, result);
  return success();
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

  Value packed = rewriter.create<LLVM::UndefOp>(loc, packedType);
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
