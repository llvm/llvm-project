//===- FuncToLLVM.cpp - Func to LLVM dialect conversion -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert MLIR Func and builtin dialects
// into the LLVM IR dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/FormatVariadic.h"
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_CONVERTFUNCTOLLVMPASS
#define GEN_PASS_DEF_SETLLVMMODULEDATALAYOUTPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define PASS_NAME "convert-func-to-llvm"

static constexpr StringRef varargsAttrName = "func.varargs";
static constexpr StringRef linkageAttrName = "llvm.linkage";
static constexpr StringRef barePtrAttrName = "llvm.bareptr";

/// Return `true` if the `op` should use bare pointer calling convention.
static bool shouldUseBarePtrCallConv(Operation *op,
                                     const LLVMTypeConverter *typeConverter) {
  return (op && op->hasAttr(barePtrAttrName)) ||
         typeConverter->getOptions().useBarePtrCallConv;
}

/// Only retain those attributes that are not constructed by
/// `LLVMFuncOp::build`.
static void filterFuncAttributes(FunctionOpInterface func,
                                 SmallVectorImpl<NamedAttribute> &result) {
  for (const NamedAttribute &attr : func->getDiscardableAttrs()) {
    if (attr.getName() == linkageAttrName ||
        attr.getName() == varargsAttrName ||
        attr.getName() == LLVM::LLVMDialect::getReadnoneAttrName())
      continue;
    result.push_back(attr);
  }
}

/// Propagate argument/results attributes.
static void propagateArgResAttrs(OpBuilder &builder, bool resultStructType,
                                 FunctionOpInterface funcOp,
                                 LLVM::LLVMFuncOp wrapperFuncOp) {
  auto argAttrs = funcOp.getAllArgAttrs();
  if (!resultStructType) {
    if (auto resAttrs = funcOp.getAllResultAttrs())
      wrapperFuncOp.setAllResultAttrs(resAttrs);
    if (argAttrs)
      wrapperFuncOp.setAllArgAttrs(argAttrs);
  } else {
    SmallVector<Attribute> argAttributes;
    // Only modify the argument and result attributes when the result is now
    // an argument.
    if (argAttrs) {
      argAttributes.push_back(builder.getDictionaryAttr({}));
      argAttributes.append(argAttrs.begin(), argAttrs.end());
      wrapperFuncOp.setAllArgAttrs(argAttributes);
    }
  }
  cast<FunctionOpInterface>(wrapperFuncOp.getOperation())
      .setVisibility(funcOp.getVisibility());
}

/// Creates an auxiliary function with pointer-to-memref-descriptor-struct
/// arguments instead of unpacked arguments. This function can be called from C
/// by passing a pointer to a C struct corresponding to a memref descriptor.
/// Similarly, returned memrefs are passed via pointers to a C struct that is
/// passed as additional argument.
/// Internally, the auxiliary function unpacks the descriptor into individual
/// components and forwards them to `newFuncOp` and forwards the results to
/// the extra arguments.
static void wrapForExternalCallers(OpBuilder &rewriter, Location loc,
                                   const LLVMTypeConverter &typeConverter,
                                   FunctionOpInterface funcOp,
                                   LLVM::LLVMFuncOp newFuncOp) {
  auto type = cast<FunctionType>(funcOp.getFunctionType());
  auto [wrapperFuncType, resultStructType] =
      typeConverter.convertFunctionTypeCWrapper(type);

  SmallVector<NamedAttribute> attributes;
  filterFuncAttributes(funcOp, attributes);

  auto wrapperFuncOp = LLVM::LLVMFuncOp::create(
      rewriter, loc, llvm::formatv("_mlir_ciface_{0}", funcOp.getName()).str(),
      wrapperFuncType, LLVM::Linkage::External, /*dsoLocal=*/false,
      /*cconv=*/LLVM::CConv::C, /*comdat=*/nullptr, attributes);
  propagateArgResAttrs(rewriter, !!resultStructType, funcOp, wrapperFuncOp);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(wrapperFuncOp.addEntryBlock(rewriter));

  SmallVector<Value, 8> args;
  size_t argOffset = resultStructType ? 1 : 0;
  for (auto [index, argType] : llvm::enumerate(type.getInputs())) {
    Value arg = wrapperFuncOp.getArgument(index + argOffset);
    if (auto memrefType = dyn_cast<MemRefType>(argType)) {
      Value loaded = LLVM::LoadOp::create(
          rewriter, loc, typeConverter.convertType(memrefType), arg);
      MemRefDescriptor::unpack(rewriter, loc, loaded, memrefType, args);
      continue;
    }
    if (isa<UnrankedMemRefType>(argType)) {
      Value loaded = LLVM::LoadOp::create(
          rewriter, loc, typeConverter.convertType(argType), arg);
      UnrankedMemRefDescriptor::unpack(rewriter, loc, loaded, args);
      continue;
    }

    args.push_back(arg);
  }

  auto call = LLVM::CallOp::create(rewriter, loc, newFuncOp, args);

  if (resultStructType) {
    LLVM::StoreOp::create(rewriter, loc, call.getResult(),
                          wrapperFuncOp.getArgument(0));
    LLVM::ReturnOp::create(rewriter, loc, ValueRange{});
  } else {
    LLVM::ReturnOp::create(rewriter, loc, call.getResults());
  }
}

/// Creates an auxiliary function with pointer-to-memref-descriptor-struct
/// arguments instead of unpacked arguments. Creates a body for the (external)
/// `newFuncOp` that allocates a memref descriptor on stack, packs the
/// individual arguments into this descriptor and passes a pointer to it into
/// the auxiliary function. If the result of the function cannot be directly
/// returned, we write it to a special first argument that provides a pointer
/// to a corresponding struct. This auxiliary external function is now
/// compatible with functions defined in C using pointers to C structs
/// corresponding to a memref descriptor.
static void wrapExternalFunction(OpBuilder &builder, Location loc,
                                 const LLVMTypeConverter &typeConverter,
                                 FunctionOpInterface funcOp,
                                 LLVM::LLVMFuncOp newFuncOp) {
  OpBuilder::InsertionGuard guard(builder);

  auto [wrapperType, resultStructType] =
      typeConverter.convertFunctionTypeCWrapper(
          cast<FunctionType>(funcOp.getFunctionType()));
  // This conversion can only fail if it could not convert one of the argument
  // types. But since it has been applied to a non-wrapper function before, it
  // should have failed earlier and not reach this point at all.
  assert(wrapperType && "unexpected type conversion failure");

  SmallVector<NamedAttribute, 4> attributes;
  filterFuncAttributes(funcOp, attributes);

  // Create the auxiliary function.
  auto wrapperFunc = LLVM::LLVMFuncOp::create(
      builder, loc, llvm::formatv("_mlir_ciface_{0}", funcOp.getName()).str(),
      wrapperType, LLVM::Linkage::External, /*dsoLocal=*/false,
      /*cconv=*/LLVM::CConv::C, /*comdat=*/nullptr, attributes);
  propagateArgResAttrs(builder, !!resultStructType, funcOp, wrapperFunc);

  // The wrapper that we synthetize here should only be visible in this module.
  newFuncOp.setLinkage(LLVM::Linkage::Private);
  builder.setInsertionPointToStart(newFuncOp.addEntryBlock(builder));

  // Get a ValueRange containing arguments.
  FunctionType type = cast<FunctionType>(funcOp.getFunctionType());
  SmallVector<Value, 8> args;
  args.reserve(type.getNumInputs());
  ValueRange wrapperArgsRange(newFuncOp.getArguments());

  if (resultStructType) {
    // Allocate the struct on the stack and pass the pointer.
    Type resultType = cast<LLVM::LLVMFunctionType>(wrapperType).getParamType(0);
    Value one = LLVM::ConstantOp::create(
        builder, loc, typeConverter.convertType(builder.getIndexType()),
        builder.getIntegerAttr(builder.getIndexType(), 1));
    Value result =
        LLVM::AllocaOp::create(builder, loc, resultType, resultStructType, one);
    args.push_back(result);
  }

  // Iterate over the inputs of the original function and pack values into
  // memref descriptors if the original type is a memref.
  for (Type input : type.getInputs()) {
    Value arg;
    int numToDrop = 1;
    auto memRefType = dyn_cast<MemRefType>(input);
    auto unrankedMemRefType = dyn_cast<UnrankedMemRefType>(input);
    if (memRefType || unrankedMemRefType) {
      numToDrop = memRefType
                      ? MemRefDescriptor::getNumUnpackedValues(memRefType)
                      : UnrankedMemRefDescriptor::getNumUnpackedValues();
      Value packed =
          memRefType
              ? MemRefDescriptor::pack(builder, loc, typeConverter, memRefType,
                                       wrapperArgsRange.take_front(numToDrop))
              : UnrankedMemRefDescriptor::pack(
                    builder, loc, typeConverter, unrankedMemRefType,
                    wrapperArgsRange.take_front(numToDrop));

      auto ptrTy = LLVM::LLVMPointerType::get(builder.getContext());
      Value one = LLVM::ConstantOp::create(
          builder, loc, typeConverter.convertType(builder.getIndexType()),
          builder.getIntegerAttr(builder.getIndexType(), 1));
      Value allocated = LLVM::AllocaOp::create(
          builder, loc, ptrTy, packed.getType(), one, /*alignment=*/0);
      LLVM::StoreOp::create(builder, loc, packed, allocated);
      arg = allocated;
    } else {
      arg = wrapperArgsRange[0];
    }

    args.push_back(arg);
    wrapperArgsRange = wrapperArgsRange.drop_front(numToDrop);
  }
  assert(wrapperArgsRange.empty() && "did not map some of the arguments");

  auto call = LLVM::CallOp::create(builder, loc, wrapperFunc, args);

  if (resultStructType) {
    Value result =
        LLVM::LoadOp::create(builder, loc, resultStructType, args.front());
    LLVM::ReturnOp::create(builder, loc, result);
  } else {
    LLVM::ReturnOp::create(builder, loc, call.getResults());
  }
}

/// Inserts `llvm.load` ops in the function body to restore the expected pointee
/// value from `llvm.byval`/`llvm.byref` function arguments that were converted
/// to LLVM pointer types.
static void restoreByValRefArgumentType(
    ConversionPatternRewriter &rewriter, const LLVMTypeConverter &typeConverter,
    ArrayRef<std::optional<NamedAttribute>> byValRefNonPtrAttrs,
    LLVM::LLVMFuncOp funcOp) {
  // Nothing to do for function declarations.
  if (funcOp.isExternal())
    return;

  ConversionPatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&funcOp.getFunctionBody().front());

  for (const auto &[arg, byValRefAttr] :
       llvm::zip(funcOp.getArguments(), byValRefNonPtrAttrs)) {
    // Skip argument if no `llvm.byval` or `llvm.byref` attribute.
    if (!byValRefAttr)
      continue;

    // Insert load to retrieve the actual argument passed by value/reference.
    assert(isa<LLVM::LLVMPointerType>(arg.getType()) &&
           "Expected LLVM pointer type for argument with "
           "`llvm.byval`/`llvm.byref` attribute");
    Type resTy = typeConverter.convertType(
        cast<TypeAttr>(byValRefAttr->getValue()).getValue());

    Value valueArg = LLVM::LoadOp::create(rewriter, arg.getLoc(), resTy, arg);
    rewriter.replaceUsesOfBlockArgument(arg, valueArg);
  }
}

FailureOr<LLVM::LLVMFuncOp> mlir::convertFuncOpToLLVMFuncOp(
    FunctionOpInterface funcOp, ConversionPatternRewriter &rewriter,
    const LLVMTypeConverter &converter, SymbolTableCollection *symbolTables) {
  // Check the funcOp has `FunctionType`.
  auto funcTy = dyn_cast<FunctionType>(funcOp.getFunctionType());
  if (!funcTy)
    return rewriter.notifyMatchFailure(
        funcOp, "Only support FunctionOpInterface with FunctionType");

  // Convert the original function arguments. They are converted using the
  // LLVMTypeConverter provided to this legalization pattern.
  auto varargsAttr = funcOp->getAttrOfType<BoolAttr>(varargsAttrName);
  // Gather `llvm.byval` and `llvm.byref` arguments whose type convertion was
  // overriden with an LLVM pointer type for later processing.
  SmallVector<std::optional<NamedAttribute>> byValRefNonPtrAttrs;
  TypeConverter::SignatureConversion result(funcOp.getNumArguments());
  auto llvmType = dyn_cast_or_null<LLVM::LLVMFunctionType>(
      converter.convertFunctionSignature(
          funcOp, varargsAttr && varargsAttr.getValue(),
          shouldUseBarePtrCallConv(funcOp, &converter), result,
          byValRefNonPtrAttrs));
  if (!llvmType)
    return rewriter.notifyMatchFailure(funcOp, "signature conversion failed");

  // Check for unsupported variadic functions.
  if (!shouldUseBarePtrCallConv(funcOp, &converter))
    if (funcOp->getAttrOfType<UnitAttr>(
            LLVM::LLVMDialect::getEmitCWrapperAttrName()))
      if (llvmType.isVarArg())
        return funcOp.emitError("C interface for variadic functions is not "
                                "supported yet.");

  // Create an LLVM function, use external linkage by default until MLIR
  // functions have linkage.
  LLVM::Linkage linkage = LLVM::Linkage::External;
  if (funcOp->hasAttr(linkageAttrName)) {
    auto attr =
        dyn_cast<mlir::LLVM::LinkageAttr>(funcOp->getAttr(linkageAttrName));
    if (!attr) {
      funcOp->emitError() << "Contains " << linkageAttrName
                          << " attribute not of type LLVM::LinkageAttr";
      return rewriter.notifyMatchFailure(
          funcOp, "Contains linkage attribute not of type LLVM::LinkageAttr");
    }
    linkage = attr.getLinkage();
  }

  // Check for invalid attributes.
  StringRef readnoneAttrName = LLVM::LLVMDialect::getReadnoneAttrName();
  if (funcOp->hasAttr(readnoneAttrName)) {
    auto attr = funcOp->getAttrOfType<UnitAttr>(readnoneAttrName);
    if (!attr) {
      funcOp->emitError() << "Contains " << readnoneAttrName
                          << " attribute not of type UnitAttr";
      return rewriter.notifyMatchFailure(
          funcOp, "Contains readnone attribute not of type UnitAttr");
    }
  }

  SmallVector<NamedAttribute, 4> attributes;
  filterFuncAttributes(funcOp, attributes);

  Operation *symbolTableOp = funcOp->getParentWithTrait<OpTrait::SymbolTable>();

  if (symbolTables && symbolTableOp) {
    SymbolTable &symbolTable = symbolTables->getSymbolTable(symbolTableOp);
    symbolTable.remove(funcOp);
  }

  auto newFuncOp = LLVM::LLVMFuncOp::create(
      rewriter, funcOp.getLoc(), funcOp.getName(), llvmType, linkage,
      /*dsoLocal=*/false, /*cconv=*/LLVM::CConv::C, /*comdat=*/nullptr,
      attributes);

  if (symbolTables && symbolTableOp) {
    auto ip = rewriter.getInsertionPoint();
    SymbolTable &symbolTable = symbolTables->getSymbolTable(symbolTableOp);
    symbolTable.insert(newFuncOp, ip);
  }

  cast<FunctionOpInterface>(newFuncOp.getOperation())
      .setVisibility(funcOp.getVisibility());

  // Create a memory effect attribute corresponding to readnone.
  if (funcOp->hasAttr(readnoneAttrName)) {
    auto memoryAttr = LLVM::MemoryEffectsAttr::get(
        rewriter.getContext(),
        {LLVM::ModRefInfo::NoModRef, LLVM::ModRefInfo::NoModRef,
         LLVM::ModRefInfo::NoModRef});
    newFuncOp.setMemoryEffectsAttr(memoryAttr);
  }

  // Propagate argument/result attributes to all converted arguments/result
  // obtained after converting a given original argument/result.
  if (ArrayAttr resAttrDicts = funcOp.getAllResultAttrs()) {
    assert(!resAttrDicts.empty() && "expected array to be non-empty");
    if (funcOp.getNumResults() == 1)
      newFuncOp.setAllResultAttrs(resAttrDicts);
  }
  if (ArrayAttr argAttrDicts = funcOp.getAllArgAttrs()) {
    SmallVector<Attribute> newArgAttrs(
        cast<LLVM::LLVMFunctionType>(llvmType).getNumParams());
    for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
      // Some LLVM IR attribute have a type attached to them. During FuncOp ->
      // LLVMFuncOp conversion these types may have changed. Account for that
      // change by converting attributes' types as well.
      SmallVector<NamedAttribute, 4> convertedAttrs;
      auto attrsDict = cast<DictionaryAttr>(argAttrDicts[i]);
      convertedAttrs.reserve(attrsDict.size());
      for (const NamedAttribute &attr : attrsDict) {
        const auto convert = [&](const NamedAttribute &attr) {
          return TypeAttr::get(converter.convertType(
              cast<TypeAttr>(attr.getValue()).getValue()));
        };
        if (attr.getName().getValue() ==
            LLVM::LLVMDialect::getByValAttrName()) {
          convertedAttrs.push_back(rewriter.getNamedAttr(
              LLVM::LLVMDialect::getByValAttrName(), convert(attr)));
        } else if (attr.getName().getValue() ==
                   LLVM::LLVMDialect::getByRefAttrName()) {
          convertedAttrs.push_back(rewriter.getNamedAttr(
              LLVM::LLVMDialect::getByRefAttrName(), convert(attr)));
        } else if (attr.getName().getValue() ==
                   LLVM::LLVMDialect::getStructRetAttrName()) {
          convertedAttrs.push_back(rewriter.getNamedAttr(
              LLVM::LLVMDialect::getStructRetAttrName(), convert(attr)));
        } else if (attr.getName().getValue() ==
                   LLVM::LLVMDialect::getInAllocaAttrName()) {
          convertedAttrs.push_back(rewriter.getNamedAttr(
              LLVM::LLVMDialect::getInAllocaAttrName(), convert(attr)));
        } else {
          convertedAttrs.push_back(attr);
        }
      }
      auto mapping = result.getInputMapping(i);
      assert(mapping && "unexpected deletion of function argument");
      // Only attach the new argument attributes if there is a one-to-one
      // mapping from old to new types. Otherwise, attributes might be
      // attached to types that they do not support.
      if (mapping->size == 1) {
        newArgAttrs[mapping->inputNo] =
            DictionaryAttr::get(rewriter.getContext(), convertedAttrs);
        continue;
      }
      // TODO: Implement custom handling for types that expand to multiple
      // function arguments.
      for (size_t j = 0; j < mapping->size; ++j)
        newArgAttrs[mapping->inputNo + j] =
            DictionaryAttr::get(rewriter.getContext(), {});
    }
    if (!newArgAttrs.empty())
      newFuncOp.setAllArgAttrs(rewriter.getArrayAttr(newArgAttrs));
  }

  rewriter.inlineRegionBefore(funcOp.getFunctionBody(), newFuncOp.getBody(),
                              newFuncOp.end());
  // Convert just the entry block. The remaining unstructured control flow is
  // converted by ControlFlowToLLVM.
  if (!newFuncOp.getBody().empty())
    rewriter.applySignatureConversion(&newFuncOp.getBody().front(), result,
                                      &converter);

  // Fix the type mismatch between the materialized `llvm.ptr` and the expected
  // pointee type in the function body when converting `llvm.byval`/`llvm.byref`
  // function arguments.
  restoreByValRefArgumentType(rewriter, converter, byValRefNonPtrAttrs,
                              newFuncOp);

  if (!shouldUseBarePtrCallConv(funcOp, &converter)) {
    if (funcOp->getAttrOfType<UnitAttr>(
            LLVM::LLVMDialect::getEmitCWrapperAttrName())) {
      if (newFuncOp.isExternal())
        wrapExternalFunction(rewriter, funcOp->getLoc(), converter, funcOp,
                             newFuncOp);
      else
        wrapForExternalCallers(rewriter, funcOp->getLoc(), converter, funcOp,
                               newFuncOp);
    }
  }

  return newFuncOp;
}

namespace {

/// FuncOp legalization pattern that converts MemRef arguments to pointers to
/// MemRef descriptors (LLVM struct data types) containing all the MemRef type
/// information.
class FuncOpConversion : public ConvertOpToLLVMPattern<func::FuncOp> {
  SymbolTableCollection *symbolTables = nullptr;

public:
  explicit FuncOpConversion(const LLVMTypeConverter &converter,
                            SymbolTableCollection *symbolTables = nullptr)
      : ConvertOpToLLVMPattern(converter), symbolTables(symbolTables) {}

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<LLVM::LLVMFuncOp> newFuncOp = mlir::convertFuncOpToLLVMFuncOp(
        cast<FunctionOpInterface>(funcOp.getOperation()), rewriter,
        *getTypeConverter(), symbolTables);
    if (failed(newFuncOp))
      return rewriter.notifyMatchFailure(funcOp, "Could not convert funcop");

    rewriter.eraseOp(funcOp);
    return success();
  }
};

struct ConstantOpLowering : public ConvertOpToLLVMPattern<func::ConstantOp> {
  using ConvertOpToLLVMPattern<func::ConstantOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(func::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = typeConverter->convertType(op.getResult().getType());
    if (!type || !LLVM::isCompatibleType(type))
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    auto newOp =
        LLVM::AddressOfOp::create(rewriter, op.getLoc(), type, op.getValue());
    for (const NamedAttribute &attr : op->getAttrs()) {
      if (attr.getName().strref() == "value")
        continue;
      newOp->setAttr(attr.getName(), attr.getValue());
    }
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

// A CallOp automatically promotes MemRefType to a sequence of alloca/store and
// passes the pointer to the MemRef across function boundaries.
template <typename CallOpType>
struct CallOpInterfaceLowering : public ConvertOpToLLVMPattern<CallOpType> {
  using ConvertOpToLLVMPattern<CallOpType>::ConvertOpToLLVMPattern;
  using Super = CallOpInterfaceLowering<CallOpType>;
  using Base = ConvertOpToLLVMPattern<CallOpType>;
  using Adaptor = typename ConvertOpToLLVMPattern<CallOpType>::OneToNOpAdaptor;

  LogicalResult matchAndRewriteImpl(CallOpType callOp, Adaptor adaptor,
                                    ConversionPatternRewriter &rewriter,
                                    bool useBarePtrCallConv = false) const {
    // Pack the result types into a struct.
    Type packedResult = nullptr;
    SmallVector<SmallVector<Type>> groupedResultTypes;
    unsigned numResults = callOp.getNumResults();
    auto resultTypes = llvm::to_vector<4>(callOp.getResultTypes());
    int64_t numConvertedTypes = 0;
    if (numResults != 0) {
      if (!(packedResult = this->getTypeConverter()->packFunctionResults(
                resultTypes, useBarePtrCallConv, &groupedResultTypes,
                &numConvertedTypes)))
        return failure();
    }

    if (useBarePtrCallConv) {
      for (auto it : callOp->getOperands()) {
        Type operandType = it.getType();
        if (isa<UnrankedMemRefType>(operandType)) {
          // Unranked memref is not supported in the bare pointer calling
          // convention.
          return failure();
        }
      }
    }
    auto promoted = this->getTypeConverter()->promoteOperands(
        callOp.getLoc(), /*opOperands=*/callOp->getOperands(),
        adaptor.getOperands(), rewriter, useBarePtrCallConv);
    auto newOp = LLVM::CallOp::create(rewriter, callOp.getLoc(),
                                      packedResult ? TypeRange(packedResult)
                                                   : TypeRange(),
                                      promoted, callOp->getAttrs());

    newOp.getProperties().operandSegmentSizes = {
        static_cast<int32_t>(promoted.size()), 0};
    newOp.getProperties().op_bundle_sizes = rewriter.getDenseI32ArrayAttr({});

    // Helper function that extracts an individual result from the return value
    // of the new call op. llvm.call ops support only 0 or 1 result. In case of
    // 2 or more results, the results are packed into a structure.
    //
    // The new call op may have more than 2 results because:
    // a. The original call op has more than 2 results.
    // b. An original op result type-converted to more than 1 result.
    auto getUnpackedResult = [&](unsigned i) -> Value {
      assert(numConvertedTypes > 0 && "convert op has no results");
      if (numConvertedTypes == 1) {
        assert(i == 0 && "out of bounds: converted op has only one result");
        return newOp->getResult(0);
      }
      // Results have been converted to a structure. Extract individual results
      // from the structure.
      return LLVM::ExtractValueOp::create(rewriter, callOp.getLoc(),
                                          newOp->getResult(0), i);
    };

    // Group the results into a vector of vectors, such that it is clear which
    // original op result is replaced with which range of values. (In case of a
    // 1:N conversion, there can be multiple replacements for a single result.)
    SmallVector<SmallVector<Value>> results;
    results.reserve(numResults);
    unsigned counter = 0;
    for (unsigned i = 0; i < numResults; ++i) {
      SmallVector<Value> &group = results.emplace_back();
      for (unsigned j = 0, e = groupedResultTypes[i].size(); j < e; ++j)
        group.push_back(getUnpackedResult(counter++));
    }

    // Special handling for MemRef types.
    for (unsigned i = 0; i < numResults; ++i) {
      Type origType = resultTypes[i];
      auto memrefType = dyn_cast<MemRefType>(origType);
      auto unrankedMemrefType = dyn_cast<UnrankedMemRefType>(origType);
      if (useBarePtrCallConv && memrefType) {
        // For the bare-ptr calling convention, promote memref results to
        // descriptors.
        assert(results[i].size() == 1 && "expected one converted result");
        results[i].front() = MemRefDescriptor::fromStaticShape(
            rewriter, callOp.getLoc(), *this->getTypeConverter(), memrefType,
            results[i].front());
      }
      if (unrankedMemrefType) {
        assert(!useBarePtrCallConv && "unranked memref is not supported in the "
                                      "bare-ptr calling convention");
        assert(results[i].size() == 1 && "expected one converted result");
        Value desc = this->copyUnrankedDescriptor(
            rewriter, callOp.getLoc(), unrankedMemrefType, results[i].front(),
            /*toDynamic=*/false);
        if (!desc)
          return failure();
        results[i].front() = desc;
      }
    }

    rewriter.replaceOpWithMultiple(callOp, results);
    return success();
  }
};

class CallOpLowering : public CallOpInterfaceLowering<func::CallOp> {
public:
  explicit CallOpLowering(const LLVMTypeConverter &typeConverter,
                          SymbolTableCollection *symbolTables = nullptr,
                          PatternBenefit benefit = 1)
      : CallOpInterfaceLowering<func::CallOp>(typeConverter, benefit),
        symbolTables(symbolTables) {}

  LogicalResult
  matchAndRewrite(func::CallOp callOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    bool useBarePtrCallConv = false;
    if (getTypeConverter()->getOptions().useBarePtrCallConv) {
      useBarePtrCallConv = true;
    } else if (symbolTables != nullptr) {
      // Fast lookup.
      Operation *callee =
          symbolTables->lookupNearestSymbolFrom(callOp, callOp.getCalleeAttr());
      useBarePtrCallConv =
          callee != nullptr && callee->hasAttr(barePtrAttrName);
    } else {
      // Warning: This is a linear lookup.
      Operation *callee =
          SymbolTable::lookupNearestSymbolFrom(callOp, callOp.getCalleeAttr());
      useBarePtrCallConv =
          callee != nullptr && callee->hasAttr(barePtrAttrName);
    }
    return matchAndRewriteImpl(callOp, adaptor, rewriter, useBarePtrCallConv);
  }

private:
  SymbolTableCollection *symbolTables = nullptr;
};

struct CallIndirectOpLowering
    : public CallOpInterfaceLowering<func::CallIndirectOp> {
  using Super::Super;

  LogicalResult
  matchAndRewrite(func::CallIndirectOp callIndirectOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return matchAndRewriteImpl(callIndirectOp, adaptor, rewriter);
  }
};

struct UnrealizedConversionCastOpLowering
    : public ConvertOpToLLVMPattern<UnrealizedConversionCastOp> {
  using ConvertOpToLLVMPattern<
      UnrealizedConversionCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> convertedTypes;
    if (succeeded(typeConverter->convertTypes(op.getOutputs().getTypes(),
                                              convertedTypes)) &&
        convertedTypes == adaptor.getInputs().getTypes()) {
      rewriter.replaceOp(op, adaptor.getInputs());
      return success();
    }

    convertedTypes.clear();
    if (succeeded(typeConverter->convertTypes(adaptor.getInputs().getTypes(),
                                              convertedTypes)) &&
        convertedTypes == op.getOutputs().getType()) {
      rewriter.replaceOp(op, adaptor.getInputs());
      return success();
    }
    return failure();
  }
};

// Special lowering pattern for `ReturnOps`.  Unlike all other operations,
// `ReturnOp` interacts with the function signature and must have as many
// operands as the function has return values.  Because in LLVM IR, functions
// can only return 0 or 1 value, we pack multiple values into a structure type.
// Emit `PoisonOp` followed by `InsertValueOp`s to create such structure if
// necessary before returning it
struct ReturnOpLowering : public ConvertOpToLLVMPattern<func::ReturnOp> {
  using ConvertOpToLLVMPattern<func::ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value, 4> updatedOperands;

    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    bool useBarePtrCallConv =
        shouldUseBarePtrCallConv(funcOp, this->getTypeConverter());

    for (auto [oldOperand, newOperands] :
         llvm::zip_equal(op->getOperands(), adaptor.getOperands())) {
      Type oldTy = oldOperand.getType();
      if (auto memRefType = dyn_cast<MemRefType>(oldTy)) {
        assert(newOperands.size() == 1 && "expected one converted result");
        if (useBarePtrCallConv &&
            getTypeConverter()->canConvertToBarePtr(memRefType)) {
          // For the bare-ptr calling convention, extract the aligned pointer to
          // be returned from the memref descriptor.
          MemRefDescriptor memrefDesc(newOperands.front());
          updatedOperands.push_back(memrefDesc.allocatedPtr(rewriter, loc));
          continue;
        }
      } else if (auto unrankedMemRefType =
                     dyn_cast<UnrankedMemRefType>(oldTy)) {
        assert(newOperands.size() == 1 && "expected one converted result");
        if (useBarePtrCallConv) {
          // Unranked memref is not supported in the bare pointer calling
          // convention.
          return failure();
        }
        Value updatedDesc =
            copyUnrankedDescriptor(rewriter, loc, unrankedMemRefType,
                                   newOperands.front(), /*toDynamic=*/true);
        if (!updatedDesc)
          return failure();
        updatedOperands.push_back(updatedDesc);
        continue;
      }

      llvm::append_range(updatedOperands, newOperands);
    }

    // If ReturnOp has 0 or 1 operand, create it and return immediately.
    if (updatedOperands.size() <= 1) {
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

    Value packed = LLVM::PoisonOp::create(rewriter, loc, packedType);
    for (auto [idx, operand] : llvm::enumerate(updatedOperands)) {
      packed = LLVM::InsertValueOp::create(rewriter, loc, packed, operand, idx);
    }
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(), packed,
                                                op->getAttrs());
    return success();
  }
};
} // namespace

void mlir::populateFuncToLLVMFuncOpConversionPattern(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns,
    SymbolTableCollection *symbolTables) {
  patterns.add<FuncOpConversion>(converter, symbolTables);
}

void mlir::populateFuncToLLVMConversionPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns,
    SymbolTableCollection *symbolTables) {
  populateFuncToLLVMFuncOpConversionPattern(converter, patterns, symbolTables);
  patterns.add<CallIndirectOpLowering>(converter);
  patterns.add<CallOpLowering>(converter, symbolTables);
  patterns.add<ConstantOpLowering>(converter);
  patterns.add<ReturnOpLowering>(converter);
}

namespace {
/// A pass converting Func operations into the LLVM IR dialect.
struct ConvertFuncToLLVMPass
    : public impl::ConvertFuncToLLVMPassBase<ConvertFuncToLLVMPass> {
  using Base::Base;

  /// Run the dialect converter on the module.
  void runOnOperation() override {
    ModuleOp m = getOperation();
    StringRef dataLayout;
    auto dataLayoutAttr = dyn_cast_or_null<StringAttr>(
        m->getAttr(LLVM::LLVMDialect::getDataLayoutAttrName()));
    if (dataLayoutAttr)
      dataLayout = dataLayoutAttr.getValue();

    if (failed(LLVM::LLVMDialect::verifyDataLayoutString(
            dataLayout, [this](const Twine &message) {
              getOperation().emitError() << message.str();
            }))) {
      signalPassFailure();
      return;
    }

    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();

    LowerToLLVMOptions options(&getContext(),
                               dataLayoutAnalysis.getAtOrAbove(m));
    options.useBarePtrCallConv = useBarePtrCallConv;
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);
    options.dataLayout = llvm::DataLayout(dataLayout);

    LLVMTypeConverter typeConverter(&getContext(), options,
                                    &dataLayoutAnalysis);

    RewritePatternSet patterns(&getContext());
    SymbolTableCollection symbolTables;

    populateFuncToLLVMConversionPatterns(typeConverter, patterns,
                                         &symbolTables);

    LLVMConversionTarget target(getContext());
    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();
  }
};

struct SetLLVMModuleDataLayoutPass
    : public impl::SetLLVMModuleDataLayoutPassBase<
          SetLLVMModuleDataLayoutPass> {
  using Base::Base;

  /// Run the dialect converter on the module.
  void runOnOperation() override {
    if (failed(LLVM::LLVMDialect::verifyDataLayoutString(
            this->dataLayout, [this](const Twine &message) {
              getOperation().emitError() << message.str();
            }))) {
      signalPassFailure();
      return;
    }
    ModuleOp m = getOperation();
    m->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(),
               StringAttr::get(m.getContext(), this->dataLayout));
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ConvertToLLVMPatternInterface implementation
//===----------------------------------------------------------------------===//

namespace {
/// Implement the interface to convert Func to LLVM.
struct FuncToLLVMDialectInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;
  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  }
};
} // namespace

void mlir::registerConvertFuncToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, func::FuncDialect *dialect) {
    dialect->addInterfaces<FuncToLLVMDialectInterface>();
  });
}
