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
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include <algorithm>
#include <functional>
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

  auto wrapperFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
      loc, llvm::formatv("_mlir_ciface_{0}", funcOp.getName()).str(),
      wrapperFuncType, LLVM::Linkage::External, /*dsoLocal=*/false,
      /*cconv=*/LLVM::CConv::C, /*comdat=*/nullptr, attributes);
  propagateArgResAttrs(rewriter, !!resultStructType, funcOp, wrapperFuncOp);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(wrapperFuncOp.addEntryBlock());

  SmallVector<Value, 8> args;
  size_t argOffset = resultStructType ? 1 : 0;
  for (auto [index, argType] : llvm::enumerate(type.getInputs())) {
    Value arg = wrapperFuncOp.getArgument(index + argOffset);
    if (auto memrefType = dyn_cast<MemRefType>(argType)) {
      Value loaded = rewriter.create<LLVM::LoadOp>(
          loc, typeConverter.convertType(memrefType), arg);
      MemRefDescriptor::unpack(rewriter, loc, loaded, memrefType, args);
      continue;
    }
    if (isa<UnrankedMemRefType>(argType)) {
      Value loaded = rewriter.create<LLVM::LoadOp>(
          loc, typeConverter.convertType(argType), arg);
      UnrankedMemRefDescriptor::unpack(rewriter, loc, loaded, args);
      continue;
    }

    args.push_back(arg);
  }

  auto call = rewriter.create<LLVM::CallOp>(loc, newFuncOp, args);

  if (resultStructType) {
    rewriter.create<LLVM::StoreOp>(loc, call.getResult(),
                                   wrapperFuncOp.getArgument(0));
    rewriter.create<LLVM::ReturnOp>(loc, ValueRange{});
  } else {
    rewriter.create<LLVM::ReturnOp>(loc, call.getResults());
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
  auto wrapperFunc = builder.create<LLVM::LLVMFuncOp>(
      loc, llvm::formatv("_mlir_ciface_{0}", funcOp.getName()).str(),
      wrapperType, LLVM::Linkage::External, /*dsoLocal=*/false,
      /*cconv=*/LLVM::CConv::C, /*comdat=*/nullptr, attributes);
  propagateArgResAttrs(builder, !!resultStructType, funcOp, wrapperFunc);

  // The wrapper that we synthetize here should only be visible in this module.
  newFuncOp.setLinkage(LLVM::Linkage::Private);
  builder.setInsertionPointToStart(newFuncOp.addEntryBlock());

  // Get a ValueRange containing arguments.
  FunctionType type = cast<FunctionType>(funcOp.getFunctionType());
  SmallVector<Value, 8> args;
  args.reserve(type.getNumInputs());
  ValueRange wrapperArgsRange(newFuncOp.getArguments());

  if (resultStructType) {
    // Allocate the struct on the stack and pass the pointer.
    Type resultType = cast<LLVM::LLVMFunctionType>(wrapperType).getParamType(0);
    Value one = builder.create<LLVM::ConstantOp>(
        loc, typeConverter.convertType(builder.getIndexType()),
        builder.getIntegerAttr(builder.getIndexType(), 1));
    Value result =
        builder.create<LLVM::AllocaOp>(loc, resultType, resultStructType, one);
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
      Value one = builder.create<LLVM::ConstantOp>(
          loc, typeConverter.convertType(builder.getIndexType()),
          builder.getIntegerAttr(builder.getIndexType(), 1));
      Value allocated = builder.create<LLVM::AllocaOp>(
          loc, ptrTy, packed.getType(), one, /*alignment=*/0);
      builder.create<LLVM::StoreOp>(loc, packed, allocated);
      arg = allocated;
    } else {
      arg = wrapperArgsRange[0];
    }

    args.push_back(arg);
    wrapperArgsRange = wrapperArgsRange.drop_front(numToDrop);
  }
  assert(wrapperArgsRange.empty() && "did not map some of the arguments");

  auto call = builder.create<LLVM::CallOp>(loc, wrapperFunc, args);

  if (resultStructType) {
    Value result =
        builder.create<LLVM::LoadOp>(loc, resultStructType, args.front());
    builder.create<LLVM::ReturnOp>(loc, result);
  } else {
    builder.create<LLVM::ReturnOp>(loc, call.getResults());
  }
}

/// Modifies the body of the function to construct the `MemRefDescriptor` from
/// the bare pointer calling convention lowering of `memref` types.
static void modifyFuncOpToUseBarePtrCallingConv(
    ConversionPatternRewriter &rewriter, Location loc,
    const LLVMTypeConverter &typeConverter, LLVM::LLVMFuncOp funcOp,
    TypeRange oldArgTypes) {
  if (funcOp.getBody().empty())
    return;

  // Promote bare pointers from memref arguments to memref descriptors at the
  // beginning of the function so that all the memrefs in the function have a
  // uniform representation.
  Block *entryBlock = &funcOp.getBody().front();
  auto blockArgs = entryBlock->getArguments();
  assert(blockArgs.size() == oldArgTypes.size() &&
         "The number of arguments and types doesn't match");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(entryBlock);
  for (auto it : llvm::zip(blockArgs, oldArgTypes)) {
    BlockArgument arg = std::get<0>(it);
    Type argTy = std::get<1>(it);

    // Unranked memrefs are not supported in the bare pointer calling
    // convention. We should have bailed out before in the presence of
    // unranked memrefs.
    assert(!isa<UnrankedMemRefType>(argTy) &&
           "Unranked memref is not supported");
    auto memrefTy = dyn_cast<MemRefType>(argTy);
    if (!memrefTy)
      continue;

    // Replace barePtr with a placeholder (undef), promote barePtr to a ranked
    // or unranked memref descriptor and replace placeholder with the last
    // instruction of the memref descriptor.
    // TODO: The placeholder is needed to avoid replacing barePtr uses in the
    // MemRef descriptor instructions. We may want to have a utility in the
    // rewriter to properly handle this use case.
    Location loc = funcOp.getLoc();
    auto placeholder = rewriter.create<LLVM::UndefOp>(
        loc, typeConverter.convertType(memrefTy));
    rewriter.replaceUsesOfBlockArgument(arg, placeholder);

    Value desc = MemRefDescriptor::fromStaticShape(rewriter, loc, typeConverter,
                                                   memrefTy, arg);
    rewriter.replaceOp(placeholder, {desc});
  }
}

FailureOr<LLVM::LLVMFuncOp>
mlir::convertFuncOpToLLVMFuncOp(FunctionOpInterface funcOp,
                                ConversionPatternRewriter &rewriter,
                                const LLVMTypeConverter &converter) {
  // Check the funcOp has `FunctionType`.
  auto funcTy = dyn_cast<FunctionType>(funcOp.getFunctionType());
  if (!funcTy)
    return rewriter.notifyMatchFailure(
        funcOp, "Only support FunctionOpInterface with FunctionType");

  // Convert the original function arguments. They are converted using the
  // LLVMTypeConverter provided to this legalization pattern.
  auto varargsAttr = funcOp->getAttrOfType<BoolAttr>(varargsAttrName);
  TypeConverter::SignatureConversion result(funcOp.getNumArguments());
  auto llvmType = converter.convertFunctionSignature(
      funcTy, varargsAttr && varargsAttr.getValue(),
      shouldUseBarePtrCallConv(funcOp, &converter), result);
  if (!llvmType)
    return rewriter.notifyMatchFailure(funcOp, "signature conversion failed");

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

  SmallVector<NamedAttribute, 4> attributes;
  filterFuncAttributes(funcOp, attributes);
  auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
      funcOp.getLoc(), funcOp.getName(), llvmType, linkage,
      /*dsoLocal=*/false, /*cconv=*/LLVM::CConv::C, /*comdat=*/nullptr,
      attributes);
  cast<FunctionOpInterface>(newFuncOp.getOperation())
      .setVisibility(funcOp.getVisibility());

  // Create a memory effect attribute corresponding to readnone.
  StringRef readnoneAttrName = LLVM::LLVMDialect::getReadnoneAttrName();
  if (funcOp->hasAttr(readnoneAttrName)) {
    auto attr = funcOp->getAttrOfType<UnitAttr>(readnoneAttrName);
    if (!attr) {
      funcOp->emitError() << "Contains " << readnoneAttrName
                          << " attribute not of type UnitAttr";
      return rewriter.notifyMatchFailure(
          funcOp, "Contains readnone attribute not of type UnitAttr");
    }
    auto memoryAttr = LLVM::MemoryEffectsAttr::get(
        rewriter.getContext(),
        {LLVM::ModRefInfo::NoModRef, LLVM::ModRefInfo::NoModRef,
         LLVM::ModRefInfo::NoModRef});
    newFuncOp.setMemoryAttr(memoryAttr);
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
  if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), converter,
                                         &result))) {
    return rewriter.notifyMatchFailure(funcOp,
                                       "region types conversion failed");
  }

  return newFuncOp;
}

namespace {

struct FuncOpConversionBase : public ConvertOpToLLVMPattern<func::FuncOp> {
protected:
  using ConvertOpToLLVMPattern<func::FuncOp>::ConvertOpToLLVMPattern;

  // Convert input FuncOp to LLVMFuncOp by using the LLVMTypeConverter provided
  // to this legalization pattern.
  FailureOr<LLVM::LLVMFuncOp>
  convertFuncOpToLLVMFuncOp(func::FuncOp funcOp,
                            ConversionPatternRewriter &rewriter) const {
    return mlir::convertFuncOpToLLVMFuncOp(
        cast<FunctionOpInterface>(funcOp.getOperation()), rewriter,
        *getTypeConverter());
  }
};

/// FuncOp legalization pattern that converts MemRef arguments to pointers to
/// MemRef descriptors (LLVM struct data types) containing all the MemRef type
/// information.
struct FuncOpConversion : public FuncOpConversionBase {
  FuncOpConversion(const LLVMTypeConverter &converter)
      : FuncOpConversionBase(converter) {}

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<LLVM::LLVMFuncOp> newFuncOp =
        convertFuncOpToLLVMFuncOp(funcOp, rewriter);
    if (failed(newFuncOp))
      return rewriter.notifyMatchFailure(funcOp, "Could not convert funcop");

    if (!shouldUseBarePtrCallConv(funcOp, this->getTypeConverter())) {
      if (funcOp->getAttrOfType<UnitAttr>(
              LLVM::LLVMDialect::getEmitCWrapperAttrName())) {
        if (newFuncOp->isVarArg())
          return funcOp->emitError("C interface for variadic functions is not "
                                   "supported yet.");

        if (newFuncOp->isExternal())
          wrapExternalFunction(rewriter, funcOp->getLoc(), *getTypeConverter(),
                               funcOp, *newFuncOp);
        else
          wrapForExternalCallers(rewriter, funcOp->getLoc(),
                                 *getTypeConverter(), funcOp, *newFuncOp);
      }
    } else {
      modifyFuncOpToUseBarePtrCallingConv(rewriter, funcOp->getLoc(),
                                          *getTypeConverter(), *newFuncOp,
                                          funcOp.getFunctionType().getInputs());
    }

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
        rewriter.create<LLVM::AddressOfOp>(op.getLoc(), type, op.getValue());
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

  LogicalResult matchAndRewriteImpl(CallOpType callOp,
                                    typename CallOpType::Adaptor adaptor,
                                    ConversionPatternRewriter &rewriter,
                                    bool useBarePtrCallConv = false) const {
    // Pack the result types into a struct.
    Type packedResult = nullptr;
    unsigned numResults = callOp.getNumResults();
    auto resultTypes = llvm::to_vector<4>(callOp.getResultTypes());

    if (numResults != 0) {
      if (!(packedResult = this->getTypeConverter()->packFunctionResults(
                resultTypes, useBarePtrCallConv)))
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
    auto newOp = rewriter.create<LLVM::CallOp>(
        callOp.getLoc(), packedResult ? TypeRange(packedResult) : TypeRange(),
        promoted, callOp->getAttrs());

    SmallVector<Value, 4> results;
    if (numResults < 2) {
      // If < 2 results, packing did not do anything and we can just return.
      results.append(newOp.result_begin(), newOp.result_end());
    } else {
      // Otherwise, it had been converted to an operation producing a structure.
      // Extract individual results from the structure and return them as list.
      results.reserve(numResults);
      for (unsigned i = 0; i < numResults; ++i) {
        results.push_back(rewriter.create<LLVM::ExtractValueOp>(
            callOp.getLoc(), newOp->getResult(0), i));
      }
    }

    if (useBarePtrCallConv) {
      // For the bare-ptr calling convention, promote memref results to
      // descriptors.
      assert(results.size() == resultTypes.size() &&
             "The number of arguments and types doesn't match");
      this->getTypeConverter()->promoteBarePtrsToDescriptors(
          rewriter, callOp.getLoc(), resultTypes, results);
    } else if (failed(this->copyUnrankedDescriptors(rewriter, callOp.getLoc(),
                                                    resultTypes, results,
                                                    /*toDynamic=*/false))) {
      return failure();
    }

    rewriter.replaceOp(callOp, results);
    return success();
  }
};

class CallOpLowering : public CallOpInterfaceLowering<func::CallOp> {
public:
  CallOpLowering(const LLVMTypeConverter &typeConverter,
                 // Can be nullptr.
                 const SymbolTable *symbolTable, PatternBenefit benefit = 1)
      : CallOpInterfaceLowering<func::CallOp>(typeConverter, benefit),
        symbolTable(symbolTable) {}

  LogicalResult
  matchAndRewrite(func::CallOp callOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    bool useBarePtrCallConv = false;
    if (getTypeConverter()->getOptions().useBarePtrCallConv) {
      useBarePtrCallConv = true;
    } else if (symbolTable != nullptr) {
      // Fast lookup.
      Operation *callee =
          symbolTable->lookup(callOp.getCalleeAttr().getValue());
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
  const SymbolTable *symbolTable = nullptr;
};

struct CallIndirectOpLowering
    : public CallOpInterfaceLowering<func::CallIndirectOp> {
  using Super::Super;

  LogicalResult
  matchAndRewrite(func::CallIndirectOp callIndirectOp, OpAdaptor adaptor,
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
// Emit `UndefOp` followed by `InsertValueOp`s to create such structure if
// necessary before returning it
struct ReturnOpLowering : public ConvertOpToLLVMPattern<func::ReturnOp> {
  using ConvertOpToLLVMPattern<func::ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned numArguments = op.getNumOperands();
    SmallVector<Value, 4> updatedOperands;

    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    bool useBarePtrCallConv =
        shouldUseBarePtrCallConv(funcOp, this->getTypeConverter());
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
};
} // namespace

void mlir::populateFuncToLLVMFuncOpConversionPattern(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<FuncOpConversion>(converter);
}

void mlir::populateFuncToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns,
    const SymbolTable *symbolTable) {
  populateFuncToLLVMFuncOpConversionPattern(converter, patterns);
  patterns.add<CallIndirectOpLowering>(converter);
  patterns.add<CallOpLowering>(converter, symbolTable);
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

    std::optional<SymbolTable> optSymbolTable = std::nullopt;
    const SymbolTable *symbolTable = nullptr;
    if (!options.useBarePtrCallConv) {
      optSymbolTable.emplace(m);
      symbolTable = &optSymbolTable.value();
    }

    RewritePatternSet patterns(&getContext());
    populateFuncToLLVMConversionPatterns(typeConverter, patterns, symbolTable);

    // TODO(https://github.com/llvm/llvm-project/issues/70982): Remove these in
    // favor of their dedicated conversion passes.
    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

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
