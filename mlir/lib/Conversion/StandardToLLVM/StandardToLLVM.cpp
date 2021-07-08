//===- StandardToLLVM.cpp - Standard to LLVM dialect conversion -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert MLIR standard and builtin dialects
// into the LLVM IR dialect.
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include <functional>

using namespace mlir;

#define PASS_NAME "convert-std-to-llvm"

/// Only retain those attributes that are not constructed by
/// `LLVMFuncOp::build`. If `filterArgAttrs` is set, also filter out argument
/// attributes.
static void filterFuncAttributes(ArrayRef<NamedAttribute> attrs,
                                 bool filterArgAttrs,
                                 SmallVectorImpl<NamedAttribute> &result) {
  for (const auto &attr : attrs) {
    if (attr.first == SymbolTable::getSymbolAttrName() ||
        attr.first == function_like_impl::getTypeAttrName() ||
        attr.first == "std.varargs" ||
        (filterArgAttrs &&
         attr.first == function_like_impl::getArgDictAttrName()))
      continue;
    result.push_back(attr);
  }
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
                                   LLVMTypeConverter &typeConverter,
                                   FuncOp funcOp, LLVM::LLVMFuncOp newFuncOp) {
  auto type = funcOp.getType();
  SmallVector<NamedAttribute, 4> attributes;
  filterFuncAttributes(funcOp->getAttrs(), /*filterArgAttrs=*/false,
                       attributes);
  Type wrapperFuncType;
  bool resultIsNowArg;
  std::tie(wrapperFuncType, resultIsNowArg) =
      typeConverter.convertFunctionTypeCWrapper(type);
  auto wrapperFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
      loc, llvm::formatv("_mlir_ciface_{0}", funcOp.getName()).str(),
      wrapperFuncType, LLVM::Linkage::External, /*dsoLocal*/ false, attributes);

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(wrapperFuncOp.addEntryBlock());

  SmallVector<Value, 8> args;
  size_t argOffset = resultIsNowArg ? 1 : 0;
  for (auto &en : llvm::enumerate(type.getInputs())) {
    Value arg = wrapperFuncOp.getArgument(en.index() + argOffset);
    if (auto memrefType = en.value().dyn_cast<MemRefType>()) {
      Value loaded = rewriter.create<LLVM::LoadOp>(loc, arg);
      MemRefDescriptor::unpack(rewriter, loc, loaded, memrefType, args);
      continue;
    }
    if (en.value().isa<UnrankedMemRefType>()) {
      Value loaded = rewriter.create<LLVM::LoadOp>(loc, arg);
      UnrankedMemRefDescriptor::unpack(rewriter, loc, loaded, args);
      continue;
    }

    args.push_back(arg);
  }

  auto call = rewriter.create<LLVM::CallOp>(loc, newFuncOp, args);

  if (resultIsNowArg) {
    rewriter.create<LLVM::StoreOp>(loc, call.getResult(0),
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
                                 LLVMTypeConverter &typeConverter,
                                 FuncOp funcOp, LLVM::LLVMFuncOp newFuncOp) {
  OpBuilder::InsertionGuard guard(builder);

  Type wrapperType;
  bool resultIsNowArg;
  std::tie(wrapperType, resultIsNowArg) =
      typeConverter.convertFunctionTypeCWrapper(funcOp.getType());
  // This conversion can only fail if it could not convert one of the argument
  // types. But since it has been applied to a non-wrapper function before, it
  // should have failed earlier and not reach this point at all.
  assert(wrapperType && "unexpected type conversion failure");

  SmallVector<NamedAttribute, 4> attributes;
  filterFuncAttributes(funcOp->getAttrs(), /*filterArgAttrs=*/false,
                       attributes);

  // Create the auxiliary function.
  auto wrapperFunc = builder.create<LLVM::LLVMFuncOp>(
      loc, llvm::formatv("_mlir_ciface_{0}", funcOp.getName()).str(),
      wrapperType, LLVM::Linkage::External, /*dsoLocal*/ false, attributes);

  builder.setInsertionPointToStart(newFuncOp.addEntryBlock());

  // Get a ValueRange containing arguments.
  FunctionType type = funcOp.getType();
  SmallVector<Value, 8> args;
  args.reserve(type.getNumInputs());
  ValueRange wrapperArgsRange(newFuncOp.getArguments());

  if (resultIsNowArg) {
    // Allocate the struct on the stack and pass the pointer.
    Type resultType =
        wrapperType.cast<LLVM::LLVMFunctionType>().getParamType(0);
    Value one = builder.create<LLVM::ConstantOp>(
        loc, typeConverter.convertType(builder.getIndexType()),
        builder.getIntegerAttr(builder.getIndexType(), 1));
    Value result = builder.create<LLVM::AllocaOp>(loc, resultType, one);
    args.push_back(result);
  }

  // Iterate over the inputs of the original function and pack values into
  // memref descriptors if the original type is a memref.
  for (auto &en : llvm::enumerate(type.getInputs())) {
    Value arg;
    int numToDrop = 1;
    auto memRefType = en.value().dyn_cast<MemRefType>();
    auto unrankedMemRefType = en.value().dyn_cast<UnrankedMemRefType>();
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

      auto ptrTy = LLVM::LLVMPointerType::get(packed.getType());
      Value one = builder.create<LLVM::ConstantOp>(
          loc, typeConverter.convertType(builder.getIndexType()),
          builder.getIntegerAttr(builder.getIndexType(), 1));
      Value allocated =
          builder.create<LLVM::AllocaOp>(loc, ptrTy, one, /*alignment=*/0);
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

  if (resultIsNowArg) {
    Value result = builder.create<LLVM::LoadOp>(loc, args.front());
    builder.create<LLVM::ReturnOp>(loc, ValueRange{result});
  } else {
    builder.create<LLVM::ReturnOp>(loc, call.getResults());
  }
}

namespace {

struct FuncOpConversionBase : public ConvertOpToLLVMPattern<FuncOp> {
protected:
  using ConvertOpToLLVMPattern<FuncOp>::ConvertOpToLLVMPattern;

  // Convert input FuncOp to LLVMFuncOp by using the LLVMTypeConverter provided
  // to this legalization pattern.
  LLVM::LLVMFuncOp
  convertFuncOpToLLVMFuncOp(FuncOp funcOp,
                            ConversionPatternRewriter &rewriter) const {
    // Convert the original function arguments. They are converted using the
    // LLVMTypeConverter provided to this legalization pattern.
    auto varargsAttr = funcOp->getAttrOfType<BoolAttr>("std.varargs");
    TypeConverter::SignatureConversion result(funcOp.getNumArguments());
    auto llvmType = getTypeConverter()->convertFunctionSignature(
        funcOp.getType(), varargsAttr && varargsAttr.getValue(), result);
    if (!llvmType)
      return nullptr;

    // Propagate argument attributes to all converted arguments obtained after
    // converting a given original argument.
    SmallVector<NamedAttribute, 4> attributes;
    filterFuncAttributes(funcOp->getAttrs(), /*filterArgAttrs=*/true,
                         attributes);
    if (ArrayAttr argAttrDicts = funcOp.getAllArgAttrs()) {
      SmallVector<Attribute, 4> newArgAttrs(
          llvmType.cast<LLVM::LLVMFunctionType>().getNumParams());
      for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
        auto mapping = result.getInputMapping(i);
        assert(mapping.hasValue() &&
               "unexpected deletion of function argument");
        for (size_t j = 0; j < mapping->size; ++j)
          newArgAttrs[mapping->inputNo + j] = argAttrDicts[i];
      }
      attributes.push_back(
          rewriter.getNamedAttr(function_like_impl::getArgDictAttrName(),
                                rewriter.getArrayAttr(newArgAttrs)));
    }

    // Create an LLVM function, use external linkage by default until MLIR
    // functions have linkage.
    auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        funcOp.getLoc(), funcOp.getName(), llvmType, LLVM::Linkage::External,
        /*dsoLocal*/ false, attributes);
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter,
                                           &result)))
      return nullptr;

    return newFuncOp;
  }
};

/// FuncOp legalization pattern that converts MemRef arguments to pointers to
/// MemRef descriptors (LLVM struct data types) containing all the MemRef type
/// information.
static constexpr StringRef kEmitIfaceAttrName = "llvm.emit_c_interface";
struct FuncOpConversion : public FuncOpConversionBase {
  FuncOpConversion(LLVMTypeConverter &converter)
      : FuncOpConversionBase(converter) {}

  LogicalResult
  matchAndRewrite(FuncOp funcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto newFuncOp = convertFuncOpToLLVMFuncOp(funcOp, rewriter);
    if (!newFuncOp)
      return failure();

    if (getTypeConverter()->getOptions().emitCWrappers ||
        funcOp->getAttrOfType<UnitAttr>(kEmitIfaceAttrName)) {
      if (newFuncOp.isExternal())
        wrapExternalFunction(rewriter, funcOp.getLoc(), *getTypeConverter(),
                             funcOp, newFuncOp);
      else
        wrapForExternalCallers(rewriter, funcOp.getLoc(), *getTypeConverter(),
                               funcOp, newFuncOp);
    }

    rewriter.eraseOp(funcOp);
    return success();
  }
};

/// FuncOp legalization pattern that converts MemRef arguments to bare pointers
/// to the MemRef element type. This will impact the calling convention and ABI.
struct BarePtrFuncOpConversion : public FuncOpConversionBase {
  using FuncOpConversionBase::FuncOpConversionBase;

  LogicalResult
  matchAndRewrite(FuncOp funcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Store the type of memref-typed arguments before the conversion so that we
    // can promote them to MemRef descriptor at the beginning of the function.
    SmallVector<Type, 8> oldArgTypes =
        llvm::to_vector<8>(funcOp.getType().getInputs());

    auto newFuncOp = convertFuncOpToLLVMFuncOp(funcOp, rewriter);
    if (!newFuncOp)
      return failure();
    if (newFuncOp.getBody().empty()) {
      rewriter.eraseOp(funcOp);
      return success();
    }

    // Promote bare pointers from memref arguments to memref descriptors at the
    // beginning of the function so that all the memrefs in the function have a
    // uniform representation.
    Block *entryBlock = &newFuncOp.getBody().front();
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
      assert(!argTy.isa<UnrankedMemRefType>() &&
             "Unranked memref is not supported");
      auto memrefTy = argTy.dyn_cast<MemRefType>();
      if (!memrefTy)
        continue;

      // Replace barePtr with a placeholder (undef), promote barePtr to a ranked
      // or unranked memref descriptor and replace placeholder with the last
      // instruction of the memref descriptor.
      // TODO: The placeholder is needed to avoid replacing barePtr uses in the
      // MemRef descriptor instructions. We may want to have a utility in the
      // rewriter to properly handle this use case.
      Location loc = funcOp.getLoc();
      auto placeholder = rewriter.create<LLVM::UndefOp>(loc, memrefTy);
      rewriter.replaceUsesOfBlockArgument(arg, placeholder);

      Value desc = MemRefDescriptor::fromStaticShape(
          rewriter, loc, *getTypeConverter(), memrefTy, arg);
      rewriter.replaceOp(placeholder, {desc});
    }

    rewriter.eraseOp(funcOp);
    return success();
  }
};

// Straightforward lowerings.
using AbsFOpLowering = VectorConvertToLLVMPattern<AbsFOp, LLVM::FAbsOp>;
using AddFOpLowering = VectorConvertToLLVMPattern<AddFOp, LLVM::FAddOp>;
using AddIOpLowering = VectorConvertToLLVMPattern<AddIOp, LLVM::AddOp>;
using AndOpLowering = VectorConvertToLLVMPattern<AndOp, LLVM::AndOp>;
using CeilFOpLowering = VectorConvertToLLVMPattern<CeilFOp, LLVM::FCeilOp>;
using CopySignOpLowering =
    VectorConvertToLLVMPattern<CopySignOp, LLVM::CopySignOp>;
using CosOpLowering = VectorConvertToLLVMPattern<math::CosOp, LLVM::CosOp>;
using DivFOpLowering = VectorConvertToLLVMPattern<DivFOp, LLVM::FDivOp>;
using ExpOpLowering = VectorConvertToLLVMPattern<math::ExpOp, LLVM::ExpOp>;
using Exp2OpLowering = VectorConvertToLLVMPattern<math::Exp2Op, LLVM::Exp2Op>;
using FPExtOpLowering = VectorConvertToLLVMPattern<FPExtOp, LLVM::FPExtOp>;
using FPToSIOpLowering = VectorConvertToLLVMPattern<FPToSIOp, LLVM::FPToSIOp>;
using FPToUIOpLowering = VectorConvertToLLVMPattern<FPToUIOp, LLVM::FPToUIOp>;
using FPTruncOpLowering = VectorConvertToLLVMPattern<FPTruncOp, LLVM::FPTruncOp>;
using FloorFOpLowering = VectorConvertToLLVMPattern<FloorFOp, LLVM::FFloorOp>;
using FmaFOpLowering = VectorConvertToLLVMPattern<FmaFOp, LLVM::FMAOp>;
using Log10OpLowering =
    VectorConvertToLLVMPattern<math::Log10Op, LLVM::Log10Op>;
using Log2OpLowering = VectorConvertToLLVMPattern<math::Log2Op, LLVM::Log2Op>;
using LogOpLowering = VectorConvertToLLVMPattern<math::LogOp, LLVM::LogOp>;
using MulFOpLowering = VectorConvertToLLVMPattern<MulFOp, LLVM::FMulOp>;
using MulIOpLowering = VectorConvertToLLVMPattern<MulIOp, LLVM::MulOp>;
using NegFOpLowering = VectorConvertToLLVMPattern<NegFOp, LLVM::FNegOp>;
using OrOpLowering = VectorConvertToLLVMPattern<OrOp, LLVM::OrOp>;
using PowFOpLowering = VectorConvertToLLVMPattern<math::PowFOp, LLVM::PowOp>;
using RemFOpLowering = VectorConvertToLLVMPattern<RemFOp, LLVM::FRemOp>;
using SIToFPOpLowering = VectorConvertToLLVMPattern<SIToFPOp, LLVM::SIToFPOp>;
using SelectOpLowering = VectorConvertToLLVMPattern<SelectOp, LLVM::SelectOp>;
using SignExtendIOpLowering =
    VectorConvertToLLVMPattern<SignExtendIOp, LLVM::SExtOp>;
using ShiftLeftOpLowering =
    OneToOneConvertToLLVMPattern<ShiftLeftOp, LLVM::ShlOp>;
using SignedDivIOpLowering =
    VectorConvertToLLVMPattern<SignedDivIOp, LLVM::SDivOp>;
using SignedRemIOpLowering =
    VectorConvertToLLVMPattern<SignedRemIOp, LLVM::SRemOp>;
using SignedShiftRightOpLowering =
    OneToOneConvertToLLVMPattern<SignedShiftRightOp, LLVM::AShrOp>;
using SinOpLowering = VectorConvertToLLVMPattern<math::SinOp, LLVM::SinOp>;
using SqrtOpLowering = VectorConvertToLLVMPattern<math::SqrtOp, LLVM::SqrtOp>;
using SubFOpLowering = VectorConvertToLLVMPattern<SubFOp, LLVM::FSubOp>;
using SubIOpLowering = VectorConvertToLLVMPattern<SubIOp, LLVM::SubOp>;
using TruncateIOpLowering = VectorConvertToLLVMPattern<TruncateIOp, LLVM::TruncOp>;
using UIToFPOpLowering = VectorConvertToLLVMPattern<UIToFPOp, LLVM::UIToFPOp>;
using UnsignedDivIOpLowering =
    VectorConvertToLLVMPattern<UnsignedDivIOp, LLVM::UDivOp>;
using UnsignedRemIOpLowering =
    VectorConvertToLLVMPattern<UnsignedRemIOp, LLVM::URemOp>;
using UnsignedShiftRightOpLowering =
    OneToOneConvertToLLVMPattern<UnsignedShiftRightOp, LLVM::LShrOp>;
using XOrOpLowering = VectorConvertToLLVMPattern<XOrOp, LLVM::XOrOp>;
using ZeroExtendIOpLowering =
    VectorConvertToLLVMPattern<ZeroExtendIOp, LLVM::ZExtOp>;

/// Lower `std.assert`. The default lowering calls the `abort` function if the
/// assertion is violated and has no effect otherwise. The failure message is
/// ignored by the default lowering but should be propagated by any custom
/// lowering.
struct AssertOpLowering : public ConvertOpToLLVMPattern<AssertOp> {
  using ConvertOpToLLVMPattern<AssertOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AssertOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    AssertOp::Adaptor transformed(operands);

    // Insert the `abort` declaration if necessary.
    auto module = op->getParentOfType<ModuleOp>();
    auto abortFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("abort");
    if (!abortFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      auto abortFuncTy = LLVM::LLVMFunctionType::get(getVoidType(), {});
      abortFunc = rewriter.create<LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(),
                                                    "abort", abortFuncTy);
    }

    // Split block at `assert` operation.
    Block *opBlock = rewriter.getInsertionBlock();
    auto opPosition = rewriter.getInsertionPoint();
    Block *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

    // Generate IR to call `abort`.
    Block *failureBlock = rewriter.createBlock(opBlock->getParent());
    rewriter.create<LLVM::CallOp>(loc, abortFunc, llvm::None);
    rewriter.create<LLVM::UnreachableOp>(loc);

    // Generate assertion test.
    rewriter.setInsertionPointToEnd(opBlock);
    rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(
        op, transformed.arg(), continuationBlock, failureBlock);

    return success();
  }
};

struct ConstantOpLowering : public ConvertOpToLLVMPattern<ConstantOp> {
  using ConvertOpToLLVMPattern<ConstantOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // If constant refers to a function, convert it to "addressof".
    if (auto symbolRef = op.getValue().dyn_cast<FlatSymbolRefAttr>()) {
      auto type = typeConverter->convertType(op.getResult().getType());
      if (!type || !LLVM::isCompatibleType(type))
        return rewriter.notifyMatchFailure(op, "failed to convert result type");

      auto newOp = rewriter.create<LLVM::AddressOfOp>(op.getLoc(), type,
                                                      symbolRef.getValue());
      for (const NamedAttribute &attr : op->getAttrs()) {
        if (attr.first.strref() == "value")
          continue;
        newOp->setAttr(attr.first, attr.second);
      }
      rewriter.replaceOp(op, newOp->getResults());
      return success();
    }

    // Calling into other scopes (non-flat reference) is not supported in LLVM.
    if (op.getValue().isa<SymbolRefAttr>())
      return rewriter.notifyMatchFailure(
          op, "referring to a symbol outside of the current module");

    return LLVM::detail::oneToOneRewrite(
        op, LLVM::ConstantOp::getOperationName(), operands, *getTypeConverter(),
        rewriter);
  }
};

struct AllocOpLowering : public AllocLikeOpLLVMLowering {
  AllocOpLowering(LLVMTypeConverter &converter)
      : AllocLikeOpLLVMLowering(memref::AllocOp::getOperationName(),
                                converter) {}

  std::tuple<Value, Value> allocateBuffer(ConversionPatternRewriter &rewriter,
                                          Location loc, Value sizeBytes,
                                          Operation *op) const override {
    // Heap allocations.
    memref::AllocOp allocOp = cast<memref::AllocOp>(op);
    MemRefType memRefType = allocOp.getType();

    Value alignment;
    if (auto alignmentAttr = allocOp.alignment()) {
      alignment = createIndexConstant(rewriter, loc, *alignmentAttr);
    } else if (!memRefType.getElementType().isSignlessIntOrIndexOrFloat()) {
      // In the case where no alignment is specified, we may want to override
      // `malloc's` behavior. `malloc` typically aligns at the size of the
      // biggest scalar on a target HW. For non-scalars, use the natural
      // alignment of the LLVM type given by the LLVM DataLayout.
      alignment = getSizeInBytes(loc, memRefType.getElementType(), rewriter);
    }

    if (alignment) {
      // Adjust the allocation size to consider alignment.
      sizeBytes = rewriter.create<LLVM::AddOp>(loc, sizeBytes, alignment);
    }

    // Allocate the underlying buffer and store a pointer to it in the MemRef
    // descriptor.
    Type elementPtrType = this->getElementPtrType(memRefType);
    auto allocFuncOp = LLVM::lookupOrCreateMallocFn(
        allocOp->getParentOfType<ModuleOp>(), getIndexType());
    auto results = createLLVMCall(rewriter, loc, allocFuncOp, {sizeBytes},
                                  getVoidPtrType());
    Value allocatedPtr =
        rewriter.create<LLVM::BitcastOp>(loc, elementPtrType, results[0]);

    Value alignedPtr = allocatedPtr;
    if (alignment) {
      // Compute the aligned type pointer.
      Value allocatedInt =
          rewriter.create<LLVM::PtrToIntOp>(loc, getIndexType(), allocatedPtr);
      Value alignmentInt =
          createAligned(rewriter, loc, allocatedInt, alignment);
      alignedPtr =
          rewriter.create<LLVM::IntToPtrOp>(loc, elementPtrType, alignmentInt);
    }

    return std::make_tuple(allocatedPtr, alignedPtr);
  }
};

struct AlignedAllocOpLowering : public AllocLikeOpLLVMLowering {
  AlignedAllocOpLowering(LLVMTypeConverter &converter)
      : AllocLikeOpLLVMLowering(memref::AllocOp::getOperationName(),
                                converter) {}

  /// Returns the memref's element size in bytes using the data layout active at
  /// `op`.
  // TODO: there are other places where this is used. Expose publicly?
  unsigned getMemRefEltSizeInBytes(MemRefType memRefType, Operation *op) const {
    const DataLayout *layout = &defaultLayout;
    if (const DataLayoutAnalysis *analysis =
            getTypeConverter()->getDataLayoutAnalysis()) {
      layout = &analysis->getAbove(op);
    }
    Type elementType = memRefType.getElementType();
    if (auto memRefElementType = elementType.dyn_cast<MemRefType>())
      return getTypeConverter()->getMemRefDescriptorSize(memRefElementType,
                                                         *layout);
    if (auto memRefElementType = elementType.dyn_cast<UnrankedMemRefType>())
      return getTypeConverter()->getUnrankedMemRefDescriptorSize(
          memRefElementType, *layout);
    return layout->getTypeSize(elementType);
  }

  /// Returns true if the memref size in bytes is known to be a multiple of
  /// factor assuming the data layout active at `op`.
  bool isMemRefSizeMultipleOf(MemRefType type, uint64_t factor,
                              Operation *op) const {
    uint64_t sizeDivisor = getMemRefEltSizeInBytes(type, op);
    for (unsigned i = 0, e = type.getRank(); i < e; i++) {
      if (type.isDynamic(type.getDimSize(i)))
        continue;
      sizeDivisor = sizeDivisor * type.getDimSize(i);
    }
    return sizeDivisor % factor == 0;
  }

  /// Returns the alignment to be used for the allocation call itself.
  /// aligned_alloc requires the allocation size to be a power of two, and the
  /// allocation size to be a multiple of alignment,
  int64_t getAllocationAlignment(memref::AllocOp allocOp) const {
    if (Optional<uint64_t> alignment = allocOp.alignment())
      return *alignment;

    // Whenever we don't have alignment set, we will use an alignment
    // consistent with the element type; since the allocation size has to be a
    // power of two, we will bump to the next power of two if it already isn't.
    auto eltSizeBytes = getMemRefEltSizeInBytes(allocOp.getType(), allocOp);
    return std::max(kMinAlignedAllocAlignment,
                    llvm::PowerOf2Ceil(eltSizeBytes));
  }

  std::tuple<Value, Value> allocateBuffer(ConversionPatternRewriter &rewriter,
                                          Location loc, Value sizeBytes,
                                          Operation *op) const override {
    // Heap allocations.
    memref::AllocOp allocOp = cast<memref::AllocOp>(op);
    MemRefType memRefType = allocOp.getType();
    int64_t alignment = getAllocationAlignment(allocOp);
    Value allocAlignment = createIndexConstant(rewriter, loc, alignment);

    // aligned_alloc requires size to be a multiple of alignment; we will pad
    // the size to the next multiple if necessary.
    if (!isMemRefSizeMultipleOf(memRefType, alignment, op))
      sizeBytes = createAligned(rewriter, loc, sizeBytes, allocAlignment);

    Type elementPtrType = this->getElementPtrType(memRefType);
    auto allocFuncOp = LLVM::lookupOrCreateAlignedAllocFn(
        allocOp->getParentOfType<ModuleOp>(), getIndexType());
    auto results =
        createLLVMCall(rewriter, loc, allocFuncOp, {allocAlignment, sizeBytes},
                       getVoidPtrType());
    Value allocatedPtr =
        rewriter.create<LLVM::BitcastOp>(loc, elementPtrType, results[0]);

    return std::make_tuple(allocatedPtr, allocatedPtr);
  }

  /// The minimum alignment to use with aligned_alloc (has to be a power of 2).
  static constexpr uint64_t kMinAlignedAllocAlignment = 16UL;

  /// Default layout to use in absence of the corresponding analysis.
  DataLayout defaultLayout;
};

// Out of line definition, required till C++17.
constexpr uint64_t AlignedAllocOpLowering::kMinAlignedAllocAlignment;

struct AllocaOpLowering : public AllocLikeOpLLVMLowering {
  AllocaOpLowering(LLVMTypeConverter &converter)
      : AllocLikeOpLLVMLowering(memref::AllocaOp::getOperationName(),
                                converter) {}

  /// Allocates the underlying buffer using the right call. `allocatedBytePtr`
  /// is set to null for stack allocations. `accessAlignment` is set if
  /// alignment is needed post allocation (for eg. in conjunction with malloc).
  std::tuple<Value, Value> allocateBuffer(ConversionPatternRewriter &rewriter,
                                          Location loc, Value sizeBytes,
                                          Operation *op) const override {

    // With alloca, one gets a pointer to the element type right away.
    // For stack allocations.
    auto allocaOp = cast<memref::AllocaOp>(op);
    auto elementPtrType = this->getElementPtrType(allocaOp.getType());

    auto allocatedElementPtr = rewriter.create<LLVM::AllocaOp>(
        loc, elementPtrType, sizeBytes,
        allocaOp.alignment() ? *allocaOp.alignment() : 0);

    return std::make_tuple(allocatedElementPtr, allocatedElementPtr);
  }
};

struct AllocaScopeOpLowering
    : public ConvertOpToLLVMPattern<memref::AllocaScopeOp> {
  using ConvertOpToLLVMPattern<memref::AllocaScopeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::AllocaScopeOp allocaScopeOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);
    Location loc = allocaScopeOp.getLoc();

    // Split the current block before the AllocaScopeOp to create the inlining
    // point.
    auto *currentBlock = rewriter.getInsertionBlock();
    auto *remainingOpsBlock =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *continueBlock;
    if (allocaScopeOp.getNumResults() == 0) {
      continueBlock = remainingOpsBlock;
    } else {
      continueBlock = rewriter.createBlock(remainingOpsBlock,
                                           allocaScopeOp.getResultTypes());
      rewriter.create<BranchOp>(loc, remainingOpsBlock);
    }

    // Inline body region.
    Block *beforeBody = &allocaScopeOp.bodyRegion().front();
    Block *afterBody = &allocaScopeOp.bodyRegion().back();
    rewriter.inlineRegionBefore(allocaScopeOp.bodyRegion(), continueBlock);

    // Save stack and then branch into the body of the region.
    rewriter.setInsertionPointToEnd(currentBlock);
    auto stackSaveOp =
        rewriter.create<LLVM::StackSaveOp>(loc, getVoidPtrType());
    rewriter.create<BranchOp>(loc, beforeBody);

    // Replace the alloca_scope return with a branch that jumps out of the body.
    // Stack restore before leaving the body region.
    rewriter.setInsertionPointToEnd(afterBody);
    auto returnOp =
        cast<memref::AllocaScopeReturnOp>(afterBody->getTerminator());
    auto branchOp = rewriter.replaceOpWithNewOp<BranchOp>(
        returnOp, continueBlock, returnOp.results());

    // Insert stack restore before jumping out the body of the region.
    rewriter.setInsertionPoint(branchOp);
    rewriter.create<LLVM::StackRestoreOp>(loc, stackSaveOp);

    // Replace the op with values return from the body region.
    rewriter.replaceOp(allocaScopeOp, continueBlock->getArguments());

    return success();
  }
};

/// Copies the shaped descriptor part to (if `toDynamic` is set) or from
/// (otherwise) the dynamically allocated memory for any operands that were
/// unranked descriptors originally.
static LogicalResult copyUnrankedDescriptors(OpBuilder &builder, Location loc,
                                             LLVMTypeConverter &typeConverter,
                                             TypeRange origTypes,
                                             SmallVectorImpl<Value> &operands,
                                             bool toDynamic) {
  assert(origTypes.size() == operands.size() &&
         "expected as may original types as operands");

  // Find operands of unranked memref type and store them.
  SmallVector<UnrankedMemRefDescriptor, 4> unrankedMemrefs;
  for (unsigned i = 0, e = operands.size(); i < e; ++i)
    if (origTypes[i].isa<UnrankedMemRefType>())
      unrankedMemrefs.emplace_back(operands[i]);

  if (unrankedMemrefs.empty())
    return success();

  // Compute allocation sizes.
  SmallVector<Value, 4> sizes;
  UnrankedMemRefDescriptor::computeSizes(builder, loc, typeConverter,
                                         unrankedMemrefs, sizes);

  // Get frequently used types.
  MLIRContext *context = builder.getContext();
  Type voidPtrType = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
  auto i1Type = IntegerType::get(context, 1);
  Type indexType = typeConverter.getIndexType();

  // Find the malloc and free, or declare them if necessary.
  auto module = builder.getInsertionPoint()->getParentOfType<ModuleOp>();
  LLVM::LLVMFuncOp freeFunc, mallocFunc;
  if (toDynamic)
    mallocFunc = LLVM::lookupOrCreateMallocFn(module, indexType);
  if (!toDynamic)
    freeFunc = LLVM::lookupOrCreateFreeFn(module);

  // Initialize shared constants.
  Value zero =
      builder.create<LLVM::ConstantOp>(loc, i1Type, builder.getBoolAttr(false));

  unsigned unrankedMemrefPos = 0;
  for (unsigned i = 0, e = operands.size(); i < e; ++i) {
    Type type = origTypes[i];
    if (!type.isa<UnrankedMemRefType>())
      continue;
    Value allocationSize = sizes[unrankedMemrefPos++];
    UnrankedMemRefDescriptor desc(operands[i]);

    // Allocate memory, copy, and free the source if necessary.
    Value memory =
        toDynamic
            ? builder.create<LLVM::CallOp>(loc, mallocFunc, allocationSize)
                  .getResult(0)
            : builder.create<LLVM::AllocaOp>(loc, voidPtrType, allocationSize,
                                             /*alignment=*/0);

    Value source = desc.memRefDescPtr(builder, loc);
    builder.create<LLVM::MemcpyOp>(loc, memory, source, allocationSize, zero);
    if (!toDynamic)
      builder.create<LLVM::CallOp>(loc, freeFunc, source);

    // Create a new descriptor. The same descriptor can be returned multiple
    // times, attempting to modify its pointer can lead to memory leaks
    // (allocated twice and overwritten) or double frees (the caller does not
    // know if the descriptor points to the same memory).
    Type descriptorType = typeConverter.convertType(type);
    if (!descriptorType)
      return failure();
    auto updatedDesc =
        UnrankedMemRefDescriptor::undef(builder, loc, descriptorType);
    Value rank = desc.rank(builder, loc);
    updatedDesc.setRank(builder, loc, rank);
    updatedDesc.setMemRefDescPtr(builder, loc, memory);

    operands[i] = updatedDesc;
  }

  return success();
}

// A CallOp automatically promotes MemRefType to a sequence of alloca/store and
// passes the pointer to the MemRef across function boundaries.
template <typename CallOpType>
struct CallOpInterfaceLowering : public ConvertOpToLLVMPattern<CallOpType> {
  using ConvertOpToLLVMPattern<CallOpType>::ConvertOpToLLVMPattern;
  using Super = CallOpInterfaceLowering<CallOpType>;
  using Base = ConvertOpToLLVMPattern<CallOpType>;

  LogicalResult
  matchAndRewrite(CallOpType callOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    typename CallOpType::Adaptor transformed(operands);

    // Pack the result types into a struct.
    Type packedResult = nullptr;
    unsigned numResults = callOp.getNumResults();
    auto resultTypes = llvm::to_vector<4>(callOp.getResultTypes());

    if (numResults != 0) {
      if (!(packedResult =
                this->getTypeConverter()->packFunctionResults(resultTypes)))
        return failure();
    }

    auto promoted = this->getTypeConverter()->promoteOperands(
        callOp.getLoc(), /*opOperands=*/callOp->getOperands(), operands,
        rewriter);
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
        auto type =
            this->typeConverter->convertType(callOp.getResult(i).getType());
        results.push_back(rewriter.create<LLVM::ExtractValueOp>(
            callOp.getLoc(), type, newOp->getResult(0),
            rewriter.getI64ArrayAttr(i)));
      }
    }

    if (this->getTypeConverter()->getOptions().useBarePtrCallConv) {
      // For the bare-ptr calling convention, promote memref results to
      // descriptors.
      assert(results.size() == resultTypes.size() &&
             "The number of arguments and types doesn't match");
      this->getTypeConverter()->promoteBarePtrsToDescriptors(
          rewriter, callOp.getLoc(), resultTypes, results);
    } else if (failed(copyUnrankedDescriptors(rewriter, callOp.getLoc(),
                                              *this->getTypeConverter(),
                                              resultTypes, results,
                                              /*toDynamic=*/false))) {
      return failure();
    }

    rewriter.replaceOp(callOp, results);
    return success();
  }
};

struct CallOpLowering : public CallOpInterfaceLowering<CallOp> {
  using Super::Super;
};

struct CallIndirectOpLowering : public CallOpInterfaceLowering<CallIndirectOp> {
  using Super::Super;
};

// A `dealloc` is converted into a call to `free` on the underlying data buffer.
// The memref descriptor being an SSA value, there is no need to clean it up
// in any way.
struct DeallocOpLowering : public ConvertOpToLLVMPattern<memref::DeallocOp> {
  using ConvertOpToLLVMPattern<memref::DeallocOp>::ConvertOpToLLVMPattern;

  explicit DeallocOpLowering(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern<memref::DeallocOp>(converter) {}

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    assert(operands.size() == 1 && "dealloc takes one operand");
    memref::DeallocOp::Adaptor transformed(operands);

    // Insert the `free` declaration if it is not already present.
    auto freeFunc = LLVM::lookupOrCreateFreeFn(op->getParentOfType<ModuleOp>());
    MemRefDescriptor memref(transformed.memref());
    Value casted = rewriter.create<LLVM::BitcastOp>(
        op.getLoc(), getVoidPtrType(),
        memref.allocatedPtr(rewriter, op.getLoc()));
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange(), rewriter.getSymbolRefAttr(freeFunc), casted);
    return success();
  }
};

/// Returns the LLVM type of the global variable given the memref type `type`.
static Type convertGlobalMemrefTypeToLLVM(MemRefType type,
                                          LLVMTypeConverter &typeConverter) {
  // LLVM type for a global memref will be a multi-dimension array. For
  // declarations or uninitialized global memrefs, we can potentially flatten
  // this to a 1D array. However, for memref.global's with an initial value,
  // we do not intend to flatten the ElementsAttribute when going from std ->
  // LLVM dialect, so the LLVM type needs to me a multi-dimension array.
  Type elementType = typeConverter.convertType(type.getElementType());
  Type arrayTy = elementType;
  // Shape has the outermost dim at index 0, so need to walk it backwards
  for (int64_t dim : llvm::reverse(type.getShape()))
    arrayTy = LLVM::LLVMArrayType::get(arrayTy, dim);
  return arrayTy;
}

/// GlobalMemrefOp is lowered to a LLVM Global Variable.
struct GlobalMemrefOpLowering
    : public ConvertOpToLLVMPattern<memref::GlobalOp> {
  using ConvertOpToLLVMPattern<memref::GlobalOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::GlobalOp global, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType type = global.type().cast<MemRefType>();
    if (!isConvertibleAndHasIdentityMaps(type))
      return failure();

    Type arrayTy = convertGlobalMemrefTypeToLLVM(type, *getTypeConverter());

    LLVM::Linkage linkage =
        global.isPublic() ? LLVM::Linkage::External : LLVM::Linkage::Private;

    Attribute initialValue = nullptr;
    if (!global.isExternal() && !global.isUninitialized()) {
      auto elementsAttr = global.initial_value()->cast<ElementsAttr>();
      initialValue = elementsAttr;

      // For scalar memrefs, the global variable created is of the element type,
      // so unpack the elements attribute to extract the value.
      if (type.getRank() == 0)
        initialValue = elementsAttr.getValue({});
    }

    rewriter.replaceOpWithNewOp<LLVM::GlobalOp>(
        global, arrayTy, global.constant(), linkage, global.sym_name(),
        initialValue, /*alignment=*/0, type.getMemorySpaceAsInt());
    return success();
  }
};

/// GetGlobalMemrefOp is lowered into a Memref descriptor with the pointer to
/// the first element stashed into the descriptor. This reuses
/// `AllocLikeOpLowering` to reuse the Memref descriptor construction.
struct GetGlobalMemrefOpLowering : public AllocLikeOpLLVMLowering {
  GetGlobalMemrefOpLowering(LLVMTypeConverter &converter)
      : AllocLikeOpLLVMLowering(memref::GetGlobalOp::getOperationName(),
                                converter) {}

  /// Buffer "allocation" for memref.get_global op is getting the address of
  /// the global variable referenced.
  std::tuple<Value, Value> allocateBuffer(ConversionPatternRewriter &rewriter,
                                          Location loc, Value sizeBytes,
                                          Operation *op) const override {
    auto getGlobalOp = cast<memref::GetGlobalOp>(op);
    MemRefType type = getGlobalOp.result().getType().cast<MemRefType>();
    unsigned memSpace = type.getMemorySpaceAsInt();

    Type arrayTy = convertGlobalMemrefTypeToLLVM(type, *getTypeConverter());
    auto addressOf = rewriter.create<LLVM::AddressOfOp>(
        loc, LLVM::LLVMPointerType::get(arrayTy, memSpace), getGlobalOp.name());

    // Get the address of the first element in the array by creating a GEP with
    // the address of the GV as the base, and (rank + 1) number of 0 indices.
    Type elementType = typeConverter->convertType(type.getElementType());
    Type elementPtrType = LLVM::LLVMPointerType::get(elementType, memSpace);

    SmallVector<Value, 4> operands = {addressOf};
    operands.insert(operands.end(), type.getRank() + 1,
                    createIndexConstant(rewriter, loc, 0));
    auto gep = rewriter.create<LLVM::GEPOp>(loc, elementPtrType, operands);

    // We do not expect the memref obtained using `memref.get_global` to be
    // ever deallocated. Set the allocated pointer to be known bad value to
    // help debug if that ever happens.
    auto intPtrType = getIntPtrType(memSpace);
    Value deadBeefConst =
        createIndexAttrConstant(rewriter, op->getLoc(), intPtrType, 0xdeadbeef);
    auto deadBeefPtr =
        rewriter.create<LLVM::IntToPtrOp>(loc, elementPtrType, deadBeefConst);

    // Both allocated and aligned pointers are same. We could potentially stash
    // a nullptr for the allocated pointer since we do not expect any dealloc.
    return std::make_tuple(deadBeefPtr, gep);
  }
};

// A `expm1` is converted into `exp - 1`.
struct ExpM1OpLowering : public ConvertOpToLLVMPattern<math::ExpM1Op> {
  using ConvertOpToLLVMPattern<math::ExpM1Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(math::ExpM1Op op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    math::ExpM1Op::Adaptor transformed(operands);
    auto operandType = transformed.operand().getType();

    if (!operandType || !LLVM::isCompatibleType(operandType))
      return failure();

    auto loc = op.getLoc();
    auto resultType = op.getResult().getType();
    auto floatType = getElementTypeOrSelf(resultType).cast<FloatType>();
    auto floatOne = rewriter.getFloatAttr(floatType, 1.0);

    if (!operandType.isa<LLVM::LLVMArrayType>()) {
      LLVM::ConstantOp one;
      if (LLVM::isCompatibleVectorType(operandType)) {
        one = rewriter.create<LLVM::ConstantOp>(
            loc, operandType,
            SplatElementsAttr::get(resultType.cast<ShapedType>(), floatOne));
      } else {
        one = rewriter.create<LLVM::ConstantOp>(loc, operandType, floatOne);
      }
      auto exp = rewriter.create<LLVM::ExpOp>(loc, transformed.operand());
      rewriter.replaceOpWithNewOp<LLVM::FSubOp>(op, operandType, exp, one);
      return success();
    }

    auto vectorType = resultType.dyn_cast<VectorType>();
    if (!vectorType)
      return rewriter.notifyMatchFailure(op, "expected vector result type");

    return LLVM::detail::handleMultidimensionalVectors(
        op.getOperation(), operands, *getTypeConverter(),
        [&](Type llvm1DVectorTy, ValueRange operands) {
          auto splatAttr = SplatElementsAttr::get(
              mlir::VectorType::get(
                  {LLVM::getVectorNumElements(llvm1DVectorTy).getFixedValue()},
                  floatType),
              floatOne);
          auto one =
              rewriter.create<LLVM::ConstantOp>(loc, llvm1DVectorTy, splatAttr);
          auto exp =
              rewriter.create<LLVM::ExpOp>(loc, llvm1DVectorTy, operands[0]);
          return rewriter.create<LLVM::FSubOp>(loc, llvm1DVectorTy, exp, one);
        },
        rewriter);
  }
};

// A `log1p` is converted into `log(1 + ...)`.
struct Log1pOpLowering : public ConvertOpToLLVMPattern<math::Log1pOp> {
  using ConvertOpToLLVMPattern<math::Log1pOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(math::Log1pOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    math::Log1pOp::Adaptor transformed(operands);
    auto operandType = transformed.operand().getType();

    if (!operandType || !LLVM::isCompatibleType(operandType))
      return rewriter.notifyMatchFailure(op, "unsupported operand type");

    auto loc = op.getLoc();
    auto resultType = op.getResult().getType();
    auto floatType = getElementTypeOrSelf(resultType).cast<FloatType>();
    auto floatOne = rewriter.getFloatAttr(floatType, 1.0);

    if (!operandType.isa<LLVM::LLVMArrayType>()) {
      LLVM::ConstantOp one =
          LLVM::isCompatibleVectorType(operandType)
              ? rewriter.create<LLVM::ConstantOp>(
                    loc, operandType,
                    SplatElementsAttr::get(resultType.cast<ShapedType>(),
                                           floatOne))
              : rewriter.create<LLVM::ConstantOp>(loc, operandType, floatOne);

      auto add = rewriter.create<LLVM::FAddOp>(loc, operandType, one,
                                               transformed.operand());
      rewriter.replaceOpWithNewOp<LLVM::LogOp>(op, operandType, add);
      return success();
    }

    auto vectorType = resultType.dyn_cast<VectorType>();
    if (!vectorType)
      return rewriter.notifyMatchFailure(op, "expected vector result type");

    return LLVM::detail::handleMultidimensionalVectors(
        op.getOperation(), operands, *getTypeConverter(),
        [&](Type llvm1DVectorTy, ValueRange operands) {
          auto splatAttr = SplatElementsAttr::get(
              mlir::VectorType::get(
                  {LLVM::getVectorNumElements(llvm1DVectorTy).getFixedValue()},
                  floatType),
              floatOne);
          auto one =
              rewriter.create<LLVM::ConstantOp>(loc, llvm1DVectorTy, splatAttr);
          auto add = rewriter.create<LLVM::FAddOp>(loc, llvm1DVectorTy, one,
                                                   operands[0]);
          return rewriter.create<LLVM::LogOp>(loc, llvm1DVectorTy, add);
        },
        rewriter);
  }
};

// A `rsqrt` is converted into `1 / sqrt`.
struct RsqrtOpLowering : public ConvertOpToLLVMPattern<math::RsqrtOp> {
  using ConvertOpToLLVMPattern<math::RsqrtOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(math::RsqrtOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    math::RsqrtOp::Adaptor transformed(operands);
    auto operandType = transformed.operand().getType();

    if (!operandType || !LLVM::isCompatibleType(operandType))
      return failure();

    auto loc = op.getLoc();
    auto resultType = op.getResult().getType();
    auto floatType = getElementTypeOrSelf(resultType).cast<FloatType>();
    auto floatOne = rewriter.getFloatAttr(floatType, 1.0);

    if (!operandType.isa<LLVM::LLVMArrayType>()) {
      LLVM::ConstantOp one;
      if (LLVM::isCompatibleVectorType(operandType)) {
        one = rewriter.create<LLVM::ConstantOp>(
            loc, operandType,
            SplatElementsAttr::get(resultType.cast<ShapedType>(), floatOne));
      } else {
        one = rewriter.create<LLVM::ConstantOp>(loc, operandType, floatOne);
      }
      auto sqrt = rewriter.create<LLVM::SqrtOp>(loc, transformed.operand());
      rewriter.replaceOpWithNewOp<LLVM::FDivOp>(op, operandType, one, sqrt);
      return success();
    }

    auto vectorType = resultType.dyn_cast<VectorType>();
    if (!vectorType)
      return failure();

    return LLVM::detail::handleMultidimensionalVectors(
        op.getOperation(), operands, *getTypeConverter(),
        [&](Type llvm1DVectorTy, ValueRange operands) {
          auto splatAttr = SplatElementsAttr::get(
              mlir::VectorType::get(
                  {LLVM::getVectorNumElements(llvm1DVectorTy).getFixedValue()},
                  floatType),
              floatOne);
          auto one =
              rewriter.create<LLVM::ConstantOp>(loc, llvm1DVectorTy, splatAttr);
          auto sqrt =
              rewriter.create<LLVM::SqrtOp>(loc, llvm1DVectorTy, operands[0]);
          return rewriter.create<LLVM::FDivOp>(loc, llvm1DVectorTy, one, sqrt);
        },
        rewriter);
  }
};

struct MemRefCastOpLowering : public ConvertOpToLLVMPattern<memref::CastOp> {
  using ConvertOpToLLVMPattern<memref::CastOp>::ConvertOpToLLVMPattern;

  LogicalResult match(memref::CastOp memRefCastOp) const override {
    Type srcType = memRefCastOp.getOperand().getType();
    Type dstType = memRefCastOp.getType();

    // memref::CastOp reduce to bitcast in the ranked MemRef case and can be
    // used for type erasure. For now they must preserve underlying element type
    // and require source and result type to have the same rank. Therefore,
    // perform a sanity check that the underlying structs are the same. Once op
    // semantics are relaxed we can revisit.
    if (srcType.isa<MemRefType>() && dstType.isa<MemRefType>())
      return success(typeConverter->convertType(srcType) ==
                     typeConverter->convertType(dstType));

    // At least one of the operands is unranked type
    assert(srcType.isa<UnrankedMemRefType>() ||
           dstType.isa<UnrankedMemRefType>());

    // Unranked to unranked cast is disallowed
    return !(srcType.isa<UnrankedMemRefType>() &&
             dstType.isa<UnrankedMemRefType>())
               ? success()
               : failure();
  }

  void rewrite(memref::CastOp memRefCastOp, ArrayRef<Value> operands,
               ConversionPatternRewriter &rewriter) const override {
    memref::CastOp::Adaptor transformed(operands);

    auto srcType = memRefCastOp.getOperand().getType();
    auto dstType = memRefCastOp.getType();
    auto targetStructType = typeConverter->convertType(memRefCastOp.getType());
    auto loc = memRefCastOp.getLoc();

    // For ranked/ranked case, just keep the original descriptor.
    if (srcType.isa<MemRefType>() && dstType.isa<MemRefType>())
      return rewriter.replaceOp(memRefCastOp, {transformed.source()});

    if (srcType.isa<MemRefType>() && dstType.isa<UnrankedMemRefType>()) {
      // Casting ranked to unranked memref type
      // Set the rank in the destination from the memref type
      // Allocate space on the stack and copy the src memref descriptor
      // Set the ptr in the destination to the stack space
      auto srcMemRefType = srcType.cast<MemRefType>();
      int64_t rank = srcMemRefType.getRank();
      // ptr = AllocaOp sizeof(MemRefDescriptor)
      auto ptr = getTypeConverter()->promoteOneMemRefDescriptor(
          loc, transformed.source(), rewriter);
      // voidptr = BitCastOp srcType* to void*
      auto voidPtr =
          rewriter.create<LLVM::BitcastOp>(loc, getVoidPtrType(), ptr)
              .getResult();
      // rank = ConstantOp srcRank
      auto rankVal = rewriter.create<LLVM::ConstantOp>(
          loc, typeConverter->convertType(rewriter.getIntegerType(64)),
          rewriter.getI64IntegerAttr(rank));
      // undef = UndefOp
      UnrankedMemRefDescriptor memRefDesc =
          UnrankedMemRefDescriptor::undef(rewriter, loc, targetStructType);
      // d1 = InsertValueOp undef, rank, 0
      memRefDesc.setRank(rewriter, loc, rankVal);
      // d2 = InsertValueOp d1, voidptr, 1
      memRefDesc.setMemRefDescPtr(rewriter, loc, voidPtr);
      rewriter.replaceOp(memRefCastOp, (Value)memRefDesc);

    } else if (srcType.isa<UnrankedMemRefType>() && dstType.isa<MemRefType>()) {
      // Casting from unranked type to ranked.
      // The operation is assumed to be doing a correct cast. If the destination
      // type mismatches the unranked the type, it is undefined behavior.
      UnrankedMemRefDescriptor memRefDesc(transformed.source());
      // ptr = ExtractValueOp src, 1
      auto ptr = memRefDesc.memRefDescPtr(rewriter, loc);
      // castPtr = BitCastOp i8* to structTy*
      auto castPtr =
          rewriter
              .create<LLVM::BitcastOp>(
                  loc, LLVM::LLVMPointerType::get(targetStructType), ptr)
              .getResult();
      // struct = LoadOp castPtr
      auto loadOp = rewriter.create<LLVM::LoadOp>(loc, castPtr);
      rewriter.replaceOp(memRefCastOp, loadOp.getResult());
    } else {
      llvm_unreachable("Unsupported unranked memref to unranked memref cast");
    }
  }
};

struct MemRefCopyOpLowering : public ConvertOpToLLVMPattern<memref::CopyOp> {
  using ConvertOpToLLVMPattern<memref::CopyOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::CopyOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    memref::CopyOp::Adaptor adaptor(operands);
    auto srcType = op.source().getType().cast<BaseMemRefType>();
    auto targetType = op.target().getType().cast<BaseMemRefType>();

    // First make sure we have an unranked memref descriptor representation.
    auto makeUnranked = [&, this](Value ranked, BaseMemRefType type) {
      auto rank = rewriter.create<LLVM::ConstantOp>(
          loc, getIndexType(), rewriter.getIndexAttr(type.getRank()));
      auto *typeConverter = getTypeConverter();
      auto ptr =
          typeConverter->promoteOneMemRefDescriptor(loc, ranked, rewriter);
      auto voidPtr =
          rewriter.create<LLVM::BitcastOp>(loc, getVoidPtrType(), ptr)
              .getResult();
      auto unrankedType =
          UnrankedMemRefType::get(type.getElementType(), type.getMemorySpace());
      return UnrankedMemRefDescriptor::pack(rewriter, loc, *typeConverter,
                                            unrankedType,
                                            ValueRange{rank, voidPtr});
    };

    Value unrankedSource = srcType.hasRank()
                               ? makeUnranked(adaptor.source(), srcType)
                               : adaptor.source();
    Value unrankedTarget = targetType.hasRank()
                               ? makeUnranked(adaptor.target(), targetType)
                               : adaptor.target();

    // Now promote the unranked descriptors to the stack.
    auto one = rewriter.create<LLVM::ConstantOp>(loc, getIndexType(),
                                                 rewriter.getIndexAttr(1));
    auto promote = [&](Value desc) {
      auto ptrType = LLVM::LLVMPointerType::get(desc.getType());
      auto allocated =
          rewriter.create<LLVM::AllocaOp>(loc, ptrType, ValueRange{one});
      rewriter.create<LLVM::StoreOp>(loc, desc, allocated);
      return allocated;
    };

    auto sourcePtr = promote(unrankedSource);
    auto targetPtr = promote(unrankedTarget);

    auto elemSize = rewriter.create<LLVM::ConstantOp>(
        loc, getIndexType(),
        rewriter.getIndexAttr(srcType.getElementTypeBitWidth() / 8));
    auto copyFn = LLVM::lookupOrCreateMemRefCopyFn(
        op->getParentOfType<ModuleOp>(), getIndexType(), sourcePtr.getType());
    rewriter.create<LLVM::CallOp>(loc, copyFn,
                                  ValueRange{elemSize, sourcePtr, targetPtr});
    rewriter.eraseOp(op);

    return success();
  }
};

/// Extracts allocated, aligned pointers and offset from a ranked or unranked
/// memref type. In unranked case, the fields are extracted from the underlying
/// ranked descriptor.
static void extractPointersAndOffset(Location loc,
                                     ConversionPatternRewriter &rewriter,
                                     LLVMTypeConverter &typeConverter,
                                     Value originalOperand,
                                     Value convertedOperand,
                                     Value *allocatedPtr, Value *alignedPtr,
                                     Value *offset = nullptr) {
  Type operandType = originalOperand.getType();
  if (operandType.isa<MemRefType>()) {
    MemRefDescriptor desc(convertedOperand);
    *allocatedPtr = desc.allocatedPtr(rewriter, loc);
    *alignedPtr = desc.alignedPtr(rewriter, loc);
    if (offset != nullptr)
      *offset = desc.offset(rewriter, loc);
    return;
  }

  unsigned memorySpace =
      operandType.cast<UnrankedMemRefType>().getMemorySpaceAsInt();
  Type elementType = operandType.cast<UnrankedMemRefType>().getElementType();
  Type llvmElementType = typeConverter.convertType(elementType);
  Type elementPtrPtrType = LLVM::LLVMPointerType::get(
      LLVM::LLVMPointerType::get(llvmElementType, memorySpace));

  // Extract pointer to the underlying ranked memref descriptor and cast it to
  // ElemType**.
  UnrankedMemRefDescriptor unrankedDesc(convertedOperand);
  Value underlyingDescPtr = unrankedDesc.memRefDescPtr(rewriter, loc);

  *allocatedPtr = UnrankedMemRefDescriptor::allocatedPtr(
      rewriter, loc, underlyingDescPtr, elementPtrPtrType);
  *alignedPtr = UnrankedMemRefDescriptor::alignedPtr(
      rewriter, loc, typeConverter, underlyingDescPtr, elementPtrPtrType);
  if (offset != nullptr) {
    *offset = UnrankedMemRefDescriptor::offset(
        rewriter, loc, typeConverter, underlyingDescPtr, elementPtrPtrType);
  }
}

struct MemRefReinterpretCastOpLowering
    : public ConvertOpToLLVMPattern<memref::ReinterpretCastOp> {
  using ConvertOpToLLVMPattern<
      memref::ReinterpretCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp castOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    memref::ReinterpretCastOp::Adaptor adaptor(operands,
                                               castOp->getAttrDictionary());
    Type srcType = castOp.source().getType();

    Value descriptor;
    if (failed(convertSourceMemRefToDescriptor(rewriter, srcType, castOp,
                                               adaptor, &descriptor)))
      return failure();
    rewriter.replaceOp(castOp, {descriptor});
    return success();
  }

private:
  LogicalResult convertSourceMemRefToDescriptor(
      ConversionPatternRewriter &rewriter, Type srcType,
      memref::ReinterpretCastOp castOp,
      memref::ReinterpretCastOp::Adaptor adaptor, Value *descriptor) const {
    MemRefType targetMemRefType =
        castOp.getResult().getType().cast<MemRefType>();
    auto llvmTargetDescriptorTy = typeConverter->convertType(targetMemRefType)
                                      .dyn_cast_or_null<LLVM::LLVMStructType>();
    if (!llvmTargetDescriptorTy)
      return failure();

    // Create descriptor.
    Location loc = castOp.getLoc();
    auto desc = MemRefDescriptor::undef(rewriter, loc, llvmTargetDescriptorTy);

    // Set allocated and aligned pointers.
    Value allocatedPtr, alignedPtr;
    extractPointersAndOffset(loc, rewriter, *getTypeConverter(),
                             castOp.source(), adaptor.source(), &allocatedPtr,
                             &alignedPtr);
    desc.setAllocatedPtr(rewriter, loc, allocatedPtr);
    desc.setAlignedPtr(rewriter, loc, alignedPtr);

    // Set offset.
    if (castOp.isDynamicOffset(0))
      desc.setOffset(rewriter, loc, adaptor.offsets()[0]);
    else
      desc.setConstantOffset(rewriter, loc, castOp.getStaticOffset(0));

    // Set sizes and strides.
    unsigned dynSizeId = 0;
    unsigned dynStrideId = 0;
    for (unsigned i = 0, e = targetMemRefType.getRank(); i < e; ++i) {
      if (castOp.isDynamicSize(i))
        desc.setSize(rewriter, loc, i, adaptor.sizes()[dynSizeId++]);
      else
        desc.setConstantSize(rewriter, loc, i, castOp.getStaticSize(i));

      if (castOp.isDynamicStride(i))
        desc.setStride(rewriter, loc, i, adaptor.strides()[dynStrideId++]);
      else
        desc.setConstantStride(rewriter, loc, i, castOp.getStaticStride(i));
    }
    *descriptor = desc;
    return success();
  }
};

struct MemRefReshapeOpLowering
    : public ConvertOpToLLVMPattern<memref::ReshapeOp> {
  using ConvertOpToLLVMPattern<memref::ReshapeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::ReshapeOp reshapeOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto *op = reshapeOp.getOperation();
    memref::ReshapeOp::Adaptor adaptor(operands, op->getAttrDictionary());
    Type srcType = reshapeOp.source().getType();

    Value descriptor;
    if (failed(convertSourceMemRefToDescriptor(rewriter, srcType, reshapeOp,
                                               adaptor, &descriptor)))
      return failure();
    rewriter.replaceOp(op, {descriptor});
    return success();
  }

private:
  LogicalResult
  convertSourceMemRefToDescriptor(ConversionPatternRewriter &rewriter,
                                  Type srcType, memref::ReshapeOp reshapeOp,
                                  memref::ReshapeOp::Adaptor adaptor,
                                  Value *descriptor) const {
    // Conversion for statically-known shape args is performed via
    // `memref_reinterpret_cast`.
    auto shapeMemRefType = reshapeOp.shape().getType().cast<MemRefType>();
    if (shapeMemRefType.hasStaticShape())
      return failure();

    // The shape is a rank-1 tensor with unknown length.
    Location loc = reshapeOp.getLoc();
    MemRefDescriptor shapeDesc(adaptor.shape());
    Value resultRank = shapeDesc.size(rewriter, loc, 0);

    // Extract address space and element type.
    auto targetType =
        reshapeOp.getResult().getType().cast<UnrankedMemRefType>();
    unsigned addressSpace = targetType.getMemorySpaceAsInt();
    Type elementType = targetType.getElementType();

    // Create the unranked memref descriptor that holds the ranked one. The
    // inner descriptor is allocated on stack.
    auto targetDesc = UnrankedMemRefDescriptor::undef(
        rewriter, loc, typeConverter->convertType(targetType));
    targetDesc.setRank(rewriter, loc, resultRank);
    SmallVector<Value, 4> sizes;
    UnrankedMemRefDescriptor::computeSizes(rewriter, loc, *getTypeConverter(),
                                           targetDesc, sizes);
    Value underlyingDescPtr = rewriter.create<LLVM::AllocaOp>(
        loc, getVoidPtrType(), sizes.front(), llvm::None);
    targetDesc.setMemRefDescPtr(rewriter, loc, underlyingDescPtr);

    // Extract pointers and offset from the source memref.
    Value allocatedPtr, alignedPtr, offset;
    extractPointersAndOffset(loc, rewriter, *getTypeConverter(),
                             reshapeOp.source(), adaptor.source(),
                             &allocatedPtr, &alignedPtr, &offset);

    // Set pointers and offset.
    Type llvmElementType = typeConverter->convertType(elementType);
    auto elementPtrPtrType = LLVM::LLVMPointerType::get(
        LLVM::LLVMPointerType::get(llvmElementType, addressSpace));
    UnrankedMemRefDescriptor::setAllocatedPtr(rewriter, loc, underlyingDescPtr,
                                              elementPtrPtrType, allocatedPtr);
    UnrankedMemRefDescriptor::setAlignedPtr(rewriter, loc, *getTypeConverter(),
                                            underlyingDescPtr,
                                            elementPtrPtrType, alignedPtr);
    UnrankedMemRefDescriptor::setOffset(rewriter, loc, *getTypeConverter(),
                                        underlyingDescPtr, elementPtrPtrType,
                                        offset);

    // Use the offset pointer as base for further addressing. Copy over the new
    // shape and compute strides. For this, we create a loop from rank-1 to 0.
    Value targetSizesBase = UnrankedMemRefDescriptor::sizeBasePtr(
        rewriter, loc, *getTypeConverter(), underlyingDescPtr,
        elementPtrPtrType);
    Value targetStridesBase = UnrankedMemRefDescriptor::strideBasePtr(
        rewriter, loc, *getTypeConverter(), targetSizesBase, resultRank);
    Value shapeOperandPtr = shapeDesc.alignedPtr(rewriter, loc);
    Value oneIndex = createIndexConstant(rewriter, loc, 1);
    Value resultRankMinusOne =
        rewriter.create<LLVM::SubOp>(loc, resultRank, oneIndex);

    Block *initBlock = rewriter.getInsertionBlock();
    Type indexType = getTypeConverter()->getIndexType();
    Block::iterator remainingOpsIt = std::next(rewriter.getInsertionPoint());

    Block *condBlock = rewriter.createBlock(initBlock->getParent(), {},
                                            {indexType, indexType});

    // Iterate over the remaining ops in initBlock and move them to condBlock.
    BlockAndValueMapping map;
    for (auto it = remainingOpsIt, e = initBlock->end(); it != e; ++it) {
      rewriter.clone(*it, map);
      rewriter.eraseOp(&*it);
    }

    rewriter.setInsertionPointToEnd(initBlock);
    rewriter.create<LLVM::BrOp>(loc, ValueRange({resultRankMinusOne, oneIndex}),
                                condBlock);
    rewriter.setInsertionPointToStart(condBlock);
    Value indexArg = condBlock->getArgument(0);
    Value strideArg = condBlock->getArgument(1);

    Value zeroIndex = createIndexConstant(rewriter, loc, 0);
    Value pred = rewriter.create<LLVM::ICmpOp>(
        loc, IntegerType::get(rewriter.getContext(), 1),
        LLVM::ICmpPredicate::sge, indexArg, zeroIndex);

    Block *bodyBlock =
        rewriter.splitBlock(condBlock, rewriter.getInsertionPoint());
    rewriter.setInsertionPointToStart(bodyBlock);

    // Copy size from shape to descriptor.
    Type llvmIndexPtrType = LLVM::LLVMPointerType::get(indexType);
    Value sizeLoadGep = rewriter.create<LLVM::GEPOp>(
        loc, llvmIndexPtrType, shapeOperandPtr, ValueRange{indexArg});
    Value size = rewriter.create<LLVM::LoadOp>(loc, sizeLoadGep);
    UnrankedMemRefDescriptor::setSize(rewriter, loc, *getTypeConverter(),
                                      targetSizesBase, indexArg, size);

    // Write stride value and compute next one.
    UnrankedMemRefDescriptor::setStride(rewriter, loc, *getTypeConverter(),
                                        targetStridesBase, indexArg, strideArg);
    Value nextStride = rewriter.create<LLVM::MulOp>(loc, strideArg, size);

    // Decrement loop counter and branch back.
    Value decrement = rewriter.create<LLVM::SubOp>(loc, indexArg, oneIndex);
    rewriter.create<LLVM::BrOp>(loc, ValueRange({decrement, nextStride}),
                                condBlock);

    Block *remainder =
        rewriter.splitBlock(bodyBlock, rewriter.getInsertionPoint());

    // Hook up the cond exit to the remainder.
    rewriter.setInsertionPointToEnd(condBlock);
    rewriter.create<LLVM::CondBrOp>(loc, pred, bodyBlock, llvm::None, remainder,
                                    llvm::None);

    // Reset position to beginning of new remainder block.
    rewriter.setInsertionPointToStart(remainder);

    *descriptor = targetDesc;
    return success();
  }
};

struct DialectCastOpLowering
    : public ConvertOpToLLVMPattern<LLVM::DialectCastOp> {
  using ConvertOpToLLVMPattern<LLVM::DialectCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(LLVM::DialectCastOp castOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    LLVM::DialectCastOp::Adaptor transformed(operands);
    if (transformed.in().getType() !=
        typeConverter->convertType(castOp.getType())) {
      return failure();
    }
    rewriter.replaceOp(castOp, transformed.in());
    return success();
  }
};

// A `dim` is converted to a constant for static sizes and to an access to the
// size stored in the memref descriptor for dynamic sizes.
struct DimOpLowering : public ConvertOpToLLVMPattern<memref::DimOp> {
  using ConvertOpToLLVMPattern<memref::DimOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::DimOp dimOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Type operandType = dimOp.source().getType();
    if (operandType.isa<UnrankedMemRefType>()) {
      rewriter.replaceOp(dimOp, {extractSizeOfUnrankedMemRef(
                                    operandType, dimOp, operands, rewriter)});

      return success();
    }
    if (operandType.isa<MemRefType>()) {
      rewriter.replaceOp(dimOp, {extractSizeOfRankedMemRef(
                                    operandType, dimOp, operands, rewriter)});
      return success();
    }
    llvm_unreachable("expected MemRefType or UnrankedMemRefType");
  }

private:
  Value extractSizeOfUnrankedMemRef(Type operandType, memref::DimOp dimOp,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter &rewriter) const {
    Location loc = dimOp.getLoc();
    memref::DimOp::Adaptor transformed(operands);

    auto unrankedMemRefType = operandType.cast<UnrankedMemRefType>();
    auto scalarMemRefType =
        MemRefType::get({}, unrankedMemRefType.getElementType());
    unsigned addressSpace = unrankedMemRefType.getMemorySpaceAsInt();

    // Extract pointer to the underlying ranked descriptor and bitcast it to a
    // memref<element_type> descriptor pointer to minimize the number of GEP
    // operations.
    UnrankedMemRefDescriptor unrankedDesc(transformed.source());
    Value underlyingRankedDesc = unrankedDesc.memRefDescPtr(rewriter, loc);
    Value scalarMemRefDescPtr = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(typeConverter->convertType(scalarMemRefType),
                                   addressSpace),
        underlyingRankedDesc);

    // Get pointer to offset field of memref<element_type> descriptor.
    Type indexPtrTy = LLVM::LLVMPointerType::get(
        getTypeConverter()->getIndexType(), addressSpace);
    Value two = rewriter.create<LLVM::ConstantOp>(
        loc, typeConverter->convertType(rewriter.getI32Type()),
        rewriter.getI32IntegerAttr(2));
    Value offsetPtr = rewriter.create<LLVM::GEPOp>(
        loc, indexPtrTy, scalarMemRefDescPtr,
        ValueRange({createIndexConstant(rewriter, loc, 0), two}));

    // The size value that we have to extract can be obtained using GEPop with
    // `dimOp.index() + 1` index argument.
    Value idxPlusOne = rewriter.create<LLVM::AddOp>(
        loc, createIndexConstant(rewriter, loc, 1), transformed.index());
    Value sizePtr = rewriter.create<LLVM::GEPOp>(loc, indexPtrTy, offsetPtr,
                                                 ValueRange({idxPlusOne}));
    return rewriter.create<LLVM::LoadOp>(loc, sizePtr);
  }

  Value extractSizeOfRankedMemRef(Type operandType, memref::DimOp dimOp,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const {
    Location loc = dimOp.getLoc();
    memref::DimOp::Adaptor transformed(operands);
    // Take advantage if index is constant.
    MemRefType memRefType = operandType.cast<MemRefType>();
    if (Optional<int64_t> index = dimOp.getConstantIndex()) {
      int64_t i = index.getValue();
      if (memRefType.isDynamicDim(i)) {
        // extract dynamic size from the memref descriptor.
        MemRefDescriptor descriptor(transformed.source());
        return descriptor.size(rewriter, loc, i);
      }
      // Use constant for static size.
      int64_t dimSize = memRefType.getDimSize(i);
      return createIndexConstant(rewriter, loc, dimSize);
    }
    Value index = dimOp.index();
    int64_t rank = memRefType.getRank();
    MemRefDescriptor memrefDescriptor(transformed.source());
    return memrefDescriptor.size(rewriter, loc, index, rank);
  }
};

struct RankOpLowering : public ConvertOpToLLVMPattern<RankOp> {
  using ConvertOpToLLVMPattern<RankOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RankOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type operandType = op.memrefOrTensor().getType();
    if (auto unrankedMemRefType = operandType.dyn_cast<UnrankedMemRefType>()) {
      UnrankedMemRefDescriptor desc(RankOp::Adaptor(operands).memrefOrTensor());
      rewriter.replaceOp(op, {desc.rank(rewriter, loc)});
      return success();
    }
    if (auto rankedMemRefType = operandType.dyn_cast<MemRefType>()) {
      rewriter.replaceOp(
          op, {createIndexConstant(rewriter, loc, rankedMemRefType.getRank())});
      return success();
    }
    return failure();
  }
};

// Common base for load and store operations on MemRefs.  Restricts the match
// to supported MemRef types. Provides functionality to emit code accessing a
// specific element of the underlying data buffer.
template <typename Derived>
struct LoadStoreOpLowering : public ConvertOpToLLVMPattern<Derived> {
  using ConvertOpToLLVMPattern<Derived>::ConvertOpToLLVMPattern;
  using ConvertOpToLLVMPattern<Derived>::isConvertibleAndHasIdentityMaps;
  using Base = LoadStoreOpLowering<Derived>;

  LogicalResult match(Derived op) const override {
    MemRefType type = op.getMemRefType();
    return isConvertibleAndHasIdentityMaps(type) ? success() : failure();
  }
};

// Load operation is lowered to obtaining a pointer to the indexed element
// and loading it.
struct LoadOpLowering : public LoadStoreOpLowering<memref::LoadOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(memref::LoadOp loadOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    memref::LoadOp::Adaptor transformed(operands);
    auto type = loadOp.getMemRefType();

    Value dataPtr =
        getStridedElementPtr(loadOp.getLoc(), type, transformed.memref(),
                             transformed.indices(), rewriter);
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(loadOp, dataPtr);
    return success();
  }
};

// Store operation is lowered to obtaining a pointer to the indexed element,
// and storing the given value to it.
struct StoreOpLowering : public LoadStoreOpLowering<memref::StoreOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = op.getMemRefType();
    memref::StoreOp::Adaptor transformed(operands);

    Value dataPtr =
        getStridedElementPtr(op.getLoc(), type, transformed.memref(),
                             transformed.indices(), rewriter);
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, transformed.value(),
                                               dataPtr);
    return success();
  }
};

// The prefetch operation is lowered in a way similar to the load operation
// except that the llvm.prefetch operation is used for replacement.
struct PrefetchOpLowering : public LoadStoreOpLowering<memref::PrefetchOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(memref::PrefetchOp prefetchOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    memref::PrefetchOp::Adaptor transformed(operands);
    auto type = prefetchOp.getMemRefType();
    auto loc = prefetchOp.getLoc();

    Value dataPtr = getStridedElementPtr(loc, type, transformed.memref(),
                                         transformed.indices(), rewriter);

    // Replace with llvm.prefetch.
    auto llvmI32Type = typeConverter->convertType(rewriter.getIntegerType(32));
    auto isWrite = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI32Type, rewriter.getI32IntegerAttr(prefetchOp.isWrite()));
    auto localityHint = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI32Type,
        rewriter.getI32IntegerAttr(prefetchOp.localityHint()));
    auto isData = rewriter.create<LLVM::ConstantOp>(
        loc, llvmI32Type, rewriter.getI32IntegerAttr(prefetchOp.isDataCache()));

    rewriter.replaceOpWithNewOp<LLVM::Prefetch>(prefetchOp, dataPtr, isWrite,
                                                localityHint, isData);
    return success();
  }
};

// The lowering of index_cast becomes an integer conversion since index becomes
// an integer.  If the bit width of the source and target integer types is the
// same, just erase the cast.  If the target type is wider, sign-extend the
// value, otherwise truncate it.
struct IndexCastOpLowering : public ConvertOpToLLVMPattern<IndexCastOp> {
  using ConvertOpToLLVMPattern<IndexCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(IndexCastOp indexCastOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    IndexCastOpAdaptor transformed(operands);

    auto targetType =
        typeConverter->convertType(indexCastOp.getResult().getType());
    auto targetElementType =
        typeConverter
            ->convertType(getElementTypeOrSelf(indexCastOp.getResult()))
            .cast<IntegerType>();
    auto sourceElementType =
        getElementTypeOrSelf(transformed.in()).cast<IntegerType>();
    unsigned targetBits = targetElementType.getWidth();
    unsigned sourceBits = sourceElementType.getWidth();

    if (targetBits == sourceBits)
      rewriter.replaceOp(indexCastOp, transformed.in());
    else if (targetBits < sourceBits)
      rewriter.replaceOpWithNewOp<LLVM::TruncOp>(indexCastOp, targetType,
                                                 transformed.in());
    else
      rewriter.replaceOpWithNewOp<LLVM::SExtOp>(indexCastOp, targetType,
                                                transformed.in());
    return success();
  }
};

// Convert std.cmp predicate into the LLVM dialect CmpPredicate.  The two
// enums share the numerical values so just cast.
template <typename LLVMPredType, typename StdPredType>
static LLVMPredType convertCmpPredicate(StdPredType pred) {
  return static_cast<LLVMPredType>(pred);
}

struct CmpIOpLowering : public ConvertOpToLLVMPattern<CmpIOp> {
  using ConvertOpToLLVMPattern<CmpIOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CmpIOp cmpiOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    CmpIOpAdaptor transformed(operands);
    auto operandType = transformed.lhs().getType();
    auto resultType = cmpiOp.getResult().getType();

    // Handle the scalar and 1D vector cases.
    if (!operandType.isa<LLVM::LLVMArrayType>()) {
      rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
          cmpiOp, typeConverter->convertType(resultType),
          convertCmpPredicate<LLVM::ICmpPredicate>(cmpiOp.getPredicate()),
          transformed.lhs(), transformed.rhs());
      return success();
    }

    auto vectorType = resultType.dyn_cast<VectorType>();
    if (!vectorType)
      return rewriter.notifyMatchFailure(cmpiOp, "expected vector result type");

    return LLVM::detail::handleMultidimensionalVectors(
        cmpiOp.getOperation(), operands, *getTypeConverter(),
        [&](Type llvm1DVectorTy, ValueRange operands) {
          CmpIOpAdaptor transformed(operands);
          return rewriter.create<LLVM::ICmpOp>(
              cmpiOp.getLoc(), llvm1DVectorTy,
              convertCmpPredicate<LLVM::ICmpPredicate>(cmpiOp.getPredicate()),
              transformed.lhs(), transformed.rhs());
        },
        rewriter);

    return success();
  }
};

struct CmpFOpLowering : public ConvertOpToLLVMPattern<CmpFOp> {
  using ConvertOpToLLVMPattern<CmpFOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CmpFOp cmpfOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    CmpFOpAdaptor transformed(operands);
    auto operandType = transformed.lhs().getType();
    auto resultType = cmpfOp.getResult().getType();

    // Handle the scalar and 1D vector cases.
    if (!operandType.isa<LLVM::LLVMArrayType>()) {
      rewriter.replaceOpWithNewOp<LLVM::FCmpOp>(
          cmpfOp, typeConverter->convertType(resultType),
          convertCmpPredicate<LLVM::FCmpPredicate>(cmpfOp.getPredicate()),
          transformed.lhs(), transformed.rhs());
      return success();
    }

    auto vectorType = resultType.dyn_cast<VectorType>();
    if (!vectorType)
      return rewriter.notifyMatchFailure(cmpfOp, "expected vector result type");

    return LLVM::detail::handleMultidimensionalVectors(
        cmpfOp.getOperation(), operands, *getTypeConverter(),
        [&](Type llvm1DVectorTy, ValueRange operands) {
          CmpFOpAdaptor transformed(operands);
          return rewriter.create<LLVM::FCmpOp>(
              cmpfOp.getLoc(), llvm1DVectorTy,
              convertCmpPredicate<LLVM::FCmpPredicate>(cmpfOp.getPredicate()),
              transformed.lhs(), transformed.rhs());
        },
        rewriter);
  }
};

// Base class for LLVM IR lowering terminator operations with successors.
template <typename SourceOp, typename TargetOp>
struct OneToOneLLVMTerminatorLowering
    : public ConvertOpToLLVMPattern<SourceOp> {
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;
  using Super = OneToOneLLVMTerminatorLowering<SourceOp, TargetOp>;

  LogicalResult
  matchAndRewrite(SourceOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TargetOp>(op, operands, op->getSuccessors(),
                                          op->getAttrs());
    return success();
  }
};

// Special lowering pattern for `ReturnOps`.  Unlike all other operations,
// `ReturnOp` interacts with the function signature and must have as many
// operands as the function has return values.  Because in LLVM IR, functions
// can only return 0 or 1 value, we pack multiple values into a structure type.
// Emit `UndefOp` followed by `InsertValueOp`s to create such structure if
// necessary before returning it
struct ReturnOpLowering : public ConvertOpToLLVMPattern<ReturnOp> {
  using ConvertOpToLLVMPattern<ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    unsigned numArguments = op.getNumOperands();
    SmallVector<Value, 4> updatedOperands;

    if (getTypeConverter()->getOptions().useBarePtrCallConv) {
      // For the bare-ptr calling convention, extract the aligned pointer to
      // be returned from the memref descriptor.
      for (auto it : llvm::zip(op->getOperands(), operands)) {
        Type oldTy = std::get<0>(it).getType();
        Value newOperand = std::get<1>(it);
        if (oldTy.isa<MemRefType>()) {
          MemRefDescriptor memrefDesc(newOperand);
          newOperand = memrefDesc.alignedPtr(rewriter, loc);
        } else if (oldTy.isa<UnrankedMemRefType>()) {
          // Unranked memref is not supported in the bare pointer calling
          // convention.
          return failure();
        }
        updatedOperands.push_back(newOperand);
      }
    } else {
      updatedOperands = llvm::to_vector<4>(operands);
      (void)copyUnrankedDescriptors(rewriter, loc, *getTypeConverter(),
                                    op.getOperands().getTypes(),
                                    updatedOperands,
                                    /*toDynamic=*/true);
    }

    // If ReturnOp has 0 or 1 operand, create it and return immediately.
    if (numArguments == 0) {
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(), ValueRange(),
                                                  op->getAttrs());
      return success();
    }
    if (numArguments == 1) {
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(
          op, TypeRange(), updatedOperands, op->getAttrs());
      return success();
    }

    // Otherwise, we need to pack the arguments into an LLVM struct type before
    // returning.
    auto packedType = getTypeConverter()->packFunctionResults(
        llvm::to_vector<4>(op.getOperandTypes()));

    Value packed = rewriter.create<LLVM::UndefOp>(loc, packedType);
    for (unsigned i = 0; i < numArguments; ++i) {
      packed = rewriter.create<LLVM::InsertValueOp>(
          loc, packedType, packed, updatedOperands[i],
          rewriter.getI64ArrayAttr(i));
    }
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(), packed,
                                                op->getAttrs());
    return success();
  }
};

// FIXME: this should be tablegen'ed as well.
struct BranchOpLowering
    : public OneToOneLLVMTerminatorLowering<BranchOp, LLVM::BrOp> {
  using Super::Super;
};
struct CondBranchOpLowering
    : public OneToOneLLVMTerminatorLowering<CondBranchOp, LLVM::CondBrOp> {
  using Super::Super;
};
struct SwitchOpLowering
    : public OneToOneLLVMTerminatorLowering<SwitchOp, LLVM::SwitchOp> {
  using Super::Super;
};

// The Splat operation is lowered to an insertelement + a shufflevector
// operation. Splat to only 1-d vector result types are lowered.
struct SplatOpLowering : public ConvertOpToLLVMPattern<SplatOp> {
  using ConvertOpToLLVMPattern<SplatOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SplatOp splatOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    VectorType resultType = splatOp.getType().dyn_cast<VectorType>();
    if (!resultType || resultType.getRank() != 1)
      return failure();

    // First insert it into an undef vector so we can shuffle it.
    auto vectorType = typeConverter->convertType(splatOp.getType());
    Value undef = rewriter.create<LLVM::UndefOp>(splatOp.getLoc(), vectorType);
    auto zero = rewriter.create<LLVM::ConstantOp>(
        splatOp.getLoc(),
        typeConverter->convertType(rewriter.getIntegerType(32)),
        rewriter.getZeroAttr(rewriter.getIntegerType(32)));

    auto v = rewriter.create<LLVM::InsertElementOp>(
        splatOp.getLoc(), vectorType, undef, splatOp.getOperand(), zero);

    int64_t width = splatOp.getType().cast<VectorType>().getDimSize(0);
    SmallVector<int32_t, 4> zeroValues(width, 0);

    // Shuffle the value across the desired number of elements.
    ArrayAttr zeroAttrs = rewriter.getI32ArrayAttr(zeroValues);
    rewriter.replaceOpWithNewOp<LLVM::ShuffleVectorOp>(splatOp, v, undef,
                                                       zeroAttrs);
    return success();
  }
};

// The Splat operation is lowered to an insertelement + a shufflevector
// operation. Splat to only 2+-d vector result types are lowered by the
// SplatNdOpLowering, the 1-d case is handled by SplatOpLowering.
struct SplatNdOpLowering : public ConvertOpToLLVMPattern<SplatOp> {
  using ConvertOpToLLVMPattern<SplatOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SplatOp splatOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    SplatOp::Adaptor adaptor(operands);
    VectorType resultType = splatOp.getType().dyn_cast<VectorType>();
    if (!resultType || resultType.getRank() == 1)
      return failure();

    // First insert it into an undef vector so we can shuffle it.
    auto loc = splatOp.getLoc();
    auto vectorTypeInfo =
        LLVM::detail::extractNDVectorTypeInfo(resultType, *getTypeConverter());
    auto llvmNDVectorTy = vectorTypeInfo.llvmNDVectorTy;
    auto llvm1DVectorTy = vectorTypeInfo.llvm1DVectorTy;
    if (!llvmNDVectorTy || !llvm1DVectorTy)
      return failure();

    // Construct returned value.
    Value desc = rewriter.create<LLVM::UndefOp>(loc, llvmNDVectorTy);

    // Construct a 1-D vector with the splatted value that we insert in all the
    // places within the returned descriptor.
    Value vdesc = rewriter.create<LLVM::UndefOp>(loc, llvm1DVectorTy);
    auto zero = rewriter.create<LLVM::ConstantOp>(
        loc, typeConverter->convertType(rewriter.getIntegerType(32)),
        rewriter.getZeroAttr(rewriter.getIntegerType(32)));
    Value v = rewriter.create<LLVM::InsertElementOp>(loc, llvm1DVectorTy, vdesc,
                                                     adaptor.input(), zero);

    // Shuffle the value across the desired number of elements.
    int64_t width = resultType.getDimSize(resultType.getRank() - 1);
    SmallVector<int32_t, 4> zeroValues(width, 0);
    ArrayAttr zeroAttrs = rewriter.getI32ArrayAttr(zeroValues);
    v = rewriter.create<LLVM::ShuffleVectorOp>(loc, v, v, zeroAttrs);

    // Iterate of linear index, convert to coords space and insert splatted 1-D
    // vector in each position.
    nDVectorIterate(vectorTypeInfo, rewriter, [&](ArrayAttr position) {
      desc = rewriter.create<LLVM::InsertValueOp>(loc, llvmNDVectorTy, desc, v,
                                                  position);
    });
    rewriter.replaceOp(splatOp, desc);
    return success();
  }
};

/// Conversion pattern that transforms a subview op into:
///   1. An `llvm.mlir.undef` operation to create a memref descriptor
///   2. Updates to the descriptor to introduce the data ptr, offset, size
///      and stride.
/// The subview op is replaced by the descriptor.
struct SubViewOpLowering : public ConvertOpToLLVMPattern<memref::SubViewOp> {
  using ConvertOpToLLVMPattern<memref::SubViewOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::SubViewOp subViewOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = subViewOp.getLoc();

    auto sourceMemRefType = subViewOp.source().getType().cast<MemRefType>();
    auto sourceElementTy =
        typeConverter->convertType(sourceMemRefType.getElementType());

    auto viewMemRefType = subViewOp.getType();
    auto inferredType = memref::SubViewOp::inferResultType(
                            subViewOp.getSourceType(),
                            extractFromI64ArrayAttr(subViewOp.static_offsets()),
                            extractFromI64ArrayAttr(subViewOp.static_sizes()),
                            extractFromI64ArrayAttr(subViewOp.static_strides()))
                            .cast<MemRefType>();
    auto targetElementTy =
        typeConverter->convertType(viewMemRefType.getElementType());
    auto targetDescTy = typeConverter->convertType(viewMemRefType);
    if (!sourceElementTy || !targetDescTy || !targetElementTy ||
        !LLVM::isCompatibleType(sourceElementTy) ||
        !LLVM::isCompatibleType(targetElementTy) ||
        !LLVM::isCompatibleType(targetDescTy))
      return failure();

    // Extract the offset and strides from the type.
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto successStrides = getStridesAndOffset(inferredType, strides, offset);
    if (failed(successStrides))
      return failure();

    // Create the descriptor.
    if (!LLVM::isCompatibleType(operands.front().getType()))
      return failure();
    MemRefDescriptor sourceMemRef(operands.front());
    auto targetMemRef = MemRefDescriptor::undef(rewriter, loc, targetDescTy);

    // Copy the buffer pointer from the old descriptor to the new one.
    Value extracted = sourceMemRef.allocatedPtr(rewriter, loc);
    Value bitcastPtr = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(targetElementTy,
                                   viewMemRefType.getMemorySpaceAsInt()),
        extracted);
    targetMemRef.setAllocatedPtr(rewriter, loc, bitcastPtr);

    // Copy the aligned pointer from the old descriptor to the new one.
    extracted = sourceMemRef.alignedPtr(rewriter, loc);
    bitcastPtr = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(targetElementTy,
                                   viewMemRefType.getMemorySpaceAsInt()),
        extracted);
    targetMemRef.setAlignedPtr(rewriter, loc, bitcastPtr);

    auto shape = viewMemRefType.getShape();
    auto inferredShape = inferredType.getShape();
    size_t inferredShapeRank = inferredShape.size();
    size_t resultShapeRank = shape.size();
    llvm::SmallDenseSet<unsigned> unusedDims =
        computeRankReductionMask(inferredShape, shape).getValue();

    // Extract strides needed to compute offset.
    SmallVector<Value, 4> strideValues;
    strideValues.reserve(inferredShapeRank);
    for (unsigned i = 0; i < inferredShapeRank; ++i)
      strideValues.push_back(sourceMemRef.stride(rewriter, loc, i));

    // Offset.
    auto llvmIndexType = typeConverter->convertType(rewriter.getIndexType());
    if (!ShapedType::isDynamicStrideOrOffset(offset)) {
      targetMemRef.setConstantOffset(rewriter, loc, offset);
    } else {
      Value baseOffset = sourceMemRef.offset(rewriter, loc);
      // `inferredShapeRank` may be larger than the number of offset operands
      // because of trailing semantics. In this case, the offset is guaranteed
      // to be interpreted as 0 and we can just skip the extra dimensions.
      for (unsigned i = 0, e = std::min(inferredShapeRank,
                                        subViewOp.getMixedOffsets().size());
           i < e; ++i) {
        Value offset =
            // TODO: need OpFoldResult ODS adaptor to clean this up.
            subViewOp.isDynamicOffset(i)
                ? operands[subViewOp.getIndexOfDynamicOffset(i)]
                : rewriter.create<LLVM::ConstantOp>(
                      loc, llvmIndexType,
                      rewriter.getI64IntegerAttr(subViewOp.getStaticOffset(i)));
        Value mul = rewriter.create<LLVM::MulOp>(loc, offset, strideValues[i]);
        baseOffset = rewriter.create<LLVM::AddOp>(loc, baseOffset, mul);
      }
      targetMemRef.setOffset(rewriter, loc, baseOffset);
    }

    // Update sizes and strides.
    SmallVector<OpFoldResult> mixedSizes = subViewOp.getMixedSizes();
    SmallVector<OpFoldResult> mixedStrides = subViewOp.getMixedStrides();
    assert(mixedSizes.size() == mixedStrides.size() &&
           "expected sizes and strides of equal length");
    for (int i = inferredShapeRank - 1, j = resultShapeRank - 1;
         i >= 0 && j >= 0; --i) {
      if (unusedDims.contains(i))
        continue;

      // `i` may overflow subViewOp.getMixedSizes because of trailing semantics.
      // In this case, the size is guaranteed to be interpreted as Dim and the
      // stride as 1.
      Value size, stride;
      if (static_cast<unsigned>(i) >= mixedSizes.size()) {
        size = rewriter.create<LLVM::DialectCastOp>(
            loc, llvmIndexType,
            rewriter.create<memref::DimOp>(loc, subViewOp.source(), i));
        stride = rewriter.create<LLVM::ConstantOp>(
            loc, llvmIndexType, rewriter.getI64IntegerAttr(1));
      } else {
        // TODO: need OpFoldResult ODS adaptor to clean this up.
        size =
            subViewOp.isDynamicSize(i)
                ? operands[subViewOp.getIndexOfDynamicSize(i)]
                : rewriter.create<LLVM::ConstantOp>(
                      loc, llvmIndexType,
                      rewriter.getI64IntegerAttr(subViewOp.getStaticSize(i)));
        if (!ShapedType::isDynamicStrideOrOffset(strides[i])) {
          stride = rewriter.create<LLVM::ConstantOp>(
              loc, llvmIndexType, rewriter.getI64IntegerAttr(strides[i]));
        } else {
          stride = subViewOp.isDynamicStride(i)
                       ? operands[subViewOp.getIndexOfDynamicStride(i)]
                       : rewriter.create<LLVM::ConstantOp>(
                             loc, llvmIndexType,
                             rewriter.getI64IntegerAttr(
                                 subViewOp.getStaticStride(i)));
          stride = rewriter.create<LLVM::MulOp>(loc, stride, strideValues[i]);
        }
      }
      targetMemRef.setSize(rewriter, loc, j, size);
      targetMemRef.setStride(rewriter, loc, j, stride);
      j--;
    }

    rewriter.replaceOp(subViewOp, {targetMemRef});
    return success();
  }
};

/// Conversion pattern that transforms a transpose op into:
///   1. A function entry `alloca` operation to allocate a ViewDescriptor.
///   2. A load of the ViewDescriptor from the pointer allocated in 1.
///   3. Updates to the ViewDescriptor to introduce the data ptr, offset, size
///      and stride. Size and stride are permutations of the original values.
///   4. A store of the resulting ViewDescriptor to the alloca'ed pointer.
/// The transpose op is replaced by the alloca'ed pointer.
class TransposeOpLowering : public ConvertOpToLLVMPattern<memref::TransposeOp> {
public:
  using ConvertOpToLLVMPattern<memref::TransposeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::TransposeOp transposeOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = transposeOp.getLoc();
    memref::TransposeOpAdaptor adaptor(operands);
    MemRefDescriptor viewMemRef(adaptor.in());

    // No permutation, early exit.
    if (transposeOp.permutation().isIdentity())
      return rewriter.replaceOp(transposeOp, {viewMemRef}), success();

    auto targetMemRef = MemRefDescriptor::undef(
        rewriter, loc, typeConverter->convertType(transposeOp.getShapedType()));

    // Copy the base and aligned pointers from the old descriptor to the new
    // one.
    targetMemRef.setAllocatedPtr(rewriter, loc,
                                 viewMemRef.allocatedPtr(rewriter, loc));
    targetMemRef.setAlignedPtr(rewriter, loc,
                               viewMemRef.alignedPtr(rewriter, loc));

    // Copy the offset pointer from the old descriptor to the new one.
    targetMemRef.setOffset(rewriter, loc, viewMemRef.offset(rewriter, loc));

    // Iterate over the dimensions and apply size/stride permutation.
    for (auto en : llvm::enumerate(transposeOp.permutation().getResults())) {
      int sourcePos = en.index();
      int targetPos = en.value().cast<AffineDimExpr>().getPosition();
      targetMemRef.setSize(rewriter, loc, targetPos,
                           viewMemRef.size(rewriter, loc, sourcePos));
      targetMemRef.setStride(rewriter, loc, targetPos,
                             viewMemRef.stride(rewriter, loc, sourcePos));
    }

    rewriter.replaceOp(transposeOp, {targetMemRef});
    return success();
  }
};

/// Conversion pattern that transforms an op into:
///   1. An `llvm.mlir.undef` operation to create a memref descriptor
///   2. Updates to the descriptor to introduce the data ptr, offset, size
///      and stride.
/// The view op is replaced by the descriptor.
struct ViewOpLowering : public ConvertOpToLLVMPattern<memref::ViewOp> {
  using ConvertOpToLLVMPattern<memref::ViewOp>::ConvertOpToLLVMPattern;

  // Build and return the value for the idx^th shape dimension, either by
  // returning the constant shape dimension or counting the proper dynamic size.
  Value getSize(ConversionPatternRewriter &rewriter, Location loc,
                ArrayRef<int64_t> shape, ValueRange dynamicSizes,
                unsigned idx) const {
    assert(idx < shape.size());
    if (!ShapedType::isDynamic(shape[idx]))
      return createIndexConstant(rewriter, loc, shape[idx]);
    // Count the number of dynamic dims in range [0, idx]
    unsigned nDynamic = llvm::count_if(shape.take_front(idx), [](int64_t v) {
      return ShapedType::isDynamic(v);
    });
    return dynamicSizes[nDynamic];
  }

  // Build and return the idx^th stride, either by returning the constant stride
  // or by computing the dynamic stride from the current `runningStride` and
  // `nextSize`. The caller should keep a running stride and update it with the
  // result returned by this function.
  Value getStride(ConversionPatternRewriter &rewriter, Location loc,
                  ArrayRef<int64_t> strides, Value nextSize,
                  Value runningStride, unsigned idx) const {
    assert(idx < strides.size());
    if (!MemRefType::isDynamicStrideOrOffset(strides[idx]))
      return createIndexConstant(rewriter, loc, strides[idx]);
    if (nextSize)
      return runningStride
                 ? rewriter.create<LLVM::MulOp>(loc, runningStride, nextSize)
                 : nextSize;
    assert(!runningStride);
    return createIndexConstant(rewriter, loc, 1);
  }

  LogicalResult
  matchAndRewrite(memref::ViewOp viewOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = viewOp.getLoc();
    memref::ViewOpAdaptor adaptor(operands);

    auto viewMemRefType = viewOp.getType();
    auto targetElementTy =
        typeConverter->convertType(viewMemRefType.getElementType());
    auto targetDescTy = typeConverter->convertType(viewMemRefType);
    if (!targetDescTy || !targetElementTy ||
        !LLVM::isCompatibleType(targetElementTy) ||
        !LLVM::isCompatibleType(targetDescTy))
      return viewOp.emitWarning("Target descriptor type not converted to LLVM"),
             failure();

    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto successStrides = getStridesAndOffset(viewMemRefType, strides, offset);
    if (failed(successStrides))
      return viewOp.emitWarning("cannot cast to non-strided shape"), failure();
    assert(offset == 0 && "expected offset to be 0");

    // Create the descriptor.
    MemRefDescriptor sourceMemRef(adaptor.source());
    auto targetMemRef = MemRefDescriptor::undef(rewriter, loc, targetDescTy);

    // Field 1: Copy the allocated pointer, used for malloc/free.
    Value allocatedPtr = sourceMemRef.allocatedPtr(rewriter, loc);
    auto srcMemRefType = viewOp.source().getType().cast<MemRefType>();
    Value bitcastPtr = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(targetElementTy,
                                   srcMemRefType.getMemorySpaceAsInt()),
        allocatedPtr);
    targetMemRef.setAllocatedPtr(rewriter, loc, bitcastPtr);

    // Field 2: Copy the actual aligned pointer to payload.
    Value alignedPtr = sourceMemRef.alignedPtr(rewriter, loc);
    alignedPtr = rewriter.create<LLVM::GEPOp>(loc, alignedPtr.getType(),
                                              alignedPtr, adaptor.byte_shift());
    bitcastPtr = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(targetElementTy,
                                   srcMemRefType.getMemorySpaceAsInt()),
        alignedPtr);
    targetMemRef.setAlignedPtr(rewriter, loc, bitcastPtr);

    // Field 3: The offset in the resulting type must be 0. This is because of
    // the type change: an offset on srcType* may not be expressible as an
    // offset on dstType*.
    targetMemRef.setOffset(rewriter, loc,
                           createIndexConstant(rewriter, loc, offset));

    // Early exit for 0-D corner case.
    if (viewMemRefType.getRank() == 0)
      return rewriter.replaceOp(viewOp, {targetMemRef}), success();

    // Fields 4 and 5: Update sizes and strides.
    if (strides.back() != 1)
      return viewOp.emitWarning("cannot cast to non-contiguous shape"),
             failure();
    Value stride = nullptr, nextSize = nullptr;
    for (int i = viewMemRefType.getRank() - 1; i >= 0; --i) {
      // Update size.
      Value size =
          getSize(rewriter, loc, viewMemRefType.getShape(), adaptor.sizes(), i);
      targetMemRef.setSize(rewriter, loc, i, size);
      // Update stride.
      stride = getStride(rewriter, loc, strides, nextSize, stride, i);
      targetMemRef.setStride(rewriter, loc, i, stride);
      nextSize = size;
    }

    rewriter.replaceOp(viewOp, {targetMemRef});
    return success();
  }
};

struct AssumeAlignmentOpLowering
    : public ConvertOpToLLVMPattern<memref::AssumeAlignmentOp> {
  using ConvertOpToLLVMPattern<
      memref::AssumeAlignmentOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::AssumeAlignmentOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    memref::AssumeAlignmentOp::Adaptor transformed(operands);
    Value memref = transformed.memref();
    unsigned alignment = op.alignment();
    auto loc = op.getLoc();

    MemRefDescriptor memRefDescriptor(memref);
    Value ptr = memRefDescriptor.alignedPtr(rewriter, memref.getLoc());

    // Emit llvm.assume(memref.alignedPtr & (alignment - 1) == 0). Notice that
    // the asserted memref.alignedPtr isn't used anywhere else, as the real
    // users like load/store/views always re-extract memref.alignedPtr as they
    // get lowered.
    //
    // This relies on LLVM's CSE optimization (potentially after SROA), since
    // after CSE all memref.alignedPtr instances get de-duplicated into the same
    // pointer SSA value.
    auto intPtrType =
        getIntPtrType(memRefDescriptor.getElementPtrType().getAddressSpace());
    Value zero = createIndexAttrConstant(rewriter, loc, intPtrType, 0);
    Value mask =
        createIndexAttrConstant(rewriter, loc, intPtrType, alignment - 1);
    Value ptrValue = rewriter.create<LLVM::PtrToIntOp>(loc, intPtrType, ptr);
    rewriter.create<LLVM::AssumeOp>(
        loc, rewriter.create<LLVM::ICmpOp>(
                 loc, LLVM::ICmpPredicate::eq,
                 rewriter.create<LLVM::AndOp>(loc, ptrValue, mask), zero));

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

/// Try to match the kind of a std.atomic_rmw to determine whether to use a
/// lowering to llvm.atomicrmw or fallback to llvm.cmpxchg.
static Optional<LLVM::AtomicBinOp> matchSimpleAtomicOp(AtomicRMWOp atomicOp) {
  switch (atomicOp.kind()) {
  case AtomicRMWKind::addf:
    return LLVM::AtomicBinOp::fadd;
  case AtomicRMWKind::addi:
    return LLVM::AtomicBinOp::add;
  case AtomicRMWKind::assign:
    return LLVM::AtomicBinOp::xchg;
  case AtomicRMWKind::maxs:
    return LLVM::AtomicBinOp::max;
  case AtomicRMWKind::maxu:
    return LLVM::AtomicBinOp::umax;
  case AtomicRMWKind::mins:
    return LLVM::AtomicBinOp::min;
  case AtomicRMWKind::minu:
    return LLVM::AtomicBinOp::umin;
  default:
    return llvm::None;
  }
  llvm_unreachable("Invalid AtomicRMWKind");
}

namespace {

struct AtomicRMWOpLowering : public LoadStoreOpLowering<AtomicRMWOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(AtomicRMWOp atomicOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(match(atomicOp)))
      return failure();
    auto maybeKind = matchSimpleAtomicOp(atomicOp);
    if (!maybeKind)
      return failure();
    AtomicRMWOp::Adaptor adaptor(operands);
    auto resultType = adaptor.value().getType();
    auto memRefType = atomicOp.getMemRefType();
    auto dataPtr =
        getStridedElementPtr(atomicOp.getLoc(), memRefType, adaptor.memref(),
                             adaptor.indices(), rewriter);
    rewriter.replaceOpWithNewOp<LLVM::AtomicRMWOp>(
        atomicOp, resultType, *maybeKind, dataPtr, adaptor.value(),
        LLVM::AtomicOrdering::acq_rel);
    return success();
  }
};

/// Wrap a llvm.cmpxchg operation in a while loop so that the operation can be
/// retried until it succeeds in atomically storing a new value into memory.
///
///      +---------------------------------+
///      |   <code before the AtomicRMWOp> |
///      |   <compute initial %loaded>     |
///      |   br loop(%loaded)              |
///      +---------------------------------+
///             |
///  -------|   |
///  |      v   v
///  |   +--------------------------------+
///  |   | loop(%loaded):                 |
///  |   |   <body contents>              |
///  |   |   %pair = cmpxchg              |
///  |   |   %ok = %pair[0]               |
///  |   |   %new = %pair[1]              |
///  |   |   cond_br %ok, end, loop(%new) |
///  |   +--------------------------------+
///  |          |        |
///  |-----------        |
///                      v
///      +--------------------------------+
///      | end:                           |
///      |   <code after the AtomicRMWOp> |
///      +--------------------------------+
///
struct GenericAtomicRMWOpLowering
    : public LoadStoreOpLowering<GenericAtomicRMWOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(GenericAtomicRMWOp atomicOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = atomicOp.getLoc();
    GenericAtomicRMWOp::Adaptor adaptor(operands);
    Type valueType = typeConverter->convertType(atomicOp.getResult().getType());

    // Split the block into initial, loop, and ending parts.
    auto *initBlock = rewriter.getInsertionBlock();
    auto *loopBlock =
        rewriter.createBlock(initBlock->getParent(),
                             std::next(Region::iterator(initBlock)), valueType);
    auto *endBlock = rewriter.createBlock(
        loopBlock->getParent(), std::next(Region::iterator(loopBlock)));

    // Operations range to be moved to `endBlock`.
    auto opsToMoveStart = atomicOp->getIterator();
    auto opsToMoveEnd = initBlock->back().getIterator();

    // Compute the loaded value and branch to the loop block.
    rewriter.setInsertionPointToEnd(initBlock);
    auto memRefType = atomicOp.memref().getType().cast<MemRefType>();
    auto dataPtr = getStridedElementPtr(loc, memRefType, adaptor.memref(),
                                        adaptor.indices(), rewriter);
    Value init = rewriter.create<LLVM::LoadOp>(loc, dataPtr);
    rewriter.create<LLVM::BrOp>(loc, init, loopBlock);

    // Prepare the body of the loop block.
    rewriter.setInsertionPointToStart(loopBlock);

    // Clone the GenericAtomicRMWOp region and extract the result.
    auto loopArgument = loopBlock->getArgument(0);
    BlockAndValueMapping mapping;
    mapping.map(atomicOp.getCurrentValue(), loopArgument);
    Block &entryBlock = atomicOp.body().front();
    for (auto &nestedOp : entryBlock.without_terminator()) {
      Operation *clone = rewriter.clone(nestedOp, mapping);
      mapping.map(nestedOp.getResults(), clone->getResults());
    }
    Value result = mapping.lookup(entryBlock.getTerminator()->getOperand(0));

    // Prepare the epilog of the loop block.
    // Append the cmpxchg op to the end of the loop block.
    auto successOrdering = LLVM::AtomicOrdering::acq_rel;
    auto failureOrdering = LLVM::AtomicOrdering::monotonic;
    auto boolType = IntegerType::get(rewriter.getContext(), 1);
    auto pairType = LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                                     {valueType, boolType});
    auto cmpxchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
        loc, pairType, dataPtr, loopArgument, result, successOrdering,
        failureOrdering);
    // Extract the %new_loaded and %ok values from the pair.
    Value newLoaded = rewriter.create<LLVM::ExtractValueOp>(
        loc, valueType, cmpxchg, rewriter.getI64ArrayAttr({0}));
    Value ok = rewriter.create<LLVM::ExtractValueOp>(
        loc, boolType, cmpxchg, rewriter.getI64ArrayAttr({1}));

    // Conditionally branch to the end or back to the loop depending on %ok.
    rewriter.create<LLVM::CondBrOp>(loc, ok, endBlock, ArrayRef<Value>(),
                                    loopBlock, newLoaded);

    rewriter.setInsertionPointToEnd(endBlock);
    moveOpsRange(atomicOp.getResult(), newLoaded, std::next(opsToMoveStart),
                 std::next(opsToMoveEnd), rewriter);

    // The 'result' of the atomic_rmw op is the newly loaded value.
    rewriter.replaceOp(atomicOp, {newLoaded});

    return success();
  }

private:
  // Clones a segment of ops [start, end) and erases the original.
  void moveOpsRange(ValueRange oldResult, ValueRange newResult,
                    Block::iterator start, Block::iterator end,
                    ConversionPatternRewriter &rewriter) const {
    BlockAndValueMapping mapping;
    mapping.map(oldResult, newResult);
    SmallVector<Operation *, 2> opsToErase;
    for (auto it = start; it != end; ++it) {
      rewriter.clone(*it, mapping);
      opsToErase.push_back(&*it);
    }
    for (auto *it : opsToErase)
      rewriter.eraseOp(it);
  }
};

} // namespace

/// Collect a set of patterns to convert from the Standard dialect to LLVM.
void mlir::populateStdToLLVMNonMemoryConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  // FIXME: this should be tablegen'ed
  // clang-format off
  patterns.add<
      AbsFOpLowering,
      AddFOpLowering,
      AddIOpLowering,
      AllocaOpLowering,
      AllocaScopeOpLowering,
      AndOpLowering,
      AssertOpLowering,
      AtomicRMWOpLowering,
      BranchOpLowering,
      CallIndirectOpLowering,
      CallOpLowering,
      CeilFOpLowering,
      CmpFOpLowering,
      CmpIOpLowering,
      CondBranchOpLowering,
      CopySignOpLowering,
      CosOpLowering,
      ConstantOpLowering,
      DialectCastOpLowering,
      DivFOpLowering,
      ExpOpLowering,
      Exp2OpLowering,
      ExpM1OpLowering,
      FloorFOpLowering,
      FmaFOpLowering,
      GenericAtomicRMWOpLowering,
      LogOpLowering,
      Log10OpLowering,
      Log1pOpLowering,
      Log2OpLowering,
      FPExtOpLowering,
      FPToSIOpLowering,
      FPToUIOpLowering,
      FPTruncOpLowering,
      IndexCastOpLowering,
      MulFOpLowering,
      MulIOpLowering,
      NegFOpLowering,
      OrOpLowering,
      PowFOpLowering,
      PrefetchOpLowering,
      RemFOpLowering,
      ReturnOpLowering,
      RsqrtOpLowering,
      SIToFPOpLowering,
      SelectOpLowering,
      ShiftLeftOpLowering,
      SignExtendIOpLowering,
      SignedDivIOpLowering,
      SignedRemIOpLowering,
      SignedShiftRightOpLowering,
      SinOpLowering,
      SplatOpLowering,
      SplatNdOpLowering,
      SqrtOpLowering,
      SubFOpLowering,
      SubIOpLowering,
      SwitchOpLowering,
      TruncateIOpLowering,
      UIToFPOpLowering,
      UnsignedDivIOpLowering,
      UnsignedRemIOpLowering,
      UnsignedShiftRightOpLowering,
      XOrOpLowering,
      ZeroExtendIOpLowering>(converter);
  // clang-format on
}

void mlir::populateStdToLLVMMemoryConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
      AssumeAlignmentOpLowering,
      DimOpLowering,
      GlobalMemrefOpLowering,
      GetGlobalMemrefOpLowering,
      LoadOpLowering,
      MemRefCastOpLowering,
      MemRefCopyOpLowering,
      MemRefReinterpretCastOpLowering,
      MemRefReshapeOpLowering,
      RankOpLowering,
      StoreOpLowering,
      SubViewOpLowering,
      TransposeOpLowering,
      ViewOpLowering>(converter);
  // clang-format on
  auto allocLowering = converter.getOptions().allocLowering;
  if (allocLowering == LowerToLLVMOptions::AllocLowering::AlignedAlloc)
    patterns.add<AlignedAllocOpLowering, DeallocOpLowering>(converter);
  else if (allocLowering == LowerToLLVMOptions::AllocLowering::Malloc)
    patterns.add<AllocOpLowering, DeallocOpLowering>(converter);
}

void mlir::populateStdToLLVMFuncOpConversionPattern(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  if (converter.getOptions().useBarePtrCallConv)
    patterns.add<BarePtrFuncOpConversion>(converter);
  else
    patterns.add<FuncOpConversion>(converter);
}

void mlir::populateStdToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                               RewritePatternSet &patterns) {
  populateStdToLLVMFuncOpConversionPattern(converter, patterns);
  populateStdToLLVMNonMemoryConversionPatterns(converter, patterns);
  populateStdToLLVMMemoryConversionPatterns(converter, patterns);
}

namespace {
/// A pass converting MLIR operations into the LLVM IR dialect.
struct LLVMLoweringPass : public ConvertStandardToLLVMBase<LLVMLoweringPass> {
  LLVMLoweringPass() = default;
  LLVMLoweringPass(bool useBarePtrCallConv, bool emitCWrappers,
                   unsigned indexBitwidth, bool useAlignedAlloc,
                   const llvm::DataLayout &dataLayout) {
    this->useBarePtrCallConv = useBarePtrCallConv;
    this->emitCWrappers = emitCWrappers;
    this->indexBitwidth = indexBitwidth;
    this->useAlignedAlloc = useAlignedAlloc;
    this->dataLayout = dataLayout.getStringRepresentation();
  }

  /// Run the dialect converter on the module.
  void runOnOperation() override {
    if (useBarePtrCallConv && emitCWrappers) {
      getOperation().emitError()
          << "incompatible conversion options: bare-pointer calling convention "
             "and C wrapper emission";
      signalPassFailure();
      return;
    }
    if (failed(LLVM::LLVMDialect::verifyDataLayoutString(
            this->dataLayout, [this](const Twine &message) {
              getOperation().emitError() << message.str();
            }))) {
      signalPassFailure();
      return;
    }

    ModuleOp m = getOperation();
    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();

    LowerToLLVMOptions options(&getContext(),
                               dataLayoutAnalysis.getAtOrAbove(m));
    options.useBarePtrCallConv = useBarePtrCallConv;
    options.emitCWrappers = emitCWrappers;
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);
    options.allocLowering =
        (useAlignedAlloc ? LowerToLLVMOptions::AllocLowering::AlignedAlloc
                         : LowerToLLVMOptions::AllocLowering::Malloc);
    options.dataLayout = llvm::DataLayout(this->dataLayout);

    LLVMTypeConverter typeConverter(&getContext(), options,
                                    &dataLayoutAnalysis);

    RewritePatternSet patterns(&getContext());
    populateStdToLLVMConversionPatterns(typeConverter, patterns);

    LLVMConversionTarget target(getContext());
    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();
    m->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(),
               StringAttr::get(m.getContext(), this->dataLayout));
  }
};
} // end namespace

Value AllocLikeOpLLVMLowering::createAligned(
    ConversionPatternRewriter &rewriter, Location loc, Value input,
    Value alignment) {
  Value one = createIndexAttrConstant(rewriter, loc, alignment.getType(), 1);
  Value bump = rewriter.create<LLVM::SubOp>(loc, alignment, one);
  Value bumped = rewriter.create<LLVM::AddOp>(loc, input, bump);
  Value mod = rewriter.create<LLVM::URemOp>(loc, bumped, alignment);
  return rewriter.create<LLVM::SubOp>(loc, bumped, mod);
}

LogicalResult AllocLikeOpLLVMLowering::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  MemRefType memRefType = getMemRefResultType(op);
  if (!isConvertibleAndHasIdentityMaps(memRefType))
    return rewriter.notifyMatchFailure(op, "incompatible memref type");
  auto loc = op->getLoc();

  // Get actual sizes of the memref as values: static sizes are constant
  // values and dynamic sizes are passed to 'alloc' as operands.  In case of
  // zero-dimensional memref, assume a scalar (size 1).
  SmallVector<Value, 4> sizes;
  SmallVector<Value, 4> strides;
  Value sizeBytes;
  this->getMemRefDescriptorSizes(loc, memRefType, operands, rewriter, sizes,
                                 strides, sizeBytes);

  // Allocate the underlying buffer.
  Value allocatedPtr;
  Value alignedPtr;
  std::tie(allocatedPtr, alignedPtr) =
      this->allocateBuffer(rewriter, loc, sizeBytes, op);

  // Create the MemRef descriptor.
  auto memRefDescriptor = this->createMemRefDescriptor(
      loc, memRefType, allocatedPtr, alignedPtr, sizes, strides, rewriter);

  // Return the final value of the descriptor.
  rewriter.replaceOp(op, {memRefDescriptor});
  return success();
}

mlir::LLVMConversionTarget::LLVMConversionTarget(MLIRContext &ctx)
    : ConversionTarget(ctx) {
  this->addLegalDialect<LLVM::LLVMDialect>();
  this->addIllegalOp<LLVM::DialectCastOp>();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createLowerToLLVMPass() {
  return std::make_unique<LLVMLoweringPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createLowerToLLVMPass(const LowerToLLVMOptions &options) {
  auto allocLowering = options.allocLowering;
  // There is no way to provide additional patterns for pass, so
  // AllocLowering::None will always fail.
  assert(allocLowering != LowerToLLVMOptions::AllocLowering::None &&
         "LLVMLoweringPass doesn't support AllocLowering::None");
  bool useAlignedAlloc =
      (allocLowering == LowerToLLVMOptions::AllocLowering::AlignedAlloc);
  return std::make_unique<LLVMLoweringPass>(
      options.useBarePtrCallConv, options.emitCWrappers,
      options.getIndexBitwidth(), useAlignedAlloc, options.dataLayout);
}

