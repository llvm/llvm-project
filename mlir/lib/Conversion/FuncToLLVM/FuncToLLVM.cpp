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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include <algorithm>
#include <functional>

namespace mlir {
#define GEN_PASS_DEF_CONVERTFUNCTOLLVM
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define PASS_NAME "convert-func-to-llvm"

static constexpr StringRef varargsAttrName = "func.varargs";
static constexpr StringRef linkageAttrName = "llvm.linkage";

/// Only retain those attributes that are not constructed by
/// `LLVMFuncOp::build`. If `filterArgAttrs` is set, also filter out argument
/// attributes.
static void filterFuncAttributes(func::FuncOp func, bool filterArgAndResAttrs,
                                 SmallVectorImpl<NamedAttribute> &result) {
  for (const NamedAttribute &attr : func->getAttrs()) {
    if (attr.getName() == SymbolTable::getSymbolAttrName() ||
        attr.getName() == func.getFunctionTypeAttrName() ||
        attr.getName() == linkageAttrName ||
        attr.getName() == varargsAttrName ||
        attr.getName() == LLVM::LLVMDialect::getReadnoneAttrName() ||
        (filterArgAndResAttrs &&
         (attr.getName() == func.getArgAttrsAttrName() ||
          attr.getName() == func.getResAttrsAttrName())))
      continue;
    result.push_back(attr);
  }
}

/// Helper function for wrapping all attributes into a single DictionaryAttr
static auto wrapAsStructAttrs(OpBuilder &b, ArrayAttr attrs) {
  return DictionaryAttr::get(
      b.getContext(),
      b.getNamedAttr(LLVM::LLVMDialect::getStructAttrsAttrName(), attrs));
}

/// Combines all result attributes into a single DictionaryAttr
/// and prepends to argument attrs.
/// This is intended to be used to format the attributes for a C wrapper
/// function when the result(s) is converted to the first function argument
/// (in the multiple return case, all returns get wrapped into a single
/// argument). The total number of argument attributes should be equal to
/// (number of function arguments) + 1.
static void
prependResAttrsToArgAttrs(OpBuilder &builder,
                          SmallVectorImpl<NamedAttribute> &attributes,
                          func::FuncOp func) {
  size_t numArguments = func.getNumArguments();
  auto allAttrs = SmallVector<Attribute>(
      numArguments + 1, DictionaryAttr::get(builder.getContext()));
  NamedAttribute *argAttrs = nullptr;
  for (auto *it = attributes.begin(); it != attributes.end();) {
    if (it->getName() == func.getArgAttrsAttrName()) {
      auto arrayAttrs = it->getValue().cast<ArrayAttr>();
      assert(arrayAttrs.size() == numArguments &&
             "Number of arg attrs and args should match");
      std::copy(arrayAttrs.begin(), arrayAttrs.end(), allAttrs.begin() + 1);
      argAttrs = it;
    } else if (it->getName() == func.getResAttrsAttrName()) {
      auto arrayAttrs = it->getValue().cast<ArrayAttr>();
      assert(!arrayAttrs.empty() && "expected array to be non-empty");
      allAttrs[0] = (arrayAttrs.size() == 1)
                        ? arrayAttrs[0]
                        : wrapAsStructAttrs(builder, arrayAttrs);
      it = attributes.erase(it);
      continue;
    }
    it++;
  }

  auto newArgAttrs = builder.getNamedAttr(func.getArgAttrsAttrName(),
                                          builder.getArrayAttr(allAttrs));
  if (!argAttrs) {
    attributes.emplace_back(newArgAttrs);
    return;
  }
  *argAttrs = newArgAttrs;
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
                                   func::FuncOp funcOp,
                                   LLVM::LLVMFuncOp newFuncOp) {
  auto type = funcOp.getFunctionType();
  SmallVector<NamedAttribute, 4> attributes;
  filterFuncAttributes(funcOp, /*filterArgAndResAttrs=*/false, attributes);
  auto [wrapperFuncType, resultIsNowArg] =
      typeConverter.convertFunctionTypeCWrapper(type);
  if (resultIsNowArg)
    prependResAttrsToArgAttrs(rewriter, attributes, funcOp);
  auto wrapperFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
      loc, llvm::formatv("_mlir_ciface_{0}", funcOp.getName()).str(),
      wrapperFuncType, LLVM::Linkage::External, /*dsoLocal*/ false,
      /*cconv*/ LLVM::CConv::C, attributes);

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
                                 LLVMTypeConverter &typeConverter,
                                 func::FuncOp funcOp,
                                 LLVM::LLVMFuncOp newFuncOp) {
  OpBuilder::InsertionGuard guard(builder);

  auto [wrapperType, resultIsNowArg] =
      typeConverter.convertFunctionTypeCWrapper(funcOp.getFunctionType());
  // This conversion can only fail if it could not convert one of the argument
  // types. But since it has been applied to a non-wrapper function before, it
  // should have failed earlier and not reach this point at all.
  assert(wrapperType && "unexpected type conversion failure");

  SmallVector<NamedAttribute, 4> attributes;
  filterFuncAttributes(funcOp, /*filterArgAndResAttrs=*/false, attributes);

  if (resultIsNowArg)
    prependResAttrsToArgAttrs(builder, attributes, funcOp);
  // Create the auxiliary function.
  auto wrapperFunc = builder.create<LLVM::LLVMFuncOp>(
      loc, llvm::formatv("_mlir_ciface_{0}", funcOp.getName()).str(),
      wrapperType, LLVM::Linkage::External, /*dsoLocal*/ false,
      /*cconv*/ LLVM::CConv::C, attributes);

  builder.setInsertionPointToStart(newFuncOp.addEntryBlock());

  // Get a ValueRange containing arguments.
  FunctionType type = funcOp.getFunctionType();
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
    builder.create<LLVM::ReturnOp>(loc, result);
  } else {
    builder.create<LLVM::ReturnOp>(loc, call.getResults());
  }
}

namespace {

struct FuncOpConversionBase : public ConvertOpToLLVMPattern<func::FuncOp> {
protected:
  using ConvertOpToLLVMPattern<func::FuncOp>::ConvertOpToLLVMPattern;

  // Convert input FuncOp to LLVMFuncOp by using the LLVMTypeConverter provided
  // to this legalization pattern.
  LLVM::LLVMFuncOp
  convertFuncOpToLLVMFuncOp(func::FuncOp funcOp,
                            ConversionPatternRewriter &rewriter) const {
    // Convert the original function arguments. They are converted using the
    // LLVMTypeConverter provided to this legalization pattern.
    auto varargsAttr = funcOp->getAttrOfType<BoolAttr>(varargsAttrName);
    TypeConverter::SignatureConversion result(funcOp.getNumArguments());
    auto llvmType = getTypeConverter()->convertFunctionSignature(
        funcOp.getFunctionType(), varargsAttr && varargsAttr.getValue(),
        result);
    if (!llvmType)
      return nullptr;

    // Propagate argument/result attributes to all converted arguments/result
    // obtained after converting a given original argument/result.
    SmallVector<NamedAttribute, 4> attributes;
    filterFuncAttributes(funcOp, /*filterArgAndResAttrs=*/true, attributes);
    if (ArrayAttr resAttrDicts = funcOp.getAllResultAttrs()) {
      assert(!resAttrDicts.empty() && "expected array to be non-empty");
      auto newResAttrDicts =
          (funcOp.getNumResults() == 1)
              ? resAttrDicts
              : rewriter.getArrayAttr(
                    {wrapAsStructAttrs(rewriter, resAttrDicts)});
      attributes.push_back(
          rewriter.getNamedAttr(funcOp.getResAttrsAttrName(), newResAttrDicts));
    }
    if (ArrayAttr argAttrDicts = funcOp.getAllArgAttrs()) {
      SmallVector<Attribute, 4> newArgAttrs(
          llvmType.cast<LLVM::LLVMFunctionType>().getNumParams());
      for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
        // Some LLVM IR attribute have a type attached to them. During FuncOp ->
        // LLVMFuncOp conversion these types may have changed. Account for that
        // change by converting attributes' types as well.
        SmallVector<NamedAttribute, 4> convertedAttrs;
        auto attrsDict = argAttrDicts[i].cast<DictionaryAttr>();
        convertedAttrs.reserve(attrsDict.size());
        for (const NamedAttribute &attr : attrsDict) {
          const auto convert = [&](const NamedAttribute &attr) {
            return TypeAttr::get(getTypeConverter()->convertType(
                attr.getValue().cast<TypeAttr>().getValue()));
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
        for (size_t j = 0; j < mapping->size; ++j)
          newArgAttrs[mapping->inputNo + j] =
              DictionaryAttr::get(rewriter.getContext(), convertedAttrs);
      }
      attributes.push_back(rewriter.getNamedAttr(
          funcOp.getArgAttrsAttrName(), rewriter.getArrayAttr(newArgAttrs)));
    }

    // Create an LLVM function, use external linkage by default until MLIR
    // functions have linkage.
    LLVM::Linkage linkage = LLVM::Linkage::External;
    if (funcOp->hasAttr(linkageAttrName)) {
      auto attr =
          funcOp->getAttr(linkageAttrName).dyn_cast<mlir::LLVM::LinkageAttr>();
      if (!attr) {
        funcOp->emitError() << "Contains " << linkageAttrName
                            << " attribute not of type LLVM::LinkageAttr";
        return nullptr;
      }
      linkage = attr.getLinkage();
    }

    // Create a memory effect attribute corresponding to readnone.
    StringRef readnoneAttrName = LLVM::LLVMDialect::getReadnoneAttrName();
    LLVM::MemoryEffectsAttr memoryAttr = {};
    if (funcOp->hasAttr(readnoneAttrName)) {
      auto attr = funcOp->getAttrOfType<UnitAttr>(readnoneAttrName);
      if (!attr) {
        funcOp->emitError() << "Contains " << readnoneAttrName
                            << " attribute not of type UnitAttr";
        return nullptr;
      }
      memoryAttr = LLVM::MemoryEffectsAttr::get(rewriter.getContext(),
                                                {LLVM::ModRefInfo::NoModRef,
                                                 LLVM::ModRefInfo::NoModRef,
                                                 LLVM::ModRefInfo::NoModRef});
    }
    auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        funcOp.getLoc(), funcOp.getName(), llvmType, linkage,
        /*dsoLocal*/ false, /*cconv*/ LLVM::CConv::C, attributes);
    // If the memory attribute was created, add it to the function.
    if (memoryAttr)
      newFuncOp.setMemoryAttr(memoryAttr);
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
struct FuncOpConversion : public FuncOpConversionBase {
  FuncOpConversion(LLVMTypeConverter &converter)
      : FuncOpConversionBase(converter) {}

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newFuncOp = convertFuncOpToLLVMFuncOp(funcOp, rewriter);
    if (!newFuncOp)
      return failure();

    if (funcOp->getAttrOfType<UnitAttr>(
            LLVM::LLVMDialect::getEmitCWrapperAttrName())) {
      if (newFuncOp.isVarArg())
        return funcOp->emitError("C interface for variadic functions is not "
                                 "supported yet.");

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
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // TODO: bare ptr conversion could be handled by argument materialization
    // and most of the code below would go away. But to do this, we would need a
    // way to distinguish between FuncOp and other regions in the
    // addArgumentMaterialization hook.

    // Store the type of memref-typed arguments before the conversion so that we
    // can promote them to MemRef descriptor at the beginning of the function.
    SmallVector<Type, 8> oldArgTypes =
        llvm::to_vector<8>(funcOp.getFunctionType().getInputs());

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
      auto placeholder = rewriter.create<LLVM::UndefOp>(
          loc, getTypeConverter()->convertType(memrefTy));
      rewriter.replaceUsesOfBlockArgument(arg, placeholder);

      Value desc = MemRefDescriptor::fromStaticShape(
          rewriter, loc, *getTypeConverter(), memrefTy, arg);
      rewriter.replaceOp(placeholder, {desc});
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

  LogicalResult
  matchAndRewrite(CallOpType callOp, typename CallOpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
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
        callOp.getLoc(), /*opOperands=*/callOp->getOperands(),
        adaptor.getOperands(), rewriter);
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

    if (this->getTypeConverter()->getOptions().useBarePtrCallConv) {
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

struct CallOpLowering : public CallOpInterfaceLowering<func::CallOp> {
  using Super::Super;
};

struct CallIndirectOpLowering
    : public CallOpInterfaceLowering<func::CallIndirectOp> {
  using Super::Super;
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

    if (getTypeConverter()->getOptions().useBarePtrCallConv) {
      // For the bare-ptr calling convention, extract the aligned pointer to
      // be returned from the memref descriptor.
      for (auto it : llvm::zip(op->getOperands(), adaptor.getOperands())) {
        Type oldTy = std::get<0>(it).getType();
        Value newOperand = std::get<1>(it);
        if (oldTy.isa<MemRefType>() && getTypeConverter()->canConvertToBarePtr(
                                           oldTy.cast<BaseMemRefType>())) {
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
    auto packedType =
        getTypeConverter()->packFunctionResults(op.getOperandTypes());

    Value packed = rewriter.create<LLVM::UndefOp>(loc, packedType);
    for (auto &it : llvm::enumerate(updatedOperands)) {
      packed = rewriter.create<LLVM::InsertValueOp>(loc, packed, it.value(),
                                                    it.index());
    }
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(), packed,
                                                op->getAttrs());
    return success();
  }
};
} // namespace

void mlir::populateFuncToLLVMFuncOpConversionPattern(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  if (converter.getOptions().useBarePtrCallConv)
    patterns.add<BarePtrFuncOpConversion>(converter);
  else
    patterns.add<FuncOpConversion>(converter);
}

void mlir::populateFuncToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                RewritePatternSet &patterns) {
  populateFuncToLLVMFuncOpConversionPattern(converter, patterns);
  // clang-format off
  patterns.add<
      CallIndirectOpLowering,
      CallOpLowering,
      ConstantOpLowering,
      ReturnOpLowering>(converter);
  // clang-format on
}

namespace {
/// A pass converting Func operations into the LLVM IR dialect.
struct ConvertFuncToLLVMPass
    : public impl::ConvertFuncToLLVMBase<ConvertFuncToLLVMPass> {
  ConvertFuncToLLVMPass() = default;
  ConvertFuncToLLVMPass(bool useBarePtrCallConv, unsigned indexBitwidth,
                        bool useAlignedAlloc,
                        const llvm::DataLayout &dataLayout) {
    this->useBarePtrCallConv = useBarePtrCallConv;
    this->indexBitwidth = indexBitwidth;
    this->dataLayout = dataLayout.getStringRepresentation();
  }

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
    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();

    LowerToLLVMOptions options(&getContext(),
                               dataLayoutAnalysis.getAtOrAbove(m));
    options.useBarePtrCallConv = useBarePtrCallConv;
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);
    options.dataLayout = llvm::DataLayout(this->dataLayout);

    LLVMTypeConverter typeConverter(&getContext(), options,
                                    &dataLayoutAnalysis);

    RewritePatternSet patterns(&getContext());
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);

    // TODO: Remove these in favor of their dedicated conversion passes.
    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

    LLVMConversionTarget target(getContext());
    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();

    m->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(),
               StringAttr::get(m.getContext(), this->dataLayout));
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertFuncToLLVMPass() {
  return std::make_unique<ConvertFuncToLLVMPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertFuncToLLVMPass(const LowerToLLVMOptions &options) {
  auto allocLowering = options.allocLowering;
  // There is no way to provide additional patterns for pass, so
  // AllocLowering::None will always fail.
  assert(allocLowering != LowerToLLVMOptions::AllocLowering::None &&
         "ConvertFuncToLLVMPass doesn't support AllocLowering::None");
  bool useAlignedAlloc =
      (allocLowering == LowerToLLVMOptions::AllocLowering::AlignedAlloc);
  return std::make_unique<ConvertFuncToLLVMPass>(
      options.useBarePtrCallConv, options.getIndexBitwidth(), useAlignedAlloc,
      options.dataLayout);
}
