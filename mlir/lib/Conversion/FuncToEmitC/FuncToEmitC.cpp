//===- FuncToEmitC.cpp - Func to EmitC Patterns -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert the Func dialect to the EmitC
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/FuncToEmitC/FuncToEmitC.h"

#include "mlir/Conversion/ConvertToEmitC/ToEmitCInterface.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Multi-return struct helpers
//===----------------------------------------------------------------------===//

// Looks up or creates an `emitc.class` named after `types` in the nearest
// enclosing symbol table of `op`, suitable for packing those types as plain
// struct fields (field0, field1, ...). If the class already exists it is
// verified to have exactly the right fields and no methods. Returns the
// corresponding !emitc.opaque<"struct ..."> type on success.
static FailureOr<emitc::OpaqueType>
getOrCreateMultiReturnType(ConversionPatternRewriter &rewriter, Location loc,
                           Operation *op, TypeRange types) {
  // Build the struct name from the types, e.g. "return_i32_i32". Each type is
  // printed and non-alphanumeric characters are replaced with '_'.
  std::string structName = "return";
  for (Type type : types) {
    std::string typeName;
    llvm::raw_string_ostream os(typeName);
    type.print(os);
    std::replace_if(
        typeName.begin(), typeName.end(),
        [](char c) { return !llvm::isAlnum(c); }, '_');
    structName += "_" + typeName;
  }

  // Find the enclosing symbol table and the direct child op within it that
  // contains `op`; the class will be inserted immediately before that child.
  Operation *symbolTableOp = SymbolTable::getNearestSymbolTable(op);
  Operation *insertBefore = op;
  while (insertBefore->getParentOp() != symbolTableOp)
    insertBefore = insertBefore->getParentOp();

  if (Operation *sym = SymbolTable::lookupSymbolIn(symbolTableOp, structName)) {
    auto classOp = dyn_cast<emitc::ClassOp>(sym);
    if (!classOp)
      return emitError(loc) << "symbol '" << structName
                            << "' exists but is not an emitc.class";

    if (classOp.getClassType() != emitc::ClassType::struct_)
      return emitError(loc)
             << "existing class '" << structName << "' is not a struct";

    SmallVector<emitc::FieldOp> fields;
    for (Operation &bodyOp : classOp.getBody().front()) {
      if (isa<emitc::FuncOp>(bodyOp))
        return emitError(loc) << "existing class '" << structName
                              << "' has methods; expected a plain struct";
      if (auto fieldOp = dyn_cast<emitc::FieldOp>(bodyOp))
        fields.push_back(fieldOp);
    }
    if (fields.size() != types.size())
      return emitError(loc) << "existing class '" << structName
                            << "' has wrong number of fields";
    for (auto [i, fieldOp] : llvm::enumerate(fields)) {
      if (fieldOp.getSymName() != "field" + std::to_string(i))
        return emitError(loc) << "existing class '" << structName
                              << "': unexpected field name at index " << i;
      if (fieldOp.getTypeAttr().getValue() != types[i])
        return emitError(loc) << "existing class '" << structName
                              << "': wrong type for field " << i;
    }
  } else {
    // Create the ClassOp before `insertBefore`, then restore the insertion
    // point.
    auto savedIP = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(insertBefore);

    emitc::ClassOp classOp = emitc::ClassOp::create(rewriter, loc, structName,
                                                    /*final_specifier=*/false,
                                                    emitc::ClassType::struct_);
    rewriter.createBlock(&classOp.getBody());
    rewriter.setInsertionPointToStart(&classOp.getBody().front());

    for (auto [i, type] : llvm::enumerate(types)) {
      auto fieldName = rewriter.getStringAttr("field" + std::to_string(i));
      emitc::FieldOp::create(rewriter, loc, fieldName, TypeAttr::get(type),
                             nullptr);
    }

    rewriter.restoreInsertionPoint(savedIP);
  }
  return emitc::OpaqueType::get(rewriter.getContext(), "struct " + structName);
}

// Packs multiple SSA values into an emitc.class struct variable and loads the
// result as a single SSA value of the opaque struct type.
static Value packValuesIntoStruct(ConversionPatternRewriter &rewriter,
                                  Location loc, ValueRange values,
                                  emitc::OpaqueType structType) {
  MLIRContext *ctx = rewriter.getContext();
  auto noInit = emitc::OpaqueAttr::get(ctx, "");
  Value structLv =
      emitc::VariableOp::create(rewriter, loc,
                                emitc::LValueType::get(structType), noInit)
          .getResult();
  for (auto [i, val] : llvm::enumerate(values)) {
    Value fieldLv =
        emitc::MemberOp::create(
            rewriter, loc, emitc::LValueType::get(val.getType()),
            rewriter.getStringAttr("field" + std::to_string(i)), structLv)
            .getResult();
    emitc::AssignOp::create(rewriter, loc, fieldLv, val);
  }
  return emitc::LoadOp::create(rewriter, loc, structType, structLv).getResult();
}

/// Implement the interface to convert Func to EmitC.
struct FuncToEmitCDialectInterface : public ConvertToEmitCPatternInterface {
  FuncToEmitCDialectInterface(Dialect *dialect)
      : ConvertToEmitCPatternInterface(dialect) {}

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToEmitCConversionPatterns(
      ConversionTarget &target, TypeConverter &typeConverter,
      RewritePatternSet &patterns, std::optional<bool> lowerToCpp) const final {
    populateFuncToEmitCPatterns(typeConverter, patterns,
                                lowerToCpp.value_or(true));
  }
};
} // namespace

void mlir::registerConvertFuncToEmitCInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, func::FuncDialect *dialect) {
    dialect->addInterfaces<FuncToEmitCDialectInterface>();
  });
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {
class CallOpConversion final : public OpConversionPattern<func::CallOp> {
public:
  CallOpConversion(const TypeConverter &typeConverter, MLIRContext *ctx,
                   bool lowerToCpp)
      : OpConversionPattern<func::CallOp>(typeConverter, ctx),
        lowerToCpp(lowerToCpp) {}

  LogicalResult
  matchAndRewrite(func::CallOp callOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Do not convert multiple-return functions if lowering target is Cpp.
    // The translator will emit the return values as an std::tuple.
    if (callOp.getNumResults() > 1 && lowerToCpp)
      return rewriter.notifyMatchFailure(
          callOp, "only functions with zero or one result can be converted");

    SmallVector<Type> convertedResultTypes;
    for (Type t : callOp.getResultTypes()) {
      Type resultType = getTypeConverter()->convertType(t);
      if (!resultType)
        return rewriter.notifyMatchFailure(callOp,
                                           "result type conversion failed");
      if (isa<emitc::ArrayType>(resultType))
        return rewriter.notifyMatchFailure(
            callOp, "function calls returning arrays are not supported");
      convertedResultTypes.push_back(resultType);
    }

    if (callOp.getNumResults() <= 1) {
      rewriter.replaceOpWithNewOp<emitc::CallOp>(
          callOp, callOp.getResultTypes(), adaptor.getOperands(),
          callOp->getAttrs());
      return success();
    }

    // Multi-result call: determine the struct type.
    Location loc = callOp.getLoc();

    auto structType =
        getOrCreateMultiReturnType(rewriter, loc, callOp, convertedResultTypes);
    if (failed(structType))
      return rewriter.notifyMatchFailure(callOp,
                                         "incompatible multi-return struct");

    // Emit a call returning the packed struct.
    Value structVal =
        emitc::CallOp::create(rewriter, loc, callOp.getCalleeAttr(),
                              TypeRange{*structType}, adaptor.getOperands())
            .getResult(0);

    // Unpack struct fields to replace the original multiple results.
    MLIRContext *ctx = rewriter.getContext();
    auto noInit = emitc::OpaqueAttr::get(ctx, "");
    Value structLv =
        emitc::VariableOp::create(rewriter, loc,
                                  emitc::LValueType::get(*structType), noInit)
            .getResult();
    emitc::AssignOp::create(rewriter, loc, structLv, structVal);
    SmallVector<Value> results;
    for (auto [i, result] : llvm::enumerate(callOp.getResults())) {
      if (result.use_empty()) {
        results.push_back(Value()); // No replacement needed.
        continue;
      }
      Type fieldType = convertedResultTypes[i];
      StringAttr fieldName =
          rewriter.getStringAttr("field" + std::to_string(i));
      Value fieldLv = emitc::MemberOp::create(rewriter, loc,
                                              emitc::LValueType::get(fieldType),
                                              fieldName, structLv)
                          .getResult();
      results.push_back(
          emitc::LoadOp::create(rewriter, loc, fieldType, fieldLv).getResult());
    }

    rewriter.replaceOp(callOp, results);
    return success();
  }

private:
  bool lowerToCpp;
};

class FuncOpConversion final : public OpConversionPattern<func::FuncOp> {
public:
  FuncOpConversion(const TypeConverter &typeConverter, MLIRContext *ctx,
                   bool lowerToCpp)
      : OpConversionPattern<func::FuncOp>(typeConverter, ctx),
        lowerToCpp(lowerToCpp) {}

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FunctionType fnType = funcOp.getFunctionType();

    // Do not convert multiple-return functions if lowering target is Cpp.
    // The translator will emit the return values as an std::tuple.
    if (fnType.getNumResults() > 1 && lowerToCpp)
      return rewriter.notifyMatchFailure(
          funcOp, "only functions with zero or one result can be converted");

    TypeConverter::SignatureConversion signatureConverter(
        fnType.getNumInputs());
    for (const auto &argType : enumerate(fnType.getInputs())) {
      auto convertedType = getTypeConverter()->convertType(argType.value());
      if (!convertedType)
        return rewriter.notifyMatchFailure(funcOp,
                                           "argument type conversion failed");
      signatureConverter.addInputs(argType.index(), convertedType);
    }

    SmallVector<Type> convertedResultTypes;
    for (Type t : fnType.getResults()) {
      Type resultType = getTypeConverter()->convertType(t);
      if (!resultType)
        return rewriter.notifyMatchFailure(funcOp,
                                           "result type conversion failed");
      if (isa<emitc::ArrayType>(resultType))
        return rewriter.notifyMatchFailure(
            funcOp, "functions returning arrays are not supported");
      convertedResultTypes.push_back(resultType);
    }

    Type resultType;
    if (fnType.getNumResults() == 1) {
      resultType = convertedResultTypes[0];
    } else if (fnType.getNumResults() > 1) {
      auto structTypeOrErr = getOrCreateMultiReturnType(
          rewriter, funcOp.getLoc(), funcOp, convertedResultTypes);
      if (failed(structTypeOrErr))
        return rewriter.notifyMatchFailure(funcOp,
                                           "incompatible multi-return struct");
      resultType = *structTypeOrErr;
    }

    // Create the converted `emitc.func` op.
    emitc::FuncOp newFuncOp = emitc::FuncOp::create(
        rewriter, funcOp.getLoc(), funcOp.getName(),
        FunctionType::get(rewriter.getContext(),
                          signatureConverter.getConvertedTypes(),
                          resultType ? TypeRange(resultType) : TypeRange()));

    // Copy over all attributes other than the function name and type.
    for (const auto &namedAttr : funcOp->getAttrs()) {
      if (namedAttr.getName() != funcOp.getFunctionTypeAttrName() &&
          namedAttr.getName() != SymbolTable::getSymbolAttrName())
        newFuncOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    // Add `extern` to specifiers if `func.func` is declaration only.
    if (funcOp.isDeclaration()) {
      ArrayAttr specifiers = rewriter.getStrArrayAttr({"extern"});
      newFuncOp.setSpecifiersAttr(specifiers);
    }

    // Add `static` to specifiers if `func.func` is private but not a
    // declaration.
    if (funcOp.isPrivate() && !funcOp.isDeclaration()) {
      ArrayAttr specifiers = rewriter.getStrArrayAttr({"static"});
      newFuncOp.setSpecifiersAttr(specifiers);
    }

    if (!funcOp.isDeclaration()) {
      rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                  newFuncOp.end());
      if (failed(rewriter.convertRegionTypes(
              &newFuncOp.getBody(), *getTypeConverter(), &signatureConverter)))
        return failure();
    }
    rewriter.eraseOp(funcOp);

    return success();
  }

private:
  bool lowerToCpp;
};

class ReturnOpConversion final : public OpConversionPattern<func::ReturnOp> {
public:
  ReturnOpConversion(const TypeConverter &typeConverter, MLIRContext *ctx,
                     bool lowerToCpp)
      : OpConversionPattern<func::ReturnOp>(typeConverter, ctx),
        lowerToCpp(lowerToCpp) {}

  LogicalResult
  matchAndRewrite(func::ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Do not convert multiple-return functions if lowering target is Cpp.
    // The translator will emit the return values as an std::tuple.
    if (returnOp.getNumOperands() > 1 && lowerToCpp)
      return rewriter.notifyMatchFailure(
          returnOp, "only zero or one operand is supported");

    if (llvm::any_of(adaptor.getOperands(), [](Value operand) {
          return isa<emitc::ArrayType>(operand.getType());
        }))
      return rewriter.notifyMatchFailure(returnOp,
                                         "returning arrays is not supported");

    if (returnOp.getNumOperands() <= 1) {
      rewriter.replaceOpWithNewOp<emitc::ReturnOp>(
          returnOp,
          returnOp.getNumOperands() ? adaptor.getOperands()[0] : nullptr);
      return success();
    }

    // Multi-operand return: pack values into a struct.
    Location loc = returnOp.getLoc();

    auto structType = getOrCreateMultiReturnType(rewriter, loc, returnOp,
                                                 adaptor.getOperands());
    if (failed(structType))
      return rewriter.notifyMatchFailure(returnOp,
                                         "incompatible multi-return struct");

    Value structVal =
        packValuesIntoStruct(rewriter, loc, adaptor.getOperands(), *structType);
    rewriter.replaceOpWithNewOp<emitc::ReturnOp>(returnOp, structVal);
    return success();
  }

private:
  bool lowerToCpp;
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::populateFuncToEmitCPatterns(const TypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       bool lowerToCpp) {
  MLIRContext *ctx = patterns.getContext();

  patterns.add<CallOpConversion, FuncOpConversion, ReturnOpConversion>(
      typeConverter, ctx, lowerToCpp);
}
