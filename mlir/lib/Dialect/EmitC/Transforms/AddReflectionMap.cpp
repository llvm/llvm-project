//===- AddReflectionMap.cpp - Add a reflection map to a class --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Dialect/EmitC/Transforms/Transforms.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace emitc;

namespace mlir {
namespace emitc {
#define GEN_PASS_DEF_ADDREFLECTIONMAPPASS
#include "mlir/Dialect/EmitC/Transforms/Passes.h.inc"

namespace {
constexpr const char *mapLibraryHeader = "map";
constexpr const char *stringLibraryHeader = "string";

IncludeOp addHeader(OpBuilder &builder, ModuleOp module, StringRef headerName) {
  StringAttr includeAttr = builder.getStringAttr(headerName);
  return IncludeOp::create(
      builder, module.getLoc(), includeAttr,
      /*is_standard_include=*/builder.getUnitAttr());
}

class AddReflectionMapPass
    : public impl::AddReflectionMapPassBase<AddReflectionMapPass> {
  using AddReflectionMapPassBase::AddReflectionMapPassBase;
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    populateAddReflectionMapPatterns(patterns, fieldAttrName, excludedFieldAttrs);

    walkAndApplyPatterns(moduleOp, std::move(patterns));
    bool hasMapHdr = false;
    bool hasStringHdr = false;
    for (auto &op : *moduleOp.getBody()) {
      IncludeOp includeOp = llvm::dyn_cast<IncludeOp>(op);
      if (!includeOp)
        continue;

      if (includeOp.getIsStandardInclude()) {
        auto include = includeOp.getInclude();
        
        hasMapHdr = include == mapLibraryHeader;
        hasStringHdr = include == stringLibraryHeader;
      }

      if (hasMapHdr && hasStringHdr)
        return;
    }

    mlir::OpBuilder builder(moduleOp.getBody(), moduleOp.getBody()->begin());
    if (!hasMapHdr)
      addHeader(builder, moduleOp, mapLibraryHeader);

    if (!hasStringHdr)
      addHeader(builder, moduleOp, stringLibraryHeader);
  }
};

} // namespace
} // namespace emitc
} // namespace mlir

class AddReflectionMapClass : public OpRewritePattern<ClassOp> {
public:
  AddReflectionMapClass(MLIRContext *context, StringRef attrName,
                        llvm::ArrayRef<std::string> excludedFieldAttrs)
      : OpRewritePattern<ClassOp>(context), fieldAttrName(attrName),
        excludedFieldAttrs(excludedFieldAttrs.begin(), excludedFieldAttrs.end()) {}

  LogicalResult matchAndRewrite(ClassOp classOp,
                                PatternRewriter &rewriter) const override {
    MLIRContext *context = rewriter.getContext();

    emitc::OpaqueType mapType = mlir::emitc::OpaqueType::get(
        context, "const std::map<std::string, char*>");

    // Collect all field names
    std::vector<std::pair<std::string, std::string>> fieldNames;
    classOp.walk([&](FieldOp fieldOp) {
      if (Attribute attr = fieldOp->getAttrDictionary().get(fieldAttrName)) {
        if (ArrayAttr arrayAttr = dyn_cast<mlir::ArrayAttr>(attr)) {
          StringAttr stringAttr = cast<mlir::StringAttr>(arrayAttr[0]);
          fieldNames.emplace_back(stringAttr.getValue().str(),
                                  fieldOp.getName().str());
          return;
        }
      }

      bool shouldIgnore = false;
      for (const std::string &ignoreAttr : excludedFieldAttrs) {
        if (fieldOp->hasAttr(ignoreAttr)) {
          shouldIgnore = true;
          break;
        }
      }

      if (shouldIgnore)
        return;

      fieldOp.emitError()
          << "FieldOp must have a dictionary attribute named '"
          << fieldAttrName << "'"
          << "with an array containing a string attribute";
    });

    std::string reflectionMapContents;
    reflectionMapContents += "{ ";
    for (size_t i = 0, numFields = fieldNames.size(); i < numFields; ++i) {
      reflectionMapContents += llvm::formatv(
          "{ \"{0}\", reinterpret_cast<char*>(&{1}) }{2}", fieldNames[i].first,
          fieldNames[i].second, (i < numFields - 1) ? ", " : "");
    }
    reflectionMapContents += " }";

    if (FuncOp executeFunc =
            classOp.lookupSymbol<FuncOp>("operator()"))
      rewriter.setInsertionPoint(executeFunc);
    else {
      classOp.emitError() << "ClassOp must contain a function named 'operator()' "
                             "to add reflection map";
      return failure();
    }

    FieldOp reflectionMapField = FieldOp::create(
        rewriter, classOp.getLoc(), rewriter.getStringAttr("reflectionMap"),
        TypeAttr::get(mapType), emitc::OpaqueAttr::get(context, reflectionMapContents));

    // Create getBufferForName method
    emitc::OpaqueType nameType = emitc::OpaqueType::get(rewriter.getContext(), "std::string");
    emitc::OpaqueType charType = emitc::OpaqueType::get(rewriter.getContext(), "char");
    emitc::PointerType valType = emitc::PointerType::get(rewriter.getContext(), charType);
    FuncOp getBufferForNameFunc = FuncOp::create(
        rewriter, reflectionMapField->getLoc(), "getBufferForName",
        FunctionType::get(rewriter.getContext(), {nameType}, {valType}));

    Block *body = rewriter.createBlock(&getBufferForNameFunc.getBody(), {}, {nameType}, {reflectionMapField->getLoc()});
    rewriter.setInsertionPointToStart(body);
    GetFieldOp mapField = GetFieldOp::create(
        rewriter, reflectionMapField->getLoc(), mapType, "reflectionMap");
    Value nameArg = body->getArgument(0);
    MemberCallOpaqueOp lookupCall = MemberCallOpaqueOp::create(
        rewriter, reflectionMapField->getLoc(), valType, mapField.getResult(),
        "at", ArrayAttr{}, ArrayAttr{}, ValueRange{nameArg});
    ReturnOp::create(rewriter, reflectionMapField->getLoc(), lookupCall.getResult(0));

    return success();
  }

private:
  /// The name of the attribute on FieldOps that contains the field name
  /// metadata for the reflection map.
  StringRef fieldAttrName;

  /// Attributes that, if present on a field, exclude it from the
  /// reflection map.
  llvm::SmallVector<std::string> excludedFieldAttrs;
};

void mlir::emitc::populateAddReflectionMapPatterns(
    RewritePatternSet &patterns, StringRef fieldAttrName,
    llvm::ArrayRef<std::string> excludedFieldAttrs) {
  patterns.add<AddReflectionMapClass>(patterns.getContext(), fieldAttrName,
                                      excludedFieldAttrs);
}