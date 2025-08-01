//===- AddReflectionMap.cpp - Add a reflection map to a class -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
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
  return builder.create<emitc::IncludeOp>(
      module.getLoc(), includeAttr,
      /*is_standard_include=*/builder.getUnitAttr());
}

class AddReflectionMapPass
    : public impl::AddReflectionMapPassBase<AddReflectionMapPass> {
  using AddReflectionMapPassBase::AddReflectionMapPassBase;
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    RewritePatternSet patterns(&getContext());
    populateAddReflectionMapPatterns(patterns, namedAttribute);

    walkAndApplyPatterns(module, std::move(patterns));
    bool hasMapHdr = false;
    bool hasStringHdr = false;
    for (auto &op : *module.getBody()) {
      emitc::IncludeOp includeOp = llvm::dyn_cast<mlir::emitc::IncludeOp>(op);
      if (!includeOp)
        continue;
      if (includeOp.getIsStandardInclude()) {
        if (includeOp.getInclude() == mapLibraryHeader)
          hasMapHdr = true;
        if (includeOp.getInclude() == stringLibraryHeader)
          hasStringHdr = true;
      }
      if (hasMapHdr && hasStringHdr)
        return;
    }

    mlir::OpBuilder builder(module.getBody(), module.getBody()->begin());
    if (!hasMapHdr)
      addHeader(builder, module, mapLibraryHeader);

    if (!hasStringHdr)
      addHeader(builder, module, stringLibraryHeader);
  }
};

} // namespace
} // namespace emitc
} // namespace mlir

class AddReflectionMapClass : public OpRewritePattern<emitc::ClassOp> {
public:
  AddReflectionMapClass(MLIRContext *context, StringRef attrName)
      : OpRewritePattern<emitc::ClassOp>(context), attributeName(attrName) {}

  LogicalResult matchAndRewrite(mlir::emitc::ClassOp classOp,
                                PatternRewriter &rewriter) const override {
    MLIRContext *context = rewriter.getContext();

    emitc::OpaqueType mapType = mlir::emitc::OpaqueType::get(
        context, "const std::map<std::string, char*>");

    // Collect all field names
    std::vector<std::pair<std::string, std::string>> fieldNames;
    classOp.walk([&](mlir::emitc::FieldOp fieldOp) {
      if (ArrayAttr arrayAttr = cast<mlir::ArrayAttr>(
              fieldOp->getAttrDictionary().get(attributeName))) {
        StringAttr stringAttr = cast<mlir::StringAttr>(arrayAttr[0]);
        fieldNames.emplace_back(stringAttr.getValue().str(),
                                fieldOp.getName().str());

      } else {
        fieldOp.emitError()
            << "FieldOp must have a dictionary attribute named '"
            << attributeName << "'"
            << "with an array containing a string attribute";
      }
    });

    std::string mapString;
    mapString += "{ ";
    for (size_t i = 0; i < fieldNames.size(); ++i) {
      mapString += llvm::formatv(
          "{ \"{0}\", reinterpret_cast<char*>(&{1}) }{2}", fieldNames[i].first,
          fieldNames[i].second, (i < fieldNames.size() - 1) ? ", " : "");
    }
    mapString += " }";

    if (emitc::FuncOp executeFunc =
            classOp.lookupSymbol<mlir::emitc::FuncOp>("execute"))
      rewriter.setInsertionPoint(executeFunc);
    else {
      classOp.emitError() << "ClassOp must contain a function named 'execute' "
                             "to add reflection map";
      return failure();
    }

    rewriter.create<emitc::FieldOp>(
        classOp.getLoc(), rewriter.getStringAttr("reflectionMap"),
        TypeAttr::get(mapType), emitc::OpaqueAttr::get(context, mapString));
    return success();
  }

private:
  StringRef attributeName;
};

void mlir::emitc::populateAddReflectionMapPatterns(RewritePatternSet &patterns,
                                                   StringRef namedAttribute) {
  patterns.add<AddReflectionMapClass>(patterns.getContext(), namedAttribute);
}
