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
    if (!hasMapHdr) {
      addHeader(builder, module, mapLibraryHeader);
    }
    if (!hasStringHdr) {
      addHeader(builder, module, stringLibraryHeader);
    }
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
      if (mlir::Attribute attrsAttr =
              fieldOp->getAttrDictionary().get("attrs")) {
        if (DictionaryAttr innerDictAttr =
                dyn_cast<mlir::DictionaryAttr>(attrsAttr)) {
          ArrayAttr arrayAttr = dyn_cast<mlir::ArrayAttr>(
              innerDictAttr.getNamed(attributeName)->getValue());
          if (!arrayAttr.empty()) {
            StringAttr stringAttr = dyn_cast<mlir::StringAttr>(arrayAttr[0]);
            std::string indexPath = stringAttr.getValue().str();
            fieldNames.emplace_back(indexPath, fieldOp.getName().str());
          }
          if (arrayAttr.size() > 1) {
            fieldOp.emitError() << attributeName
                                << " attribute must "
                                   "contain at most one value, but found "
                                << arrayAttr.size() << " values.";
            return;
          }
        }
      }
    });

    // Construct the map initializer string
    std::string mapInitializer = "{ ";
    for (size_t i = 0; i < fieldNames.size(); ++i) {
      mapInitializer += " { \"" + fieldNames[i].first + "\", " +
                        "reinterpret_cast<char*>(&" + fieldNames[i].second +
                        ")";
      mapInitializer += " }";
      if (i < fieldNames.size() - 1)
        mapInitializer += ", ";
    }
    mapInitializer += " }";

    emitc::OpaqueType returnType = mlir::emitc::OpaqueType::get(
        context, "const std::map<std::string, char*>");

    emitc::FuncOp executeFunc =
        classOp.lookupSymbol<mlir::emitc::FuncOp>("execute");
    if (executeFunc)
      rewriter.setInsertionPoint(executeFunc);
    else
      classOp.emitError() << "ClassOp must contain a function named 'execute' "
                             "to add reflection map";

    // Create the getFeatures function
    emitc::FuncOp getFeaturesFunc = rewriter.create<mlir::emitc::FuncOp>(
        classOp.getLoc(), "getFeatures",
        rewriter.getFunctionType({}, {returnType}));

    // Add the body of the getFeatures function
    Block *funcBody = getFeaturesFunc.addEntryBlock();
    rewriter.setInsertionPointToStart(funcBody);

    // Create the constant map
    emitc::ConstantOp bufferMap = rewriter.create<emitc::ConstantOp>(
        classOp.getLoc(), mapType,
        emitc::OpaqueAttr::get(context, mapInitializer));

    rewriter.create<mlir::emitc::ReturnOp>(classOp.getLoc(),
                                           bufferMap.getResult());

    return success();
  }

private:
  StringRef attributeName;
};

void mlir::emitc::populateAddReflectionMapPatterns(RewritePatternSet &patterns,
                                                   StringRef namedAttribute) {
  patterns.add<AddReflectionMapClass>(patterns.getContext(), namedAttribute);
}
