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
constexpr const char *kMapLibraryHeader = "map";
constexpr const char *kStringLibraryHeader = "string";
class AddReflectionMapPass
    : public impl::AddReflectionMapPassBase<AddReflectionMapPass> {
  using AddReflectionMapPassBase::AddReflectionMapPassBase;
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    RewritePatternSet patterns(&getContext());
    populateAddReflectionMapPatterns(patterns, namedAttribute);

    walkAndApplyPatterns(module, std::move(patterns));
    bool hasMap = false;
    bool hasString = false;
    for (auto &op : *module.getBody()) {
      emitc::IncludeOp includeOp = llvm::dyn_cast<mlir::emitc::IncludeOp>(op);
      if (!includeOp)
        continue;
      if (includeOp.getIsStandardInclude()) {
        if (includeOp.getInclude() == kMapLibraryHeader)
          hasMap = true;
        if (includeOp.getInclude() == kStringLibraryHeader)
          hasString = true;
      }
    }

    if (hasMap && hasString)
      return;

    mlir::OpBuilder builder(module.getBody(), module.getBody()->begin());
    if (!hasMap) {
      StringAttr includeAttr = builder.getStringAttr(kMapLibraryHeader);
      builder.create<mlir::emitc::IncludeOp>(
          module.getLoc(), includeAttr,
          /*is_standard_include=*/builder.getUnitAttr());
    }
    if (!hasString) {
      StringAttr includeAttr = builder.getStringAttr(kStringLibraryHeader);
      builder.create<emitc::IncludeOp>(
          module.getLoc(), includeAttr,
          /*is_standard_include=*/builder.getUnitAttr());
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
    mlir::MLIRContext *context = rewriter.getContext();
    emitc::OpaqueType stringViewType =
        mlir::emitc::OpaqueType::get(rewriter.getContext(), "std::string_view");
    emitc::OpaqueType charType =
        mlir::emitc::OpaqueType::get(rewriter.getContext(), "char");
    emitc::OpaqueType mapType = mlir::emitc::OpaqueType::get(
        rewriter.getContext(), "const std::map<std::string, char*>");

    FunctionType funcType =
        rewriter.getFunctionType({stringViewType}, {charType});
    emitc::FuncOp executeFunc =
        classOp.lookupSymbol<mlir::emitc::FuncOp>("execute");
    if (executeFunc)
      rewriter.setInsertionPoint(executeFunc);
    else
      classOp.emitError() << "ClassOp must contain a function named 'execute' "
                             "to add reflection map";

    emitc::FuncOp getBufferFunc = rewriter.create<mlir::emitc::FuncOp>(
        classOp.getLoc(), "getBufferForName", funcType);

    Block *funcBody = getBufferFunc.addEntryBlock();
    rewriter.setInsertionPointToStart(funcBody);

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

    std::string mapInitializer = "{ ";
    for (size_t i = 0; i < fieldNames.size(); ++i) {
      mapInitializer += " { \"" + fieldNames[i].first + "\", " +
                        "reinterpret_cast<char*>(&" + fieldNames[i].second +
                        ")",
          mapInitializer += " }";
      if (i < fieldNames.size() - 1)
        mapInitializer += ", ";
    }
    mapInitializer += " }";

    emitc::OpaqueType iteratorType = mlir::emitc::OpaqueType::get(
        context, "std::map<std::string, char*>::const_iterator");

    emitc::ConstantOp bufferMap = rewriter.create<emitc::ConstantOp>(
        classOp.getLoc(), mapType,
        emitc::OpaqueAttr::get(context, mapInitializer));

    mlir::Value nameArg = getBufferFunc.getArgument(0);
    emitc::CallOpaqueOp it = rewriter.create<emitc::CallOpaqueOp>(
        classOp.getLoc(), iteratorType, rewriter.getStringAttr("find"),
        mlir::ValueRange{bufferMap.getResult(), nameArg});
    emitc::CallOpaqueOp endIt = rewriter.create<emitc::CallOpaqueOp>(
        classOp.getLoc(), iteratorType, rewriter.getStringAttr("end"),
        bufferMap.getResult());
    emitc::CallOpaqueOp isEnd = rewriter.create<emitc::CallOpaqueOp>(
        classOp.getLoc(), rewriter.getI1Type(),
        "operator==", mlir::ValueRange{it.getResult(0), endIt.getResult(0)});
    emitc::ConstantOp nullPtr = rewriter.create<emitc::ConstantOp>(
        classOp.getLoc(), charType, emitc::OpaqueAttr::get(context, "nullptr"));
    emitc::CallOpaqueOp second = rewriter.create<emitc::CallOpaqueOp>(
        classOp.getLoc(), charType, "second", it.getResult(0));

    emitc::ConditionalOp result = rewriter.create<emitc::ConditionalOp>(
        classOp.getLoc(), charType, isEnd.getResult(0), nullPtr.getResult(),
        second.getResult(0));

    rewriter.create<emitc::ReturnOp>(classOp.getLoc(), result.getResult());

    return success();
  }

private:
  StringRef attributeName;
};

void mlir::emitc::populateAddReflectionMapPatterns(RewritePatternSet &patterns,
                                                   StringRef namedAttribute) {
  patterns.add<AddReflectionMapClass>(patterns.getContext(), namedAttribute);
}
