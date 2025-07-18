/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

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
class AddReflectionMapPass
    : public impl::AddReflectionMapPassBase<AddReflectionMapPass> {
  using AddReflectionMapPassBase::AddReflectionMapPassBase;
  void runOnOperation() override {
    Operation *rootOp = getOperation();

    RewritePatternSet patterns(&getContext());
    populateAddReflectionMapPatterns(patterns);

    walkAndApplyPatterns(rootOp, std::move(patterns));
  }
};

} // namespace
} // namespace emitc
} // namespace mlir

class AddReflectionMapClass : public OpRewritePattern<emitc::ClassOp> {
public:
  AddReflectionMapClass(MLIRContext *context)
      : OpRewritePattern<emitc::ClassOp>(context) {}

  LogicalResult matchAndRewrite(mlir::emitc::ClassOp classOp,
                                PatternRewriter &rewriter) const override {
    mlir::MLIRContext *context = rewriter.getContext();
    emitc::OpaqueType stringViewType =
        mlir::emitc::OpaqueType::get(rewriter.getContext(), "std::string_view");
    emitc::OpaqueType charPtrType =
        mlir::emitc::OpaqueType::get(rewriter.getContext(), "char");
    emitc::OpaqueType mapType = mlir::emitc::OpaqueType::get(
        rewriter.getContext(), "const std::map<std::string, char*>");

    FunctionType funcType =
        rewriter.getFunctionType({stringViewType}, {charPtrType});
    emitc::FuncOp executeFunc =
        classOp.lookupSymbol<mlir::emitc::FuncOp>("execute");
    rewriter.setInsertionPoint(executeFunc);

    emitc::FuncOp getBufferFunc = rewriter.create<mlir::emitc::FuncOp>(
        classOp.getLoc(), "getBufferForName", funcType);

    Block *funcBody = getBufferFunc.addEntryBlock();
    rewriter.setInsertionPointToStart(funcBody);

    // Collect all field names
    SmallVector<std::string> fieldNames;
    classOp.walk([&](mlir::emitc::FieldOp fieldOp) {
      if (mlir::Attribute attrsAttr =
              fieldOp->getAttrDictionary().get("attrs")) {
        if (DictionaryAttr innerDictAttr =
                dyn_cast<mlir::DictionaryAttr>(attrsAttr)) {
          auto indexPathAttr =
              innerDictAttr.getNamed("tf_saved_model.index_path");
          ArrayAttr arrayAttr =
              dyn_cast<mlir::ArrayAttr>(indexPathAttr->getValue());
          if (!arrayAttr.empty()) {
            StringAttr stringAttr = dyn_cast<mlir::StringAttr>(arrayAttr[0]);
            std::string indexPath = stringAttr.getValue().str();
            fieldNames.push_back(indexPath);
          }
          if (arrayAttr.size() > 1) {
            fieldOp.emitError() << "tf_saved_model.index_path attribute must "
                                   "contain at most one value, but found "
                                << arrayAttr.size() << " values.";
            return;
          }
        }
      }
    });

    std::string mapInitializer = "{{";
    for (size_t i = 0; i < fieldNames.size(); ++i) {
      mapInitializer += "\"" + fieldNames[i] + "\", " +
                        "reinterpret_cast<char*>(&" + fieldNames[i] + ")",
          mapInitializer += "}";
      if (i < fieldNames.size() - 1)
        mapInitializer += ", {";
    }
    mapInitializer += "}";

    auto iteratorType = mlir::emitc::OpaqueType::get(
        context, "std::map<std::string, char*>::const_iterator");
    auto boolType = rewriter.getI1Type();
    // 5. Create the constant map
    auto bufferMap = rewriter.create<emitc::ConstantOp>(
        classOp.getLoc(), mapType,
        emitc::OpaqueAttr::get(context, mapInitializer));

    // 6. Get the function argument
    mlir::Value nameArg = getBufferFunc.getArgument(0);

    // 7. Create the find call
    auto it = rewriter.create<emitc::CallOpaqueOp>(
        classOp.getLoc(), iteratorType, rewriter.getStringAttr("find"),
        mlir::ValueRange{bufferMap.getResult(), nameArg});

    // 8. Create the end call
    auto endIt = rewriter.create<emitc::CallOpaqueOp>(
        classOp.getLoc(), iteratorType, rewriter.getStringAttr("end"),
        bufferMap.getResult());

    // 9. Create the operator== call
    auto isEnd = rewriter.create<emitc::CallOpaqueOp>(
        classOp.getLoc(), boolType,
        "operator==", mlir::ValueRange{it.getResult(0), endIt.getResult(0)});

    // 10. Create the nullptr constant
    auto nullPtr = rewriter.create<emitc::ConstantOp>(
        classOp.getLoc(), charPtrType,
        emitc::OpaqueAttr::get(context, "nullptr"));

    // 11. Create the second call
    auto second = rewriter.create<emitc::CallOpaqueOp>(
        classOp.getLoc(), charPtrType, "second", it.getResult(0));

    // 12. Create the conditional
    auto result = rewriter.create<emitc::ConditionalOp>(
        classOp.getLoc(), charPtrType, isEnd.getResult(0), nullPtr.getResult(),
        second.getResult(0));

    // 13. Create return
    rewriter.create<emitc::ReturnOp>(classOp.getLoc(), result.getResult());

    return success();
  }
};

void mlir::emitc::populateAddReflectionMapPatterns(
    RewritePatternSet &patterns) {
  patterns.add<AddReflectionMapClass>(patterns.getContext());
}
