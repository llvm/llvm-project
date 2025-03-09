//===- SelectPass.cpp - Select pass code ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// SelectPass dynamically selects pass pipeline to run based on root op
// attribute.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
#define GEN_PASS_DEF_SELECTPASS
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct SelectPass final : public impl::SelectPassBase<SelectPass> {
  using SelectPassBase::SelectPassBase;

  SelectPass(
      std::string name_, std::string selectCondName_,
      ArrayRef<std::pair<StringRef, std::function<void(OpPassManager &)>>>
          populateFuncs) {
    name = std::move(name_);
    selectCondName = std::move(selectCondName_);

    SmallVector<std::string> selectVals;
    SmallVector<std::string> selectPpls;
    selectVals.reserve(populateFuncs.size());
    selectPpls.reserve(populateFuncs.size());
    selectPassManagers.reserve(populateFuncs.size());
    for (auto &&[name, populate] : populateFuncs) {
      selectVals.emplace_back(name);

      auto &pm = selectPassManagers.emplace_back();
      populate(pm);

      llvm::raw_string_ostream os(selectPpls.emplace_back());
      pm.printAsTextualPipeline(os);
    }

    selectValues = selectVals;
    selectPipelines = selectPpls;
  }

  LogicalResult initializeOptions(
      StringRef options,
      function_ref<LogicalResult(const Twine &)> errorHandler) override {
    if (failed(SelectPassBase::initializeOptions(options, errorHandler)))
      return failure();

    if (selectCondName.empty())
      return errorHandler("select-cond-name is empty");

    if (selectValues.size() != selectPipelines.size())
      return errorHandler("values and pipelines size mismatch");

    selectPassManagers.resize(selectPipelines.size());

    for (auto &&[i, pipeline] : llvm::enumerate(selectPipelines)) {
      if (failed(parsePassPipeline(pipeline, selectPassManagers[i])))
        return errorHandler("failed to parse pipeline");
    }

    return success();
  }

  LogicalResult initialize(MLIRContext *context) override {
    condAttrName = StringAttr::get(context, selectCondName);

    selectAttrs.reserve(selectAttrs.size());
    for (StringRef value : selectValues)
      selectAttrs.emplace_back(StringAttr::get(context, value));

    return success();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    for (const OpPassManager &pipeline : selectPassManagers)
      pipeline.getDependentDialects(registry);
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    Attribute condAttrValue = op->getAttr(condAttrName);
    if (!condAttrValue) {
      op->emitError("condition attribute not present: ") << condAttrName;
      return signalPassFailure();
    }

    for (auto &&[value, pm] :
         llvm::zip_equal(selectAttrs, selectPassManagers)) {
      if (value != condAttrValue)
        continue;

      if (failed(runPipeline(pm, op)))
        return signalPassFailure();

      return;
    }

    // TODO: add a default pipeline option.
    op->emitError("unhandled condition value: ") << condAttrValue;
    return signalPassFailure();
  }

protected:
  StringRef getName() const override { return name; }

private:
  StringAttr condAttrName;
  SmallVector<Attribute> selectAttrs;
  SmallVector<OpPassManager> selectPassManagers;
};
} // namespace

std::unique_ptr<Pass> mlir::createSelectPass(
    std::string name, std::string selectCondName,
    ArrayRef<std::pair<StringRef, std::function<void(OpPassManager &)>>>
        populateFuncs) {
  return std::make_unique<SelectPass>(std::move(name),
                                      std::move(selectCondName), populateFuncs);
}
