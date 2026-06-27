//===- WrapFuncInClass.cpp - Wrap Emitc Funcs in classes -------------===//
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
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace emitc;

namespace mlir {
namespace emitc {
#define GEN_PASS_DEF_WRAPFUNCINCLASSPASS
#include "mlir/Dialect/EmitC/Transforms/Passes.h.inc"

namespace {
struct WrapFuncInClassPass
    : public impl::WrapFuncInClassPassBase<WrapFuncInClassPass> {
  using WrapFuncInClassPassBase::WrapFuncInClassPassBase;
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();

    DenseMap<FuncOp, llvm::DenseSet<GlobalOp>> globalsUsedByFuncs;

    SymbolTableCollection symbolTable;
    moduleOp.walk([&globalsUsedByFuncs, &symbolTable](FuncOp funcOp) {
      funcOp.walk([&globalsUsedByFuncs, &symbolTable,
                   &funcOp](GetGlobalOp getGlobalOp) {
        if (auto globalOp = symbolTable.lookupNearestSymbolFrom<GlobalOp>(
                getGlobalOp, getGlobalOp.getNameAttr())) {
          globalsUsedByFuncs[funcOp].insert(globalOp);
        }
      });
    });

    RewritePatternSet patterns(&getContext());
    populateWrapFuncInClass(patterns, funcName, globalsUsedByFuncs);

    walkAndApplyPatterns(moduleOp, std::move(patterns));

    DenseSet<GlobalOp> globalsToErase;
    for (auto &[_, globals] : globalsUsedByFuncs)
      globalsToErase.insert_range(globals);

    for (GlobalOp globalOp : globalsToErase)
      globalOp.erase();
  }
};

} // namespace
} // namespace emitc
} // namespace mlir

class WrapFuncInClass : public OpRewritePattern<FuncOp> {
public:
  WrapFuncInClass(
      MLIRContext *context, StringRef funcName,
      const DenseMap<FuncOp, llvm::DenseSet<GlobalOp>> &globalsToMove)
      : OpRewritePattern<FuncOp>(context), funcName(funcName),
        globalsToMove(globalsToMove) {}

  LogicalResult matchAndRewrite(FuncOp funcOp,
                                PatternRewriter &rewriter) const override {

    auto className = funcOp.getSymNameAttr().str() + "Class";
    ClassOp newClassOp = ClassOp::create(rewriter, funcOp.getLoc(), className);

    SmallVector<std::pair<StringAttr, TypeAttr>> fields;
    rewriter.createBlock(&newClassOp.getBody());
    rewriter.setInsertionPointToStart(&newClassOp.getBody().front());

    auto argAttrs = funcOp.getArgAttrs();
    for (auto [idx, val] : llvm::enumerate(funcOp.getArguments())) {
      StringAttr fieldName =
          rewriter.getStringAttr("fieldName" + std::to_string(idx));

      TypeAttr typeAttr = TypeAttr::get(val.getType());
      fields.push_back({fieldName, typeAttr});

      FieldOp fieldop = FieldOp::create(rewriter, funcOp->getLoc(), fieldName,
                                        typeAttr, nullptr);

      if (argAttrs && idx < argAttrs->size()) {
        fieldop->setDiscardableAttrs(funcOp.getArgAttrDict(idx));
      }
    }

    auto globalsIt = globalsToMove.find(funcOp);
    if (globalsIt != globalsToMove.end()) {
      for (auto global : globalsIt->second) {
        FieldOp::create(rewriter, funcOp->getLoc(), global.getSymNameAttr(),
                        global.getTypeAttr(), global.getInitialValueAttr());
      }
    }

    rewriter.setInsertionPointToEnd(&newClassOp.getBody().front());
    FunctionType funcType = funcOp.getFunctionType();
    Location loc = funcOp.getLoc();
    FuncOp newFuncOp = FuncOp::create(rewriter, loc, (funcName), funcType);

    rewriter.createBlock(&newFuncOp.getBody());
    newFuncOp.getBody().takeBody(funcOp.getBody());

    rewriter.setInsertionPointToStart(&newFuncOp.getBody().front());
    std::vector<Value> newArguments;
    newArguments.reserve(fields.size());
    for (auto &[fieldName, attr] : fields) {
      GetFieldOp arg =
          GetFieldOp::create(rewriter, loc, attr.getValue(), fieldName);
      newArguments.push_back(arg);
    }

    for (auto [oldArg, newArg] :
         llvm::zip(newFuncOp.getArguments(), newArguments)) {
      rewriter.replaceAllUsesWith(oldArg, newArg);
    }

    llvm::BitVector argsToErase(newFuncOp.getNumArguments(), true);
    if (failed(newFuncOp.eraseArguments(argsToErase)))
      newFuncOp->emitOpError("failed to erase all arguments using BitVector");

    newFuncOp.walk([&](GetGlobalOp getGlobalOp) {
      rewriter.setInsertionPoint(getGlobalOp);
      GetFieldOp getFieldOp =
          GetFieldOp::create(rewriter, getGlobalOp.getLoc(),
                             getGlobalOp.getType(), getGlobalOp.getNameAttr());
      rewriter.replaceOp(getGlobalOp, getFieldOp);
    });

    rewriter.replaceOp(funcOp, newClassOp);
    return success();
  }

private:
  /// Name of the newly generated member function with body matching the input
  /// function.
  std::string funcName;

  /// Map of FuncOp and the GlobalOps it uses which need to be moved into the
  /// ClassOp wrapper.
  DenseMap<FuncOp, llvm::DenseSet<GlobalOp>> globalsToMove;
};

void mlir::emitc::populateWrapFuncInClass(
    RewritePatternSet &patterns, StringRef funcName,
    DenseMap<FuncOp, DenseSet<GlobalOp>> &globalsToMove) {
  patterns.add<WrapFuncInClass>(patterns.getContext(), funcName, globalsToMove);
}
