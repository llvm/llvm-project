//===- InferAliasScopeAttrs.cpp - Infer LLVM alias scope attributes -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/Transforms/InferAliasScopeAttrs.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace LLVM {
#define GEN_PASS_DEF_LLVMINFERALIASSCOPEATTRIBUTES
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"
} // namespace LLVM
} // namespace mlir

using namespace mlir;

static Value getBasePtr(Operation *op) {
  // TODO: we need a common interface to get the base ptr.
  if (auto loadOp = dyn_cast<memref::LoadOp>(op))
    return loadOp.getMemRef();

  if (auto storeOp = dyn_cast<memref::StoreOp>(op))
    return storeOp.getMemRef();

  return nullptr;
}

namespace {

struct LLVMInferAliasScopeAttrs
    : public LLVM::impl::LLVMInferAliasScopeAttributesBase<
          LLVMInferAliasScopeAttrs> {
  void runOnOperation() override {
    SmallVector<Operation *> memOps;
    getOperation().walk([&](MemoryEffectOpInterface op) {
      if ((op.hasEffect<MemoryEffects::Read>() ||
           op.hasEffect<MemoryEffects::Write>()) &&
          getBasePtr(op))
        memOps.emplace_back(op);
    });

    if (memOps.empty())
      return markAllAnalysesPreserved();

    auto &aliasAnalysis = getAnalysis<AliasAnalysis>();
    MLIRContext *ctx = &getContext();

    LLVM::AliasScopeDomainAttr domain;
    llvm::SmallDenseMap<Operation *, Attribute> aliasScopes;
    auto getScope = [&](Operation *op) -> LLVM::AliasScopeAttr {
      if (!domain)
        domain = LLVM::AliasScopeDomainAttr::get(ctx);

      auto scope =
          cast_if_present<LLVM::AliasScopeAttr>(aliasScopes.lookup(op));
      if (scope)
        return scope;

      scope = LLVM::AliasScopeAttr::get(domain);
      aliasScopes[op] = scope;
      return scope;
    };

    DenseMap<Operation *, llvm::SmallSetVector<Attribute, 4>> noaliasScopes;

    // TODO: This is quadratic in the number of memOps, can we do better?
    for (Operation *op : memOps) {
      for (Operation *otherOp : memOps) {
        if (op == otherOp)
          continue;

        Value basePtr = getBasePtr(op);
        assert(basePtr && "Expected base ptr");
        Value otherBasePtr = getBasePtr(otherOp);
        assert(otherBasePtr && "Expected base ptr");
        if (!aliasAnalysis.alias(basePtr, otherBasePtr).isNo())
          continue;

        noaliasScopes[op].insert(getScope(otherOp));
      }
    }

    if (noaliasScopes.empty())
      return markAllAnalysesPreserved();

    auto aliasScopesName =
        StringAttr::get(ctx, LLVM::LLVMDialect::getAliasScopesAttrName());
    auto noaliasName =
        StringAttr::get(ctx, LLVM::LLVMDialect::getNoAliasAttrName());

    // We are intentionally using discardable attributes here because those are
    // generally not robust against codegen transformations (e.g. inlining) and
    // this pass is intended to be run just before *-to-llvm conversion.
    for (Operation *op : memOps) {
      if (auto aliasScope = aliasScopes.lookup(op))
        op->setAttr(aliasScopesName, ArrayAttr::get(ctx, {aliasScope}));

      auto it = noaliasScopes.find(op);
      if (it != noaliasScopes.end())
        op->setAttr(noaliasName, ArrayAttr::get(ctx, it->second.getArrayRef()));
    }
  }
};
} // namespace
