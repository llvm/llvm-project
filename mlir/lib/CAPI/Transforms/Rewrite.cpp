//===- Rewrite.cpp - C API for Rewrite Patterns ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Rewrite.h"
#include "mlir-c/Transforms.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

inline mlir::RewritePatternSet &unwrap(MlirRewritePatternSet module) {
  assert(module.ptr && "unexpected null module");
  return *(static_cast<mlir::RewritePatternSet *>(module.ptr));
}

inline MlirRewritePatternSet wrap(mlir::RewritePatternSet *module) {
  return {module};
}

inline mlir::FrozenRewritePatternSet *
unwrap(MlirFrozenRewritePatternSet module) {
  assert(module.ptr && "unexpected null module");
  return static_cast<mlir::FrozenRewritePatternSet *>(module.ptr);
}

inline MlirFrozenRewritePatternSet wrap(mlir::FrozenRewritePatternSet *module) {
  return {module};
}

MlirFrozenRewritePatternSet mlirFreezeRewritePattern(MlirRewritePatternSet op) {
  auto *m = new mlir::FrozenRewritePatternSet(std::move(unwrap(op)));
  op.ptr = nullptr;
  return wrap(m);
}

void mlirFrozenRewritePatternSetDestroy(MlirFrozenRewritePatternSet op) {
  delete unwrap(op);
  op.ptr = nullptr;
}

MlirLogicalResult
mlirApplyPatternsAndFoldGreedily(MlirModule op,
                                 MlirFrozenRewritePatternSet patterns,
                                 MlirGreedyRewriteDriverConfig) {
  return wrap(
      mlir::applyPatternsAndFoldGreedily(unwrap(op), *unwrap(patterns)));
}

#if MLIR_ENABLE_PDL_IN_PATTERNMATCH
inline mlir::PDLPatternModule *unwrap(MlirPDLPatternModule module) {
  assert(module.ptr && "unexpected null module");
  return static_cast<mlir::PDLPatternModule *>(module.ptr);
}

inline MlirPDLPatternModule wrap(mlir::PDLPatternModule *module) {
  return {module};
}

MlirPDLPatternModule mlirPDLPatternModuleFromModule(MlirModule op) {
  return wrap(new mlir::PDLPatternModule(
      mlir::OwningOpRef<mlir::ModuleOp>(unwrap(op))));
}

void mlirPDLPatternModuleDestroy(MlirPDLPatternModule op) {
  delete unwrap(op);
  op.ptr = nullptr;
}

MlirRewritePatternSet
mlirRewritePatternSetFromPDLPatternModule(MlirPDLPatternModule op) {
  auto *m = new mlir::RewritePatternSet(std::move(*unwrap(op)));
  op.ptr = nullptr;
  return wrap(m);
}
#endif // MLIR_ENABLE_PDL_IN_PATTERNMATCH
