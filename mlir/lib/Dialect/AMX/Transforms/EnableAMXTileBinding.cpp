//===- EnableAMXTileBinding.cpp - Enable tile binding for Intel AMX -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass enables the tile register binding semantic for Intel® Advanced
// Matrix Extensions (Intel® AMX). Intuitively, this pass analyses the tile
// binding hints set by users, legalize the hints and automatically configures
// needed hardware context. The AMX tile register usage in lowered intrinsics
// would strictly respect the given hints, enforced in lowering pass
// `--convert-vector-to-llvm`.
//
// Note that if this pass is not invoked prior to `--convert-vector-to-llvm`,
// the AMX lowering would ignore the binding info and fallback to original
// scheme.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMX/AMXDialect.h"
#include "mlir/Dialect/AMX/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#define DEBUG_TYPE "enable-amx-tile-binding"

namespace mlir {
namespace amx {

#define GEN_PASS_DEF_ENABLEAMXTILEBINDING
#include "mlir/Dialect/AMX/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//

/// A class for analyzing tile register binding for each tile vector.
class TileBindingAnalysis {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileBindingAnalysis)
  explicit TileBindingAnalysis(Operation *);
};

TileBindingAnalysis::TileBindingAnalysis(Operation *root) {}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct EnableAMXTileBindingPass
    : public impl::EnableAMXTileBindingBase<EnableAMXTileBindingPass> {
  void runOnOperation() override {
    // 0. Get AnalyseInfo for each concerned Value (mixed used of tmul & normal
    // vector operations?)
    TileBindingAnalysis &analysis = getAnalysis<TileBindingAnalysis>();

    // 1. Propagate binding info to AMX Ops
    //
    // 2. Analyse tile scopes & expand them maximally
    //
    // 3. insert tile config/release according to tile scopes
  }
};

} // namespace amx
} // namespace mlir
