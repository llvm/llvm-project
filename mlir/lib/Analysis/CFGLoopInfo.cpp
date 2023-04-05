//===- CFGLoopInfo.cpp - LoopInfo analysis for region bodies --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/CFGLoopInfo.h"

using namespace mlir;

CFGLoop::CFGLoop(mlir::Block *block)
    : llvm::LoopBase<mlir::Block, CFGLoop>(block) {}

CFGLoopInfo::CFGLoopInfo(
    const llvm::DominatorTreeBase<mlir::Block, false> &domTree) {
  analyze(domTree);
}
