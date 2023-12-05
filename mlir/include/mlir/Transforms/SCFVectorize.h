//===- SCFVectorize.h - ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_SCFVECTORIZE_H_
#define MLIR_TRANSFORMS_SCFVECTORIZE_H_

#include <memory>
#include <optional>

namespace mlir {
class OpBuilder;
class Pass;
struct LogicalResult;
namespace scf {
class ParallelOp;
}
} // namespace mlir

namespace mlir {
struct SCFVectorizeInfo {
  unsigned dim = 0;
  unsigned factor = 0;
  unsigned count = 0;
  bool masked = false;
};

std::optional<SCFVectorizeInfo> getLoopVectorizeInfo(mlir::scf::ParallelOp loop,
                                                     unsigned dim,
                                                     unsigned vectorBitWidth);

struct SCFVectorizeParams {
  unsigned dim = 0;
  unsigned factor = 0;
  bool masked = false;
};

mlir::LogicalResult vectorizeLoop(mlir::OpBuilder &builder,
                                  mlir::scf::ParallelOp loop,
                                  const SCFVectorizeParams &params);

std::unique_ptr<mlir::Pass> createSCFVectorizePass();
} // namespace mlir

#endif // MLIR_TRANSFORMS_SCFVECTORIZE_H_