//===- ParallelLoopMapper.cpp - Parallel Loop Mapper unit tests -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Transforms/ParallelLoopMapper.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace {

TEST(ParallelLoopMapperTest, TestSetMappingAttrMultipleProcMapping) {
  MLIRContext context;
  Builder b(&context);
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<scf::SCFDialect>();
  context.getOrLoadDialect<gpu::GPUDialect>();

  const char *const code = R"mlir(
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    scf.parallel (%i) = (%c1) to (%c3) step (%c1) {
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(code, &context);
  scf::ParallelOp ploopOp =
      *(*module).getBody()->getOps<scf::ParallelOp>().begin();
  auto BlockXMapping = b.getAttr<gpu::ParallelLoopDimMappingAttr>(
      gpu::Processor::BlockX, b.getDimIdentityMap(), b.getDimIdentityMap());
  SmallVector<gpu::ParallelLoopDimMappingAttr, 1> mapping = {BlockXMapping,
                                                             BlockXMapping};
  EXPECT_FALSE(succeeded(gpu::setMappingAttr(ploopOp, mapping)));
}

} // end anonymous namespace
