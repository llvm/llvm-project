//===- OpenACCStandaloneTest.cpp - Standalone tests -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace mlir;

// This executable deliberately links MLIROpenACCDialect without the GPU
// operation implementation or the standard OpenACC external models.
TEST(OpenACCStandalone, LoadingDoesNotLoadOtherDialects) {
  MLIRContext context;
  context.loadDialect<acc::OpenACCDialect>();
  EXPECT_EQ(context.getLoadedDialect("gpu"), nullptr);
  EXPECT_EQ(context.getLoadedDialect("llvm"), nullptr);
  EXPECT_EQ(context.getLoadedDialect("memref"), nullptr);
  EXPECT_TRUE(OperationName("acc.parallel", &context).isRegistered());
  EXPECT_TRUE(OperationName("acc.compute_region", &context).isRegistered());
  EXPECT_TRUE(OperationName("acc.gpu_shared_memory", &context).isRegistered());
  EXPECT_FALSE(OperationName("gpu.launch", &context).isRegistered());

  auto dim = acc::GPUParallelDimAttr::get(&context, gpu::Processor::ThreadY);
  EXPECT_EQ(dim.getProcessor(), gpu::Processor::ThreadY);
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << dim;
  EXPECT_EQ(text, "#acc.par_dim<thread_y>");
  EXPECT_EQ(context.getLoadedDialect("gpu"), nullptr);
}
