//===- OpenACCUtilsGPUTest.cpp - Unit tests for OpenACC GPU utilities ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/OpenACCUtilsGPU.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::acc;

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class OpenACCUtilsGPUTest : public ::testing::Test {
protected:
  OpenACCUtilsGPUTest() : b(&context), loc(UnknownLoc::get(&context)) {
    context.loadDialect<arith::ArithDialect, gpu::GPUDialect>();
  }

  void SetUp() override {
    module = ModuleOp::create(b, loc);
    b.setInsertionPointToStart(module->getBody());
  }

  MLIRContext context;
  OpBuilder b;
  Location loc;
  OwningOpRef<ModuleOp> module;
};

//===----------------------------------------------------------------------===//
// getOrCreateGPUModule Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsGPUTest, getOrCreateGPUModuleCreatesWhenMissing) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);

  // First call should create the GPU module
  auto gpuMod = getOrCreateGPUModule(*module, /*create=*/true);
  ASSERT_TRUE(gpuMod.has_value());
  EXPECT_EQ(gpuMod->getName(), kDefaultGPUModuleName);

  // Module should now have the container module attribute
  EXPECT_TRUE(
      (*module)->hasAttr(gpu::GPUDialect::getContainerModuleAttrName()));
}

TEST_F(OpenACCUtilsGPUTest, getOrCreateGPUModuleReturnsExisting) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);

  // Create a GPU module first
  auto gpuMod1 = getOrCreateGPUModule(*module, /*create=*/true);
  ASSERT_TRUE(gpuMod1.has_value());

  // Second call should return the same module
  auto gpuMod2 = getOrCreateGPUModule(*module, /*create=*/true);
  ASSERT_TRUE(gpuMod2.has_value());
  EXPECT_EQ(gpuMod1->getOperation(), gpuMod2->getOperation());
}

TEST_F(OpenACCUtilsGPUTest, getOrCreateGPUModuleNoCreateReturnsNullopt) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);

  // With create=false and no existing GPU module, should return nullopt
  auto gpuMod = getOrCreateGPUModule(*module, /*create=*/false);
  EXPECT_FALSE(gpuMod.has_value());
}

TEST_F(OpenACCUtilsGPUTest, getOrCreateGPUModuleCustomName) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);

  // Create with custom name
  auto gpuMod =
      getOrCreateGPUModule(*module, /*create=*/true, "custom_gpu_module");
  ASSERT_TRUE(gpuMod.has_value());
  EXPECT_EQ(gpuMod->getName(), "custom_gpu_module");
}

TEST_F(OpenACCUtilsGPUTest, getOrCreateGPUModuleEmptyNameUsesDefault) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);

  // Empty name should use default
  auto gpuMod = getOrCreateGPUModule(*module, /*create=*/true, "");
  ASSERT_TRUE(gpuMod.has_value());
  EXPECT_EQ(gpuMod->getName(), kDefaultGPUModuleName);
}

//===----------------------------------------------------------------------===//
// getGPUSize / getGPUThreadId Tests
//===----------------------------------------------------------------------===//

TEST_F(OpenACCUtilsGPUTest, getGPUSizeFromLaunch) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  b.setInsertionPointToStart(module->getBody());

  auto c128 = arith::ConstantIndexOp::create(b, loc, 128);
  auto c4 = arith::ConstantIndexOp::create(b, loc, 4);
  auto c1 = arith::ConstantIndexOp::create(b, loc, 1);
  auto launch = gpu::LaunchOp::create(b, loc, c4, c1, c1, c128, c1, c1);
  llvm::DenseMap<gpu::Processor, Value> dimensionOps;

  // getGPUSize returns the in-kernel size SSA values (block arguments), which
  // are the values exposed by getBlockSize()/getGridSize().
  EXPECT_EQ(getGPUSize(gpu::Processor::ThreadX, launch, dimensionOps),
            launch.getBlockSize().x);
  EXPECT_EQ(getGPUSize(gpu::Processor::BlockX, launch, dimensionOps),
            launch.getGridSize().x);
}

TEST_F(OpenACCUtilsGPUTest, getGPUSizeFromDimensionMap) {
  auto c128 = arith::ConstantIndexOp::create(b, loc, 128);
  llvm::DenseMap<gpu::Processor, Value> dimensionOps;
  dimensionOps[gpu::Processor::ThreadX] = c128.getResult();

  EXPECT_EQ(getGPUSize(gpu::Processor::ThreadX, nullptr, dimensionOps),
            c128.getResult());
}

TEST_F(OpenACCUtilsGPUTest, getGPUThreadIdFromLaunch) {
  OwningOpRef<ModuleOp> module = ModuleOp::create(b, loc);
  b.setInsertionPointToStart(module->getBody());

  auto c128 = arith::ConstantIndexOp::create(b, loc, 128);
  auto c4 = arith::ConstantIndexOp::create(b, loc, 4);
  auto c1 = arith::ConstantIndexOp::create(b, loc, 1);
  auto launch = gpu::LaunchOp::create(b, loc, c4, c1, c1, c128, c1, c1);
  llvm::DenseMap<gpu::Processor, Value> indexOps;

  EXPECT_EQ(getGPUThreadId(gpu::Processor::ThreadX, launch, indexOps),
            launch.getThreadIds().x);
  EXPECT_EQ(getGPUThreadId(gpu::Processor::BlockX, launch, indexOps),
            launch.getBlockIds().x);
}

TEST_F(OpenACCUtilsGPUTest, getGPUThreadIdFromIndexMap) {
  auto c7 = arith::ConstantIndexOp::create(b, loc, 7);
  llvm::DenseMap<gpu::Processor, Value> indexOps;
  indexOps[gpu::Processor::BlockX] = c7.getResult();

  EXPECT_EQ(getGPUThreadId(gpu::Processor::BlockX, nullptr, indexOps),
            c7.getResult());
}
