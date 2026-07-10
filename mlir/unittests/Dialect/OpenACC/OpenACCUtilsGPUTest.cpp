//===- OpenACCUtilsGPUTest.cpp - Unit tests for OpenACC GPU utilities ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/OpenACCUtilsGPU.h"
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
    context.loadDialect<gpu::GPUDialect>();
  }

  MLIRContext context;
  OpBuilder b;
  Location loc;
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
