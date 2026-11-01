//===- OpenACCExtensionsTest.cpp - Extension tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/OpenACCToLLVM/ACCToLLVM.h"
#include "mlir/Conversion/OpenACCToSCF/ConvertOpenACCToSCF.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/Extensions/AllExtensions.h"
#include "mlir/Dialect/OpenACC/Extensions/MemRefInterfaces.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "gtest/gtest.h"

using namespace mlir;

TEST(OpenACCExtensions, PassRegistersStandardModels) {
  DialectRegistry registry;
  acc::createOffloadTargetVerifier()->getDependentDialects(registry);
  MLIRContext context(registry);
  context.loadDialect<acc::OpenACCDialect, gpu::GPUDialect>();
  EXPECT_TRUE(OperationName("gpu.launch", &context)
                  .hasInterface<acc::OffloadRegionOpInterface>());
  EXPECT_TRUE(isa<acc::PointerLikeType>(
      MemRefType::get({4}, IntegerType::get(&context, 32))));
}

TEST(OpenACCExtensions, ConversionPassRegistersStandardModels) {
  for (auto createPass :
       {createConvertACCToLLVMPass, createConvertOpenACCToSCFPass}) {
    auto pass = createPass();
    SCOPED_TRACE(pass->getName().str());
    DialectRegistry registry;
    pass->getDependentDialects(registry);
    MLIRContext context(registry);
    context.loadDialect<acc::OpenACCDialect, LLVM::LLVMDialect>();
    EXPECT_TRUE(isa<acc::PointerLikeType>(
        MemRefType::get({4}, IntegerType::get(&context, 32))));
    EXPECT_TRUE(
        isa<acc::PointerLikeType>(LLVM::LLVMPointerType::get(&context)));
    EXPECT_TRUE(OperationName("llvm.mlir.addressof", &context)
                    .hasInterface<acc::AddressOfGlobalOpInterface>());
  }
}

TEST(OpenACCExtensions, MemRefTypesWithoutMemRefOperations) {
  DialectRegistry registry;
  acc::registerMemRefInterfaces(registry);
  MLIRContext context(registry);
  context.loadDialect<acc::OpenACCDialect>();
  auto type = MemRefType::get({4}, IntegerType::get(&context, 32));
  EXPECT_TRUE(isa<acc::PointerLikeType>(type));
  EXPECT_EQ(context.getLoadedDialect("gpu"), nullptr);
}

TEST(OpenACCExtensions, EitherLoadingOrder) {
  for (bool accFirst : {false, true}) {
    DialectRegistry registry;
    acc::registerAllExtensions(registry);
    MLIRContext context(registry);
    EXPECT_EQ(context.getLoadedDialect("gpu"), nullptr);
    if (accFirst) {
      context.loadDialect<acc::OpenACCDialect>();
      EXPECT_EQ(context.getLoadedDialect("gpu"), nullptr);
      context.loadDialect<gpu::GPUDialect, LLVM::LLVMDialect,
                          memref::MemRefDialect>();
    } else {
      context.loadDialect<gpu::GPUDialect, LLVM::LLVMDialect,
                          memref::MemRefDialect>();
      context.loadDialect<acc::OpenACCDialect>();
    }

    auto space =
        gpu::AddressSpaceAttr::get(&context, gpu::AddressSpace::Workgroup);
    EXPECT_EQ(space.getDialect().getNamespace(), "gpu");
    EXPECT_EQ(gpu::AsyncTokenType::get(&context).getDialect().getNamespace(),
              "gpu");
    auto type = MemRefType::get({4}, IntegerType::get(&context, 32),
                                MemRefLayoutAttrInterface{}, space);
    auto pointer = dyn_cast<acc::PointerLikeType>(type);
    ASSERT_TRUE(pointer);
    EXPECT_TRUE(pointer.isDeviceAccessible({}, nullptr));
    EXPECT_TRUE(
        isa<acc::PointerLikeType>(LLVM::LLVMPointerType::get(&context)));

    OperationName launch("gpu.launch", &context);
    EXPECT_TRUE(launch.hasInterface<acc::OffloadRegionOpInterface>());
    EXPECT_TRUE(OperationName("memref.global", &context)
                    .hasInterface<acc::GlobalVariableOpInterface>());
    EXPECT_TRUE(OperationName("llvm.mlir.addressof", &context)
                    .hasInterface<acc::AddressOfGlobalOpInterface>());
  }
}

TEST(OpenACCExtensions, RegistrationAfterLoading) {
  MLIRContext context;
  context.loadDialect<acc::OpenACCDialect, gpu::GPUDialect,
                      memref::MemRefDialect, LLVM::LLVMDialect>();
  EXPECT_FALSE(OperationName("gpu.launch", &context)
                   .hasInterface<acc::OffloadRegionOpInterface>());
  DialectRegistry registry;
  acc::registerAllExtensions(registry);
  context.appendDialectRegistry(registry);
  EXPECT_TRUE(OperationName("gpu.launch", &context)
                  .hasInterface<acc::OffloadRegionOpInterface>());
  EXPECT_TRUE(isa<acc::PointerLikeType>(
      MemRefType::get({4}, IntegerType::get(&context, 32))));
}
