//===- OpenACCRuntimeUtilsTest.cpp - OpenACC runtime utility tests --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/OpenACCRuntimeUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

#include <optional>

using namespace mlir;
using namespace mlir::acc;

TEST(OpenACCRuntimeCallConfigTest, UsesDialectDeviceTypeEncodingsByDefault) {
  ACCRuntimeCallConfig config;
  for (uint32_t value = 0; value <= getMaxEnumValForDeviceType(); ++value) {
    std::optional<DeviceType> deviceType = symbolizeDeviceType(value);
    if (deviceType)
      EXPECT_EQ(config.getDeviceTypeRuntimeValue(*deviceType), value);
  }
}

TEST(OpenACCRuntimeCallConfigTest, UsesDialectMapFlagEncodingsByDefault) {
  ACCRuntimeCallConfig config;
  for (unsigned bit = 0; bit != 32; ++bit) {
    uint32_t value = 1u << bit;
    std::optional<MapFlags> flag = symbolizeMapFlags(value);
    if (flag)
      EXPECT_EQ(config.getMapFlagsRuntimeValue(*flag), value);
  }
}

TEST(OpenACCRuntimeCallConfigTest, CombinesConfiguredMapFlagEncodings) {
  ACCRuntimeCallConfig config;
  config.setMapFlagRuntimeValue(MapFlags::to, 0x40);
  config.setMapFlagRuntimeValue(MapFlags::from, 0x100);

  EXPECT_EQ(config.getMapFlagsRuntimeValue(MapFlags::to | MapFlags::from),
            0x140);
}

TEST(OpenACCRuntimeCallConfigTest, PostProcessesMapFlags) {
  ACCRuntimeCallConfig config;
  EXPECT_EQ(config.postProcessMapFlags(nullptr, MapFlags::to), MapFlags::to);

  Operation *seenMapOp = nullptr;
  config.setMapFlagsPostProcessFn([&](Operation *mapOp, MapFlags flags) {
    seenMapOp = mapOp;
    return flags | MapFlags::delete_;
  });

  EXPECT_EQ(config.postProcessMapFlags(nullptr, MapFlags::from),
            MapFlags::from | MapFlags::delete_);
  EXPECT_EQ(seenMapOp, nullptr);
}

TEST(OpenACCRuntimeCallConfigTest, FormatsConfiguredMapFlags) {
  ACCRuntimeCallConfig config;
  config.setMapFlagRuntimeValue(MapFlags::to, 0x40);
  config.setMapFlagRuntimeValue(MapFlags::from, 0x100);

  EXPECT_EQ(config.formatMapFlags(MapFlags::to | MapFlags::from),
            "to,from (320 / 0x140)");
}

TEST(OpenACCDataDescriptorTest, StatesTheKindOfEveryDescriptor) {
  EXPECT_EQ(getDataDescriptorKind(DataDescriptor::AccDataDescGeneric),
            DataDescKind::none);
  EXPECT_EQ(getDataDescriptorKind(DataDescriptor::AccDataDescCFI),
            DataDescKind::cfi);
  EXPECT_EQ(getDataDescriptorKind(DataDescriptor::AccDataDescMemRef),
            DataDescKind::memref);
  EXPECT_EQ(getDataDescriptorKind(DataDescriptor::AccDataDescOpenACC),
            DataDescKind::openacc);
  EXPECT_EQ(getDataDescriptorName(DataDescriptor::AccDataDescMemRef),
            "AccDataDescMemRef");
}

TEST(OpenACCDataDescriptorTest, BuildsTheDeclaredLayouts) {
  MLIRContext ctx;
  ctx.loadDialect<LLVM::LLVMDialect>();
  Type i8Ty = IntegerType::get(&ctx, 8);
  Type i32Ty = IntegerType::get(&ctx, 32);
  Type i64Ty = IntegerType::get(&ctx, 64);
  Type ptrTy = LLVM::LLVMPointerType::get(&ctx);

  EXPECT_EQ(
      getDataDescriptorType(&ctx, DataDescriptor::AccDataDescGeneric).getBody(),
      ArrayRef<Type>({i32Ty, i32Ty}));
  EXPECT_EQ(
      getDataDescriptorType(&ctx, DataDescriptor::AccDataDescCFI).getBody(),
      ArrayRef<Type>({i32Ty, ptrTy}));
  EXPECT_EQ(
      getDataDescriptorType(&ctx, DataDescriptor::AccDataDescMemRef).getBody(),
      ArrayRef<Type>({i32Ty, i8Ty, i64Ty, ptrTy}));

  Type baseTy = getDataDescriptorType(&ctx, DataDescriptor::AccDataDescMemRef);
  EXPECT_EQ(
      getDataDescriptorType(&ctx, DataDescriptor::AccDataDescOpenACC, baseTy)
          .getBody(),
      ArrayRef<Type>({baseTy, i8Ty, i64Ty, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy}));
}

TEST(OpenACCDataDescriptorTest, NumbersFieldsInDeclarationOrder) {
  EXPECT_EQ(getDataDescriptorFieldIndex(AccDataDescMemRefField::Version), 0);
  EXPECT_EQ(getDataDescriptorFieldIndex(AccDataDescMemRefField::Rank), 1);
  EXPECT_EQ(getDataDescriptorFieldIndex(AccDataDescMemRefField::ElementSize),
            2);
  EXPECT_EQ(
      getDataDescriptorFieldIndex(AccDataDescMemRefField::MemRefDescriptor), 3);

  EXPECT_EQ(getDataDescriptorFieldIndex(AccDataDescOpenACCField::Base), 0);
  EXPECT_EQ(getDataDescriptorFieldIndex(AccDataDescOpenACCField::StartIndices),
            7);
}
