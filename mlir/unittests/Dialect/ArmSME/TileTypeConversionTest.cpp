//===- TileTypeConversionTest.cpp - Tests ArmSME tile type conversion -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArmSMEToLLVM/ArmSMEToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"

#include "gtest/gtest.h"

using namespace mlir;

class ArmSMETest : public ::testing::Test {
protected:
  ArmSMETest() { context.getOrLoadDialect<mlir::arm_sme::ArmSMEDialect>(); }

  mlir::MLIRContext context;
};

TEST_F(ArmSMETest, TestTileTypeConversion) {
  LLVMTypeConverter llvmConverter(&context);
  LLVMTypeConverter llvmConverterWithArmSMEConversion(&context);

  RewritePatternSet patterns(&context);
  populateArmSMEToLLVMConversionPatterns(llvmConverterWithArmSMEConversion,
                                         patterns);

  Type i32 = IntegerType::get(&context, 32);
  auto smeTileType = VectorType::get({4, 4}, i32, {true, true});

  // An unmodified LLVMTypeConverer should fail to convert an ArmSME tile type.
  {
    SmallVector<Type> convertedType;
    ASSERT_TRUE(failed(llvmConverter.convertType(smeTileType, convertedType)));
  }

  // An updated LLVMTypeConverer should return the ArmSME tile vector type
  // unchanged.
  {
    SmallVector<Type> convertedType;
    ASSERT_TRUE(succeeded(llvmConverterWithArmSMEConversion.convertType(
        smeTileType, convertedType)));
    ASSERT_EQ(ArrayRef<Type>(convertedType), ArrayRef<Type>{smeTileType});
  }
}
