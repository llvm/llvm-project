//===- ResourceTest.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../test/lib/Dialect/Test/TestAttributes.h"
#include "../../test/lib/Dialect/Test/TestDialect.h"
#include "mlir/Parser/Parser.h"

#include "gmock/gmock.h"

using namespace mlir;

namespace {
TEST(MLIRParser, ResourceKeyConflict) {
  std::string moduleStr = R"mlir(
    "test.use1"() {attr = #test.e1di64_elements<blob1> : tensor<3xi64> } : () -> ()

    {-#
      dialect_resources: {
        test: {
          blob1: "0x08000000010000000000000002000000000000000300000000000000"
        }
      }
    #-}
  )mlir";
  std::string moduleStr2 = R"mlir(
    "test.use2"() {attr = #test.e1di64_elements<blob1> : tensor<3xi64> } : () -> ()

    {-#
      dialect_resources: {
        test: {
          blob1: "0x08000000040000000000000005000000000000000600000000000000"
        }
      }
    #-}
  )mlir";

  MLIRContext context;
  context.loadDialect<test::TestDialect>();

  // Parse both modules into the same context so that we ensure the conflicting
  // resources have been loaded.
  OwningOpRef<ModuleOp> module1 =
      parseSourceString<ModuleOp>(moduleStr, &context);
  OwningOpRef<ModuleOp> module2 =
      parseSourceString<ModuleOp>(moduleStr2, &context);
  ASSERT_TRUE(module1 && module2);

  // Merge the two modules so that we can test printing the remapped resources.
  Block *block = module1->getBody();
  block->getOperations().splice(block->end(),
                                module2->getBody()->getOperations());

  // Check that conflicting resources were remapped.
  std::string outputStr;
  {
    llvm::raw_string_ostream os(outputStr);
    module1->print(os);
  }
  StringRef output(outputStr);
  EXPECT_TRUE(
      output.contains("\"test.use1\"() {attr = #test.e1di64_elements<blob1>"));
  EXPECT_TRUE(output.contains(
      "blob1: \"0x08000000010000000000000002000000000000000300000000000000\""));
  EXPECT_TRUE(output.contains(
      "\"test.use2\"() {attr = #test.e1di64_elements<blob1_1>"));
  EXPECT_TRUE(output.contains(
      "blob1_1: "
      "\"0x08000000040000000000000005000000000000000600000000000000\""));
}
} // namespace
