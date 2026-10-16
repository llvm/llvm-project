//===- AddressSpacesTest.cpp - TTI address space enumeration test ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests that AMDGPU reports the address spaces it gives a meaning to, as
// listed in the AMDGPU Address Spaces table in AMDGPUUsage.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUUnitTests.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Target/TargetMachine.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST_F(AMDGPUTestBase, ReportsDocumentedAddressSpaces) {
  StringRef ModuleString = R"(
  define amdgpu_kernel void @test() {
    ret void
  }
  )";
  LLVMContext Context;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(ModuleString, Err, Context);
  ASSERT_TRUE(M) << Err.getMessage();

  Function *F = M->getFunction("test");
  ASSERT_TRUE(F);

  auto TM =
      createAMDGPUTargetMachine(Triple("amdgcn-amd-amdhsa"), "gfx900", "");
  ASSERT_TRUE(TM);
  TargetTransformInfo TTI = TM->getTargetTransformInfo(*F);

  using namespace AMDGPUAS;
  EXPECT_THAT(TTI.getAddressSpaces(),
              ::testing::ElementsAre(
                  FLAT_ADDRESS, GLOBAL_ADDRESS, REGION_ADDRESS, LOCAL_ADDRESS,
                  CONSTANT_ADDRESS, PRIVATE_ADDRESS, CONSTANT_ADDRESS_32BIT,
                  BUFFER_FAT_POINTER, BUFFER_RESOURCE, BUFFER_STRIDED_POINTER,
                  STREAMOUT_REGISTER));
}
