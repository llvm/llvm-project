//===- llvm/unittests/Target/DirectX/PointerTypeAnalysisTests.cpp ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DirectXTargetMachine.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SourceMgr.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::dxil;

namespace {
class UniqueResourceFromUseTest : public testing::Test {
protected:
  PassBuilder *PB;
  ModuleAnalysisManager *MAM;
  LLVMContext *Context;
  virtual void SetUp() {
    Context = new LLVMContext();
    MAM = new ModuleAnalysisManager();
    PB = new PassBuilder();
    PB->registerModuleAnalyses(*MAM);
    MAM->registerPass([&] { return DXILResourceTypeAnalysis(); });
    MAM->registerPass([&] { return DXILResourceAnalysis(); });
  }

  std::unique_ptr<Module> parseAsm(StringRef Asm) {
    SMDiagnostic Error;
    std::unique_ptr<Module> M = parseAssemblyString(Asm, Error, *Context);
    EXPECT_TRUE(M) << "Bad assembly?: " << Error.getMessage();
    return M;
  }

  virtual void TearDown() {
    delete PB;
    delete MAM;
    delete Context;
  }
};

// Test that several calls to decrement on the same resource don't raise a
// Diagnositic and resolves to a single decrement entry
TEST_F(UniqueResourceFromUseTest, TestResourceCounterDecrement) {
  StringRef Assembly = R"(
define void @main() {
entry:
  %handle = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding(i32 1, i32 2, i32 3, i32 4, ptr null)
  call i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", float, 1, 0) %handle, i8 -1)
  call i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", float, 1, 0) %handle, i8 -1)
  call i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", float, 1, 0) %handle, i8 -1)
  ret void
}
  )";

  auto M = parseAsm(Assembly);

  DXILResourceMap &DRM = MAM->getResult<DXILResourceAnalysis>(*M);

  for (const Function &F : M->functions()) {
    if (F.getIntrinsicID() != Intrinsic::dx_resource_handlefrombinding)
      continue;

    for (const User *U : F.users()) {
      const CallInst *CI = cast<CallInst>(U);
      const auto *const Binding = DRM.find(CI);
      ASSERT_EQ(Binding->CounterDirection, ResourceCounterDirection::Decrement);
    }
  }
}

// Test that several calls to increment on the same resource don't raise a
// Diagnositic and resolves to a single increment entry
TEST_F(UniqueResourceFromUseTest, TestResourceCounterIncrement) {
  StringRef Assembly = R"(
define void @main() {
entry:
  %handle = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding(i32 1, i32 2, i32 3, i32 4, ptr null)
  call i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", float, 1, 0) %handle, i8 1)
  call i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", float, 1, 0) %handle, i8 1)
  call i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", float, 1, 0) %handle, i8 1)
  ret void
}
  )";

  auto M = parseAsm(Assembly);

  DXILResourceMap &DRM = MAM->getResult<DXILResourceAnalysis>(*M);

  for (const Function &F : M->functions()) {
    if (F.getIntrinsicID() != Intrinsic::dx_resource_handlefrombinding)
      continue;

    for (const User *U : F.users()) {
      const CallInst *CI = cast<CallInst>(U);
      const auto *const Binding = DRM.find(CI);
      ASSERT_EQ(Binding->CounterDirection, ResourceCounterDirection::Increment);
    }
  }
}

// Test that looking up a resource that doesn't have the counter updated
// resoves to unknown
TEST_F(UniqueResourceFromUseTest, TestResourceCounterUnknown) {
  StringRef Assembly = R"(
define void @main() {
entry:
  %handle = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding(i32 1, i32 2, i32 3, i32 4, ptr null)
  ret void
}
  )";

  auto M = parseAsm(Assembly);

  DXILResourceMap &DRM = MAM->getResult<DXILResourceAnalysis>(*M);

  for (const Function &F : M->functions()) {
    if (F.getIntrinsicID() != Intrinsic::dx_resource_handlefrombinding)
      continue;

    for (const User *U : F.users()) {
      const CallInst *CI = cast<CallInst>(U);
      const auto *const Binding = DRM.find(CI);
      ASSERT_EQ(Binding->CounterDirection, ResourceCounterDirection::Unknown);
    }
  }
}

// Test that multiple different resources with unique incs/decs aren't
// marked invalid
TEST_F(UniqueResourceFromUseTest, TestResourceCounterMultiple) {
  StringRef Assembly = R"(
define void @main() {
entry:
  %handle1 = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding(i32 1, i32 2, i32 3, i32 4, ptr null)
  %handle2 = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding(i32 4, i32 3, i32 2, i32 1, ptr null)
  call i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", float, 1, 0) %handle1, i8 -1)
  call i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", float, 1, 0) %handle2, i8 1)
  ret void
}
  )";

  auto M = parseAsm(Assembly);

  DXILResourceMap &DRM = MAM->getResult<DXILResourceAnalysis>(*M);

  ResourceCounterDirection Dirs[2] = {ResourceCounterDirection::Decrement,
                                      ResourceCounterDirection::Increment};
  ResourceCounterDirection *Dir = Dirs;

  for (const Function &F : M->functions()) {
    if (F.getIntrinsicID() != Intrinsic::dx_resource_handlefrombinding)
      continue;

    uint32_t ExpectedDirsIndex = 0;
    for (const User *U : F.users()) {
      const CallInst *CI = cast<CallInst>(U);
      const auto *const Binding = DRM.find(CI);
      ASSERT_TRUE(ExpectedDirsIndex < 2);
      ASSERT_EQ(Binding->CounterDirection, Dir[ExpectedDirsIndex]);
      ExpectedDirsIndex++;
    }
  }
}

// Test that single different resources with unique incs/decs is marked invalid
TEST_F(UniqueResourceFromUseTest, TestResourceCounterInvalid) {
  StringRef Assembly = R"(
define void @main() {
entry:
  %handle = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding(i32 1, i32 2, i32 3, i32 4, ptr null)
  call i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", float, 1, 0) %handle, i8 -1)
  call i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", float, 1, 0) %handle, i8 1)
  ret void
}
  )";

  auto M = parseAsm(Assembly);

  DXILResourceMap &DRM = MAM->getResult<DXILResourceAnalysis>(*M);

  for (const Function &F : M->functions()) {
    if (F.getIntrinsicID() != Intrinsic::dx_resource_handlefrombinding)
      continue;

    for (const User *U : F.users()) {
      const CallInst *CI = cast<CallInst>(U);
      const auto *const Binding = DRM.find(CI);
      ASSERT_EQ(Binding->CounterDirection, ResourceCounterDirection::Invalid);
    }
  }
}

} // namespace
