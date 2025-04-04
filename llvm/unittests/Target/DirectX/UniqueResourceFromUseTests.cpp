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
    MAM->registerPass([&] { return DXILResourceBindingAnalysis(); });
    MAM->registerPass([&] { return DXILResourceCounterDirectionAnalysis(); });
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

TEST_F(UniqueResourceFromUseTest, TestTrivialUse) {
  StringRef Assembly = R"(
define void @main() {
entry:
  %handle = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding(i32 1, i32 2, i32 3, i32 4, i1 false)
  call void @a.func(target("dx.RawBuffer", float, 1, 0) %handle)
  call void @a.func(target("dx.RawBuffer", float, 1, 0) %handle)
  ret void
}

declare void @a.func(target("dx.RawBuffer", float, 1, 0) %handle)
  )";

  auto M = parseAsm(Assembly);

  const DXILBindingMap &DBM = MAM->getResult<DXILResourceBindingAnalysis>(*M);
  for (const Function &F : M->functions()) {
    if (F.getName() != "a.func") {
      continue;
    }

    unsigned CalledResources = 0;

    for (const User *U : F.users()) {
      const CallInst *CI = cast<CallInst>(U);
      const Value *Handle = CI->getArgOperand(0);
      const auto Bindings = DBM.findByUse(Handle);
      ASSERT_EQ(Bindings.size(), 1u)
          << "Handle should resolve into one resource";

      auto Binding = Bindings[0]->getBinding();
      EXPECT_EQ(0u, Binding.RecordID);
      EXPECT_EQ(1u, Binding.Space);
      EXPECT_EQ(2u, Binding.LowerBound);
      EXPECT_EQ(3u, Binding.Size);

      CalledResources++;
    }

    EXPECT_EQ(2u, CalledResources)
        << "Expected 2 resolved call to create resource";
  }
}

TEST_F(UniqueResourceFromUseTest, TestIndirectUse) {
  StringRef Assembly = R"(
define void @foo() {
  %handle = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding(i32 1, i32 2, i32 3, i32 4, i1 false)
  %handle2 = call target("dx.RawBuffer", float, 1, 0) @ind.func(target("dx.RawBuffer", float, 1, 0) %handle)
  %handle3 = call target("dx.RawBuffer", float, 1, 0) @ind.func(target("dx.RawBuffer", float, 1, 0) %handle2)
  %handle4 = call target("dx.RawBuffer", float, 1, 0) @ind.func(target("dx.RawBuffer", float, 1, 0) %handle3)
  call void @a.func(target("dx.RawBuffer", float, 1, 0) %handle4)
  ret void
}

declare void @a.func(target("dx.RawBuffer", float, 1, 0) %handle)
declare target("dx.RawBuffer", float, 1, 0) @ind.func(target("dx.RawBuffer", float, 1, 0) %handle)
  )";

  auto M = parseAsm(Assembly);

  const DXILBindingMap &DBM = MAM->getResult<DXILResourceBindingAnalysis>(*M);
  for (const Function &F : M->functions()) {
    if (F.getName() != "a.func") {
      continue;
    }

    unsigned CalledResources = 0;

    for (const User *U : F.users()) {
      const CallInst *CI = cast<CallInst>(U);
      const Value *Handle = CI->getArgOperand(0);
      const auto Bindings = DBM.findByUse(Handle);
      ASSERT_EQ(Bindings.size(), 1u)
          << "Handle should resolve into one resource";

      auto Binding = Bindings[0]->getBinding();
      EXPECT_EQ(0u, Binding.RecordID);
      EXPECT_EQ(1u, Binding.Space);
      EXPECT_EQ(2u, Binding.LowerBound);
      EXPECT_EQ(3u, Binding.Size);

      CalledResources++;
    }

    EXPECT_EQ(1u, CalledResources)
        << "Expected 1 resolved call to create resource";
  }
}

TEST_F(UniqueResourceFromUseTest, TestAmbigousIndirectUse) {
  StringRef Assembly = R"(
define void @foo() {
  %foo = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding(i32 1, i32 1, i32 1, i32 1, i1 false)
  %bar = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding(i32 2, i32 2, i32 2, i32 2, i1 false)
  %baz = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding(i32 3, i32 3, i32 3, i32 3, i1 false)
  %bat = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding(i32 4, i32 4, i32 4, i32 4, i1 false)
  %a = call target("dx.RawBuffer", float, 1, 0) @ind.func(target("dx.RawBuffer", float, 1, 0) %foo, target("dx.RawBuffer", float, 1, 0) %bar)
  %b = call target("dx.RawBuffer", float, 1, 0) @ind.func(target("dx.RawBuffer", float, 1, 0) %baz, target("dx.RawBuffer", float, 1, 0) %bat)
  %handle = call target("dx.RawBuffer", float, 1, 0) @ind.func(target("dx.RawBuffer", float, 1, 0) %a, target("dx.RawBuffer", float, 1, 0) %b)
  call void @a.func(target("dx.RawBuffer", float, 1, 0) %handle)
  ret void
}

declare void @a.func(target("dx.RawBuffer", float, 1, 0) %handle)
declare target("dx.RawBuffer", float, 1, 0) @ind.func(target("dx.RawBuffer", float, 1, 0) %x, target("dx.RawBuffer", float, 1, 0) %y)
  )";

  auto M = parseAsm(Assembly);

  const DXILBindingMap &DBM = MAM->getResult<DXILResourceBindingAnalysis>(*M);
  for (const Function &F : M->functions()) {
    if (F.getName() != "a.func") {
      continue;
    }

    unsigned CalledResources = 0;

    for (const User *U : F.users()) {
      const CallInst *CI = cast<CallInst>(U);
      const Value *Handle = CI->getArgOperand(0);
      const auto Bindings = DBM.findByUse(Handle);
      ASSERT_EQ(Bindings.size(), 4u)
          << "Handle should resolve into four resources";

      auto Binding = Bindings[0]->getBinding();
      EXPECT_EQ(0u, Binding.RecordID);
      EXPECT_EQ(1u, Binding.Space);
      EXPECT_EQ(1u, Binding.LowerBound);
      EXPECT_EQ(1u, Binding.Size);

      Binding = Bindings[1]->getBinding();
      EXPECT_EQ(1u, Binding.RecordID);
      EXPECT_EQ(2u, Binding.Space);
      EXPECT_EQ(2u, Binding.LowerBound);
      EXPECT_EQ(2u, Binding.Size);

      Binding = Bindings[2]->getBinding();
      EXPECT_EQ(2u, Binding.RecordID);
      EXPECT_EQ(3u, Binding.Space);
      EXPECT_EQ(3u, Binding.LowerBound);
      EXPECT_EQ(3u, Binding.Size);

      Binding = Bindings[3]->getBinding();
      EXPECT_EQ(3u, Binding.RecordID);
      EXPECT_EQ(4u, Binding.Space);
      EXPECT_EQ(4u, Binding.LowerBound);
      EXPECT_EQ(4u, Binding.Size);

      CalledResources++;
    }

    EXPECT_EQ(1u, CalledResources)
        << "Expected 1 resolved call to create resource";
  }
}

TEST_F(UniqueResourceFromUseTest, TestConditionalUse) {
  StringRef Assembly = R"(
define void @foo(i32 %n) {
entry:
  %x = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding(i32 1, i32 1, i32 1, i32 1, i1 false)
  %y = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding(i32 4, i32 4, i32 4, i32 4, i1 false)
  %cond = icmp eq i32 %n, 0
  br i1 %cond, label %bb.true, label %bb.false

bb.true:
  %handle_t = call target("dx.RawBuffer", float, 1, 0) @ind.func(target("dx.RawBuffer", float, 1, 0) %x)
  br label %bb.exit

bb.false:
  %handle_f = call target("dx.RawBuffer", float, 1, 0) @ind.func(target("dx.RawBuffer", float, 1, 0) %y)
  br label %bb.exit

bb.exit:
  %handle = phi target("dx.RawBuffer", float, 1, 0) [ %handle_t, %bb.true ], [ %handle_f, %bb.false ]
  call void @a.func(target("dx.RawBuffer", float, 1, 0) %handle)
  ret void
}

declare void @a.func(target("dx.RawBuffer", float, 1, 0) %handle)
declare target("dx.RawBuffer", float, 1, 0) @ind.func(target("dx.RawBuffer", float, 1, 0) %x)
  )";

  auto M = parseAsm(Assembly);

  const DXILBindingMap &DBM = MAM->getResult<DXILResourceBindingAnalysis>(*M);
  for (const Function &F : M->functions()) {
    if (F.getName() != "a.func") {
      continue;
    }

    unsigned CalledResources = 0;

    for (const User *U : F.users()) {
      const CallInst *CI = cast<CallInst>(U);
      const Value *Handle = CI->getArgOperand(0);
      const auto Bindings = DBM.findByUse(Handle);
      ASSERT_EQ(Bindings.size(), 2u)
          << "Handle should resolve into four resources";

      auto Binding = Bindings[0]->getBinding();
      EXPECT_EQ(0u, Binding.RecordID);
      EXPECT_EQ(1u, Binding.Space);
      EXPECT_EQ(1u, Binding.LowerBound);
      EXPECT_EQ(1u, Binding.Size);

      Binding = Bindings[1]->getBinding();
      EXPECT_EQ(1u, Binding.RecordID);
      EXPECT_EQ(4u, Binding.Space);
      EXPECT_EQ(4u, Binding.LowerBound);
      EXPECT_EQ(4u, Binding.Size);

      CalledResources++;
    }

    EXPECT_EQ(1u, CalledResources)
        << "Expected 1 resolved call to create resource";
  }
}

// Test that several calls to decrement on the same resource don't raise a
// Diagnositic and resolves to a single decrement entry
TEST_F(UniqueResourceFromUseTest, TestResourceCounterDecrement) {
  StringRef Assembly = R"(
define void @main() {
entry:
  %handle = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding(i32 1, i32 2, i32 3, i32 4, i1 false)
  call i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", float, 1, 0) %handle, i8 -1)
  call i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", float, 1, 0) %handle, i8 -1)
  call i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", float, 1, 0) %handle, i8 -1)
  ret void
}
  )";

  auto M = parseAsm(Assembly);

  const DXILBindingMap &DBM = MAM->getResult<DXILResourceBindingAnalysis>(*M);
  const DXILResourceCounterDirectionMap &DCDM =
      MAM->getResult<DXILResourceCounterDirectionAnalysis>(*M);

  for (const Function &F : M->functions()) {
    if (F.getIntrinsicID() != Intrinsic::dx_resource_handlefrombinding) {
      continue;
    }

    for (const User *U : F.users()) {
      const CallInst *CI = cast<CallInst>(U);
      const auto *const Binding = DBM.find(CI);
      ASSERT_EQ(DCDM[*Binding], ResourceCounterDirection::Decrement);
    }
  }
}

// Test that several calls to increment on the same resource don't raise a
// Diagnositic and resolves to a single increment entry
TEST_F(UniqueResourceFromUseTest, TestResourceCounterIncrement) {
  StringRef Assembly = R"(
define void @main() {
entry:
  %handle = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding(i32 1, i32 2, i32 3, i32 4, i1 false)
  call i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", float, 1, 0) %handle, i8 1)
  call i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", float, 1, 0) %handle, i8 1)
  call i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", float, 1, 0) %handle, i8 1)
  ret void
}
  )";

  auto M = parseAsm(Assembly);

  const DXILBindingMap &DBM = MAM->getResult<DXILResourceBindingAnalysis>(*M);
  const DXILResourceCounterDirectionMap &DCDM =
      MAM->getResult<DXILResourceCounterDirectionAnalysis>(*M);

  for (const Function &F : M->functions()) {
    if (F.getIntrinsicID() != Intrinsic::dx_resource_handlefrombinding) {
      continue;
    }

    for (const User *U : F.users()) {
      const CallInst *CI = cast<CallInst>(U);
      const auto *const Binding = DBM.find(CI);
      ASSERT_EQ(DCDM[*Binding], ResourceCounterDirection::Increment);
    }
  }
}

// Test that looking up a resource that doesn't have the counter updated
// resoves to unknown
TEST_F(UniqueResourceFromUseTest, TestResourceCounterUnknown) {
  StringRef Assembly = R"(
define void @main() {
entry:
  %handle = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding(i32 1, i32 2, i32 3, i32 4, i1 false)
  ret void
}
  )";

  auto M = parseAsm(Assembly);

  const DXILBindingMap &DBM = MAM->getResult<DXILResourceBindingAnalysis>(*M);
  const DXILResourceCounterDirectionMap &DCDM =
      MAM->getResult<DXILResourceCounterDirectionAnalysis>(*M);

  for (const Function &F : M->functions()) {
    if (F.getIntrinsicID() != Intrinsic::dx_resource_handlefrombinding) {
      continue;
    }

    for (const User *U : F.users()) {
      const CallInst *CI = cast<CallInst>(U);
      const auto *const Binding = DBM.find(CI);
      ASSERT_EQ(DCDM[*Binding], ResourceCounterDirection::Unknown);
    }
  }
}

// Test that multiple different resources with unique incs/decs aren't
// marked invalid
TEST_F(UniqueResourceFromUseTest, TestResourceCounterMultiple) {
  StringRef Assembly = R"(
define void @main() {
entry:
  %handle1 = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding(i32 1, i32 2, i32 3, i32 4, i1 false)
  %handle2 = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding(i32 4, i32 3, i32 2, i32 1, i1 false)
  call i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", float, 1, 0) %handle1, i8 -1)
  call i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", float, 1, 0) %handle2, i8 1)
  ret void
}
  )";

  auto M = parseAsm(Assembly);

  const DXILBindingMap &DBM = MAM->getResult<DXILResourceBindingAnalysis>(*M);
  const DXILResourceCounterDirectionMap &DCDM =
      MAM->getResult<DXILResourceCounterDirectionAnalysis>(*M);

  ResourceCounterDirection Dirs[2] = {ResourceCounterDirection::Decrement,
                                      ResourceCounterDirection::Increment};
  ResourceCounterDirection *Dir = Dirs;

  for (const Function &F : M->functions()) {
    if (F.getIntrinsicID() != Intrinsic::dx_resource_handlefrombinding) {
      continue;
    }

    for (const User *U : F.users()) {
      const CallInst *CI = cast<CallInst>(U);
      const auto *const Binding = DBM.find(CI);
      ASSERT_EQ(DCDM[*Binding], *Dir);
      Dir++;
    }
  }
}

// Test that single different resources with unique incs/decs is marked invalid
TEST_F(UniqueResourceFromUseTest, TestResourceCounterInvalid) {
  StringRef Assembly = R"(
define void @main() {
entry:
  %handle = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding(i32 1, i32 2, i32 3, i32 4, i1 false)
  call i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", float, 1, 0) %handle, i8 -1)
  call i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", float, 1, 0) %handle, i8 1)
  ret void
}
  )";

  auto M = parseAsm(Assembly);

  const DXILBindingMap &DBM = MAM->getResult<DXILResourceBindingAnalysis>(*M);
  const DXILResourceCounterDirectionMap &DCDM =
      MAM->getResult<DXILResourceCounterDirectionAnalysis>(*M);

  for (const Function &F : M->functions()) {
    if (F.getIntrinsicID() != Intrinsic::dx_resource_handlefrombinding) {
      continue;
    }

    for (const User *U : F.users()) {
      const CallInst *CI = cast<CallInst>(U);
      const auto *const Binding = DBM.find(CI);
      ASSERT_EQ(DCDM[*Binding], ResourceCounterDirection::Invalid);
    }
  }
}

} // namespace
