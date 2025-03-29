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

  virtual void SetUp() {
    MAM = new ModuleAnalysisManager();
    PB = new PassBuilder();
    PB->registerModuleAnalyses(*MAM);
    MAM->registerPass([&] { return DXILResourceTypeAnalysis(); });
    MAM->registerPass([&] { return DXILResourceBindingAnalysis(); });
  }

  virtual void TearDown() {
    delete PB;
    delete MAM;
  }
};

TEST_F(UniqueResourceFromUseTest, TestTrivialUse) {
  StringRef Assembly = R"(
define void @main() {
entry:
  %handle = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32 1, i32 2, i32 3, i32 4, i1 false)
  call void @a.func(target("dx.RawBuffer", float, 1, 0) %handle)
  call void @a.func(target("dx.RawBuffer", float, 1, 0) %handle)
  ret void
}

declare target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32, i32, i32, i32, i1)
declare void @a.func(target("dx.RawBuffer", float, 1, 0) %handle)
  )";

  LLVMContext Context;
  SMDiagnostic Error;
  auto M = parseAssemblyString(Assembly, Error, Context);
  ASSERT_TRUE(M) << "Bad assembly?";

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

      auto Binding = Bindings[0].getBinding();
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
  %handle = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32 1, i32 2, i32 3, i32 4, i1 false)
  %handle2 = call target("dx.RawBuffer", float, 1, 0) @ind.func(target("dx.RawBuffer", float, 1, 0) %handle)
  %handle3 = call target("dx.RawBuffer", float, 1, 0) @ind.func(target("dx.RawBuffer", float, 1, 0) %handle2)
  %handle4 = call target("dx.RawBuffer", float, 1, 0) @ind.func(target("dx.RawBuffer", float, 1, 0) %handle3)
  call void @a.func(target("dx.RawBuffer", float, 1, 0) %handle4)
  ret void
}

declare target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32, i32, i32, i32, i1)
declare void @a.func(target("dx.RawBuffer", float, 1, 0) %handle)
declare target("dx.RawBuffer", float, 1, 0) @ind.func(target("dx.RawBuffer", float, 1, 0) %handle)
  )";

  LLVMContext Context;
  SMDiagnostic Error;
  auto M = parseAssemblyString(Assembly, Error, Context);
  ASSERT_TRUE(M) << "Bad assembly?";

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

      auto Binding = Bindings[0].getBinding();
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
  %foo = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32 1, i32 1, i32 1, i32 1, i1 false)
  %bar = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32 2, i32 2, i32 2, i32 2, i1 false)
  %baz = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32 3, i32 3, i32 3, i32 3, i1 false)
  %bat = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32 4, i32 4, i32 4, i32 4, i1 false)
  %a = call target("dx.RawBuffer", float, 1, 0) @ind.func(target("dx.RawBuffer", float, 1, 0) %foo, target("dx.RawBuffer", float, 1, 0) %bar)
  %b = call target("dx.RawBuffer", float, 1, 0) @ind.func(target("dx.RawBuffer", float, 1, 0) %baz, target("dx.RawBuffer", float, 1, 0) %bat)
  %handle = call target("dx.RawBuffer", float, 1, 0) @ind.func(target("dx.RawBuffer", float, 1, 0) %a, target("dx.RawBuffer", float, 1, 0) %b)
  call void @a.func(target("dx.RawBuffer", float, 1, 0) %handle)
  ret void
}

declare target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32, i32, i32, i32, i1)
declare void @a.func(target("dx.RawBuffer", float, 1, 0) %handle)
declare target("dx.RawBuffer", float, 1, 0) @ind.func(target("dx.RawBuffer", float, 1, 0) %x, target("dx.RawBuffer", float, 1, 0) %y)
  )";

  LLVMContext Context;
  SMDiagnostic Error;
  auto M = parseAssemblyString(Assembly, Error, Context);
  ASSERT_TRUE(M) << "Bad assembly?";

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

      auto Binding = Bindings[0].getBinding();
      EXPECT_EQ(0u, Binding.RecordID);
      EXPECT_EQ(1u, Binding.Space);
      EXPECT_EQ(1u, Binding.LowerBound);
      EXPECT_EQ(1u, Binding.Size);

      Binding = Bindings[1].getBinding();
      EXPECT_EQ(1u, Binding.RecordID);
      EXPECT_EQ(2u, Binding.Space);
      EXPECT_EQ(2u, Binding.LowerBound);
      EXPECT_EQ(2u, Binding.Size);

      Binding = Bindings[2].getBinding();
      EXPECT_EQ(2u, Binding.RecordID);
      EXPECT_EQ(3u, Binding.Space);
      EXPECT_EQ(3u, Binding.LowerBound);
      EXPECT_EQ(3u, Binding.Size);

      Binding = Bindings[3].getBinding();
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
  %x = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32 1, i32 1, i32 1, i32 1, i1 false)
  %y = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32 4, i32 4, i32 4, i32 4, i1 false)
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

declare target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32, i32, i32, i32, i1)
declare void @a.func(target("dx.RawBuffer", float, 1, 0) %handle)
declare target("dx.RawBuffer", float, 1, 0) @ind.func(target("dx.RawBuffer", float, 1, 0) %x)
  )";

  LLVMContext Context;
  SMDiagnostic Error;
  auto M = parseAssemblyString(Assembly, Error, Context);
  ASSERT_TRUE(M) << "Bad assembly?";

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

      auto Binding = Bindings[0].getBinding();
      EXPECT_EQ(0u, Binding.RecordID);
      EXPECT_EQ(1u, Binding.Space);
      EXPECT_EQ(1u, Binding.LowerBound);
      EXPECT_EQ(1u, Binding.Size);

      Binding = Bindings[1].getBinding();
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

} // namespace
