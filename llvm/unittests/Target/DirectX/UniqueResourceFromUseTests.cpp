//===- llvm/unittests/Target/DirectX/PointerTypeAnalysisTests.cpp ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DirectXIRPasses/PointerTypeAnalysis.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/TypedPointerType.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/Debugify.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::Contains;
using ::testing::Pair;

using namespace llvm;
using namespace llvm::dxil;

template <typename T> struct IsA {
  friend bool operator==(const Value *V, const IsA &) { return isa<T>(V); }
};

TEST(UniqueResourceFromUse, TestTrivialUse) {
  StringRef Assembly = R"(
define void @main() {
entry:
  %handle = call target("dx.RawBuffer", float, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_f32_1_0t(i32 1, i32 2, i32 3, i32 4, i1 false)
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
  DebugifyCustomPassManager Passes;
  Passes.add(createDXILResourceTypeWrapperPassPass());
  DXILResourceBindingWrapperPass* RBPass = new DXILResourceBindingWrapperPass();
  Passes.add(RBPass);
  Passes.run(*M);

  const DXILBindingMap &DBM = RBPass->getBindingMap();
  for (const Function& F : M->functions()) {
    if (F.getName() != "a.func") {
      continue;
    }

    unsigned CalledResources = 0;

    for (const User* U : F.users()) {
      const CallInst* CI = dyn_cast<CallInst>(U);
      ASSERT_TRUE(CI) << "All users of @a.func must be CallInst";

      const Value* Handle = CI->getArgOperand(0);

      const auto* It = DBM.findByUse(Handle);
      ASSERT_TRUE(It != DBM.end()) << "Handle should resolve into resource";

      const llvm::dxil::ResourceBindingInfo::ResourceBinding& Binding = It->getBinding();
      EXPECT_EQ(0u, Binding.RecordID);
      EXPECT_EQ(1u, Binding.Space);
      EXPECT_EQ(2u, Binding.LowerBound);
      EXPECT_EQ(3u, Binding.Size);

      CalledResources++;
    }

    EXPECT_EQ(1u, CalledResources) << "Expected exactly 1 resolved call to create resource";
  }

}
