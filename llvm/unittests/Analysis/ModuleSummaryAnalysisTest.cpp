//===- ModuleSummaryAnalysisTest.cpp - Module summary analysis tests ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TargetTransformInfoImpl.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Transforms/Utils/AssignGUID.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;

using CallTypeIdInfo = FunctionSummary::CallTypeIdInfo;

/// The call type ID of the test target: the calling convention and the number
/// of parameters, so that the tests can predict it.
static uint64_t testCallTypeId(CallingConv::ID CC, unsigned NumParams) {
  return (uint64_t(CC) << 32) | (NumParams + 1);
}

static const char *ModuleIR = R"IR(
@fp = global ptr @local_taken

define void @external(i32 %x) {
  ret void
}

define internal void @local_not_taken() {
  ret void
}

define internal void @local_taken() {
  ret void
}

define spir_kernel void @kernel(ptr %fp) {
  call void @local_not_taken()
  call void %fp(i32 1)
  call void %fp(i32 2)
  call fastcc void %fp(i32 3)
  ret void
}

define void @unknown_code() {
  call void asm sideeffect "", ""()
  call void inttoptr (i64 4096 to ptr)()
  ret void
}
)IR";

namespace {

/// A target that uses call type IDs.
class CallTypeIdTTIImpl : public TargetTransformInfoImplBase {
public:
  explicit CallTypeIdTTIImpl(const DataLayout &DL)
      : TargetTransformInfoImplBase(DL) {}

  std::optional<uint64_t> getCallTypeId(FunctionType *FTy,
                                        CallingConv::ID CC) const override {
    return testCallTypeId(CC, FTy->getNumParams());
  }
};

class ModuleSummaryAnalysisTest : public testing::Test {
protected:
  void SetUp() override {
    SMDiagnostic Err;
    M = parseAssemblyString(ModuleIR, Err, Context);
    ASSERT_TRUE(M) << Err.getMessage().str();
    // The summary refers to the global values by their GUIDs.
    AssignGUIDPass::runOnModule(*M);
  }

  ModuleSummaryIndex buildIndex(const TargetTransformInfo *TTI) {
    ProfileSummaryInfo PSI(*M);
    return buildModuleSummaryIndex(
        *M, /*GetBFICallback=*/nullptr, &PSI,
        [](const Function &) -> const StackSafetyInfo * { return nullptr; },
        [TTI](const Function &) { return TTI; });
  }

  const CallTypeIdInfo *getCallTypeIdInfo(const ModuleSummaryIndex &Index,
                                          StringRef Name) {
    const GlobalValueSummary *Summary =
        Index.getGlobalValueSummary(*M->getFunction(Name));
    return cast<FunctionSummary>(Summary)->getCallTypeIdInfo();
  }

  LLVMContext Context;
  std::unique_ptr<Module> M;
};

TEST_F(ModuleSummaryAnalysisTest, CallTypeIds) {
  TargetTransformInfo TTI(
      std::make_unique<CallTypeIdTTIImpl>(M->getDataLayout()));
  ModuleSummaryIndex Index = buildIndex(&TTI);

  // An indirect call can reach a function with non-local linkage.
  const CallTypeIdInfo *External = getCallTypeIdInfo(Index, "external");
  ASSERT_NE(External, nullptr);
  EXPECT_EQ(External->CallTypeId, testCallTypeId(CallingConv::C, 1));
  EXPECT_TRUE(External->IndirectCallTypeIds.empty());
  EXPECT_FALSE(External->HasCallWithoutTypeId);

  // A local function has a call type ID only if its address is taken. Without
  // one, the summary still has call type IDs, all empty.
  const CallTypeIdInfo *NotTaken = getCallTypeIdInfo(Index, "local_not_taken");
  ASSERT_NE(NotTaken, nullptr);
  EXPECT_EQ(NotTaken->CallTypeId, 0u);
  const CallTypeIdInfo *Taken = getCallTypeIdInfo(Index, "local_taken");
  ASSERT_NE(Taken, nullptr);
  EXPECT_EQ(Taken->CallTypeId, testCallTypeId(CallingConv::C, 0));

  // A kernel cannot be called. The call type IDs of its indirect calls are
  // distinct, and the calling convention is part of the ID. A direct call has
  // no call type ID.
  const CallTypeIdInfo *Kernel = getCallTypeIdInfo(Index, "kernel");
  ASSERT_NE(Kernel, nullptr);
  EXPECT_EQ(Kernel->CallTypeId, 0u);
  EXPECT_THAT(Kernel->IndirectCallTypeIds,
              testing::ElementsAre(testCallTypeId(CallingConv::C, 1),
                                   testCallTypeId(CallingConv::Fast, 1)));
  EXPECT_FALSE(Kernel->HasCallWithoutTypeId);

  // A call to inline assembly or to an absolute address has no call type ID.
  const CallTypeIdInfo *Unknown = getCallTypeIdInfo(Index, "unknown_code");
  ASSERT_NE(Unknown, nullptr);
  EXPECT_EQ(Unknown->CallTypeId, testCallTypeId(CallingConv::C, 0));
  EXPECT_TRUE(Unknown->IndirectCallTypeIds.empty());
  EXPECT_TRUE(Unknown->HasCallWithoutTypeId);
}

TEST_F(ModuleSummaryAnalysisTest, CallTypeIdsInBitcode) {
  TargetTransformInfo TTI(
      std::make_unique<CallTypeIdTTIImpl>(M->getDataLayout()));
  ModuleSummaryIndex Index = buildIndex(&TTI);

  SmallVector<char, 0> Buffer;
  raw_svector_ostream OS(Buffer);
  WriteBitcodeToFile(*M, OS, /*ShouldPreserveUseListOrder=*/false, &Index);
  Expected<std::unique_ptr<ModuleSummaryIndex>> ReadIndex =
      getModuleSummaryIndex(
          MemoryBufferRef(StringRef(Buffer.data(), Buffer.size()), "test"));
  ASSERT_THAT_EXPECTED(ReadIndex, Succeeded());

  for (const Function &F : *M) {
    const CallTypeIdInfo *Written = getCallTypeIdInfo(Index, F.getName());
    const CallTypeIdInfo *Read = getCallTypeIdInfo(**ReadIndex, F.getName());
    ASSERT_NE(Written, nullptr);
    ASSERT_NE(Read, nullptr);
    EXPECT_EQ(Read->CallTypeId, Written->CallTypeId) << F.getName();
    EXPECT_EQ(Read->IndirectCallTypeIds, Written->IndirectCallTypeIds)
        << F.getName();
    EXPECT_EQ(Read->HasCallWithoutTypeId, Written->HasCallWithoutTypeId)
        << F.getName();
  }
}

TEST_F(ModuleSummaryAnalysisTest, NoCallTypeIdsByDefault) {
  // The default target does not use call type IDs, and neither does a summary
  // without a TargetTransformInfo.
  TargetTransformInfo DefaultTTI(M->getDataLayout());
  ModuleSummaryIndex WithDefaultTTI = buildIndex(&DefaultTTI);
  ModuleSummaryIndex WithoutTTI = buildIndex(nullptr);
  for (const Function &F : *M) {
    EXPECT_EQ(getCallTypeIdInfo(WithDefaultTTI, F.getName()), nullptr)
        << F.getName();
    EXPECT_EQ(getCallTypeIdInfo(WithoutTTI, F.getName()), nullptr)
        << F.getName();
  }
}

} // namespace
