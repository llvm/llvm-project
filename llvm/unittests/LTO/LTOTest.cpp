//===- llvm/unittest/LTO/LTOTest.cpp - Unit tests for LTO -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/LTO/LTO.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace llvm;
using ::testing::_;
using ::testing::AtLeast;
using ::testing::Return;

namespace {

class LTOTest : public ::testing::Test {
protected:
  LLVMContext Context;
  std::string TripleName;
  std::unique_ptr<TargetMachine> TM;

  std::unique_ptr<Module> createEmptyModule();

public:
  static void SetUpTestSuite();
  void SetUp() override;
};

void LTOTest::SetUpTestSuite() {
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
}

void LTOTest::SetUp() {
  TripleName = Triple::normalize(sys::getDefaultTargetTriple());
  std::string Error;
  const auto *TheTarget = TargetRegistry::lookupTarget(TripleName, Error);
  if (!TheTarget)
    GTEST_SKIP();
  TM.reset(TheTarget->createTargetMachine(TripleName, "", "", TargetOptions(),
                                          std::nullopt));
  if (!TM)
    GTEST_SKIP();
}

std::unique_ptr<Module> LTOTest::createEmptyModule() {
  auto M = std::make_unique<Module>("Empty", Context);
  M->setTargetTriple(TripleName);
  M->setDataLayout(TM->createDataLayout());
  return M;
}

static std::unique_ptr<MemoryBuffer>
writeBitcodeToMemoryBuffer(const Module &M) {
  SmallString<0> Buffer;
  raw_svector_ostream OS(Buffer);
  WriteBitcodeToFile(M, OS);
  return std::make_unique<SmallVectorMemoryBuffer>(std::move(Buffer));
}

static void
addMemBufToLto(lto::LTO &Lto, MemoryBufferRef MB,
               ArrayRef<lto::SymbolResolution> SymRes = std::nullopt) {
  auto InputOrError = lto::InputFile::create(MB);
  ASSERT_TRUE(!!InputOrError) << toString(InputOrError.takeError());
  auto AddResult = Lto.add(std::move(*InputOrError), SymRes);
  ASSERT_TRUE(!AddResult) << toString(std::move(AddResult));
}

static void runLto(lto::LTO &Lto) {
  std::vector<SmallString<0>> AddStreamBufs;
  auto AddStreamFn = [&AddStreamBufs](size_t task,
                                      const Twine & /*moduleName*/) {
    return std::make_unique<CachedFileStream>(
        std::make_unique<raw_svector_ostream>(AddStreamBufs[task]));
  };
  AddStreamBufs.resize(Lto.getMaxTasks());
  auto RunResult = Lto.run(AddStreamFn);
  ASSERT_TRUE(!RunResult) << toString(std::move(RunResult));
}

struct MockBeforePassFunc {
  MOCK_METHOD(bool, Op, (StringRef, Any));
  bool operator()(StringRef Name, Any IR) { return Op(Name, IR); }
};

TEST_F(LTOTest, PassInstrumentationHook) {
  MockBeforePassFunc MBPF;
  EXPECT_CALL(MBPF, Op(_, _)).Times(AtLeast(1));

  lto::Config LtoConfig;
  LtoConfig.PassInstrumentationHook = [&](PassInstrumentationCallbacks &PIC) {
    PIC.registerShouldRunOptionalPassCallback(std::ref(MBPF));
  };
  lto::LTO LtoTest(std::move(LtoConfig));

  auto Module = createEmptyModule();
  auto ModuleMemBuf = writeBitcodeToMemoryBuffer(*Module);
  ASSERT_NO_FATAL_FAILURE(
      addMemBufToLto(LtoTest, ModuleMemBuf->getMemBufferRef()));
  ASSERT_NO_FATAL_FAILURE(runLto(LtoTest));
}

} // end anonymous namespace
