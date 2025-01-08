//===- llvm/unittest/CodeGen/AMDGPUMetadataTest.cpp -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Test that amdgpu metadata that is added in a pass is read by the asm emitter
/// and stored in the ELF.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

namespace llvm {

namespace {
// Pass that adds global metadata
struct AddMetadataPass : public ModulePass {
  std::string PalMDString;

public:
  static char ID;
  AddMetadataPass(std::string PalMDString)
      : ModulePass(ID), PalMDString(PalMDString) {}
  bool runOnModule(Module &M) override {
    auto &Ctx = M.getContext();
    auto *MD = M.getOrInsertNamedMetadata("amdgpu.pal.metadata.msgpack");
    auto *PalMD = MDString::get(Ctx, PalMDString);
    auto *TMD = MDTuple::get(Ctx, {PalMD});
    MD->addOperand(TMD);
    return true;
  }
};
char AddMetadataPass::ID = 0;
} // end anonymous namespace

class AMDGPUSelectionDAGTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    InitializeAllTargets();
    InitializeAllTargetMCs();
  }

  void SetUp() override {
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget("amdgcn--amdpal", Error);
    if (!T)
      GTEST_SKIP();

    TargetOptions Options;
    TM = std::unique_ptr<TargetMachine>(T->createTargetMachine(
        "amdgcn--amdpal", "gfx1010", "", Options, std::nullopt));
    if (!TM)
      GTEST_SKIP();

    LLVMContext Context;
    std::unique_ptr<Module> M(new Module("TestModule", Context));
    M->setDataLayout(TM->createDataLayout());

    legacy::PassManager PM;
    PM.add(new AddMetadataPass(PalMDString));
    raw_svector_ostream OutStream(Elf);
    if (TM->addPassesToEmitFile(PM, OutStream, nullptr,
                                CodeGenFileType::ObjectFile))
      report_fatal_error("Target machine cannot emit a file of this type");

    PM.run(*M);
  }

  static std::string PalMDString;

  LLVMContext Context;
  std::unique_ptr<TargetMachine> TM;
  std::unique_ptr<Module> M;
  SmallString<1024> Elf;
};
std::string AMDGPUSelectionDAGTest::PalMDString =
    "\x81\xB0"
    "amdpal.pipelines\x91\x81\xA4.api\xA6Vulkan";

TEST_F(AMDGPUSelectionDAGTest, checkMetadata) {
  // Check that the string is contained in the ELF
  EXPECT_NE(Elf.find("Vulkan"), std::string::npos);
}

} // end namespace llvm
