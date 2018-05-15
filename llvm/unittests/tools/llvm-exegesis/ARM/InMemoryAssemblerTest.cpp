//===-- InMemoryAssemblerTest.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InMemoryAssembler.h"
#include "ARMInstrInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <memory>

namespace exegesis {
namespace {

using llvm::MCInstBuilder;
using testing::ElementsAre;

class MachineFunctionGeneratorTest : public ::testing::Test {
protected:
  MachineFunctionGeneratorTest()
      : TT("armv7-none-linux-gnueabi"), CpuName("") {}

  static void SetUpTestCase() {
    LLVMInitializeARMTargetInfo();
    LLVMInitializeARMTargetMC();
    LLVMInitializeARMTarget();
    LLVMInitializeARMAsmPrinter();
  }

  std::unique_ptr<llvm::LLVMTargetMachine> createTargetMachine() {
    std::string Error;
    const llvm::Target *TheTarget =
        llvm::TargetRegistry::lookupTarget(TT, Error);
    assert(TheTarget);
    const llvm::TargetOptions Options;
    return std::unique_ptr<llvm::LLVMTargetMachine>(
        static_cast<llvm::LLVMTargetMachine *>(TheTarget->createTargetMachine(
            TT, CpuName, "", Options, llvm::Reloc::Model::Static)));
  }

private:
  const std::string TT;
  const std::string CpuName;
};

TEST_F(MachineFunctionGeneratorTest, JitFunction) {
  JitFunctionContext Context(createTargetMachine());
  JitFunction Function(std::move(Context), {});
  ASSERT_THAT(Function.getFunctionBytes().str(),
              ElementsAre(0x1e, 0xff, 0x2f, 0xe1));
}

TEST_F(MachineFunctionGeneratorTest, JitFunctionADDrr) {
  JitFunctionContext Context(createTargetMachine());
  JitFunction Function(std::move(Context), {MCInstBuilder(llvm::ARM::ADDrr)
                                                .addReg(llvm::ARM::R0)
                                                .addReg(llvm::ARM::R0)
                                                .addReg(llvm::ARM::R0)
                                                .addImm(llvm::ARMCC::AL)
                                                .addReg(0)
                                                .addReg(0)});
  ASSERT_THAT(Function.getFunctionBytes().str(),
              ElementsAre(0x00, 0x00, 0x80, 0xe0, 0x1e, 0xff, 0x2f, 0xe1));
}

} // namespace
} // namespace exegesis
