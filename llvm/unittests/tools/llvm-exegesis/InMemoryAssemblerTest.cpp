//===-- InMemoryAssemblerTest.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InMemoryAssembler.h"
#include "X86InstrInfo.h"
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
using llvm::X86::EAX;
using llvm::X86::MOV32ri;
using llvm::X86::MOV64ri32;
using llvm::X86::RAX;
using llvm::X86::XOR32rr;
using testing::ElementsAre;

class MachineFunctionGeneratorTest : public ::testing::Test {
protected:
  MachineFunctionGeneratorTest()
      : TT(llvm::sys::getProcessTriple()),
        CpuName(llvm::sys::getHostCPUName().str()) {}

  static void SetUpTestCase() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
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

TEST_F(MachineFunctionGeneratorTest, DISABLED_JitFunction) {
  JitFunctionContext Context(createTargetMachine());
  JitFunction Function(std::move(Context), {});
  ASSERT_THAT(Function.getFunctionBytes().str(), ElementsAre(0xc3));
  // FIXME: Check that the function runs without errors. Right now this is
  // disabled because it fails on some bots.
  // Function();
}

TEST_F(MachineFunctionGeneratorTest, DISABLED_JitFunctionXOR32rr) {
  JitFunctionContext Context(createTargetMachine());
  JitFunction Function(
      std::move(Context),
      {MCInstBuilder(XOR32rr).addReg(EAX).addReg(EAX).addReg(EAX)});
  ASSERT_THAT(Function.getFunctionBytes().str(), ElementsAre(0x31, 0xc0, 0xc3));
  // Function();
}

TEST_F(MachineFunctionGeneratorTest, DISABLED_JitFunctionMOV64ri) {
  JitFunctionContext Context(createTargetMachine());
  JitFunction Function(std::move(Context),
                       {MCInstBuilder(MOV64ri32).addReg(RAX).addImm(42)});
  ASSERT_THAT(Function.getFunctionBytes().str(),
              ElementsAre(0x48, 0xc7, 0xc0, 0x2a, 0x00, 0x00, 0x00, 0xc3));
  // Function();
}

TEST_F(MachineFunctionGeneratorTest, DISABLED_JitFunctionMOV32ri) {
  JitFunctionContext Context(createTargetMachine());
  JitFunction Function(std::move(Context),
                       {MCInstBuilder(MOV32ri).addReg(EAX).addImm(42)});
  ASSERT_THAT(Function.getFunctionBytes().str(),
              ElementsAre(0xb8, 0x2a, 0x00, 0x00, 0x00, 0xc3));
  // Function();
}

} // namespace
} // namespace exegesis
