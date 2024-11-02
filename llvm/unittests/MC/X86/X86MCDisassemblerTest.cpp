//===- X86MCDisassemblerTest.cpp - Tests for X86 MCDisassembler -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCDisassembler/MCSymbolizer.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

struct Context {
  const char *TripleName = "x86_64-unknown-elf";
  std::unique_ptr<MCRegisterInfo> MRI;
  std::unique_ptr<MCAsmInfo> MAI;
  std::unique_ptr<MCContext> Ctx;
  std::unique_ptr<MCSubtargetInfo> STI;
  std::unique_ptr<MCDisassembler> DisAsm;

  Context() {
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86TargetMC();
    LLVMInitializeX86Disassembler();

    // If we didn't build x86, do not run the test.
    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget(TripleName, Error);
    if (!TheTarget)
      return;

    MRI.reset(TheTarget->createMCRegInfo(TripleName));
    MAI.reset(TheTarget->createMCAsmInfo(*MRI, TripleName, MCTargetOptions()));
    STI.reset(TheTarget->createMCSubtargetInfo(TripleName, "", ""));
    Ctx = std::make_unique<MCContext>(Triple(TripleName), MAI.get(), MRI.get(),
                                      STI.get());

    DisAsm.reset(TheTarget->createMCDisassembler(*STI, *Ctx));
  }

  operator MCContext &() { return *Ctx; };
};

Context &getContext() {
  static Context Ctxt;
  return Ctxt;
}

class X86MCSymbolizerTest : public MCSymbolizer {
public:
  X86MCSymbolizerTest(MCContext &MC) : MCSymbolizer(MC, nullptr) {}
  ~X86MCSymbolizerTest() {}

  struct OpInfo {
    int64_t Value = 0;
    uint64_t Offset = 0;
    uint64_t Size;
  };
  std::vector<OpInfo> Operands;
  uint64_t InstructionSize = 0;

  void reset() {
    Operands.clear();
    InstructionSize = 0;
  }

  bool tryAddingSymbolicOperand(MCInst &Inst, raw_ostream &CStream,
                                int64_t Value, uint64_t Address, bool IsBranch,
                                uint64_t Offset, uint64_t OpSize,
                                uint64_t InstSize) override {
    Operands.push_back({Value, Offset, OpSize});
    InstructionSize = InstSize;
    return false;
  }

  void tryAddingPcLoadReferenceComment(raw_ostream &cStream, int64_t Value,
                                       uint64_t Address) override {}
};

} // namespace

TEST(X86Disassembler, X86MCSymbolizerTest) {
  X86MCSymbolizerTest *TestSymbolizer = new X86MCSymbolizerTest(getContext());
  getContext().DisAsm->setSymbolizer(
      std::unique_ptr<MCSymbolizer>(TestSymbolizer));

  MCDisassembler::DecodeStatus Status;
  MCInst Inst;
  uint64_t InstSize;

  auto checkBytes = [&](ArrayRef<uint8_t> Bytes) {
    TestSymbolizer->reset();
    Status =
        getContext().DisAsm->getInstruction(Inst, InstSize, Bytes, 0, nulls());
    ASSERT_TRUE(Status == MCDisassembler::Success);
    EXPECT_EQ(TestSymbolizer->InstructionSize, InstSize);
  };

  auto checkOperand = [&](size_t OpNo, int64_t Value, uint64_t Offset,
                          uint64_t Size) {
    ASSERT_TRUE(TestSymbolizer->Operands.size() > OpNo);
    EXPECT_EQ(TestSymbolizer->Operands[OpNo].Value, Value);
    EXPECT_EQ(TestSymbolizer->Operands[OpNo].Offset, Offset);
    EXPECT_EQ(TestSymbolizer->Operands[OpNo].Size, Size);
  };

  // movq    $0x80000, 0x80000
  checkBytes(
      {0x48, 0xc7, 0x04, 0x25, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x08, 0x00});
  checkOperand(0, 0x80000, 4, 4);
  checkOperand(1, 0x80000, 8, 4);

  // movq   $0x2a, 0x123(%rax,%r14,8)
  checkBytes(
      {0x4a, 0xc7, 0x84, 0xf0, 0x23, 0x01, 0x00, 0x00, 0x2a, 0x00, 0x00, 0x00});
  checkOperand(0, 291, 4, 4);
  checkOperand(1, 42, 8, 4);

  // movq   $0xffffffffffffefe8, -0x1(%rip)
  // Test that the value of the rip-relative operand is set correctly.
  // The instruction address is 0 and the size is 12 bytes.
  checkBytes(
      {0x48, 0xc7, 0x05, 0xff, 0xff, 0xff, 0xff, 0xe8, 0xef, 0xff, 0xff});
  checkOperand(0, /*next instr address*/ 11 - /*disp*/ 1, 3, 4);
  checkOperand(1, 0xffffffffffffefe8, 7, 4);

  // movq   $0xfffffffffffffef5, (%r12)
  // Test that the displacement operand has a size of 0, since it is not
  // explicitly specified in the instruction.
  checkBytes({0x49, 0xc7, 0x04, 0x24, 0xf5, 0xfe, 0xff, 0xff});
  checkOperand(0, 0, 4, 0);
  checkOperand(1, 0xfffffffffffffef5, 4, 4);
}
