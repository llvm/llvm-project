//===- SystemZMCDisassemblerTest.cpp - Tests for SystemZ MCDisassembler ---===//
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
  static constexpr char TripleName[] = "systemz-unknown";
  Triple TT;
  std::unique_ptr<MCRegisterInfo> MRI;
  std::unique_ptr<MCAsmInfo> MAI;
  std::unique_ptr<MCContext> Ctx;
  std::unique_ptr<MCSubtargetInfo> STI;
  std::unique_ptr<MCDisassembler> DisAsm;

  Context() : TT(TripleName) {
    LLVMInitializeSystemZTargetInfo();
    LLVMInitializeSystemZTargetMC();
    LLVMInitializeSystemZDisassembler();

    // If we didn't build SystemZ, do not run the test.
    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget(TripleName, Error);
    if (!TheTarget)
      return;

    MRI.reset(TheTarget->createMCRegInfo(TT));
    MAI.reset(TheTarget->createMCAsmInfo(*MRI, TT, MCTargetOptions()));
    STI.reset(TheTarget->createMCSubtargetInfo(TT, "", ""));
    Ctx = std::make_unique<MCContext>(TT, MAI.get(), MRI.get(), STI.get());

    DisAsm.reset(TheTarget->createMCDisassembler(*STI, *Ctx));
  }

  operator MCContext &() { return *Ctx; };
};

Context &getContext() {
  static Context Ctxt;
  return Ctxt;
}

class SystemZMCSymbolizerTest : public MCSymbolizer {
public:
  SystemZMCSymbolizerTest(MCContext &MC) : MCSymbolizer(MC, nullptr) {}
  ~SystemZMCSymbolizerTest() {}

  bool tryAddingSymbolicOperand([[maybe_unused]] MCInst &Inst,
                                [[maybe_unused]] raw_ostream &CStream,
                                [[maybe_unused]] int64_t Value,
                                [[maybe_unused]] uint64_t Address,
                                [[maybe_unused]] bool IsBranch,
                                [[maybe_unused]] uint64_t Offset,
                                [[maybe_unused]] uint64_t OpSize,
                                [[maybe_unused]] uint64_t InstSize) override {
    return true;
  }

  void
  tryAddingPcLoadReferenceComment([[maybe_unused]] raw_ostream &cStream,
                                  [[maybe_unused]] int64_t Value,
                                  [[maybe_unused]] uint64_t Address) override {}
};

} // namespace

TEST(SystemZDisassembler, SystemZMCSymbolizerTest) {
  SystemZMCSymbolizerTest *TestSymbolizer =
      new SystemZMCSymbolizerTest(getContext());
  getContext().DisAsm->setSymbolizer(
      std::unique_ptr<MCSymbolizer>(TestSymbolizer));

  MCInst Inst;
  uint64_t InstSize;

  // Check that the SystemZ disassembler sets the comment stream before calling
  // MCDisassembler::tryAddingSymbolicOperand. This will fail an assert if it
  // does not do that.
  MCDisassembler::DecodeStatus Status = getContext().DisAsm->getInstruction(
      Inst, InstSize,
      // lgrl   %r1, 0x1234
      {0xc4, 0x18, 0x00, 0x00, 0x9a, 0x1a}, 0, nulls());
  ASSERT_TRUE(Status == MCDisassembler::Success);
  EXPECT_EQ(InstSize, uint64_t{6});
}
