//===- LoongArchMCDisassemblerTest.cpp - LoongArch disassembler tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCDisassembler/MCSymbolizer.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <array>
#include <cassert>
#include <memory>
#include <string>

using namespace llvm;

namespace {

struct Context {
  const Triple TT{"loongarch64-unknown-linux-gnu"};
  std::unique_ptr<MCRegisterInfo> MRI;
  std::unique_ptr<MCAsmInfo> MAI;
  std::unique_ptr<MCContext> Ctx;
  std::unique_ptr<MCSubtargetInfo> STI;
  std::unique_ptr<MCDisassembler> DisAsm;

  Context() {
    LLVMInitializeLoongArchTargetInfo();
    LLVMInitializeLoongArchTargetMC();
    LLVMInitializeLoongArchDisassembler();

    // If we didn't build LoongArch, do not run the test.
    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);
    if (!TheTarget)
      return;

    MRI.reset(TheTarget->createMCRegInfo(TT));
    MAI.reset(TheTarget->createMCAsmInfo(*MRI, TT, MCTargetOptions()));
    STI.reset(TheTarget->createMCSubtargetInfo(TT, "", ""));
    Ctx = std::make_unique<MCContext>(TT, *MAI, *MRI, *STI);
    DisAsm.reset(TheTarget->createMCDisassembler(*STI, *Ctx));
  }

  operator MCContext &() { return *Ctx; }
};

Context &getContext() {
  static Context Ctx;
  return Ctx;
}

class LoongArchMCSymbolizerTest : public MCSymbolizer {
public:
  struct Call {
    raw_ostream *CommentStream;
    int64_t Value;
    uint64_t Address;
    bool IsBranch;
    uint64_t Offset;
    uint64_t OpSize;
    uint64_t InstSize;
  };

  explicit LoongArchMCSymbolizerTest(MCContext &Ctx)
      : MCSymbolizer(Ctx, nullptr) {}

  SmallVector<Call, 1> Calls;
  bool AddSymbolicOperand = false;

  void reset(bool AddSymbol = false) {
    Calls.clear();
    AddSymbolicOperand = AddSymbol;
  }

  bool tryAddingSymbolicOperand(MCInst &Inst, raw_ostream &CommentStream,
                                int64_t Value, uint64_t Address, bool IsBranch,
                                uint64_t Offset, uint64_t OpSize,
                                uint64_t InstSize) override {
    Calls.push_back(
        {&CommentStream, Value, Address, IsBranch, Offset, OpSize, InstSize});
    if (!AddSymbolicOperand)
      return false;

    Inst.addOperand(MCOperand::createExpr(
        MCSymbolRefExpr::create(Ctx.getOrCreateSymbol("symbol"), Ctx)));
    return true;
  }

  void tryAddingPcLoadReferenceComment(raw_ostream &, int64_t,
                                       uint64_t) override {}
};

struct ImmediateTestCase {
  std::array<uint8_t, 4> Bytes;
  int64_t SymbolicValue;
  int64_t DecodedImmediate;
  unsigned OperandIndex;
  bool IsPCRelBranch;
};

} // namespace

TEST(LoongArchDisassembler, ReportsDecodedImmediatesToSymbolizer) {
  auto *Symbolizer = new LoongArchMCSymbolizerTest(getContext());
  getContext().DisAsm->setSymbolizer(std::unique_ptr<MCSymbolizer>(Symbolizer));

  constexpr uint64_t Address = 0x1000;
  const ImmediateTestCase TestCases[] = {
      // ori $a0, $a1, 0x345
      {{0xa4, 0x14, 0x8d, 0x03}, 0x345, 0x345, 2, false},
      // pcalau12i $a0, -1
      {{0xe4, 0xff, 0xff, 0x1b}, -1, -1, 1, false},
      // beq $a0, $a1, 8
      {{0x85, 0x08, 0x00, 0x58}, 0x1008, 8, 2, true},
      // beqz $a0, 8
      {{0x80, 0x08, 0x00, 0x40}, 0x1008, 8, 1, true},
      // bceqz $fcc0, 8
      {{0x00, 0x08, 0x00, 0x48}, 0x1008, 8, 1, true},
      // bcnez $fcc1, -4
      {{0x3f, 0xfd, 0xff, 0x4b}, 0x0ffc, -4, 1, true},
      // b 8
      {{0x00, 0x08, 0x00, 0x50}, 0x1008, 8, 0, true},
      // bl -4
      {{0xff, 0xff, 0xff, 0x57}, 0x0ffc, -4, 0, true},
      // jirl $ra, $a0, 8
      {{0x81, 0x08, 0x00, 0x4c}, 8, 8, 2, false},
  };

  std::string CommentStorage;
  raw_string_ostream CommentStream(CommentStorage);

  for (const ImmediateTestCase &TestCase : TestCases) {
    Symbolizer->reset();

    MCInst Inst;
    uint64_t InstSize;
    const MCDisassembler::DecodeStatus Status =
        getContext().DisAsm->getInstruction(Inst, InstSize, TestCase.Bytes,
                                            Address, CommentStream);
    ASSERT_EQ(Status, MCDisassembler::Success);
    ASSERT_EQ(InstSize, 4u);

    ASSERT_EQ(Symbolizer->Calls.size(), 1u);
    const LoongArchMCSymbolizerTest::Call &Call = Symbolizer->Calls.front();
    EXPECT_EQ(Call.CommentStream, &CommentStream);
    EXPECT_EQ(Call.Value, TestCase.SymbolicValue);
    EXPECT_EQ(Call.Address, Address);
    EXPECT_EQ(Call.IsBranch, TestCase.IsPCRelBranch);
    EXPECT_EQ(Call.Offset, 0u);
    EXPECT_EQ(Call.OpSize, 0u);
    EXPECT_EQ(Call.InstSize, 4u);

    ASSERT_GT(Inst.getNumOperands(), TestCase.OperandIndex);
    const MCOperand &Operand = Inst.getOperand(TestCase.OperandIndex);
    ASSERT_TRUE(Operand.isImm());
    EXPECT_EQ(Operand.getImm(), TestCase.DecodedImmediate);
  }
}

TEST(LoongArchDisassembler, UsesSymbolicImmediateFromSymbolizer) {
  auto *Symbolizer = new LoongArchMCSymbolizerTest(getContext());
  getContext().DisAsm->setSymbolizer(std::unique_ptr<MCSymbolizer>(Symbolizer));
  Symbolizer->reset(/*AddSymbol=*/true);

  MCInst Inst;
  uint64_t InstSize;
  const MCDisassembler::DecodeStatus Status =
      getContext().DisAsm->getInstruction(Inst, InstSize,
                                          // pcalau12i $a0, 0
                                          {0x04, 0x00, 0x00, 0x1a}, 0x1000,
                                          nulls());

  ASSERT_EQ(Status, MCDisassembler::Success);
  ASSERT_EQ(InstSize, 4u);
  ASSERT_EQ(Symbolizer->Calls.size(), 1u);
  ASSERT_EQ(Inst.getNumOperands(), 2u);
  EXPECT_TRUE(Inst.getOperand(1).isExpr());
}
