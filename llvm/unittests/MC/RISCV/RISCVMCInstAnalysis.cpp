//===- RISCVMCInstAnalysis.cpp - Unit tests for RISCV MCInstrAnalysis ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"
#include <cstdint>
#include <memory>

using namespace llvm;

namespace {

struct TestContext {
  const char *TripleName = "riscv32-unknown-elf";
  std::unique_ptr<MCRegisterInfo> MRI;
  std::unique_ptr<MCAsmInfo> MAI;
  std::unique_ptr<MCInstrInfo> MII;
  std::unique_ptr<MCInstrAnalysis> MIA;
  std::unique_ptr<MCContext> Ctx;
  std::unique_ptr<const MCSubtargetInfo> STI;

  TestContext(const char *TripleName) {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllDisassemblers();

    std::string Error;
    const Target *TheTarget = TargetRegistry::lookupTarget(TripleName, Error);
    if (!TheTarget)
      return;

    MRI.reset(TheTarget->createMCRegInfo(TripleName));
    MII.reset(TheTarget->createMCInstrInfo());
    MIA.reset(TheTarget->createMCInstrAnalysis(MII.get()));
    MCTargetOptions MCOptions;
    MAI.reset(TheTarget->createMCAsmInfo(*MRI, TripleName, MCOptions));
    Ctx = std::make_unique<MCContext>(Triple(TripleName), MAI.get(), MRI.get(),
                                      /*MSTI=*/nullptr);
    const char *MCPU = "generic";
    std::string Features = ""; // No extensions, just RV32I
    STI.reset(TheTarget->createMCSubtargetInfo(TripleName, MCPU, Features));
  }

  operator bool() { return Ctx.get() && MIA.get(); }
  operator MCContext &() { return *Ctx; }
  MCInstrAnalysis &getInstrAnalysis() { return *MIA; }
};

TestContext &getTestContext() {
  static TestContext Ctx;
  return Ctx;
}

} // end anonymous namespace

// Helper to create an MCInst with register and immediate operands
static MCInst makeInst(unsigned Opcode, std::initializer_list<MCOperand> Ops) {
  MCInst Inst;
  Inst.setOpcode(Opcode);
  for (const auto &Op : Ops)
    Inst.addOperand(Op);
  return Inst;
}

TEST(RISCVMCInstrAnalysis, FindTargetAddressWithRegisterState) {
  if (!getTestContext())
    GTEST_SKIP();
  auto &MIA = getTestContext().getInstrAnalysis();

  // Set up variables.
  uint64_t Addr = 0x2, Target;
  uint32_t mask = ~(0);      // All bits set to 1.
  uint64_t lowerImm = 0xfff; // Lower 12 bits set to 1.
  bool found;

  // ------------ Test 1 --------------
  // Verifies accuracy of the result when ADDI (and related instructions, see
  // switch case in RISCVTargetDesc.cpp) is involved in an instruction sequence.
  // ADDI only retains the bottom XLEN bits, making it dependent on the target.
  // This masking is especially apparent in 32-bit targets as overflow must
  // correctly be discarded.
  uint64_t upperImm1 = 0 | mask;
  upperImm1 >>= 12; // Bottom 20 bits set to 1.

  // 1. AUIPC x5, 0xFFFFF.
  MCInst auipc = makeInst(
      /*AUIPC=*/23, {MCOperand::createReg(5), MCOperand::createImm(upperImm1)});
  // 2. ADDI x5, x5, 0xFFF.
  MCInst addi =
      makeInst(/*ADDI=*/13, {MCOperand::createReg(5), MCOperand::createReg(5),
                             MCOperand::createImm(lowerImm)});

  MIA.updateState(auipc, Addr);
  found = MIA.findTargetAddress(addi, Addr, 4, Target, nullptr);
  EXPECT_TRUE(found);

  uint64_t expected1 =
      ((upperImm1 << 12) + lowerImm + Addr) & maskTrailingOnes<uint64_t>(32);
  EXPECT_EQ(Target, expected1);
  MIA.resetState();

  // ------------------- Test 2 -----------------
  // Verifies the correctness when ADDIW is involved in a sequence. ADDIW
  // performs a sign extension up to 64 bits based on the 32nd bit after the
  // result has been computed.
  mask >>= 1; // Bottom 31 bits set to 1.

  uint64_t upper_bits2 = 0 | mask;
  upper_bits2 >>= 12; // Bottom 19 bits set to 1.

  // 1. AUIPC x6, 0x7FFFF.
  MCInst auipc2 = makeInst(/*AUIPC=*/23, {MCOperand::createReg(6),
                                          MCOperand::createImm(upper_bits2)});
  // 2. ADDIW x6, x6, 0xFFF.
  MCInst addiw =
      makeInst(/*ADDIW=*/15, {MCOperand::createReg(6), MCOperand::createReg(6),
                              MCOperand::createImm(lowerImm)});

  MIA.updateState(auipc2, Addr);
  found = MIA.findTargetAddress(addiw, Addr, 4, Target, nullptr);
  EXPECT_TRUE(found);

  uint64_t expected2 = SignExtend64((upper_bits2 << 12) + lowerImm + Addr, 32);
  EXPECT_EQ(Target, expected2);
  MIA.resetState();

  // All other instructions simply add the immediate to the value loaded in the
  // corresponding register by the previous instruction in the sequence. This
  // correctness check is implicitly performed in the tests above. For more
  // tests related to this feature, check
  // llvm/test/tools/llvm-objdump/RISCV/riscv64-ar-coverage.s
}
