//===- GIMatchTableExecutorTest.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutor.h"
#include "GISelMITest.h"
#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutorImpl.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(GlobalISelLEB128Test, fastDecodeULEB128) {
#define EXPECT_DECODE_ULEB128_EQ(EXPECTED, VALUE)                              \
  do {                                                                         \
    uint64_t ActualSize = 0;                                                   \
    uint64_t Actual = GIMatchTableExecutor::fastDecodeULEB128(                 \
        reinterpret_cast<const uint8_t *>(VALUE), ActualSize);                 \
    EXPECT_EQ(sizeof(VALUE) - 1, ActualSize);                                  \
    EXPECT_EQ(EXPECTED, Actual);                                               \
  } while (0)

  EXPECT_DECODE_ULEB128_EQ(0u, "\x00");
  EXPECT_DECODE_ULEB128_EQ(1u, "\x01");
  EXPECT_DECODE_ULEB128_EQ(63u, "\x3f");
  EXPECT_DECODE_ULEB128_EQ(64u, "\x40");
  EXPECT_DECODE_ULEB128_EQ(0x7fu, "\x7f");
  EXPECT_DECODE_ULEB128_EQ(0x80u, "\x80\x01");
  EXPECT_DECODE_ULEB128_EQ(0x81u, "\x81\x01");
  EXPECT_DECODE_ULEB128_EQ(0x90u, "\x90\x01");
  EXPECT_DECODE_ULEB128_EQ(0xffu, "\xff\x01");
  EXPECT_DECODE_ULEB128_EQ(0x100u, "\x80\x02");
  EXPECT_DECODE_ULEB128_EQ(0x101u, "\x81\x02");
  EXPECT_DECODE_ULEB128_EQ(4294975616ULL, "\x80\xc1\x80\x80\x10");

  // Decode ULEB128 with extra padding bytes
  EXPECT_DECODE_ULEB128_EQ(0u, "\x80\x00");
  EXPECT_DECODE_ULEB128_EQ(0u, "\x80\x80\x00");
  EXPECT_DECODE_ULEB128_EQ(0x7fu, "\xff\x00");
  EXPECT_DECODE_ULEB128_EQ(0x7fu, "\xff\x80\x00");
  EXPECT_DECODE_ULEB128_EQ(0x80u, "\x80\x81\x00");
  EXPECT_DECODE_ULEB128_EQ(0x80u, "\x80\x81\x80\x00");
  EXPECT_DECODE_ULEB128_EQ(0x80u, "\x80\x81\x80\x80\x80\x80\x80\x80\x80\x00");
  EXPECT_DECODE_ULEB128_EQ(0x80000000'00000000ul,
                           "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x01");

#undef EXPECT_DECODE_ULEB128_EQ
}

namespace {

/// Runs the match table that GIReplaceRegWithConstant/FConstant emits on a
/// G_IMPLICIT_DEF of type \p Ty, and returns the register that now holds the
/// constant.
class BuildConstantInPlaceExecutor : public GIMatchTableExecutor {
public:
  using PredicateBitset = Bitset<1>;
  using ComplexMatcherMemFn = ComplexRendererFns (
      BuildConstantInPlaceExecutor::*)(MachineOperand &) const;
  using CustomRendererFn = void (BuildConstantInPlaceExecutor::*)(
      MachineInstrBuilder &, const MachineInstr &, int) const;

  static const char *getName() { return "BuildConstantInPlaceExecutor"; }
  void setupGeneratedPerFunctionState(MachineFunction &) override {}

  Register run(MachineIRBuilder &B, LLT Ty, uint8_t Opcode, uint64_t Bits) {
    auto Root = B.buildUndef(Ty);
    Register RootReg = Root.getReg(0);
    auto Copy = B.buildCopy(Ty, Root);

    SmallVector<uint8_t> Table = {Opcode, /*InsnID*/ 0, /*OpIdx*/ 0};
    Table.append(reinterpret_cast<uint8_t *>(&Bits),
                 reinterpret_cast<uint8_t *>(&Bits + 1));
    Table.push_back(GIR_EraseRootFromParent_Done);

    ExecInfoTy<PredicateBitset, ComplexMatcherMemFn, CustomRendererFn> ExecInfo(
        &Ty, 1, nullptr, nullptr, nullptr);
    MatcherState State(/*MaxRenderers=*/0);
    State.MIs.push_back(Root);
    const TargetSubtargetInfo &STI = B.getMF().getSubtarget();
    EXPECT_TRUE(executeMatchTable(*this, State, ExecInfo, B, Table.data(),
                                  *STI.getInstrInfo(), *B.getMRI(),
                                  *STI.getRegisterInfo(), *STI.getRegBankInfo(),
                                  PredicateBitset(), /*CoverageInfo=*/nullptr));

    // The constant must define the root's register rather than a new one.
    Register Reg = Copy->getOperand(1).getReg();
    EXPECT_EQ(Reg, RootReg);
    EXPECT_TRUE(B.getMRI()->hasOneDef(Reg));
    return Reg;
  }

  APFloat runFP(MachineIRBuilder &B, LLT Ty, uint64_t Bits) {
    Register Reg = run(B, Ty, GIR_BuildFConstantInPlace, Bits);
    // Read the immediate directly: the look-through constant helpers rebuild
    // the APFloat from its bit width, which turns bf16 into half.
    return getConstantFPVRegVal(Reg, *B.getMRI())->getValueAPF();
  }
};

TEST_F(AArch64GISelMITest, BuildConstantInPlace) {
  setUp();
  if (!TM)
    GTEST_SKIP();

  BuildConstantInPlaceExecutor Exec;
  Register Reg =
      Exec.run(B, LLT::scalar(32), GIR_BuildConstantInPlace, uint64_t(-1));
  EXPECT_EQ(getIConstantVRegSExtVal(Reg, *B.getMRI()), -1);

  Reg = Exec.run(B, LLT::fixed_vector(2, 32), GIR_BuildConstantInPlace, 7);
  EXPECT_EQ(getIConstantSplatSExtVal(Reg, *B.getMRI()), 7);
}

TEST_F(AArch64GISelMITest, BuildFConstantInPlaceTruncatesBits) {
  setUp();
  if (!TM)
    GTEST_SKIP();

  BuildConstantInPlaceExecutor Exec;
  // -1 is truncated to the type's width instead of asserting in APInt.
  for (LLT Ty : {LLT::float16(), LLT::float32()})
    EXPECT_TRUE(Exec.runFP(B, Ty, ~0ULL).bitcastToAPInt().isAllOnes());
}

TEST_F(AArch64GISelMITest, BuildFConstantInPlaceFPSemantics) {
  setUp();
  if (!TM)
    GTEST_SKIP();

  BuildConstantInPlaceExecutor Exec;
  // 0x3f80 is 1.0 as bf16, but 1.875 as half.
  EXPECT_TRUE(Exec.runFP(B, LLT::bfloat16(), 0x3f80).isExactlyValue(1.0));
  // s16 could be either, so only +0.0 (the same bits in both) is accepted.
  EXPECT_TRUE(Exec.runFP(B, LLT::scalar(16), 0).isPosZero());
  EXPECT_DEATH_IF_SUPPORTED(Exec.runFP(B, LLT::scalar(16), 0x3f80),
                            "does not distinguish half from bfloat");
  EXPECT_DEATH_IF_SUPPORTED(Exec.runFP(B, LLT::float128(), 1),
                            "wider than 64 bits");
}

} // namespace
