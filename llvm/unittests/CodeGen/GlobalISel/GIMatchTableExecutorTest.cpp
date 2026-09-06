//===- GIMatchTableExecutorTest.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutor.h"
#include "GISelMITest.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutorImpl.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class TestGIMatchTableExecutor : public Combiner {
public:
  TestGIMatchTableExecutor(MachineFunction &MF, const CombinerInfo &CInfo)
      : Combiner(MF, CInfo, /*VT*/ nullptr) {}

  static const char *getName() { return "test-gimatchtable-executor"; }

  void setupGeneratedPerFunctionState(MachineFunction &MF) override {}
  bool tryCombineAll(MachineInstr &I) const override { return false; }

  bool runCustomAction(unsigned FnID, const MatcherState &State,
                       NewMIVector &OutMIs) const override {
    assert((FnID == 1 || FnID == 2) && "Expected a valid FnID");
    MachineInstr &Root = *State.MIs[0];
    MachineBasicBlock &MBB = *Root.getParent();
    MachineInstrBuilder MIB =
        BuildMI(MBB, Root.getIterator(), Root.getDebugLoc(),
                Root.getMF()->getSubtarget().getInstrInfo()->get(
                    TargetOpcode::G_SUB))
            .addDef(Root.getOperand(0).getReg())
            .addUse(Root.getOperand(1).getReg())
            .addUse(Root.getOperand(2).getReg());
    if (FnID == 1)
      MIB.setMIFlags(MachineInstr::NoSWrap);
    OutMIs.push_back(MIB);
    return true;
  }

  bool run(ArrayRef<MachineInstr *> MIs, MachineIRBuilder &Builder,
           const uint8_t *MatchTable, const TargetInstrInfo &TII,
           MachineRegisterInfo &MRI, const TargetRegisterInfo &TRI,
           const RegisterBankInfo &RBI) const {
    MatcherState State(/*MaxRenderers=*/0);
    State.MIs.append(MIs.begin(), MIs.end());
    using PredicateBitset = Bitset<1>;
    using ComplexMatcherMemFn =
        ComplexRendererFns (TestGIMatchTableExecutor::*)(MachineOperand &)
            const;
    using CustomRendererFn = void (TestGIMatchTableExecutor::*)(
        MachineInstrBuilder &, const MachineInstr &, int64_t) const;
    const ExecInfoTy<PredicateBitset, ComplexMatcherMemFn, CustomRendererFn>
        ExecInfo(nullptr, 0, nullptr, nullptr, nullptr);
    PredicateBitset AvailableFeatures;
    return executeMatchTable(*const_cast<TestGIMatchTableExecutor *>(this),
                             State, ExecInfo, Builder, MatchTable, TII, MRI,
                             TRI, RBI, AvailableFeatures, nullptr);
  }

  bool run(MachineInstr &Root, MachineIRBuilder &Builder,
           const uint8_t *MatchTable, const TargetInstrInfo &TII,
           MachineRegisterInfo &MRI, const TargetRegisterInfo &TRI,
           const RegisterBankInfo &RBI) const {
    MachineInstr *RootMI = &Root;
    return run(ArrayRef<MachineInstr *>(RootMI), Builder, MatchTable, TII, MRI,
               TRI, RBI);
  }
};

static void appendU16(SmallVectorImpl<uint8_t> &Table, uint16_t Value) {
  Table.push_back(Value & 0xff);
  Table.push_back((Value >> 8) & 0xff);
}

static void appendU32(SmallVectorImpl<uint8_t> &Table, uint32_t Value) {
  Table.push_back(Value & 0xff);
  Table.push_back((Value >> 8) & 0xff);
  Table.push_back((Value >> 16) & 0xff);
  Table.push_back((Value >> 24) & 0xff);
}

static CombinerInfo getTestCombinerInfo() {
  return CombinerInfo(/*AllowIllegalOps*/ true, /*ShouldLegalizeIllegal*/ false,
                      /*LInfo*/ nullptr, /*OptEnabled*/ true,
                      /*OptSize*/ false, /*MinSize*/ false);
}

} // namespace

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

TEST_F(AArch64GISelMITest, MatchTableExplicitMIFlagsOverrideDefaultPoisonDrop) {
  setUp("");
  if (!TM)
    GTEST_SKIP();

  Register RootReg = MRI->createGenericVirtualRegister(LLT::scalar(64));
  auto Root =
      B.buildInstr(TargetOpcode::G_ADD, {RootReg}, {Copies[1], Copies[2]});
  Root->setFlags(MachineInstr::NoUWrap | MachineInstr::NoSWrap);
  CombinerInfo CInfo = getTestCombinerInfo();
  TestGIMatchTableExecutor Executor(*MF, CInfo);
  Executor.setupMF(*MF, nullptr);

  SmallVector<uint8_t, 32> MatchTable;
  MatchTable.push_back(GIR_BuildMI);
  MatchTable.push_back(0); // InsnID
  appendU16(MatchTable, TargetOpcode::G_SUB);
  MatchTable.push_back(GIR_BuildMI);
  MatchTable.push_back(1); // InsnID
  appendU16(MatchTable, TargetOpcode::G_MUL);
  MatchTable.push_back(GIR_SetMIFlags);
  MatchTable.push_back(0); // InsnID
  appendU32(MatchTable, MachineInstr::NoSWrap);
  MatchTable.push_back(GIR_CopyMIFlags);
  MatchTable.push_back(1); // InsnID
  MatchTable.push_back(0); // OldInsnID
  MatchTable.push_back(GIR_Done);

  const TargetSubtargetInfo &STI = MF->getSubtarget();
  EXPECT_TRUE(Executor.run(*Root, B, MatchTable.data(), *STI.getInstrInfo(),
                           *MRI, *STI.getRegisterInfo(),
                           *STI.getRegBankInfo()));

  MachineInstr *SetFlagsMI = nullptr;
  MachineInstr *CopiedFlagsMI = nullptr;
  for (MachineInstr &MI : *EntryMBB) {
    if (MI.getOpcode() == TargetOpcode::G_SUB)
      SetFlagsMI = &MI;
    if (MI.getOpcode() == TargetOpcode::G_MUL)
      CopiedFlagsMI = &MI;
  }
  ASSERT_NE(SetFlagsMI, nullptr);
  EXPECT_FALSE(SetFlagsMI->getFlag(MachineInstr::NoUWrap));
  EXPECT_TRUE(SetFlagsMI->getFlag(MachineInstr::NoSWrap));

  ASSERT_NE(CopiedFlagsMI, nullptr);
  EXPECT_TRUE(CopiedFlagsMI->getFlag(MachineInstr::NoUWrap));
  EXPECT_TRUE(CopiedFlagsMI->getFlag(MachineInstr::NoSWrap));
}

TEST_F(AArch64GISelMITest, MatchTableExplicitUnsetMIFlagsBlocksPropagation) {
  setUp("");
  if (!TM)
    GTEST_SKIP();

  Register RootReg = MRI->createGenericVirtualRegister(LLT::scalar(64));
  auto Root =
      B.buildInstr(TargetOpcode::G_ADD, {RootReg}, {Copies[1], Copies[2]});
  Root->setFlags(MachineInstr::NoUWrap | MachineInstr::NoSWrap |
                 MachineInstr::FmNsz);

  CombinerInfo CInfo = getTestCombinerInfo();
  TestGIMatchTableExecutor Executor(*MF, CInfo);
  Executor.setupMF(*MF, nullptr);

  SmallVector<uint8_t, 32> MatchTable;
  MatchTable.push_back(GIR_BuildMI);
  MatchTable.push_back(0); // InsnID
  appendU16(MatchTable, TargetOpcode::G_SUB);
  MatchTable.push_back(GIR_SetMIFlags);
  MatchTable.push_back(0); // InsnID
  appendU32(MatchTable, MachineInstr::NoUWrap | MachineInstr::FmNsz);
  MatchTable.push_back(GIR_UnsetMIFlags);
  MatchTable.push_back(0); // InsnID
  appendU32(MatchTable, MachineInstr::NoUWrap | MachineInstr::FmNsz);
  MatchTable.push_back(GIR_Done);

  const TargetSubtargetInfo &STI = MF->getSubtarget();
  EXPECT_TRUE(Executor.run(*Root, B, MatchTable.data(), *STI.getInstrInfo(),
                           *MRI, *STI.getRegisterInfo(),
                           *STI.getRegBankInfo()));

  MachineInstr *BuiltMI = nullptr;
  for (MachineInstr &MI : *EntryMBB) {
    if (MI.getOpcode() == TargetOpcode::G_SUB) {
      BuiltMI = &MI;
      break;
    }
  }
  ASSERT_NE(BuiltMI, nullptr);
  EXPECT_FALSE(BuiltMI->getFlag(MachineInstr::NoUWrap));
  EXPECT_FALSE(BuiltMI->getFlag(MachineInstr::FmNsz));
  EXPECT_FALSE(BuiltMI->getFlag(MachineInstr::NoSWrap));
}

TEST_F(AArch64GISelMITest, MatchTableCustomActionDropsRootMIFlags) {
  setUp("");
  if (!TM)
    GTEST_SKIP();

  Register RootReg = MRI->createGenericVirtualRegister(LLT::scalar(64));
  auto Root =
      B.buildInstr(TargetOpcode::G_ADD, {RootReg}, {Copies[1], Copies[2]});
  Root->setFlags(MachineInstr::NoUWrap | MachineInstr::NoSWrap);

  CombinerInfo CInfo = getTestCombinerInfo();
  TestGIMatchTableExecutor Executor(*MF, CInfo);
  Executor.setupMF(*MF, nullptr);

  SmallVector<uint8_t, 16> MatchTable;
  MatchTable.push_back(GIR_DoneWithCustomAction);
  appendU16(MatchTable, 1); // FnID

  const TargetSubtargetInfo &STI = MF->getSubtarget();
  EXPECT_TRUE(Executor.run(*Root, B, MatchTable.data(), *STI.getInstrInfo(),
                           *MRI, *STI.getRegisterInfo(),
                           *STI.getRegBankInfo()));

  MachineInstr *BuiltMI = nullptr;
  for (MachineInstr &MI : *EntryMBB) {
    if (MI.getOpcode() == TargetOpcode::G_SUB) {
      BuiltMI = &MI;
      break;
    }
  }
  ASSERT_NE(BuiltMI, nullptr);
  EXPECT_FALSE(BuiltMI->getFlag(MachineInstr::NoUWrap));
  EXPECT_TRUE(BuiltMI->getFlag(MachineInstr::NoSWrap));
}

TEST_F(AArch64GISelMITest, MatchTableCustomActionDropsUnsetRootMIFlags) {
  setUp("");
  if (!TM)
    GTEST_SKIP();

  Register RootReg = MRI->createGenericVirtualRegister(LLT::scalar(64));
  auto Root =
      B.buildInstr(TargetOpcode::G_ADD, {RootReg}, {Copies[1], Copies[2]});
  Root->setFlags(MachineInstr::NoUWrap | MachineInstr::NoSWrap);

  CombinerInfo CInfo = getTestCombinerInfo();
  TestGIMatchTableExecutor Executor(*MF, CInfo);
  Executor.setupMF(*MF, nullptr);

  SmallVector<uint8_t, 16> MatchTable;
  MatchTable.push_back(GIR_DoneWithCustomAction);
  appendU16(MatchTable, 2); // FnID

  const TargetSubtargetInfo &STI = MF->getSubtarget();
  EXPECT_TRUE(Executor.run(*Root, B, MatchTable.data(), *STI.getInstrInfo(),
                           *MRI, *STI.getRegisterInfo(),
                           *STI.getRegBankInfo()));

  MachineInstr *BuiltMI = nullptr;
  for (MachineInstr &MI : *EntryMBB) {
    if (MI.getOpcode() == TargetOpcode::G_SUB) {
      BuiltMI = &MI;
      break;
    }
  }
  ASSERT_NE(BuiltMI, nullptr);
  EXPECT_FALSE(BuiltMI->getFlag(MachineInstr::NoUWrap));
  EXPECT_FALSE(BuiltMI->getFlag(MachineInstr::NoSWrap));
}
