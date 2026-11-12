//===- RenameIndependentSubregsTest.cpp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/RenameIndependentSubregs.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

#include <memory>

using namespace llvm;

namespace {

class RenameIndependentSubregsTest : public testing::Test {
protected:
  LLVMContext Context;
  std::unique_ptr<TargetMachine> TM;
  std::unique_ptr<Module> M;
  std::unique_ptr<MachineModuleInfo> MMI;
  std::unique_ptr<MIRParser> MIR;

  LoopAnalysisManager LAM;
  MachineFunctionAnalysisManager MFAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  static void SetUpTestCase() {
    InitializeAllTargets();
    InitializeAllTargetMCs();
  }

  void SetUp() override {
    auto &RegisteredOptions = cl::getRegisteredOptions();
    auto It = RegisteredOptions.find("enable-subreg-liveness");
    if (It == RegisteredOptions.end())
      GTEST_SKIP();

    // Force-enable subreg liveness tracking, as AArch64 doesn't enable it by
    // default
    if (It->second->addOccurrence(0, It->second->ArgStr, "", false))
      GTEST_SKIP();

    Triple TT("aarch64-none-linux-gnu");
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget("", TT, Error);
    if (!T)
      GTEST_SKIP();

    TargetOptions TMOptions;
    TM.reset(
        T->createTargetMachine(TT, "generic", "", TMOptions, std::nullopt));
    if (!TM)
      GTEST_SKIP();

    MMI = std::make_unique<MachineModuleInfo>(TM.get());

    PassBuilder PB(TM.get());
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.registerMachineFunctionAnalyses(MFAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM, &MFAM);
    MAM.registerPass([&] { return MachineModuleAnalysis(*MMI); });
  }

  bool parseMIR(StringRef MIRCode) {
    std::unique_ptr<MemoryBuffer> MBuffer = MemoryBuffer::getMemBuffer(MIRCode);
    MIR = createMIRParser(std::move(MBuffer), Context);
    if (!MIR)
      return false;

    M = MIR->parseIRModule();
    if (!M)
      return false;
    M->setDataLayout(TM->createDataLayout());

    if (MIR->parseMachineFunctions(*M, MAM)) {
      M.reset();
      return false;
    }
    return true;
  }

  MachineFunction &getMF(StringRef Name) {
    return FAM.getResult<MachineFunctionAnalysis>(*M->getFunction(Name))
        .getMF();
  }

  static StringRef getTestMIR() {
    return R"MIR(
--- |
  define void @test() {
  entry:
    unreachable
  }
...
---
name:              test
tracksRegLiveness: true
isSSA:             false
noPhis:            true
registers:
  - { id: 0, class: fpr128, preferred-register: '' }
  - { id: 1, class: fpr16, preferred-register: '' }
  - { id: 3, class: gpr32common, preferred-register: '' }
  - { id: 4, class: fpr32, preferred-register: '' }
  - { id: 5, class: fpr64, preferred-register: '' }
  - { id: 6, class: fpr64, preferred-register: '' }
  - { id: 7, class: fpr128, preferred-register: '' }
body:             |
  bb.0.entry:
    successors: %bb.1(0x40000000), %bb.2(0x40000000)

    %1:fpr16 = IMPLICIT_DEF
    %3:gpr32common = COPY $wzr
    undef %0.hsub:fpr128 = COPY %1
    CBZW killed %3, %bb.2
    B %bb.1

  bb.1:
    successors: %bb.4(0x80000000)

    B %bb.4

  bb.2:
    successors: %bb.4(0x80000000), %bb.3(0x00000000)

    INLINEASM_BR &"", 1 /* sideeffect attdialect */, 6029321 /* reguse:FPR128 */, %0.hsub, 13 /* imm */, %bb.4
    B %bb.3

  bb.3:
    $q0 = IMPLICIT_DEF
    RET_ReallyLR implicit $q0

  bb.4 (machine-block-address-taken, inlineasm-br-indirect-target):
    %7:fpr128 = COPY %0
    %4:fpr32 = COPY %0.ssub
    %5:fpr64 = IMPLICIT_DEF
    undef %0.dsub:fpr128 = COPY %5
    %6:fpr64 = COPY killed %0.dsub
    $q0 = IMPLICIT_DEF
    RET_ReallyLR implicit $q0
...
)MIR";
  }

  void shapeUnsafeInterval(MachineFunction &MF, LiveIntervals &LIS) {
    auto &MRI = MF.getRegInfo();

    Register Reg0 = Register::index2VirtReg(0);
    LiveInterval &LI = LIS.getInterval(Reg0);
    LI.clear();
    LI.clearSubRanges();

    BumpPtrAllocator &Alloc = LIS.getVNInfoAllocator();
    const TargetRegisterInfo &TRI = *MRI.getTargetRegisterInfo();
    const SlotIndexes &Indexes = *LIS.getSlotIndexes();

    MachineBasicBlock &BB0 = *std::next(MF.begin(), 0);
    MachineBasicBlock &BB2 = *std::next(MF.begin(), 2);
    MachineBasicBlock &BB4 = *std::next(MF.begin(), 4);

    MachineInstr &DefH = *std::next(BB0.begin(), 2);
    MachineInstr &InlineAsmBr = *BB2.begin();
    MachineInstr &UseSSub = *std::next(BB4.begin(), 1);
    MachineInstr &DefD = *std::next(BB4.begin(), 3);
    MachineInstr &DefQ = *std::next(BB4.begin(), 5);

    unsigned HSub = DefH.getOperand(0).getSubReg();
    unsigned SSub = UseSSub.getOperand(1).getSubReg();
    unsigned DSub = DefD.getOperand(0).getSubReg();

    auto *HSR = LI.createSubRange(Alloc, TRI.getSubRegIndexLaneMask(HSub));
    auto *SSR = LI.createSubRange(Alloc, TRI.getSubRegIndexLaneMask(SSub));
    auto *DSR = LI.createSubRange(Alloc, TRI.getSubRegIndexLaneMask(DSub));

    SlotIndex HDef = Indexes.getInstructionIndex(DefH).getRegSlot();
    SlotIndex InsertIdx = Indexes.getInstructionIndex(InlineAsmBr);
    SlotIndex PredEndPrev = Indexes.getMBBEndIdx(&BB2).getPrevSlot();
    SlotIndex BB4Start = Indexes.getMBBStartIdx(&BB4);
    SlotIndex UseSSubIdx = Indexes.getInstructionIndex(UseSSub);
    SlotIndex DefDBase = Indexes.getInstructionIndex(DefD);
    SlotIndex DDef = DefDBase.getRegSlot();
    SlotIndex QDefBase = Indexes.getInstructionIndex(DefQ);

    ASSERT_LT(InsertIdx, PredEndPrev);

    // The MIR only needs to provide the CFG and instruction positions used by
    // findPHICopyInsertPoint(): bb.2 contains INLINEASM_BR, bb.4 is the
    // indirect target / PHI-def block, and bb.1 is the other predecessor.

    // One equivalence class is the overlapping hsub/ssub component:
    // - HVal is live in bb.2 at the INLINEASM_BR copy insertion point.
    // - SVal is a PHI-def at the start of bb.4.
    // - That same class is dead at the end of bb.2, so repairing the missing
    //   PHI live-in would need to insert an edge IMPLICIT_DEF in bb.2.
    //
    // Previously, the pass would attempt the split and eventually assert when
    // trying to add the new segment for the inserted IMPLICIT_DEF
    VNInfo *HVal = HSR->getNextValue(HDef, Alloc);
    HSR->addSegment({HDef, PredEndPrev, HVal});
    HSR->addSegment({BB4Start, UseSSubIdx, HVal});

    VNInfo *SVal = SSR->getNextValue(BB4Start, Alloc);
    ASSERT_TRUE(SVal->isPHIDef());
    SSR->addSegment({BB4Start, DefDBase, SVal});

    // A second disconnected component is required so the pass actually
    // considers splitting the interval before the safety check rejects it.
    VNInfo *DVal = DSR->getNextValue(DDef, Alloc);
    DSR->addSegment({DDef, QDefBase, DVal});

    LIS.constructMainRangeFromSubranges(LI);
    ASSERT_GE(LI.valnos.size(), 2u);

    // The crafted interval is intentionally inconsistent with the parsed MIR to
    // model the specific state that would trigger the assertion in addSegment
  }
};

TEST_F(RenameIndependentSubregsTest, SkipsSplitWhenPHIRepairWouldOverlapClass) {
  ASSERT_TRUE(parseMIR(getTestMIR()));

  MachineFunction &MF = getMF("test");
  LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);
  shapeUnsafeInterval(MF, LIS);

  unsigned Before = MF.getRegInfo().getNumVirtRegs();
  RenameIndependentSubregsPass Pass;
  Pass.run(MF, MFAM);

  EXPECT_EQ(MF.getRegInfo().getNumVirtRegs(), Before);
}

} // namespace
