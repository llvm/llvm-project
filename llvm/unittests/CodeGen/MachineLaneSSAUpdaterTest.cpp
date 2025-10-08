//===- MachineLaneSSAUpdaterTest.cpp - Unit tests for MachineLaneSSAUpdater -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineLaneSSAUpdater.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/LaneBitmask.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

using namespace llvm;

// TestPass needs to be defined outside anonymous namespace for INITIALIZE_PASS
struct TestPass : public MachineFunctionPass {
  static char ID;
  TestPass() : MachineFunctionPass(ID) {}
};

char TestPass::ID = 0;

namespace llvm {
  void initializeTestPassPass(PassRegistry &);
}

INITIALIZE_PASS(TestPass, "testpass", "testpass", false, false)

namespace {

void initLLVM() {
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  PassRegistry *Registry = PassRegistry::getPassRegistry();
  initializeCore(*Registry);
  initializeCodeGen(*Registry);
}

// Helper to create a target machine for AMDGPU
std::unique_ptr<TargetMachine> createTargetMachine() {
  Triple TT("amdgcn--");
  std::string Error;
  const Target *T = TargetRegistry::lookupTarget("", TT, Error);
  if (!T)
    return nullptr;
    
  TargetOptions Options;
  return std::unique_ptr<TargetMachine>(
      T->createTargetMachine(TT, "gfx900", "", Options, std::nullopt,
                             std::nullopt, CodeGenOptLevel::Aggressive));
}

// Helper to parse MIR string with legacy PassManager
std::unique_ptr<Module> parseMIR(LLVMContext &Context,
                                 legacy::PassManagerBase &PM,
                                 std::unique_ptr<MIRParser> &MIR,
                                 const TargetMachine &TM, StringRef MIRCode) {
  SMDiagnostic Diagnostic;
  std::unique_ptr<MemoryBuffer> MBuffer = MemoryBuffer::getMemBuffer(MIRCode);
  MIR = createMIRParser(std::move(MBuffer), Context);
  if (!MIR)
    return nullptr;

  std::unique_ptr<Module> M = MIR->parseIRModule();
  if (!M)
    return nullptr;

  M->setDataLayout(TM.createDataLayout());

  MachineModuleInfoWrapperPass *MMIWP = new MachineModuleInfoWrapperPass(&TM);
  if (MIR->parseMachineFunctions(*M, MMIWP->getMMI()))
    return nullptr;
  PM.add(MMIWP);

  return M;
}

template <typename AnalysisType>
struct TestPassT : public TestPass {
  typedef std::function<void(MachineFunction&, AnalysisType&)> TestFx;

  TestPassT() {
    // We should never call this but always use PM.add(new TestPass(...))
    abort();
  }
  
  TestPassT(TestFx T, bool ShouldPass)
      : T(T), ShouldPass(ShouldPass) {
    initializeTestPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    AnalysisType &A = getAnalysis<AnalysisType>();
    T(MF, A);
    bool VerifyResult = MF.verify(this, /* Banner=*/nullptr,
                                   /*OS=*/&llvm::errs(),
                                   /* AbortOnError=*/false);
    EXPECT_EQ(VerifyResult, ShouldPass);
    return true;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<AnalysisType>();
    AU.addPreserved<AnalysisType>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  
private:
  TestFx T;
  bool ShouldPass;
};

template <typename AnalysisType>
static void doTest(StringRef MIRFunc,
                   typename TestPassT<AnalysisType>::TestFx T,
                   bool ShouldPass = true) {
  initLLVM();
  
  LLVMContext Context;
  std::unique_ptr<TargetMachine> TM = createTargetMachine();
  if (!TM)
    GTEST_SKIP() << "AMDGPU target not available";

  legacy::PassManager PM;
  std::unique_ptr<MIRParser> MIR;
  std::unique_ptr<Module> M = parseMIR(Context, PM, MIR, *TM, MIRFunc);
  ASSERT_TRUE(M);

  PM.add(new TestPassT<AnalysisType>(T, ShouldPass));

  PM.run(*M);
}

static void liveIntervalsTest(StringRef MIRFunc,
                              TestPassT<LiveIntervalsWrapperPass>::TestFx T,
                              bool ShouldPass = true) {
  SmallString<512> S;
  StringRef MIRString = (Twine(R"MIR(
--- |
  define amdgpu_kernel void @func() { ret void }
...
---
name: func
tracksRegLiveness: true
registers:
  - { id: 0, class: vgpr_32 }
body: |
  bb.0:
)MIR") + Twine(MIRFunc) + Twine("...\n")).toNullTerminatedStringRef(S);

  doTest<LiveIntervalsWrapperPass>(MIRString, T, ShouldPass);
}

//===----------------------------------------------------------------------===//
// Test 1: Insert new definition and verify SSA repair with PHI insertion
//===----------------------------------------------------------------------===//

TEST(MachineLaneSSAUpdaterTest, NewDefInsertsPhiAndRewritesUses) {
  liveIntervalsTest(R"MIR(
    %0:vgpr_32 = V_MOV_B32_e32 0, implicit $exec
    S_BRANCH %bb.1

  bb.1:
    successors: %bb.2, %bb.3
    %1:vgpr_32 = V_ADD_U32_e32 %0, %0, implicit $exec
    $sgpr0 = S_MOV_B32 0
    $sgpr1 = S_MOV_B32 1
    S_CMP_LG_U32 $sgpr0, $sgpr1, implicit-def $scc
    S_CBRANCH_SCC1 %bb.3, implicit $scc

  bb.2:
    successors: %bb.4
    %2:vgpr_32 = V_ADD_U32_e32 %1, %1, implicit $exec
    S_BRANCH %bb.4

  bb.3:
    successors: %bb.4
    S_NOP 0

  bb.4:
    %5:vgpr_32 = V_ADD_U32_e32 %1, %1, implicit $exec
    S_ENDPGM 0
)MIR",
    [](MachineFunction &MF, LiveIntervalsWrapperPass &LISWrapper) {
      LiveIntervals &LIS = LISWrapper.getLIS();
      MachineDominatorTree MDT(MF);
      const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
      MachineRegisterInfo &MRI = MF.getRegInfo();
      
      // Verify we have 5 blocks as expected
      ASSERT_EQ(MF.size(), 5u) << "Should have bb.0, bb.1, bb.2, bb.3, bb.4";
      
      MachineBasicBlock *BB1 = MF.getBlockNumbered(1);
      MachineBasicBlock *BB3 = MF.getBlockNumbered(3);
      MachineBasicBlock *BB4 = MF.getBlockNumbered(4);
      
      // Get %1 which is defined in bb.1 (first non-PHI instruction)
      MachineInstr *OrigDefMI = &*BB1->getFirstNonPHI();
      ASSERT_TRUE(OrigDefMI) << "Could not find instruction in bb.1";
      ASSERT_TRUE(OrigDefMI->getNumOperands() > 0) << "Instruction has no operands";
      
      Register OrigReg = OrigDefMI->getOperand(0).getReg();
      ASSERT_TRUE(OrigReg.isValid()) << "Could not get destination register %1 from bb.1";
      
      // Count uses before SSA repair
      unsigned UseCountBefore = 0;
      for (const MachineInstr &MI : MRI.use_instructions(OrigReg)) {
        (void)MI;
        ++UseCountBefore;
      }
      ASSERT_GT(UseCountBefore, 0u) << "Original register should have uses";
      
      // Find V_MOV_B32_e32 instruction in bb.0 to get its opcode
      MachineBasicBlock *BB0 = MF.getBlockNumbered(0);
      MachineInstr *MovInst = &*BB0->begin();
      unsigned MovOpcode = MovInst->getOpcode();
      Register ExecReg = MovInst->getOperand(2).getReg(); // Get EXEC register
      
      // Create a new definition in bb.3 that defines OrigReg (violating SSA)
      // This creates a scenario where bb.4 needs a PHI to merge values from bb.2 and bb.3
      const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
      auto InsertPt = BB3->getFirstNonPHI();
      MachineInstr *NewDefMI = BuildMI(*BB3, InsertPt, DebugLoc(), 
                                        TII->get(MovOpcode), OrigReg)
                                   .addImm(42)
                                   .addReg(ExecReg, RegState::Implicit);
      
      // Set MachineFunction properties to allow PHIs and indicate SSA form
      MF.getProperties().set(MachineFunctionProperties::Property::IsSSA);
      MF.getProperties().reset(MachineFunctionProperties::Property::NoPHIs);
      
      // NOW TEST MachineLaneSSAUpdater: call repairSSAForNewDef
      // Before: %1 defined in bb.1, used in bb.2 and bb.4
      //         NewDefMI in bb.3 also defines %1 (violating SSA!)
      // After repair: NewDefMI will define a new vreg, bb.4 gets PHI
      MachineLaneSSAUpdater Updater(MF, LIS, MDT, *TRI);
      Register NewReg = Updater.repairSSAForNewDef(*NewDefMI, OrigReg);
      
      // VERIFY RESULTS:
      
      // 1. NewReg should be valid and different from OrigReg
      EXPECT_TRUE(NewReg.isValid()) << "Updater should create a new register";
      EXPECT_NE(NewReg, OrigReg) << "New register should be different from original";
      
      // 2. NewDefMI should now define NewReg (not OrigReg)
      EXPECT_EQ(NewDefMI->getOperand(0).getReg(), NewReg) << "NewDefMI should now define the new register";
      
      
      // 3. Check if PHI nodes were inserted in bb.4
      bool FoundPHI = false;
      for (MachineInstr &MI : *BB4) {
        if (MI.isPHI()) {
          FoundPHI = true;
          break;
        }
      }
      EXPECT_TRUE(FoundPHI) << "SSA repair should have inserted PHI node in bb.4";
      
      // 4. Verify LiveIntervals are still valid
      EXPECT_TRUE(LIS.hasInterval(NewReg)) << "New register should have live interval";
      EXPECT_TRUE(LIS.hasInterval(OrigReg)) << "Original register should still have live interval";
      
      // Note: MachineFunction verification happens in TestPassT::runOnMachineFunction
      // If verification fails, print the MachineFunction for debugging
      if (!MF.verify(nullptr, /* Banner=*/nullptr, /*OS=*/nullptr, /* AbortOnError=*/false)) {
        llvm::errs() << "MachineFunction verification failed after SSA repair:\n";
        MF.print(llvm::errs());
        LIS.print(llvm::errs());
      }
    });
}

//===----------------------------------------------------------------------===//
// Test 2: Multiple PHI insertions in nested control flow
//
// CFG structure (from user's diagram):
//        bb.0
//          |
//        bb.1 (%1 = original def)
//        /  \
//     bb.2  bb.3
//      |    /  \
//      | bb.4  bb.5 (new def inserted here)
//      |   \  /
//      |   bb.6 (needs first PHI: %X = PHI %1,bb.4  NewDef,bb.5)
//       \  /
//        bb.7 (needs second PHI: %Y = PHI %1,bb.2  %X,bb.6)
//          |
//        bb.8 (use)
//
// Key insight: IDF(bb.5) = {bb.6, bb.7}
// - bb.6 needs PHI because it's reachable from bb.4 (has %1) and bb.5 (has new def)
// - bb.7 needs PHI because it's reachable from bb.2 (has %1) and bb.6 (has PHI result %X)
//
// This truly requires TWO PHI nodes for proper SSA form!
//===----------------------------------------------------------------------===//

TEST(MachineLaneSSAUpdaterTest, MultiplePhiInsertion) {
  liveIntervalsTest(R"MIR(
    %0:vgpr_32 = V_MOV_B32_e32 0, implicit $exec
    S_BRANCH %bb.1

  bb.1:
    successors: %bb.2, %bb.3
    %1:vgpr_32 = V_ADD_U32_e32 %0, %0, implicit $exec
    $sgpr0 = S_MOV_B32 0
    $sgpr1 = S_MOV_B32 1
    S_CMP_LG_U32 $sgpr0, $sgpr1, implicit-def $scc
    S_CBRANCH_SCC1 %bb.3, implicit $scc

  bb.2:
    successors: %bb.7
    %2:vgpr_32 = V_ADD_U32_e32 %1, %1, implicit $exec
    S_BRANCH %bb.7

  bb.3:
    successors: %bb.4, %bb.5
    $sgpr2 = S_MOV_B32 0
    $sgpr3 = S_MOV_B32 1
    S_CMP_LG_U32 $sgpr2, $sgpr3, implicit-def $scc
    S_CBRANCH_SCC1 %bb.5, implicit $scc

  bb.4:
    successors: %bb.6
    %3:vgpr_32 = V_ADD_U32_e32 %1, %1, implicit $exec
    S_BRANCH %bb.6

  bb.5:
    successors: %bb.6
    S_NOP 0

  bb.6:
    successors: %bb.7
    %4:vgpr_32 = V_SUB_U32_e32 %1, %1, implicit $exec

  bb.7:
    successors: %bb.8
    %5:vgpr_32 = V_AND_B32_e32 %1, %1, implicit $exec
    S_BRANCH %bb.8

  bb.8:
    %6:vgpr_32 = V_OR_B32_e32 %1, %1, implicit $exec
    S_ENDPGM 0
)MIR",
    [](MachineFunction &MF, LiveIntervalsWrapperPass &LISWrapper) {
      LiveIntervals &LIS = LISWrapper.getLIS();
      MachineDominatorTree MDT(MF);
      const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
      MachineRegisterInfo &MRI = MF.getRegInfo();
      
      // Verify we have the expected number of blocks
      ASSERT_EQ(MF.size(), 9u) << "Should have bb.0 through bb.8";
      
      MachineBasicBlock *BB1 = MF.getBlockNumbered(1);
      MachineBasicBlock *BB5 = MF.getBlockNumbered(5);
      MachineBasicBlock *BB6 = MF.getBlockNumbered(6);
      MachineBasicBlock *BB7 = MF.getBlockNumbered(7);
      
      // Get %1 which is defined in bb.1
      MachineInstr *OrigDefMI = &*BB1->getFirstNonPHI();
      Register OrigReg = OrigDefMI->getOperand(0).getReg();
      ASSERT_TRUE(OrigReg.isValid()) << "Could not get original register";
      
      // Count uses of %1 before SSA repair
      unsigned UseCountBefore = 0;
      for (const MachineInstr &MI : MRI.use_instructions(OrigReg)) {
        (void)MI;
        ++UseCountBefore;
      }
      ASSERT_GT(UseCountBefore, 0u) << "Original register should have uses";
      llvm::errs() << "Original register has " << UseCountBefore << " uses before SSA repair\n";
      
      // Get V_MOV opcode from bb.0
      MachineBasicBlock *BB0 = MF.getBlockNumbered(0);
      MachineInstr *MovInst = &*BB0->begin();
      unsigned MovOpcode = MovInst->getOpcode();
      Register ExecReg = MovInst->getOperand(2).getReg();
      
      // Insert new definition in bb.5 that defines OrigReg (violating SSA)
      const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
      auto InsertPt = BB5->getFirstNonPHI();
      MachineInstr *NewDefMI = BuildMI(*BB5, InsertPt, DebugLoc(), 
                                        TII->get(MovOpcode), OrigReg)
                                   .addImm(100)
                                   .addReg(ExecReg, RegState::Implicit);
      
      // Set MachineFunction properties
      MF.getProperties().set(MachineFunctionProperties::Property::IsSSA);
      MF.getProperties().reset(MachineFunctionProperties::Property::NoPHIs);
      
      // Call MachineLaneSSAUpdater
      MachineLaneSSAUpdater Updater(MF, LIS, MDT, *TRI);
      Register NewReg = Updater.repairSSAForNewDef(*NewDefMI, OrigReg);
      
      EXPECT_TRUE(NewReg.isValid()) << "Updater should create a new register";
      EXPECT_NE(NewReg, OrigReg) << "New register should be different from original";
      EXPECT_EQ(NewDefMI->getOperand(0).getReg(), NewReg) << "NewDefMI should now define the new register";
      
      // Count PHI nodes inserted and track their locations
      unsigned PHICount = 0;
      std::map<unsigned, unsigned> PHIsPerBlock;
      for (MachineBasicBlock &MBB : MF) {
        unsigned BlockPHIs = 0;
        for (MachineInstr &MI : MBB) {
          if (MI.isPHI()) {
            ++PHICount;
            ++BlockPHIs;
            llvm::errs() << "Found PHI in BB#" << MBB.getNumber() << ": ";
            MI.print(llvm::errs());
          }
        }
        if (BlockPHIs > 0) {
          PHIsPerBlock[MBB.getNumber()] = BlockPHIs;
        }
      }
      
      llvm::errs() << "Total PHI nodes inserted: " << PHICount << "\n";
      
      // Check for first PHI in bb.6 (joins bb.4 and bb.5)
      bool FoundPHIInBB6 = false;
      for (MachineInstr &MI : *BB6) {
        if (MI.isPHI()) {
          FoundPHIInBB6 = true;
          llvm::errs() << "First PHI in bb.6: ";
          MI.print(llvm::errs());
          // Verify it has 2 incoming values (4 operands: 2 x (reg, mbb))
          unsigned NumIncoming = (MI.getNumOperands() - 1) / 2;
          EXPECT_EQ(NumIncoming, 2u) << "First PHI in bb.6 should have 2 incoming values (from bb.4 and bb.5)";
          break;
        }
      }
      EXPECT_TRUE(FoundPHIInBB6) << "Should have first PHI in bb.6 (joins bb.4 with %1 and bb.5 with new def)";
      
      // Check for second PHI in bb.7 (joins bb.2 and bb.6)
      bool FoundPHIInBB7 = false;
      for (MachineInstr &MI : *BB7) {
        if (MI.isPHI()) {
          FoundPHIInBB7 = true;
          llvm::errs() << "Second PHI in bb.7: ";
          MI.print(llvm::errs());
          // Verify it has 2 incoming values (4 operands: 2 x (reg, mbb))
          unsigned NumIncoming = (MI.getNumOperands() - 1) / 2;
          EXPECT_EQ(NumIncoming, 2u) << "Second PHI in bb.7 should have 2 incoming values (from bb.2 with %1 and bb.6 with first PHI result)";
          break;
        }
      }
      EXPECT_TRUE(FoundPHIInBB7) << "Should have second PHI in bb.7 (joins bb.2 with %1 and bb.6 with first PHI)";
      
      // Should have exactly 2 PHIs
      EXPECT_EQ(PHICount, 2u) << "Should have inserted exactly TWO PHI nodes (one at bb.6, one at bb.7)";
      
      // Verify LiveIntervals are valid
      EXPECT_TRUE(LIS.hasInterval(NewReg)) << "New register should have live interval";
      EXPECT_TRUE(LIS.hasInterval(OrigReg)) << "Original register should have live interval";
      
      // Debug output if verification fails
      if (!MF.verify(nullptr, nullptr, nullptr, false)) {
        llvm::errs() << "MachineFunction verification failed:\n";
        MF.print(llvm::errs());
        LIS.print(llvm::errs());
      }
    });
}

//===----------------------------------------------------------------------===//
// Test 3: Subregister lane tracking with partial register updates
//
// This tests the "LaneAware" part of MachineLaneSSAUpdater.
//
// Scenario:
//   - Start with a 64-bit register %1 (has sub0 and sub1 lanes)
//   - Insert a new definition that only updates sub0 (lower 32 bits)
//   - The SSA updater should:
//     1. Track that only sub0 lane is modified (not sub1)
//     2. Create PHI that merges only the sub0 lane
//     3. Preserve the original sub1 lane
//
// CFG:
//     bb.0
//       |
//     bb.1 (%1 = 64-bit def, both lanes)
//     /  \
//  bb.2  bb.3 (new def updates only %X.sub0)
//     \  /
//     bb.4 (needs PHI for sub0 lane only)
//       |
//     bb.5 (use both lanes)
//===----------------------------------------------------------------------===//

TEST(MachineLaneSSAUpdaterTest, SubregisterLaneTracking) {
  liveIntervalsTest(R"MIR(
    %0:vgpr_32 = V_MOV_B32_e32 0, implicit $exec
    S_BRANCH %bb.1

  bb.1:
    successors: %bb.2, %bb.3
    ; Create vregs in order: %1, %2, %3
    %1:vgpr_32 = V_MOV_B32_e32 10, implicit $exec
    %2:vgpr_32 = V_MOV_B32_e32 20, implicit $exec
    %3:vreg_64 = REG_SEQUENCE %1, %subreg.sub0, %2, %subreg.sub1
    $sgpr0 = S_MOV_B32 0
    $sgpr1 = S_MOV_B32 1
    S_CMP_LG_U32 $sgpr0, $sgpr1, implicit-def $scc
    S_CBRANCH_SCC1 %bb.3, implicit $scc

  bb.2:
    successors: %bb.4
    ; Use sub0 lane only
    %4:vgpr_32 = V_ADD_U32_e32 %3.sub0, %3.sub0, implicit $exec
    S_BRANCH %bb.4

  bb.3:
    successors: %bb.4
    S_NOP 0

  bb.4:
    successors: %bb.5
    ; Use both sub0 and sub1 lanes separately
    %5:vgpr_32 = V_ADD_U32_e32 %3.sub0, %3.sub1, implicit $exec
    S_BRANCH %bb.5

  bb.5:
    ; Use full 64-bit register (tests REG_SEQUENCE path after PHI)
    %6:vreg_64 = V_LSHLREV_B64_e64 0, %3, implicit $exec
    S_ENDPGM 0
)MIR",
    [](MachineFunction &MF, LiveIntervalsWrapperPass &LISWrapper) {
      LiveIntervals &LIS = LISWrapper.getLIS();
      MachineDominatorTree MDT(MF);
      const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
      MachineRegisterInfo &MRI = MF.getRegInfo();
      
      // Verify we have the expected number of blocks
      ASSERT_EQ(MF.size(), 6u) << "Should have bb.0 through bb.5";
      
      MachineBasicBlock *BB3 = MF.getBlockNumbered(3);
      
      // Get the 64-bit register %3 (vreg_64) from the MIR
      Register Reg64 = Register::index2VirtReg(3);
      ASSERT_TRUE(Reg64.isValid()) << "Register %3 should be valid";
      
      const TargetRegisterClass *RC64 = MRI.getRegClass(Reg64);
      ASSERT_EQ(TRI->getRegSizeInBits(*RC64), 64u) << "Register %3 should be 64-bit";
      llvm::errs() << "Using 64-bit register: %" << Reg64.virtRegIndex() << " (raw: " << Reg64 << ")\n";
      
      // Verify it has subranges for lane tracking
      ASSERT_TRUE(LIS.hasInterval(Reg64)) << "Register should have live interval";
      LiveInterval &LI = LIS.getInterval(Reg64);
      if (LI.hasSubRanges()) {
        llvm::errs() << "Register has subranges (lane tracking active)\n";
        for (const LiveInterval::SubRange &SR : LI.subranges()) {
          llvm::errs() << "  Lane mask: " << PrintLaneMask(SR.LaneMask) << "\n";
        }
      } else {
        llvm::errs() << "Warning: Register does not have subranges\n";
      }
      
      // Find the subreg index for a 32-bit subreg of the 64-bit register
      unsigned Sub0Idx = 0;
      for (unsigned Idx = 1, E = TRI->getNumSubRegIndices(); Idx <= E; ++Idx) {
        const TargetRegisterClass *SubRC = TRI->getSubRegisterClass(RC64, Idx);
        if (SubRC && TRI->getRegSizeInBits(*SubRC) == 32) {
          Sub0Idx = Idx;
          break;
        }
      }
      ASSERT_NE(Sub0Idx, 0u) << "Could not find 32-bit subregister index";
      LaneBitmask Sub0Mask = TRI->getSubRegIndexLaneMask(Sub0Idx);
      llvm::errs() << "Sub0 index=" << Sub0Idx << " (" << TRI->getSubRegIndexName(Sub0Idx) 
                   << "), mask=" << PrintLaneMask(Sub0Mask) << "\n";
      
      // Insert new definition in bb.3 that defines Reg64.sub0 (partial update, violating SSA)
      // Use V_MOV with immediate - no liveness dependencies
      // It's the caller's responsibility to ensure source operands are valid
      const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
      auto InsertPt = BB3->getFirstNonPHI();
      
      // Get V_MOV opcode and EXEC register from bb.0
      MachineBasicBlock *BB0 = MF.getBlockNumbered(0);
      MachineInstr *MovInst = &*BB0->begin();
      unsigned MovOpcode = MovInst->getOpcode();
      Register ExecReg = MovInst->getOperand(2).getReg();
      
      // Create a 32-bit temporary register
      Register TempReg = MRI.createVirtualRegister(TRI->getSubRegisterClass(RC64, Sub0Idx));
      
      // Insert both instructions first (V_MOV and COPY)
      MachineInstr *TempMI = BuildMI(*BB3, InsertPt, DebugLoc(), TII->get(MovOpcode), TempReg)
          .addImm(99)
          .addReg(ExecReg, RegState::Implicit);
      
      MachineInstr *NewDefMI = BuildMI(*BB3, InsertPt, DebugLoc(), 
                                        TII->get(TargetOpcode::COPY))
          .addReg(Reg64, RegState::Define, Sub0Idx)  // %3.sub0 = (violates SSA)
          .addReg(TempReg);                           // COPY from temp
      
      // Caller's responsibility: index instructions and create live intervals
      // Do this AFTER inserting both instructions so uses are visible
      LIS.InsertMachineInstrInMaps(*TempMI);
      LIS.InsertMachineInstrInMaps(*NewDefMI);
      LIS.createAndComputeVirtRegInterval(TempReg);
      
      // Set MachineFunction properties
      MF.getProperties().set(MachineFunctionProperties::Property::IsSSA);
      MF.getProperties().reset(MachineFunctionProperties::Property::NoPHIs);
      
      // Call MachineLaneSSAUpdater to repair the SSA violation
      // This should create a new vreg for the subreg def and insert lane-aware PHIs
      MachineLaneSSAUpdater Updater(MF, LIS, MDT, *TRI);
      Register NewReg = Updater.repairSSAForNewDef(*NewDefMI, Reg64);
      
      llvm::errs() << "SSA repair created new register: %" << NewReg.virtRegIndex() << " (raw: " << NewReg << ")\n";
      
      // VERIFY RESULTS:
      
      // 1. NewReg should be a 32-bit register (for sub0), not 64-bit
      EXPECT_TRUE(NewReg.isValid()) << "Updater should create a new register";
      EXPECT_NE(NewReg, Reg64) << "New register should be different from original";
      
      const TargetRegisterClass *NewRC = MRI.getRegClass(NewReg);
      EXPECT_EQ(TRI->getRegSizeInBits(*NewRC), 32u) << "New register should be 32-bit (subreg class)";
      
      // 2. NewDefMI should now define NewReg (not Reg64.sub0)
      EXPECT_EQ(NewDefMI->getOperand(0).getReg(), NewReg) << "NewDefMI should now define new 32-bit register";
      EXPECT_EQ(NewDefMI->getOperand(0).getSubReg(), 0u) << "NewDefMI should no longer have subreg index";
      
      // 3. Verify PHIs were inserted where needed
      MachineBasicBlock *BB4 = MF.getBlockNumbered(4);
      bool FoundPHI = false;
      for (MachineInstr &MI : *BB4) {
        if (MI.isPHI()) {
          FoundPHI = true;
          llvm::errs() << "Found PHI in bb.4: ";
          MI.print(llvm::errs());
          break;
        }
      }
      EXPECT_TRUE(FoundPHI) << "Should have inserted PHI for sub0 lane in bb.4";
      
      // 4. Verify LiveIntervals are valid
      EXPECT_TRUE(LIS.hasInterval(NewReg)) << "New register should have live interval";
      
      // Debug output if verification fails
      if (!MF.verify(nullptr, nullptr, nullptr, false)) {
        llvm::errs() << "MachineFunction verification failed:\n";
        MF.print(llvm::errs());
        LIS.print(llvm::errs());
      }
    });
}

//===----------------------------------------------------------------------===//
// Test 4: Subreg def → Full register PHI (REG_SEQUENCE before PHI)
//
// This tests the critical case where:
// - Input MIR has a PHI that expects full 64-bit register from both paths
// - We insert a subreg definition (X.sub0) on one path
// - The updater must build a REG_SEQUENCE before the PHI to combine:
//     NewSubreg (sub0) + OriginalReg.sub1 → FullReg for PHI
//
// CFG:
//     bb.0 (entry)
//       |
//     bb.1 (X=1, full 64-bit def)
//      / \
//   bb.2  bb.3
//   (Y=2)  / \
//     |  bb.4  bb.5 (NEW DEF: X.sub0 = 3) ← inserted by test
//     |    \  /
//     |    bb.6 (first join: bb.4 + bb.5, may need REG_SEQUENCE)
//     |    /
//     \  /
//     bb.7 (second join: PHI Z = PHI(Y, bb.2, X, bb.6)) ← already in input MIR
//       |
//     bb.8 (use Z)
//
// Expected: REG_SEQUENCE in bb.6 before branching to bb.7
//===----------------------------------------------------------------------===//

TEST(MachineLaneSSAUpdaterTest, SubregDefToFullRegPHI) {
  liveIntervalsTest(R"MIR(
    %0:vgpr_32 = V_MOV_B32_e32 0, implicit $exec
    S_BRANCH %bb.1

  bb.1:
    successors: %bb.2, %bb.3
    ; X = 1 (full 64-bit register)
    %1:vgpr_32 = V_MOV_B32_e32 10, implicit $exec
    %2:vgpr_32 = V_MOV_B32_e32 11, implicit $exec
    %3:vreg_64 = REG_SEQUENCE %1, %subreg.sub0, %2, %subreg.sub1
    $sgpr0 = S_MOV_B32 0
    $sgpr1 = S_MOV_B32 1
    S_CMP_LG_U32 $sgpr0, $sgpr1, implicit-def $scc
    S_CBRANCH_SCC1 %bb.3, implicit $scc

  bb.2:
    successors: %bb.7
    ; Y = 2 (full 64-bit register, different from X)
    %4:vgpr_32 = V_MOV_B32_e32 20, implicit $exec
    %5:vgpr_32 = V_MOV_B32_e32 21, implicit $exec
    %6:vreg_64 = REG_SEQUENCE %4, %subreg.sub0, %5, %subreg.sub1
    S_BRANCH %bb.7

  bb.3:
    successors: %bb.4, %bb.5
    $sgpr2 = S_MOV_B32 0
    $sgpr3 = S_MOV_B32 1
    S_CMP_LG_U32 $sgpr2, $sgpr3, implicit-def $scc
    S_CBRANCH_SCC1 %bb.5, implicit $scc

  bb.4:
    successors: %bb.6
    S_NOP 0
    S_BRANCH %bb.6

  bb.5:
    successors: %bb.6
    ; New def will be inserted here: X.sub0 = 3
    S_NOP 0

  bb.6:
    successors: %bb.7
    S_BRANCH %bb.7

  bb.7:
    ; PHI already in input MIR, expects full 64-bit from both paths
    %7:vreg_64 = PHI %6:vreg_64, %bb.2, %3:vreg_64, %bb.6
    S_BRANCH %bb.8

  bb.8:
    ; Use Z
    %8:vreg_64 = V_LSHLREV_B64_e64 0, %7, implicit $exec
    S_ENDPGM 0
)MIR",
    [](MachineFunction &MF, LiveIntervalsWrapperPass &LISWrapper) {
      LiveIntervals &LIS = LISWrapper.getLIS();
      MachineDominatorTree MDT(MF);
      const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
      MachineRegisterInfo &MRI = MF.getRegInfo();
      
      ASSERT_EQ(MF.size(), 9u) << "Should have bb.0 through bb.8";
      
      MachineBasicBlock *BB5 = MF.getBlockNumbered(5); // New def inserted here
      MachineBasicBlock *BB6 = MF.getBlockNumbered(6); // First join (bb.4 + bb.5)
      MachineBasicBlock *BB7 = MF.getBlockNumbered(7); // PHI block (bb.2 + bb.6)
      
      // Get register X (%3, the 64-bit register from bb.1)
      Register RegX = Register::index2VirtReg(3);
      ASSERT_TRUE(RegX.isValid()) << "Register %3 (X) should be valid";
      
      const TargetRegisterClass *RC64 = MRI.getRegClass(RegX);
      ASSERT_EQ(TRI->getRegSizeInBits(*RC64), 64u) << "Register X should be 64-bit";
      
      // Find sub0 index (32-bit subregister)
      unsigned Sub0Idx = 0;
      for (unsigned Idx = 1, E = TRI->getNumSubRegIndices(); Idx <= E; ++Idx) {
        const TargetRegisterClass *SubRC = TRI->getSubRegisterClass(RC64, Idx);
        if (SubRC && TRI->getRegSizeInBits(*SubRC) == 32) {
          Sub0Idx = Idx;
          break;
        }
      }
      ASSERT_NE(Sub0Idx, 0u) << "Could not find 32-bit subregister index";
      
      // Insert new definition in bb.5: X.sub0 = 3
      const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
      auto InsertPt = BB5->getFirstNonPHI();
      
      // Get V_MOV opcode and EXEC register
      MachineBasicBlock *BB0 = MF.getBlockNumbered(0);
      MachineInstr *MovInst = &*BB0->begin();
      unsigned MovOpcode = MovInst->getOpcode();
      Register ExecReg = MovInst->getOperand(2).getReg();
      
      // Create temporary register
      Register TempReg = MRI.createVirtualRegister(TRI->getSubRegisterClass(RC64, Sub0Idx));
      
      MachineInstr *TempMI = BuildMI(*BB5, InsertPt, DebugLoc(), TII->get(MovOpcode), TempReg)
          .addImm(30)
          .addReg(ExecReg, RegState::Implicit);
      
      MachineInstr *NewDefMI = BuildMI(*BB5, InsertPt, DebugLoc(), 
                                        TII->get(TargetOpcode::COPY))
          .addReg(RegX, RegState::Define, Sub0Idx)  // X.sub0 = 
          .addReg(TempReg);
      
      // Index instructions and create live interval for temp
      LIS.InsertMachineInstrInMaps(*TempMI);
      LIS.InsertMachineInstrInMaps(*NewDefMI);
      LIS.createAndComputeVirtRegInterval(TempReg);
      
      // Set MachineFunction properties
      MF.getProperties().set(MachineFunctionProperties::Property::IsSSA);
      MF.getProperties().reset(MachineFunctionProperties::Property::NoPHIs);
      
      // Call SSA updater
      MachineLaneSSAUpdater Updater(MF, LIS, MDT, *TRI);
      Register NewReg = Updater.repairSSAForNewDef(*NewDefMI, RegX);
      
      llvm::errs() << "SSA repair created new register: %" << NewReg.virtRegIndex() << " (raw: " << NewReg << ")\n";
      
      // VERIFY RESULTS:
      
      // 1. New register should be 32-bit (subreg class)
      EXPECT_TRUE(NewReg.isValid());
      EXPECT_NE(NewReg, RegX);
      const TargetRegisterClass *NewRC = MRI.getRegClass(NewReg);
      EXPECT_EQ(TRI->getRegSizeInBits(*NewRC), 32u) << "New register should be 32-bit";
      
      // 2. NewDefMI should now define NewReg without subreg index
      EXPECT_EQ(NewDefMI->getOperand(0).getReg(), NewReg);
      EXPECT_EQ(NewDefMI->getOperand(0).getSubReg(), 0u);
      
      // 3. Check the existing PHI in bb.7
      bool FoundPHI = false;
      Register PHIReg;
      MachineInstr *PHI = nullptr;
      for (MachineInstr &MI : *BB7) {
        if (MI.isPHI()) {
          FoundPHI = true;
          PHI = &MI;
          PHIReg = MI.getOperand(0).getReg();
          llvm::errs() << "PHI in bb.7 after SSA repair: ";
          MI.print(llvm::errs());
          break;
        }
      }
      ASSERT_TRUE(FoundPHI) << "Should have PHI in bb.7 (from input MIR)";
      
      // 4. CRITICAL: Check for REG_SEQUENCE in bb.6 (first join, before branch to PHI)
      // The updater must build REG_SEQUENCE to provide full register to the PHI
      bool FoundREGSEQ = false;
      for (MachineInstr &MI : *BB6) {
        if (MI.getOpcode() == TargetOpcode::REG_SEQUENCE) {
          FoundREGSEQ = true;
          llvm::errs() << "Found REG_SEQUENCE in bb.6: ";
          MI.print(llvm::errs());
          
          // Should combine new sub0 with original sub1
          EXPECT_GE(MI.getNumOperands(), 5u) << "REG_SEQUENCE should have result + 2 source pairs";
          break;
        }
      }
      EXPECT_TRUE(FoundREGSEQ) << "Should have built REG_SEQUENCE in bb.6 to provide full register to PHI in bb.7";
      
      // 5. Verify LiveIntervals
      EXPECT_TRUE(LIS.hasInterval(NewReg));
      EXPECT_TRUE(LIS.hasInterval(PHIReg));
      
      // Debug output if verification fails
      if (!MF.verify(nullptr, nullptr, nullptr, false)) {
        llvm::errs() << "MachineFunction verification failed:\n";
        MF.print(llvm::errs());
        LIS.print(llvm::errs());
      }
    });
}

} // anonymous namespace
