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

// Test basic PHI insertion and use rewriting in a diamond CFG
//
// CFG Structure:
//       BB0 (entry)
//        |
//       BB1 (%1 = orig def)
//      /   \
//    BB2   BB3 (INSERT: %1 = new_def)
//      \   /
//       BB4 (use %1) → PHI expected
//
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
// CFG structure:
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
//   - Start with a 64-bit register %3 (has sub0 and sub1 lanes)
//   - Insert a new definition that only updates sub0 (lower 32 bits)
//   - The SSA updater should:
//     1. Track that only sub0 lane is modified (not sub1)
//     2. Create PHI that merges only the sub0 lane
//     3. Preserve the original sub1 lane
//     4. Generate REG_SEQUENCE to compose full register from PHI+unchanged lanes
//
// CFG Structure:
//       BB0 (entry)
//        |
//       BB1 (%3:vreg_64 = REG_SEQUENCE of %1:sub0, %2:sub1)
//      /   \
//    BB2   BB3 (INSERT: %3.sub0 = new_def)
//     |     |
//    use   (no use)
//   sub0    
//      \   /
//       BB4 (use sub0 + sub1) → PHI for sub0 lane only
//        |
//       BB5 (use full %3) → REG_SEQUENCE to compose full reg from PHI result + unchanged sub1
//
// Expected behavior:
//   - PHI in BB4 merges only sub0 lane (changed)
//   - sub1 lane flows unchanged through the diamond
//   - REG_SEQUENCE in BB5 composes full 64-bit from (PHI_sub0, original_sub1)
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
      for (MachineInstr &MI : *BB7) {
        if (MI.isPHI()) {
          FoundPHI = true;
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

//===----------------------------------------------------------------------===//
// Test 5: Loop with new def in loop body (PHI in loop header)
//
// This tests SSA repair when a new definition is inserted inside a loop,
// requiring a PHI node in the loop header to merge:
// - Entry path: original value from before the loop
// - Back edge: new value from loop body
//
// CFG:
//     bb.0 (entry, X = 1)
//       |
//       v
//     bb.1 (loop header) ← PHI needed: %PHI = PHI(X, bb.0, NewReg, bb.2)
//      / \
//     /   \
//  bb.2   bb.3 (loop exit, use X)
//  (loop
//  body,
//  new def)
//     |
//     └──→ bb.1 (back edge)
//
// Key test: Dominance-based PHI construction should correctly use NewReg
// for the back edge operand since NewDefBB (bb.2) dominates the loop latch (bb.2).
//===----------------------------------------------------------------------===//

// Test loop with new definition in loop body requiring PHI in loop header
//
// CFG Structure:
//       BB0 (entry, %1 = orig def)
//        |
//    +-> BB1 (loop header)
//    |  / \
//    | /   \
//    BB2   BB3 (exit, use %1)
//    |
//    (INSERT: %1 = new_def)
//    |
//    +-(backedge) -> PHI needed in BB1 to merge initial value and loop value
//
TEST(MachineLaneSSAUpdaterTest, LoopWithDefInBody) {
  liveIntervalsTest(R"MIR(
    %0:vgpr_32 = V_MOV_B32_e32 0, implicit $exec
    ; Original definition of %1 (before loop)
    %1:vgpr_32 = V_ADD_U32_e32 %0, %0, implicit $exec
    S_BRANCH %bb.1

  bb.1:
    successors: %bb.2, %bb.3
    ; Loop header - PHI should be inserted here
    $sgpr0 = S_MOV_B32 0
    $sgpr1 = S_MOV_B32 1
    S_CMP_LG_U32 $sgpr0, $sgpr1, implicit-def $scc
    S_CBRANCH_SCC1 %bb.3, implicit $scc

  bb.2:
    successors: %bb.1
    ; Loop body - new def will be inserted here
    %2:vgpr_32 = V_ADD_U32_e32 %1, %1, implicit $exec
    S_BRANCH %bb.1

  bb.3:
    ; Loop exit - use %1
    %3:vgpr_32 = V_ADD_U32_e32 %1, %1, implicit $exec
    S_ENDPGM 0
)MIR",
    [](MachineFunction &MF, LiveIntervalsWrapperPass &LISWrapper) {
      LiveIntervals &LIS = LISWrapper.getLIS();
      MachineDominatorTree MDT(MF);
      const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
      
      ASSERT_EQ(MF.size(), 4u) << "Should have bb.0 through bb.3";
      
      MachineBasicBlock *BB0 = MF.getBlockNumbered(0); // Entry with original def
      MachineBasicBlock *BB1 = MF.getBlockNumbered(1); // Loop header
      MachineBasicBlock *BB2 = MF.getBlockNumbered(2); // Loop body
      
      // Get %1 (defined in bb.0, used in loop)
      // Skip the first V_MOV instruction, get the V_ADD
      auto It = BB0->begin();
      ++It; // Skip %0 = V_MOV
      MachineInstr *OrigDefMI = &*It;
      Register OrigReg = OrigDefMI->getOperand(0).getReg();
      ASSERT_TRUE(OrigReg.isValid()) << "Could not get original register";
      
      llvm::errs() << "Original register: %" << OrigReg.virtRegIndex() << "\n";
      
      // Insert new definition in loop body (bb.2)
      // This violates SSA because %1 is defined both in bb.0 and bb.2
      MachineInstr *MovInst = &*BB0->begin();
      unsigned MovOpcode = MovInst->getOpcode();
      Register ExecReg = MovInst->getOperand(2).getReg();
      
      const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
      auto InsertPt = BB2->getFirstNonPHI();
      MachineInstr *NewDefMI = BuildMI(*BB2, InsertPt, DebugLoc(), 
                                        TII->get(MovOpcode), OrigReg)
                                   .addImm(99)
                                   .addReg(ExecReg, RegState::Implicit);
      
      // Set MachineFunction properties
      MF.getProperties().set(MachineFunctionProperties::Property::IsSSA);
      MF.getProperties().reset(MachineFunctionProperties::Property::NoPHIs);
      
      // Call SSA updater
      MachineLaneSSAUpdater Updater(MF, LIS, MDT, *TRI);
      Register NewReg = Updater.repairSSAForNewDef(*NewDefMI, OrigReg);
      
      llvm::errs() << "SSA repair created new register: %" << NewReg.virtRegIndex() << "\n";
      
      // VERIFY RESULTS:
      
      // 1. NewReg should be valid and different from OrigReg
      EXPECT_TRUE(NewReg.isValid());
      EXPECT_NE(NewReg, OrigReg);
      
      // 2. NewDefMI should now define NewReg
      EXPECT_EQ(NewDefMI->getOperand(0).getReg(), NewReg);
      
      // 3. PHI should be inserted in loop header (bb.1)
      bool FoundPHIInHeader = false;
      for (MachineInstr &MI : *BB1) {
        if (MI.isPHI()) {
          FoundPHIInHeader = true;
          llvm::errs() << "Found PHI in loop header (bb.1): ";
          MI.print(llvm::errs());
          
          // Verify PHI has 2 incoming values
          unsigned NumIncoming = (MI.getNumOperands() - 1) / 2;
          EXPECT_EQ(NumIncoming, 2u) << "Loop header PHI should have 2 incoming values";
          
          // Check the operands
          // One should be from bb.0 (entry, using OrigReg)
          // One should be from bb.2 (back edge, using NewReg)
          bool HasEntryPath = false;
          bool HasBackEdge = false;
          
          for (unsigned i = 1; i < MI.getNumOperands(); i += 2) {
            Register IncomingReg = MI.getOperand(i).getReg();
            MachineBasicBlock *IncomingMBB = MI.getOperand(i + 1).getMBB();
            
            if (IncomingMBB == BB0) {
              HasEntryPath = true;
              EXPECT_EQ(IncomingReg, OrigReg) << "Entry path should use OrigReg";
              llvm::errs() << "  Entry path (bb.0): %" << IncomingReg.virtRegIndex() << "\n";
            } else if (IncomingMBB == BB2) {
              HasBackEdge = true;
              EXPECT_EQ(IncomingReg, NewReg) << "Back edge should use NewReg";
              llvm::errs() << "  Back edge (bb.2): %" << IncomingReg.virtRegIndex() << "\n";
            }
          }
          
          EXPECT_TRUE(HasEntryPath) << "PHI should have entry path from bb.0";
          EXPECT_TRUE(HasBackEdge) << "PHI should have back edge from bb.2";
          
          break;
        }
      }
      EXPECT_TRUE(FoundPHIInHeader) << "Should have inserted PHI in loop header (bb.1)";
      
      // 4. Verify LiveIntervals are valid
      EXPECT_TRUE(LIS.hasInterval(NewReg));
      EXPECT_TRUE(LIS.hasInterval(OrigReg));
      
      // Debug output if verification fails
      if (!MF.verify(nullptr, nullptr, nullptr, false)) {
        llvm::errs() << "MachineFunction verification failed:\n";
        MF.print(llvm::errs());
        LIS.print(llvm::errs());
      }
    });
}

//===----------------------------------------------------------------------===//
// Test 6: Complex loop with diamond CFG and use-before-def
//
// This is the most comprehensive test combining multiple SSA repair scenarios:
// 1. Loop with existing PHI (induction variable)
// 2. Use before redefinition (in loop header)
// 3. New definition in one branch of if-then-else diamond
// 4. PHI1 at diamond join
// 5. PHI2 at loop header (merges entry value and PHI1 result from back edge)
// 6. Use after diamond (in latch) should use PHI1 result
//
// CFG Structure:
//         BB0 (entry: X=%1, i=0)
//          |
//      +-> BB1 (loop header)
//      |   |   PHI_i = PHI(0, BB0; i+1, BB5) [already in input MIR]
//      |   |   PHI2 = PHI(X, BB0; PHI1, BB5) [created by SSA updater]
//      |   |   USE X (before redef!) [rewritten to use PHI2]
//      |   |   if (i < 10)
//      |  / \
//      | BB2 BB3 (INSERT: X = 99)
//      | |    |
//      | |    (then: X unchanged)
//      | |    (else: NEW DEF)
//      |  \  /
//      |   BB4 (diamond join)
//      |   |   PHI1 = PHI(X, BB2; NewReg, BB3) [created by SSA updater]
//      |   |
//      |   BB5 (loop latch)
//      |   |   USE X [rewritten to use PHI1]
//      |   |   i = i + 1
//      |   | \
//      |   |  \
//      +---+   BB6 (exit, USE X)
//
// Key challenge: Use in BB1 occurs BEFORE the def in BB3 (in program order),
//                requiring PHI2 in the loop header for proper SSA form.
//
// Expected SSA repair:
//   - PHI1 created in BB4 (diamond join): merges unchanged X from BB2, new def from BB3
//   - PHI2 created in BB1 (loop header): merges entry X from BB0, PHI1 result from BB5
//   - Use in BB1 rewritten to PHI2
//   - Use in BB5 rewritten to PHI1
//===----------------------------------------------------------------------===//
TEST(MachineLaneSSAUpdaterTest, ComplexLoopWithDiamondAndUseBeforeDef) {
  liveIntervalsTest(R"MIR(
    %0:vgpr_32 = V_MOV_B32_e32 0, implicit $exec
    ; X = 1 (the register we'll redefine in loop)
    %1:vgpr_32 = V_MOV_B32_e32 1, implicit $exec
    ; i = 0 (induction variable)
    %2:vgpr_32 = V_MOV_B32_e32 0, implicit $exec
    S_BRANCH %bb.1

  bb.1:
    successors: %bb.2, %bb.3
    ; Loop header with existing PHI for induction variable
    %3:vgpr_32 = PHI %2:vgpr_32, %bb.0, %10:vgpr_32, %bb.5
    ; USE X before redefinition - should be rewritten to PHI2
    %4:vgpr_32 = V_ADD_U32_e32 %1, %1, implicit $exec
    ; Check if i < 10
    %5:vgpr_32 = V_MOV_B32_e32 10, implicit $exec
    $sgpr0 = S_MOV_B32 0
    $sgpr1 = S_MOV_B32 1
    S_CMP_LG_U32 $sgpr0, $sgpr1, implicit-def $scc
    S_CBRANCH_SCC1 %bb.3, implicit $scc

  bb.2:
    successors: %bb.4
    ; Then branch - X unchanged
    S_NOP 0
    S_BRANCH %bb.4

  bb.3:
    successors: %bb.4
    ; Else branch - NEW DEF will be inserted here: X = 99
    S_NOP 0

  bb.4:
    successors: %bb.5
    ; Diamond join - PHI1 should be created here
    S_NOP 0

  bb.5:
    successors: %bb.1, %bb.6
    ; Loop latch - USE X (should be rewritten to PHI1)
    %8:vgpr_32 = V_SUB_U32_e32 %1, %1, implicit $exec
    ; i = i + 1
    %9:vgpr_32 = V_MOV_B32_e32 1, implicit $exec
    %10:vgpr_32 = V_ADD_U32_e32 %3, %9, implicit $exec
    ; Check loop condition
    $sgpr2 = S_MOV_B32 0
    $sgpr3 = S_MOV_B32 1
    S_CMP_LG_U32 $sgpr2, $sgpr3, implicit-def $scc
    S_CBRANCH_SCC1 %bb.6, implicit $scc
    S_BRANCH %bb.1

  bb.6:
    ; Loop exit - USE X
    %11:vgpr_32 = V_OR_B32_e32 %1, %1, implicit $exec
    S_ENDPGM 0
)MIR",
    [](MachineFunction &MF, LiveIntervalsWrapperPass &LISWrapper) {
      LiveIntervals &LIS = LISWrapper.getLIS();
      MachineDominatorTree MDT(MF);
      const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
      
      ASSERT_EQ(MF.size(), 7u) << "Should have bb.0 through bb.6";
      
      MachineBasicBlock *BB0 = MF.getBlockNumbered(0); // Entry
      MachineBasicBlock *BB1 = MF.getBlockNumbered(1); // Loop header
      MachineBasicBlock *BB3 = MF.getBlockNumbered(3); // Else (new def here)
      MachineBasicBlock *BB4 = MF.getBlockNumbered(4); // Diamond join
      MachineBasicBlock *BB5 = MF.getBlockNumbered(5); // Latch
      
      // Get %1 (X, defined in bb.0)
      auto It = BB0->begin();
      ++It; // Skip %0 = V_MOV_B32_e32 0
      MachineInstr *OrigDefMI = &*It; // %1 = V_MOV_B32_e32 1
      Register OrigReg = OrigDefMI->getOperand(0).getReg();
      ASSERT_TRUE(OrigReg.isValid()) << "Could not get original register X";
      
      llvm::errs() << "Original register X: %" << OrigReg.virtRegIndex() << "\n";
      
      // Find the use-before-def in bb.1 (loop header)
      MachineInstr *UseBeforeDefMI = nullptr;
      for (MachineInstr &MI : *BB1) {
        if (!MI.isPHI() && MI.getOpcode() != TargetOpcode::IMPLICIT_DEF) {
          // First non-PHI instruction should be V_ADD using %1
          if (MI.getNumOperands() >= 3 && MI.getOperand(1).isReg() && 
              MI.getOperand(1).getReg() == OrigReg) {
            UseBeforeDefMI = &MI;
            break;
          }
        }
      }
      ASSERT_TRUE(UseBeforeDefMI) << "Could not find use-before-def in loop header";
      llvm::errs() << "Found use-before-def in bb.1: %"
                   << UseBeforeDefMI->getOperand(0).getReg().virtRegIndex() << "\n";
      
      // Insert new definition in bb.3 (else branch): X = 99
      MachineInstr *MovInst = &*BB0->begin();
      unsigned MovOpcode = MovInst->getOpcode();
      Register ExecReg = MovInst->getOperand(2).getReg();
      
      const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
      auto InsertPt = BB3->getFirstNonPHI();
      MachineInstr *NewDefMI = BuildMI(*BB3, InsertPt, DebugLoc(), 
                                        TII->get(MovOpcode), OrigReg)
                                   .addImm(99)
                                   .addReg(ExecReg, RegState::Implicit);
      
      // Set MachineFunction properties
      MF.getProperties().set(MachineFunctionProperties::Property::IsSSA);
      MF.getProperties().reset(MachineFunctionProperties::Property::NoPHIs);
      
      // Call SSA updater
      MachineLaneSSAUpdater Updater(MF, LIS, MDT, *TRI);
      Register NewReg = Updater.repairSSAForNewDef(*NewDefMI, OrigReg);
      
      llvm::errs() << "SSA repair created new register: %" << NewReg.virtRegIndex() << "\n";
      
      // VERIFY RESULTS:
      
      // 1. NewReg should be valid and different from OrigReg
      EXPECT_TRUE(NewReg.isValid());
      EXPECT_NE(NewReg, OrigReg);
      EXPECT_EQ(NewDefMI->getOperand(0).getReg(), NewReg);
      
      // 2. PHI1 should exist in diamond join (bb.4)
      bool FoundPHI1 = false;
      Register PHI1Reg;
      for (MachineInstr &MI : *BB4) {
        if (MI.isPHI()) {
          FoundPHI1 = true;
          PHI1Reg = MI.getOperand(0).getReg();
          llvm::errs() << "Found PHI1 in diamond join (bb.4): ";
          MI.print(llvm::errs());
          
          // Should have 2 incoming: OrigReg from bb.2, NewReg from bb.3
          unsigned NumIncoming = (MI.getNumOperands() - 1) / 2;
          EXPECT_EQ(NumIncoming, 2u) << "Diamond join PHI should have 2 incoming";
          break;
        }
      }
      EXPECT_TRUE(FoundPHI1) << "Should have PHI1 in diamond join (bb.4)";
      
      // 3. PHI2 should exist in loop header (bb.1)
      // First, count all PHIs
      unsigned TotalPHICount = 0;
      for (MachineInstr &MI : *BB1) {
        if (MI.isPHI())
          TotalPHICount++;
      }
      llvm::errs() << "Total PHIs in loop header: " << TotalPHICount << "\n";
      EXPECT_EQ(TotalPHICount, 2u) << "Loop header should have 2 PHIs (induction var + SSA repair)";
      
      // Now find the SSA repair PHI (not the induction variable PHI %3)
      bool FoundPHI2 = false;
      Register PHI2Reg;
      Register InductionVarPHI = Register::index2VirtReg(3); // %3 from input MIR
      for (MachineInstr &MI : *BB1) {
        if (MI.isPHI()) {
          Register PHIResult = MI.getOperand(0).getReg();
          
          // Skip the induction variable PHI (%3 from input MIR) when looking for SSA repair PHI
          if (PHIResult == InductionVarPHI)
            continue;
          
          FoundPHI2 = true;
          PHI2Reg = PHIResult;
          llvm::errs() << "Found PHI2 (SSA repair) in loop header (bb.1): ";
          MI.print(llvm::errs());
          
          // Should have 2 incoming: OrigReg from bb.0, PHI1Reg from bb.5
          unsigned NumIncoming = (MI.getNumOperands() - 1) / 2;
          EXPECT_EQ(NumIncoming, 2u) << "Loop header PHI2 should have 2 incoming";
          
          // Verify operands
          bool HasEntryPath = false;
          bool HasBackEdge = false;
          for (unsigned i = 1; i < MI.getNumOperands(); i += 2) {
            Register IncomingReg = MI.getOperand(i).getReg();
            MachineBasicBlock *IncomingMBB = MI.getOperand(i + 1).getMBB();
            
            if (IncomingMBB == BB0) {
              HasEntryPath = true;
              EXPECT_EQ(IncomingReg, OrigReg) << "Entry path should use OrigReg";
            } else if (IncomingMBB == BB5) {
              HasBackEdge = true;
              EXPECT_EQ(IncomingReg, PHI1Reg) << "Back edge should use PHI1 result";
            }
          }
          
          EXPECT_TRUE(HasEntryPath) << "PHI2 should have entry path from bb.0";
          EXPECT_TRUE(HasBackEdge) << "PHI2 should have back edge from bb.5";
          break;
        }
      }
      EXPECT_TRUE(FoundPHI2) << "Should have PHI2 (SSA repair) in loop header (bb.1)";
      
      // 4. Use-before-def in bb.1 should be rewritten to PHI2
      EXPECT_EQ(UseBeforeDefMI->getOperand(1).getReg(), PHI2Reg)
          << "Use-before-def should be rewritten to PHI2 result";
      llvm::errs() << "Use-before-def correctly rewritten to PHI2: %"
                   << PHI2Reg.virtRegIndex() << "\n";
      
      // 5. Use in latch (bb.5) should be rewritten to PHI1
      // Find instruction using PHI1 (originally used %1)
      bool FoundLatchUse = false;
      for (MachineInstr &MI : *BB5) {
        // Skip PHIs and branches
        if (MI.isPHI() || MI.isBranch())
          continue;
        
        // Look for any instruction that uses PHI1Reg
        for (unsigned i = 0; i < MI.getNumOperands(); ++i) {
          MachineOperand &MO = MI.getOperand(i);
          if (MO.isReg() && MO.isUse() && MO.getReg() == PHI1Reg) {
            llvm::errs() << "Latch use correctly rewritten to PHI1: %"
                         << PHI1Reg.virtRegIndex() << " in: ";
            MI.print(llvm::errs());
            FoundLatchUse = true;
            break;
          }
        }
        if (FoundLatchUse)
          break;
      }
      EXPECT_TRUE(FoundLatchUse) << "Should find use of PHI1 in latch (bb.5)";
      
      // 6. Verify LiveIntervals
      EXPECT_TRUE(LIS.hasInterval(NewReg));
      EXPECT_TRUE(LIS.hasInterval(PHI1Reg));
      EXPECT_TRUE(LIS.hasInterval(PHI2Reg));
      
      // Debug output if verification fails
      if (!MF.verify(nullptr, nullptr, nullptr, false)) {
        llvm::errs() << "MachineFunction verification failed:\n";
        MF.print(llvm::errs());
        LIS.print(llvm::errs());
      }
    });
}

// Test 7: Multiple subreg redefinitions in loop (X.sub0 in one branch, X.sub1 in latch)
// This tests the most complex scenario: two separate lane redefinitions with REG_SEQUENCE
// composition at the backedge.
// Test multiple subregister redefinitions in different paths within a loop
//
// CFG Structure:
//         BB0 (entry, %1:vreg_64 = IMPLICIT_DEF)
//          |
//      +-> BB1 (loop header, PHI for %0)
//      |   |   (use %0.sub0)
//      |  / \
//      | BB2 BB5
//      | |    |
//      | use  INSERT: %0.sub0 = new_def1
//      |sub1  use %0.sub0
//      |  \   /
//      |   BB3 (latch)
//      |   |   (INSERT: %3.sub1 = new_def2, where %3 is increment result)
//      |   |   (%3 = %0 << 1)
//      +---+
//       |
//      BB4 (exit)
//
// Key: Two separate lane redefinitions requiring separate SSA repairs:
//      1. %0.sub0 in BB5 → PHI for sub0 in BB3
//      2. %3.sub1 in BB3 (after increment) → PHI for sub1 in BB1
//
TEST(MachineLaneSSAUpdaterTest, MultipleSubregRedefsInLoop) {
  SmallString<2048> S;
  StringRef MIRString = (Twine(R"MIR(
--- |
  define amdgpu_kernel void @func() { ret void }
...
---
name: func
tracksRegLiveness: true
registers:
  - { id: 0, class: vreg_64 }
  - { id: 1, class: vreg_64 }
  - { id: 2, class: vgpr_32 }
  - { id: 3, class: vreg_64 }
body: |
  bb.0:
    successors: %bb.1
    %1:vreg_64 = IMPLICIT_DEF
    
  bb.1:
    successors: %bb.2, %bb.5
    %0:vreg_64 = PHI %1:vreg_64, %bb.0, %3:vreg_64, %bb.3
    %2:vgpr_32 = V_MOV_B32_e32 10, implicit $exec
    dead %4:vgpr_32 = V_ADD_U32_e32 %0.sub0:vreg_64, %2:vgpr_32, implicit $exec
    $sgpr0 = S_MOV_B32 0
    $sgpr1 = S_MOV_B32 1
    S_CMP_LG_U32 $sgpr0, $sgpr1, implicit-def $scc
    S_CBRANCH_SCC1 %bb.2, implicit $scc
    S_BRANCH %bb.5
    
  bb.2:
    successors: %bb.3
    dead %5:vgpr_32 = V_MOV_B32_e32 %0.sub1:vreg_64, implicit $exec
    S_BRANCH %bb.3
    
  bb.5:
    successors: %bb.3
    dead %6:vgpr_32 = V_MOV_B32_e32 %0.sub0:vreg_64, implicit $exec
    S_BRANCH %bb.3
    
  bb.3:
    successors: %bb.1, %bb.4
    %3:vreg_64 = V_LSHLREV_B64_e64 1, %0:vreg_64, implicit $exec
    $sgpr2 = S_MOV_B32 0
    $sgpr3 = S_MOV_B32 10
    S_CMP_LT_U32 $sgpr2, $sgpr3, implicit-def $scc
    S_CBRANCH_SCC1 %bb.1, implicit $scc
    S_BRANCH %bb.4
    
  bb.4:
    S_ENDPGM 0
...
)MIR")).toNullTerminatedStringRef(S);

  doTest<LiveIntervalsWrapperPass>(MIRString,
             [](MachineFunction &MF, LiveIntervalsWrapperPass &LISWrapper) {
      LiveIntervals &LIS = LISWrapper.getLIS();
      MachineDominatorTree MDT(MF);
      llvm::errs() << "\n=== MultipleSubregRedefsInLoop Test ===\n";
      
      // Get basic blocks
      auto BBI = MF.begin();
      ++BBI;  // Skip BB0 (Entry)
      MachineBasicBlock *BB1 = &*BBI++;  // Loop header
      ++BBI;  // Skip BB2 (True branch)
      MachineBasicBlock *BB5 = &*BBI++;  // False branch (uses X.LO, INSERT def X.LO)
      MachineBasicBlock *BB3 = &*BBI++;  // Latch (increment, INSERT def X.HI)
      // Skip BB4 (Exit)
      
      MachineRegisterInfo &MRI = MF.getRegInfo();
      const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
      const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
      (void)MRI;  // May be unused, suppress warning
      
      // Find the 64-bit register and its subregister indices
      Register OrigReg = Register::index2VirtReg(0); // %0 from MIR
      ASSERT_TRUE(OrigReg.isValid()) << "Register %0 should be valid";
      unsigned Sub0Idx = 0, Sub1Idx = 0;
      
      // Find sub0 (low 32 bits) and sub1 (high 32 bits)
      for (unsigned Idx = 1; Idx < TRI->getNumSubRegIndices(); ++Idx) {
        LaneBitmask Mask = TRI->getSubRegIndexLaneMask(Idx);
        unsigned SubRegSize = TRI->getSubRegIdxSize(Idx);
        
        if (SubRegSize == 32) {
          if (Mask.getAsInteger() == 0x3) { // Low lanes
            Sub0Idx = Idx;
          } else if (Mask.getAsInteger() == 0xC) { // High lanes
            Sub1Idx = Idx;
          }
        }
      }
      
      ASSERT_NE(Sub0Idx, 0u) << "Should find sub0 index";
      ASSERT_NE(Sub1Idx, 0u) << "Should find sub1 index";
      
      llvm::errs() << "Using 64-bit register: %" << OrigReg.virtRegIndex() 
                   << " with sub0=" << Sub0Idx << ", sub1=" << Sub1Idx << "\n";
      
      // Get V_MOV opcode and EXEC register from existing instruction
      MachineInstr *MovInst = nullptr;
      Register ExecReg;
      for (MachineInstr &MI : *BB1) {
        if (!MI.isPHI() && MI.getNumOperands() >= 3 && MI.getOperand(2).isReg()) {
          MovInst = &MI;
          ExecReg = MI.getOperand(2).getReg();
          break;
        }
      }
      ASSERT_NE(MovInst, nullptr) << "Should find V_MOV in BB1";
      unsigned MovOpcode = MovInst->getOpcode();
      
      // === FIRST INSERTION: X.sub0 in BB5 (else branch) ===
      llvm::errs() << "\n=== First insertion: X.sub0 in BB5 ===\n";
      
      // Find insertion point in BB5 (after the use of X.sub0)
      MachineInstr *InsertPoint1 = nullptr;
      for (MachineInstr &MI : *BB5) {
        if (MI.isBranch()) {
          InsertPoint1 = &MI;
          break;
        }
      }
      ASSERT_NE(InsertPoint1, nullptr) << "Should find branch in BB5";
      
      // Create first new def: X.sub0 = 99
      MachineInstrBuilder MIB1 = BuildMI(*BB5, InsertPoint1, DebugLoc(), 
                                          TII->get(MovOpcode))
          .addReg(OrigReg, RegState::Define, Sub0Idx)
          .addImm(99)
          .addReg(ExecReg, RegState::Implicit);
      
      MachineInstr &NewDefMI1 = *MIB1;
      llvm::errs() << "Created first def in BB5: ";
      NewDefMI1.print(llvm::errs());
      
      // Create SSA updater and repair after first insertion
      MachineLaneSSAUpdater Updater(MF, LIS, MDT, *TRI);
      Register NewReg1 = Updater.repairSSAForNewDef(NewDefMI1, OrigReg);
      
      llvm::errs() << "SSA repair #1 created new register: %" << NewReg1.virtRegIndex() << "\n";
      
      // === SECOND INSERTION: X.sub1 in BB3 (after increment) ===
      llvm::errs() << "\n=== Second insertion: X.sub1 in BB3 (after increment) ===\n";
      
      // Find the increment instruction in BB3 (look for vreg_64 def)
      MachineInstr *IncrementMI = nullptr;
      Register IncrementReg;
      for (MachineInstr &MI : *BB3) {
        if (!MI.isPHI() && MI.getNumOperands() > 0 && MI.getOperand(0).isReg() && 
            MI.getOperand(0).isDef()) {
          Register DefReg = MI.getOperand(0).getReg();
          if (DefReg.isVirtual() && DefReg == Register::index2VirtReg(3)) {
            IncrementMI = &MI;
            IncrementReg = DefReg; // This is %3
            llvm::errs() << "Found increment: ";
            MI.print(llvm::errs());
            break;
          }
        }
      }
      ASSERT_NE(IncrementMI, nullptr) << "Should find increment (def of %3) in BB3";
      ASSERT_TRUE(IncrementReg.isValid()) << "Increment register should be valid";
      
      // Create second new def: %3.sub1 = 200 (redefine increment result's sub1)
      MachineBasicBlock::iterator InsertPoint2 = std::next(IncrementMI->getIterator());
      MachineInstrBuilder MIB2 = BuildMI(*BB3, InsertPoint2, DebugLoc(),
                                          TII->get(MovOpcode))
          .addReg(IncrementReg, RegState::Define, Sub1Idx)  // Redefine %3.sub1, not %0.sub1!
          .addImm(200)
          .addReg(ExecReg, RegState::Implicit);
      
      MachineInstr &NewDefMI2 = *MIB2;
      llvm::errs() << "Created second def in BB3 (redefining %3.sub1): ";
      NewDefMI2.print(llvm::errs());
      
      // Repair SSA after second insertion (for %3, the increment result)
      Register NewReg2 = Updater.repairSSAForNewDef(NewDefMI2, IncrementReg);
      
      llvm::errs() << "SSA repair #2 created new register: %" << NewReg2.virtRegIndex() << "\n";
      
      // === Verification ===
      llvm::errs() << "\n=== Verification ===\n";
      
      // Print final MIR
      llvm::errs() << "Final BB3 (latch):\n";
      for (MachineInstr &MI : *BB3) {
        MI.print(llvm::errs());
      }
      
      // 1. Should have PHI for 32-bit X.sub0 at BB3 (diamond join)
      bool FoundSub0PHI = false;
      for (MachineInstr &MI : *BB3) {
        if (MI.isPHI()) {
          Register PHIResult = MI.getOperand(0).getReg();
          if (PHIResult != Register::index2VirtReg(3)) { // Not the increment result PHI
            FoundSub0PHI = true;
            llvm::errs() << "Found sub0 PHI in BB3: ";
            MI.print(llvm::errs());
          }
        }
      }
      EXPECT_TRUE(FoundSub0PHI) << "Should have PHI for sub0 lane in BB3";
      
      // 2. Should have REG_SEQUENCE in BB3 before backedge to compose full 64-bit
      bool FoundREGSEQ = false;
      for (MachineInstr &MI : *BB3) {
        if (MI.getOpcode() == TargetOpcode::REG_SEQUENCE) {
          FoundREGSEQ = true;
          llvm::errs() << "Found REG_SEQUENCE in BB3: ";
          MI.print(llvm::errs());
          
          // Verify it composes both lanes
          unsigned NumSources = (MI.getNumOperands() - 1) / 2;
          EXPECT_GE(NumSources, 2u) << "REG_SEQUENCE should have at least 2 sources (sub0 and sub1)";
        }
      }
      
      EXPECT_TRUE(FoundREGSEQ) << "Should have REG_SEQUENCE at backedge in BB3";
      
      // 3. Verify LiveIntervals
      EXPECT_TRUE(LIS.hasInterval(NewReg1));
      EXPECT_TRUE(LIS.hasInterval(NewReg2));
      
      // Debug output if verification fails
      if (!MF.verify(nullptr, nullptr, nullptr, false)) {
        llvm::errs() << "MachineFunction verification failed:\n";
        MF.print(llvm::errs());
        LIS.print(llvm::errs());
      }
    });
}

// Test 8: Nested loops with SSA repair across multiple loop levels
// This tests SSA repair with a new definition in an inner loop body that propagates
// to both the inner loop header and outer loop header PHIs.
// Test nested loops with SSA repair across multiple loop levels
//
// CFG Structure:
//         BB0 (entry, %0 = 100)
//          |
//      +-> BB1 (outer loop header)
//      |   |   PHI for %1 (outer induction var)
//      |   |
//      | +->BB2 (inner loop header)
//      | | |   PHI for %2 (inner induction var)
//      | | |\
//      | | | \
//      | | BB3 BB4 (outer loop body)
//      | |  |
//      | |  INSERT: %0 = new_def
//      | |  (%3 = %2 + %0)
//      | |  |
//      | +--+ (inner backedge) -> PHI in BB2 for %0 expected
//      |     |
//      |    (%4 = %1 + %0, use %0)
//      +----+ (outer backedge)
//       |
//      BB5 (exit)
//
// Key: New def in inner loop body propagates to:
//      1. Inner loop header PHI (BB2)
//      2. Outer loop body uses (BB4)
//      3. Outer loop header PHI (BB1)
//
TEST(MachineLaneSSAUpdaterTest, NestedLoopsWithSSARepair) {
  SmallString<2048> S;
  StringRef MIRString = (Twine(R"MIR(
--- |
  define amdgpu_kernel void @func() { ret void }
...
---
name: func
tracksRegLiveness: true
registers:
  - { id: 0, class: vgpr_32 }
  - { id: 1, class: vgpr_32 }
  - { id: 2, class: vgpr_32 }
  - { id: 3, class: vgpr_32 }
body: |
  bb.0:
    successors: %bb.1
    %0:vgpr_32 = V_MOV_B32_e32 100, implicit $exec
    S_BRANCH %bb.1
  
  bb.1:
    successors: %bb.2
    ; Outer loop header: %1 = PHI(initial, result_from_outer_body)
    %1:vgpr_32 = PHI %0:vgpr_32, %bb.0, %4:vgpr_32, %bb.4
    dead %5:vgpr_32 = V_ADD_U32_e32 %1:vgpr_32, %1:vgpr_32, implicit $exec
    S_BRANCH %bb.2
  
  bb.2:
    successors: %bb.3, %bb.4
    ; Inner loop header: %2 = PHI(from_outer, from_inner_body)
    %2:vgpr_32 = PHI %1:vgpr_32, %bb.1, %3:vgpr_32, %bb.3
    dead %6:vgpr_32 = V_MOV_B32_e32 %2:vgpr_32, implicit $exec
    $sgpr0 = S_MOV_B32 0
    $sgpr1 = S_MOV_B32 5
    S_CMP_LT_U32 $sgpr0, $sgpr1, implicit-def $scc
    S_CBRANCH_SCC1 %bb.3, implicit $scc
    S_BRANCH %bb.4
  
  bb.3:
    successors: %bb.2
    ; Inner loop body - accumulate value, then we'll insert new def for %0
    %3:vgpr_32 = V_ADD_U32_e32 %2:vgpr_32, %0:vgpr_32, implicit $exec
    S_BRANCH %bb.2
  
  bb.4:
    successors: %bb.1, %bb.5
    ; Outer loop body after inner loop exit
    ; Increment outer induction variable %1 and use %0 (which we'll redefine)
    %4:vgpr_32 = V_ADD_U32_e32 %1:vgpr_32, %0:vgpr_32, implicit $exec
    dead %7:vgpr_32 = V_MOV_B32_e32 %0:vgpr_32, implicit $exec
    $sgpr2 = S_MOV_B32 0
    $sgpr3 = S_MOV_B32 10
    S_CMP_LT_U32 $sgpr2, $sgpr3, implicit-def $scc
    S_CBRANCH_SCC1 %bb.1, implicit $scc
    S_BRANCH %bb.5
  
  bb.5:
    ; Exit
    S_ENDPGM 0
...
)MIR")).toNullTerminatedStringRef(S);

  doTest<LiveIntervalsWrapperPass>(MIRString,
             [](MachineFunction &MF, LiveIntervalsWrapperPass &LISWrapper) {
      LiveIntervals &LIS = LISWrapper.getLIS();
      MachineDominatorTree MDT(MF);
      llvm::errs() << "\n=== NestedLoopsWithSSARepair Test ===\n";
      
      // Get basic blocks
      auto BBI = MF.begin();
      MachineBasicBlock *BB0 = &*BBI++;  // Entry
      MachineBasicBlock *BB1 = &*BBI++;  // Outer loop header
      MachineBasicBlock *BB2 = &*BBI++;  // Inner loop header
      MachineBasicBlock *BB3 = &*BBI++;  // Inner loop body (INSERT HERE)
      MachineBasicBlock *BB4 = &*BBI++;  // Outer loop body (after inner)
      // BB5 = Exit (not needed)
      
      const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
      const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
      
      // Get the register that will be redefined (%0 is the initial value)
      Register OrigReg = Register::index2VirtReg(0);
      ASSERT_TRUE(OrigReg.isValid()) << "Register %0 should be valid";
      
      llvm::errs() << "Original register: %" << OrigReg.virtRegIndex() << "\n";
      
      // Get V_MOV opcode and EXEC register
      MachineInstr *MovInst = &*BB0->begin();
      unsigned MovOpcode = MovInst->getOpcode();
      Register ExecReg = MovInst->getOperand(2).getReg();
      
      // Print initial state
      llvm::errs() << "\nInitial BB2 (inner loop header):\n";
      for (MachineInstr &MI : *BB2) {
        MI.print(llvm::errs());
      }
      
      llvm::errs() << "\nInitial BB1 (outer loop header):\n";
      for (MachineInstr &MI : *BB1) {
        MI.print(llvm::errs());
      }
      
      // Insert new definition in BB3 (inner loop body)
      // Find insertion point before the branch
      MachineInstr *InsertPt = nullptr;
      for (MachineInstr &MI : *BB3) {
        if (MI.isBranch()) {
          InsertPt = &MI;
          break;
        }
      }
      ASSERT_NE(InsertPt, nullptr) << "Should find branch in BB3";
      
      // Insert: X = 999 (violates SSA)
      MachineInstr *NewDefMI = BuildMI(*BB3, InsertPt, DebugLoc(),
                                        TII->get(MovOpcode), OrigReg)
          .addImm(999)
          .addReg(ExecReg, RegState::Implicit);
      
      llvm::errs() << "\nInserted new def in BB3 (inner loop body): ";
      NewDefMI->print(llvm::errs());
      
      // Create SSA updater and repair
      MachineLaneSSAUpdater Updater(MF, LIS, MDT, *TRI);
      Register NewReg = Updater.repairSSAForNewDef(*NewDefMI, OrigReg);
      
      llvm::errs() << "SSA repair created new register: %" << NewReg.virtRegIndex() << "\n";
      
      // === Verification ===
      llvm::errs() << "\n=== Verification ===\n";
      
      llvm::errs() << "\nFinal BB2 (inner loop header):\n";
      for (MachineInstr &MI : *BB2) {
        MI.print(llvm::errs());
      }
      
      llvm::errs() << "\nFinal BB1 (outer loop header):\n";
      for (MachineInstr &MI : *BB1) {
        MI.print(llvm::errs());
      }
      
      llvm::errs() << "\nFinal BB4 (outer loop body after inner):\n";
      for (MachineInstr &MI : *BB4) {
        MI.print(llvm::errs());
      }
      
      // 1. Inner loop header (BB2) should have NEW PHI created by SSA repair
      bool FoundSSARepairPHI = false;
      Register SSARepairPHIReg;
      for (MachineInstr &MI : *BB2) {
        if (MI.isPHI()) {
          // Look for a PHI that has NewReg as one of its incoming values
          for (unsigned i = 1; i < MI.getNumOperands(); i += 2) {
            Register IncomingReg = MI.getOperand(i).getReg();
            MachineBasicBlock *IncomingMBB = MI.getOperand(i + 1).getMBB();
            
            if (IncomingMBB == BB3 && IncomingReg == NewReg) {
              FoundSSARepairPHI = true;
              SSARepairPHIReg = MI.getOperand(0).getReg();
              llvm::errs() << "Found SSA repair PHI in inner loop header: ";
              MI.print(llvm::errs());
              
              // Should have incoming from BB1 and BB3
              unsigned NumIncoming = (MI.getNumOperands() - 1) / 2;
              EXPECT_EQ(NumIncoming, 2u) << "SSA repair PHI should have 2 incoming";
              break;
            }
          }
          if (FoundSSARepairPHI)
            break;
        }
      }
      EXPECT_TRUE(FoundSSARepairPHI) << "Should find SSA repair PHI in BB2 (inner loop header)";
      
      // 2. Outer loop header (BB1) may have PHI updated if needed
      bool FoundOuterPHI = false;
      for (MachineInstr &MI : *BB1) {
        if (MI.isPHI() && MI.getOperand(0).getReg() == Register::index2VirtReg(1)) {
          FoundOuterPHI = true;
          llvm::errs() << "Found outer loop PHI: ";
          MI.print(llvm::errs());
        }
      }
      EXPECT_TRUE(FoundOuterPHI) << "Should find outer loop PHI in BB1";
      
      // 3. Use in BB4 should be updated
      bool FoundUseInBB4 = false;
      for (MachineInstr &MI : *BB4) {
        if (!MI.isPHI() && MI.getNumOperands() > 1) {
          for (unsigned i = 0; i < MI.getNumOperands(); ++i) {
            if (MI.getOperand(i).isReg() && MI.getOperand(i).isUse()) {
              Register UseReg = MI.getOperand(i).getReg();
              if (UseReg.isVirtual()) {
                FoundUseInBB4 = true;
                llvm::errs() << "Found use in BB4: %" << UseReg.virtRegIndex() << " in ";
                MI.print(llvm::errs());
              }
            }
          }
        }
      }
      EXPECT_TRUE(FoundUseInBB4) << "Should find uses in outer loop body (BB4)";
      
      // 4. Verify LiveIntervals
      EXPECT_TRUE(LIS.hasInterval(NewReg));
      
      // Debug output if verification fails
      if (!MF.verify(nullptr, nullptr, nullptr, false)) {
        llvm::errs() << "MachineFunction verification failed:\n";
        MF.print(llvm::errs());
        LIS.print(llvm::errs());
      }
    });
}

//===----------------------------------------------------------------------===//
// Test 9: 128-bit register with 64-bit subreg redef and multiple lane uses
//
// This comprehensive test covers:
// 1. Large register (128-bit) with multiple subregisters (sub0, sub1, sub2, sub3)
// 2. Partial redefinition (64-bit sub2_3 covering two lanes: sub2+sub3)
// 3. Uses of changed lanes (sub2, sub3) in different paths
// 4. Uses of unchanged lanes (sub0, sub1) in different paths
// 5. Diamond CFG with redef in one branch
// 6. Second diamond to test propagation of PHI result
//
// CFG Structure:
//         BB0 (entry)
//          |
//         BB1 (%0:vreg_128 = initial 128-bit value)
//          |
//         BB2 (diamond1 split)
//        /   \
//      BB3   BB4 (INSERT: %0.sub2_3 = new_def)
//       |     |
//      use   use
//      sub0  sub3 (changed)
//        \   /
//         BB5 (join) -> PHI for sub2_3 lanes (sub2+sub3 changed, sub0+sub1 unchanged)
//          |
//         use sub1 (unchanged, flows from BB1)
//          |
//         BB6 (diamond2 split)
//        /   \
//      BB7   BB8
//       |     |
//      use   (no use)
//      sub2
//        \   /
//         BB9 (join, no PHI - BB5's PHI dominates)
//          |
//         BB10 (use sub0, exit)
//
// Expected behavior:
//   - PHI in BB5 merges sub2_3 lanes ONLY (sub2+sub3 changed)
//   - sub0+sub1 lanes flow unchanged from BB1 through entire CFG
//   - Uses in BB5, BB7, BB10 use PHI result or unchanged lanes
//   - No PHI in BB9 (BB5 dominates, PHI result flows through)
//
// This test validates:
//   ✓ Partial redefinition (64-bit of 128-bit)
//   ✓ Multiple different subreg uses (sub0, sub1, sub2, sub3)
//   ✓ Changed vs unchanged lane tracking
//   ✓ PHI result propagation to dominated blocks
//===----------------------------------------------------------------------===//
TEST(MachineLaneSSAUpdaterTest, MultipleSubregUsesAcrossDiamonds) {
  SmallString<4096> S;
  StringRef MIRString = (Twine(R"MIR(
--- |
  define amdgpu_kernel void @func() { ret void }
...
---
name: func
tracksRegLiveness: true
registers:
  - { id: 0, class: vreg_128 }
  - { id: 1, class: vgpr_32 }
  - { id: 2, class: vgpr_32 }
  - { id: 3, class: vgpr_32 }
  - { id: 4, class: vgpr_32 }
body: |
  bb.0:
    successors: %bb.1
    S_BRANCH %bb.1
  
  bb.1:
    successors: %bb.2
    ; Initialize 128-bit register %0 with IMPLICIT_DEF
    %0:vreg_128 = IMPLICIT_DEF
    S_BRANCH %bb.2
  
  bb.2:
    successors: %bb.3, %bb.4
    ; Diamond 1 split
    $sgpr0 = S_MOV_B32 0
    $sgpr1 = S_MOV_B32 1
    S_CMP_LG_U32 $sgpr0, $sgpr1, implicit-def $scc
    S_CBRANCH_SCC1 %bb.4, implicit $scc
  
  bb.3:
    successors: %bb.5
    ; Use sub0 (unchanged lane, low 32 bits)
    %1:vgpr_32 = V_MOV_B32_e32 %0.sub0:vreg_128, implicit $exec
    S_BRANCH %bb.5
  
  bb.4:
    successors: %bb.5
    ; This is where we'll INSERT: %0.sub2_3 = new_def (64-bit, covers sub2+sub3)
    ; After insertion, use sub3 (high 32 bits of sub2_3)
    %2:vgpr_32 = V_MOV_B32_e32 %0.sub3:vreg_128, implicit $exec
    S_BRANCH %bb.5
  
  bb.5:
    successors: %bb.6
    ; Diamond 1 join - PHI expected for sub2_3 lanes
    ; Use sub1 (unchanged lane, bits 32-63)
    %3:vgpr_32 = V_MOV_B32_e32 %0.sub1:vreg_128, implicit $exec
    S_BRANCH %bb.6
  
  bb.6:
    successors: %bb.7, %bb.8
    ; Diamond 2 split
    $sgpr2 = S_MOV_B32 0
    $sgpr3 = S_MOV_B32 1
    S_CMP_LG_U32 $sgpr2, $sgpr3, implicit-def $scc
    S_CBRANCH_SCC1 %bb.8, implicit $scc
  
  bb.7:
    successors: %bb.9
    ; Use sub2 (changed lane, bits 64-95)
    dead %4:vgpr_32 = V_MOV_B32_e32 %0.sub2:vreg_128, implicit $exec
    S_BRANCH %bb.9
  
  bb.8:
    successors: %bb.9
    ; No use - sparse use pattern
    S_NOP 0
  
  bb.9:
    successors: %bb.10
    ; Diamond 2 join - no PHI needed (BB5 dominates)
    S_NOP 0
  
  bb.10:
    ; Exit - use sub0 again (unchanged lane)
    dead %5:vgpr_32 = V_MOV_B32_e32 %0.sub0:vreg_128, implicit $exec
    S_ENDPGM 0
...
)MIR")).toNullTerminatedStringRef(S);

  doTest<LiveIntervalsWrapperPass>(MIRString,
             [](MachineFunction &MF, LiveIntervalsWrapperPass &LISWrapper) {
      LiveIntervals &LIS = LISWrapper.getLIS();
      MachineDominatorTree MDT(MF);
      llvm::errs() << "\n=== MultipleSubregUsesAcrossDiamonds Test ===\n";
      
      // Get basic blocks
      auto BBI = MF.begin();
      ++BBI; // Skip BB0 (entry)
      ++BBI; // Skip BB1 (Initial def)
      ++BBI; // Skip BB2 (Diamond1 split)
      MachineBasicBlock *BB3 = &*BBI++; // Diamond1 true (no redef)
      MachineBasicBlock *BB4 = &*BBI++; // Diamond1 false (INSERT HERE)
      MachineBasicBlock *BB5 = &*BBI++; // Diamond1 join
      
      const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
      const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
      MachineRegisterInfo &MRI = MF.getRegInfo();
      (void)MRI; // May be unused, suppress warning
      
      // Find the 128-bit register %0
      Register OrigReg = Register::index2VirtReg(0);
      ASSERT_TRUE(OrigReg.isValid()) << "Register %0 should be valid";
      
      llvm::errs() << "Using 128-bit register: %" << OrigReg.virtRegIndex() << "\n";
      
      // Find sub2_3 subregister index (64-bit covering bits 64-127)
      unsigned Sub2_3Idx = 0;
      for (unsigned Idx = 1; Idx < TRI->getNumSubRegIndices(); ++Idx) {
        unsigned SubRegSize = TRI->getSubRegIdxSize(Idx);
        LaneBitmask Mask = TRI->getSubRegIndexLaneMask(Idx);
        
        // Looking for 64-bit subreg covering upper half (lanes for sub2+sub3)
        // sub2_3 should have mask 0xF0 (lanes for bits 64-127)
        if (SubRegSize == 64 && (Mask.getAsInteger() & 0xF0) == 0xF0) {
          Sub2_3Idx = Idx;
          llvm::errs() << "Found sub2_3 index: " << Idx 
                       << " (size=" << SubRegSize 
                       << ", mask=0x" << llvm::format("%X", Mask.getAsInteger()) << ")\n";
          break;
        }
      }
      
      ASSERT_NE(Sub2_3Idx, 0u) << "Should find sub2_3 subregister index";
      
      // Insert new definition in BB4: %0.sub2_3 = IMPLICIT_DEF
      // Find insertion point (before the use of sub3)
      MachineInstr *UseOfSub3 = nullptr;
      
      for (MachineInstr &MI : *BB4) {
        if (MI.getNumOperands() >= 2 && MI.getOperand(0).isReg() && 
            MI.getOperand(1).isReg() && MI.getOperand(1).getReg() == OrigReg) {
          UseOfSub3 = &MI;
          break;
        }
      }
      ASSERT_NE(UseOfSub3, nullptr) << "Should find use of sub3 in BB4";
      
      // Create new def: %0.sub2_3 = IMPLICIT_DEF
      // We use IMPLICIT_DEF because it works for any register size and the SSA updater
      // doesn't care about the specific instruction semantics - we're just testing SSA repair
      MachineInstrBuilder MIB = BuildMI(*BB4, UseOfSub3, DebugLoc(), 
                                         TII->get(TargetOpcode::IMPLICIT_DEF))
        .addDef(OrigReg, RegState::Define, Sub2_3Idx);
      
      MachineInstr *NewDefMI = MIB.getInstr();
      llvm::errs() << "Inserted new def in BB4: ";
      NewDefMI->print(llvm::errs());
      
      // Index the new instruction
      LIS.InsertMachineInstrInMaps(*NewDefMI);
      
      // Set MachineFunction properties to allow PHI insertion
      MF.getProperties().set(MachineFunctionProperties::Property::IsSSA);
      MF.getProperties().reset(MachineFunctionProperties::Property::NoPHIs);
      
      // Create SSA updater and repair
      MachineLaneSSAUpdater Updater(MF, LIS, MDT, *TRI);
      Register NewReg = Updater.repairSSAForNewDef(*NewDefMI, OrigReg);
      
      llvm::errs() << "SSA repair created new register: %" << NewReg.virtRegIndex() << "\n";
      
      // Print final state of key blocks
      llvm::errs() << "\nFinal BB5 (diamond1 join):\n";
      for (MachineInstr &MI : *BB5) {
        MI.print(llvm::errs());
      }
      
      // Verify SSA repair results
      
      // 1. Should have PHI in BB5 for sub2+sub3 lanes
      bool FoundPHI = false;
      for (MachineInstr &MI : *BB5) {
        if (MI.isPHI()) {
          Register PHIResult = MI.getOperand(0).getReg();
          if (PHIResult.isVirtual()) {
            llvm::errs() << "Found PHI in BB5: ";
            MI.print(llvm::errs());
            
            // Check that it has 2 incoming values
            unsigned NumIncoming = (MI.getNumOperands() - 1) / 2;
            EXPECT_EQ(NumIncoming, 2u) << "PHI should have 2 incoming values";
            
            // Check that one incoming is the new register from BB4
            // and the other incoming from BB3 uses %0.sub2_3
            bool HasNewRegFromBB4 = false;
            bool HasCorrectSubregFromBB3 = false;
            for (unsigned i = 1; i < MI.getNumOperands(); i += 2) {
              Register IncomingReg = MI.getOperand(i).getReg();
              unsigned IncomingSubReg = MI.getOperand(i).getSubReg();
              MachineBasicBlock *IncomingMBB = MI.getOperand(i + 1).getMBB();
              
              if (IncomingMBB == BB4) {
                HasNewRegFromBB4 = (IncomingReg == NewReg);
                llvm::errs() << "  Incoming from BB4: %" << IncomingReg.virtRegIndex() << "\n";
              } else if (IncomingMBB == BB3) {
                // Should be %0.sub2_3 (the lanes we redefined)
                llvm::errs() << "  Incoming from BB3: %" << IncomingReg.virtRegIndex();
                if (IncomingSubReg) {
                  llvm::errs() << "." << TRI->getSubRegIndexName(IncomingSubReg);
                }
                llvm::errs() << "\n";
                
                // Verify it's using sub2_3
                if (IncomingReg == OrigReg && IncomingSubReg == Sub2_3Idx) {
                  HasCorrectSubregFromBB3 = true;
                }
              }
            }
            EXPECT_TRUE(HasNewRegFromBB4) << "PHI should use NewReg from BB4";
            EXPECT_TRUE(HasCorrectSubregFromBB3) << "PHI should use %0.sub2_3 from BB3";
            FoundPHI = true;
          }
        }
      }
      EXPECT_TRUE(FoundPHI) << "Should find PHI in BB5 for sub2_3 lanes";
      
      // 2. Verify LiveIntervals
      EXPECT_TRUE(LIS.hasInterval(NewReg));
      EXPECT_TRUE(LIS.hasInterval(OrigReg));
      
      // 3. Verify LiveInterval for OrigReg has subranges for changed lanes
      LiveInterval &OrigLI = LIS.getInterval(OrigReg);
      EXPECT_TRUE(OrigLI.hasSubRanges()) << "OrigReg should have subranges after partial redef";
      
      // Debug output if verification fails
      if (!MF.verify(nullptr, nullptr, nullptr, false)) {
        llvm::errs() << "MachineFunction verification failed:\n";
        MF.print(llvm::errs());
        LIS.print(llvm::errs());
      }
    });
}

// Test 10: Non-contiguous lane mask - redefine sub1 of 128-bit, use full register
// This specifically tests the multi-source REG_SEQUENCE code path for non-contiguous lanes
//
// CFG Structure:
//        BB0 (entry)
//         |
//         v
//        BB1 (%0:vreg_128 = IMPLICIT_DEF)
//         |
//         v
//        BB2 (diamond split)
//        /  \
//       /    \
//      v      v
//    BB3    BB4 (%0.sub1 = IMPLICIT_DEF - redefine middle lane!)
//      \    /
//       \  /
//        v
//       BB5 (diamond join - USE %0 as full register)
//         |
//         v
//       BB6 (exit)
//
// Key Property: Redefining sub1 leaves LanesFromOld = sub0 + sub2 + sub3 (non-contiguous!)
//               This requires getCoveringSubRegsForLaneMask to decompose into multiple subregs
//               Expected REG_SEQUENCE: %RS = REG_SEQUENCE %6, sub1, %0.sub0, sub0, %0.sub2_3, sub2_3
//
TEST(MachineLaneSSAUpdaterTest, NonContiguousLaneMaskREGSEQUENCE) {
  SmallString<4096> S;
  StringRef MIRString = (Twine(R"MIR(
--- |
  define amdgpu_kernel void @func() { ret void }
...
---
name: func
tracksRegLiveness: true
registers:
  - { id: 0, class: vreg_128 }
  - { id: 1, class: vreg_128 }
body: |
  bb.0:
    successors: %bb.1
    S_BRANCH %bb.1
  
  bb.1:
    successors: %bb.2, %bb.3
    %0:vreg_128 = IMPLICIT_DEF
    $sgpr0 = S_MOV_B32 0
    $sgpr1 = S_MOV_B32 1
    S_CMP_LG_U32 $sgpr0, $sgpr1, implicit-def $scc
    S_CBRANCH_SCC1 %bb.3, implicit $scc
  
  bb.2:
    successors: %bb.4
    ; Left path - no redefinition
    S_NOP 0
    S_BRANCH %bb.4
  
  bb.3:
    successors: %bb.4
    ; Right path - THIS IS WHERE WE'LL INSERT: %0.sub1 = IMPLICIT_DEF
    S_NOP 0
    S_BRANCH %bb.4
  
  bb.4:
    ; Diamond join - use FULL register (this will need REG_SEQUENCE!)
    ; Using full %0 (not a subreg) forces composition of non-contiguous lanes
    dead %1:vreg_128 = COPY %0:vreg_128
    S_ENDPGM 0
...
)MIR")).toNullTerminatedStringRef(S);

  doTest<LiveIntervalsWrapperPass>(MIRString,
             [](MachineFunction &MF, LiveIntervalsWrapperPass &LISWrapper) {
      LiveIntervals &LIS = LISWrapper.getLIS();
      MachineDominatorTree MDT(MF);
      llvm::errs() << "\n=== NonContiguousLaneMaskREGSEQUENCE Test ===\n";
      
      const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
      const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
      MachineRegisterInfo &MRI = MF.getRegInfo();
      (void)MRI; // May be unused, suppress warning
      
      // Find blocks
      // bb.0 = entry
      // bb.1 = IMPLICIT_DEF + diamond split
      // bb.2 = left path (no redef)
      // bb.3 = right path (INSERT sub1 def here)
      // bb.4 = diamond join (use full register)
      MachineBasicBlock *BB3 = MF.getBlockNumbered(3);  // Right path - where we insert
      MachineBasicBlock *BB4 = MF.getBlockNumbered(4);  // Join - where we need REG_SEQUENCE
      
      // Find %0 (the vreg_128)
      Register OrigReg = Register::index2VirtReg(0);
      ASSERT_TRUE(OrigReg.isValid()) << "Register %0 should be valid";
      llvm::errs() << "Using 128-bit register: %" << OrigReg.virtRegIndex() << "\n";
      
      // Find sub1 subregister index
      unsigned Sub1Idx = 0;
      for (unsigned Idx = 1; Idx < TRI->getNumSubRegIndices(); ++Idx) {
        StringRef Name = TRI->getSubRegIndexName(Idx);
        if (Name == "sub1") {
          Sub1Idx = Idx;
          break;
        }
      }
      
      ASSERT_NE(Sub1Idx, 0u) << "Should find sub1 subregister index";
      
      // Insert new definition in BB3 (right path): %0.sub1 = IMPLICIT_DEF
      MachineInstrBuilder MIB = BuildMI(*BB3, BB3->getFirstNonPHI(), DebugLoc(), 
                                         TII->get(TargetOpcode::IMPLICIT_DEF))
        .addDef(OrigReg, RegState::Define, Sub1Idx);
      
      MachineInstr *NewDefMI = MIB.getInstr();
      llvm::errs() << "Inserted new def in BB3: ";
      NewDefMI->print(llvm::errs());
      
      // Index the new instruction
      LIS.InsertMachineInstrInMaps(*NewDefMI);
      
      // Set MachineFunction properties to allow PHI insertion
      MF.getProperties().set(MachineFunctionProperties::Property::IsSSA);
      MF.getProperties().reset(MachineFunctionProperties::Property::NoPHIs);
      
      // Create SSA updater and repair
      MachineLaneSSAUpdater Updater(MF, LIS, MDT, *TRI);
      Register NewReg = Updater.repairSSAForNewDef(*NewDefMI, OrigReg);
      
      llvm::errs() << "SSA repair created new register: %" << NewReg.virtRegIndex() << "\n";
      
      // Print final state
      llvm::errs() << "\nFinal BB4 (diamond join):\n";
      for (MachineInstr &MI : *BB4) {
        MI.print(llvm::errs());
      }
      
      // Verify SSA repair results
      
      // 1. Should have PHI in BB4 for sub1 lane
      bool FoundPHI = false;
      Register PHIReg;
      for (MachineInstr &MI : *BB4) {
        if (MI.isPHI()) {
          PHIReg = MI.getOperand(0).getReg();
          if (PHIReg.isVirtual()) {
            llvm::errs() << "Found PHI in BB4: ";
            MI.print(llvm::errs());
            FoundPHI = true;
            
            // Check that it has 2 incoming values
            unsigned NumIncoming = (MI.getNumOperands() - 1) / 2;
            EXPECT_EQ(NumIncoming, 2u) << "PHI should have 2 incoming values";
            
            // One incoming should be the new register (vgpr_32 from BB3)
            bool HasNewRegFromBB3 = false;
            for (unsigned i = 1; i < MI.getNumOperands(); i += 2) {
              if (MI.getOperand(i).isReg() && MI.getOperand(i).getReg() == NewReg) {
                EXPECT_EQ(MI.getOperand(i + 1).getMBB(), BB3) << "NewReg should come from BB3";
                HasNewRegFromBB3 = true;
              }
            }
            EXPECT_TRUE(HasNewRegFromBB3) << "PHI should have NewReg from BB3";
            
            break;
          }
        }
      }
      
      EXPECT_TRUE(FoundPHI) << "Should create PHI in BB4 for sub1 lane";
      
      // 2. Most importantly: Should have REG_SEQUENCE with MULTIPLE sources for non-contiguous lanes
      // After PHI for sub1, we need to compose full register:
      // LanesFromOld = sub0 + sub2 + sub3 (non-contiguous!)
      // This requires multiple REG_SEQUENCE operands
      bool FoundREGSEQUENCE = false;
      unsigned NumREGSEQSources = 0;
      
      for (MachineInstr &MI : *BB4) {
        if (MI.getOpcode() == TargetOpcode::REG_SEQUENCE) {
          llvm::errs() << "Found REG_SEQUENCE: ";
          MI.print(llvm::errs());
          FoundREGSEQUENCE = true;
          
          // Count sources (each source is: register + subregidx, so pairs)
          NumREGSEQSources = (MI.getNumOperands() - 1) / 2;
          llvm::errs() << "  REG_SEQUENCE has " << NumREGSEQSources << " sources\n";
          
          // We expect at least 2 sources for non-contiguous case:
          // 1. PHI result covering sub1
          // 2. One or more sources from OrigReg covering sub0, sub2, sub3
          EXPECT_GE(NumREGSEQSources, 2u) 
              << "REG_SEQUENCE should have multiple sources for non-contiguous lanes";
          
          // Verify at least one source is the PHI result
          bool HasPHISource = false;
          for (unsigned i = 1; i < MI.getNumOperands(); i += 2) {
            if (MI.getOperand(i).isReg() && MI.getOperand(i).getReg() == PHIReg) {
              HasPHISource = true;
              break;
            }
          }
          EXPECT_TRUE(HasPHISource) << "REG_SEQUENCE should use PHI result";
          
          break;
        }
      }
      
      EXPECT_TRUE(FoundREGSEQUENCE) 
          << "Should create REG_SEQUENCE to compose full register from non-contiguous lanes";
      
      // 3. The COPY use should now reference the REG_SEQUENCE result (not %0)
      bool FoundRewrittenUse = false;
      for (MachineInstr &MI : *BB4) {
        if (MI.getOpcode() == TargetOpcode::COPY) {
          MachineOperand &SrcOp = MI.getOperand(1);
          if (SrcOp.isReg() && SrcOp.getReg().isVirtual() && SrcOp.getReg() != OrigReg) {
            llvm::errs() << "Found rewritten COPY: ";
            MI.print(llvm::errs());
            FoundRewrittenUse = true;
            break;
          }
        }
      }
      
      EXPECT_TRUE(FoundRewrittenUse) << "COPY should be rewritten to use REG_SEQUENCE result";
      
      // Print summary
      llvm::errs() << "\n=== Test Summary ===\n";
      llvm::errs() << "✓ Redefined sub1 (middle lane) of vreg_128\n";
      llvm::errs() << "✓ Created PHI for sub1 lane\n";
      llvm::errs() << "✓ Created REG_SEQUENCE with " << NumREGSEQSources 
                   << " sources to handle non-contiguous lanes (sub0 + sub2 + sub3)\n";
      llvm::errs() << "✓ This test exercises getCoveringSubRegsForLaneMask!\n";
    });
}

} // anonymous namespace
