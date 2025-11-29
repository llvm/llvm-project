//===- MachineLaneSSAUpdaterSpillReloadTest.cpp - Spill/Reload tests -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for MachineLaneSSAUpdater focusing on spill/reload scenarios.
//
// NOTE: This file is currently a placeholder for future spiller-specific tests.
// Analysis showed that repairSSAForNewDef() is sufficient for spill/reload 
// scenarios - no special spill handling is needed. The spiller workflow is:
//   1. Insert reload instruction before use
//   2. Call repairSSAForNewDef(ReloadMI, SpilledReg)
//   3. Done! Uses are rewritten, LiveIntervals naturally pruned
//
// Future spiller-specific scenarios (if needed) can be added here.
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
struct SpillReloadTestPass : public MachineFunctionPass {
  static char ID;
  SpillReloadTestPass() : MachineFunctionPass(ID) {}
};

char SpillReloadTestPass::ID = 0;

namespace llvm {
  void initializeSpillReloadTestPassPass(PassRegistry &);
}

INITIALIZE_PASS(SpillReloadTestPass, "spillreloadtestpass", 
                "spillreloadtestpass", false, false)

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
struct SpillReloadTestPassT : public SpillReloadTestPass {
  typedef std::function<void(MachineFunction&, AnalysisType&)> TestFx;

  SpillReloadTestPassT() {
    // We should never call this but always use PM.add(new SpillReloadTestPass(...))
    abort();
  }
  
  SpillReloadTestPassT(TestFx T, bool ShouldPass)
      : T(T), ShouldPass(ShouldPass) {
    initializeSpillReloadTestPassPass(*PassRegistry::getPassRegistry());
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
                   typename SpillReloadTestPassT<AnalysisType>::TestFx T,
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

  PM.add(new SpillReloadTestPassT<AnalysisType>(T, ShouldPass));

  PM.run(*M);
}

static void liveIntervalsTest(StringRef MIRFunc,
                              SpillReloadTestPassT<LiveIntervalsWrapperPass>::TestFx T,
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
// Spill/Reload Tests
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Test 1: Simple Linear Spill/Reload
//===----------------------------------------------------------------------===//
//
// This test demonstrates that repairSSAForNewDef() works for spill/reload
// scenarios without any special handling.
//
// CFG Structure:
//    BB0 (entry)
//     |   %0 = initial_def
//     |
//    BB1 (intermediate)
//     |   some operations
//     |
//    BB2 (reload & use)
//     |   %0 = RELOAD (simulated as V_MOV_B32)
//     |   use %0
//
// Scenario:
// - %0 is defined in BB0 and used in BB2
// - Insert a reload instruction in BB2 that redefines %0 (violating SSA)
// - Call repairSSAForNewDef() to fix the SSA violation
// - Verify that uses are rewritten and LiveIntervals are correct
//
// Expected Behavior:
// - Reload renamed to define a new register
// - Uses after reload rewritten to new register
// - OrigReg's LiveInterval naturally pruned to BB0 only
// - No PHI needed (linear CFG)
//
TEST(MachineLaneSSAUpdaterSpillReloadTest, SimpleLinearSpillReload) {
  liveIntervalsTest(R"MIR(
    %0:vgpr_32 = V_MOV_B32_e32 42, implicit $exec
    S_BRANCH %bb.1

  bb.1:
    successors: %bb.2
    %1:vgpr_32 = V_MOV_B32_e32 100, implicit $exec
    S_BRANCH %bb.2

  bb.2:
    %2:vgpr_32 = V_ADD_U32_e32 %0, %1, implicit $exec
    S_ENDPGM 0
)MIR",
    [](MachineFunction &MF, LiveIntervalsWrapperPass &LISWrapper) {
      LiveIntervals &LIS = LISWrapper.getLIS();
      MachineDominatorTree MDT(MF);
      const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
      
      // Verify we have 3 blocks as expected
      ASSERT_EQ(MF.size(), 3u) << "Should have bb.0, bb.1, bb.2";
      
      MachineBasicBlock *BB0 = MF.getBlockNumbered(0);
      MachineBasicBlock *BB2 = MF.getBlockNumbered(2);
      
      // Find %0 definition in BB0 (first instruction should be V_MOV_B32)
      MachineInstr *OrigDefMI = &*BB0->begin();
      ASSERT_TRUE(OrigDefMI && OrigDefMI->getNumOperands() > 0);
      Register OrigReg = OrigDefMI->getOperand(0).getReg();
      ASSERT_TRUE(OrigReg.isValid()) << "Should have valid original register %0";
      
      // STEP 1: Insert reload instruction in BB2 before the use
      // This creates a second definition of %0, violating SSA
      const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
      auto InsertPt = BB2->getFirstNonPHI();
      
      // Get opcode and register from the existing V_MOV_B32 in BB0
      unsigned MovOpcode = OrigDefMI->getOpcode();
      Register ExecReg = OrigDefMI->getOperand(2).getReg();
      
      // Insert reload: %0 = V_MOV_B32 999 (simulating load from stack)
      // This violates SSA because %0 is already defined in BB0
      MachineInstr *ReloadMI = BuildMI(*BB2, InsertPt, DebugLoc(),
                                        TII->get(MovOpcode), OrigReg)
                                   .addImm(999)  // Simulated reload value
                                   .addReg(ExecReg, RegState::Implicit);
      
      // Set MachineFunction properties to allow SSA
      MF.getProperties().set(MachineFunctionProperties::Property::IsSSA);
      MF.getProperties().reset(MachineFunctionProperties::Property::NoPHIs);
      
      // STEP 2: Call repairSSAForNewDef to fix the SSA violation
      // This will:
      //   - Rename the reload to define a new register
      //   - Rewrite uses dominated by the reload
      //   - Naturally prune OrigReg's LiveInterval via recomputation
      MachineLaneSSAUpdater Updater(MF, LIS, MDT, *TRI);
      Register ReloadReg = Updater.repairSSAForNewDef(*ReloadMI, OrigReg);
      
      // VERIFY RESULTS:
      
      // 1. ReloadReg should be valid and different from OrigReg
      EXPECT_TRUE(ReloadReg.isValid()) << "Updater should return valid register";
      EXPECT_NE(ReloadReg, OrigReg) << "Reload register should be different from original";
      
      // 2. ReloadMI should define the new ReloadReg (not OrigReg)
      EXPECT_EQ(ReloadMI->getOperand(0).getReg(), ReloadReg) 
          << "ReloadMI should define new reload register";
      
      // 3. Verify the ReloadReg has a valid LiveInterval
      EXPECT_TRUE(LIS.hasInterval(ReloadReg)) 
          << "Reload register should have live interval";
      
      // 4. No PHI should be inserted (linear CFG, reload dominates subsequent uses)
      bool FoundPHI = false;
      for (MachineBasicBlock &MBB : MF) {
        for (MachineInstr &MI : MBB) {
          if (MI.isPHI()) {
            FoundPHI = true;
            break;
          }
        }
      }
      EXPECT_FALSE(FoundPHI) 
          << "Linear CFG should not require PHI nodes";
      
      // 5. Verify OrigReg's LiveInterval was naturally pruned
      //    It should only cover BB0 now (definition to end of block)
      EXPECT_TRUE(LIS.hasInterval(OrigReg)) 
          << "Original register should still have live interval";
      const LiveInterval &OrigLI = LIS.getInterval(OrigReg);
      
      // The performSSARepair recomputation naturally prunes OrigReg
      // because all uses in BB2 were rewritten to ReloadReg
      SlotIndex OrigEnd = OrigLI.endIndex();
      
      // OrigReg should not extend into BB2 where ReloadReg took over
      SlotIndex BB2Start = LIS.getMBBStartIdx(BB2);
      EXPECT_LE(OrigEnd, BB2Start) 
          << "Original register should not extend into BB2 after reload";
    });
}

} // anonymous namespace

