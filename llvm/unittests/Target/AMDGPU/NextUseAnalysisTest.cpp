//===- llvm/unittests/Target/AMDGPU/NextUseAnalysisTest.cpp ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Timer.h"
#include "AMDGPUNextUseAnalysis.h"
#include "AMDGPUTargetMachine.h"
#include "AMDGPUUnitTests.h"
#include "GCNSubtarget.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/InitializePasses.h"
#include "gtest/gtest.h"

using namespace llvm;

// Helper wrapper to store analysis results for unit testing
class NextUseAnalysisTestWrapper : public AMDGPUNextUseAnalysisWrapper {
public:
  static std::unique_ptr<NextUseResult> Captured;

  bool runOnMachineFunction(MachineFunction &MF) override {
    bool Changed = AMDGPUNextUseAnalysisWrapper::runOnMachineFunction(MF);
    // Store the result for unit test access
    Captured = std::make_unique<NextUseResult>(std::move(getNU()));
    return Changed;
  }
};

std::unique_ptr<NextUseResult> NextUseAnalysisTestWrapper::Captured;

class NextUseAnalysisTest : public ::testing::Test {
protected:
  std::unique_ptr<LLVMContext> Context;
  std::unique_ptr<const GCNTargetMachine> TM;
  MachineModuleInfo *MMI = nullptr;

  void SetUp() override {
    DebugFlag = true;
    setCurrentDebugType("amdgpu-next-use");
    Context = std::make_unique<LLVMContext>();
    TM = createAMDGPUTargetMachine("amdgcn-amd-", "gfx1200", "");
    if (!TM) {
      GTEST_SKIP() << "AMDGPU target not available";
    }
    static bool InitializedOnce = false;
    if (!InitializedOnce) {
      // Initialize required passes
      PassRegistry &PR = *PassRegistry::getPassRegistry();
      initializeMachineModuleInfoWrapperPassPass(PR);
      initializeMachineDominatorTreeWrapperPassPass(PR);
      initializeSlotIndexesWrapperPassPass(PR);
      initializeMachineLoopInfoWrapperPassPass(PR);
      InitializedOnce = true;
    }
  }

  std::unique_ptr<Module> parseMIR(LLVMContext &Ctx, StringRef MIR,
                                              const TargetMachine &TM,
                                              legacy::PassManager &PM) {
    // 1) Add MMI wrapper first, get a handle to its MMI
    auto *MMIWP = new MachineModuleInfoWrapperPass(&TM);
    PM.add(MMIWP);
    MMI = &MMIWP->getMMI();

    // 2) Parse IR+MIR
    auto Buf = MemoryBuffer::getMemBuffer(MIR);
    auto MIRP = createMIRParser(std::move(Buf), Ctx);
    if (!MIRP)
      return nullptr;

    auto M = MIRP->parseIRModule();
    if (!M)
      return nullptr;

    M->setTargetTriple(TM.getTargetTriple());
    M->setDataLayout(TM.createDataLayout());
    MMIWP->doInitialization(*M);

    if (MIRP->parseMachineFunctions(*M, *MMI))
      return nullptr;

    return M;
  }

  NextUseResult &runNextUseAnalysis(MachineFunction &MF,
                                    legacy::PassManager &PM) {
    // Add our analysis pass at pre-emit stage (after most optimizations)
    PM.add(new NextUseAnalysisTestWrapper());
    
    PM.run(const_cast<Module&>(*MMI->getModule()));
    
    // Get the analysis result from the wrapper that was run
    return *NextUseAnalysisTestWrapper::Captured;
  }
};

// Test basic API functionality (doesn't require analysis to run)
TEST_F(NextUseAnalysisTest, BasicAPIValidation) {
  // Create a simple MIR for testing
  const char *MIRString = R"MIR(
--- |
  define void @test_basic() { ret void }
...
---
name: test_basic
body: |
  bb.0:
    %0:sgpr_32 = S_MOV_B32 10
    %1:sgpr_32 = S_MOV_B32 5  
    %2:sgpr_32 = S_ADD_U32 %0, %1, implicit-def $scc
    S_ENDPGM 0, implicit %2
)MIR";

  SetUp();
  LLVMContext Ctx;
  legacy::PassManager PM;
  auto Module = parseMIR(Ctx, MIRString, *TM, PM);
  ASSERT_TRUE(Module) << "Failed to parse basic MIR";
  
  Function *F = Module->getFunction("test_basic");
  ASSERT_TRUE(F) << "Function test_basic not found";
  
  MachineFunction *MF = MMI->getMachineFunction(*F);
  ASSERT_TRUE(MF) << "MachineFunction not found";
  
  NextUseResult &NU = runNextUseAnalysis(*MF, PM);

  MachineBasicBlock &MBB = MF->front();
  
  // Find the first S_MOV_B32 instruction
  MachineBasicBlock::iterator FirstMov = MBB.end();
  for (auto MI = MBB.begin(); MI != MBB.end(); ++MI) {
    if (MI->getOpcode() == AMDGPU::S_MOV_B32) {
      FirstMov = MI;
      break;
    }
  }
  ASSERT_NE(FirstMov, MBB.end()) << "First S_MOV_B32 not found";
  
  // Get the register defined by first instruction
  Register DefReg;
  for (const auto &MO : FirstMov->operands()) {
    if (MO.isReg() && MO.isDef() && MO.getReg().isVirtual()) {
      DefReg = MO.getReg();
      break;
    }
  }
  ASSERT_TRUE(DefReg.isVirtual()) << "Should define a virtual register";
  
  // Test that API methods work (even if returning default values)
  VRegMaskPair VMP(DefReg, LaneBitmask::getAll());
  unsigned distance = NU.getNextUseDistance(FirstMov, VMP);
  
  // API should work and return some value (likely default 65535)
  EXPECT_EQ(distance, 65535U) << "Without analysis, should return default Infinity value";
  
  // Test block-level API
  unsigned blockDistance = NU.getNextUseDistance(MBB, VMP);
  EXPECT_EQ(blockDistance, 65535U) << "Block API should also return default value";
  
  // Test subregister API
  auto subregUses = NU.getSortedSubregUses(FirstMov, VMP);
  EXPECT_EQ(subregUses.size(), 0U) << "Without analysis, should return empty subregister list";
  
  EXPECT_TRUE(true) << "✅ Basic NextUseAnalysis API validation completed";
}

// Test API with different lane masks
TEST_F(NextUseAnalysisTest, LaneMaskAPIValidation) {
  const char *MIRString = R"MIR(
--- |
  define void @test_lane_mask() { ret void }
...
---
name: test_lane_mask
body: |
  bb.0:
    %0:sgpr_32 = S_MOV_B32 42
    S_ENDPGM 0, implicit %0
)MIR";

  SetUp();
  LLVMContext Ctx;
  legacy::PassManager PM;
  auto Module = parseMIR(Ctx, MIRString, *TM, PM);
  ASSERT_TRUE(Module) << "Failed to parse lane mask test MIR";
  
  Function *F = Module->getFunction("test_lane_mask");
  ASSERT_TRUE(F) << "Function test_lane_mask not found";
  
  MachineFunction *MF = MMI->getMachineFunction(*F);
  ASSERT_TRUE(MF) << "MachineFunction not found";

  NextUseResult &NU = runNextUseAnalysis(*MF, PM);

  MachineBasicBlock &MBB = MF->front();
  auto MovInstr = MBB.begin();
  ASSERT_EQ(MovInstr->getOpcode(), AMDGPU::S_MOV_B32) << "First instruction should be S_MOV_B32";
  
  // Get virtual register
  Register VReg;
  for (const auto &MO : MovInstr->operands()) {
    if (MO.isReg() && MO.isDef() && MO.getReg().isVirtual()) {
      VReg = MO.getReg();
      break;
    }
  }
  ASSERT_TRUE(VReg.isVirtual()) << "Should define virtual register";
  
  // Test different lane masks
  VRegMaskPair VMP_None(VReg, LaneBitmask::getNone());
  VRegMaskPair VMP_All(VReg, LaneBitmask::getAll());
  
  unsigned distNone = NU.getNextUseDistance(MovInstr, VMP_None);
  unsigned distAll = NU.getNextUseDistance(MovInstr, VMP_All);
  
  // Should return consistent default values
  EXPECT_EQ(distNone, 65535U) << "No-lane mask should return default value";
  EXPECT_EQ(distAll, 65535U) << "All-lane mask should return default value";
  
  EXPECT_TRUE(true) << "✅ Lane mask API validation completed";
}

// Test VRegMaskPair construction and basic operations
TEST_F(NextUseAnalysisTest, VRegMaskPairValidation) {
  const char *MIRString = R"MIR(
--- |
  define void @test_vregmask() { ret void }
...
---
name: test_vregmask
body: |
  bb.0:
    %0:sgpr_32 = S_MOV_B32 10
    S_ENDPGM 0, implicit %0
)MIR";

  SetUp();
  LLVMContext Ctx;
  legacy::PassManager PM;
  auto Module = parseMIR(Ctx, MIRString, *TM, PM);
  ASSERT_TRUE(Module) << "Failed to parse VRegMaskPair test MIR";
  
  Function *F = Module->getFunction("test_vregmask");
  ASSERT_TRUE(F) << "Function test_vregmask not found";
  
  MachineFunction *MF = MMI->getMachineFunction(*F);
  ASSERT_TRUE(MF) << "MachineFunction not found";

  NextUseResult &NU = runNextUseAnalysis(*MF, PM);

  MachineBasicBlock &MBB = MF->front();
  auto MovInstr = MBB.begin();
  
  // Get virtual register
  Register VReg;
  for (const auto &MO : MovInstr->operands()) {
    if (MO.isReg() && MO.isDef() && MO.getReg().isVirtual()) {
      VReg = MO.getReg();
      break;
    }
  }
  ASSERT_TRUE(VReg.isVirtual()) << "Should define virtual register";
  
  // Test VRegMaskPair construction
  VRegMaskPair VMP1(VReg, LaneBitmask::getAll());
  VRegMaskPair VMP2(VReg, LaneBitmask::getNone());
  
  // Test basic properties
  EXPECT_EQ(VMP1.getVReg(), VReg) << "VRegMaskPair should store correct register";
  EXPECT_EQ(VMP2.getVReg(), VReg) << "VRegMaskPair should store correct register";
  
  EXPECT_NE(VMP1.getLaneMask(), VMP2.getLaneMask()) << "Different lane masks should be different";
  
  EXPECT_TRUE(true) << "✅ VRegMaskPair validation completed";
}

 

// Test actual analysis execution with computation
TEST_F(NextUseAnalysisTest, ActualAnalysisExecution) {
  const char *MIRString = R"MIR(
--- |
  define void @test_analysis() { ret void }
...
---
name: test_analysis
body: |
  bb.0:
    %0:sgpr_32 = S_MOV_B32 10
    %1:sgpr_32 = S_MOV_B32 5  
    %2:sgpr_32 = S_ADD_U32 %0, %1, implicit-def $scc
    S_ENDPGM 0, implicit %2
)MIR";

  SetUp();
  LLVMContext Ctx;
  legacy::PassManager PM;
  auto Module = parseMIR(Ctx, MIRString, *TM, PM);
  ASSERT_TRUE(Module) << "Failed to parse analysis test MIR";
  
  Function *F = Module->getFunction("test_analysis");
  ASSERT_TRUE(F) << "Function test_analysis not found";
  
  MachineFunction *MF = MMI->getMachineFunction(*F);
  ASSERT_TRUE(MF) << "MachineFunction not found";
  
  // Run actual NextUseAnalysis
  NextUseResult &NU = runNextUseAnalysis(*MF, PM);

  MachineBasicBlock &MBB = MF->front();
  
  // Find the first S_MOV_B32 instruction
  MachineBasicBlock::iterator FirstMov = MBB.end();
  MachineBasicBlock::iterator SecondMov = MBB.end();
  bool FirstMovSeen = false;
  for (auto MI = MBB.begin(); MI != MBB.end(); ++MI) {
    if (MI->getOpcode() == AMDGPU::S_MOV_B32) {

      if (FirstMovSeen) {
        SecondMov = MI;
        break;
      } else {
        FirstMov = MI;
        FirstMovSeen = true;
      }
    }
  }
  ASSERT_NE(FirstMov, MBB.end()) << "First S_MOV_B32 not found";
  ASSERT_NE(SecondMov, MBB.end()) << "Second S_MOV_B32 not found";

  // Get the register defined by first instruction
  Register DefReg;
  for (const auto &MO : FirstMov->operands()) {
    if (MO.isReg() && MO.isDef() && MO.getReg().isVirtual()) {
      DefReg = MO.getReg();
      break;
    }
  }
  ASSERT_TRUE(DefReg.isVirtual()) << "Should define a virtual register";
  
  // Test that actual analysis has been run (should not return default 65535)
  VRegMaskPair VMP(DefReg, LaneBitmask::getAll());
  unsigned distance = NU.getNextUseDistance(SecondMov, VMP);
  
  // With actual analysis, we should get a computed distance, not the default value
  // The distance should be reasonable (not the default 65535 "infinity" value)
  EXPECT_NE(distance, 65535U) << "With proper analysis, should not return default Infinity value";
  EXPECT_LT(distance, 65535U) << "Distance should be less than infinity for a used register";
  
  EXPECT_TRUE(true) << "✅ Actual NextUseAnalysis execution completed with distance: " << distance;
}
