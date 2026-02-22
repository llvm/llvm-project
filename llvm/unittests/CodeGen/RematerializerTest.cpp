//===- RematerializerTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Rematerializer.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineDomTreeUpdater.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"
#include <memory>

using namespace llvm;
using RegisterIdx = Rematerializer::RegisterIdx;

class RematerializerTest : public testing::Test {
public:
  LLVMContext Context;
  std::unique_ptr<TargetMachine> TM;
  std::unique_ptr<Module> M;
  std::unique_ptr<MachineModuleInfo> MMI;

  std::unique_ptr<MIRParser> MIR;
  std::unique_ptr<SmallVector<Rematerializer::RegionBoundaries>> Regions;
  std::unique_ptr<Rematerializer> Remater;
  MachineFunction *MF;

  LoopAnalysisManager LAM;
  MachineFunctionAnalysisManager MFAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;

  ModulePassManager MPM;
  FunctionPassManager FPM;
  MachineFunctionPassManager MFPM;
  ModuleAnalysisManager MAM;

  static void SetUpTestCase() {
    InitializeAllTargets();
    InitializeAllTargetMCs();
  }

  void SetUp() override {
    Triple TargetTriple("amdgcn--");
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget("", TargetTriple, Error);
    if (!T)
      GTEST_SKIP();
    TargetOptions Options;
    TM = std::unique_ptr<TargetMachine>(T->createTargetMachine(
        TargetTriple, "gfx950", "", Options, std::nullopt));
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

  bool parseMIRAndInit(StringRef MIRCode, StringRef FunName,
                       bool SupportRollback) {
    SMDiagnostic Diagnostic;
    std::unique_ptr<MemoryBuffer> MBuffer = MemoryBuffer::getMemBuffer(MIRCode);
    MIR = createMIRParser(std::move(MBuffer), Context);
    if (!MIR)
      return false;

    M = MIR->parseIRModule();
    M->setDataLayout(TM->createDataLayout());

    if (MIR->parseMachineFunctions(*M, MAM)) {
      M.reset();
      return false;
    }

    MF = &FAM.getResult<MachineFunctionAnalysis>(*M->getFunction(FunName))
              .getMF();
    LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(*MF);

    // Create regions for the rematerializer. Both MBBs and terminator MIs
    // delimitate regions.
    Regions = std::make_unique<SmallVector<Rematerializer::RegionBoundaries>>();
    MachineInstr *FirstMI = nullptr;
    for (MachineBasicBlock &MBB : *MF) {
      for (MachineInstr &MI : MBB) {
        if (!FirstMI)
          FirstMI = &MI;
        if (MI.isTerminator()) {
          if (FirstMI != &MI)
            Regions->push_back({FirstMI, MI});
          FirstMI = nullptr;
        }
      }
      // End the region at the end of the block.
      if (FirstMI) {
        Regions->push_back({FirstMI, MBB.end()});
        FirstMI = nullptr;
      }
    }

    Remater = std::make_unique<Rematerializer>(*MF, *Regions, LIS);
    Remater->analyze(SupportRollback);
    return true;
  }

  MachineFunction &getMF() { return *MF; }
  Rematerializer &getRematerializer() { return *Remater; }

  /// Returns the number of users of register \p RegIdx.
  unsigned getNumUsers(RegisterIdx RegIdx) {
    unsigned NumUsers = 0;
    for (const auto &[_, RegionUses] : Remater->getReg(RegIdx).Uses)
      NumUsers += RegionUses.size();
    return NumUsers;
  }

  /// Returns the size of region \p RegionIdx.
  unsigned getRegionSize(unsigned RegionIdx) {
    const Rematerializer::RegionBoundaries &Region = (*Regions)[RegionIdx];
    return std::distance(Region.first, Region.second);
  }
};

/// Asserts that region RegionIdx contains RegionSize instructions.
#define ASSERT_REGION_SIZE(RegionIdx, RegionSize)                              \
  ASSERT_EQ(getRegionSize(RegionIdx), RegionSize)

/// Asserts that regions have sizes RegionSizes, which must be an iterable
/// object with the same number of elements as the number of regions.
#define ASSERT_REGION_SIZES(RegionSizes)                                       \
  {                                                                            \
    ASSERT_EQ(RegionSizes.size(), Regions->size());                            \
    for (const auto [RegionIdx, ExpectedSize] : enumerate(RegionSizes))        \
      ASSERT_REGION_SIZE(RegionIdx, ExpectedSize);                             \
  }

/// Expects that register RegIdx in the rematerializer has a total of N users.
#define EXPECT_NUM_USERS(RegIdx, N)                                            \
  EXPECT_EQ(getNumUsers(RegIdx), static_cast<unsigned>(N))

/// Expects that register RegIdx in the remterializer hsa no users.
#define EXPECT_NO_USERS(RegIdx) EXPECT_NUM_USERS(RegIdx, 0)

/// Expects that rematerialized register RegIdx has origin OriginIdx, is defined
/// in region DefRegionIdx, and has a total of NumUsers users.
#define EXPECT_REMAT(RegIdx, OriginIdx, DefRegionIdx, NumUsers)                \
  {                                                                            \
    const Rematerializer::Reg &RematReg = Remater.getReg(RegIdx);              \
    EXPECT_EQ(Remater.getOriginOf(RegIdx), OriginIdx);                         \
    EXPECT_EQ(RematReg.DefRegion, DefRegionIdx);                               \
    EXPECT_NUM_USERS(RegIdx, NumUsers);                                        \
  }

/// Rematerializes a tree of registers to a single user in different ways using
/// the dependency reuse mechanics and the coarse-grained or more fine-grained
/// API. Rollback rematerializations in-between each different wave of
/// rematerializations.
TEST_F(RematerializerTest, TreeRematRollback) {
  StringRef MIR = R"(
name:            TreeRematRollback
tracksRegLiveness: true
machineFunctionInfo:
  isEntryFunction: true
body:             |
  bb.0:
    %0:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 0, implicit $exec, implicit $mode
    %1:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 1, implicit $exec, implicit $mode
    %2:vgpr_32 = V_ADD_U32_e32 %0, %1, implicit $exec
    %3:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 3, implicit $exec, implicit $mode
    %4:vgpr_32 = V_ADD_U32_e32 %2, %3, implicit $exec
  
  bb.1:
    S_NOP 0, implicit %4
    S_ENDPGM 0
...
)";
  ASSERT_TRUE(
      parseMIRAndInit(MIR, "TreeRematRollback", /*SupportRollback=*/true));
  Rematerializer &Remater = getRematerializer();
  Rematerializer::DependencyReuseInfo DRI;

  // MBB/Region indices.
  const unsigned MBB0 = 0, MBB1 = 1;
  SmallVector<unsigned, 2> RegionSizes{5, 1};
  ASSERT_REGION_SIZES(RegionSizes);

  // Indices of rematerializable registers.
  unsigned NumRegs = 0;
  const RegisterIdx Cst0 = NumRegs++, Cst1 = NumRegs++, Add01 = NumRegs++,
                    Cst3 = NumRegs++, Add23 = NumRegs++;
  ASSERT_EQ(Remater.getNumRegs(), NumRegs);

  // Rematerialize Add23 with all transitive dependencies.
  {
    Remater.rematerializeToRegion(/*RootIdx=*/Add23, /*UseRegion=*/MBB1, DRI);
    Remater.updateLiveIntervals();

    // None of the original registers have any users, but they still are in the
    // MIR because we enabled rollback support.
    EXPECT_NO_USERS(Cst0);
    EXPECT_NO_USERS(Cst1);
    EXPECT_NO_USERS(Add01);
    EXPECT_NO_USERS(Cst3);
    EXPECT_NO_USERS(Add23);

    // Copies of all MIs were inserted into the second MBB.
    RegionSizes[MBB1] += 5;
    ASSERT_REGION_SIZES(RegionSizes);
    NumRegs += 5;
    ASSERT_EQ(Remater.getNumRegs(), NumRegs);
  }

  // After rollback all rematerializations are removed from the MIR.
  Remater.rollbackRematsOf(Add23);
  RegionSizes[MBB1] -= 5;
  ASSERT_REGION_SIZES(RegionSizes);

  // Rematerialize Add23 only with its direct dependencies, reuse the rest.
  {
    DRI.clear().reuse(Cst0).reuse(Cst1);
    Remater.rematerializeToRegion(/*RootIdx=*/Add23, /*UseRegion=*/MBB1, DRI);
    Remater.updateLiveIntervals();

    // Re-used registers have rematerializations as their single user (original
    // users are dead). Rematerialized registers have no users.
    EXPECT_NUM_USERS(Cst0, 1);
    EXPECT_NUM_USERS(Cst1, 1);
    EXPECT_NO_USERS(Add01);
    EXPECT_NO_USERS(Cst3);
    EXPECT_NO_USERS(Add23);

    // Only immediate dependencies are copied to the second MBB.
    RegionSizes[MBB1] += 3;
    ASSERT_REGION_SIZES(RegionSizes);
    NumRegs += 3;
    ASSERT_EQ(Remater.getNumRegs(), NumRegs);
  }

  // After rollback all rematerializations are removed from the MIR.
  Remater.rollbackRematsOf(Add23);
  RegionSizes[MBB1] -= 3;
  ASSERT_REGION_SIZES(RegionSizes);

  // Rematerialize Add23 only with its direct dependencies as before, but
  // with as fine-grained operations as possible.
  {
    MachineInstr *NopMI = &*(*Regions)[MBB1].first;

    DRI.clear().reuse(Cst0).reuse(Cst1);
    const RegisterIdx RematAdd01 =
        Remater.rematerializeToPos(/*RootIdx=*/Add01, NopMI, DRI);
    // This adds an additional user to the used constants, and does not change
    // existing users for the original register.
    EXPECT_NO_USERS(RematAdd01);
    EXPECT_NUM_USERS(Add01, 1);
    EXPECT_NUM_USERS(Cst0, 2);
    EXPECT_NUM_USERS(Cst1, 2);

    DRI.clear();
    const RegisterIdx RematCst3 =
        Remater.rematerializeToPos(/*RootIdx=*/Cst3, NopMI, DRI);
    // This does not change existing users for the original register.
    EXPECT_NO_USERS(RematCst3);
    EXPECT_NUM_USERS(Cst3, 1);

    DRI.clear().useRemat(Add01, RematAdd01).useRemat(Cst3, RematCst3);
    const RegisterIdx RematAdd23 =
        Remater.rematerializeToPos(/*RootIdx=*/Add23, NopMI, DRI);
    // This adds a user to used rematerializations, and does not change existing
    // users for the original register.
    EXPECT_NO_USERS(RematAdd23);
    EXPECT_NUM_USERS(Add23, 1);
    EXPECT_NUM_USERS(RematAdd01, 1);
    EXPECT_NUM_USERS(RematCst3, 1);

    // Finally transfer the NOP user from the original to the rematerialized
    // register.
    Remater.transferUser(Add23, RematAdd23, *NopMI);
    EXPECT_NO_USERS(Add23);
    EXPECT_NUM_USERS(RematAdd23, 1);

    RegionSizes[MBB1] += 3;
    ASSERT_REGION_SIZES(RegionSizes);
    NumRegs += 3;
    ASSERT_EQ(Remater.getNumRegs(), NumRegs);
  }

  // This time don't rollback; commit the rematerializations. This finally
  // deletes unused registers in the first block. However the number of
  // registers tracked by the rematerializer doesn't change.
  Remater.updateLiveIntervals();
  Remater.commitRematerializations();
  RegionSizes[MBB0] -= 3;
  ASSERT_REGION_SIZES(RegionSizes);
  ASSERT_EQ(Remater.getNumRegs(), NumRegs);

  EXPECT_TRUE(getMF().verify());
}

/// Rematerializes a single register to multiple regions, tracking that
/// rematerializations are linked correctly and making sure that the original
/// register is deleted automatically when it no longer has any uses.
TEST_F(RematerializerTest, MultiRegionsRemat) {
  StringRef MIR = R"(
name:            MultiRegionsRemat
tracksRegLiveness: true
machineFunctionInfo:
  isEntryFunction: true
body:             |
  bb.0:
    %0:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 0, implicit $exec, implicit $mode
  
  bb.1:
    S_NOP 0, implicit %0, implicit %0

  bb.2:
    S_NOP 0, implicit %0
    S_NOP 0, implicit %0

  bb.3:
    S_NOP 0, implicit %0
    S_ENDPGM 0
...
)";
  ASSERT_TRUE(
      parseMIRAndInit(MIR, "MultiRegionsRemat", /*SupportRollback=*/false));
  Rematerializer &Remater = getRematerializer();
  Rematerializer::DependencyReuseInfo DRI;

  // MBB/Region indices.
  const unsigned MBB0 = 0, MBB1 = 1, MBB2 = 2, MBB3 = 3;
  SmallVector<unsigned, 2> RegionSizes{1, 1, 2, 1};
  ASSERT_REGION_SIZES(RegionSizes);

  // Indices of rematerializable registers.
  const RegisterIdx Cst0 = 0;
  ASSERT_EQ(Remater.getNumRegs(), 1U);

  // Rematerialization to MBB1.
  const RegisterIdx RematBB1 =
      Remater.rematerializeToRegion(/*RootIdx=*/Cst0, /*UseRegion=*/MBB1, DRI);
  ++RegionSizes[MBB1];
  ASSERT_REGION_SIZES(RegionSizes);
  EXPECT_REMAT(/*RegIdx=*/RematBB1, /*OriginIdx=*/Cst0, /*DefRegionIdx=*/MBB1,
               /*NumUsers=*/1);

  // Rematerialization to MBB2.
  DRI.clear();
  const RegisterIdx RematBB2 =
      Remater.rematerializeToRegion(/*RootIdx=*/Cst0, /*UseRegion=*/MBB2, DRI);
  ++RegionSizes[MBB2];
  ASSERT_REGION_SIZES(RegionSizes);
  EXPECT_REMAT(/*RegIdx=*/RematBB2, /*OriginIdx=*/Cst0, /*DefRegionIdx=*/MBB2,
               /*NumUsers=*/2);

  // Rematerialization to MBB3. Rematerializing to the last original user
  // deletes the original register.
  DRI.clear();
  const RegisterIdx RematBB3 =
      Remater.rematerializeToRegion(/*RootIdx=*/Cst0, /*UseRegion=*/MBB3, DRI);
  --RegionSizes[MBB0];
  ++RegionSizes[MBB3];
  ASSERT_REGION_SIZES(RegionSizes);
  EXPECT_REMAT(/*RegIdx=*/RematBB3, /*OriginIdx=*/Cst0, /*DefRegionIdx=*/MBB3,
               /*NumUsers=*/1);

  Remater.updateLiveIntervals();
  EXPECT_TRUE(getMF().verify());
}

/// Rematerializes a tree of register with some unrematerializable operands to a
/// final destination in two steps, creating rematerializations of
/// rematerializations in the process. Make sure that origins of
/// rematerializations are always original registers.
TEST_F(RematerializerTest, MultiStep) {
  StringRef MIR = R"(
name:            MultiStep
tracksRegLiveness: true
machineFunctionInfo:
  isEntryFunction: true
body:             |
  bb.0:
    %0:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 0, implicit $exec, implicit $mode
    %1:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 1, implicit $exec, implicit $mode, implicit-def $m0
    %2:vgpr_32 = V_ADD_U32_e32 %0, %1, implicit $exec
    S_NOP 0, implicit %0
  
  bb.1:
    %3:vgpr_32 = V_ADD_U32_e32 %2, %2, implicit $exec

  bb.2:
    S_NOP 0, implicit %3
    S_ENDPGM 0
...
)";
  ASSERT_TRUE(parseMIRAndInit(MIR, "MultiStep", /*SupportRollback=*/false));
  Rematerializer &Remater = getRematerializer();
  Rematerializer::DependencyReuseInfo DRI;

  // MBB/Region indices.
  const unsigned MBB0 = 0, MBB1 = 1, MBB2 = 2;
  SmallVector<unsigned, 2> RegionSizes{4, 1, 1};
  ASSERT_REGION_SIZES(RegionSizes);

  // Indices of rematerializable registers.
  unsigned NumRegs = 0;
  const RegisterIdx Cst0 = NumRegs++, Add01 = NumRegs++, Add22 = NumRegs++;
  ASSERT_EQ(Remater.getNumRegs(), NumRegs);

  // Rematerialize Add01 from the first to the second block along with its
  // single rematerializable dependency (constant 0). The constant 1 has an
  // implicit def that is non-ignorable so it cannot be rematerialized. The
  // constant 0 remains in the first block because it has a user there, but the
  // add is deleted.
  Remater.rematerializeToRegion(/*RootIdx=*/Add01, /*UseRegion=*/MBB1, DRI);
  const RegisterIdx RematCst0 = NumRegs++, RematAdd01 = NumRegs++;
  RegionSizes[MBB0] -= 1;
  RegionSizes[MBB1] += 2;
  ASSERT_REGION_SIZES(RegionSizes);
  EXPECT_REMAT(/*RegIdx=*/RematCst0, /*OriginIdx=*/Cst0, /*DefRegionIdx=*/MBB1,
               /*NumUsers=*/1);
  EXPECT_REMAT(/*RegIdx=*/RematAdd01, /*OriginIdx=*/Add01,
               /*DefRegionIdx=*/MBB1,
               /*NumUsers=*/1);

  // We are going to re-rematerialize a register so the LIS need to be
  // up-to-date.
  Remater.updateLiveIntervals();

  // Rematerialize Add22 from the second to the third block, which will
  // also indirectly rematerialize RematAdd01; make sure the latter's
  // rematerializations's origin is the original register, not RematAdd01.
  DRI.clear().reuse(RematCst0);
  Remater.rematerializeToRegion(/*RootIdx=*/Add22, /*UseRegion=*/MBB2, DRI);
  const RegisterIdx RematRematAdd01 = NumRegs++, RematAdd22 = NumRegs++;
  RegionSizes[MBB1] -= 2;
  RegionSizes[MBB2] += 2;
  ASSERT_REGION_SIZES(RegionSizes);
  EXPECT_REMAT(/*RegIdx=*/RematRematAdd01, /*OriginIdx=*/Add01,
               /*DefRegionIdx=*/MBB2,
               /*NumUsers=*/1);
  EXPECT_REMAT(/*RegIdx=*/RematAdd22, /*OriginIdx=*/Add22,
               /*DefRegionIdx=*/MBB2,
               /*NumUsers=*/1);

  Remater.updateLiveIntervals();
  EXPECT_TRUE(getMF().verify());
}

/// Checks that it is possible to rematerialize inside a region that was
/// rendered empty by previous rematerializations (as long as the region ends
/// with a terminator).
TEST_F(RematerializerTest, EmptyRegion) {
  StringRef MIR = R"(
name:            EmptyRegion
tracksRegLiveness: true
machineFunctionInfo:
  isEntryFunction: true
body:             |
  bb.0:
    %0:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 0, implicit $exec, implicit $mode
    %1:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 1, implicit $exec, implicit $mode

  bb.1:
    %2:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 2, implicit $exec, implicit $mode
    
  bb.2:
    %3:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 3, implicit $exec, implicit $mode
    S_BRANCH %bb.3

  bb.3:
    S_NOP 0, implicit %0, implicit %1
    S_NOP 0, implicit %2, implicit %3
    S_ENDPGM 0
...
)";
  ASSERT_TRUE(parseMIRAndInit(MIR, "EmptyRegion", /*SupportRollback=*/false));
  Rematerializer &Remater = getRematerializer();
  Rematerializer::DependencyReuseInfo DRI;

  // MBB/Region indices.
  const unsigned MBB0 = 0, MBB1 = 1, MBB2 = 2, MBB3 = 3;
  SmallVector<unsigned, 2> RegionSizes{2, 1, 1, 2};
  ASSERT_REGION_SIZES(RegionSizes);

  // Indices of rematerializable registers.
  unsigned NumRegs = 0;
  const RegisterIdx Cst0 = NumRegs++, Cst1 = NumRegs++, Cst2 = NumRegs++,
                    Cst3 = NumRegs++;
  ASSERT_EQ(Remater.getNumRegs(), NumRegs);

  // After rematerializing %2 and %3 to bb.3, their respective original defining
  // regions are empty. %2's region ends at the end of its parent block, whereas
  // %3's region ends at a terminator MI (S_BRANCH).
  Remater.rematerializeToRegion(/*RootIdx=*/Cst2, /*UseRegion=*/MBB3, DRI);
  Remater.rematerializeToRegion(/*RootIdx=*/Cst3, /*UseRegion=*/MBB3, DRI);
  RegionSizes[MBB1] -= 1;
  RegionSizes[MBB2] -= 1;
  RegionSizes[MBB3] += 2;
  ASSERT_REGION_SIZES(RegionSizes);

  // We can move %0 and %1 to bb.2 because the boundary of bb.2's region points
  // to the region terminator (S_BRANCH), which is a valid position to insert
  // before. We couldn't move them to bb.1 however, since after %2 is
  // rematerialized there is no MI left to reference inside the region.
  RegisterIdx RematCst0 =
      Remater.rematerializeToPos(/*RootIdx=*/Cst0, (*Regions)[MBB2].first, DRI);
  RegisterIdx RematCst1 =
      Remater.rematerializeToPos(/*RootIdx=*/Cst1, (*Regions)[MBB2].first, DRI);
  Remater.transferRegionUsers(Cst0, RematCst0, MBB3);
  Remater.transferRegionUsers(Cst1, RematCst1, MBB3);
  RegionSizes[MBB0] -= 2;
  RegionSizes[MBB2] += 2;
  ASSERT_REGION_SIZES(RegionSizes);

  Remater.updateLiveIntervals();
  EXPECT_TRUE(getMF().verify());
}

/// Checks that only registers with a single definition are rematerializable,
/// even when registers are made up of multiple sub-registers each with their
/// own definition.
TEST_F(RematerializerTest, SubReg) {
  StringRef MIR = R"(
name:            SubReg
tracksRegLiveness: true
machineFunctionInfo:
  isEntryFunction: true
body:             |
  bb.0:
    undef %01.sub0:vreg_64_align2 = nofpexcept V_CVT_I32_F64_e32 0, implicit $exec, implicit $mode
    %01.sub1:vreg_64_align2 = nofpexcept V_CVT_I32_F64_e32 1, implicit $exec, implicit $mode
    
    undef %2.sub0:vreg_64_align2 = nofpexcept V_CVT_I32_F64_e32 2, implicit $exec, implicit $mode
    
    undef %34.sub0:vreg_64_align2 = nofpexcept V_CVT_I32_F64_e32 3, implicit $exec, implicit $mode

  bb.1:
    %34.sub1:vreg_64_align2 = nofpexcept V_CVT_I32_F64_e32 4, implicit $exec, implicit $mode
    S_NOP 0, implicit %01, implicit %2, implicit %34
    S_ENDPGM 0
...
)";
  ASSERT_TRUE(parseMIRAndInit(MIR, "SubReg", /*SupportRollback=*/false));
  Rematerializer &Remater = getRematerializer();
  Rematerializer::DependencyReuseInfo DRI;

  // MBB/Region indices.
  const unsigned MBB0 = 0, MBB1 = 1;
  SmallVector<unsigned, 2> RegionSizes{4, 2};
  ASSERT_REGION_SIZES(RegionSizes);

  // Indices of rematerializable registers.
  unsigned NumRegs = 0;
  const RegisterIdx Cst2 = NumRegs++;
  ASSERT_EQ(Remater.getNumRegs(), NumRegs);

  RegisterIdx RematCst2 =
      Remater.rematerializeToRegion(/*RootIdx=*/Cst2, /*UseRegion=*/MBB1, DRI);
  RegionSizes[MBB0] -= 1;
  RegionSizes[MBB1] += 1;
  ASSERT_REGION_SIZES(RegionSizes);
  EXPECT_REMAT(/*RegIdx=*/RematCst2, /*OriginIdx=*/Cst2,
               /*DefRegionIdx=*/MBB1,
               /*NumUsers=*/1);

  Remater.updateLiveIntervals();
  EXPECT_TRUE(getMF().verify());
}
