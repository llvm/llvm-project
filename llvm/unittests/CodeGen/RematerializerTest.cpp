//===- RematerializerTest.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Rematerializer.h"
#include "CodeGenTestBase.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/RegisterPressure.h"
#include "llvm/Support/TargetSelect.h"

using namespace llvm;
using RegisterIdx = Rematerializer::RegisterIdx;

namespace {
/// Wraps a rematerializer (with pointer-like access semantics through ->) next
/// to other members generally used by unit tests.
struct RematerializerWrapper {
  MachineFunction &MF;
  LiveIntervals &LIS;
  Rematerializer Remater;

  /// Region sizes for regions passed to the rematerializer. Initialized at
  /// construction to the correct value, can then be modified to track expected
  /// changes.
  SmallVector<unsigned> RegionSizes;
  /// Number of rematerializable registers identified by the rematerializer.
  /// Initialized at construction to the correct value, can then be modified to
  /// track expected changes.
  unsigned NumRematRegs;

  using RegionBoundaries = Rematerializer::RegionBoundaries;

  RematerializerWrapper(MachineFunction &MF,
                        SmallVectorImpl<RegionBoundaries> &Regions,
                        LiveIntervals &LIS)
      : MF(MF), LIS(LIS), Remater(MF, Regions, LIS) {
    for (const RegionBoundaries &Region : Regions)
      RegionSizes.push_back(std::distance(Region.first, Region.second));
    Remater.analyze();
    NumRematRegs = Remater.getNumRegs();
  }

  Rematerializer *operator->() { return &Remater; }
  const Rematerializer *operator->() const { return &Remater; }
  Rematerializer &operator*() { return Remater; }
  const Rematerializer &operator*() const { return Remater; }

  /// Returns the number of users of rematerializable register \p RegIdx.
  unsigned getNumUsers(RegisterIdx RegIdx) const {
    unsigned NumUsers = 0;
    for (const auto &[_, RegionUses] : Remater.getReg(RegIdx).Uses)
      NumUsers += RegionUses.size();
    return NumUsers;
  }

  /// Returns the number of MIs in region \p RegionIdx.
  unsigned getRegionSize(unsigned RegionIdx) const {
    const RegionBoundaries &Region = Remater.getRegion(RegionIdx);
    return std::distance(Region.first, Region.second);
  }

  /// Expects that \p NumMIs were added to region \p RegionIdx.
  RematerializerWrapper &addMIs(unsigned RegionIdx, unsigned NumMIs) {
    RegionSizes[RegionIdx] += NumMIs;
    return *this;
  }

  /// Expects that \p NumMIs were removed from region \p RegionIdx.
  RematerializerWrapper &removeMIs(unsigned RegionIdx, unsigned NumMIs) {
    RegionSizes[RegionIdx] -= NumMIs;
    return *this;
  }

  /// Expects that \p NumMIs were move from region \p FromRegionIdx to region \p
  /// ToRegionIdx.
  RematerializerWrapper &moveMIs(unsigned FromRegionIdx, unsigned ToRegionIdx,
                                 unsigned NumMIs) {
    return removeMIs(FromRegionIdx, NumMIs).addMIs(ToRegionIdx, NumMIs);
  }

  /// Expects that \p NumRegs rematerializable registers were added to the
  /// rematerializer.
  RematerializerWrapper &addRematRegs(unsigned NumRegs) {
    NumRematRegs += NumRegs;
    return *this;
  }
};

class RematerializerTest : public CodeGenTestBase {
public:
  static void SetUpTestCase() {
#if LLVM_HAS_AMDGPU_TARGET
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetMC();
#endif
  }

  void SetUp() override { setUpImpl("amdgcn--", "gfx950", ""); }

  using RematerializerTestFn = std::function<void(RematerializerWrapper &RW)>;

  void rematerializerTest(StringRef MIRBody, RematerializerTestFn Test) {
    SmallString<512> S;
    StringRef MIRString = (Twine(R"MIR(
---
name: func
tracksRegLiveness: true
machineFunctionInfo:
  isEntryFunction: true
body:             |
)MIR") + Twine(MIRBody) + Twine("...\n"))
                              .toNullTerminatedStringRef(S);
    ASSERT_TRUE(parseMIR(MIRString));
    MachineFunction &MF = getMF("func");
    LiveIntervals &LIS = MFAM.getResult<LiveIntervalsAnalysis>(MF);

    SmallVector<Rematerializer::RegionBoundaries> Regions;
    MachineInstr *FirstMI = nullptr;
    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : MBB) {
        if (!FirstMI)
          FirstMI = &MI;
        if (MI.isTerminator()) {
          if (FirstMI != &MI)
            Regions.push_back({FirstMI, MI});
          FirstMI = nullptr;
        }
      }
      if (FirstMI) {
        Regions.push_back({FirstMI, MBB.end()});
        FirstMI = nullptr;
      }
    }

    RematerializerWrapper RW(MF, Regions, LIS);
    Test(RW);

    RW->updateLiveIntervals();
    EXPECT_TRUE(MF.verify());
  }
};
} // namespace

/// All custon asserts/expects assume that a RematerializerWrapper is in scope
/// and named RW.

/// Asserts that the number of expected rematerializable registers indeed tracks
/// the actual number correctly.
#define ASSERT_NUM_REMAT_REGS() ASSERT_EQ(RW->getNumRegs(), RW.NumRematRegs)

/// Asserts that all regions match expected sizes from the test rematerializer.
#define ASSERT_REGION_SIZES()                                                  \
  {                                                                            \
    for (const auto [RegionIdx, ExpectedSize] : enumerate(RW.RegionSizes))     \
      ASSERT_EQ(RW.getRegionSize(RegionIdx), ExpectedSize);                    \
  }

/// Expects that register RegIdx in the rematerializer has a total of N users.
#define EXPECT_NUM_USERS(RegIdx, N)                                            \
  EXPECT_EQ(RW.getNumUsers(RegIdx), static_cast<unsigned>(N))

/// Expects that register RegIdx in the rematerializer has no users.
#define EXPECT_NO_USERS(RegIdx) EXPECT_NUM_USERS(RegIdx, 0)

/// Expects that rematerialized register RegIdx has origin OriginIdx, is defined
/// in region DefRegionIdx, and has a total of NumUsers users.
#define EXPECT_REMAT(RegIdx, OriginIdx, DefRegionIdx, NumUsers)                \
  {                                                                            \
    const Rematerializer::Reg &RematReg = RW->getReg(RegIdx);                  \
    EXPECT_EQ(RW->getOriginOf(RegIdx), OriginIdx);                             \
    EXPECT_EQ(RematReg.DefRegion, DefRegionIdx);                               \
    EXPECT_NUM_USERS(RegIdx, NumUsers);                                        \
  }

/// Rematerializes a tree of registers to a single user in different ways using
/// the dependency reuse mechanics and the coarse-grained or more fine-grained
/// API. Rollback rematerializations in-between each different wave of
/// rematerializations.
TEST_F(RematerializerTest, TreeRematRollback) {
  StringRef MIRBody = R"MIR(
  bb.0:
    %0:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 0, implicit $exec, implicit $mode
    %1:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 1, implicit $exec, implicit $mode
    %2:vgpr_32 = V_ADD_U32_e32 %0, %1, implicit $exec
    %3:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 3, implicit $exec, implicit $mode
    %4:vgpr_32 = V_ADD_U32_e32 %2, %3, implicit $exec

  bb.1:
    S_NOP 0, implicit %4
    S_ENDPGM 0
)MIR";
  rematerializerTest(MIRBody, [](RematerializerWrapper &RW) {
    Rematerializer::DependencyReuseInfo DRI;
    Rollbacker Rollbacker;
    RW->addListener(&Rollbacker);

    const unsigned MBB0 = 0, MBB1 = 1;
    const RegisterIdx Cst0 = 0, Cst1 = 1, Add01 = 2, Cst3 = 3, Add23 = 4;

    // Rematerialize Add23 with all transitive dependencies.
    RW->rematerializeToRegion(Add23, MBB1, DRI);
    RW->updateLiveIntervals();

    EXPECT_NO_USERS(Cst0);
    EXPECT_NO_USERS(Cst1);
    EXPECT_NO_USERS(Add01);
    EXPECT_NO_USERS(Cst3);
    EXPECT_NO_USERS(Add23);

    RW.moveMIs(MBB0, MBB1, 5).addRematRegs(5);
    ASSERT_REGION_SIZES();
    ASSERT_NUM_REMAT_REGS();

    // After rollback all rematerializations are removed from the MIR.
    Rollbacker.rollback(*RW);
    RW.moveMIs(MBB1, MBB0, 5);
    ASSERT_REGION_SIZES();

    // Rematerialize Add23 only with its direct dependencies, reuse the rest.
    DRI.clear().reuse(Cst0).reuse(Cst1);
    RW->rematerializeToRegion(Add23, MBB1, DRI);
    RW->updateLiveIntervals();

    EXPECT_NUM_USERS(Cst0, 1);
    EXPECT_NUM_USERS(Cst1, 1);
    EXPECT_NO_USERS(Add01);
    EXPECT_NO_USERS(Cst3);
    EXPECT_NO_USERS(Add23);

    RW.moveMIs(MBB0, MBB1, 3).addRematRegs(3);
    ASSERT_REGION_SIZES();
    ASSERT_NUM_REMAT_REGS();

    // After rollback all rematerializations are removed from the MIR.
    Rollbacker.rollback(*RW);
    RW.moveMIs(MBB1, MBB0, 3);
    ASSERT_REGION_SIZES();

    // Rematerialize Add23 only with its direct dependencies as before, but
    // with as fine-grained operations as possible.
    MachineInstr *NopMI = &*RW->getRegion(MBB1).first;

    DRI.clear().reuse(Cst0).reuse(Cst1);
    const RegisterIdx RematAdd01 =
        RW->rematerializeToPos(Add01, MBB1, NopMI, DRI);
    EXPECT_NO_USERS(RematAdd01);
    EXPECT_NUM_USERS(Add01, 1);
    EXPECT_NUM_USERS(Cst0, 2);
    EXPECT_NUM_USERS(Cst1, 2);

    DRI.clear();
    const RegisterIdx RematCst3 =
        RW->rematerializeToPos(Cst3, MBB1, NopMI, DRI);
    EXPECT_NO_USERS(RematCst3);
    EXPECT_NUM_USERS(Cst3, 1);

    DRI.clear().useRemat(Add01, RematAdd01).useRemat(Cst3, RematCst3);
    const RegisterIdx RematAdd23 =
        RW->rematerializeToPos(Add23, MBB1, NopMI, DRI);
    EXPECT_NO_USERS(RematAdd23);
    EXPECT_NUM_USERS(Add23, 1);
    EXPECT_NUM_USERS(RematAdd01, 1);
    EXPECT_NUM_USERS(RematCst3, 1);

    RW->transferUser(Add23, RematAdd23, MBB1, *NopMI);
    EXPECT_NO_USERS(Add23);
    EXPECT_NUM_USERS(RematAdd23, 1);

    RW.moveMIs(MBB0, MBB1, 3).addRematRegs(3);
    ASSERT_REGION_SIZES();
    ASSERT_NUM_REMAT_REGS();
  });
}

/// To rematerialize %3 along with all its dependencies before its only use in
/// bb.1, we must first rematerialize %0 and %1 (in any order), then %2, and
/// finally %3. The rematerializer had a rematerialization order bug wherein,
/// because %0 is also used directly in the MI defining %3, it was
/// rematerialized after %2, breaking the invariant that dependencies of a
/// register must always be rematerialized before the register itself.
TEST_F(RematerializerTest, MultiplePathsRematOrder) {
  StringRef MIRBody = R"MIR(
  bb.0:
    %0:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 0, implicit $exec, implicit $mode
    %1:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 1, implicit $exec, implicit $mode
    %2:vgpr_32 = V_ADD_U32_e32 %0, %1, implicit $exec
    %3:vgpr_32 = V_ADD_U32_e32 %0, %2, implicit $exec

  bb.1:
    S_NOP 0, implicit %3
    S_ENDPGM 0
)MIR";
  rematerializerTest(MIRBody, [](RematerializerWrapper &RW) {
    Rematerializer::DependencyReuseInfo DRI;
    const unsigned MBB1 = 1;
    const RegisterIdx Add02 = 3;
    RW->rematerializeToRegion(Add02, MBB1, DRI);
  });
}

/// Rematerializes a single register to multiple regions, tracking that
/// rematerializations are linked correctly and making sure that the original
/// register is deleted automatically when it no longer has any uses.
TEST_F(RematerializerTest, MultiRegionsRemat) {
  StringRef MIRBody = R"MIR(
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
)MIR";
  rematerializerTest(MIRBody, [](RematerializerWrapper &RW) {
    Rematerializer::DependencyReuseInfo DRI;

    const unsigned MBB0 = 0, MBB1 = 1, MBB2 = 2, MBB3 = 3;
    const RegisterIdx Cst0 = 0;

    // Rematerialization to MBB1.
    const RegisterIdx RematBB1 = RW->rematerializeToRegion(Cst0, MBB1, DRI);
    RW.addMIs(MBB1, 1);
    ASSERT_REGION_SIZES();
    EXPECT_REMAT(RematBB1, Cst0, MBB1, 1);

    // Rematerialization to MBB2.
    DRI.clear();
    const RegisterIdx RematBB2 = RW->rematerializeToRegion(Cst0, MBB2, DRI);
    RW.addMIs(MBB2, 1);
    ASSERT_REGION_SIZES();
    EXPECT_REMAT(RematBB2, Cst0, MBB2, 2);

    // Rematerialization to MBB3. Rematerializing to the last original user
    // deletes the original register.
    DRI.clear();
    const RegisterIdx RematBB3 = RW->rematerializeToRegion(Cst0, MBB3, DRI);
    RW.moveMIs(MBB0, MBB3, 1);
    ASSERT_REGION_SIZES();
    EXPECT_REMAT(RematBB3, Cst0, MBB3, 1);
  });
}

/// Rematerializes a tree of register with some unrematerializable operands to a
/// final destination in two steps, creating rematerializations of
/// rematerializations in the process. Make sure that origins of
/// rematerializations are always original registers.
TEST_F(RematerializerTest, MultiStep) {
  StringRef MIRBody = R"MIR(
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
)MIR";
  rematerializerTest(MIRBody, [](RematerializerWrapper &RW) {
    Rematerializer::DependencyReuseInfo DRI;

    const unsigned MBB0 = 0, MBB1 = 1, MBB2 = 2;
    const RegisterIdx Cst0 = 0, Add01 = 1, Add22 = 2, RematCst0 = 3,
                      RematAdd01 = 4, RematRematAdd01 = 5, RematAdd22 = 6;

    // Rematerialize Add01 from the first to the second block along with its
    // single rematerializable dependency (constant 0). The constant 1 has an
    // implicit def that is non-ignorable so it cannot be rematerialized. The
    // constant 0 remains in the first block because it has a user there, but
    // the add is deleted.
    RW->rematerializeToRegion(Add01, MBB1, DRI);
    RW.removeMIs(MBB0, 1).addMIs(MBB1, 2);
    ASSERT_REGION_SIZES();
    EXPECT_REMAT(RematCst0, Cst0, MBB1, 1);
    EXPECT_REMAT(RematAdd01, Add01, MBB1, 1);

    // We are going to re-rematerialize a register so the LIS need to be
    // up-to-date.
    RW->updateLiveIntervals();

    // Rematerialize Add22 from the second to the third block, which will also
    // indirectly rematerialize RematAdd01; make sure the latter's
    // rematerialization's origin is the original register, not RematAdd01.
    DRI.clear().reuse(RematCst0);
    RW->rematerializeToRegion(Add22, MBB2, DRI);
    RW.moveMIs(MBB1, MBB2, 2);
    ASSERT_REGION_SIZES();
    EXPECT_REMAT(RematRematAdd01, Add01, MBB2, 1);
    EXPECT_REMAT(RematAdd22, Add22, MBB2, 1);
  });
}

/// Checks that it is possible to rematerialize inside a region that was
/// rendered empty by previous rematerializations (as long as the region ends
/// with a terminator).
TEST_F(RematerializerTest, EmptyRegion) {
  StringRef MIRBody = R"MIR(
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
)MIR";
  rematerializerTest(MIRBody, [](RematerializerWrapper &RW) {
    Rematerializer::DependencyReuseInfo DRI;

    const unsigned MBB0 = 0, MBB1 = 1, MBB2 = 2, MBB3 = 3;
    const RegisterIdx Cst0 = 0, Cst1 = 1, Cst2 = 2, Cst3 = 3;

    // After rematerializing %2 and %3 to bb.3, their respective original
    // defining regions are empty. %2's region ends at the end of its parent
    // block, whereas %3's region ends at a terminator MI (S_BRANCH).
    RW->rematerializeToRegion(Cst2, MBB3, DRI);
    RW->rematerializeToRegion(Cst3, MBB3, DRI.clear());
    RW.removeMIs(MBB1, 1).removeMIs(MBB2, 1).addMIs(MBB3, 2);
    ASSERT_REGION_SIZES();

    // Move %0 to the empty MBB1 block/region.
    const RegisterIdx RematCst0 =
        RW->rematerializeToRegion(Cst0, MBB1, DRI.clear());
    RW->transferRegionUsers(Cst0, RematCst0, MBB3);

    // Move %1 to the empty MBB2 region, right before the S_BRANCH terminator.
    const RegisterIdx RematCst1 = RW->rematerializeToPos(
        Cst1, MBB2, RW->getRegion(MBB2).first, DRI.clear());
    RW->transferRegionUsers(Cst1, RematCst1, MBB3);

    RW.removeMIs(MBB0, 2).addMIs(MBB1, 1).addMIs(MBB2, 1);
    ASSERT_REGION_SIZES();
  });
}

/// Checks that only registers with a single definition are rematerializable,
/// even when registers are made up of multiple sub-registers each with their
/// own definition.
TEST_F(RematerializerTest, SubReg) {
  StringRef MIRBody = R"MIR(
  bb.0:
    undef %01.sub0:vreg_64_align2 = nofpexcept V_CVT_I32_F64_e32 0, implicit $exec, implicit $mode
    %01.sub1:vreg_64_align2 = nofpexcept V_CVT_I32_F64_e32 1, implicit $exec, implicit $mode

    undef %2.sub0:vreg_64_align2 = nofpexcept V_CVT_I32_F64_e32 2, implicit $exec, implicit $mode

    undef %34.sub0:vreg_64_align2 = nofpexcept V_CVT_I32_F64_e32 3, implicit $exec, implicit $mode

  bb.1:
    %34.sub1:vreg_64_align2 = nofpexcept V_CVT_I32_F64_e32 4, implicit $exec, implicit $mode
    S_NOP 0, implicit %01, implicit %2, implicit %34
    S_ENDPGM 0
)MIR";
  rematerializerTest(MIRBody, [](RematerializerWrapper &RW) {
    Rematerializer::DependencyReuseInfo DRI;

    const unsigned MBB0 = 0, MBB1 = 1;
    const RegisterIdx Cst2 = 0;

    RegisterIdx RematCst2 = RW->rematerializeToRegion(Cst2, MBB1, DRI);
    RW.moveMIs(MBB0, MBB1, 1);
    ASSERT_REGION_SIZES();
    EXPECT_REMAT(RematCst2, Cst2, MBB1, 1);
  });
}

/// The rematerializer had a bug where re-creating the interval of a
/// non-rematerializable super-register defined over multiple MIs, some of which
/// defining entirely dead subregisters, could cause a crash when changing the
/// order of sub-definitions (for example during scheduling) because the
/// re-created interval could end up with multiple connected components, which
/// is illegal. The solution is to split separate components of the interval in
/// such cases.
TEST_F(RematerializerTest, SplitSubRegDeadDef) {
  StringRef MIRBody = R"MIR(
  bb.0:
    undef %0.sub0:vreg_64 = IMPLICIT_DEF
    %0.sub1:vreg_64 = IMPLICIT_DEF
    %1:vgpr_32 = V_ADD_U32_e32 %0.sub0, %0.sub0, implicit $exec
    
  bb.1:
    S_NOP 0, implicit %1
    S_ENDPGM 0
)MIR";
  rematerializerTest(MIRBody, [](RematerializerWrapper &RW) {
    // Replicates the scheduler's effect on LIS on an intra-block move of MI.
    auto MoveMIAndAdjustLiveness = [&](MachineInstr &MI) {
      RW.LIS.handleMove(MI);
      const MachineRegisterInfo &MRI = RW.MF.getRegInfo();
      const TargetRegisterInfo &TRI = *RW.MF.getSubtarget().getRegisterInfo();
      RegisterOperands RegOpers;
      RegOpers.collect(MI, TRI, MRI, true, /*IgnoreDead=*/false);
      SlotIndex Sub1Slot = RW.LIS.getInstructionIndex(MI).getRegSlot();
      RegOpers.adjustLaneLiveness(RW.LIS, MRI, Sub1Slot, &MI);
    };

    MachineBasicBlock &MBB0 = *RW.MF.getBlockNumbered(0);
    MachineInstr &Sub0Def = *MBB0.begin();
    MachineInstr &Sub1Def = *MBB0.begin()->getNextNode();

    // Flip %0's subdefinition order. After the move, the definitions look like:
    // undef %0.sub1:vreg_64 = IMPLICIT_DEF
    // undef %0.sub0:vreg_64 = IMPLICIT_DEF
    MBB0.splice(Sub0Def.getIterator(), &MBB0, Sub1Def.getIterator());
    MoveMIAndAdjustLiveness(Sub1Def);

    // Rematerialize %1 to bb.1. This triggers a live-interval update of %0 when
    // calling Remater.updateLiveIntervals(), during which its interval is
    // split.
    Rematerializer::DependencyReuseInfo DRI;
    const unsigned MBB1 = 1;
    const RegisterIdx Add = 0;
    RW->rematerializeToRegion(Add, MBB1, DRI);
    RW->updateLiveIntervals();

    // If we didn't split %0 before, its definitions would now look like:
    // dead undef %0.sub1:vreg_64 = IMPLICIT_DEF
    // undef %0.sub0:vreg_64 = IMPLICIT_DEF
    //
    // Trying to flip back %0's definition order then triggers an
    // error in LIS.handleMove because its live interval is made up of multiple
    // connected components.
    ASSERT_NE(Sub0Def.getOperand(0).getReg(), Sub1Def.getOperand(0).getReg());
    MBB0.splice(MBB0.end(), &MBB0, Sub1Def.getIterator());
    MoveMIAndAdjustLiveness(Sub1Def);
  });
}

/// Checks that rollback works as expected when the rollback listener is added
/// mid-rematerializations.
TEST_F(RematerializerTest, Rollback) {
  StringRef MIRBody = R"MIR(
  bb.0:
    %0:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 0, implicit $exec, implicit $mode
    %1:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 1, implicit $exec, implicit $mode

  bb.1:
    S_NOP 0, implicit %0, implicit %1

  bb.2:
    S_NOP 0, implicit %0, implicit %1
    S_ENDPGM 0
)MIR";
  rematerializerTest(MIRBody, [](RematerializerWrapper &RW) {
    Rematerializer::DependencyReuseInfo DRI;

    const unsigned MBB0 = 0, MBB1 = 1, MBB2 = 2;
    const RegisterIdx Cst0 = 0, Cst1 = 1;

    // Rematerialize %0 to MBB1, taking one user from the original register.
    RegisterIdx RematCst0MBB1 = RW->rematerializeToRegion(Cst0, MBB1, DRI);
    RW.addMIs(MBB1, 1).addRematRegs(1);
    ASSERT_REGION_SIZES();
    ASSERT_NUM_REMAT_REGS();

    Rollbacker Rollback;
    RW->addListener(&Rollback);

    // Rematerialize %0 to MBB2 and %1 to MBB1/MBB2; each rematerialization ends
    // up with a single user and both original registers are deleted.
    RegisterIdx RematCst0MBB2 =
        RW->rematerializeToRegion(Cst0, MBB2, DRI.clear());
    RegisterIdx RematCst1MBB1 =
        RW->rematerializeToRegion(Cst1, MBB1, DRI.clear());
    RegisterIdx RematCst1MBB2 =
        RW->rematerializeToRegion(Cst1, MBB2, DRI.clear());

    RW.removeMIs(MBB0, 2).addMIs(MBB1, 1).addMIs(MBB2, 2).addRematRegs(3);
    ASSERT_REGION_SIZES();
    ASSERT_NUM_REMAT_REGS();

    EXPECT_NO_USERS(Cst0);
    EXPECT_NO_USERS(Cst1);
    EXPECT_NUM_USERS(RematCst0MBB1, 1);
    EXPECT_NUM_USERS(RematCst0MBB2, 1);
    EXPECT_NUM_USERS(RematCst1MBB1, 1);
    EXPECT_NUM_USERS(RematCst1MBB2, 1);

    // Rollback all changes since the rollbacker was added. The first
    // rematerialization of %0 to MBB1 happened before so it is not rolled back.
    // However %0 is re-created because it was deleted after.
    Rollback.rollback(*RW);

    RW.addMIs(MBB0, 2).removeMIs(MBB1, 1).removeMIs(MBB2, 2);
    ASSERT_REGION_SIZES();
    ASSERT_NUM_REMAT_REGS();

    EXPECT_NUM_USERS(Cst0, 1);
    EXPECT_NUM_USERS(Cst1, 2);
    EXPECT_NUM_USERS(RematCst0MBB1, 1);
    EXPECT_NO_USERS(RematCst0MBB2);
    EXPECT_NO_USERS(RematCst1MBB1);
    EXPECT_NO_USERS(RematCst1MBB2);
  });
}

/// Checks that rollback re-creates MIs at correct positions when the order of
/// register deletions forces the re-creation logic to iterate through multiple
/// deleted registers' respective insert position to find a valid one.
TEST_F(RematerializerTest, RollbackInvalidInsertPos) {
  StringRef MIRBody = R"MIR(
  bb.0:
    %0:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 0, implicit $exec, implicit $mode
    %1:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 1, implicit $exec, implicit $mode
    %2:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 2, implicit $exec, implicit $mode
    %3:vgpr_32 = nofpexcept V_CVT_I32_F64_e32 3, implicit $exec, implicit $mode

  bb.1:
    S_NOP 0, implicit %0, implicit %1, implicit %2, implicit %3
    S_ENDPGM 0
)MIR";
  rematerializerTest(MIRBody, [](RematerializerWrapper &RW) {
    Rematerializer::DependencyReuseInfo DRI;
    Rollbacker Rollback;
    RW->addListener(&Rollback);

    const unsigned MBB0 = 0, MBB1 = 1;
    const RegisterIdx Cst0 = 0, Cst1 = 1, Cst2 = 2, Cst3 = 3;

    // Rematerialize %0 to MBB1, deleting the original register.
    RW->rematerializeToRegion(Cst0, MBB1, DRI);
    RW.moveMIs(MBB0, MBB1, 1);
    ASSERT_REGION_SIZES();

    // Rematerialize %1 to MBB1, deleting the original register.
    RW->rematerializeToRegion(Cst1, MBB1, DRI.clear());
    RW.moveMIs(MBB0, MBB1, 1);
    ASSERT_REGION_SIZES();

    // Rematerialize %2 to MBB1, deleting the original register.
    RW->rematerializeToRegion(Cst2, MBB1, DRI.clear());
    RW.moveMIs(MBB0, MBB1, 1);
    ASSERT_REGION_SIZES();

    // Now rollback and check for correct instruction order in the original
    // defining region.
    Rollback.rollback(*RW);
    RW.moveMIs(MBB1, MBB0, 3);
    ASSERT_REGION_SIZES();

    MachineInstr &DefCst0 = *RW->getReg(Cst0).DefMI;
    MachineInstr &DefCst1 = *RW->getReg(Cst1).DefMI;
    MachineInstr &DefCst2 = *RW->getReg(Cst2).DefMI;
    MachineInstr &DefCst3 = *RW->getReg(Cst3).DefMI;
    EXPECT_EQ(std::next(DefCst0.getIterator()), DefCst1.getIterator());
    EXPECT_EQ(std::next(DefCst1.getIterator()), DefCst2.getIterator());
    EXPECT_EQ(std::next(DefCst2.getIterator()), DefCst3.getIterator());
  });
}
