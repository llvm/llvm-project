#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace llvm {
  void initializeTestPassPass(PassRegistry &);
}

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

/// Create a TargetMachine. As we lack a dedicated always available target for
/// unittests, we go for "AMDGPU" to be able to test normal and subregister
/// liveranges.
std::unique_ptr<LLVMTargetMachine> createTargetMachine() {
  Triple TargetTriple("amdgcn--");
  std::string Error;
  const Target *T = TargetRegistry::lookupTarget("", TargetTriple, Error);
  if (!T)
    return nullptr;

  TargetOptions Options;
  return std::unique_ptr<LLVMTargetMachine>(static_cast<LLVMTargetMachine*>(
      T->createTargetMachine("AMDGPU", "gfx900", "", Options, None, None,
                             CodeGenOpt::Aggressive)));
}

std::unique_ptr<Module> parseMIR(LLVMContext &Context,
    legacy::PassManagerBase &PM, std::unique_ptr<MIRParser> &MIR,
    const LLVMTargetMachine &TM, StringRef MIRCode, const char *FuncName) {
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

typedef std::function<void(MachineFunction&,LiveIntervals&)> LiveIntervalTest;

struct TestPass : public MachineFunctionPass {
  static char ID;
  TestPass() : MachineFunctionPass(ID) {
    // We should never call this but always use PM.add(new TestPass(...))
    abort();
  }
  TestPass(LiveIntervalTest T) : MachineFunctionPass(ID), T(T) {
    initializeTestPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    LiveIntervals &LIS = getAnalysis<LiveIntervals>();
    T(MF, LIS);
    EXPECT_TRUE(MF.verify(this));
    return true;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<LiveIntervals>();
    AU.addPreserved<LiveIntervals>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
private:
  LiveIntervalTest T;
};

static MachineInstr &getMI(MachineFunction &MF, unsigned At,
                           unsigned BlockNum) {
  MachineBasicBlock &MBB = *MF.getBlockNumbered(BlockNum);

  unsigned I = 0;
  for (MachineInstr &MI : MBB) {
    if (I == At)
      return MI;
    ++I;
  }
  llvm_unreachable("Instruction not found");
}

/**
 * Move instruction number \p From in front of instruction number \p To and
 * update affected liveness intervals with LiveIntervalAnalysis::handleMove().
 */
static void testHandleMove(MachineFunction &MF, LiveIntervals &LIS,
                           unsigned From, unsigned To, unsigned BlockNum = 0) {
  MachineInstr &FromInstr = getMI(MF, From, BlockNum);
  MachineInstr &ToInstr = getMI(MF, To, BlockNum);

  MachineBasicBlock &MBB = *FromInstr.getParent();
  MBB.splice(ToInstr.getIterator(), &MBB, FromInstr.getIterator());
  LIS.handleMove(FromInstr, true);
}

/**
 * Move instructions numbered \p From inclusive through instruction number
 * \p To into a newly formed bundle and update affected liveness intervals
 * with LiveIntervalAnalysis::handleMoveIntoNewBundle().
 */
static void testHandleMoveIntoNewBundle(MachineFunction &MF, LiveIntervals &LIS,
                                        unsigned From, unsigned To,
                                        unsigned BlockNum = 0) {
  MachineInstr &FromInstr = getMI(MF, From, BlockNum);
  MachineInstr &ToInstr = getMI(MF, To, BlockNum);
  MachineBasicBlock &MBB = *FromInstr.getParent();
  MachineBasicBlock::instr_iterator I = FromInstr.getIterator();

  // Build bundle
  finalizeBundle(MBB, I, std::next(ToInstr.getIterator()));

  // Update LiveIntervals
  MachineBasicBlock::instr_iterator BundleStart = std::prev(I);
  LIS.handleMoveIntoNewBundle(*BundleStart, true);
}

static void liveIntervalTest(StringRef MIRFunc, LiveIntervalTest T) {
  LLVMContext Context;
  std::unique_ptr<LLVMTargetMachine> TM = createTargetMachine();
  // This test is designed for the X86 backend; stop if it is not available.
  if (!TM)
    return;

  legacy::PassManager PM;

  SmallString<160> S;
  StringRef MIRString = (Twine(R"MIR(
---
...
name: func
registers:
  - { id: 0, class: sreg_64 }
body: |
  bb.0:
)MIR") + Twine(MIRFunc) + Twine("...\n")).toNullTerminatedStringRef(S);
  std::unique_ptr<MIRParser> MIR;
  std::unique_ptr<Module> M = parseMIR(Context, PM, MIR, *TM, MIRString,
                                       "func");
  ASSERT_TRUE(M);

  PM.add(new TestPass(T));

  PM.run(*M);
}

} // End of anonymous namespace.

char TestPass::ID = 0;
INITIALIZE_PASS(TestPass, "testpass", "testpass", false, false)

TEST(LiveIntervalTest, MoveUpDef) {
  // Value defined.
  liveIntervalTest(R"MIR(
    S_NOP 0
    S_NOP 0
    early-clobber %0 = IMPLICIT_DEF
    S_NOP 0, implicit %0
)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 2, 1);
  });
}

TEST(LiveIntervalTest, MoveUpRedef) {
  liveIntervalTest(R"MIR(
    %0 = IMPLICIT_DEF
    S_NOP 0
    %0 = IMPLICIT_DEF implicit %0(tied-def 0)
    S_NOP 0, implicit %0
)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 2, 1);
  });
}

TEST(LiveIntervalTest, MoveUpEarlyDef) {
  liveIntervalTest(R"MIR(
    S_NOP 0
    S_NOP 0
    early-clobber %0 = IMPLICIT_DEF
    S_NOP 0, implicit %0
)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 2, 1);
  });
}

TEST(LiveIntervalTest, MoveUpEarlyRedef) {
  liveIntervalTest(R"MIR(
    %0 = IMPLICIT_DEF
    S_NOP 0
    early-clobber %0 = IMPLICIT_DEF implicit %0(tied-def 0)
    S_NOP 0, implicit %0
)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 2, 1);
  });
}

TEST(LiveIntervalTest, MoveUpKill) {
  liveIntervalTest(R"MIR(
    %0 = IMPLICIT_DEF
    S_NOP 0
    S_NOP 0, implicit %0
)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 2, 1);
  });
}

TEST(LiveIntervalTest, MoveUpKillFollowing) {
  liveIntervalTest(R"MIR(
    %0 = IMPLICIT_DEF
    S_NOP 0
    S_NOP 0, implicit %0
    S_NOP 0, implicit %0
)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 2, 1);
  });
}

// TODO: Construct a situation where we have intervals following a hole
// while still having connected components.

TEST(LiveIntervalTest, MoveDownDef) {
  // Value defined.
  liveIntervalTest(R"MIR(
    S_NOP 0
    early-clobber %0 = IMPLICIT_DEF
    S_NOP 0
    S_NOP 0, implicit %0
)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 1, 2);
  });
}

TEST(LiveIntervalTest, MoveDownRedef) {
  liveIntervalTest(R"MIR(
    %0 = IMPLICIT_DEF
    %0 = IMPLICIT_DEF implicit %0(tied-def 0)
    S_NOP 0
    S_NOP 0, implicit %0
)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 1, 2);
  });
}

TEST(LiveIntervalTest, MoveDownEarlyDef) {
  liveIntervalTest(R"MIR(
    S_NOP 0
    early-clobber %0 = IMPLICIT_DEF
    S_NOP 0
    S_NOP 0, implicit %0
)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 1, 2);
  });
}

TEST(LiveIntervalTest, MoveDownEarlyRedef) {
  liveIntervalTest(R"MIR(
    %0 = IMPLICIT_DEF
    early-clobber %0 = IMPLICIT_DEF implicit %0(tied-def 0)
    S_NOP 0
    S_NOP 0, implicit %0
)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 1, 2);
  });
}

TEST(LiveIntervalTest, MoveDownKill) {
  liveIntervalTest(R"MIR(
    %0 = IMPLICIT_DEF
    S_NOP 0, implicit %0
    S_NOP 0
)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 1, 2);
  });
}

TEST(LiveIntervalTest, MoveDownKillFollowing) {
  liveIntervalTest(R"MIR(
    %0 = IMPLICIT_DEF
    S_NOP 0
    S_NOP 0, implicit %0
    S_NOP 0, implicit %0
)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 1, 2);
  });
}

TEST(LiveIntervalTest, MoveUndefUse) {
  liveIntervalTest(R"MIR(
    %0 = IMPLICIT_DEF
    S_NOP 0, implicit undef %0
    S_NOP 0, implicit %0
    S_NOP 0
)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 1, 3);
  });
}

TEST(LiveIntervalTest, MoveUpValNos) {
  // handleMoveUp() had a bug where it would reuse the value number of the
  // destination segment, even though we have no guarantee that this valno
  // wasn't used in other segments.
  liveIntervalTest(R"MIR(
    successors: %bb.1, %bb.2
    %0 = IMPLICIT_DEF
    S_CBRANCH_VCCNZ %bb.2, implicit undef $vcc
    S_BRANCH %bb.1
  bb.2:
    S_NOP 0, implicit %0
  bb.1:
    successors: %bb.2
    %0 = IMPLICIT_DEF implicit %0(tied-def 0)
    %0 = IMPLICIT_DEF implicit %0(tied-def 0)
    %0 = IMPLICIT_DEF implicit %0(tied-def 0)
    S_BRANCH %bb.2
)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 2, 0, 2);
  });
}

TEST(LiveIntervalTest, MoveOverUndefUse0) {
  // findLastUseBefore() used by handleMoveUp() must ignore undef operands.
  liveIntervalTest(R"MIR(
    %0 = IMPLICIT_DEF
    S_NOP 0
    S_NOP 0, implicit undef %0
    %0 = IMPLICIT_DEF implicit %0(tied-def 0)
)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 3, 1);
  });
}

TEST(LiveIntervalTest, MoveOverUndefUse1) {
  // findLastUseBefore() used by handleMoveUp() must ignore undef operands.
  liveIntervalTest(R"MIR(
    $sgpr0 = IMPLICIT_DEF
    S_NOP 0
    S_NOP 0, implicit undef $sgpr0
    $sgpr0 = IMPLICIT_DEF implicit $sgpr0(tied-def 0)
)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 3, 1);
  });
}

TEST(LiveIntervalTest, SubRegMoveDown) {
  // Subregister ranges can have holes inside a basic block. Check for a
  // movement of the form 32->150 in a liverange [16, 32) [100,200).
  liveIntervalTest(R"MIR(
    successors: %bb.1, %bb.2
    %0 = IMPLICIT_DEF
    S_CBRANCH_VCCNZ %bb.2, implicit undef $vcc
    S_BRANCH %bb.1
  bb.2:
    successors: %bb.1
    S_NOP 0, implicit %0.sub0
    S_NOP 0, implicit %0.sub1
    S_NOP 0
    undef %0.sub0 = IMPLICIT_DEF
    %0.sub1 = IMPLICIT_DEF
  bb.1:
    S_NOP 0, implicit %0
)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
    // Scheduler behaviour: Clear def,read-undef flag and move.
    MachineInstr &MI = getMI(MF, 3, /*BlockNum=*/1);
    MI.getOperand(0).setIsUndef(false);
    testHandleMove(MF, LIS, 1, 4, /*BlockNum=*/1);
  });
}

TEST(LiveIntervalTest, SubRegMoveUp) {
  // handleMoveUp had a bug not updating valno of segment incoming to bb.2
  // after swapping subreg definitions.
  liveIntervalTest(R"MIR(
    successors: %bb.1, %bb.2
    undef %0.sub0 = IMPLICIT_DEF
    %0.sub1 = IMPLICIT_DEF
    S_CBRANCH_VCCNZ %bb.2, implicit undef $vcc
    S_BRANCH %bb.1
  bb.1:
    S_NOP 0, implicit %0.sub1
  bb.2:
    S_NOP 0, implicit %0.sub1
)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 1, 0);
  });
}

TEST(LiveIntervalTest, DeadSubRegMoveUp) {
  // handleMoveUp had a bug where moving a dead subreg def into the middle of
  // an earlier segment resulted in an invalid live range.
  liveIntervalTest(R"MIR(
    undef %125.sub0:vreg_128 = V_MOV_B32_e32 0, implicit $exec
    %125.sub1:vreg_128 = COPY %125.sub0
    %125.sub2:vreg_128 = COPY %125.sub0
    undef %51.sub0:vreg_128 = V_MOV_B32_e32 898625526, implicit $exec
    %51.sub1:vreg_128 = COPY %51.sub0
    %51.sub2:vreg_128 = COPY %51.sub0
    %52:vgpr_32 = V_MOV_B32_e32 986714345, implicit $exec
    %54:vgpr_32 = V_MOV_B32_e32 1742342378, implicit $exec
    %57:vgpr_32 = V_MOV_B32_e32 3168768712, implicit $exec
    %59:vgpr_32 = V_MOV_B32_e32 1039972644, implicit $exec
    %60:vgpr_32 = V_MAD_F32 0, %52, 0, undef %61:vgpr_32, 0, %59, 0, 0, implicit $exec
    %63:vgpr_32 = V_ADD_F32_e32 %51.sub3, undef %64:vgpr_32, implicit $exec
    dead %66:vgpr_32 = V_MAD_F32 0, %60, 0, undef %67:vgpr_32, 0, %125.sub2, 0, 0, implicit $exec
    undef %124.sub1:vreg_128 = V_MAD_F32 0, %57, 0, undef %70:vgpr_32, 0, %125.sub1, 0, 0, implicit $exec
    %124.sub0:vreg_128 = V_MAD_F32 0, %54, 0, undef %73:vgpr_32, 0, %125.sub0, 0, 0, implicit $exec
    dead undef %125.sub3:vreg_128 = V_MAC_F32_e32 %63, undef %76:vgpr_32, %125.sub3, implicit $exec
)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 15, 12);
  });
}

TEST(LiveIntervalTest, TestMoveSubRegDefAcrossUseDef) {
  liveIntervalTest(R"MIR(
    %1:vreg_64 = IMPLICIT_DEF

  bb.1:
    %2:vgpr_32 = V_MOV_B32_e32 2, implicit $exec
    %3:vgpr_32 = V_ADD_U32_e32 %2, %1.sub0, implicit $exec
    undef %1.sub0:vreg_64 = V_ADD_U32_e32 %2, %2, implicit $exec
    %1.sub1:vreg_64 = COPY %2
    S_NOP 0, implicit %1.sub1
    S_BRANCH %bb.1

)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
     MachineInstr &UndefSubregDef = getMI(MF, 2, 1);
     // The scheduler clears undef from subregister defs before moving
     UndefSubregDef.getOperand(0).setIsUndef(false);
     testHandleMove(MF, LIS, 3, 1, 1);
  });
}

TEST(LiveIntervalTest, TestMoveSubRegDefAcrossUseDefMulti) {
  liveIntervalTest(R"MIR(
    %1:vreg_96 = IMPLICIT_DEF

  bb.1:
    %2:vgpr_32 = V_MOV_B32_e32 2, implicit $exec
    %3:vgpr_32 = V_ADD_U32_e32 %2, %1.sub0, implicit $exec
    undef %1.sub0:vreg_96 = V_ADD_U32_e32 %2, %2, implicit $exec
    %1.sub1:vreg_96 = COPY %2
    %1.sub2:vreg_96 = COPY %2
    S_NOP 0, implicit %1.sub1, implicit %1.sub2
    S_BRANCH %bb.1

)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
     MachineInstr &UndefSubregDef = getMI(MF, 2, 1);
     // The scheduler clears undef from subregister defs before moving
     UndefSubregDef.getOperand(0).setIsUndef(false);
     testHandleMove(MF, LIS, 4, 1, 1);
  });
}

TEST(LiveIntervalTest, BundleUse) {
  liveIntervalTest(R"MIR(
    %0 = IMPLICIT_DEF
    S_NOP 0
    S_NOP 0, implicit %0
    S_NOP 0
)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMoveIntoNewBundle(MF, LIS, 1, 2);
  });
}

TEST(LiveIntervalTest, BundleDef) {
  liveIntervalTest(R"MIR(
    %0 = IMPLICIT_DEF
    S_NOP 0
    S_NOP 0, implicit %0
    S_NOP 0
)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMoveIntoNewBundle(MF, LIS, 0, 1);
  });
}

TEST(LiveIntervalTest, BundleRedef) {
  liveIntervalTest(R"MIR(
    %0 = IMPLICIT_DEF
    S_NOP 0
    %0 = IMPLICIT_DEF implicit %0(tied-def 0)
    S_NOP 0, implicit %0
)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMoveIntoNewBundle(MF, LIS, 1, 2);
  });
}

TEST(LiveIntervalTest, BundleInternalUse) {
  liveIntervalTest(R"MIR(
    %0 = IMPLICIT_DEF
    S_NOP 0
    S_NOP 0, implicit %0
    S_NOP 0
)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMoveIntoNewBundle(MF, LIS, 0, 2);
  });
}

TEST(LiveIntervalTest, BundleUndefUse) {
  liveIntervalTest(R"MIR(
    %0 = IMPLICIT_DEF
    S_NOP 0
    S_NOP 0, implicit undef %0
    S_NOP 0
)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMoveIntoNewBundle(MF, LIS, 1, 2);
  });
}

TEST(LiveIntervalTest, BundleSubRegUse) {
  liveIntervalTest(R"MIR(
    successors: %bb.1, %bb.2
    undef %0.sub0 = IMPLICIT_DEF
    %0.sub1 = IMPLICIT_DEF
    S_CBRANCH_VCCNZ %bb.2, implicit undef $vcc
    S_BRANCH %bb.1
  bb.1:
    S_NOP 0
    S_NOP 0, implicit %0.sub1
  bb.2:
    S_NOP 0, implicit %0.sub1
)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMoveIntoNewBundle(MF, LIS, 0, 1, 1);
  });
}

TEST(LiveIntervalTest, BundleSubRegDef) {
  liveIntervalTest(R"MIR(
    successors: %bb.1, %bb.2
    undef %0.sub0 = IMPLICIT_DEF
    %0.sub1 = IMPLICIT_DEF
    S_CBRANCH_VCCNZ %bb.2, implicit undef $vcc
    S_BRANCH %bb.1
  bb.1:
    S_NOP 0
    S_NOP 0, implicit %0.sub1
  bb.2:
    S_NOP 0, implicit %0.sub1
)MIR", [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMoveIntoNewBundle(MF, LIS, 0, 1, 0);
  });
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  initLLVM();
  return RUN_ALL_TESTS();
}
