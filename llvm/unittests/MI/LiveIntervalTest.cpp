#include "gtest/gtest.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/IR/LegacyPassManager.h"

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
/// unittests, we go for "x86_64" which should be available in most builds.
std::unique_ptr<TargetMachine> createTargetMachine() {
  Triple TargetTriple("x86_64--");
  std::string Error;
  const Target *T = TargetRegistry::lookupTarget("", TargetTriple, Error);
  if (!T)
    return nullptr;

  TargetOptions Options;
  return std::unique_ptr<TargetMachine>(
      T->createTargetMachine("x86_64", "", "", Options, None,
                             CodeModel::Default, CodeGenOpt::Aggressive));
}

std::unique_ptr<Module> parseMIR(LLVMContext &Context,
    legacy::PassManagerBase &PM, std::unique_ptr<MIRParser> &MIR,
    const TargetMachine &TM, StringRef MIRCode, const char *FuncName) {
  SMDiagnostic Diagnostic;
  std::unique_ptr<MemoryBuffer> MBuffer = MemoryBuffer::getMemBuffer(MIRCode);
  MIR = createMIRParser(std::move(MBuffer), Context);
  if (!MIR)
    return nullptr;

  std::unique_ptr<Module> M = MIR->parseLLVMModule();
  if (!M)
    return nullptr;

  M->setDataLayout(TM.createDataLayout());

  Function *F = M->getFunction(FuncName);
  if (!F)
    return nullptr;

  const LLVMTargetMachine &LLVMTM = static_cast<const LLVMTargetMachine&>(TM);
  LLVMTM.addMachineModuleInfo(PM);
  LLVMTM.addMachineFunctionAnalysis(PM, MIR.get());

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

/**
 * Move instruction number \p From in front of instruction number \p To and
 * update affected liveness intervals with LiveIntervalAnalysis::handleMove().
 */
static void testHandleMove(MachineFunction &MF, LiveIntervals &LIS,
                           unsigned From, unsigned To, unsigned BlockNum = 0) {
  MachineBasicBlock &MBB = *MF.getBlockNumbered(BlockNum);

  unsigned I = 0;
  MachineInstr *FromInstr = nullptr;
  MachineInstr *ToInstr = nullptr;
  for (MachineInstr &MI : MBB) {
    if (I == From)
      FromInstr = &MI;
    if (I == To)
      ToInstr = &MI;
    ++I;
  }
  assert(FromInstr != nullptr && ToInstr != nullptr);

  MBB.splice(ToInstr->getIterator(), &MBB, FromInstr->getIterator());
  LIS.handleMove(*FromInstr, true);
}

static void liveIntervalTest(StringRef MIRFunc, LiveIntervalTest T) {
  LLVMContext Context;
  std::unique_ptr<TargetMachine> TM = createTargetMachine();
  // This test is designed for the X86 backend; stop if it is not available.
  if (!TM)
    return;

  legacy::PassManager PM;

  SmallString<160> S;
  StringRef MIRString = (Twine(
"---\n"
"...\n"
"name: func\n"
"registers:\n"
"  - { id: 0, class: gr64 }\n"
"body: |\n"
"  bb.0:\n"
  ) + Twine(MIRFunc) + Twine("...\n")).toNullTerminatedStringRef(S);
  std::unique_ptr<MIRParser> MIR;
  std::unique_ptr<Module> M = parseMIR(Context, PM, MIR, *TM, MIRString,
                                       "func");

  PM.add(new TestPass(T));

  PM.run(*M);
}

} // End of anonymous namespace.

char TestPass::ID = 0;
INITIALIZE_PASS(TestPass, "testpass", "testpass", false, false)

TEST(LiveIntervalTest, MoveUpDef) {
  // Value defined.
  liveIntervalTest(
"    NOOP\n"
"    NOOP\n"
"    early-clobber %0 = IMPLICIT_DEF\n"
"    RETQ %0\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 2, 1);
  });
}

TEST(LiveIntervalTest, MoveUpRedef) {
  liveIntervalTest(
"    %0 = IMPLICIT_DEF\n"
"    NOOP\n"
"    %0 = IMPLICIT_DEF implicit %0(tied-def 0)\n"
"    RETQ %0\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 2, 1);
  });
}

TEST(LiveIntervalTest, MoveUpEarlyDef) {
  liveIntervalTest(
"    NOOP\n"
"    NOOP\n"
"    early-clobber %0 = IMPLICIT_DEF\n"
"    RETQ %0\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 2, 1);
  });
}

TEST(LiveIntervalTest, MoveUpEarlyRedef) {
  liveIntervalTest(
"    %0 = IMPLICIT_DEF\n"
"    NOOP\n"
"    early-clobber %0 = IMPLICIT_DEF implicit %0(tied-def 0)\n"
"    RETQ %0\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 2, 1);
  });
}

TEST(LiveIntervalTest, MoveUpKill) {
  liveIntervalTest(
"    %0 = IMPLICIT_DEF\n"
"    NOOP\n"
"    NOOP implicit %0\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 2, 1);
  });
}

TEST(LiveIntervalTest, MoveUpKillFollowing) {
  liveIntervalTest(
"    %0 = IMPLICIT_DEF\n"
"    NOOP\n"
"    NOOP implicit %0\n"
"    RETQ %0\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 2, 1);
  });
}

// TODO: Construct a situation where we have intervals following a hole
// while still having connected components.

TEST(LiveIntervalTest, MoveDownDef) {
  // Value defined.
  liveIntervalTest(
"    NOOP\n"
"    early-clobber %0 = IMPLICIT_DEF\n"
"    NOOP\n"
"    RETQ %0\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 1, 2);
  });
}

TEST(LiveIntervalTest, MoveDownRedef) {
  liveIntervalTest(
"    %0 = IMPLICIT_DEF\n"
"    %0 = IMPLICIT_DEF implicit %0(tied-def 0)\n"
"    NOOP\n"
"    RETQ %0\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 1, 2);
  });
}

TEST(LiveIntervalTest, MoveDownEarlyDef) {
  liveIntervalTest(
"    NOOP\n"
"    early-clobber %0 = IMPLICIT_DEF\n"
"    NOOP\n"
"    RETQ %0\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 1, 2);
  });
}

TEST(LiveIntervalTest, MoveDownEarlyRedef) {
  liveIntervalTest(
"    %0 = IMPLICIT_DEF\n"
"    early-clobber %0 = IMPLICIT_DEF implicit %0(tied-def 0)\n"
"    NOOP\n"
"    RETQ %0\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 1, 2);
  });
}

TEST(LiveIntervalTest, MoveDownKill) {
  liveIntervalTest(
"    %0 = IMPLICIT_DEF\n"
"    NOOP implicit %0\n"
"    NOOP\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 1, 2);
  });
}

TEST(LiveIntervalTest, MoveDownKillFollowing) {
  liveIntervalTest(
"    %0 = IMPLICIT_DEF\n"
"    NOOP\n"
"    NOOP implicit %0\n"
"    RETQ %0\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 1, 2);
  });
}

TEST(LiveIntervalTest, MoveUndefUse) {
  liveIntervalTest(
"    %0 = IMPLICIT_DEF\n"
"    NOOP implicit undef %0\n"
"    NOOP implicit %0\n"
"    NOOP\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 1, 3);
  });
}

TEST(LiveIntervalTest, MoveUpValNos) {
  // handleMoveUp() had a bug where it would reuse the value number of the
  // destination segment, even though we have no guarntee that this valno wasn't
  // used in other segments.
  liveIntervalTest(
"    successors: %bb.1, %bb.2\n"
"    %0 = IMPLICIT_DEF\n"
"    JG_1 %bb.2, implicit %eflags\n"
"    JMP_1 %bb.1\n"
"  bb.2:\n"
"    NOOP implicit %0\n"
"  bb.1:\n"
"    successors: %bb.2\n"
"    %0 = IMPLICIT_DEF implicit %0(tied-def 0)\n"
"    %0 = IMPLICIT_DEF implicit %0(tied-def 0)\n"
"    %0 = IMPLICIT_DEF implicit %0(tied-def 0)\n"
"    JMP_1 %bb.2\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 2, 0, 2);
  });
}

TEST(LiveIntervalTest, MoveOverUndefUse0) {
  // findLastUseBefore() used by handleMoveUp() must ignore undef operands.
  liveIntervalTest(
"    %0 = IMPLICIT_DEF\n"
"    NOOP\n"
"    NOOP implicit undef %0\n"
"    %0 = IMPLICIT_DEF implicit %0(tied-def 0)\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 3, 1);
  });
}

TEST(LiveIntervalTest, MoveOverUndefUse1) {
  // findLastUseBefore() used by handleMoveUp() must ignore undef operands.
  liveIntervalTest(
"    %rax = IMPLICIT_DEF\n"
"    NOOP\n"
"    NOOP implicit undef %rax\n"
"    %rax = IMPLICIT_DEF implicit %rax(tied-def 0)\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    testHandleMove(MF, LIS, 3, 1);
  });
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  initLLVM();
  return RUN_ALL_TESTS();
}
