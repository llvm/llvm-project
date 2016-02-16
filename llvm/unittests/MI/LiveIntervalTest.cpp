#include "gtest/gtest.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
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
      T->createTargetMachine("x86_64", "", "", Options, Reloc::Default,
                             CodeModel::Default, CodeGenOpt::Aggressive));
}

std::unique_ptr<Module> parseMIR(legacy::PassManagerBase &PM,
    std::unique_ptr<MIRParser> &MIR, const TargetMachine &TM,
    StringRef MIRCode, const char *FuncName) {
  LLVMContext &Context = getGlobalContext();

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

  MachineModuleInfo *MMI = new MachineModuleInfo(
      *TM.getMCAsmInfo(), *TM.getMCRegisterInfo(), nullptr);
  PM.add(MMI);

  MachineFunctionAnalysis *MFA = new MachineFunctionAnalysis(TM, MIR.get());
  PM.add(MFA);

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
void Move(MachineFunction &MF, LiveIntervals &LIS, unsigned From, unsigned To) {
  MachineBasicBlock &MBB = MF.front();

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
  LIS.handleMove(FromInstr, true);
}

void DoLiveIntervalTest(StringRef MIRFunc, LiveIntervalTest T) {
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
  std::unique_ptr<Module> M = parseMIR(PM, MIR, *TM, MIRString, "func");

  PM.add(new TestPass(T));

  PM.run(*M);
}

} // End of anonymous namespace.

char TestPass::ID = 0;
INITIALIZE_PASS(TestPass, "testpass", "testpass", false, false)

TEST(LiveIntervalTest, MoveUpDef) {
  // Value defined.
  DoLiveIntervalTest(
"    NOOP\n"
"    NOOP\n"
"    early-clobber %0 = IMPLICIT_DEF\n"
"    RETQ %0\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    Move(MF, LIS, 2, 1);
  });
}

TEST(LiveIntervalTest, MoveUpRedef) {
  DoLiveIntervalTest(
"    %0 = IMPLICIT_DEF\n"
"    NOOP\n"
"    %0 = IMPLICIT_DEF implicit %0(tied-def 0)\n"
"    RETQ %0\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    Move(MF, LIS, 2, 1);
  });
}

TEST(LiveIntervalTest, MoveUpEarlyDef) {
  DoLiveIntervalTest(
"    NOOP\n"
"    NOOP\n"
"    early-clobber %0 = IMPLICIT_DEF\n"
"    RETQ %0\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    Move(MF, LIS, 2, 1);
  });
}

TEST(LiveIntervalTest, MoveUpEarlyRedef) {
  DoLiveIntervalTest(
"    %0 = IMPLICIT_DEF\n"
"    NOOP\n"
"    early-clobber %0 = IMPLICIT_DEF implicit %0(tied-def 0)\n"
"    RETQ %0\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    Move(MF, LIS, 2, 1);
  });
}

TEST(LiveIntervalTest, MoveUpKill) {
  DoLiveIntervalTest(
"    %0 = IMPLICIT_DEF\n"
"    NOOP\n"
"    NOOP implicit %0\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    Move(MF, LIS, 2, 1);
  });
}

TEST(LiveIntervalTest, MoveUpKillFollowing) {
  DoLiveIntervalTest(
"    %0 = IMPLICIT_DEF\n"
"    NOOP\n"
"    NOOP implicit %0\n"
"    RETQ %0\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    Move(MF, LIS, 2, 1);
  });
}

// TODO: Construct a situation where we have intervals following a hole
// while still having connected components.

TEST(LiveIntervalTest, MoveDownDef) {
  // Value defined.
  DoLiveIntervalTest(
"    NOOP\n"
"    early-clobber %0 = IMPLICIT_DEF\n"
"    NOOP\n"
"    RETQ %0\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    Move(MF, LIS, 1, 2);
  });
}

TEST(LiveIntervalTest, MoveDownRedef) {
  DoLiveIntervalTest(
"    %0 = IMPLICIT_DEF\n"
"    %0 = IMPLICIT_DEF implicit %0(tied-def 0)\n"
"    NOOP\n"
"    RETQ %0\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    Move(MF, LIS, 1, 2);
  });
}

TEST(LiveIntervalTest, MoveDownEarlyDef) {
  DoLiveIntervalTest(
"    NOOP\n"
"    early-clobber %0 = IMPLICIT_DEF\n"
"    NOOP\n"
"    RETQ %0\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    Move(MF, LIS, 1, 2);
  });
}

TEST(LiveIntervalTest, MoveDownEarlyRedef) {
  DoLiveIntervalTest(
"    %0 = IMPLICIT_DEF\n"
"    early-clobber %0 = IMPLICIT_DEF implicit %0(tied-def 0)\n"
"    NOOP\n"
"    RETQ %0\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    Move(MF, LIS, 1, 2);
  });
}

TEST(LiveIntervalTest, MoveDownKill) {
  DoLiveIntervalTest(
"    %0 = IMPLICIT_DEF\n"
"    NOOP implicit %0\n"
"    NOOP\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    Move(MF, LIS, 1, 2);
  });
}

TEST(LiveIntervalTest, MoveDownKillFollowing) {
  DoLiveIntervalTest(
"    %0 = IMPLICIT_DEF\n"
"    NOOP\n"
"    NOOP implicit %0\n"
"    RETQ %0\n",
  [](MachineFunction &MF, LiveIntervals &LIS) {
    Move(MF, LIS, 1, 2);
  });
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  initLLVM();
  return RUN_ALL_TESTS();
}
