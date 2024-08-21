#include "llvm/CodeGen/GlobalISel/InstructionSelect.h"
#include "GISelMITest.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelector.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/LegacyPassManager.h"

namespace {

class EraseMockInstructionSelector : public InstructionSelector {
public:
  int NumSelected = 0;
  SmallVector<MachineInstr *> MIs;

  bool select(MachineInstr &MI) override {
    ++NumSelected;
    switch (NumSelected) {
    case 1:
      EXPECT_EQ(&MI, MIs[8]);
      // Erase previous instructions
      MIs[7]->eraseFromParent();
      MIs[6]->eraseFromParent();
      // Don't erase this MI before step 3 to prevent DCE
      return true;
    case 2:
      EXPECT_EQ(&MI, MIs[5]);
      // Erase previous instructions reversed
      MIs[3]->eraseFromParent();
      MIs[4]->eraseFromParent();
      MI.eraseFromParent();
      return true;
    case 3:
      EXPECT_EQ(&MI, MIs[2]);
      MIs[8]->eraseFromParent();
      // Erase first instructions
      MIs[0]->eraseFromParent();
      MIs[1]->eraseFromParent();
      MI.eraseFromParent();
      return true;
    default:
      ADD_FAILURE();
      return false;
    }
  }

  void setupGeneratedPerFunctionState(MachineFunction &MF) override {}
};

TEST_F(AArch64GISelMITest, TestInstructionSelectErase) {
  StringRef MIRString = R"(
   $x0 = COPY %2(s64)
   $x0 = COPY %2(s64)
   $x0 = COPY %2(s64)
   $x0 = COPY %2(s64)
   $x0 = COPY %2(s64)
   $x0 = COPY %2(s64)
)";
  setUp(MIRString);
  if (!TM)
    GTEST_SKIP();

  legacy::PassManager PM;
  std::unique_ptr<TargetPassConfig> TPC(TM->createPassConfig(PM));

  EraseMockInstructionSelector ISel;
  ISel.TPC = TPC.get();
  for (auto &MI : *EntryMBB) {
    ISel.MIs.push_back(&MI);
  }

  InstructionSelect ISelPass;
  ISelPass.setInstructionSelector(&ISel);
  ISelPass.selectMachineFunction(*MF);
  EXPECT_EQ(ISel.NumSelected, 3);
}

} // namespace
