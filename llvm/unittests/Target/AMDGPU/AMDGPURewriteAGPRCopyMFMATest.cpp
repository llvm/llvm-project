#include "AMDGPURewriteAGPRCopyMFMA.h"
#include "AMDGPUTargetMachine.h"
#include "AMDGPUUnitTests.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class AMDGPURewriteAGPRCopyMFMATest : public testing::Test {
protected:
  std::unique_ptr<LLVMContext> Context;
  std::unique_ptr<Module> M;
  std::unique_ptr<const TargetMachine> TM;
  std::unique_ptr<MachineModuleInfo> MMI;
  std::unique_ptr<MachineFunction> MF;
  std::unique_ptr<MachineDominatorTree> MDT;

  void SetUp() override {
    TM = createAMDGPUTargetMachine("amdgcn-amd-amdhsa", "gfx90a", "");
    if (!TM)
      return;

    Context = std::make_unique<LLVMContext>();
    M = std::make_unique<Module>("TestModule", *Context);
    MMI = std::make_unique<MachineModuleInfo>(TM.get());
  }

  void createMachineFunction() {
    M->setDataLayout(TM->createDataLayout());
    FunctionType *FTy = FunctionType::get(Type::getVoidTy(*Context), false);
    Function *F = Function::Create(FTy, Function::ExternalLinkage, "test", *M);
    MF = std::make_unique<MachineFunction>(*F, *TM, *TM->getSubtargetImpl(*F),
                                           MMI->getContext(), 0);
  }
};

TEST_F(AMDGPURewriteAGPRCopyMFMATest, JointDominanceDiamond) {
  if (!TM)
    return;
  createMachineFunction();

  // Create CFG:
  // Entry -> StoreBB
  // Entry -> NoStoreBB
  // StoreBB -> MergeBB
  // NoStoreBB -> MergeBB

  MachineBasicBlock *Entry = MF->CreateMachineBasicBlock();
  MachineBasicBlock *StoreBB = MF->CreateMachineBasicBlock();
  MachineBasicBlock *NoStoreBB = MF->CreateMachineBasicBlock();
  MachineBasicBlock *MergeBB = MF->CreateMachineBasicBlock();

  MF->push_back(Entry);
  MF->push_back(StoreBB);
  MF->push_back(NoStoreBB);
  MF->push_back(MergeBB);

  Entry->addSuccessor(StoreBB);
  Entry->addSuccessor(NoStoreBB);

  StoreBB->addSuccessor(MergeBB);

  NoStoreBB->addSuccessor(MergeBB);

  MDT = std::make_unique<MachineDominatorTree>();
  MDT->recalculate(*MF);

  const TargetInstrInfo *TII = MF->getSubtarget().getInstrInfo();
  DebugLoc DL;

  MachineInstr *StoreMI =
      BuildMI(StoreBB, DL, TII->get(TargetOpcode::INLINEASM))
          .addExternalSymbol("store")
          .addImm(1 /* SideEffects */);

  MachineInstr *LoadMI = BuildMI(MergeBB, DL, TII->get(TargetOpcode::INLINEASM))
                             .addExternalSymbol("load")
                             .addImm(1);

  int Slot = MF->getFrameInfo().CreateSpillStackObject(4, Align(4));
  StoreMI->addOperand(*MF, MachineOperand::CreateFI(Slot));
  LoadMI->addOperand(*MF, MachineOperand::CreateFI(Slot));

  SmallVector<MachineInstr *, 4> Stores = {StoreMI};
  SmallVector<MachineInstr *, 4> Loads = {LoadMI};

  bool Result =
      AMDGPU::checkAGPRCopyMFMAJointDominance(*MF, *MDT, Stores, Loads, Slot);
  EXPECT_FALSE(Result);
}

TEST_F(AMDGPURewriteAGPRCopyMFMATest, LoadBeforeStoreInLoop) {
  if (!TM)
    return;
  createMachineFunction();

  // Entry -> LoopBB -> Exit
  // LoopBB -> LoopBB

  MachineBasicBlock *Entry = MF->CreateMachineBasicBlock();
  MachineBasicBlock *LoopBB = MF->CreateMachineBasicBlock();
  MachineBasicBlock *Exit = MF->CreateMachineBasicBlock();

  MF->push_back(Entry);
  MF->push_back(LoopBB);
  MF->push_back(Exit);

  Entry->addSuccessor(LoopBB);
  LoopBB->addSuccessor(LoopBB);
  LoopBB->addSuccessor(Exit);

  MDT = std::make_unique<MachineDominatorTree>();
  MDT->recalculate(*MF);

  const TargetInstrInfo *TII = MF->getSubtarget().getInstrInfo();
  DebugLoc DL;

  MachineInstr *LoadMI = BuildMI(LoopBB, DL, TII->get(TargetOpcode::INLINEASM))
                             .addExternalSymbol("load")
                             .addImm(1);

  MachineInstr *StoreMI = BuildMI(LoopBB, DL, TII->get(TargetOpcode::INLINEASM))
                              .addExternalSymbol("store")
                              .addImm(1);

  int Slot = MF->getFrameInfo().CreateSpillStackObject(4, Align(4));
  StoreMI->addOperand(*MF, MachineOperand::CreateFI(Slot));
  LoadMI->addOperand(*MF, MachineOperand::CreateFI(Slot));

  SmallVector<MachineInstr *, 4> Stores = {StoreMI};
  SmallVector<MachineInstr *, 4> Loads = {LoadMI};

  bool Result =
      AMDGPU::checkAGPRCopyMFMAJointDominance(*MF, *MDT, Stores, Loads, Slot);
  EXPECT_FALSE(Result);
}

TEST_F(AMDGPURewriteAGPRCopyMFMATest, DominatedByPredecessor) {
  if (!TM)
    return;
  createMachineFunction();

  // Entry -> StoreBB -> LoadBB

  MachineBasicBlock *Entry = MF->CreateMachineBasicBlock();
  MachineBasicBlock *StoreBB = MF->CreateMachineBasicBlock();
  MachineBasicBlock *LoadBB = MF->CreateMachineBasicBlock();

  MF->push_back(Entry);
  MF->push_back(StoreBB);
  MF->push_back(LoadBB);

  Entry->addSuccessor(StoreBB);
  StoreBB->addSuccessor(LoadBB);

  MDT = std::make_unique<MachineDominatorTree>();
  MDT->recalculate(*MF);

  const TargetInstrInfo *TII = MF->getSubtarget().getInstrInfo();
  DebugLoc DL;

  MachineInstr *StoreMI =
      BuildMI(StoreBB, DL, TII->get(TargetOpcode::INLINEASM))
          .addExternalSymbol("store")
          .addImm(1);

  MachineInstr *LoadMI = BuildMI(LoadBB, DL, TII->get(TargetOpcode::INLINEASM))
                             .addExternalSymbol("load")
                             .addImm(1);

  int Slot = MF->getFrameInfo().CreateSpillStackObject(4, Align(4));
  StoreMI->addOperand(*MF, MachineOperand::CreateFI(Slot));
  LoadMI->addOperand(*MF, MachineOperand::CreateFI(Slot));

  SmallVector<MachineInstr *, 4> Stores = {StoreMI};
  SmallVector<MachineInstr *, 4> Loads = {LoadMI};

  bool Result =
      AMDGPU::checkAGPRCopyMFMAJointDominance(*MF, *MDT, Stores, Loads, Slot);
  EXPECT_TRUE(Result);
}

} // namespace
