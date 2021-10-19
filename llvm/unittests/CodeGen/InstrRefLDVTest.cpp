//===------------- llvm/unittest/CodeGen/InstrRefLDVTest.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include "../lib/CodeGen/LiveDebugValues/InstrRefBasedImpl.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace LiveDebugValues;

// Include helper functions to ease the manipulation of MachineFunctions
#include "MFCommon.inc"

class InstrRefLDVTest : public testing::Test {
public:
  friend class InstrRefBasedLDV;
  using MLocTransferMap = InstrRefBasedLDV::MLocTransferMap;

  LLVMContext Ctx;
  Module Mod;
  std::unique_ptr<TargetMachine> Machine;
  std::unique_ptr<MachineFunction> MF;
  std::unique_ptr<MachineDominatorTree> DomTree;
  DICompileUnit *OurCU;
  DIFile *OurFile;
  DISubprogram *OurFunc;
  DILexicalBlock *OurBlock, *AnotherBlock;
  DISubprogram *ToInlineFunc;
  DILexicalBlock *ToInlineBlock;
  DILocalVariable *FuncVariable;
  DIBasicType *LongInt;
  DIExpression *EmptyExpr;

  DebugLoc OutermostLoc, InBlockLoc, NotNestedBlockLoc, InlinedLoc;

  MachineBasicBlock *MBB0, *MBB1, *MBB2, *MBB3, *MBB4;

  std::unique_ptr<InstrRefBasedLDV> LDV;
  std::unique_ptr<MLocTracker> MTracker;
  std::unique_ptr<VLocTracker> VTracker;

  InstrRefLDVTest() : Ctx(), Mod("beehives", Ctx) {
  }

  void SetUp() {
    // Boilerplate that creates a MachineFunction and associated blocks.

    Mod.setDataLayout("e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-f64:32:64-f80:32-n8:16:32-S128");
    Triple TargetTriple("x86_64--");
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget("", TargetTriple, Error);
    if (!T)
      GTEST_SKIP();

    TargetOptions Options;
    Machine = std::unique_ptr<TargetMachine>(T->createTargetMachine(
        "X86", "", "", Options, None, None, CodeGenOpt::Aggressive));

    auto Type = FunctionType::get(Type::getVoidTy(Ctx), false);
    auto F = Function::Create(Type, GlobalValue::ExternalLinkage, "Test", &Mod);

    unsigned FunctionNum = 42;
    MachineModuleInfo MMI((LLVMTargetMachine*)&*Machine);
    const TargetSubtargetInfo &STI = *Machine->getSubtargetImpl(*F);

    MF = std::make_unique<MachineFunction>(*F, (LLVMTargetMachine&)*Machine, STI, FunctionNum, MMI);

    // Create metadata: CU, subprogram, some blocks and an inline function
    // scope.
    DIBuilder DIB(Mod);
    OurFile = DIB.createFile("xyzzy.c", "/cave");
    OurCU =
        DIB.createCompileUnit(dwarf::DW_LANG_C99, OurFile, "nou", false, "", 0);
    auto OurSubT = DIB.createSubroutineType(DIB.getOrCreateTypeArray(None));
    OurFunc =
        DIB.createFunction(OurCU, "bees", "", OurFile, 1, OurSubT, 1,
                           DINode::FlagZero, DISubprogram::SPFlagDefinition);
    F->setSubprogram(OurFunc);
    OurBlock = DIB.createLexicalBlock(OurFunc, OurFile, 2, 3);
    AnotherBlock = DIB.createLexicalBlock(OurFunc, OurFile, 2, 6);
    ToInlineFunc =
        DIB.createFunction(OurFile, "shoes", "", OurFile, 10, OurSubT, 10,
                           DINode::FlagZero, DISubprogram::SPFlagDefinition);

    // Make some nested scopes.
    OutermostLoc = DILocation::get(Ctx, 3, 1, OurFunc);
    InBlockLoc = DILocation::get(Ctx, 4, 1, OurBlock);
    InlinedLoc = DILocation::get(Ctx, 10, 1, ToInlineFunc, InBlockLoc.get());

    // Make a scope that isn't nested within the others.
    NotNestedBlockLoc = DILocation::get(Ctx, 4, 1, AnotherBlock);

    LongInt = DIB.createBasicType("long", 64, llvm::dwarf::DW_ATE_unsigned);
    FuncVariable = DIB.createAutoVariable(OurFunc, "lala", OurFile, 1, LongInt);
    EmptyExpr = DIExpression::get(Ctx, {});

    DIB.finalize();
  }

  Register getRegByName(const char *WantedName) {
    auto *TRI = MF->getRegInfo().getTargetRegisterInfo();
    // Slow, but works.
    for (unsigned int I = 1; I < TRI->getNumRegs(); ++I) {
      const char *Name = TRI->getName(I);
      if (strcmp(WantedName, Name) == 0)
        return I;
    }

    // If this ever fails, something is very wrong with this unit test.
    llvm_unreachable("Can't find register by name");
  }

  InstrRefBasedLDV *setupLDVObj() {
    // Create a new LDV object, and plug some relevant object ptrs into it.
    LDV = std::make_unique<InstrRefBasedLDV>();
    const TargetSubtargetInfo &STI = MF->getSubtarget();
    LDV->TII = STI.getInstrInfo();
    LDV->TRI = STI.getRegisterInfo();
    LDV->TFI = STI.getFrameLowering();
    LDV->MFI = &MF->getFrameInfo();

    DomTree = std::make_unique<MachineDominatorTree>(*MF);
    LDV->DomTree = &*DomTree;

    // Future work: unit tests for mtracker / vtracker / ttracker.

    // Setup things like the artifical block map, and BlockNo <=> RPO Order
    // mappings.
    LDV->initialSetup(*MF);
    LDV->LS.initialize(*MF);
    addMTracker();
    addVTracker();
    return &*LDV;
  }

  void addMTracker() {
    ASSERT_TRUE(LDV);
    // Add a machine-location-tracking object to LDV. Don't initialize any
    // register locations within it though.
    const TargetSubtargetInfo &STI = MF->getSubtarget();
    MTracker = std::make_unique<MLocTracker>(
          *MF, *LDV->TII, *LDV->TRI, *STI.getTargetLowering());
    LDV->MTracker = &*MTracker;
  }

  void addVTracker() {
    ASSERT_TRUE(LDV);
    VTracker = std::make_unique<VLocTracker>();
    LDV->VTracker = &*VTracker;
  }

  // Some routines for bouncing into LDV,
  void buildMLocValueMap(ValueIDNum **MInLocs, ValueIDNum **MOutLocs,
                         SmallVectorImpl<MLocTransferMap> &MLocTransfer) {
    LDV->buildMLocValueMap(*MF, MInLocs, MOutLocs, MLocTransfer);
  }

  Optional<ValueIDNum>
  pickVPHILoc(const MachineBasicBlock &MBB, const DebugVariable &Var,
              const InstrRefBasedLDV::LiveIdxT &LiveOuts, ValueIDNum **MOutLocs,
              const SmallVectorImpl<const MachineBasicBlock *> &BlockOrders) {
    return LDV->pickVPHILoc(MBB, Var, LiveOuts, MOutLocs, BlockOrders);
  }

  bool vlocJoin(MachineBasicBlock &MBB, InstrRefBasedLDV::LiveIdxT &VLOCOutLocs,
                const SmallSet<DebugVariable, 4> &AllVars,
                SmallPtrSet<const MachineBasicBlock *, 8> &InScopeBlocks,
                SmallPtrSet<const MachineBasicBlock *, 8> &BlocksToExplore,
                DenseMap<DebugVariable, DbgValue> &InLocsT) {
    return LDV->vlocJoin(MBB, VLOCOutLocs, AllVars, InScopeBlocks,
                         BlocksToExplore, InLocsT);
  }

  void buildVLocValueMap(const DILocation *DILoc,
                    const SmallSet<DebugVariable, 4> &VarsWeCareAbout,
                    SmallPtrSetImpl<MachineBasicBlock *> &AssignBlocks,
                    InstrRefBasedLDV::LiveInsT &Output, ValueIDNum **MOutLocs,
                    ValueIDNum **MInLocs,
                    SmallVectorImpl<VLocTracker> &AllTheVLocs) {
    LDV->buildVLocValueMap(DILoc, VarsWeCareAbout, AssignBlocks, Output,
                           MOutLocs, MInLocs, AllTheVLocs);
  }

  void initValueArray(ValueIDNum **Nums, unsigned Blks, unsigned Locs) {
    for (unsigned int I = 0; I < Blks; ++I)
      for (unsigned int J = 0; J < Locs; ++J)
        Nums[I][J] = ValueIDNum::EmptyValue;
  }

  void setupSingleBlock() {
    // Add an entry block with nothing but 'ret void' in it.
    Function &F = const_cast<llvm::Function &>(MF->getFunction());
    auto *BB0 = BasicBlock::Create(Ctx, "entry", &F);
    IRBuilder<> IRB(BB0);
    IRB.CreateRetVoid();
    MBB0 = MF->CreateMachineBasicBlock(BB0);
    MF->insert(MF->end(), MBB0);
    MF->RenumberBlocks();

    setupLDVObj();
  }

  void setupDiamondBlocks() {
    //        entry
    //        /  \
    //      br1  br2
    //        \  /
    //         ret
    llvm::Function &F = const_cast<llvm::Function &>(MF->getFunction());
    auto *BB0 = BasicBlock::Create(Ctx, "a", &F);
    auto *BB1 = BasicBlock::Create(Ctx, "b", &F);
    auto *BB2 = BasicBlock::Create(Ctx, "c", &F);
    auto *BB3 = BasicBlock::Create(Ctx, "d", &F);
    IRBuilder<> IRB0(BB0), IRB1(BB1), IRB2(BB2), IRB3(BB3);
    IRB0.CreateBr(BB1);
    IRB1.CreateBr(BB2);
    IRB2.CreateBr(BB3);
    IRB3.CreateRetVoid();
    MBB0 = MF->CreateMachineBasicBlock(BB0);
    MF->insert(MF->end(), MBB0);
    MBB1 = MF->CreateMachineBasicBlock(BB1);
    MF->insert(MF->end(), MBB1);
    MBB2 = MF->CreateMachineBasicBlock(BB2);
    MF->insert(MF->end(), MBB2);
    MBB3 = MF->CreateMachineBasicBlock(BB3);
    MF->insert(MF->end(), MBB3);
    MBB0->addSuccessor(MBB1);
    MBB0->addSuccessor(MBB2);
    MBB1->addSuccessor(MBB3);
    MBB2->addSuccessor(MBB3);
    MF->RenumberBlocks();

    setupLDVObj();
  }

  void setupSimpleLoop() {
    //    entry
    //     |
    //     |/-----\
    //    loopblk |
    //     |\-----/
    //     |
    //     ret
    llvm::Function &F = const_cast<llvm::Function &>(MF->getFunction());
    auto *BB0 = BasicBlock::Create(Ctx, "entry", &F);
    auto *BB1 = BasicBlock::Create(Ctx, "loop", &F);
    auto *BB2 = BasicBlock::Create(Ctx, "ret", &F);
    IRBuilder<> IRB0(BB0), IRB1(BB1), IRB2(BB2);
    IRB0.CreateBr(BB1);
    IRB1.CreateBr(BB2);
    IRB2.CreateRetVoid();
    MBB0 = MF->CreateMachineBasicBlock(BB0);
    MF->insert(MF->end(), MBB0);
    MBB1 = MF->CreateMachineBasicBlock(BB1);
    MF->insert(MF->end(), MBB1);
    MBB2 = MF->CreateMachineBasicBlock(BB2);
    MF->insert(MF->end(), MBB2);
    MBB0->addSuccessor(MBB1);
    MBB1->addSuccessor(MBB2);
    MBB1->addSuccessor(MBB1);
    MF->RenumberBlocks();

    setupLDVObj();
  }

  void setupNestedLoops() {
    //    entry
    //     |
    //    loop1
    //     ^\
    //     | \    /-\
    //     |  loop2  |
    //     |  /   \-/
    //     ^ /
    //     join
    //     |
    //     ret
    llvm::Function &F = const_cast<llvm::Function &>(MF->getFunction());
    auto *BB0 = BasicBlock::Create(Ctx, "entry", &F);
    auto *BB1 = BasicBlock::Create(Ctx, "loop1", &F);
    auto *BB2 = BasicBlock::Create(Ctx, "loop2", &F);
    auto *BB3 = BasicBlock::Create(Ctx, "join", &F);
    auto *BB4 = BasicBlock::Create(Ctx, "ret", &F);
    IRBuilder<> IRB0(BB0), IRB1(BB1), IRB2(BB2), IRB3(BB3), IRB4(BB4);
    IRB0.CreateBr(BB1);
    IRB1.CreateBr(BB2);
    IRB2.CreateBr(BB3);
    IRB3.CreateBr(BB4);
    IRB4.CreateRetVoid();
    MBB0 = MF->CreateMachineBasicBlock(BB0);
    MF->insert(MF->end(), MBB0);
    MBB1 = MF->CreateMachineBasicBlock(BB1);
    MF->insert(MF->end(), MBB1);
    MBB2 = MF->CreateMachineBasicBlock(BB2);
    MF->insert(MF->end(), MBB2);
    MBB3 = MF->CreateMachineBasicBlock(BB3);
    MF->insert(MF->end(), MBB3);
    MBB4 = MF->CreateMachineBasicBlock(BB4);
    MF->insert(MF->end(), MBB4);
    MBB0->addSuccessor(MBB1);
    MBB1->addSuccessor(MBB2);
    MBB2->addSuccessor(MBB2);
    MBB2->addSuccessor(MBB3);
    MBB3->addSuccessor(MBB1);
    MBB3->addSuccessor(MBB4);
    MF->RenumberBlocks();

    setupLDVObj();
  }

  void setupNoDominatingLoop() {
    //           entry
    //            / \
    //           /   \
    //          /     \
    //        head1   head2
    //        ^  \   /   ^
    //        ^   \ /    ^
    //        \-joinblk -/
    //             |
    //            ret
    llvm::Function &F = const_cast<llvm::Function &>(MF->getFunction());
    auto *BB0 = BasicBlock::Create(Ctx, "entry", &F);
    auto *BB1 = BasicBlock::Create(Ctx, "head1", &F);
    auto *BB2 = BasicBlock::Create(Ctx, "head2", &F);
    auto *BB3 = BasicBlock::Create(Ctx, "joinblk", &F);
    auto *BB4 = BasicBlock::Create(Ctx, "ret", &F);
    IRBuilder<> IRB0(BB0), IRB1(BB1), IRB2(BB2), IRB3(BB3), IRB4(BB4);
    IRB0.CreateBr(BB1);
    IRB1.CreateBr(BB2);
    IRB2.CreateBr(BB3);
    IRB3.CreateBr(BB4);
    IRB4.CreateRetVoid();
    MBB0 = MF->CreateMachineBasicBlock(BB0);
    MF->insert(MF->end(), MBB0);
    MBB1 = MF->CreateMachineBasicBlock(BB1);
    MF->insert(MF->end(), MBB1);
    MBB2 = MF->CreateMachineBasicBlock(BB2);
    MF->insert(MF->end(), MBB2);
    MBB3 = MF->CreateMachineBasicBlock(BB3);
    MF->insert(MF->end(), MBB3);
    MBB4 = MF->CreateMachineBasicBlock(BB4);
    MF->insert(MF->end(), MBB4);
    MBB0->addSuccessor(MBB1);
    MBB0->addSuccessor(MBB2);
    MBB1->addSuccessor(MBB3);
    MBB2->addSuccessor(MBB3);
    MBB3->addSuccessor(MBB1);
    MBB3->addSuccessor(MBB2);
    MBB3->addSuccessor(MBB4);
    MF->RenumberBlocks();

    setupLDVObj();
  }

  void setupBadlyNestedLoops() {
    //           entry
    //             |
    //           loop1 -o
    //             | ^
    //             | ^
    //           loop2 -o
    //             | ^
    //             | ^
    //           loop3 -o
    //             |
    //            ret
    //
    // NB: the loop blocks self-loop, which is a bit too fiddly to draw on
    // accurately.
    llvm::Function &F = const_cast<llvm::Function &>(MF->getFunction());
    auto *BB0 = BasicBlock::Create(Ctx, "entry", &F);
    auto *BB1 = BasicBlock::Create(Ctx, "loop1", &F);
    auto *BB2 = BasicBlock::Create(Ctx, "loop2", &F);
    auto *BB3 = BasicBlock::Create(Ctx, "loop3", &F);
    auto *BB4 = BasicBlock::Create(Ctx, "ret", &F);
    IRBuilder<> IRB0(BB0), IRB1(BB1), IRB2(BB2), IRB3(BB3), IRB4(BB4);
    IRB0.CreateBr(BB1);
    IRB1.CreateBr(BB2);
    IRB2.CreateBr(BB3);
    IRB3.CreateBr(BB4);
    IRB4.CreateRetVoid();
    MBB0 = MF->CreateMachineBasicBlock(BB0);
    MF->insert(MF->end(), MBB0);
    MBB1 = MF->CreateMachineBasicBlock(BB1);
    MF->insert(MF->end(), MBB1);
    MBB2 = MF->CreateMachineBasicBlock(BB2);
    MF->insert(MF->end(), MBB2);
    MBB3 = MF->CreateMachineBasicBlock(BB3);
    MF->insert(MF->end(), MBB3);
    MBB4 = MF->CreateMachineBasicBlock(BB4);
    MF->insert(MF->end(), MBB4);
    MBB0->addSuccessor(MBB1);
    MBB1->addSuccessor(MBB1);
    MBB1->addSuccessor(MBB2);
    MBB2->addSuccessor(MBB1);
    MBB2->addSuccessor(MBB2);
    MBB2->addSuccessor(MBB3);
    MBB3->addSuccessor(MBB2);
    MBB3->addSuccessor(MBB3);
    MBB3->addSuccessor(MBB4);
    MF->RenumberBlocks();

    setupLDVObj();
  }
};

TEST_F(InstrRefLDVTest, MLocSingleBlock) {
  // Test some very simple properties about interpreting the transfer function.
  setupSingleBlock();

  // We should start with a single location, the stack pointer.
  ASSERT_TRUE(MTracker->getNumLocs() == 1);
  LocIdx RspLoc(0);

  // Set up live-in and live-out tables for this function: two locations (we
  // add one later) in a single block.
  ValueIDNum InLocs[2], OutLocs[2];
  ValueIDNum *InLocsPtr[1] = {&InLocs[0]};
  ValueIDNum *OutLocsPtr[1] = {&OutLocs[0]};

  // Transfer function: nothing.
  SmallVector<MLocTransferMap, 1> TransferFunc = {{}};

  // Try and build value maps...
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);

  // The result should be that RSP is marked as a live-in-PHI -- this represents
  // an argument. And as there's no transfer function, the block live-out should
  // be the same.
  EXPECT_EQ(InLocs[0], ValueIDNum(0, 0, RspLoc));
  EXPECT_EQ(OutLocs[0], ValueIDNum(0, 0, RspLoc));

  // Try again, this time initialising the in-locs to be defined by an
  // instruction. The entry block should always be re-assigned to be the
  // arguments.
  initValueArray(InLocsPtr, 1, 2);
  initValueArray(OutLocsPtr, 1, 2);
  InLocs[0] = ValueIDNum(0, 1, RspLoc);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  EXPECT_EQ(InLocs[0], ValueIDNum(0, 0, RspLoc));
  EXPECT_EQ(OutLocs[0], ValueIDNum(0, 0, RspLoc));

  // Now insert something into the transfer function to assign to the single
  // machine location.
  TransferFunc[0].insert({RspLoc, ValueIDNum(0, 1, RspLoc)});
  initValueArray(InLocsPtr, 1, 2);
  initValueArray(OutLocsPtr, 1, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  EXPECT_EQ(InLocs[0], ValueIDNum(0, 0, RspLoc));
  EXPECT_EQ(OutLocs[0], ValueIDNum(0, 1, RspLoc));
  TransferFunc[0].clear();

  // Add a new register to be tracked, and insert it into the transfer function
  // as a copy. The output of $rax should be the live-in value of $rsp.
  Register RAX = getRegByName("RAX");
  LocIdx RaxLoc = MTracker->lookupOrTrackRegister(RAX);
  TransferFunc[0].insert({RspLoc, ValueIDNum(0, 1, RspLoc)});
  TransferFunc[0].insert({RaxLoc, ValueIDNum(0, 0, RspLoc)});
  initValueArray(InLocsPtr, 1, 2);
  initValueArray(OutLocsPtr, 1, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  EXPECT_EQ(InLocs[0], ValueIDNum(0, 0, RspLoc));
  EXPECT_EQ(InLocs[1], ValueIDNum(0, 0, RaxLoc));
  EXPECT_EQ(OutLocs[0], ValueIDNum(0, 1, RspLoc));
  EXPECT_EQ(OutLocs[1], ValueIDNum(0, 0, RspLoc)); // Rax contains RspLoc.
  TransferFunc[0].clear();
}

TEST_F(InstrRefLDVTest, MLocDiamondBlocks) {
  // Test that information flows from the entry block to two successors.
  //        entry
  //        /  \
  //      br1  br2
  //        \  /
  //         ret
  setupDiamondBlocks();

  ASSERT_TRUE(MTracker->getNumLocs() == 1);
  LocIdx RspLoc(0);
  Register RAX = getRegByName("RAX");
  LocIdx RaxLoc = MTracker->lookupOrTrackRegister(RAX);

  ValueIDNum InLocs[4][2], OutLocs[4][2];
  ValueIDNum *InLocsPtr[4] = {InLocs[0], InLocs[1], InLocs[2], InLocs[3]};
  ValueIDNum *OutLocsPtr[4] = {OutLocs[0], OutLocs[1], OutLocs[2], OutLocs[3]};

  // Transfer function: start with nothing.
  SmallVector<MLocTransferMap, 1> TransferFunc;
  TransferFunc.resize(4);

  // Name some values.
  unsigned EntryBlk = 0, BrBlk1 = 1, BrBlk2 = 2, RetBlk = 3;

  ValueIDNum LiveInRsp(EntryBlk, 0, RspLoc);
  ValueIDNum RspDefInBlk0(EntryBlk, 1, RspLoc);
  ValueIDNum RspDefInBlk1(BrBlk1, 1, RspLoc);
  ValueIDNum RspDefInBlk2(BrBlk2, 1, RspLoc);
  ValueIDNum RspPHIInBlk3(RetBlk, 0, RspLoc);
  ValueIDNum RaxLiveInBlk1(BrBlk1, 0, RaxLoc);
  ValueIDNum RaxLiveInBlk2(BrBlk2, 0, RaxLoc);

  // With no transfer function, the live-in values to the entry block should
  // propagate to all live-outs and the live-ins to the two successor blocks.
  // IN ADDITION: this checks that the exit block doesn't get a PHI put in it.
  initValueArray(InLocsPtr, 4, 2);
  initValueArray(OutLocsPtr, 4, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  EXPECT_EQ(InLocs[0][0], LiveInRsp);
  EXPECT_EQ(InLocs[1][0], LiveInRsp);
  EXPECT_EQ(InLocs[2][0], LiveInRsp);
  EXPECT_EQ(InLocs[3][0], LiveInRsp);
  EXPECT_EQ(OutLocs[0][0], LiveInRsp);
  EXPECT_EQ(OutLocs[1][0], LiveInRsp);
  EXPECT_EQ(OutLocs[2][0], LiveInRsp);
  EXPECT_EQ(OutLocs[3][0], LiveInRsp);
  // (Skipped writing out locations for $rax).

  // Check that a def of $rsp in the entry block will likewise reach all the
  // successors.
  TransferFunc[0].insert({RspLoc, RspDefInBlk0});
  initValueArray(InLocsPtr, 4, 2);
  initValueArray(OutLocsPtr, 4, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  EXPECT_EQ(InLocs[0][0], LiveInRsp);
  EXPECT_EQ(InLocs[1][0], RspDefInBlk0);
  EXPECT_EQ(InLocs[2][0], RspDefInBlk0);
  EXPECT_EQ(InLocs[3][0], RspDefInBlk0);
  EXPECT_EQ(OutLocs[0][0], RspDefInBlk0);
  EXPECT_EQ(OutLocs[1][0], RspDefInBlk0);
  EXPECT_EQ(OutLocs[2][0], RspDefInBlk0);
  EXPECT_EQ(OutLocs[3][0], RspDefInBlk0);
  TransferFunc[0].clear();

  // Def in one branch of the diamond means that we need a PHI in the ret block
  TransferFunc[0].insert({RspLoc, RspDefInBlk0});
  TransferFunc[1].insert({RspLoc, RspDefInBlk1});
  initValueArray(InLocsPtr, 4, 2);
  initValueArray(OutLocsPtr, 4, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  // This value map: like above, where RspDefInBlk0 is propagated through one
  // branch of the diamond, but is def'ed in the live-outs of the other. The
  // ret / merging block should have a PHI in its live-ins.
  EXPECT_EQ(InLocs[0][0], LiveInRsp);
  EXPECT_EQ(InLocs[1][0], RspDefInBlk0);
  EXPECT_EQ(InLocs[2][0], RspDefInBlk0);
  EXPECT_EQ(InLocs[3][0], RspPHIInBlk3);
  EXPECT_EQ(OutLocs[0][0], RspDefInBlk0);
  EXPECT_EQ(OutLocs[1][0], RspDefInBlk1);
  EXPECT_EQ(OutLocs[2][0], RspDefInBlk0);
  EXPECT_EQ(OutLocs[3][0], RspPHIInBlk3);
  TransferFunc[0].clear();
  TransferFunc[1].clear();

  // If we have differeing defs in either side of the diamond, we should
  // continue to produce a PHI,
  TransferFunc[0].insert({RspLoc, RspDefInBlk0});
  TransferFunc[1].insert({RspLoc, RspDefInBlk1});
  TransferFunc[2].insert({RspLoc, RspDefInBlk2});
  initValueArray(InLocsPtr, 4, 2);
  initValueArray(OutLocsPtr, 4, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  EXPECT_EQ(InLocs[0][0], LiveInRsp);
  EXPECT_EQ(InLocs[1][0], RspDefInBlk0);
  EXPECT_EQ(InLocs[2][0], RspDefInBlk0);
  EXPECT_EQ(InLocs[3][0], RspPHIInBlk3);
  EXPECT_EQ(OutLocs[0][0], RspDefInBlk0);
  EXPECT_EQ(OutLocs[1][0], RspDefInBlk1);
  EXPECT_EQ(OutLocs[2][0], RspDefInBlk2);
  EXPECT_EQ(OutLocs[3][0], RspPHIInBlk3);
  TransferFunc[0].clear();
  TransferFunc[1].clear();
  TransferFunc[2].clear();

  // If we have defs of the same value on either side of the branch, a PHI will
  // initially be created, however value propagation should then eliminate it.
  // Encode this by copying the live-in value to $rax, and copying it to $rsp
  // from $rax in each branch of the diamond. We don't allow the definition of
  // arbitary values in transfer functions.
  TransferFunc[0].insert({RspLoc, RspDefInBlk0});
  TransferFunc[0].insert({RaxLoc, LiveInRsp});
  TransferFunc[1].insert({RspLoc, RaxLiveInBlk1});
  TransferFunc[2].insert({RspLoc, RaxLiveInBlk2});
  initValueArray(InLocsPtr, 4, 2);
  initValueArray(OutLocsPtr, 4, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  EXPECT_EQ(InLocs[0][0], LiveInRsp);
  EXPECT_EQ(InLocs[1][0], RspDefInBlk0);
  EXPECT_EQ(InLocs[2][0], RspDefInBlk0);
  EXPECT_EQ(InLocs[3][0], LiveInRsp);
  EXPECT_EQ(OutLocs[0][0], RspDefInBlk0);
  EXPECT_EQ(OutLocs[1][0], LiveInRsp);
  EXPECT_EQ(OutLocs[2][0], LiveInRsp);
  EXPECT_EQ(OutLocs[3][0], LiveInRsp);
  TransferFunc[0].clear();
  TransferFunc[1].clear();
  TransferFunc[2].clear();
}

TEST_F(InstrRefLDVTest, MLocSimpleLoop) {
  //    entry
  //     |
  //     |/-----\
  //    loopblk |
  //     |\-----/
  //     |
  //     ret
  setupSimpleLoop();

  ASSERT_TRUE(MTracker->getNumLocs() == 1);
  LocIdx RspLoc(0);
  Register RAX = getRegByName("RAX");
  LocIdx RaxLoc = MTracker->lookupOrTrackRegister(RAX);

  ValueIDNum InLocs[3][2], OutLocs[3][2];
  ValueIDNum *InLocsPtr[3] = {InLocs[0], InLocs[1], InLocs[2]};
  ValueIDNum *OutLocsPtr[3] = {OutLocs[0], OutLocs[1], OutLocs[2]};

  SmallVector<MLocTransferMap, 1> TransferFunc;
  TransferFunc.resize(3);

  // Name some values.
  unsigned EntryBlk = 0, LoopBlk = 1, RetBlk = 2;

  ValueIDNum LiveInRsp(EntryBlk, 0, RspLoc);
  ValueIDNum RspPHIInBlk1(LoopBlk, 0, RspLoc);
  ValueIDNum RspDefInBlk1(LoopBlk, 1, RspLoc);
  ValueIDNum LiveInRax(EntryBlk, 0, RaxLoc);
  ValueIDNum RaxPHIInBlk1(LoopBlk, 0, RaxLoc);
  ValueIDNum RaxPHIInBlk2(RetBlk, 0, RaxLoc);

  // Begin test with all locations being live-through.
  initValueArray(InLocsPtr, 3, 2);
  initValueArray(OutLocsPtr, 3, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  EXPECT_EQ(InLocs[0][0], LiveInRsp);
  EXPECT_EQ(InLocs[1][0], LiveInRsp);
  EXPECT_EQ(InLocs[2][0], LiveInRsp);
  EXPECT_EQ(OutLocs[0][0], LiveInRsp);
  EXPECT_EQ(OutLocs[1][0], LiveInRsp);
  EXPECT_EQ(OutLocs[2][0], LiveInRsp);

  // Add a def of $rsp to the loop block: it should be in the live-outs, but
  // should cause a PHI to be placed in the live-ins. Test the transfer function
  // by copying that PHI into $rax in the loop, then back to $rsp in the ret
  // block.
  TransferFunc[1].insert({RspLoc, RspDefInBlk1});
  TransferFunc[1].insert({RaxLoc, RspPHIInBlk1});
  TransferFunc[2].insert({RspLoc, RaxPHIInBlk2});
  initValueArray(InLocsPtr, 3, 2);
  initValueArray(OutLocsPtr, 3, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  EXPECT_EQ(InLocs[0][0], LiveInRsp);
  EXPECT_EQ(InLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(InLocs[2][0], RspDefInBlk1);
  EXPECT_EQ(OutLocs[0][0], LiveInRsp);
  EXPECT_EQ(OutLocs[1][0], RspDefInBlk1);
  EXPECT_EQ(OutLocs[2][0], RspPHIInBlk1);
  // Check rax as well,
  EXPECT_EQ(InLocs[0][1], LiveInRax);
  EXPECT_EQ(InLocs[1][1], RaxPHIInBlk1);
  EXPECT_EQ(InLocs[2][1], RspPHIInBlk1);
  EXPECT_EQ(OutLocs[0][1], LiveInRax);
  EXPECT_EQ(OutLocs[1][1], RspPHIInBlk1);
  EXPECT_EQ(OutLocs[2][1], RspPHIInBlk1);
  TransferFunc[1].clear();
  TransferFunc[2].clear();

  // As with the diamond case, a PHI will be created if there's a (implicit)
  // def in the entry block and loop block; but should be value propagated away
  // if it copies in the same value. Copy live-in $rsp to $rax, then copy it
  // into $rsp in the loop. Encoded as copying the live-in $rax value in block 1
  // to $rsp.
  TransferFunc[0].insert({RaxLoc, LiveInRsp});
  TransferFunc[1].insert({RspLoc, RaxPHIInBlk1});
  initValueArray(InLocsPtr, 3, 2);
  initValueArray(OutLocsPtr, 3, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  EXPECT_EQ(InLocs[0][0], LiveInRsp);
  EXPECT_EQ(InLocs[1][0], LiveInRsp);
  EXPECT_EQ(InLocs[2][0], LiveInRsp);
  EXPECT_EQ(OutLocs[0][0], LiveInRsp);
  EXPECT_EQ(OutLocs[1][0], LiveInRsp);
  EXPECT_EQ(OutLocs[2][0], LiveInRsp);
  // Check $rax's values.
  EXPECT_EQ(InLocs[0][1], LiveInRax);
  EXPECT_EQ(InLocs[1][1], LiveInRsp);
  EXPECT_EQ(InLocs[2][1], LiveInRsp);
  EXPECT_EQ(OutLocs[0][1], LiveInRsp);
  EXPECT_EQ(OutLocs[1][1], LiveInRsp);
  EXPECT_EQ(OutLocs[2][1], LiveInRsp);
  TransferFunc[0].clear();
  TransferFunc[1].clear();
}

TEST_F(InstrRefLDVTest, MLocNestedLoop) {
  //    entry
  //     |
  //    loop1
  //     ^\
  //     | \    /-\
  //     |  loop2  |
  //     |  /   \-/
  //     ^ /
  //     join
  //     |
  //     ret
  setupNestedLoops();

  ASSERT_TRUE(MTracker->getNumLocs() == 1);
  LocIdx RspLoc(0);
  Register RAX = getRegByName("RAX");
  LocIdx RaxLoc = MTracker->lookupOrTrackRegister(RAX);

  ValueIDNum InLocs[5][2], OutLocs[5][2];
  ValueIDNum *InLocsPtr[5] = {InLocs[0], InLocs[1], InLocs[2], InLocs[3],
                              InLocs[4]};
  ValueIDNum *OutLocsPtr[5] = {OutLocs[0], OutLocs[1], OutLocs[2], OutLocs[3],
                               OutLocs[4]};

  SmallVector<MLocTransferMap, 1> TransferFunc;
  TransferFunc.resize(5);

  unsigned EntryBlk = 0, Loop1Blk = 1, Loop2Blk = 2, JoinBlk = 3;

  ValueIDNum LiveInRsp(EntryBlk, 0, RspLoc);
  ValueIDNum RspPHIInBlk1(Loop1Blk, 0, RspLoc);
  ValueIDNum RspDefInBlk1(Loop1Blk, 1, RspLoc);
  ValueIDNum RspPHIInBlk2(Loop2Blk, 0, RspLoc);
  ValueIDNum RspDefInBlk2(Loop2Blk, 1, RspLoc);
  ValueIDNum RspDefInBlk3(JoinBlk, 1, RspLoc);
  ValueIDNum LiveInRax(EntryBlk, 0, RaxLoc);
  ValueIDNum RaxPHIInBlk1(Loop1Blk, 0, RaxLoc);
  ValueIDNum RaxPHIInBlk2(Loop2Blk, 0, RaxLoc);

  // Like the other tests: first ensure that if there's nothing in the transfer
  // function, then everything is live-through (check $rsp).
  initValueArray(InLocsPtr, 5, 2);
  initValueArray(OutLocsPtr, 5, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  EXPECT_EQ(InLocs[0][0], LiveInRsp);
  EXPECT_EQ(InLocs[1][0], LiveInRsp);
  EXPECT_EQ(InLocs[2][0], LiveInRsp);
  EXPECT_EQ(InLocs[3][0], LiveInRsp);
  EXPECT_EQ(InLocs[4][0], LiveInRsp);
  EXPECT_EQ(OutLocs[0][0], LiveInRsp);
  EXPECT_EQ(OutLocs[1][0], LiveInRsp);
  EXPECT_EQ(OutLocs[2][0], LiveInRsp);
  EXPECT_EQ(OutLocs[3][0], LiveInRsp);
  EXPECT_EQ(OutLocs[4][0], LiveInRsp);

  // A def in the inner loop means we should get PHIs at the heads of both
  // loops. Live-outs of the last three blocks will be the def, as it dominates
  // those.
  TransferFunc[2].insert({RspLoc, RspDefInBlk2});
  initValueArray(InLocsPtr, 5, 2);
  initValueArray(OutLocsPtr, 5, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  EXPECT_EQ(InLocs[0][0], LiveInRsp);
  EXPECT_EQ(InLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(InLocs[2][0], RspPHIInBlk2);
  EXPECT_EQ(InLocs[3][0], RspDefInBlk2);
  EXPECT_EQ(InLocs[4][0], RspDefInBlk2);
  EXPECT_EQ(OutLocs[0][0], LiveInRsp);
  EXPECT_EQ(OutLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(OutLocs[2][0], RspDefInBlk2);
  EXPECT_EQ(OutLocs[3][0], RspDefInBlk2);
  EXPECT_EQ(OutLocs[4][0], RspDefInBlk2);
  TransferFunc[2].clear();

  // Adding a def to the outer loop header shouldn't affect this much -- the
  // live-out of block 1 changes.
  TransferFunc[1].insert({RspLoc, RspDefInBlk1});
  TransferFunc[2].insert({RspLoc, RspDefInBlk2});
  initValueArray(InLocsPtr, 5, 2);
  initValueArray(OutLocsPtr, 5, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  EXPECT_EQ(InLocs[0][0], LiveInRsp);
  EXPECT_EQ(InLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(InLocs[2][0], RspPHIInBlk2);
  EXPECT_EQ(InLocs[3][0], RspDefInBlk2);
  EXPECT_EQ(InLocs[4][0], RspDefInBlk2);
  EXPECT_EQ(OutLocs[0][0], LiveInRsp);
  EXPECT_EQ(OutLocs[1][0], RspDefInBlk1);
  EXPECT_EQ(OutLocs[2][0], RspDefInBlk2);
  EXPECT_EQ(OutLocs[3][0], RspDefInBlk2);
  EXPECT_EQ(OutLocs[4][0], RspDefInBlk2);
  TransferFunc[1].clear();
  TransferFunc[2].clear();

  // Likewise, putting a def in the outer loop tail shouldn't affect where
  // the PHIs go, and should propagate into the ret block.

  TransferFunc[1].insert({RspLoc, RspDefInBlk1});
  TransferFunc[2].insert({RspLoc, RspDefInBlk2});
  TransferFunc[3].insert({RspLoc, RspDefInBlk3});
  initValueArray(InLocsPtr, 5, 2);
  initValueArray(OutLocsPtr, 5, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  EXPECT_EQ(InLocs[0][0], LiveInRsp);
  EXPECT_EQ(InLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(InLocs[2][0], RspPHIInBlk2);
  EXPECT_EQ(InLocs[3][0], RspDefInBlk2);
  EXPECT_EQ(InLocs[4][0], RspDefInBlk3);
  EXPECT_EQ(OutLocs[0][0], LiveInRsp);
  EXPECT_EQ(OutLocs[1][0], RspDefInBlk1);
  EXPECT_EQ(OutLocs[2][0], RspDefInBlk2);
  EXPECT_EQ(OutLocs[3][0], RspDefInBlk3);
  EXPECT_EQ(OutLocs[4][0], RspDefInBlk3);
  TransferFunc[1].clear();
  TransferFunc[2].clear();
  TransferFunc[3].clear();

  // However: if we don't def in the inner-loop, then we just have defs in the
  // head and tail of the outer loop. The inner loop should be live-through.
  TransferFunc[1].insert({RspLoc, RspDefInBlk1});
  TransferFunc[3].insert({RspLoc, RspDefInBlk3});
  initValueArray(InLocsPtr, 5, 2);
  initValueArray(OutLocsPtr, 5, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  EXPECT_EQ(InLocs[0][0], LiveInRsp);
  EXPECT_EQ(InLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(InLocs[2][0], RspDefInBlk1);
  EXPECT_EQ(InLocs[3][0], RspDefInBlk1);
  EXPECT_EQ(InLocs[4][0], RspDefInBlk3);
  EXPECT_EQ(OutLocs[0][0], LiveInRsp);
  EXPECT_EQ(OutLocs[1][0], RspDefInBlk1);
  EXPECT_EQ(OutLocs[2][0], RspDefInBlk1);
  EXPECT_EQ(OutLocs[3][0], RspDefInBlk3);
  EXPECT_EQ(OutLocs[4][0], RspDefInBlk3);
  TransferFunc[1].clear();
  TransferFunc[3].clear();

  // Check that this still works if we copy RspDefInBlk1 to $rax and then
  // copy it back into $rsp in the inner loop.
  TransferFunc[1].insert({RspLoc, RspDefInBlk1});
  TransferFunc[1].insert({RaxLoc, RspDefInBlk1});
  TransferFunc[2].insert({RspLoc, RaxPHIInBlk2});
  TransferFunc[3].insert({RspLoc, RspDefInBlk3});
  initValueArray(InLocsPtr, 5, 2);
  initValueArray(OutLocsPtr, 5, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  EXPECT_EQ(InLocs[0][0], LiveInRsp);
  EXPECT_EQ(InLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(InLocs[2][0], RspDefInBlk1);
  EXPECT_EQ(InLocs[3][0], RspDefInBlk1);
  EXPECT_EQ(InLocs[4][0], RspDefInBlk3);
  EXPECT_EQ(OutLocs[0][0], LiveInRsp);
  EXPECT_EQ(OutLocs[1][0], RspDefInBlk1);
  EXPECT_EQ(OutLocs[2][0], RspDefInBlk1);
  EXPECT_EQ(OutLocs[3][0], RspDefInBlk3);
  EXPECT_EQ(OutLocs[4][0], RspDefInBlk3);
  // Look at raxes value in the relevant blocks,
  EXPECT_EQ(InLocs[2][1], RspDefInBlk1);
  EXPECT_EQ(OutLocs[1][1], RspDefInBlk1);
  TransferFunc[1].clear();
  TransferFunc[2].clear();
  TransferFunc[3].clear();

  // If we have a single def in the tail of the outer loop, that should produce
  // a PHI at the loop head, and be live-through the inner loop.
  TransferFunc[3].insert({RspLoc, RspDefInBlk3});
  initValueArray(InLocsPtr, 5, 2);
  initValueArray(OutLocsPtr, 5, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  EXPECT_EQ(InLocs[0][0], LiveInRsp);
  EXPECT_EQ(InLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(InLocs[2][0], RspPHIInBlk1);
  EXPECT_EQ(InLocs[3][0], RspPHIInBlk1);
  EXPECT_EQ(InLocs[4][0], RspDefInBlk3);
  EXPECT_EQ(OutLocs[0][0], LiveInRsp);
  EXPECT_EQ(OutLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(OutLocs[2][0], RspPHIInBlk1);
  EXPECT_EQ(OutLocs[3][0], RspDefInBlk3);
  EXPECT_EQ(OutLocs[4][0], RspDefInBlk3);
  TransferFunc[3].clear();

  // And if we copy from $rsp to $rax in block 2, it should resolve to the PHI
  // in block 1, and we should keep that value in rax until the ret block.
  // There'll be a PHI in block 1 and 2, because we're putting a def in the
  // inner loop.
  TransferFunc[2].insert({RaxLoc, RspPHIInBlk2});
  TransferFunc[3].insert({RspLoc, RspDefInBlk3});
  initValueArray(InLocsPtr, 5, 2);
  initValueArray(OutLocsPtr, 5, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  // Examining the values of rax,
  EXPECT_EQ(InLocs[0][1], LiveInRax);
  EXPECT_EQ(InLocs[1][1], RaxPHIInBlk1);
  EXPECT_EQ(InLocs[2][1], RaxPHIInBlk2);
  EXPECT_EQ(InLocs[3][1], RspPHIInBlk1);
  EXPECT_EQ(InLocs[4][1], RspPHIInBlk1);
  EXPECT_EQ(OutLocs[0][1], LiveInRax);
  EXPECT_EQ(OutLocs[1][1], RaxPHIInBlk1);
  EXPECT_EQ(OutLocs[2][1], RspPHIInBlk1);
  EXPECT_EQ(OutLocs[3][1], RspPHIInBlk1);
  EXPECT_EQ(OutLocs[4][1], RspPHIInBlk1);
  TransferFunc[2].clear();
  TransferFunc[3].clear();
}

TEST_F(InstrRefLDVTest, MLocNoDominatingLoop) {
  //           entry
  //            / \
  //           /   \
  //          /     \
  //        head1   head2
  //        ^  \   /   ^
  //        ^   \ /    ^
  //        \-joinblk -/
  //             |
  //            ret
  setupNoDominatingLoop();

  ASSERT_TRUE(MTracker->getNumLocs() == 1);
  LocIdx RspLoc(0);
  Register RAX = getRegByName("RAX");
  LocIdx RaxLoc = MTracker->lookupOrTrackRegister(RAX);

  ValueIDNum InLocs[5][2], OutLocs[5][2];
  ValueIDNum *InLocsPtr[5] = {InLocs[0], InLocs[1], InLocs[2], InLocs[3],
                              InLocs[4]};
  ValueIDNum *OutLocsPtr[5] = {OutLocs[0], OutLocs[1], OutLocs[2], OutLocs[3],
                               OutLocs[4]};

  SmallVector<MLocTransferMap, 1> TransferFunc;
  TransferFunc.resize(5);

  unsigned EntryBlk = 0, Head1Blk = 1, Head2Blk = 2, JoinBlk = 3;

  ValueIDNum LiveInRsp(EntryBlk, 0, RspLoc);
  ValueIDNum RspPHIInBlk1(Head1Blk, 0, RspLoc);
  ValueIDNum RspDefInBlk1(Head1Blk, 1, RspLoc);
  ValueIDNum RspPHIInBlk2(Head2Blk, 0, RspLoc);
  ValueIDNum RspDefInBlk2(Head2Blk, 1, RspLoc);
  ValueIDNum RspPHIInBlk3(JoinBlk, 0, RspLoc);
  ValueIDNum RspDefInBlk3(JoinBlk, 1, RspLoc);
  ValueIDNum RaxPHIInBlk1(Head1Blk, 0, RaxLoc);
  ValueIDNum RaxPHIInBlk2(Head2Blk, 0, RaxLoc);

  // As ever, test that everything is live-through if there are no defs.
  initValueArray(InLocsPtr, 5, 2);
  initValueArray(OutLocsPtr, 5, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  EXPECT_EQ(InLocs[0][0], LiveInRsp);
  EXPECT_EQ(InLocs[1][0], LiveInRsp);
  EXPECT_EQ(InLocs[2][0], LiveInRsp);
  EXPECT_EQ(InLocs[3][0], LiveInRsp);
  EXPECT_EQ(InLocs[4][0], LiveInRsp);
  EXPECT_EQ(OutLocs[0][0], LiveInRsp);
  EXPECT_EQ(OutLocs[1][0], LiveInRsp);
  EXPECT_EQ(OutLocs[2][0], LiveInRsp);
  EXPECT_EQ(OutLocs[3][0], LiveInRsp);
  EXPECT_EQ(OutLocs[4][0], LiveInRsp);

  // Putting a def in the 'join' block will cause us to have two distinct
  // PHIs in each loop head, then on entry to the join block.
  TransferFunc[3].insert({RspLoc, RspDefInBlk3});
  initValueArray(InLocsPtr, 5, 2);
  initValueArray(OutLocsPtr, 5, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  EXPECT_EQ(InLocs[0][0], LiveInRsp);
  EXPECT_EQ(InLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(InLocs[2][0], RspPHIInBlk2);
  EXPECT_EQ(InLocs[3][0], RspPHIInBlk3);
  EXPECT_EQ(InLocs[4][0], RspDefInBlk3);
  EXPECT_EQ(OutLocs[0][0], LiveInRsp);
  EXPECT_EQ(OutLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(OutLocs[2][0], RspPHIInBlk2);
  EXPECT_EQ(OutLocs[3][0], RspDefInBlk3);
  EXPECT_EQ(OutLocs[4][0], RspDefInBlk3);
  TransferFunc[3].clear();

  // We should get the same behaviour if we put the def in either of the
  // loop heads -- it should force the other head to be a PHI.
  TransferFunc[1].insert({RspLoc, RspDefInBlk1});
  initValueArray(InLocsPtr, 5, 2);
  initValueArray(OutLocsPtr, 5, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  EXPECT_EQ(InLocs[0][0], LiveInRsp);
  EXPECT_EQ(InLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(InLocs[2][0], RspPHIInBlk2);
  EXPECT_EQ(InLocs[3][0], RspPHIInBlk3);
  EXPECT_EQ(InLocs[4][0], RspPHIInBlk3);
  EXPECT_EQ(OutLocs[0][0], LiveInRsp);
  EXPECT_EQ(OutLocs[1][0], RspDefInBlk1);
  EXPECT_EQ(OutLocs[2][0], RspPHIInBlk2);
  EXPECT_EQ(OutLocs[3][0], RspPHIInBlk3);
  EXPECT_EQ(OutLocs[4][0], RspPHIInBlk3);
  TransferFunc[1].clear();

  // Check symmetry,
  TransferFunc[2].insert({RspLoc, RspDefInBlk2});
  initValueArray(InLocsPtr, 5, 2);
  initValueArray(OutLocsPtr, 5, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  EXPECT_EQ(InLocs[0][0], LiveInRsp);
  EXPECT_EQ(InLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(InLocs[2][0], RspPHIInBlk2);
  EXPECT_EQ(InLocs[3][0], RspPHIInBlk3);
  EXPECT_EQ(InLocs[4][0], RspPHIInBlk3);
  EXPECT_EQ(OutLocs[0][0], LiveInRsp);
  EXPECT_EQ(OutLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(OutLocs[2][0], RspDefInBlk2);
  EXPECT_EQ(OutLocs[3][0], RspPHIInBlk3);
  EXPECT_EQ(OutLocs[4][0], RspPHIInBlk3);
  TransferFunc[2].clear();

  // Test some scenarios where there _shouldn't_ be any PHIs created at heads.
  // These are those PHIs are created, but value propagation eliminates them.
  // For example, lets copy rsp-livein to $rsp inside each loop head, so that
  // there's no need for a PHI in the join block. Put a def of $rsp in block 3
  // to force PHIs elsewhere.
  TransferFunc[0].insert({RaxLoc, LiveInRsp});
  TransferFunc[1].insert({RspLoc, RaxPHIInBlk1});
  TransferFunc[2].insert({RspLoc, RaxPHIInBlk2});
  TransferFunc[3].insert({RspLoc, RspDefInBlk3});
  initValueArray(InLocsPtr, 5, 2);
  initValueArray(OutLocsPtr, 5, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  EXPECT_EQ(InLocs[0][0], LiveInRsp);
  EXPECT_EQ(InLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(InLocs[2][0], RspPHIInBlk2);
  EXPECT_EQ(InLocs[3][0], LiveInRsp);
  EXPECT_EQ(InLocs[4][0], RspDefInBlk3);
  EXPECT_EQ(OutLocs[0][0], LiveInRsp);
  EXPECT_EQ(OutLocs[1][0], LiveInRsp);
  EXPECT_EQ(OutLocs[2][0], LiveInRsp);
  EXPECT_EQ(OutLocs[3][0], RspDefInBlk3);
  EXPECT_EQ(OutLocs[4][0], RspDefInBlk3);
  TransferFunc[0].clear();
  TransferFunc[1].clear();
  TransferFunc[2].clear();
  TransferFunc[3].clear();

  // In fact, if we eliminate the def in block 3, none of those PHIs are
  // necessary, as we're just repeatedly copying LiveInRsp into $rsp. They
  // should all be value propagated out.
  TransferFunc[0].insert({RaxLoc, LiveInRsp});
  TransferFunc[1].insert({RspLoc, RaxPHIInBlk1});
  TransferFunc[2].insert({RspLoc, RaxPHIInBlk2});
  initValueArray(InLocsPtr, 5, 2);
  initValueArray(OutLocsPtr, 5, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  EXPECT_EQ(InLocs[0][0], LiveInRsp);
  EXPECT_EQ(InLocs[1][0], LiveInRsp);
  EXPECT_EQ(InLocs[2][0], LiveInRsp);
  EXPECT_EQ(InLocs[3][0], LiveInRsp);
  EXPECT_EQ(InLocs[4][0], LiveInRsp);
  EXPECT_EQ(OutLocs[0][0], LiveInRsp);
  EXPECT_EQ(OutLocs[1][0], LiveInRsp);
  EXPECT_EQ(OutLocs[2][0], LiveInRsp);
  EXPECT_EQ(OutLocs[3][0], LiveInRsp);
  EXPECT_EQ(OutLocs[4][0], LiveInRsp);
  TransferFunc[0].clear();
  TransferFunc[1].clear();
  TransferFunc[2].clear();
}

TEST_F(InstrRefLDVTest, MLocBadlyNestedLoops) {
  //           entry
  //             |
  //           loop1 -o
  //             | ^
  //             | ^
  //           loop2 -o
  //             | ^
  //             | ^
  //           loop3 -o
  //             |
  //            ret
  setupBadlyNestedLoops();

  ASSERT_TRUE(MTracker->getNumLocs() == 1);
  LocIdx RspLoc(0);
  Register RAX = getRegByName("RAX");
  LocIdx RaxLoc = MTracker->lookupOrTrackRegister(RAX);

  ValueIDNum InLocs[5][2], OutLocs[5][2];
  ValueIDNum *InLocsPtr[5] = {InLocs[0], InLocs[1], InLocs[2], InLocs[3],
                              InLocs[4]};
  ValueIDNum *OutLocsPtr[5] = {OutLocs[0], OutLocs[1], OutLocs[2], OutLocs[3],
                               OutLocs[4]};

  SmallVector<MLocTransferMap, 1> TransferFunc;
  TransferFunc.resize(5);

  unsigned EntryBlk = 0, Loop1Blk = 1, Loop2Blk = 2, Loop3Blk = 3;

  ValueIDNum LiveInRsp(EntryBlk, 0, RspLoc);
  ValueIDNum RspPHIInBlk1(Loop1Blk, 0, RspLoc);
  ValueIDNum RspDefInBlk1(Loop1Blk, 1, RspLoc);
  ValueIDNum RspPHIInBlk2(Loop2Blk, 0, RspLoc);
  ValueIDNum RspPHIInBlk3(Loop3Blk, 0, RspLoc);
  ValueIDNum RspDefInBlk3(Loop3Blk, 1, RspLoc);
  ValueIDNum LiveInRax(EntryBlk, 0, RaxLoc);
  ValueIDNum RaxPHIInBlk3(Loop3Blk, 0, RaxLoc);

  // As ever, test that everything is live-through if there are no defs.
  initValueArray(InLocsPtr, 5, 2);
  initValueArray(OutLocsPtr, 5, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  EXPECT_EQ(InLocs[0][0], LiveInRsp);
  EXPECT_EQ(InLocs[1][0], LiveInRsp);
  EXPECT_EQ(InLocs[2][0], LiveInRsp);
  EXPECT_EQ(InLocs[3][0], LiveInRsp);
  EXPECT_EQ(InLocs[4][0], LiveInRsp);
  EXPECT_EQ(OutLocs[0][0], LiveInRsp);
  EXPECT_EQ(OutLocs[1][0], LiveInRsp);
  EXPECT_EQ(OutLocs[2][0], LiveInRsp);
  EXPECT_EQ(OutLocs[3][0], LiveInRsp);
  EXPECT_EQ(OutLocs[4][0], LiveInRsp);

  // A def in loop3 should cause PHIs in every loop block: they're all
  // reachable from each other.
  TransferFunc[3].insert({RspLoc, RspDefInBlk3});
  initValueArray(InLocsPtr, 5, 2);
  initValueArray(OutLocsPtr, 5, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  EXPECT_EQ(InLocs[0][0], LiveInRsp);
  EXPECT_EQ(InLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(InLocs[2][0], RspPHIInBlk2);
  EXPECT_EQ(InLocs[3][0], RspPHIInBlk3);
  EXPECT_EQ(InLocs[4][0], RspDefInBlk3);
  EXPECT_EQ(OutLocs[0][0], LiveInRsp);
  EXPECT_EQ(OutLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(OutLocs[2][0], RspPHIInBlk2);
  EXPECT_EQ(OutLocs[3][0], RspDefInBlk3);
  EXPECT_EQ(OutLocs[4][0], RspDefInBlk3);
  TransferFunc[3].clear();

  // A def in loop1 should cause a PHI in loop1, but not the other blocks.
  // loop2 and loop3 are dominated by the def in loop1, so they should have
  // that value live-through.
  TransferFunc[1].insert({RspLoc, RspDefInBlk1});
  initValueArray(InLocsPtr, 5, 2);
  initValueArray(OutLocsPtr, 5, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  EXPECT_EQ(InLocs[0][0], LiveInRsp);
  EXPECT_EQ(InLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(InLocs[2][0], RspDefInBlk1);
  EXPECT_EQ(InLocs[3][0], RspDefInBlk1);
  EXPECT_EQ(InLocs[4][0], RspDefInBlk1);
  EXPECT_EQ(OutLocs[0][0], LiveInRsp);
  EXPECT_EQ(OutLocs[1][0], RspDefInBlk1);
  EXPECT_EQ(OutLocs[2][0], RspDefInBlk1);
  EXPECT_EQ(OutLocs[3][0], RspDefInBlk1);
  EXPECT_EQ(OutLocs[4][0], RspDefInBlk1);
  TransferFunc[1].clear();

  // As with earlier tricks: copy $rsp to $rax in the entry block, then $rax
  // to $rsp in block 3. The only def of $rsp is simply copying the same value
  // back into itself, and the value of $rsp is LiveInRsp all the way through.
  // PHIs should be created, then value-propagated away...  however this
  // doesn't work in practice.
  // Consider the entry to loop3: we can determine that there's an incoming
  // PHI value from loop2, and LiveInRsp from the self-loop. This would still
  // justify having a PHI on entry to loop3. The only way to completely
  // value-propagate these PHIs away would be to speculatively explore what
  // PHIs could be eliminated and what that would lead to; which is
  // combinatorially complex.
  // Happily:
  //  a) In this scenario, we always have a tracked location for LiveInRsp
  //     anyway, so there's no loss in availability,
  //  b) Only DBG_PHIs of a register would be vunlerable to this scenario, and
  //     even then only if a true PHI became a DBG_PHI and was then optimised
  //     through branch folding to no longer be at a CFG join,
  //  c) The register allocator can spot this kind of redundant COPY easily,
  //     and eliminate it.
  //
  // This unit test left in as a reference for the limitations of this
  // approach. PHIs will be left in $rsp on entry to each block.
  TransferFunc[0].insert({RaxLoc, LiveInRsp});
  TransferFunc[3].insert({RspLoc, RaxPHIInBlk3});
  initValueArray(InLocsPtr, 5, 2);
  initValueArray(OutLocsPtr, 5, 2);
  buildMLocValueMap(InLocsPtr, OutLocsPtr, TransferFunc);
  EXPECT_EQ(InLocs[0][0], LiveInRsp);
  EXPECT_EQ(InLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(InLocs[2][0], RspPHIInBlk2);
  EXPECT_EQ(InLocs[3][0], RspPHIInBlk3);
  EXPECT_EQ(InLocs[4][0], LiveInRsp);
  EXPECT_EQ(OutLocs[0][0], LiveInRsp);
  EXPECT_EQ(OutLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(OutLocs[2][0], RspPHIInBlk2);
  EXPECT_EQ(OutLocs[3][0], LiveInRsp);
  EXPECT_EQ(OutLocs[4][0], LiveInRsp);
  // Check $rax's value. It should have $rsps value from the entry block
  // onwards.
  EXPECT_EQ(InLocs[0][1], LiveInRax);
  EXPECT_EQ(InLocs[1][1], LiveInRsp);
  EXPECT_EQ(InLocs[2][1], LiveInRsp);
  EXPECT_EQ(InLocs[3][1], LiveInRsp);
  EXPECT_EQ(InLocs[4][1], LiveInRsp);
  EXPECT_EQ(OutLocs[0][1], LiveInRsp);
  EXPECT_EQ(OutLocs[1][1], LiveInRsp);
  EXPECT_EQ(OutLocs[2][1], LiveInRsp);
  EXPECT_EQ(OutLocs[3][1], LiveInRsp);
  EXPECT_EQ(OutLocs[4][1], LiveInRsp);
}

TEST_F(InstrRefLDVTest, pickVPHILocDiamond) {
  //        entry
  //        /  \
  //      br1  br2
  //        \  /
  //         ret
  setupDiamondBlocks();

  ASSERT_TRUE(MTracker->getNumLocs() == 1);
  LocIdx RspLoc(0);
  Register RAX = getRegByName("RAX");
  LocIdx RaxLoc = MTracker->lookupOrTrackRegister(RAX);

  ValueIDNum OutLocs[4][2];
  ValueIDNum *OutLocsPtr[4] = {OutLocs[0], OutLocs[1], OutLocs[2], OutLocs[3]};

  initValueArray(OutLocsPtr, 4, 2);

  unsigned EntryBlk = 0, Br2Blk = 2, RetBlk = 3;

  ValueIDNum LiveInRsp(EntryBlk, 0, RspLoc);
  ValueIDNum LiveInRax(EntryBlk, 0, RaxLoc);
  ValueIDNum RspPHIInBlk2(Br2Blk, 0, RspLoc);
  ValueIDNum RspPHIInBlk3(RetBlk, 0, RspLoc);

  DebugVariable Var(FuncVariable, None, nullptr);
  DbgValueProperties EmptyProps(EmptyExpr, false);
  SmallVector<DenseMap<DebugVariable, DbgValue>, 32> VLiveOuts;
  VLiveOuts.resize(4);
  InstrRefBasedLDV::LiveIdxT VLiveOutIdx;
  VLiveOutIdx[MBB0] = &VLiveOuts[0];
  VLiveOutIdx[MBB1] = &VLiveOuts[1];
  VLiveOutIdx[MBB2] = &VLiveOuts[2];
  VLiveOutIdx[MBB3] = &VLiveOuts[3];

  SmallVector<const MachineBasicBlock *, 2> Preds;
  for (const auto *Pred : MBB3->predecessors())
    Preds.push_back(Pred);

  // Specify the live-outs around the joining block.
  OutLocs[1][0] = LiveInRsp;
  OutLocs[2][0] = LiveInRax;

  Optional<ValueIDNum> Result;

  // Simple case: join two distinct values on entry to the block.
  VLiveOuts[1].insert({Var,  DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLiveOuts[2].insert({Var,  DbgValue(LiveInRax, EmptyProps, DbgValue::Def)});
  Result = pickVPHILoc(*MBB3, Var, VLiveOutIdx, OutLocsPtr, Preds);
  // Should have picked a PHI in $rsp in block 3.
  EXPECT_TRUE(Result);
  if (Result) {
    EXPECT_EQ(*Result, RspPHIInBlk3);
  }

  // If the incoming values are swapped between blocks, we should not
  // successfully join. The CFG merge would select the right values, but in
  // the wrong conditions.
  std::swap(VLiveOuts[1], VLiveOuts[2]);
  Result = pickVPHILoc(*MBB3, Var, VLiveOutIdx, OutLocsPtr, Preds);
  EXPECT_FALSE(Result);

  // Swap back,
  std::swap(VLiveOuts[1], VLiveOuts[2]);
  // Setting one of these to being a constant should prohibit merging.
  VLiveOuts[1].find(Var)->second.Kind = DbgValue::Const;
  VLiveOuts[1].find(Var)->second.MO = MachineOperand::CreateImm(0);
  Result = pickVPHILoc(*MBB3, Var, VLiveOutIdx, OutLocsPtr, Preds);
  EXPECT_FALSE(Result);

  // Seeing both to being a constant -> still prohibit, it shouldn't become
  // a value in the register file anywhere.
  VLiveOuts[2].find(Var)->second = VLiveOuts[1].find(Var)->second;
  Result = pickVPHILoc(*MBB3, Var, VLiveOutIdx, OutLocsPtr, Preds);
  EXPECT_FALSE(Result);

  // NoVals shouldn't join with anything else.
  VLiveOuts[1].clear();
  VLiveOuts[2].clear();
  VLiveOuts[1].insert({Var,  DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLiveOuts[2].insert({Var,  DbgValue(2, EmptyProps, DbgValue::NoVal)});
  Result = pickVPHILoc(*MBB3, Var, VLiveOutIdx, OutLocsPtr, Preds);
  EXPECT_FALSE(Result);

  // We might merge in another VPHI in such a join. Present pickVPHILoc with
  // such a scenario: first, where one incoming edge has a VPHI with no known
  // value. This represents an edge where there was a PHI value that can't be
  // found in the register file -- we can't subsequently find a PHI here.
  VLiveOuts[2].clear();
  VLiveOuts[2].insert({Var,  DbgValue(2, EmptyProps, DbgValue::VPHI)});
  EXPECT_EQ(VLiveOuts[2].find(Var)->second.ID, ValueIDNum::EmptyValue);
  Result = pickVPHILoc(*MBB3, Var, VLiveOutIdx, OutLocsPtr, Preds);
  EXPECT_FALSE(Result);

  // However, if we know the value of the incoming VPHI, we can search for its
  // location. Use a PHI machine-value for doing this, as VPHIs should always
  // have PHI values, or they should have been eliminated.
  OutLocs[2][0] = RspPHIInBlk2;
  VLiveOuts[2].find(Var)->second.ID = RspPHIInBlk2;
  Result = pickVPHILoc(*MBB3, Var, VLiveOutIdx, OutLocsPtr, Preds);
  EXPECT_TRUE(Result);
  if (Result) {
    EXPECT_EQ(*Result, RspPHIInBlk3);
  }

  // If that value isn't available from that block, don't join.
  OutLocs[2][0] = LiveInRsp;
  Result = pickVPHILoc(*MBB3, Var, VLiveOutIdx, OutLocsPtr, Preds);
  EXPECT_FALSE(Result);

  // Check that we don't pick values when the properties disagree, for example
  // different indirectness or DIExpression.
  DIExpression *NewExpr = DIExpression::prepend(EmptyExpr, DIExpression::ApplyOffset, 4);
  DbgValueProperties PropsWithExpr(NewExpr, false);
  VLiveOuts[2].clear();
  VLiveOuts[2].insert({Var,  DbgValue(LiveInRsp, PropsWithExpr, DbgValue::Def)});
  Result = pickVPHILoc(*MBB3, Var, VLiveOutIdx, OutLocsPtr, Preds);
  EXPECT_FALSE(Result);

  DbgValueProperties PropsWithIndirect(EmptyExpr, true);
  VLiveOuts[2].clear();
  VLiveOuts[2].insert({Var,  DbgValue(LiveInRsp, PropsWithIndirect, DbgValue::Def)});
  Result = pickVPHILoc(*MBB3, Var, VLiveOutIdx, OutLocsPtr, Preds);
  EXPECT_FALSE(Result);
}

TEST_F(InstrRefLDVTest, pickVPHILocLoops) {
  setupSimpleLoop();
  //    entry
  //     |
  //     |/-----\
  //    loopblk |
  //     |\-----/
  //     |
  //     ret

  ASSERT_TRUE(MTracker->getNumLocs() == 1);
  LocIdx RspLoc(0);
  Register RAX = getRegByName("RAX");
  LocIdx RaxLoc = MTracker->lookupOrTrackRegister(RAX);

  ValueIDNum OutLocs[3][2];
  ValueIDNum *OutLocsPtr[4] = {OutLocs[0], OutLocs[1], OutLocs[2]};

  initValueArray(OutLocsPtr, 3, 2);

  unsigned EntryBlk = 0, LoopBlk = 1;

  ValueIDNum LiveInRsp(EntryBlk, 0, RspLoc);
  ValueIDNum LiveInRax(EntryBlk, 0, RaxLoc);
  ValueIDNum RspPHIInBlk1(LoopBlk, 0, RspLoc);
  ValueIDNum RaxPHIInBlk1(LoopBlk, 0, RaxLoc);

  DebugVariable Var(FuncVariable, None, nullptr);
  DbgValueProperties EmptyProps(EmptyExpr, false);
  SmallVector<DenseMap<DebugVariable, DbgValue>, 32> VLiveOuts;
  VLiveOuts.resize(3);
  InstrRefBasedLDV::LiveIdxT VLiveOutIdx;
  VLiveOutIdx[MBB0] = &VLiveOuts[0];
  VLiveOutIdx[MBB1] = &VLiveOuts[1];
  VLiveOutIdx[MBB2] = &VLiveOuts[2];

  SmallVector<const MachineBasicBlock *, 2> Preds;
  for (const auto *Pred : MBB1->predecessors())
    Preds.push_back(Pred);

  // Specify the live-outs around the joining block.
  OutLocs[0][0] = LiveInRsp;
  OutLocs[1][0] = LiveInRax;

  Optional<ValueIDNum> Result;

  // See that we can merge as normal on a backedge.
  VLiveOuts[0].insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLiveOuts[1].insert({Var, DbgValue(LiveInRax, EmptyProps, DbgValue::Def)});
  Result = pickVPHILoc(*MBB1, Var, VLiveOutIdx, OutLocsPtr, Preds);
  // Should have picked a PHI in $rsp in block 1.
  EXPECT_TRUE(Result);
  if (Result) {
    EXPECT_EQ(*Result, RspPHIInBlk1);
  }

  // And that, if the desired values aren't available, we don't merge.
  OutLocs[1][0] = LiveInRsp;
  Result = pickVPHILoc(*MBB1, Var, VLiveOutIdx, OutLocsPtr, Preds);
  EXPECT_FALSE(Result);

  // Test the backedge behaviour: PHIs that feed back into themselves can
  // carry this variables value. Feed in LiveInRsp in both $rsp and $rax
  // from the entry block, but only put an appropriate backedge PHI in $rax.
  // Only the $rax location can form the correct PHI.
  OutLocs[0][0] = LiveInRsp;
  OutLocs[0][1] = LiveInRsp;
  OutLocs[1][0] = RaxPHIInBlk1;
  OutLocs[1][1] = RaxPHIInBlk1;
  VLiveOuts[0].clear();
  VLiveOuts[1].clear();
  VLiveOuts[0].insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  // Crucially, a VPHI originating in this block:
  VLiveOuts[1].insert({Var, DbgValue(1, EmptyProps, DbgValue::VPHI)});
  Result = pickVPHILoc(*MBB1, Var, VLiveOutIdx, OutLocsPtr, Preds);
  EXPECT_TRUE(Result);
  if (Result) {
    EXPECT_EQ(*Result, RaxPHIInBlk1);
  }

  // Merging should not be permitted if there's a usable PHI on the backedge,
  // but it's in the wrong place. (Overwrite $rax).
  OutLocs[1][1] = LiveInRax;
  Result = pickVPHILoc(*MBB1, Var, VLiveOutIdx, OutLocsPtr, Preds);
  EXPECT_FALSE(Result);

  // Additionally, if the VPHI coming back on the loop backedge isn't from
  // this block (block 1), we can't merge it.
  OutLocs[1][1] = RaxPHIInBlk1;
  VLiveOuts[1].clear();
  VLiveOuts[1].insert({Var, DbgValue(0, EmptyProps, DbgValue::VPHI)});
  Result = pickVPHILoc(*MBB1, Var, VLiveOutIdx, OutLocsPtr, Preds);
  EXPECT_FALSE(Result);
}

TEST_F(InstrRefLDVTest, pickVPHILocBadlyNestedLoops) {
  // Run some tests similar to pickVPHILocLoops, with more than one backedge,
  // and check that we merge correctly over many candidate locations.
  setupBadlyNestedLoops();
  //           entry
  //             |
  //           loop1 -o
  //             | ^
  //             | ^
  //           loop2 -o
  //             | ^
  //             | ^
  //           loop3 -o
  //             |
  //            ret
  ASSERT_TRUE(MTracker->getNumLocs() == 1);
  LocIdx RspLoc(0);
  Register RAX = getRegByName("RAX");
  LocIdx RaxLoc = MTracker->lookupOrTrackRegister(RAX);
  Register RBX = getRegByName("RBX");
  LocIdx RbxLoc = MTracker->lookupOrTrackRegister(RBX);

  ValueIDNum OutLocs[5][3];
  ValueIDNum *OutLocsPtr[5] = {OutLocs[0], OutLocs[1], OutLocs[2], OutLocs[3], OutLocs[4]};

  initValueArray(OutLocsPtr, 5, 3);

  unsigned EntryBlk = 0, Loop1Blk = 1;

  ValueIDNum LiveInRsp(EntryBlk, 0, RspLoc);
  ValueIDNum LiveInRax(EntryBlk, 0, RaxLoc);
  ValueIDNum LiveInRbx(EntryBlk, 0, RbxLoc);
  ValueIDNum RspPHIInBlk1(Loop1Blk, 0, RspLoc);
  ValueIDNum RaxPHIInBlk1(Loop1Blk, 0, RaxLoc);
  ValueIDNum RbxPHIInBlk1(Loop1Blk, 0, RbxLoc);

  DebugVariable Var(FuncVariable, None, nullptr);
  DbgValueProperties EmptyProps(EmptyExpr, false);
  SmallVector<DenseMap<DebugVariable, DbgValue>, 32> VLiveOuts;
  VLiveOuts.resize(5);
  InstrRefBasedLDV::LiveIdxT VLiveOutIdx;
  VLiveOutIdx[MBB0] = &VLiveOuts[0];
  VLiveOutIdx[MBB1] = &VLiveOuts[1];
  VLiveOutIdx[MBB2] = &VLiveOuts[2];
  VLiveOutIdx[MBB3] = &VLiveOuts[3];
  VLiveOutIdx[MBB4] = &VLiveOuts[4];

  // We're going to focus on block 1.
  SmallVector<const MachineBasicBlock *, 2> Preds;
  for (const auto *Pred : MBB1->predecessors())
    Preds.push_back(Pred);

  // Specify the live-outs around the joining block. Incoming edges from the
  // entry block, self, and loop2.
  OutLocs[0][0] = LiveInRsp;
  OutLocs[1][0] = LiveInRax;
  OutLocs[2][0] = LiveInRbx;

  Optional<ValueIDNum> Result;

  // See that we can merge as normal on a backedge.
  VLiveOuts[0].insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLiveOuts[1].insert({Var, DbgValue(LiveInRax, EmptyProps, DbgValue::Def)});
  VLiveOuts[2].insert({Var, DbgValue(LiveInRbx, EmptyProps, DbgValue::Def)});
  Result = pickVPHILoc(*MBB1, Var, VLiveOutIdx, OutLocsPtr, Preds);
  // Should have picked a PHI in $rsp in block 1.
  EXPECT_TRUE(Result);
  if (Result) {
    EXPECT_EQ(*Result, RspPHIInBlk1);
  }

  // Check too that permuting the live-out locations prevents merging
  OutLocs[0][0] = LiveInRax;
  OutLocs[1][0] = LiveInRbx;
  OutLocs[2][0] = LiveInRsp;
  Result = pickVPHILoc(*MBB1, Var, VLiveOutIdx, OutLocsPtr, Preds);
  EXPECT_FALSE(Result);

  OutLocs[0][0] = LiveInRsp;
  OutLocs[1][0] = LiveInRax;
  OutLocs[2][0] = LiveInRbx;

  // Feeding a PHI back on one backedge shouldn't merge (block 1 self backedge
  // wants LiveInRax).
  OutLocs[1][0] = RspPHIInBlk1;
  Result = pickVPHILoc(*MBB1, Var, VLiveOutIdx, OutLocsPtr, Preds);
  EXPECT_FALSE(Result);

  // If the variables value on that edge is a VPHI feeding into itself, that's
  // fine.
  VLiveOuts[1].clear();
  VLiveOuts[1].insert({Var, DbgValue(1, EmptyProps, DbgValue::VPHI)});
  Result = pickVPHILoc(*MBB1, Var, VLiveOutIdx, OutLocsPtr, Preds);
  EXPECT_TRUE(Result);
  if (Result) {
    EXPECT_EQ(*Result, RspPHIInBlk1);
  }

  // Likewise: the other backedge being a VPHI from block 1 should be accepted.
  OutLocs[2][0] = RspPHIInBlk1;
  VLiveOuts[2].clear();
  VLiveOuts[2].insert({Var, DbgValue(1, EmptyProps, DbgValue::VPHI)});
  Result = pickVPHILoc(*MBB1, Var, VLiveOutIdx, OutLocsPtr, Preds);
  EXPECT_TRUE(Result);
  if (Result) {
    EXPECT_EQ(*Result, RspPHIInBlk1);
  }

  // Here's where it becomes tricky: we should not merge if there are two
  // _distinct_ backedge PHIs. We can't have a PHI that happens in both rsp
  // and rax for example. We can only pick one location as the live-in.
  OutLocs[2][0] = RaxPHIInBlk1;
  Result = pickVPHILoc(*MBB1, Var, VLiveOutIdx, OutLocsPtr, Preds);
  EXPECT_FALSE(Result);

  // The above test sources correct machine-PHI-value from two places. Now
  // try with one machine-PHI-value, but placed in two different locations
  // on the backedge. Again, we can't merge a location here, there's no
  // location that works on all paths.
  OutLocs[0][0] = LiveInRsp;
  OutLocs[1][0] = RspPHIInBlk1;
  OutLocs[2][0] = LiveInRsp;
  OutLocs[2][1] = RspPHIInBlk1;
  Result = pickVPHILoc(*MBB1, Var, VLiveOutIdx, OutLocsPtr, Preds);
  EXPECT_FALSE(Result);

  // Scatter various PHI values across the available locations. Only rbx (loc 2)
  // has the right value in both backedges -- that's the loc that should be
  // picked.
  OutLocs[0][2] = LiveInRsp;
  OutLocs[1][0] = RspPHIInBlk1;
  OutLocs[1][1] = RaxPHIInBlk1;
  OutLocs[1][2] = RbxPHIInBlk1;
  OutLocs[2][0] = LiveInRsp;
  OutLocs[2][1] = RspPHIInBlk1;
  OutLocs[2][2] = RbxPHIInBlk1;
  Result = pickVPHILoc(*MBB1, Var, VLiveOutIdx, OutLocsPtr, Preds);
  EXPECT_TRUE(Result);
  if (Result) {
    EXPECT_EQ(*Result, RbxPHIInBlk1);
  }
}

TEST_F(InstrRefLDVTest, vlocJoinDiamond) {
  //        entry
  //        /  \
  //      br1  br2
  //        \  /
  //         ret
  setupDiamondBlocks();

  ASSERT_TRUE(MTracker->getNumLocs() == 1);
  LocIdx RspLoc(0);
  Register RAX = getRegByName("RAX");
  LocIdx RaxLoc = MTracker->lookupOrTrackRegister(RAX);

  unsigned EntryBlk = 0, Br2Blk = 2, RetBlk = 3;

  ValueIDNum LiveInRsp(EntryBlk, 0, RspLoc);
  ValueIDNum LiveInRax(EntryBlk, 0, RaxLoc);
  ValueIDNum RspPHIInBlkBr2Blk(Br2Blk, 0, RspLoc);
  ValueIDNum RspPHIInBlkRetBlk(RetBlk, 0, RspLoc);

  DebugVariable Var(FuncVariable, None, nullptr);
  DbgValueProperties EmptyProps(EmptyExpr, false);
  SmallVector<DenseMap<DebugVariable, DbgValue>, 32> VLiveOuts, VLiveIns;
  VLiveOuts.resize(4);
  VLiveIns.resize(4);
  InstrRefBasedLDV::LiveIdxT VLiveOutIdx, VLiveInIdx;
  VLiveOutIdx[MBB0] = &VLiveOuts[0];
  VLiveOutIdx[MBB1] = &VLiveOuts[1];
  VLiveOutIdx[MBB2] = &VLiveOuts[2];
  VLiveOutIdx[MBB3] = &VLiveOuts[3];
  VLiveInIdx[MBB0] = &VLiveIns[0];
  VLiveInIdx[MBB1] = &VLiveIns[1];
  VLiveInIdx[MBB2] = &VLiveIns[2];
  VLiveInIdx[MBB3] = &VLiveIns[3];

  SmallPtrSet<const MachineBasicBlock *, 8> AllBlocks;
  AllBlocks.insert(MBB0);
  AllBlocks.insert(MBB1);
  AllBlocks.insert(MBB2);
  AllBlocks.insert(MBB3);

  SmallVector<const MachineBasicBlock *, 2> Preds;
  for (const auto *Pred : MBB3->predecessors())
    Preds.push_back(Pred);

  SmallSet<DebugVariable, 4> AllVars;
  AllVars.insert(Var);

  DenseMap<DebugVariable, DbgValue> JoinedLocs;

  // vlocJoin is here to propagate incoming values, and eliminate PHIs. Start
  // off by propagating a value into the merging block, number 3.
  JoinedLocs.insert({Var, DbgValue(3, EmptyProps, DbgValue::NoVal)});
  VLiveOuts[1].insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLiveOuts[2].insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  bool Result =
      vlocJoin(*MBB3, VLiveOutIdx, AllVars, AllBlocks, AllBlocks, JoinedLocs);
  EXPECT_TRUE(Result); // Output locs should have changed.
  auto It = JoinedLocs.find(Var);
  EXPECT_EQ(It->second.Kind, DbgValue::Def);
  EXPECT_EQ(It->second.ID, LiveInRsp);
  // JoinedLocs.clear(); <--- leave commented out for next test,

  // And if we did it a second time, leaving the live-ins as it was, then
  // we should report no change.
  Result =
      vlocJoin(*MBB3, VLiveOutIdx, AllVars, AllBlocks, AllBlocks, JoinedLocs);
  EXPECT_FALSE(Result);
  JoinedLocs.clear();

  // If the live-in variable values are different, but there's no PHI placed
  // in this block, then just pick a location. It should be the first (in RPO)
  // predecessor to avoid being a backedge.
  VLiveOuts[2].clear();
  VLiveOuts[2].insert({Var, DbgValue(LiveInRax, EmptyProps, DbgValue::Def)});
  JoinedLocs.insert({Var, DbgValue(3, EmptyProps, DbgValue::NoVal)});
  Result =
      vlocJoin(*MBB3, VLiveOutIdx, AllVars, AllBlocks, AllBlocks, JoinedLocs);
  EXPECT_TRUE(Result);
  It = JoinedLocs.find(Var);
  EXPECT_EQ(It->second.Kind, DbgValue::Def);
  // RPO is blocks 0 2 1 3, so LiveInRax is picked as the first predecessor
  // of this join.
  EXPECT_EQ(It->second.ID, LiveInRax);
  JoinedLocs.clear();

  // No tests for whether vlocJoin will pass-through a variable with differing
  // expressions / properties. Those can only come about due to assignments; and
  // for any assignment at all, a PHI should have been placed at the dominance
  // frontier. We rely on the IDF calculator being accurate (which is OK,
  // because so does the rest of LLVM).

  // Try placing a PHI. With differing input values (LiveInRsp, LiveInRax),
  // this PHI should not be eliminated.
  JoinedLocs.insert({Var, DbgValue(3, EmptyProps, DbgValue::VPHI)});
  Result =
      vlocJoin(*MBB3, VLiveOutIdx, AllVars, AllBlocks, AllBlocks, JoinedLocs);
  // Expect no change.
  EXPECT_FALSE(Result);
  It = JoinedLocs.find(Var);
  EXPECT_EQ(It->second.Kind, DbgValue::VPHI);
  // This should not have been assigned a fixed value.
  EXPECT_EQ(It->second.ID, ValueIDNum::EmptyValue);
  EXPECT_EQ(It->second.BlockNo, 3);
  JoinedLocs.clear();

  // Try a simple PHI elimination. Put a PHI in block 3, but LiveInRsp on both
  // incoming edges. Re-load in and out-locs with unrelated values; they're
  // irrelevant.
  VLiveOuts[1].clear();
  VLiveOuts[2].clear();
  VLiveOuts[1].insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLiveOuts[2].insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  JoinedLocs.insert({Var, DbgValue(3, EmptyProps, DbgValue::VPHI)});
  Result =
      vlocJoin(*MBB3, VLiveOutIdx, AllVars, AllBlocks, AllBlocks, JoinedLocs);
  EXPECT_TRUE(Result);
  It = JoinedLocs.find(Var);
  EXPECT_EQ(It->second.Kind, DbgValue::Def);
  EXPECT_EQ(It->second.ID, LiveInRsp);
  JoinedLocs.clear();

  // If the "current" live-in is a VPHI, but not a VPHI generated in the current
  // block, then it's the remains of an earlier value propagation. We should
  // value propagate through this merge. Even if the current incoming values
  // disagree, because we've previously determined any VPHI here is redundant.
  VLiveOuts[1].clear();
  VLiveOuts[2].clear();
  VLiveOuts[1].insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLiveOuts[2].insert({Var, DbgValue(LiveInRax, EmptyProps, DbgValue::Def)});
  JoinedLocs.insert({Var, DbgValue(2, EmptyProps, DbgValue::VPHI)});
  Result =
      vlocJoin(*MBB3, VLiveOutIdx, AllVars, AllBlocks, AllBlocks, JoinedLocs);
  EXPECT_TRUE(Result);
  It = JoinedLocs.find(Var);
  EXPECT_EQ(It->second.Kind, DbgValue::Def);
  EXPECT_EQ(It->second.ID, LiveInRax); // from block 2
  JoinedLocs.clear();

  // The above test, but test that we will install one value-propagated VPHI
  // over another.
  VLiveOuts[1].clear();
  VLiveOuts[2].clear();
  VLiveOuts[1].insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLiveOuts[2].insert({Var, DbgValue(0, EmptyProps, DbgValue::VPHI)});
  JoinedLocs.insert({Var, DbgValue(2, EmptyProps, DbgValue::VPHI)});
  Result =
      vlocJoin(*MBB3, VLiveOutIdx, AllVars, AllBlocks, AllBlocks, JoinedLocs);
  EXPECT_TRUE(Result);
  It = JoinedLocs.find(Var);
  EXPECT_EQ(It->second.Kind, DbgValue::VPHI);
  EXPECT_EQ(It->second.BlockNo, 0);
  JoinedLocs.clear();

  // We shouldn't eliminate PHIs when properties disagree.
  DbgValueProperties PropsWithIndirect(EmptyExpr, true);
  VLiveOuts[1].clear();
  VLiveOuts[2].clear();
  VLiveOuts[1].insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLiveOuts[2].insert(
      {Var, DbgValue(LiveInRsp, PropsWithIndirect, DbgValue::Def)});
  JoinedLocs.insert({Var, DbgValue(3, EmptyProps, DbgValue::VPHI)});
  Result =
      vlocJoin(*MBB3, VLiveOutIdx, AllVars, AllBlocks, AllBlocks, JoinedLocs);
  EXPECT_FALSE(Result);
  It = JoinedLocs.find(Var);
  EXPECT_EQ(It->second.Kind, DbgValue::VPHI);
  EXPECT_EQ(It->second.BlockNo, 3);
  JoinedLocs.clear();

  // Even if properties disagree, we should still value-propagate if there's no
  // PHI to be eliminated. The disagreeing values should work themselves out,
  // seeing how we've determined no PHI is necessary.
  VLiveOuts[1].clear();
  VLiveOuts[2].clear();
  VLiveOuts[1].insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLiveOuts[2].insert(
      {Var, DbgValue(LiveInRsp, PropsWithIndirect, DbgValue::Def)});
  JoinedLocs.insert({Var, DbgValue(2, EmptyProps, DbgValue::VPHI)});
  Result =
      vlocJoin(*MBB3, VLiveOutIdx, AllVars, AllBlocks, AllBlocks, JoinedLocs);
  EXPECT_TRUE(Result);
  It = JoinedLocs.find(Var);
  EXPECT_EQ(It->second.Kind, DbgValue::Def);
  EXPECT_EQ(It->second.ID, LiveInRsp);
  // Also check properties come from block 2, the first RPO predecessor to block
  // three.
  EXPECT_EQ(It->second.Properties, PropsWithIndirect);
  JoinedLocs.clear();
  VLiveIns[3].clear();

  // Again, disagreeing properties, this time the expr, should cause a PHI to
  // not be eliminated.
  DIExpression *NewExpr = DIExpression::prepend(EmptyExpr, DIExpression::ApplyOffset, 4);
  DbgValueProperties PropsWithExpr(NewExpr, false);
  VLiveOuts[1].clear();
  VLiveOuts[2].clear();
  VLiveOuts[1].insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLiveOuts[2].insert({Var, DbgValue(LiveInRsp, PropsWithExpr, DbgValue::Def)});
  JoinedLocs.insert({Var, DbgValue(3, EmptyProps, DbgValue::VPHI)});
  Result =
      vlocJoin(*MBB3, VLiveOutIdx, AllVars, AllBlocks, AllBlocks, JoinedLocs);
  EXPECT_FALSE(Result);
}

TEST_F(InstrRefLDVTest, vlocJoinLoops) {
  setupSimpleLoop();
  //    entry
  //     |
  //     |/-----\
  //    loopblk |
  //     |\-----/
  //     |
  //     ret
  ASSERT_TRUE(MTracker->getNumLocs() == 1);
  LocIdx RspLoc(0);
  Register RAX = getRegByName("RAX");
  LocIdx RaxLoc = MTracker->lookupOrTrackRegister(RAX);

  unsigned EntryBlk = 0, LoopBlk = 1;

  ValueIDNum LiveInRsp(EntryBlk, 0, RspLoc);
  ValueIDNum LiveInRax(EntryBlk, 0, RaxLoc);
  ValueIDNum RspPHIInBlk1(LoopBlk, 0, RspLoc);

  DebugVariable Var(FuncVariable, None, nullptr);
  DbgValueProperties EmptyProps(EmptyExpr, false);
  SmallVector<DenseMap<DebugVariable, DbgValue>, 32> VLiveOuts;
  VLiveOuts.resize(3);
  InstrRefBasedLDV::LiveIdxT VLiveOutIdx;
  VLiveOutIdx[MBB0] = &VLiveOuts[0];
  VLiveOutIdx[MBB1] = &VLiveOuts[1];
  VLiveOutIdx[MBB2] = &VLiveOuts[2];

  SmallPtrSet<const MachineBasicBlock *, 8> AllBlocks;
  AllBlocks.insert(MBB0);
  AllBlocks.insert(MBB1);
  AllBlocks.insert(MBB2);

  SmallVector<const MachineBasicBlock *, 2> Preds;
  for (const auto *Pred : MBB1->predecessors())
    Preds.push_back(Pred);

  SmallSet<DebugVariable, 4> AllVars;
  AllVars.insert(Var);

  DenseMap<DebugVariable, DbgValue> JoinedLocs;

  // Test some back-edge-specific behaviours of vloc join. Mostly: the fact that
  // VPHIs that arrive on backedges can be eliminated, despite having different
  // values to the predecessor.

  // First: when there's no VPHI placed already, propagate the live-in value of
  // the first RPO predecessor.
  VLiveOuts[0].insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLiveOuts[1].insert({Var, DbgValue(LiveInRax, EmptyProps, DbgValue::Def)});
  JoinedLocs.insert({Var, DbgValue(LiveInRax, EmptyProps, DbgValue::Def)});
  bool Result =
      vlocJoin(*MBB1, VLiveOutIdx, AllVars, AllBlocks, AllBlocks, JoinedLocs);
  EXPECT_TRUE(Result);
  auto It = JoinedLocs.find(Var);
  EXPECT_EQ(It->second.Kind, DbgValue::Def);
  EXPECT_EQ(It->second.ID, LiveInRsp);
  JoinedLocs.clear();

  // If there is a VPHI: don't elimiante it if there are disagreeing values.
  VLiveOuts[0].insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLiveOuts[1].insert({Var, DbgValue(LiveInRax, EmptyProps, DbgValue::Def)});
  JoinedLocs.insert({Var, DbgValue(1, EmptyProps, DbgValue::VPHI)});
  Result =
      vlocJoin(*MBB1, VLiveOutIdx, AllVars, AllBlocks, AllBlocks, JoinedLocs);
  EXPECT_FALSE(Result);
  It = JoinedLocs.find(Var);
  EXPECT_EQ(It->second.Kind, DbgValue::VPHI);
  EXPECT_EQ(It->second.BlockNo, 1);
  JoinedLocs.clear();

  // If we feed this VPHI back into itself though, we can eliminate it.
  VLiveOuts[0].clear();
  VLiveOuts[1].clear();
  VLiveOuts[0].insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLiveOuts[1].insert({Var, DbgValue(1, EmptyProps, DbgValue::VPHI)});
  JoinedLocs.insert({Var, DbgValue(1, EmptyProps, DbgValue::VPHI)});
  Result =
      vlocJoin(*MBB1, VLiveOutIdx, AllVars, AllBlocks, AllBlocks, JoinedLocs);
  EXPECT_TRUE(Result);
  It = JoinedLocs.find(Var);
  EXPECT_EQ(It->second.Kind, DbgValue::Def);
  EXPECT_EQ(It->second.ID, LiveInRsp);
  JoinedLocs.clear();

  // Don't eliminate backedge VPHIs if the predecessors have different
  // properties.
  DIExpression *NewExpr = DIExpression::prepend(EmptyExpr, DIExpression::ApplyOffset, 4);
  DbgValueProperties PropsWithExpr(NewExpr, false);
  VLiveOuts[1].clear();
  VLiveOuts[1].insert({Var, DbgValue(1, PropsWithExpr, DbgValue::VPHI)});
  JoinedLocs.insert({Var, DbgValue(1, EmptyProps, DbgValue::VPHI)});
  Result =
      vlocJoin(*MBB1, VLiveOutIdx, AllVars, AllBlocks, AllBlocks, JoinedLocs);
  EXPECT_FALSE(Result);
  It = JoinedLocs.find(Var);
  EXPECT_EQ(It->second.Kind, DbgValue::VPHI);
  EXPECT_EQ(It->second.BlockNo, 1);
  JoinedLocs.clear();

  // Backedges with VPHIs, but from the wrong block, shouldn't be eliminated.
  VLiveOuts[1].clear();
  VLiveOuts[1].insert({Var, DbgValue(0, EmptyProps, DbgValue::VPHI)});
  JoinedLocs.insert({Var, DbgValue(1, EmptyProps, DbgValue::VPHI)});
  Result =
      vlocJoin(*MBB1, VLiveOutIdx, AllVars, AllBlocks, AllBlocks, JoinedLocs);
  EXPECT_FALSE(Result);
  It = JoinedLocs.find(Var);
  EXPECT_EQ(It->second.Kind, DbgValue::VPHI);
  EXPECT_EQ(It->second.BlockNo, 1);
  JoinedLocs.clear();
}

TEST_F(InstrRefLDVTest, vlocJoinBadlyNestedLoops) {
  // Test PHI elimination in the presence of multiple backedges.
  setupBadlyNestedLoops();
  //           entry
  //             |
  //           loop1 -o
  //             | ^
  //             | ^
  //           loop2 -o
  //             | ^
  //             | ^
  //           loop3 -o
  //             |
  //            ret
  ASSERT_TRUE(MTracker->getNumLocs() == 1);
  LocIdx RspLoc(0);
  Register RAX = getRegByName("RAX");
  LocIdx RaxLoc = MTracker->lookupOrTrackRegister(RAX);
  Register RBX = getRegByName("RBX");
  LocIdx RbxLoc = MTracker->lookupOrTrackRegister(RBX);

  unsigned EntryBlk = 0;

  ValueIDNum LiveInRsp(EntryBlk, 0, RspLoc);
  ValueIDNum LiveInRax(EntryBlk, 0, RaxLoc);
  ValueIDNum LiveInRbx(EntryBlk, 0, RbxLoc);

  DebugVariable Var(FuncVariable, None, nullptr);
  DbgValueProperties EmptyProps(EmptyExpr, false);
  SmallVector<DenseMap<DebugVariable, DbgValue>, 32> VLiveOuts;
  VLiveOuts.resize(5);
  InstrRefBasedLDV::LiveIdxT VLiveOutIdx;
  VLiveOutIdx[MBB0] = &VLiveOuts[0];
  VLiveOutIdx[MBB1] = &VLiveOuts[1];
  VLiveOutIdx[MBB2] = &VLiveOuts[2];
  VLiveOutIdx[MBB3] = &VLiveOuts[3];
  VLiveOutIdx[MBB4] = &VLiveOuts[4];

  SmallPtrSet<const MachineBasicBlock *, 8> AllBlocks;
  AllBlocks.insert(MBB0);
  AllBlocks.insert(MBB1);
  AllBlocks.insert(MBB2);
  AllBlocks.insert(MBB3);
  AllBlocks.insert(MBB4);

  // We're going to focus on block 1.
  SmallVector<const MachineBasicBlock *, 3> Preds;
  for (const auto *Pred : MBB1->predecessors())
    Preds.push_back(Pred);

  SmallSet<DebugVariable, 4> AllVars;
  AllVars.insert(Var);

  DenseMap<DebugVariable, DbgValue> JoinedLocs;

  // Test a normal VPHI isn't eliminated.
  VLiveOuts[0].insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLiveOuts[1].insert({Var, DbgValue(LiveInRax, EmptyProps, DbgValue::Def)});
  VLiveOuts[2].insert({Var, DbgValue(LiveInRbx, EmptyProps, DbgValue::Def)});
  JoinedLocs.insert({Var, DbgValue(1, EmptyProps, DbgValue::VPHI)});
  bool Result =
      vlocJoin(*MBB1, VLiveOutIdx, AllVars, AllBlocks, AllBlocks, JoinedLocs);
  EXPECT_FALSE(Result);
  auto It = JoinedLocs.find(Var);
  EXPECT_EQ(It->second.Kind, DbgValue::VPHI);
  EXPECT_EQ(It->second.BlockNo, 1);
  JoinedLocs.clear();

  // Common VPHIs on backedges should merge.
  VLiveOuts[0].clear();
  VLiveOuts[1].clear();
  VLiveOuts[2].clear();
  VLiveOuts[0].insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLiveOuts[1].insert({Var, DbgValue(1, EmptyProps, DbgValue::VPHI)});
  VLiveOuts[2].insert({Var, DbgValue(1, EmptyProps, DbgValue::VPHI)});
  JoinedLocs.insert({Var, DbgValue(1, EmptyProps, DbgValue::VPHI)});
  Result =
      vlocJoin(*MBB1, VLiveOutIdx, AllVars, AllBlocks, AllBlocks, JoinedLocs);
  EXPECT_TRUE(Result);
  It = JoinedLocs.find(Var);
  EXPECT_EQ(It->second.Kind, DbgValue::Def);
  EXPECT_EQ(It->second.ID, LiveInRsp);
  JoinedLocs.clear();

  // They shouldn't merge if one of their properties is different.
  DbgValueProperties PropsWithIndirect(EmptyExpr, true);
  VLiveOuts[0].clear();
  VLiveOuts[1].clear();
  VLiveOuts[2].clear();
  VLiveOuts[0].insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLiveOuts[1].insert({Var, DbgValue(1, EmptyProps, DbgValue::VPHI)});
  VLiveOuts[2].insert({Var, DbgValue(1, PropsWithIndirect, DbgValue::VPHI)});
  JoinedLocs.insert({Var, DbgValue(1, EmptyProps, DbgValue::VPHI)});
  Result =
      vlocJoin(*MBB1, VLiveOutIdx, AllVars, AllBlocks, AllBlocks, JoinedLocs);
  EXPECT_FALSE(Result);
  It = JoinedLocs.find(Var);
  EXPECT_EQ(It->second.Kind, DbgValue::VPHI);
  EXPECT_EQ(It->second.BlockNo, 1);
  JoinedLocs.clear();

  // VPHIs from different blocks should not merge.
  VLiveOuts[0].clear();
  VLiveOuts[1].clear();
  VLiveOuts[2].clear();
  VLiveOuts[0].insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLiveOuts[1].insert({Var, DbgValue(1, EmptyProps, DbgValue::VPHI)});
  VLiveOuts[2].insert({Var, DbgValue(2, EmptyProps, DbgValue::VPHI)});
  JoinedLocs.insert({Var, DbgValue(1, EmptyProps, DbgValue::VPHI)});
  Result =
      vlocJoin(*MBB1, VLiveOutIdx, AllVars, AllBlocks, AllBlocks, JoinedLocs);
  EXPECT_FALSE(Result);
  It = JoinedLocs.find(Var);
  EXPECT_EQ(It->second.Kind, DbgValue::VPHI);
  EXPECT_EQ(It->second.BlockNo, 1);
  JoinedLocs.clear();
}

// Above are tests for picking VPHI locations, and eliminating VPHIs. No
// unit-tests are written for evaluating the transfer function as that's
// pretty straight forwards, or applying VPHI-location-picking to live-ins.
// Instead, pre-set some machine locations and apply buildVLocValueMap to the
// existing CFG patterns.
TEST_F(InstrRefLDVTest, VLocSingleBlock) {
  setupSingleBlock();

  ASSERT_TRUE(MTracker->getNumLocs() == 1);
  LocIdx RspLoc(0);

  ValueIDNum InLocs[2], OutLocs[2];
  ValueIDNum *InLocsPtr[1] = {&InLocs[0]};
  ValueIDNum *OutLocsPtr[1] = {&OutLocs[0]};

  ValueIDNum LiveInRsp = ValueIDNum(0, 0, RspLoc);
  InLocs[0] = OutLocs[0] = LiveInRsp;

  DebugVariable Var(FuncVariable, None, nullptr);
  DbgValueProperties EmptyProps(EmptyExpr, false);

  SmallSet<DebugVariable, 4> AllVars;
  AllVars.insert(Var);

  // Mild hack: rather than constructing machine instructions in each block
  // and creating lexical scopes across them, instead just tell
  // buildVLocValueMap that there's an assignment in every block. That makes
  // every block in scope.
  SmallPtrSet<MachineBasicBlock *, 4> AssignBlocks;
  AssignBlocks.insert(MBB0);

  SmallVector<VLocTracker, 1> VLocs;
  VLocs.resize(1);

  InstrRefBasedLDV::LiveInsT Output;

  // Test that, with no assignments at all, no mappings are created for the
  // variable in this function.
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output.size(), 0ul);

  // If we put an assignment in the transfer function, that should... well,
  // do nothing, because we don't store the live-outs.
  VLocs[0].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output.size(), 0ul);

  // There is pretty much nothing else of interest to test with a single block.
  // It's not relevant to the SSA-construction parts of variable values.
}

TEST_F(InstrRefLDVTest, VLocDiamondBlocks) {
  setupDiamondBlocks();
  //        entry
  //        /  \
  //      br1  br2
  //        \  /
  //         ret

  ASSERT_TRUE(MTracker->getNumLocs() == 1);
  LocIdx RspLoc(0);
  Register RAX = getRegByName("RAX");
  LocIdx RaxLoc = MTracker->lookupOrTrackRegister(RAX);

  unsigned EntryBlk = 0, RetBlk = 3;

  ValueIDNum LiveInRsp = ValueIDNum(EntryBlk, 0, RspLoc);
  ValueIDNum LiveInRax = ValueIDNum(EntryBlk, 0, RaxLoc);
  ValueIDNum RspPHIInBlk3 = ValueIDNum(RetBlk, 0, RspLoc);

  ValueIDNum InLocs[4][2], OutLocs[4][2];
  ValueIDNum *InLocsPtr[4] = {InLocs[0], InLocs[1], InLocs[2], InLocs[3]};
  ValueIDNum *OutLocsPtr[4] = {OutLocs[0], OutLocs[1], OutLocs[2], OutLocs[3]};

  initValueArray(InLocsPtr, 4, 2);
  initValueArray(OutLocsPtr, 4, 2);

  DebugVariable Var(FuncVariable, None, nullptr);
  DbgValueProperties EmptyProps(EmptyExpr, false);

  SmallSet<DebugVariable, 4> AllVars;
  AllVars.insert(Var);

  // Mild hack: rather than constructing machine instructions in each block
  // and creating lexical scopes across them, instead just tell
  // buildVLocValueMap that there's an assignment in every block. That makes
  // every block in scope.
  SmallPtrSet<MachineBasicBlock *, 4> AssignBlocks;
  AssignBlocks.insert(MBB0);
  AssignBlocks.insert(MBB1);
  AssignBlocks.insert(MBB2);
  AssignBlocks.insert(MBB3);

  SmallVector<VLocTracker, 1> VLocs;
  VLocs.resize(4);

  InstrRefBasedLDV::LiveInsT Output;

  // Start off with LiveInRsp in every location.
  for (unsigned int I = 0; I < 4; ++I) {
    InLocs[I][0] = InLocs[I][1] = LiveInRsp;
    OutLocs[I][0] = OutLocs[I][1] = LiveInRsp;
  }

  auto ClearOutputs = [&]() {
    for (auto &Elem : Output)
      Elem.clear();
  };
  Output.resize(4);

  // No assignments -> no values.
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 0ul);
  EXPECT_EQ(Output[2].size(), 0ul);
  EXPECT_EQ(Output[3].size(), 0ul);

  // An assignment in the end block should also not affect other blocks; or
  // produce any live-ins.
  VLocs[3].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 0ul);
  EXPECT_EQ(Output[2].size(), 0ul);
  EXPECT_EQ(Output[3].size(), 0ul);
  ClearOutputs();

  // Assignments in either of the side-of-diamond blocks should also not be
  // propagated anywhere.
  VLocs[3].Vars.clear();
  VLocs[2].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 0ul);
  EXPECT_EQ(Output[2].size(), 0ul);
  EXPECT_EQ(Output[3].size(), 0ul);
  VLocs[2].Vars.clear();
  ClearOutputs();

  VLocs[1].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 0ul);
  EXPECT_EQ(Output[2].size(), 0ul);
  EXPECT_EQ(Output[3].size(), 0ul);
  VLocs[1].Vars.clear();
  ClearOutputs();

  // However: putting an assignment in the first block should propagate variable
  // values through to all other blocks, as it dominates.
  VLocs[0].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  ASSERT_EQ(Output[3].size(), 1ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.ID, LiveInRsp);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.ID, LiveInRsp);
  EXPECT_EQ(Output[3][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[3][0].second.ID, LiveInRsp);
  ClearOutputs();
  VLocs[0].Vars.clear();

  // Additionally, even if that value isn't available in the register file, it
  // should still be propagated, as buildVLocValueMap shouldn't care about
  // what's in the registers (except for PHIs).
  // values through to all other blocks, as it dominates.
  VLocs[0].Vars.insert({Var, DbgValue(LiveInRax, EmptyProps, DbgValue::Def)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  ASSERT_EQ(Output[3].size(), 1ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.ID, LiveInRax);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.ID, LiveInRax);
  EXPECT_EQ(Output[3][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[3][0].second.ID, LiveInRax);
  ClearOutputs();
  VLocs[0].Vars.clear();

  // We should get a live-in to the merging block, if there are two assigns of
  // the same value in either side of the diamond.
  VLocs[1].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLocs[2].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 0ul);
  EXPECT_EQ(Output[2].size(), 0ul);
  ASSERT_EQ(Output[3].size(), 1ul);
  EXPECT_EQ(Output[3][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[3][0].second.ID, LiveInRsp);
  ClearOutputs();
  VLocs[1].Vars.clear();
  VLocs[2].Vars.clear();

  // If we assign a value in the entry block, then 'undef' on a branch, we
  // shouldn't have a live-in in the merge block.
  VLocs[0].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLocs[1].Vars.insert({Var, DbgValue(EmptyProps, DbgValue::Undef)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  EXPECT_EQ(Output[3].size(), 0ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.ID, LiveInRsp);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.ID, LiveInRsp);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[1].Vars.clear();

  // Having different values joining into the merge block should mean we have
  // no live-in in that block. Block ones LiveInRax value doesn't appear as a
  // live-in anywhere, it's block internal.
  VLocs[0].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLocs[1].Vars.insert({Var, DbgValue(LiveInRax, EmptyProps, DbgValue::Def)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  EXPECT_EQ(Output[3].size(), 0ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.ID, LiveInRsp);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.ID, LiveInRsp);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[1].Vars.clear();

  // But on the other hand, if there's a location in the register file where
  // those two values can be joined, do so.
  OutLocs[1][0] = LiveInRax;
  VLocs[0].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLocs[1].Vars.insert({Var, DbgValue(LiveInRax, EmptyProps, DbgValue::Def)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  ASSERT_EQ(Output[3].size(), 1ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.ID, LiveInRsp);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.ID, LiveInRsp);
  EXPECT_EQ(Output[3][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[3][0].second.ID, RspPHIInBlk3);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[1].Vars.clear();
}

TEST_F(InstrRefLDVTest, VLocSimpleLoop) {
  setupSimpleLoop();
  //    entry
  //     |
  //     |/-----\
  //    loopblk |
  //     |\-----/
  //     |
  //     ret

  ASSERT_TRUE(MTracker->getNumLocs() == 1);
  LocIdx RspLoc(0);
  Register RAX = getRegByName("RAX");
  LocIdx RaxLoc = MTracker->lookupOrTrackRegister(RAX);

  unsigned EntryBlk = 0, LoopBlk = 1;

  ValueIDNum LiveInRsp = ValueIDNum(EntryBlk, 0, RspLoc);
  ValueIDNum LiveInRax = ValueIDNum(EntryBlk, 0, RaxLoc);
  ValueIDNum RspPHIInBlk1 = ValueIDNum(LoopBlk, 0, RspLoc);
  ValueIDNum RspDefInBlk1 = ValueIDNum(LoopBlk, 1, RspLoc);
  ValueIDNum RaxPHIInBlk1 = ValueIDNum(LoopBlk, 0, RaxLoc);

  ValueIDNum InLocs[3][2], OutLocs[3][2];
  ValueIDNum *InLocsPtr[3] = {InLocs[0], InLocs[1], InLocs[2]};
  ValueIDNum *OutLocsPtr[3] = {OutLocs[0], OutLocs[1], OutLocs[2]};

  initValueArray(InLocsPtr, 3, 2);
  initValueArray(OutLocsPtr, 3, 2);

  DebugVariable Var(FuncVariable, None, nullptr);
  DbgValueProperties EmptyProps(EmptyExpr, false);

  SmallSet<DebugVariable, 4> AllVars;
  AllVars.insert(Var);

  SmallPtrSet<MachineBasicBlock *, 4> AssignBlocks;
  AssignBlocks.insert(MBB0);
  AssignBlocks.insert(MBB1);
  AssignBlocks.insert(MBB2);

  SmallVector<VLocTracker, 3> VLocs;
  VLocs.resize(3);

  InstrRefBasedLDV::LiveInsT Output;

  // Start off with LiveInRsp in every location.
  for (unsigned int I = 0; I < 3; ++I) {
    InLocs[I][0] = InLocs[I][1] = LiveInRsp;
    OutLocs[I][0] = OutLocs[I][1] = LiveInRsp;
  }

  auto ClearOutputs = [&]() {
    for (auto &Elem : Output)
      Elem.clear();
  };
  Output.resize(3);

  // Easy starter: a dominating assign should propagate to all blocks.
  VLocs[0].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.ID, LiveInRsp);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.ID, LiveInRsp);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[1].Vars.clear();

  // Put an undef assignment in the loop. Should get no live-in value.
  VLocs[0].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLocs[1].Vars.insert({Var, DbgValue(EmptyProps, DbgValue::Undef)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 0ul);
  EXPECT_EQ(Output[2].size(), 0ul);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[1].Vars.clear();

  // Assignment of the same value should naturally join.
  VLocs[0].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLocs[1].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.ID, LiveInRsp);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.ID, LiveInRsp);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[1].Vars.clear();

  // Assignment of different values shouldn't join with no machine PHI vals.
  // Will be live-in to exit block as it's dominated.
  VLocs[0].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLocs[1].Vars.insert({Var, DbgValue(LiveInRax, EmptyProps, DbgValue::Def)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 0ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.ID, LiveInRax);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[1].Vars.clear();

  // Install a completely unrelated PHI value, that we should not join on. Try
  // with unrelated assign in loop block again.
  InLocs[1][0] = RspPHIInBlk1;
  OutLocs[1][0] = RspDefInBlk1;
  VLocs[0].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLocs[1].Vars.insert({Var, DbgValue(LiveInRax, EmptyProps, DbgValue::Def)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 0ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.ID, LiveInRax);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[1].Vars.clear();

  // Now, if we assign RspDefInBlk1 in the loop block, we should be able to
  // find the appropriate PHI.
  InLocs[1][0] = RspPHIInBlk1;
  OutLocs[1][0] = RspDefInBlk1;
  VLocs[0].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLocs[1].Vars.insert({Var, DbgValue(RspDefInBlk1, EmptyProps, DbgValue::Def)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.ID, RspPHIInBlk1);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.ID, RspDefInBlk1);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[1].Vars.clear();

  // If the PHI happens in a different location, the live-in should happen
  // there.
  InLocs[1][0] = LiveInRsp;
  OutLocs[1][0] = LiveInRsp;
  InLocs[1][1] = RaxPHIInBlk1;
  OutLocs[1][1] = RspDefInBlk1;
  VLocs[0].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLocs[1].Vars.insert({Var, DbgValue(RspDefInBlk1, EmptyProps, DbgValue::Def)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.ID, RaxPHIInBlk1);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.ID, RspDefInBlk1);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[1].Vars.clear();

  // The PHI happening in both places should be handled too. Exactly where
  // isn't important, but if the location picked changes, this test will let
  // you know.
  InLocs[1][0] = RaxPHIInBlk1;
  OutLocs[1][0] = RspDefInBlk1;
  InLocs[1][1] = RaxPHIInBlk1;
  OutLocs[1][1] = RspDefInBlk1;
  VLocs[0].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLocs[1].Vars.insert({Var, DbgValue(RspDefInBlk1, EmptyProps, DbgValue::Def)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  // Today, the first register is picked.
  EXPECT_EQ(Output[1][0].second.ID, RspPHIInBlk1);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.ID, RspDefInBlk1);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[1].Vars.clear();

  // If the loop block looked a bit like this:
  //    %0 = PHI %1, %2
  //    [...]
  //    DBG_VALUE %0
  // Then with instr-ref it becomes:
  //    DBG_PHI %0
  //    [...]
  //    DBG_INSTR_REF
  // And we would be feeding a machine PHI-value back around the loop. However:
  // this does not mean we can eliminate the variable value PHI and use the
  // variable value from the entry block: they are distinct values that must be
  // joined at some location by the control flow.
  // [This test input would never occur naturally, the machine-PHI would be
  //  eliminated]
  InLocs[1][0] = RspPHIInBlk1;
  OutLocs[1][0] = RspPHIInBlk1;
  InLocs[1][1] = LiveInRax;
  OutLocs[1][1] = LiveInRax;
  VLocs[0].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLocs[1].Vars.insert({Var, DbgValue(RspPHIInBlk1, EmptyProps, DbgValue::Def)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.ID, RspPHIInBlk1);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.ID, RspPHIInBlk1);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[1].Vars.clear();

  // Test that we can eliminate PHIs. A PHI will be placed at the loop head
  // because there's a def in in.
  InLocs[1][0] = LiveInRsp;
  OutLocs[1][0] = LiveInRsp;
  VLocs[0].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLocs[1].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.ID, LiveInRsp);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.ID, LiveInRsp);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[1].Vars.clear();
}

// test phi elimination with the nested situation
TEST_F(InstrRefLDVTest, VLocNestedLoop) {
  //    entry
  //     |
  //    loop1
  //     ^\
  //     | \    /-\
  //     |  loop2  |
  //     |  /   \-/
  //     ^ /
  //     join
  //     |
  //     ret
  setupNestedLoops();

  ASSERT_TRUE(MTracker->getNumLocs() == 1);
  LocIdx RspLoc(0);
  Register RAX = getRegByName("RAX");
  LocIdx RaxLoc = MTracker->lookupOrTrackRegister(RAX);

  unsigned EntryBlk = 0, Loop1Blk = 1, Loop2Blk = 2;

  ValueIDNum LiveInRsp = ValueIDNum(EntryBlk, 0, RspLoc);
  ValueIDNum LiveInRax = ValueIDNum(EntryBlk, 0, RaxLoc);
  ValueIDNum RspPHIInBlk1 = ValueIDNum(Loop1Blk, 0, RspLoc);
  ValueIDNum RspPHIInBlk2 = ValueIDNum(Loop2Blk, 0, RspLoc);
  ValueIDNum RspDefInBlk2 = ValueIDNum(Loop2Blk, 1, RspLoc);

  ValueIDNum InLocs[5][2], OutLocs[5][2];
  ValueIDNum *InLocsPtr[5] = {InLocs[0], InLocs[1], InLocs[2], InLocs[3], InLocs[4]};
  ValueIDNum *OutLocsPtr[5] = {OutLocs[0], OutLocs[1], OutLocs[2], OutLocs[3], OutLocs[4]};

  initValueArray(InLocsPtr, 5, 2);
  initValueArray(OutLocsPtr, 5, 2);

  DebugVariable Var(FuncVariable, None, nullptr);
  DbgValueProperties EmptyProps(EmptyExpr, false);

  SmallSet<DebugVariable, 4> AllVars;
  AllVars.insert(Var);

  SmallPtrSet<MachineBasicBlock *, 5> AssignBlocks;
  AssignBlocks.insert(MBB0);
  AssignBlocks.insert(MBB1);
  AssignBlocks.insert(MBB2);
  AssignBlocks.insert(MBB3);
  AssignBlocks.insert(MBB4);

  SmallVector<VLocTracker, 5> VLocs;
  VLocs.resize(5);

  InstrRefBasedLDV::LiveInsT Output;

  // Start off with LiveInRsp in every location.
  for (unsigned int I = 0; I < 5; ++I) {
    InLocs[I][0] = InLocs[I][1] = LiveInRsp;
    OutLocs[I][0] = OutLocs[I][1] = LiveInRsp;
  }

  auto ClearOutputs = [&]() {
    for (auto &Elem : Output)
      Elem.clear();
  };
  Output.resize(5);

  // A dominating assign should propagate to all blocks.
  VLocs[0].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  ASSERT_EQ(Output[3].size(), 1ul);
  ASSERT_EQ(Output[4].size(), 1ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.ID, LiveInRsp);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.ID, LiveInRsp);
  EXPECT_EQ(Output[3][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[3][0].second.ID, LiveInRsp);
  EXPECT_EQ(Output[4][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[4][0].second.ID, LiveInRsp);
  ClearOutputs();
  VLocs[0].Vars.clear();

  // Test that an assign in the inner loop causes unresolved PHIs at the heads
  // of both loops, and no output location. Dominated blocks do get values.
  VLocs[0].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLocs[2].Vars.insert({Var, DbgValue(LiveInRax, EmptyProps, DbgValue::Def)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 0ul);
  EXPECT_EQ(Output[2].size(), 0ul);
  ASSERT_EQ(Output[3].size(), 1ul);
  ASSERT_EQ(Output[4].size(), 1ul);
  EXPECT_EQ(Output[3][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[3][0].second.ID, LiveInRax);
  EXPECT_EQ(Output[4][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[4][0].second.ID, LiveInRax);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[2].Vars.clear();

  // Same test, but with no assignment in block 0. We should still get values
  // in dominated blocks.
  VLocs[2].Vars.insert({Var, DbgValue(LiveInRax, EmptyProps, DbgValue::Def)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 0ul);
  EXPECT_EQ(Output[2].size(), 0ul);
  ASSERT_EQ(Output[3].size(), 1ul);
  ASSERT_EQ(Output[4].size(), 1ul);
  EXPECT_EQ(Output[3][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[3][0].second.ID, LiveInRax);
  EXPECT_EQ(Output[4][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[4][0].second.ID, LiveInRax);
  ClearOutputs();
  VLocs[2].Vars.clear();

  // Similarly, assignments in the outer loop gives location to dominated
  // blocks, but no PHI locations are found at the outer loop head.
  VLocs[0].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLocs[3].Vars.insert({Var, DbgValue(LiveInRax, EmptyProps, DbgValue::Def)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 0ul);
  EXPECT_EQ(Output[2].size(), 0ul);
  EXPECT_EQ(Output[3].size(), 0ul);
  ASSERT_EQ(Output[4].size(), 1ul);
  EXPECT_EQ(Output[4][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[4][0].second.ID, LiveInRax);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[3].Vars.clear();

  VLocs[0].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLocs[1].Vars.insert({Var, DbgValue(LiveInRax, EmptyProps, DbgValue::Def)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 0ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  ASSERT_EQ(Output[3].size(), 1ul);
  ASSERT_EQ(Output[4].size(), 1ul);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.ID, LiveInRax);
  EXPECT_EQ(Output[3][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[3][0].second.ID, LiveInRax);
  EXPECT_EQ(Output[4][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[4][0].second.ID, LiveInRax);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[1].Vars.clear();

  // With an assignment of the same value in the inner loop, we should work out
  // that all PHIs can be eliminated and the same value is live-through the
  // whole function.
  VLocs[0].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLocs[2].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 1ul);
  EXPECT_EQ(Output[2].size(), 1ul);
  ASSERT_EQ(Output[3].size(), 1ul);
  ASSERT_EQ(Output[4].size(), 1ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.ID, LiveInRsp);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.ID, LiveInRsp);
  EXPECT_EQ(Output[3][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[3][0].second.ID, LiveInRsp);
  EXPECT_EQ(Output[4][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[4][0].second.ID, LiveInRsp);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[2].Vars.clear();

  // If we have an assignment in the inner loop, and a PHI for it at the inner
  // loop head, we could find a live-in location for the inner loop. But because
  // the outer loop has no PHI, we can't find a variable value for outer loop
  // head, so can't have a live-in value for the inner loop head.
  InLocs[2][0] = RspPHIInBlk2;
  OutLocs[2][0] = LiveInRax;
  // NB: all other machine locations are LiveInRsp, disallowing a PHI in block
  // one. Even though RspPHIInBlk2 isn't available later in the function, we
  // should still produce a live-in value. The fact it's unavailable is a
  // different concern.
  VLocs[0].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLocs[2].Vars.insert({Var, DbgValue(LiveInRax, EmptyProps, DbgValue::Def)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 0ul);
  EXPECT_EQ(Output[2].size(), 0ul);
  ASSERT_EQ(Output[3].size(), 1ul);
  ASSERT_EQ(Output[4].size(), 1ul);
  EXPECT_EQ(Output[3][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[3][0].second.ID, LiveInRax);
  EXPECT_EQ(Output[4][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[4][0].second.ID, LiveInRax);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[2].Vars.clear();

  // Have an assignment in inner loop that can have a PHI resolved; and add a
  // machine value PHI to the outer loop head, so that we can find a location
  // all the way through the function.
  InLocs[1][0] = RspPHIInBlk1;
  OutLocs[1][0] = RspPHIInBlk1;
  InLocs[2][0] = RspPHIInBlk2;
  OutLocs[2][0] = RspDefInBlk2;
  InLocs[3][0] = RspDefInBlk2;
  OutLocs[3][0] = RspDefInBlk2;
  VLocs[0].Vars.insert({Var, DbgValue(LiveInRsp, EmptyProps, DbgValue::Def)});
  VLocs[2].Vars.insert({Var, DbgValue(RspDefInBlk2, EmptyProps, DbgValue::Def)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    OutLocsPtr, InLocsPtr, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  ASSERT_EQ(Output[3].size(), 1ul);
  ASSERT_EQ(Output[4].size(), 1ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.ID, RspPHIInBlk1);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.ID, RspPHIInBlk2);
  EXPECT_EQ(Output[3][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[3][0].second.ID, RspDefInBlk2);
  EXPECT_EQ(Output[4][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[4][0].second.ID, RspDefInBlk2);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[2].Vars.clear();
}

