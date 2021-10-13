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

  DebugLoc OutermostLoc, InBlockLoc, NotNestedBlockLoc, InlinedLoc;

  MachineBasicBlock *MBB1, *MBB2, *MBB3, *MBB4, *MBB5;

  std::unique_ptr<InstrRefBasedLDV> LDV;
  std::unique_ptr<MLocTracker> MTracker;

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
    addMTracker();
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

  // Some routines for bouncing into LDV,
  void buildMLocValueMap(ValueIDNum **MInLocs, ValueIDNum **MOutLocs,
                         SmallVectorImpl<MLocTransferMap> &MLocTransfer) {
    LDV->buildMLocValueMap(*MF, MInLocs, MOutLocs, MLocTransfer);
  }

  void initValueArray(ValueIDNum **Nums, unsigned Blks, unsigned Locs) {
    for (unsigned int I = 0; I < Blks; ++I)
      for (unsigned int J = 0; J < Locs; ++J)
        Nums[I][J] = ValueIDNum::EmptyValue;
  }
};

TEST_F(InstrRefLDVTest, MLocSingleBlock) {
  // Test some very simple properties about interpreting the transfer function.

  // Add an entry block with nothing but 'ret void' in it.
  Function &F = const_cast<llvm::Function &>(MF->getFunction());
  auto *BB1 = BasicBlock::Create(Ctx, "entry", &F);
  IRBuilder<> IRB(BB1);
  IRB.CreateRetVoid();
  MBB1 = MF->CreateMachineBasicBlock(BB1);
  MF->insert(MF->end(), MBB1);
  MF->RenumberBlocks();

  setupLDVObj();
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
  llvm::Function &F = const_cast<llvm::Function &>(MF->getFunction());
  auto *BB1 = BasicBlock::Create(Ctx, "a", &F);
  auto *BB2 = BasicBlock::Create(Ctx, "b", &F);
  auto *BB3 = BasicBlock::Create(Ctx, "c", &F);
  auto *BB4 = BasicBlock::Create(Ctx, "d", &F);
  IRBuilder<> IRB1(BB1), IRB2(BB2), IRB3(BB3), IRB4(BB4);
  IRB1.CreateBr(BB2);
  IRB2.CreateBr(BB3);
  IRB3.CreateBr(BB4);
  IRB4.CreateRetVoid();
  MBB1 = MF->CreateMachineBasicBlock(BB1);
  MF->insert(MF->end(), MBB1);
  MBB2 = MF->CreateMachineBasicBlock(BB2);
  MF->insert(MF->end(), MBB2);
  MBB3 = MF->CreateMachineBasicBlock(BB3);
  MF->insert(MF->end(), MBB3);
  MBB4 = MF->CreateMachineBasicBlock(BB4);
  MF->insert(MF->end(), MBB4);
  MBB1->addSuccessor(MBB2);
  MBB1->addSuccessor(MBB3);
  MBB2->addSuccessor(MBB4);
  MBB3->addSuccessor(MBB4);
  MF->RenumberBlocks();

  setupLDVObj();

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
  ValueIDNum LiveInRsp(0, 0, RspLoc);
  ValueIDNum RspDefInBlk0(0, 1, RspLoc);
  ValueIDNum RspDefInBlk1(1, 1, RspLoc);
  ValueIDNum RspDefInBlk2(2, 1, RspLoc);
  ValueIDNum RspPHIInBlk3(3, 0, RspLoc);
  ValueIDNum RaxLiveInBlk1(1, 0, RaxLoc);
  ValueIDNum RaxLiveInBlk2(2, 0, RaxLoc);

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
  llvm::Function &F = const_cast<llvm::Function &>(MF->getFunction());
  auto *BB1 = BasicBlock::Create(Ctx, "entry", &F);
  auto *BB2 = BasicBlock::Create(Ctx, "loop", &F);
  auto *BB3 = BasicBlock::Create(Ctx, "ret", &F);
  IRBuilder<> IRB1(BB1), IRB2(BB2), IRB3(BB3);
  IRB1.CreateBr(BB2);
  IRB2.CreateBr(BB3);
  IRB3.CreateRetVoid();
  MBB1 = MF->CreateMachineBasicBlock(BB1);
  MF->insert(MF->end(), MBB1);
  MBB2 = MF->CreateMachineBasicBlock(BB2);
  MF->insert(MF->end(), MBB2);
  MBB3 = MF->CreateMachineBasicBlock(BB3);
  MF->insert(MF->end(), MBB3);
  MBB1->addSuccessor(MBB2);
  MBB2->addSuccessor(MBB3);
  MBB2->addSuccessor(MBB2);
  MF->RenumberBlocks();

  setupLDVObj();

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
  ValueIDNum LiveInRsp(0, 0, RspLoc);
  ValueIDNum RspPHIInBlk1(1, 0, RspLoc);
  ValueIDNum RspDefInBlk1(1, 1, RspLoc);
  ValueIDNum LiveInRax(0, 0, RaxLoc);
  ValueIDNum RaxPHIInBlk1(1, 0, RaxLoc);
  ValueIDNum RaxPHIInBlk2(2, 0, RaxLoc);

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
  llvm::Function &F = const_cast<llvm::Function &>(MF->getFunction());
  auto *BB1 = BasicBlock::Create(Ctx, "entry", &F);
  auto *BB2 = BasicBlock::Create(Ctx, "loop1", &F);
  auto *BB3 = BasicBlock::Create(Ctx, "loop2", &F);
  auto *BB4 = BasicBlock::Create(Ctx, "join", &F);
  auto *BB5 = BasicBlock::Create(Ctx, "ret", &F);
  IRBuilder<> IRB1(BB1), IRB2(BB2), IRB3(BB3), IRB4(BB4), IRB5(BB5);
  IRB1.CreateBr(BB2);
  IRB2.CreateBr(BB3);
  IRB3.CreateBr(BB4);
  IRB4.CreateBr(BB5);
  IRB5.CreateRetVoid();
  MBB1 = MF->CreateMachineBasicBlock(BB1);
  MF->insert(MF->end(), MBB1);
  MBB2 = MF->CreateMachineBasicBlock(BB2);
  MF->insert(MF->end(), MBB2);
  MBB3 = MF->CreateMachineBasicBlock(BB3);
  MF->insert(MF->end(), MBB3);
  MBB4 = MF->CreateMachineBasicBlock(BB4);
  MF->insert(MF->end(), MBB4);
  MBB5 = MF->CreateMachineBasicBlock(BB5);
  MF->insert(MF->end(), MBB5);
  MBB1->addSuccessor(MBB2);
  MBB2->addSuccessor(MBB3);
  MBB3->addSuccessor(MBB3);
  MBB3->addSuccessor(MBB4);
  MBB4->addSuccessor(MBB2);
  MBB4->addSuccessor(MBB5);
  MF->RenumberBlocks();

  setupLDVObj();

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

  ValueIDNum LiveInRsp(0, 0, RspLoc);
  ValueIDNum RspPHIInBlk1(1, 0, RspLoc);
  ValueIDNum RspDefInBlk1(1, 1, RspLoc);
  ValueIDNum RspPHIInBlk2(2, 0, RspLoc);
  ValueIDNum RspDefInBlk2(2, 1, RspLoc);
  ValueIDNum RspDefInBlk3(3, 1, RspLoc);
  ValueIDNum LiveInRax(0, 0, RaxLoc);
  ValueIDNum RaxPHIInBlk1(1, 0, RaxLoc);
  ValueIDNum RaxPHIInBlk2(2, 0, RaxLoc);

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
  llvm::Function &F = const_cast<llvm::Function &>(MF->getFunction());
  auto *BB1 = BasicBlock::Create(Ctx, "entry", &F);
  auto *BB2 = BasicBlock::Create(Ctx, "head1", &F);
  auto *BB3 = BasicBlock::Create(Ctx, "head2", &F);
  auto *BB4 = BasicBlock::Create(Ctx, "joinblk", &F);
  auto *BB5 = BasicBlock::Create(Ctx, "ret", &F);
  IRBuilder<> IRB1(BB1), IRB2(BB2), IRB3(BB3), IRB4(BB4), IRB5(BB5);
  IRB1.CreateBr(BB2);
  IRB2.CreateBr(BB3);
  IRB3.CreateBr(BB4);
  IRB4.CreateBr(BB5);
  IRB5.CreateRetVoid();
  MBB1 = MF->CreateMachineBasicBlock(BB1);
  MF->insert(MF->end(), MBB1);
  MBB2 = MF->CreateMachineBasicBlock(BB2);
  MF->insert(MF->end(), MBB2);
  MBB3 = MF->CreateMachineBasicBlock(BB3);
  MF->insert(MF->end(), MBB3);
  MBB4 = MF->CreateMachineBasicBlock(BB4);
  MF->insert(MF->end(), MBB4);
  MBB5 = MF->CreateMachineBasicBlock(BB5);
  MF->insert(MF->end(), MBB5);
  MBB1->addSuccessor(MBB2);
  MBB1->addSuccessor(MBB3);
  MBB2->addSuccessor(MBB4);
  MBB3->addSuccessor(MBB4);
  MBB4->addSuccessor(MBB2);
  MBB4->addSuccessor(MBB3);
  MBB4->addSuccessor(MBB5);
  MF->RenumberBlocks();

  setupLDVObj();

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

  ValueIDNum LiveInRsp(0, 0, RspLoc);
  ValueIDNum RspPHIInBlk1(1, 0, RspLoc);
  ValueIDNum RspDefInBlk1(1, 1, RspLoc);
  ValueIDNum RspPHIInBlk2(2, 0, RspLoc);
  ValueIDNum RspDefInBlk2(2, 1, RspLoc);
  ValueIDNum RspPHIInBlk3(3, 0, RspLoc);
  ValueIDNum RspDefInBlk3(3, 1, RspLoc);
  ValueIDNum RaxPHIInBlk1(1, 0, RaxLoc);
  ValueIDNum RaxPHIInBlk2(2, 0, RaxLoc);

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
  //
  // NB: the loop blocks self-loop, which is a bit too fiddly to draw on
  // accurately.
  llvm::Function &F = const_cast<llvm::Function &>(MF->getFunction());
  auto *BB1 = BasicBlock::Create(Ctx, "entry", &F);
  auto *BB2 = BasicBlock::Create(Ctx, "loop1", &F);
  auto *BB3 = BasicBlock::Create(Ctx, "loop2", &F);
  auto *BB4 = BasicBlock::Create(Ctx, "loop3", &F);
  auto *BB5 = BasicBlock::Create(Ctx, "ret", &F);
  IRBuilder<> IRB1(BB1), IRB2(BB2), IRB3(BB3), IRB4(BB4), IRB5(BB5);
  IRB1.CreateBr(BB2);
  IRB2.CreateBr(BB3);
  IRB3.CreateBr(BB4);
  IRB4.CreateBr(BB5);
  IRB5.CreateRetVoid();
  MBB1 = MF->CreateMachineBasicBlock(BB1);
  MF->insert(MF->end(), MBB1);
  MBB2 = MF->CreateMachineBasicBlock(BB2);
  MF->insert(MF->end(), MBB2);
  MBB3 = MF->CreateMachineBasicBlock(BB3);
  MF->insert(MF->end(), MBB3);
  MBB4 = MF->CreateMachineBasicBlock(BB4);
  MF->insert(MF->end(), MBB4);
  MBB5 = MF->CreateMachineBasicBlock(BB5);
  MF->insert(MF->end(), MBB5);
  MBB1->addSuccessor(MBB2);
  MBB2->addSuccessor(MBB2);
  MBB2->addSuccessor(MBB3);
  MBB3->addSuccessor(MBB2);
  MBB3->addSuccessor(MBB3);
  MBB3->addSuccessor(MBB4);
  MBB4->addSuccessor(MBB3);
  MBB4->addSuccessor(MBB4);
  MBB4->addSuccessor(MBB5);
  MF->RenumberBlocks();

  setupLDVObj();

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

  ValueIDNum LiveInRsp(0, 0, RspLoc);
  ValueIDNum RspPHIInBlk1(1, 0, RspLoc);
  ValueIDNum RspDefInBlk1(1, 1, RspLoc);
  ValueIDNum RspPHIInBlk2(2, 0, RspLoc);
  ValueIDNum RspPHIInBlk3(3, 0, RspLoc);
  ValueIDNum RspDefInBlk3(3, 1, RspLoc);
  ValueIDNum LiveInRax(0, 0, RaxLoc);
  ValueIDNum RaxPHIInBlk3(3, 0, RaxLoc);

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
