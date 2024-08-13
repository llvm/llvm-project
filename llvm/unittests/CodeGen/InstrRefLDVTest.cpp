//===------------- llvm/unittest/CodeGen/InstrRefLDVTest.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/MemoryBuffer.h"
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
  std::unique_ptr<Module> Mod;
  std::unique_ptr<TargetMachine> Machine;
  std::unique_ptr<MachineFunction> MF;
  std::unique_ptr<MachineDominatorTree> DomTree;
  std::unique_ptr<MachineModuleInfo> MMI;
  DICompileUnit *OurCU;
  DIFile *OurFile;
  DISubprogram *OurFunc;
  DILexicalBlock *OurBlock, *AnotherBlock;
  DISubprogram *ToInlineFunc;
  DILexicalBlock *ToInlineBlock;
  DILocalVariable *FuncVariable;
  DIBasicType *LongInt;
  DIExpression *EmptyExpr;
  LiveDebugValues::OverlapMap Overlaps;
  LiveDebugValues::DebugVariableMap DVMap;

  DebugLoc OutermostLoc, InBlockLoc, NotNestedBlockLoc, InlinedLoc;

  MachineBasicBlock *MBB0, *MBB1, *MBB2, *MBB3, *MBB4;

  std::unique_ptr<InstrRefBasedLDV> LDV;
  std::unique_ptr<MLocTracker> MTracker;
  std::unique_ptr<VLocTracker> VTracker;

  SmallString<256> MIRStr;

  InstrRefLDVTest() : Ctx(), Mod(std::make_unique<Module>("beehives", Ctx)) {}

  void SetUp() {
    // Boilerplate that creates a MachineFunction and associated blocks.

    Mod->setDataLayout("e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-"
                       "f80:128-n8:16:32:64-S128");
    Triple TargetTriple("x86_64--");
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget("", TargetTriple, Error);
    if (!T)
      GTEST_SKIP();

    TargetOptions Options;
    Machine = std::unique_ptr<TargetMachine>(T->createTargetMachine(
        Triple::normalize("x86_64--"), "", "", Options, std::nullopt,
        std::nullopt, CodeGenOptLevel::Aggressive));

    auto Type = FunctionType::get(Type::getVoidTy(Ctx), false);
    auto F =
        Function::Create(Type, GlobalValue::ExternalLinkage, "Test", &*Mod);

    unsigned FunctionNum = 42;
    MMI = std::make_unique<MachineModuleInfo>((LLVMTargetMachine *)&*Machine);
    const TargetSubtargetInfo &STI = *Machine->getSubtargetImpl(*F);

    MF = std::make_unique<MachineFunction>(*F, (LLVMTargetMachine &)*Machine,
                                           STI, MMI->getContext(), FunctionNum);

    // Create metadata: CU, subprogram, some blocks and an inline function
    // scope.
    DIBuilder DIB(*Mod);
    OurFile = DIB.createFile("xyzzy.c", "/cave");
    OurCU =
        DIB.createCompileUnit(dwarf::DW_LANG_C99, OurFile, "nou", false, "", 0);
    auto OurSubT =
        DIB.createSubroutineType(DIB.getOrCreateTypeArray(std::nullopt));
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

  InstrRefBasedLDV *setupLDVObj(MachineFunction *MF) {
    // Create a new LDV object, and plug some relevant object ptrs into it.
    LDV = std::make_unique<InstrRefBasedLDV>();
    const TargetSubtargetInfo &STI = MF->getSubtarget();
    LDV->TII = STI.getInstrInfo();
    LDV->TRI = STI.getRegisterInfo();
    LDV->TFI = STI.getFrameLowering();
    LDV->MFI = &MF->getFrameInfo();
    LDV->MRI = &MF->getRegInfo();

    DomTree = std::make_unique<MachineDominatorTree>(*MF);
    LDV->DomTree = &*DomTree;

    // Future work: unit tests for mtracker / vtracker / ttracker.

    // Setup things like the artifical block map, and BlockNo <=> RPO Order
    // mappings.
    LDV->initialSetup(*MF);
    LDV->LS.initialize(*MF);
    addMTracker(MF);
    return &*LDV;
  }

  void addMTracker(MachineFunction *MF) {
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
    VTracker = std::make_unique<VLocTracker>(DVMap, Overlaps, EmptyExpr);
    LDV->VTracker = &*VTracker;
  }

  DbgOpID addValueDbgOp(ValueIDNum V) {
    return LDV->DbgOpStore.insert(DbgOp(V));
  }
  DbgOpID addConstDbgOp(MachineOperand MO) {
    return LDV->DbgOpStore.insert(DbgOp(MO));
  }

  // Some routines for bouncing into LDV,
  void buildMLocValueMap(FuncValueTable &MInLocs, FuncValueTable &MOutLocs,
                         SmallVectorImpl<MLocTransferMap> &MLocTransfer) {
    LDV->buildMLocValueMap(*MF, MInLocs, MOutLocs, MLocTransfer);
  }

  void placeMLocPHIs(MachineFunction &MF,
                     SmallPtrSetImpl<MachineBasicBlock *> &AllBlocks,
                     FuncValueTable &MInLocs,
                     SmallVectorImpl<MLocTransferMap> &MLocTransfer) {
    LDV->placeMLocPHIs(MF, AllBlocks, MInLocs, MLocTransfer);
  }

  bool
  pickVPHILoc(SmallVectorImpl<DbgOpID> &OutValues, const MachineBasicBlock &MBB,
              const InstrRefBasedLDV::LiveIdxT &LiveOuts,
              FuncValueTable &MOutLocs,
              const SmallVectorImpl<const MachineBasicBlock *> &BlockOrders) {
    return LDV->pickVPHILoc(OutValues, MBB, LiveOuts, MOutLocs, BlockOrders);
  }

  bool vlocJoin(MachineBasicBlock &MBB, InstrRefBasedLDV::LiveIdxT &VLOCOutLocs,
                SmallPtrSet<const MachineBasicBlock *, 8> &BlocksToExplore,
                DbgValue &InLoc) {
    return LDV->vlocJoin(MBB, VLOCOutLocs, BlocksToExplore, InLoc);
  }

  void buildVLocValueMap(const DILocation *DILoc,
                    const SmallSet<DebugVariableID, 4> &VarsWeCareAbout,
                    SmallPtrSetImpl<MachineBasicBlock *> &AssignBlocks,
                    InstrRefBasedLDV::LiveInsT &Output, FuncValueTable &MOutLocs,
                    FuncValueTable &MInLocs,
                    SmallVectorImpl<VLocTracker> &AllTheVLocs) {
    LDV->buildVLocValueMap(DILoc, VarsWeCareAbout, AssignBlocks, Output,
                           MOutLocs, MInLocs, AllTheVLocs);
  }

  void initValueArray(FuncValueTable &Nums, unsigned Blks, unsigned Locs) {
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

    setupLDVObj(&*MF);
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

    setupLDVObj(&*MF);
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

    setupLDVObj(&*MF);
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

    setupLDVObj(&*MF);
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

    setupLDVObj(&*MF);
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

    setupLDVObj(&*MF);
  }

  MachineFunction *readMIRBlock(const char *Input) {
    MIRStr.clear();
    StringRef S = Twine(Twine(R"MIR(
--- |
  target triple = "x86_64-unknown-linux-gnu"
  define void @test() { ret void }
...
---
name: test
tracksRegLiveness: true
stack:
  - { id: 0, name: '', type: spill-slot, offset: -16, size: 8, alignment: 8,
      stack-id: default, callee-saved-register: '', callee-saved-restored: true,
      debug-info-variable: '', debug-info-expression: '', debug-info-location: '' }
body:  |
   bb.0:
    liveins: $rdi, $rsi
)MIR") + Twine(Input) + Twine("...\n"))
                      .toNullTerminatedStringRef(MIRStr);
    ;

    // Clear the "test" function from MMI if it's still present.
    if (Function *Fn = Mod->getFunction("test"))
      MMI->deleteMachineFunctionFor(*Fn);

    auto MemBuf = MemoryBuffer::getMemBuffer(S, "<input>");
    auto MIRParse = createMIRParser(std::move(MemBuf), Ctx);
    Mod = MIRParse->parseIRModule();
    assert(Mod);
    Mod->setDataLayout("e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-"
                       "f80:128-n8:16:32:64-S128");

    bool Result = MIRParse->parseMachineFunctions(*Mod, *MMI);
    assert(!Result && "Failed to parse unit test machine function?");
    (void)Result;

    Function *Fn = Mod->getFunction("test");
    assert(Fn && "Failed to parse a unit test module string?");
    Fn->setSubprogram(OurFunc);
    return MMI->getMachineFunction(*Fn);
  }

  void
  produceMLocTransferFunction(MachineFunction &MF,
                              SmallVectorImpl<MLocTransferMap> &MLocTransfer,
                              unsigned MaxNumBlocks) {
    LDV->produceMLocTransferFunction(MF, MLocTransfer, MaxNumBlocks);
  }

  std::pair<FuncValueTable, FuncValueTable>
  allocValueTables(unsigned Blocks, unsigned Locs) {
    return {FuncValueTable(Blocks, Locs), FuncValueTable(Blocks, Locs)};
  }
};

TEST_F(InstrRefLDVTest, MTransferDefs) {
  MachineFunction *MF = readMIRBlock(
   "    $rax = MOV64ri 0\n"
   "    RET64 $rax\n");
  setupLDVObj(MF);

  // We should start with only SP tracked.
  EXPECT_TRUE(MTracker->getNumLocs() == 1);

  SmallVector<MLocTransferMap, 1> TransferMap;
  TransferMap.resize(1);
  produceMLocTransferFunction(*MF, TransferMap, 1);

  // Code contains only one register write: that should assign to each of the
  // aliasing registers. Test that all of them get locations, and have a
  // corresponding def at the first instr in the function.
  const char *RegNames[] = {"RAX", "HAX", "EAX", "AX", "AH", "AL"};
  EXPECT_TRUE(MTracker->getNumLocs() == 7);
  for (const char *RegName : RegNames) {
    Register R = getRegByName(RegName);
    ASSERT_TRUE(MTracker->isRegisterTracked(R));
    LocIdx L = MTracker->getRegMLoc(R);
    ValueIDNum V = MTracker->readReg(R);
    // Value of this register should be: block zero, instruction 1, and the
    // location it's defined in is itself.
    ValueIDNum ToCmp(0, 1, L);
    EXPECT_EQ(V, ToCmp);
  }

  // Do the same again, but with an aliasing write. This should write to all
  // the same registers again, except $ah and $hax (the upper 8 bits of $ax
  // and 32 bits of $rax resp.).
  MF = readMIRBlock(
   "    $rax = MOV64ri 0\n"
   "    $al = MOV8ri 0\n"
   "    RET64 $rax\n");
  setupLDVObj(MF);
  TransferMap.clear();
  TransferMap.resize(1);
  produceMLocTransferFunction(*MF, TransferMap, 1);

  auto TestRegSetSite = [&](const char *Name, unsigned InstrNum) {
    Register R = getRegByName(Name);
    ASSERT_TRUE(MTracker->isRegisterTracked(R));
    LocIdx L = MTracker->getRegMLoc(R);
    ValueIDNum V = MTracker->readMLoc(L);
    ValueIDNum ToCmp(0, InstrNum, L);
    EXPECT_EQ(V, ToCmp);
  };

  TestRegSetSite("AL", 2);
  TestRegSetSite("AH", 1);
  TestRegSetSite("AX", 2);
  TestRegSetSite("EAX", 2);
  TestRegSetSite("HAX", 1);
  TestRegSetSite("RAX", 2);

  // This call should:
  //  * Def rax via the implicit-def,
  //  * Clobber rsi/rdi and all their subregs, via the register mask
  //  * Same for rcx, despite it not being a use in the instr, it's in the mask
  //  * NOT clobber $rsp / $esp $ sp, LiveDebugValues deliberately ignores
  //    these.
  //  * NOT clobber $rbx, because it's non-volatile
  //  * Not track every other register in the machine, only those needed.
 MF = readMIRBlock(
   "    $rax = MOV64ri 0\n" // instr 1
   "    $rbx = MOV64ri 0\n" // instr 2
   "    $rcx = MOV64ri 0\n" // instr 3
   "    $rdi = MOV64ri 0\n" // instr 4
   "    $rsi = MOV64ri 0\n" // instr 5
   "    CALL64r $rax, csr_64, implicit $rsp, implicit $ssp, implicit $rdi, implicit $rsi, implicit-def $rsp, implicit-def $ssp, implicit-def $rax, implicit-def $esp, implicit-def $sp\n\n\n\n" // instr 6
   "    RET64 $rax\n"); // instr 7
  setupLDVObj(MF);
  TransferMap.clear();
  TransferMap.resize(1);
  produceMLocTransferFunction(*MF, TransferMap, 1);

  const char *RegsSetInCall[] = {"AL",  "AH",  "AX", "EAX", "HAX", "RAX",
                                 "DIL", "DIH", "DI", "EDI", "HDI", "RDI",
                                 "SIL", "SIH", "SI", "ESI", "HSI", "RSI",
                                 "CL",  "CH",  "CX", "ECX", "HCX", "RCX"};
  for (const char *RegSetInCall : RegsSetInCall)
    TestRegSetSite(RegSetInCall, 6);

  const char *RegsLeftAlone[] = {"BL", "BH", "BX", "EBX", "HBX", "RBX"};
  for (const char *RegLeftAlone : RegsLeftAlone)
    TestRegSetSite(RegLeftAlone, 2);

  // Stack pointer should be the live-in to the function, instruction zero.
  TestRegSetSite("RSP", 0);
  // These stack regs should not be tracked either. Nor the (fake) subregs.
  EXPECT_FALSE(MTracker->isRegisterTracked(getRegByName("ESP")));
  EXPECT_FALSE(MTracker->isRegisterTracked(getRegByName("SP")));
  EXPECT_FALSE(MTracker->isRegisterTracked(getRegByName("SPL")));
  EXPECT_FALSE(MTracker->isRegisterTracked(getRegByName("SPH")));
  EXPECT_FALSE(MTracker->isRegisterTracked(getRegByName("HSP")));

  // Should only be tracking: 6 x {A, B, C, DI, SI} registers = 30,
  // Plus RSP, SSP = 32.
  EXPECT_EQ(32u, MTracker->getNumLocs());


  // When we DBG_PHI something, we should track all its subregs.
  MF = readMIRBlock(
   "    DBG_PHI $rdi, 0\n"
   "    RET64\n");
  setupLDVObj(MF);
  TransferMap.clear();
  TransferMap.resize(1);
  produceMLocTransferFunction(*MF, TransferMap, 1);

  // All DI regs and RSP tracked.
  EXPECT_EQ(7u, MTracker->getNumLocs());

  // All the DI registers should have block live-in values, i.e. the argument
  // to the function.
  const char *DIRegs[] = {"DIL", "DIH", "DI", "EDI", "HDI", "RDI"};
  for (const char *DIReg : DIRegs)
    TestRegSetSite(DIReg, 0);
}

TEST_F(InstrRefLDVTest, MTransferMeta) {
  // Meta instructions should not have any effect on register values.
  SmallVector<MLocTransferMap, 1> TransferMap;
  MachineFunction *MF = readMIRBlock(
   "    $rax = MOV64ri 0\n"
   "    $rax = IMPLICIT_DEF\n"
   "    $rax = KILL killed $rax\n"
   "    RET64 $rax\n");
  setupLDVObj(MF);
  TransferMap.clear();
  TransferMap.resize(1);
  produceMLocTransferFunction(*MF, TransferMap, 1);

  LocIdx RaxLoc = MTracker->getRegMLoc(getRegByName("RAX"));
  ValueIDNum V = MTracker->readMLoc(RaxLoc);
  // Def of rax should be from instruction 1, i.e., unmodified.
  ValueIDNum Cmp(0, 1, RaxLoc);
  EXPECT_EQ(Cmp, V);
}

TEST_F(InstrRefLDVTest, MTransferCopies) {
  SmallVector<MLocTransferMap, 1> TransferMap;
  // This memory spill should be recognised, and a spill slot created.
  MachineFunction *MF = readMIRBlock(
   "    $rax = MOV64ri 0\n"
   "    MOV64mr $rsp, 1, $noreg, 16, $noreg, $rax :: (store 8 into %stack.0)\n"
   "    RET64 $rax\n");
  setupLDVObj(MF);
  TransferMap.clear();
  TransferMap.resize(1);
  produceMLocTransferFunction(*MF, TransferMap, 1);

  // Check that the spill location contains the value defined in rax by
  // instruction 1. The MIR header says -16 offset, but it's stored as -8;
  // it's not completely clear why, but here we only care about correctly
  // identifying the slot, not that all the surrounding data is correct.
  SpillLoc L = {getRegByName("RSP"), StackOffset::getFixed(-8)};
  SpillLocationNo SpillNo = *MTracker->getOrTrackSpillLoc(L);
  unsigned SpillLocID = MTracker->getLocID(SpillNo, {64, 0});
  LocIdx SpillLoc = MTracker->getSpillMLoc(SpillLocID);
  ValueIDNum V = MTracker->readMLoc(SpillLoc);
  Register RAX = getRegByName("RAX");
  LocIdx RaxLoc = MTracker->getRegMLoc(RAX);
  ValueIDNum Cmp(0, 1, RaxLoc);
  EXPECT_EQ(V, Cmp);

  // A spill and restore should be recognised.
  MF = readMIRBlock(
   "    $rax = MOV64ri 0\n"
   "    MOV64mr $rsp, 1, $noreg, 16, $noreg, $rax :: (store 8 into %stack.0)\n"
   "    $rbx = MOV64rm $rsp, 1, $noreg, 0, $noreg :: (load 8 from %stack.0)\n"
   "    RET64\n");
  setupLDVObj(MF);
  TransferMap.clear();
  TransferMap.resize(1);
  produceMLocTransferFunction(*MF, TransferMap, 1);

  // Test that rbx contains rax from instruction 1.
  RAX = getRegByName("RAX");
  RaxLoc = MTracker->getRegMLoc(RAX);
  Register RBX = getRegByName("RBX");
  LocIdx RbxLoc = MTracker->getRegMLoc(RBX);
  Cmp = ValueIDNum(0, 1, RaxLoc);
  ValueIDNum RbxVal = MTracker->readMLoc(RbxLoc);
  EXPECT_EQ(RbxVal, Cmp);

  // Testing that all the subregisters are transferred happens in
  // MTransferSubregSpills.

  // Copies and x86 movs should be recognised and honoured. In addition, all
  // of the subregisters should be copied across too.
  MF = readMIRBlock(
   "    $rax = MOV64ri 0\n"
   "    $rcx = COPY $rax\n"
   "    $rbx = MOV64rr $rcx\n"
   "    RET64\n");
  setupLDVObj(MF);
  TransferMap.clear();
  TransferMap.resize(1);
  produceMLocTransferFunction(*MF, TransferMap, 1);

  const char *ARegs[] = {"AL", "AH", "AX", "EAX", "HAX", "RAX"};
  const char *BRegs[] = {"BL", "BH", "BX", "EBX", "HBX", "RBX"};
  const char *CRegs[] = {"CL", "CH", "CX", "ECX", "HCX", "RCX"};
  auto CheckReg = [&](unsigned int I) {
    LocIdx A = MTracker->getRegMLoc(getRegByName(ARegs[I]));
    LocIdx B = MTracker->getRegMLoc(getRegByName(BRegs[I]));
    LocIdx C = MTracker->getRegMLoc(getRegByName(CRegs[I]));
    ValueIDNum ARefVal(0, 1, A);
    ValueIDNum AVal = MTracker->readMLoc(A);
    ValueIDNum BVal = MTracker->readMLoc(B);
    ValueIDNum CVal = MTracker->readMLoc(C);
    EXPECT_EQ(ARefVal, AVal);
    EXPECT_EQ(ARefVal, BVal);
    EXPECT_EQ(ARefVal, CVal);
  };

  for (unsigned int I = 0; I < 6; ++I)
    CheckReg(I);

  // When we copy to a subregister, the super-register should be def'd too: it's
  // value will have changed.
  MF = readMIRBlock(
   "    $rax = MOV64ri 0\n"
   "    $ecx = COPY $eax\n"
   "    RET64\n");
  setupLDVObj(MF);
  TransferMap.clear();
  TransferMap.resize(1);
  produceMLocTransferFunction(*MF, TransferMap, 1);

  // First four regs [al, ah, ax, eax] should be copied to *cx.
  for (unsigned int I = 0; I < 4; ++I) {
    LocIdx A = MTracker->getRegMLoc(getRegByName(ARegs[I]));
    LocIdx C = MTracker->getRegMLoc(getRegByName(CRegs[I]));
    ValueIDNum ARefVal(0, 1, A);
    ValueIDNum AVal = MTracker->readMLoc(A);
    ValueIDNum CVal = MTracker->readMLoc(C);
    EXPECT_EQ(ARefVal, AVal);
    EXPECT_EQ(ARefVal, CVal);
  }

  // But rcx should contain a value defined by the COPY.
  LocIdx RcxLoc = MTracker->getRegMLoc(getRegByName("RCX"));
  ValueIDNum RcxVal = MTracker->readMLoc(RcxLoc);
  ValueIDNum RcxDefVal(0, 2, RcxLoc); // instr 2 -> the copy
  EXPECT_EQ(RcxVal, RcxDefVal);
}

TEST_F(InstrRefLDVTest, MTransferSubregSpills) {
  SmallVector<MLocTransferMap, 1> TransferMap;
  MachineFunction *MF = readMIRBlock(
   "    $rax = MOV64ri 0\n"
   "    MOV64mr $rsp, 1, $noreg, 16, $noreg, $rax :: (store 8 into %stack.0)\n"
   "    $rbx = MOV64rm $rsp, 1, $noreg, 0, $noreg :: (load 8 from %stack.0)\n"
   "    RET64\n");
  setupLDVObj(MF);
  TransferMap.clear();
  TransferMap.resize(1);
  produceMLocTransferFunction(*MF, TransferMap, 1);

  // Check that all the subregs of rax and rbx contain the same values. One
  // should completely transfer to the other.
  const char *ARegs[] = {"AL", "AH", "AX", "EAX", "HAX", "RAX"};
  const char *BRegs[] = {"BL", "BH", "BX", "EBX", "HBX", "RBX"};
  for (unsigned int I = 0; I < 6; ++I) {
    LocIdx A = MTracker->getRegMLoc(getRegByName(ARegs[I]));
    LocIdx B = MTracker->getRegMLoc(getRegByName(BRegs[I]));
    EXPECT_EQ(MTracker->readMLoc(A), MTracker->readMLoc(B));
  }

  // Explicitly check what's in the different subreg slots, on the stack.
  // Pair up subreg idx fields with the corresponding subregister in $rax.
  MLocTracker::StackSlotPos SubRegIdxes[] = {{8, 0}, {8, 8}, {16, 0}, {32, 0}, {64, 0}};
  const char *SubRegNames[] = {"AL", "AH", "AX", "EAX", "RAX"};
  for (unsigned int I = 0; I < 5; ++I) {
    // Value number where it's defined,
    LocIdx RegLoc = MTracker->getRegMLoc(getRegByName(SubRegNames[I]));
    ValueIDNum DefNum(0, 1, RegLoc);
    // Read the corresponding subreg field from the stack.
    SpillLoc L = {getRegByName("RSP"), StackOffset::getFixed(-8)};
    SpillLocationNo SpillNo = *MTracker->getOrTrackSpillLoc(L);
    unsigned SpillID = MTracker->getLocID(SpillNo, SubRegIdxes[I]);
    LocIdx SpillLoc = MTracker->getSpillMLoc(SpillID);
    ValueIDNum SpillValue = MTracker->readMLoc(SpillLoc);
    EXPECT_EQ(DefNum, SpillValue);
  }

  // If we have exactly the same code, but we write $eax to the stack slot after
  // $rax, then we should still have exactly the same output in the lower five
  // subregisters. Storing $eax to the start of the slot will overwrite with the
  // same values. $rax, as an aliasing register, should be reset to something
  // else by that write.
  // In theory, we could try and recognise that we're writing the _same_ values
  // to the stack again, and so $rax doesn't need to be reset to something else.
  // It seems vanishingly unlikely that LLVM would generate such code though,
  // so the benefits would be small.
  MF = readMIRBlock(
   "    $rax = MOV64ri 0\n"
   "    MOV64mr $rsp, 1, $noreg, 16, $noreg, $rax :: (store 8 into %stack.0)\n"
   "    MOV32mr $rsp, 1, $noreg, 16, $noreg, $eax :: (store 4 into %stack.0)\n"
   "    $rbx = MOV64rm $rsp, 1, $noreg, 0, $noreg :: (load 8 from %stack.0)\n"
   "    RET64\n");
  setupLDVObj(MF);
  TransferMap.clear();
  TransferMap.resize(1);
  produceMLocTransferFunction(*MF, TransferMap, 1);

  // Check lower five registers up to and include $eax == $ebx,
  for (unsigned int I = 0; I < 5; ++I) {
    LocIdx A = MTracker->getRegMLoc(getRegByName(ARegs[I]));
    LocIdx B = MTracker->getRegMLoc(getRegByName(BRegs[I]));
    EXPECT_EQ(MTracker->readMLoc(A), MTracker->readMLoc(B));
  }

  // $rbx should contain something else; today it's a def at the spill point
  // of the 4 byte value.
  SpillLoc L = {getRegByName("RSP"), StackOffset::getFixed(-8)};
  SpillLocationNo SpillNo = *MTracker->getOrTrackSpillLoc(L);
  unsigned SpillID = MTracker->getLocID(SpillNo, {64, 0});
  LocIdx Spill64Loc = MTracker->getSpillMLoc(SpillID);
  ValueIDNum DefAtSpill64(0, 3, Spill64Loc);
  LocIdx RbxLoc = MTracker->getRegMLoc(getRegByName("RBX"));
  EXPECT_EQ(MTracker->readMLoc(RbxLoc), DefAtSpill64);

  // Same again, test that the lower four subreg slots on the stack are the
  // value defined by $rax in instruction 1.
  for (unsigned int I = 0; I < 4; ++I) {
    // Value number where it's defined,
    LocIdx RegLoc = MTracker->getRegMLoc(getRegByName(SubRegNames[I]));
    ValueIDNum DefNum(0, 1, RegLoc);
    // Read the corresponding subreg field from the stack.
    SpillNo = *MTracker->getOrTrackSpillLoc(L);
    SpillID = MTracker->getLocID(SpillNo, SubRegIdxes[I]);
    LocIdx SpillLoc = MTracker->getSpillMLoc(SpillID);
    ValueIDNum SpillValue = MTracker->readMLoc(SpillLoc);
    EXPECT_EQ(DefNum, SpillValue);
  }

  // Stack slot for $rax should be a different value, today it's EmptyValue.
  ValueIDNum SpillValue = MTracker->readMLoc(Spill64Loc);
  EXPECT_EQ(SpillValue, DefAtSpill64);

  // If we write something to the stack, then over-write with some register
  // from a completely different hierarchy, none of the "old" values should be
  // readable.
  // NB: slight hack, store 16 in to a 8 byte stack slot.
  MF = readMIRBlock(
   "    $rax = MOV64ri 0\n"
   "    MOV64mr $rsp, 1, $noreg, 16, $noreg, $rax :: (store 8 into %stack.0)\n"
   "    $xmm0 = IMPLICIT_DEF\n"
   "    MOVUPDmr $rsp, 1, $noreg, 16, $noreg, killed $xmm0 :: (store (s128) into %stack.0)\n"
   "    $rbx = MOV64rm $rsp, 1, $noreg, 0, $noreg :: (load 8 from %stack.0)\n"
   "    RET64\n");
  setupLDVObj(MF);
  TransferMap.clear();
  TransferMap.resize(1);
  produceMLocTransferFunction(*MF, TransferMap, 1);

  for (unsigned int I = 0; I < 5; ++I) {
    // Read subreg fields from the stack.
    SpillLocationNo SpillNo = *MTracker->getOrTrackSpillLoc(L);
    unsigned SpillID = MTracker->getLocID(SpillNo, SubRegIdxes[I]);
    LocIdx SpillLoc = MTracker->getSpillMLoc(SpillID);
    ValueIDNum SpillValue = MTracker->readMLoc(SpillLoc);

    // Value should be defined by the spill-to-xmm0 instr, get value of a def
    // at the point of the spill.
    ValueIDNum SpillDef(0, 4, SpillLoc);
    EXPECT_EQ(SpillValue, SpillDef);
  }

  // Read xmm0's position and ensure it has a value. Should be the live-in
  // value to the block, as IMPLICIT_DEF isn't a real def.
  SpillNo = *MTracker->getOrTrackSpillLoc(L);
  SpillID = MTracker->getLocID(SpillNo, {128, 0});
  LocIdx Spill128Loc = MTracker->getSpillMLoc(SpillID);
  SpillValue = MTracker->readMLoc(Spill128Loc);
  Register XMM0 = getRegByName("XMM0");
  LocIdx Xmm0Loc = MTracker->getRegMLoc(XMM0);
  EXPECT_EQ(ValueIDNum(0, 0, Xmm0Loc), SpillValue);

  // What happens if we spill ah to the stack, then load al? It should find
  // the same value.
  MF = readMIRBlock(
   "    $rax = MOV64ri 0\n"
   "    MOV8mr $rsp, 1, $noreg, 16, $noreg, $ah :: (store 1 into %stack.0)\n"
   "    $al = MOV8rm $rsp, 1, $noreg, 0, $noreg :: (load 1 from %stack.0)\n"
   "    RET64\n");
  setupLDVObj(MF);
  TransferMap.clear();
  TransferMap.resize(1);
  produceMLocTransferFunction(*MF, TransferMap, 1);

  Register AL = getRegByName("AL");
  Register AH = getRegByName("AH");
  LocIdx AlLoc = MTracker->getRegMLoc(AL);
  LocIdx AhLoc = MTracker->getRegMLoc(AH);
  ValueIDNum AHDef(0, 1, AhLoc);
  ValueIDNum ALValue = MTracker->readMLoc(AlLoc);
  EXPECT_EQ(ALValue, AHDef);
}

TEST_F(InstrRefLDVTest, MLocSingleBlock) {
  // Test some very simple properties about interpreting the transfer function.
  setupSingleBlock();

  // We should start with a single location, the stack pointer.
  ASSERT_TRUE(MTracker->getNumLocs() == 1);
  LocIdx RspLoc(0);

  // Set up live-in and live-out tables for this function: two locations (we
  // add one later) in a single block.
  auto [MOutLocs, MInLocs] = allocValueTables(1, 2);

  // Transfer function: nothing.
  SmallVector<MLocTransferMap, 1> TransferFunc;
  TransferFunc.resize(1);

  // Try and build value maps...
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);

  // The result should be that RSP is marked as a live-in-PHI -- this represents
  // an argument. And as there's no transfer function, the block live-out should
  // be the same.
  EXPECT_EQ(MInLocs[0][0], ValueIDNum(0, 0, RspLoc));
  EXPECT_EQ(MOutLocs[0][0], ValueIDNum(0, 0, RspLoc));

  // Try again, this time initialising the in-locs to be defined by an
  // instruction. The entry block should always be re-assigned to be the
  // arguments.
  initValueArray(MInLocs, 1, 2);
  initValueArray(MOutLocs, 1, 2);
  MInLocs[0][0] = ValueIDNum(0, 1, RspLoc);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  EXPECT_EQ(MInLocs[0][0], ValueIDNum(0, 0, RspLoc));
  EXPECT_EQ(MOutLocs[0][0], ValueIDNum(0, 0, RspLoc));

  // Now insert something into the transfer function to assign to the single
  // machine location.
  TransferFunc[0].insert({RspLoc, ValueIDNum(0, 1, RspLoc)});
  initValueArray(MInLocs, 1, 2);
  initValueArray(MOutLocs, 1, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  EXPECT_EQ(MInLocs[0][0], ValueIDNum(0, 0, RspLoc));
  EXPECT_EQ(MOutLocs[0][0], ValueIDNum(0, 1, RspLoc));
  TransferFunc[0].clear();

  // Add a new register to be tracked, and insert it into the transfer function
  // as a copy. The output of $rax should be the live-in value of $rsp.
  Register RAX = getRegByName("RAX");
  LocIdx RaxLoc = MTracker->lookupOrTrackRegister(RAX);
  TransferFunc[0].insert({RspLoc, ValueIDNum(0, 1, RspLoc)});
  TransferFunc[0].insert({RaxLoc, ValueIDNum(0, 0, RspLoc)});
  initValueArray(MInLocs, 1, 2);
  initValueArray(MOutLocs, 1, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  EXPECT_EQ(MInLocs[0][0], ValueIDNum(0, 0, RspLoc));
  EXPECT_EQ(MInLocs[0][1], ValueIDNum(0, 0, RaxLoc));
  EXPECT_EQ(MOutLocs[0][0], ValueIDNum(0, 1, RspLoc));
  EXPECT_EQ(MOutLocs[0][1], ValueIDNum(0, 0, RspLoc)); // Rax contains RspLoc.
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

  auto [MInLocs, MOutLocs] = allocValueTables(4, 2);

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
  initValueArray(MInLocs, 4, 2);
  initValueArray(MOutLocs, 4, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  EXPECT_EQ(MInLocs[0][0], LiveInRsp);
  EXPECT_EQ(MInLocs[1][0], LiveInRsp);
  EXPECT_EQ(MInLocs[2][0], LiveInRsp);
  EXPECT_EQ(MInLocs[3][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[0][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[1][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[2][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[3][0], LiveInRsp);
  // (Skipped writing out locations for $rax).

  // Check that a def of $rsp in the entry block will likewise reach all the
  // successors.
  TransferFunc[0].insert({RspLoc, RspDefInBlk0});
  initValueArray(MInLocs, 4, 2);
  initValueArray(MOutLocs, 4, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  EXPECT_EQ(MInLocs[0][0], LiveInRsp);
  EXPECT_EQ(MInLocs[1][0], RspDefInBlk0);
  EXPECT_EQ(MInLocs[2][0], RspDefInBlk0);
  EXPECT_EQ(MInLocs[3][0], RspDefInBlk0);
  EXPECT_EQ(MOutLocs[0][0], RspDefInBlk0);
  EXPECT_EQ(MOutLocs[1][0], RspDefInBlk0);
  EXPECT_EQ(MOutLocs[2][0], RspDefInBlk0);
  EXPECT_EQ(MOutLocs[3][0], RspDefInBlk0);
  TransferFunc[0].clear();

  // Def in one branch of the diamond means that we need a PHI in the ret block
  TransferFunc[0].insert({RspLoc, RspDefInBlk0});
  TransferFunc[1].insert({RspLoc, RspDefInBlk1});
  initValueArray(MInLocs, 4, 2);
  initValueArray(MOutLocs, 4, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  // This value map: like above, where RspDefInBlk0 is propagated through one
  // branch of the diamond, but is def'ed in the live-outs of the other. The
  // ret / merging block should have a PHI in its live-ins.
  EXPECT_EQ(MInLocs[0][0], LiveInRsp);
  EXPECT_EQ(MInLocs[1][0], RspDefInBlk0);
  EXPECT_EQ(MInLocs[2][0], RspDefInBlk0);
  EXPECT_EQ(MInLocs[3][0], RspPHIInBlk3);
  EXPECT_EQ(MOutLocs[0][0], RspDefInBlk0);
  EXPECT_EQ(MOutLocs[1][0], RspDefInBlk1);
  EXPECT_EQ(MOutLocs[2][0], RspDefInBlk0);
  EXPECT_EQ(MOutLocs[3][0], RspPHIInBlk3);
  TransferFunc[0].clear();
  TransferFunc[1].clear();

  // If we have differeing defs in either side of the diamond, we should
  // continue to produce a PHI,
  TransferFunc[0].insert({RspLoc, RspDefInBlk0});
  TransferFunc[1].insert({RspLoc, RspDefInBlk1});
  TransferFunc[2].insert({RspLoc, RspDefInBlk2});
  initValueArray(MInLocs, 4, 2);
  initValueArray(MOutLocs, 4, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  EXPECT_EQ(MInLocs[0][0], LiveInRsp);
  EXPECT_EQ(MInLocs[1][0], RspDefInBlk0);
  EXPECT_EQ(MInLocs[2][0], RspDefInBlk0);
  EXPECT_EQ(MInLocs[3][0], RspPHIInBlk3);
  EXPECT_EQ(MOutLocs[0][0], RspDefInBlk0);
  EXPECT_EQ(MOutLocs[1][0], RspDefInBlk1);
  EXPECT_EQ(MOutLocs[2][0], RspDefInBlk2);
  EXPECT_EQ(MOutLocs[3][0], RspPHIInBlk3);
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
  initValueArray(MInLocs, 4, 2);
  initValueArray(MOutLocs, 4, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  EXPECT_EQ(MInLocs[0][0], LiveInRsp);
  EXPECT_EQ(MInLocs[1][0], RspDefInBlk0);
  EXPECT_EQ(MInLocs[2][0], RspDefInBlk0);
  EXPECT_EQ(MInLocs[3][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[0][0], RspDefInBlk0);
  EXPECT_EQ(MOutLocs[1][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[2][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[3][0], LiveInRsp);
  TransferFunc[0].clear();
  TransferFunc[1].clear();
  TransferFunc[2].clear();
}

TEST_F(InstrRefLDVTest, MLocDiamondSpills) {
  // Test that defs in stack locations that require PHIs, cause PHIs to be
  // installed in aliasing locations. i.e., if there's a PHI in the lower
  // 8 bits of the stack, there should be PHIs for 16/32/64 bit locations
  // on the stack too.
  // Technically this isn't needed for accuracy: we should calculate PHIs
  // independently for each location. However, because there's an optimisation
  // that only places PHIs for the lower "interfering" parts of stack slots,
  // test for this behaviour.
  setupDiamondBlocks();

  ASSERT_TRUE(MTracker->getNumLocs() == 1);
  LocIdx RspLoc(0);

  // Create a stack location and ensure it's tracked.
  SpillLoc SL = {getRegByName("RSP"), StackOffset::getFixed(-8)};
  SpillLocationNo SpillNo = *MTracker->getOrTrackSpillLoc(SL);
  ASSERT_EQ(MTracker->getNumLocs(), 11u); // Tracks all possible stack locs.
  // Locations are: RSP, stack slots from 2^3 bits wide up to 2^9 for zmm regs,
  // then slots for sub_8bit_hi and sub_16bit_hi ({8, 8} and {16, 16}).
  // Finally, one for spilt fp80 registers.

  // Pick out the locations on the stack that various x86 regs would be written
  // to. HAX is the upper 16 bits of EAX.
  unsigned ALID = MTracker->getLocID(SpillNo, {8, 0});
  unsigned AHID = MTracker->getLocID(SpillNo, {8, 8});
  unsigned AXID = MTracker->getLocID(SpillNo, {16, 0});
  unsigned EAXID = MTracker->getLocID(SpillNo, {32, 0});
  unsigned HAXID = MTracker->getLocID(SpillNo, {16, 16});
  unsigned RAXID = MTracker->getLocID(SpillNo, {64, 0});
  LocIdx ALStackLoc = MTracker->getSpillMLoc(ALID);
  LocIdx AHStackLoc = MTracker->getSpillMLoc(AHID);
  LocIdx AXStackLoc = MTracker->getSpillMLoc(AXID);
  LocIdx EAXStackLoc = MTracker->getSpillMLoc(EAXID);
  LocIdx HAXStackLoc = MTracker->getSpillMLoc(HAXID);
  LocIdx RAXStackLoc = MTracker->getSpillMLoc(RAXID);
  // There are other locations, for things like xmm0, which we're going to
  // ignore here.

  auto [MInLocs, MOutLocs] = allocValueTables(4, 11);

  // Transfer function: start with nothing.
  SmallVector<MLocTransferMap, 1> TransferFunc;
  TransferFunc.resize(4);

  // Name some values.
  unsigned EntryBlk = 0, Blk1 = 1, RetBlk = 3;

  ValueIDNum LiveInRsp(EntryBlk, 0, RspLoc);
  ValueIDNum ALLiveIn(EntryBlk, 0, ALStackLoc);
  ValueIDNum AHLiveIn(EntryBlk, 0, AHStackLoc);
  ValueIDNum HAXLiveIn(EntryBlk, 0, HAXStackLoc);
  ValueIDNum ALPHI(RetBlk, 0, ALStackLoc);
  ValueIDNum AXPHI(RetBlk, 0, AXStackLoc);
  ValueIDNum EAXPHI(RetBlk, 0, EAXStackLoc);
  ValueIDNum HAXPHI(RetBlk, 0, HAXStackLoc);
  ValueIDNum RAXPHI(RetBlk, 0, RAXStackLoc);

  ValueIDNum ALDefInBlk1(Blk1, 1, ALStackLoc);
  ValueIDNum HAXDefInBlk1(Blk1, 1, HAXStackLoc);

  SmallPtrSet<MachineBasicBlock *, 4> AllBlocks{MBB0, MBB1, MBB2, MBB3};

  // If we put defs into one side of the diamond, for AL and HAX, then we should
  // find all aliasing positions have PHIs placed. This isn't technically what
  // the transfer function says to do: but we're testing that the optimisation
  // to reduce IDF calculation does the right thing.
  // AH should not be def'd: it don't alias AL or HAX.
  //
  // NB: we don't call buildMLocValueMap, because it will try to eliminate the
  // upper-slot PHIs, and succeed because of our slightly cooked transfer
  // function.
  TransferFunc[1].insert({ALStackLoc, ALDefInBlk1});
  TransferFunc[1].insert({HAXStackLoc, HAXDefInBlk1});
  initValueArray(MInLocs, 4, 11);
  placeMLocPHIs(*MF, AllBlocks, MInLocs, TransferFunc);
  EXPECT_EQ(MInLocs[3][ALStackLoc.asU64()], ALPHI);
  EXPECT_EQ(MInLocs[3][AXStackLoc.asU64()], AXPHI);
  EXPECT_EQ(MInLocs[3][EAXStackLoc.asU64()], EAXPHI);
  EXPECT_EQ(MInLocs[3][HAXStackLoc.asU64()], HAXPHI);
  EXPECT_EQ(MInLocs[3][RAXStackLoc.asU64()], RAXPHI);
  // AH should be left untouched,
  EXPECT_EQ(MInLocs[3][AHStackLoc.asU64()], ValueIDNum::EmptyValue);
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

  auto [MInLocs, MOutLocs] = allocValueTables(3, 2);

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
  initValueArray(MInLocs, 3, 2);
  initValueArray(MOutLocs, 3, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  EXPECT_EQ(MInLocs[0][0], LiveInRsp);
  EXPECT_EQ(MInLocs[1][0], LiveInRsp);
  EXPECT_EQ(MInLocs[2][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[0][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[1][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[2][0], LiveInRsp);

  // Add a def of $rsp to the loop block: it should be in the live-outs, but
  // should cause a PHI to be placed in the live-ins. Test the transfer function
  // by copying that PHI into $rax in the loop, then back to $rsp in the ret
  // block.
  TransferFunc[1].insert({RspLoc, RspDefInBlk1});
  TransferFunc[1].insert({RaxLoc, RspPHIInBlk1});
  TransferFunc[2].insert({RspLoc, RaxPHIInBlk2});
  initValueArray(MInLocs, 3, 2);
  initValueArray(MOutLocs, 3, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  EXPECT_EQ(MInLocs[0][0], LiveInRsp);
  EXPECT_EQ(MInLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(MInLocs[2][0], RspDefInBlk1);
  EXPECT_EQ(MOutLocs[0][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[1][0], RspDefInBlk1);
  EXPECT_EQ(MOutLocs[2][0], RspPHIInBlk1);
  // Check rax as well,
  EXPECT_EQ(MInLocs[0][1], LiveInRax);
  EXPECT_EQ(MInLocs[1][1], RaxPHIInBlk1);
  EXPECT_EQ(MInLocs[2][1], RspPHIInBlk1);
  EXPECT_EQ(MOutLocs[0][1], LiveInRax);
  EXPECT_EQ(MOutLocs[1][1], RspPHIInBlk1);
  EXPECT_EQ(MOutLocs[2][1], RspPHIInBlk1);
  TransferFunc[1].clear();
  TransferFunc[2].clear();

  // As with the diamond case, a PHI will be created if there's a (implicit)
  // def in the entry block and loop block; but should be value propagated away
  // if it copies in the same value. Copy live-in $rsp to $rax, then copy it
  // into $rsp in the loop. Encoded as copying the live-in $rax value in block 1
  // to $rsp.
  TransferFunc[0].insert({RaxLoc, LiveInRsp});
  TransferFunc[1].insert({RspLoc, RaxPHIInBlk1});
  initValueArray(MInLocs, 3, 2);
  initValueArray(MOutLocs, 3, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  EXPECT_EQ(MInLocs[0][0], LiveInRsp);
  EXPECT_EQ(MInLocs[1][0], LiveInRsp);
  EXPECT_EQ(MInLocs[2][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[0][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[1][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[2][0], LiveInRsp);
  // Check $rax's values.
  EXPECT_EQ(MInLocs[0][1], LiveInRax);
  EXPECT_EQ(MInLocs[1][1], LiveInRsp);
  EXPECT_EQ(MInLocs[2][1], LiveInRsp);
  EXPECT_EQ(MOutLocs[0][1], LiveInRsp);
  EXPECT_EQ(MOutLocs[1][1], LiveInRsp);
  EXPECT_EQ(MOutLocs[2][1], LiveInRsp);
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

  auto [MInLocs, MOutLocs] = allocValueTables(5, 2);

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
  initValueArray(MInLocs, 5, 2);
  initValueArray(MOutLocs, 5, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  EXPECT_EQ(MInLocs[0][0], LiveInRsp);
  EXPECT_EQ(MInLocs[1][0], LiveInRsp);
  EXPECT_EQ(MInLocs[2][0], LiveInRsp);
  EXPECT_EQ(MInLocs[3][0], LiveInRsp);
  EXPECT_EQ(MInLocs[4][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[0][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[1][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[2][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[3][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[4][0], LiveInRsp);

  // A def in the inner loop means we should get PHIs at the heads of both
  // loops. Live-outs of the last three blocks will be the def, as it dominates
  // those.
  TransferFunc[2].insert({RspLoc, RspDefInBlk2});
  initValueArray(MInLocs, 5, 2);
  initValueArray(MOutLocs, 5, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  EXPECT_EQ(MInLocs[0][0], LiveInRsp);
  EXPECT_EQ(MInLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(MInLocs[2][0], RspPHIInBlk2);
  EXPECT_EQ(MInLocs[3][0], RspDefInBlk2);
  EXPECT_EQ(MInLocs[4][0], RspDefInBlk2);
  EXPECT_EQ(MOutLocs[0][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(MOutLocs[2][0], RspDefInBlk2);
  EXPECT_EQ(MOutLocs[3][0], RspDefInBlk2);
  EXPECT_EQ(MOutLocs[4][0], RspDefInBlk2);
  TransferFunc[2].clear();

  // Adding a def to the outer loop header shouldn't affect this much -- the
  // live-out of block 1 changes.
  TransferFunc[1].insert({RspLoc, RspDefInBlk1});
  TransferFunc[2].insert({RspLoc, RspDefInBlk2});
  initValueArray(MInLocs, 5, 2);
  initValueArray(MOutLocs, 5, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  EXPECT_EQ(MInLocs[0][0], LiveInRsp);
  EXPECT_EQ(MInLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(MInLocs[2][0], RspPHIInBlk2);
  EXPECT_EQ(MInLocs[3][0], RspDefInBlk2);
  EXPECT_EQ(MInLocs[4][0], RspDefInBlk2);
  EXPECT_EQ(MOutLocs[0][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[1][0], RspDefInBlk1);
  EXPECT_EQ(MOutLocs[2][0], RspDefInBlk2);
  EXPECT_EQ(MOutLocs[3][0], RspDefInBlk2);
  EXPECT_EQ(MOutLocs[4][0], RspDefInBlk2);
  TransferFunc[1].clear();
  TransferFunc[2].clear();

  // Likewise, putting a def in the outer loop tail shouldn't affect where
  // the PHIs go, and should propagate into the ret block.

  TransferFunc[1].insert({RspLoc, RspDefInBlk1});
  TransferFunc[2].insert({RspLoc, RspDefInBlk2});
  TransferFunc[3].insert({RspLoc, RspDefInBlk3});
  initValueArray(MInLocs, 5, 2);
  initValueArray(MOutLocs, 5, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  EXPECT_EQ(MInLocs[0][0], LiveInRsp);
  EXPECT_EQ(MInLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(MInLocs[2][0], RspPHIInBlk2);
  EXPECT_EQ(MInLocs[3][0], RspDefInBlk2);
  EXPECT_EQ(MInLocs[4][0], RspDefInBlk3);
  EXPECT_EQ(MOutLocs[0][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[1][0], RspDefInBlk1);
  EXPECT_EQ(MOutLocs[2][0], RspDefInBlk2);
  EXPECT_EQ(MOutLocs[3][0], RspDefInBlk3);
  EXPECT_EQ(MOutLocs[4][0], RspDefInBlk3);
  TransferFunc[1].clear();
  TransferFunc[2].clear();
  TransferFunc[3].clear();

  // However: if we don't def in the inner-loop, then we just have defs in the
  // head and tail of the outer loop. The inner loop should be live-through.
  TransferFunc[1].insert({RspLoc, RspDefInBlk1});
  TransferFunc[3].insert({RspLoc, RspDefInBlk3});
  initValueArray(MInLocs, 5, 2);
  initValueArray(MOutLocs, 5, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  EXPECT_EQ(MInLocs[0][0], LiveInRsp);
  EXPECT_EQ(MInLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(MInLocs[2][0], RspDefInBlk1);
  EXPECT_EQ(MInLocs[3][0], RspDefInBlk1);
  EXPECT_EQ(MInLocs[4][0], RspDefInBlk3);
  EXPECT_EQ(MOutLocs[0][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[1][0], RspDefInBlk1);
  EXPECT_EQ(MOutLocs[2][0], RspDefInBlk1);
  EXPECT_EQ(MOutLocs[3][0], RspDefInBlk3);
  EXPECT_EQ(MOutLocs[4][0], RspDefInBlk3);
  TransferFunc[1].clear();
  TransferFunc[3].clear();

  // Check that this still works if we copy RspDefInBlk1 to $rax and then
  // copy it back into $rsp in the inner loop.
  TransferFunc[1].insert({RspLoc, RspDefInBlk1});
  TransferFunc[1].insert({RaxLoc, RspDefInBlk1});
  TransferFunc[2].insert({RspLoc, RaxPHIInBlk2});
  TransferFunc[3].insert({RspLoc, RspDefInBlk3});
  initValueArray(MInLocs, 5, 2);
  initValueArray(MOutLocs, 5, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  EXPECT_EQ(MInLocs[0][0], LiveInRsp);
  EXPECT_EQ(MInLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(MInLocs[2][0], RspDefInBlk1);
  EXPECT_EQ(MInLocs[3][0], RspDefInBlk1);
  EXPECT_EQ(MInLocs[4][0], RspDefInBlk3);
  EXPECT_EQ(MOutLocs[0][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[1][0], RspDefInBlk1);
  EXPECT_EQ(MOutLocs[2][0], RspDefInBlk1);
  EXPECT_EQ(MOutLocs[3][0], RspDefInBlk3);
  EXPECT_EQ(MOutLocs[4][0], RspDefInBlk3);
  // Look at raxes value in the relevant blocks,
  EXPECT_EQ(MInLocs[2][1], RspDefInBlk1);
  EXPECT_EQ(MOutLocs[1][1], RspDefInBlk1);
  TransferFunc[1].clear();
  TransferFunc[2].clear();
  TransferFunc[3].clear();

  // If we have a single def in the tail of the outer loop, that should produce
  // a PHI at the loop head, and be live-through the inner loop.
  TransferFunc[3].insert({RspLoc, RspDefInBlk3});
  initValueArray(MInLocs, 5, 2);
  initValueArray(MOutLocs, 5, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  EXPECT_EQ(MInLocs[0][0], LiveInRsp);
  EXPECT_EQ(MInLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(MInLocs[2][0], RspPHIInBlk1);
  EXPECT_EQ(MInLocs[3][0], RspPHIInBlk1);
  EXPECT_EQ(MInLocs[4][0], RspDefInBlk3);
  EXPECT_EQ(MOutLocs[0][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(MOutLocs[2][0], RspPHIInBlk1);
  EXPECT_EQ(MOutLocs[3][0], RspDefInBlk3);
  EXPECT_EQ(MOutLocs[4][0], RspDefInBlk3);
  TransferFunc[3].clear();

  // And if we copy from $rsp to $rax in block 2, it should resolve to the PHI
  // in block 1, and we should keep that value in rax until the ret block.
  // There'll be a PHI in block 1 and 2, because we're putting a def in the
  // inner loop.
  TransferFunc[2].insert({RaxLoc, RspPHIInBlk2});
  TransferFunc[3].insert({RspLoc, RspDefInBlk3});
  initValueArray(MInLocs, 5, 2);
  initValueArray(MOutLocs, 5, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  // Examining the values of rax,
  EXPECT_EQ(MInLocs[0][1], LiveInRax);
  EXPECT_EQ(MInLocs[1][1], RaxPHIInBlk1);
  EXPECT_EQ(MInLocs[2][1], RaxPHIInBlk2);
  EXPECT_EQ(MInLocs[3][1], RspPHIInBlk1);
  EXPECT_EQ(MInLocs[4][1], RspPHIInBlk1);
  EXPECT_EQ(MOutLocs[0][1], LiveInRax);
  EXPECT_EQ(MOutLocs[1][1], RaxPHIInBlk1);
  EXPECT_EQ(MOutLocs[2][1], RspPHIInBlk1);
  EXPECT_EQ(MOutLocs[3][1], RspPHIInBlk1);
  EXPECT_EQ(MOutLocs[4][1], RspPHIInBlk1);
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

  auto [MInLocs, MOutLocs] = allocValueTables(5, 2);

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
  initValueArray(MInLocs, 5, 2);
  initValueArray(MOutLocs, 5, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  EXPECT_EQ(MInLocs[0][0], LiveInRsp);
  EXPECT_EQ(MInLocs[1][0], LiveInRsp);
  EXPECT_EQ(MInLocs[2][0], LiveInRsp);
  EXPECT_EQ(MInLocs[3][0], LiveInRsp);
  EXPECT_EQ(MInLocs[4][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[0][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[1][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[2][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[3][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[4][0], LiveInRsp);

  // Putting a def in the 'join' block will cause us to have two distinct
  // PHIs in each loop head, then on entry to the join block.
  TransferFunc[3].insert({RspLoc, RspDefInBlk3});
  initValueArray(MInLocs, 5, 2);
  initValueArray(MOutLocs, 5, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  EXPECT_EQ(MInLocs[0][0], LiveInRsp);
  EXPECT_EQ(MInLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(MInLocs[2][0], RspPHIInBlk2);
  EXPECT_EQ(MInLocs[3][0], RspPHIInBlk3);
  EXPECT_EQ(MInLocs[4][0], RspDefInBlk3);
  EXPECT_EQ(MOutLocs[0][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(MOutLocs[2][0], RspPHIInBlk2);
  EXPECT_EQ(MOutLocs[3][0], RspDefInBlk3);
  EXPECT_EQ(MOutLocs[4][0], RspDefInBlk3);
  TransferFunc[3].clear();

  // We should get the same behaviour if we put the def in either of the
  // loop heads -- it should force the other head to be a PHI.
  TransferFunc[1].insert({RspLoc, RspDefInBlk1});
  initValueArray(MInLocs, 5, 2);
  initValueArray(MOutLocs, 5, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  EXPECT_EQ(MInLocs[0][0], LiveInRsp);
  EXPECT_EQ(MInLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(MInLocs[2][0], RspPHIInBlk2);
  EXPECT_EQ(MInLocs[3][0], RspPHIInBlk3);
  EXPECT_EQ(MInLocs[4][0], RspPHIInBlk3);
  EXPECT_EQ(MOutLocs[0][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[1][0], RspDefInBlk1);
  EXPECT_EQ(MOutLocs[2][0], RspPHIInBlk2);
  EXPECT_EQ(MOutLocs[3][0], RspPHIInBlk3);
  EXPECT_EQ(MOutLocs[4][0], RspPHIInBlk3);
  TransferFunc[1].clear();

  // Check symmetry,
  TransferFunc[2].insert({RspLoc, RspDefInBlk2});
  initValueArray(MInLocs, 5, 2);
  initValueArray(MOutLocs, 5, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  EXPECT_EQ(MInLocs[0][0], LiveInRsp);
  EXPECT_EQ(MInLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(MInLocs[2][0], RspPHIInBlk2);
  EXPECT_EQ(MInLocs[3][0], RspPHIInBlk3);
  EXPECT_EQ(MInLocs[4][0], RspPHIInBlk3);
  EXPECT_EQ(MOutLocs[0][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(MOutLocs[2][0], RspDefInBlk2);
  EXPECT_EQ(MOutLocs[3][0], RspPHIInBlk3);
  EXPECT_EQ(MOutLocs[4][0], RspPHIInBlk3);
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
  initValueArray(MInLocs, 5, 2);
  initValueArray(MOutLocs, 5, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  EXPECT_EQ(MInLocs[0][0], LiveInRsp);
  EXPECT_EQ(MInLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(MInLocs[2][0], RspPHIInBlk2);
  EXPECT_EQ(MInLocs[3][0], LiveInRsp);
  EXPECT_EQ(MInLocs[4][0], RspDefInBlk3);
  EXPECT_EQ(MOutLocs[0][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[1][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[2][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[3][0], RspDefInBlk3);
  EXPECT_EQ(MOutLocs[4][0], RspDefInBlk3);
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
  initValueArray(MInLocs, 5, 2);
  initValueArray(MOutLocs, 5, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  EXPECT_EQ(MInLocs[0][0], LiveInRsp);
  EXPECT_EQ(MInLocs[1][0], LiveInRsp);
  EXPECT_EQ(MInLocs[2][0], LiveInRsp);
  EXPECT_EQ(MInLocs[3][0], LiveInRsp);
  EXPECT_EQ(MInLocs[4][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[0][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[1][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[2][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[3][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[4][0], LiveInRsp);
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

  auto [MInLocs, MOutLocs] = allocValueTables(5, 2);

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
  initValueArray(MInLocs, 5, 2);
  initValueArray(MOutLocs, 5, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  EXPECT_EQ(MInLocs[0][0], LiveInRsp);
  EXPECT_EQ(MInLocs[1][0], LiveInRsp);
  EXPECT_EQ(MInLocs[2][0], LiveInRsp);
  EXPECT_EQ(MInLocs[3][0], LiveInRsp);
  EXPECT_EQ(MInLocs[4][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[0][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[1][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[2][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[3][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[4][0], LiveInRsp);

  // A def in loop3 should cause PHIs in every loop block: they're all
  // reachable from each other.
  TransferFunc[3].insert({RspLoc, RspDefInBlk3});
  initValueArray(MInLocs, 5, 2);
  initValueArray(MOutLocs, 5, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  EXPECT_EQ(MInLocs[0][0], LiveInRsp);
  EXPECT_EQ(MInLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(MInLocs[2][0], RspPHIInBlk2);
  EXPECT_EQ(MInLocs[3][0], RspPHIInBlk3);
  EXPECT_EQ(MInLocs[4][0], RspDefInBlk3);
  EXPECT_EQ(MOutLocs[0][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(MOutLocs[2][0], RspPHIInBlk2);
  EXPECT_EQ(MOutLocs[3][0], RspDefInBlk3);
  EXPECT_EQ(MOutLocs[4][0], RspDefInBlk3);
  TransferFunc[3].clear();

  // A def in loop1 should cause a PHI in loop1, but not the other blocks.
  // loop2 and loop3 are dominated by the def in loop1, so they should have
  // that value live-through.
  TransferFunc[1].insert({RspLoc, RspDefInBlk1});
  initValueArray(MInLocs, 5, 2);
  initValueArray(MOutLocs, 5, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  EXPECT_EQ(MInLocs[0][0], LiveInRsp);
  EXPECT_EQ(MInLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(MInLocs[2][0], RspDefInBlk1);
  EXPECT_EQ(MInLocs[3][0], RspDefInBlk1);
  EXPECT_EQ(MInLocs[4][0], RspDefInBlk1);
  EXPECT_EQ(MOutLocs[0][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[1][0], RspDefInBlk1);
  EXPECT_EQ(MOutLocs[2][0], RspDefInBlk1);
  EXPECT_EQ(MOutLocs[3][0], RspDefInBlk1);
  EXPECT_EQ(MOutLocs[4][0], RspDefInBlk1);
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
  initValueArray(MInLocs, 5, 2);
  initValueArray(MOutLocs, 5, 2);
  buildMLocValueMap(MInLocs, MOutLocs, TransferFunc);
  EXPECT_EQ(MInLocs[0][0], LiveInRsp);
  EXPECT_EQ(MInLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(MInLocs[2][0], RspPHIInBlk2);
  EXPECT_EQ(MInLocs[3][0], RspPHIInBlk3);
  EXPECT_EQ(MInLocs[4][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[0][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[1][0], RspPHIInBlk1);
  EXPECT_EQ(MOutLocs[2][0], RspPHIInBlk2);
  EXPECT_EQ(MOutLocs[3][0], LiveInRsp);
  EXPECT_EQ(MOutLocs[4][0], LiveInRsp);
  // Check $rax's value. It should have $rsps value from the entry block
  // onwards.
  EXPECT_EQ(MInLocs[0][1], LiveInRax);
  EXPECT_EQ(MInLocs[1][1], LiveInRsp);
  EXPECT_EQ(MInLocs[2][1], LiveInRsp);
  EXPECT_EQ(MInLocs[3][1], LiveInRsp);
  EXPECT_EQ(MInLocs[4][1], LiveInRsp);
  EXPECT_EQ(MOutLocs[0][1], LiveInRsp);
  EXPECT_EQ(MOutLocs[1][1], LiveInRsp);
  EXPECT_EQ(MOutLocs[2][1], LiveInRsp);
  EXPECT_EQ(MOutLocs[3][1], LiveInRsp);
  EXPECT_EQ(MOutLocs[4][1], LiveInRsp);
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

  auto [MInLocs, MOutLocs] = allocValueTables(4, 2);

  initValueArray(MOutLocs, 4, 2);

  unsigned EntryBlk = 0, Br2Blk = 2, RetBlk = 3;

  ValueIDNum LiveInRsp(EntryBlk, 0, RspLoc);
  ValueIDNum LiveInRax(EntryBlk, 0, RaxLoc);
  ValueIDNum RspPHIInBlk2(Br2Blk, 0, RspLoc);
  ValueIDNum RspPHIInBlk3(RetBlk, 0, RspLoc);
  ValueIDNum RaxPHIInBlk3(RetBlk, 0, RaxLoc);
  DbgOpID LiveInRspID = addValueDbgOp(LiveInRsp);
  DbgOpID LiveInRaxID = addValueDbgOp(LiveInRax);
  DbgOpID RspPHIInBlk2ID = addValueDbgOp(RspPHIInBlk2);
  DbgOpID RspPHIInBlk3ID = addValueDbgOp(RspPHIInBlk3);
  DbgOpID RaxPHIInBlk3ID = addValueDbgOp(RaxPHIInBlk3);
  DbgOpID ConstZeroID = addConstDbgOp(MachineOperand::CreateImm(0));

  DebugVariable Var(FuncVariable, std::nullopt, nullptr);
  DbgValueProperties EmptyProps(EmptyExpr, false, false);
  DIExpression *TwoOpExpr =
      DIExpression::get(Ctx, {dwarf::DW_OP_LLVM_arg, 0, dwarf::DW_OP_LLVM_arg,
                              1, dwarf::DW_OP_plus});
  DbgValueProperties TwoOpProps(TwoOpExpr, false, true);
  SmallVector<DbgValue, 32> VLiveOuts;
  VLiveOuts.resize(4, DbgValue(EmptyProps, DbgValue::Undef));
  InstrRefBasedLDV::LiveIdxT VLiveOutIdx;
  VLiveOutIdx[MBB0] = &VLiveOuts[0];
  VLiveOutIdx[MBB1] = &VLiveOuts[1];
  VLiveOutIdx[MBB2] = &VLiveOuts[2];
  VLiveOutIdx[MBB3] = &VLiveOuts[3];

  SmallVector<const MachineBasicBlock *, 2> Preds;
  for (const auto *Pred : MBB3->predecessors())
    Preds.push_back(Pred);

  // Specify the live-outs around the joining block.
  MOutLocs[1][0] = LiveInRsp;
  MOutLocs[2][0] = LiveInRax;

  bool Result;
  SmallVector<DbgOpID> OutValues;

  // Simple case: join two distinct values on entry to the block.
  VLiveOuts[1] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[2] = DbgValue(LiveInRaxID, EmptyProps);
  OutValues.clear();
  Result = pickVPHILoc(OutValues, *MBB3, VLiveOutIdx, MOutLocs, Preds);
  // Should have picked a PHI in $rsp in block 3.
  EXPECT_TRUE(Result);
  if (Result) {
    EXPECT_EQ(OutValues[0], RspPHIInBlk3ID);
  }

  // If the incoming values are swapped between blocks, we should not
  // successfully join. The CFG merge would select the right values, but in
  // the wrong conditions.
  std::swap(VLiveOuts[1], VLiveOuts[2]);
  OutValues.clear();
  Result = pickVPHILoc(OutValues, *MBB3, VLiveOutIdx, MOutLocs, Preds);
  EXPECT_FALSE(Result);

  // Swap back,
  std::swap(VLiveOuts[1], VLiveOuts[2]);
  // Setting one of these to being a constant should prohibit merging.
  VLiveOuts[1].setDbgOpIDs(ConstZeroID);
  OutValues.clear();
  Result = pickVPHILoc(OutValues, *MBB3, VLiveOutIdx, MOutLocs, Preds);
  EXPECT_FALSE(Result);

  // Setting both to being identical constants should result in a valid join
  // with a constant value.
  VLiveOuts[2] = VLiveOuts[1];
  OutValues.clear();
  Result = pickVPHILoc(OutValues, *MBB3, VLiveOutIdx, MOutLocs, Preds);
  EXPECT_TRUE(Result);
  if (Result) {
    EXPECT_EQ(OutValues[0], ConstZeroID);
  }

  // NoVals shouldn't join with anything else.
  VLiveOuts[1] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[2] = DbgValue(2, EmptyProps, DbgValue::NoVal);
  OutValues.clear();
  Result = pickVPHILoc(OutValues, *MBB3, VLiveOutIdx, MOutLocs, Preds);
  EXPECT_FALSE(Result);

  // We might merge in another VPHI in such a join. Present pickVPHILoc with
  // such a scenario: first, where one incoming edge has a VPHI with no known
  // value. This represents an edge where there was a PHI value that can't be
  // found in the register file -- we can't subsequently find a PHI here.
  VLiveOuts[1] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[2] = DbgValue(2, EmptyProps, DbgValue::VPHI);
  OutValues.clear();
  Result = pickVPHILoc(OutValues, *MBB3, VLiveOutIdx, MOutLocs, Preds);
  EXPECT_FALSE(Result);

  // However, if we know the value of the incoming VPHI, we can search for its
  // location. Use a PHI machine-value for doing this, as VPHIs should always
  // have PHI values, or they should have been eliminated.
  MOutLocs[2][0] = RspPHIInBlk2;
  VLiveOuts[1] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[2] = DbgValue(2, EmptyProps, DbgValue::VPHI);
  VLiveOuts[2].setDbgOpIDs(RspPHIInBlk2ID); // Set location where PHI happens.
  OutValues.clear();
  Result = pickVPHILoc(OutValues, *MBB3, VLiveOutIdx, MOutLocs, Preds);
  EXPECT_TRUE(Result);
  if (Result) {
    EXPECT_EQ(OutValues[0], RspPHIInBlk3ID);
  }

  // If that value isn't available from that block, don't join.
  MOutLocs[2][0] = LiveInRsp;
  OutValues.clear();
  Result = pickVPHILoc(OutValues, *MBB3, VLiveOutIdx, MOutLocs, Preds);
  EXPECT_FALSE(Result);

  // Check that we don't pick values when the properties disagree, for example
  // different indirectness or DIExpression.
  MOutLocs[2][0] = LiveInRax;
  DIExpression *NewExpr =
      DIExpression::prepend(EmptyExpr, DIExpression::ApplyOffset, 4);
  DbgValueProperties PropsWithExpr(NewExpr, false, false);
  VLiveOuts[1] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[2] = DbgValue(LiveInRspID, PropsWithExpr);
  OutValues.clear();
  Result = pickVPHILoc(OutValues, *MBB3, VLiveOutIdx, MOutLocs, Preds);
  EXPECT_FALSE(Result);

  DbgValueProperties PropsWithIndirect(EmptyExpr, true, false);
  VLiveOuts[1] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[2] = DbgValue(LiveInRspID, PropsWithIndirect);
  OutValues.clear();
  Result = pickVPHILoc(OutValues, *MBB3, VLiveOutIdx, MOutLocs, Preds);
  EXPECT_FALSE(Result);

  // When we have operands with values [A, B] and [B, A], we do not necessarily
  // pick identical join locations for each operand if the locations of those
  // values differ between incoming basic blocks.
  MOutLocs[1][0] = LiveInRsp;
  MOutLocs[2][0] = LiveInRax;
  MOutLocs[1][1] = LiveInRax;
  MOutLocs[2][1] = LiveInRsp;
  DbgOpID Locs0[] = {LiveInRspID, LiveInRaxID};
  DbgOpID Locs1[] = {LiveInRaxID, LiveInRspID};
  VLiveOuts[1] = DbgValue(Locs0, TwoOpProps);
  VLiveOuts[2] = DbgValue(Locs1, TwoOpProps);
  OutValues.clear();
  Result = pickVPHILoc(OutValues, *MBB3, VLiveOutIdx, MOutLocs, Preds);
  // Should have picked PHIs in block 3.
  EXPECT_TRUE(Result);
  if (Result) {
    EXPECT_EQ(OutValues[0], RspPHIInBlk3ID);
    EXPECT_EQ(OutValues[1], RaxPHIInBlk3ID);
  }

  // When joining identical constants for an operand, we should simply keep that
  // constant while joining the remaining operands as normal.
  DbgOpID Locs2[] = {LiveInRspID, ConstZeroID};
  DbgOpID Locs3[] = {LiveInRaxID, ConstZeroID};
  VLiveOuts[1] = DbgValue(Locs2, TwoOpProps);
  VLiveOuts[2] = DbgValue(Locs3, TwoOpProps);
  OutValues.clear();
  Result = pickVPHILoc(OutValues, *MBB3, VLiveOutIdx, MOutLocs, Preds);
  EXPECT_TRUE(Result);
  if (Result) {
    EXPECT_EQ(OutValues[0], RspPHIInBlk3ID);
    EXPECT_EQ(OutValues[1], ConstZeroID);
  }

  // If the debug values have different constants for the same operand, they
  // cannot be joined.
  DbgOpID ConstOneID = addConstDbgOp(MachineOperand::CreateImm(1));
  DbgOpID Locs4[] = {LiveInRaxID, ConstOneID};
  VLiveOuts[1] = DbgValue(Locs3, TwoOpProps);
  VLiveOuts[2] = DbgValue(Locs4, TwoOpProps);
  OutValues.clear();
  Result = pickVPHILoc(OutValues, *MBB3, VLiveOutIdx, MOutLocs, Preds);
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

  auto [MInLocs, MOutLocs] = allocValueTables(3, 2);

  initValueArray(MOutLocs, 3, 2);

  unsigned EntryBlk = 0, LoopBlk = 1;

  ValueIDNum LiveInRsp(EntryBlk, 0, RspLoc);
  ValueIDNum LiveInRax(EntryBlk, 0, RaxLoc);
  ValueIDNum RspPHIInBlk1(LoopBlk, 0, RspLoc);
  ValueIDNum RaxPHIInBlk1(LoopBlk, 0, RaxLoc);
  DbgOpID LiveInRspID = addValueDbgOp(LiveInRsp);
  DbgOpID LiveInRaxID = addValueDbgOp(LiveInRax);
  DbgOpID RspPHIInBlk1ID = addValueDbgOp(RspPHIInBlk1);
  DbgOpID RaxPHIInBlk1ID = addValueDbgOp(RaxPHIInBlk1);

  DebugVariable Var(FuncVariable, std::nullopt, nullptr);
  DbgValueProperties EmptyProps(EmptyExpr, false, false);
  DIExpression *TwoOpExpr =
      DIExpression::get(Ctx, {dwarf::DW_OP_LLVM_arg, 0, dwarf::DW_OP_LLVM_arg,
                              1, dwarf::DW_OP_plus});
  DbgValueProperties TwoOpProps(TwoOpExpr, false, true);
  SmallVector<DbgValue, 32> VLiveOuts;
  VLiveOuts.resize(3, DbgValue(EmptyProps, DbgValue::Undef));
  InstrRefBasedLDV::LiveIdxT VLiveOutIdx;
  VLiveOutIdx[MBB0] = &VLiveOuts[0];
  VLiveOutIdx[MBB1] = &VLiveOuts[1];
  VLiveOutIdx[MBB2] = &VLiveOuts[2];

  SmallVector<const MachineBasicBlock *, 2> Preds;
  for (const auto *Pred : MBB1->predecessors())
    Preds.push_back(Pred);

  // Specify the live-outs around the joining block.
  MOutLocs[0][0] = LiveInRsp;
  MOutLocs[1][0] = LiveInRax;

  bool Result;
  SmallVector<DbgOpID> OutValues;

  // See that we can merge as normal on a backedge.
  VLiveOuts[0] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[1] = DbgValue(LiveInRaxID, EmptyProps);
  OutValues.clear();
  Result = pickVPHILoc(OutValues, *MBB1, VLiveOutIdx, MOutLocs, Preds);
  // Should have picked a PHI in $rsp in block 1.
  EXPECT_TRUE(Result);
  if (Result) {
    EXPECT_EQ(OutValues[0], RspPHIInBlk1ID);
  }

  // And that, if the desired values aren't available, we don't merge.
  MOutLocs[1][0] = LiveInRsp;
  OutValues.clear();
  Result = pickVPHILoc(OutValues, *MBB1, VLiveOutIdx, MOutLocs, Preds);
  EXPECT_FALSE(Result);

  // Test the backedge behaviour: PHIs that feed back into themselves can
  // carry this variables value. Feed in LiveInRsp in both $rsp and $rax
  // from the entry block, but only put an appropriate backedge PHI in $rax.
  // Only the $rax location can form the correct PHI.
  MOutLocs[0][0] = LiveInRsp;
  MOutLocs[0][1] = LiveInRsp;
  MOutLocs[1][0] = RaxPHIInBlk1;
  MOutLocs[1][1] = RaxPHIInBlk1;
  VLiveOuts[0] = DbgValue(LiveInRspID, EmptyProps);
  // Crucially, a VPHI originating in this block:
  VLiveOuts[1] = DbgValue(1, EmptyProps, DbgValue::VPHI);
  OutValues.clear();
  Result = pickVPHILoc(OutValues, *MBB1, VLiveOutIdx, MOutLocs, Preds);
  EXPECT_TRUE(Result);
  if (Result) {
    EXPECT_EQ(OutValues[0], RaxPHIInBlk1ID);
  }

  // Test that we can also find a location when joining a backedge PHI with
  // a variadic debug value.
  MOutLocs[1][0] = RspPHIInBlk1;
  MOutLocs[0][1] = LiveInRax;
  DbgOpID Locs0[] = {LiveInRspID, LiveInRaxID};
  VLiveOuts[0] = DbgValue(Locs0, TwoOpProps);
  // Crucially, a VPHI originating in this block:
  VLiveOuts[1] = DbgValue(1, TwoOpProps, DbgValue::VPHI);
  OutValues.clear();
  Result = pickVPHILoc(OutValues, *MBB1, VLiveOutIdx, MOutLocs, Preds);
  EXPECT_TRUE(Result);
  if (Result) {
    EXPECT_EQ(OutValues[0], RspPHIInBlk1ID);
    EXPECT_EQ(OutValues[1], RaxPHIInBlk1ID);
  }

  // Merging should not be permitted if there's a usable PHI on the backedge,
  // but it's in the wrong place. (Overwrite $rax).
  MOutLocs[1][1] = LiveInRax;
  VLiveOuts[0] = DbgValue(LiveInRspID, EmptyProps);
  OutValues.clear();
  Result = pickVPHILoc(OutValues, *MBB1, VLiveOutIdx, MOutLocs, Preds);
  EXPECT_FALSE(Result);

  // Additionally, if the VPHI coming back on the loop backedge isn't from
  // this block (block 1), we can't merge it.
  MOutLocs[1][1] = RaxPHIInBlk1;
  VLiveOuts[0] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[1] = DbgValue(0, EmptyProps, DbgValue::VPHI);
  OutValues.clear();
  Result = pickVPHILoc(OutValues, *MBB1, VLiveOutIdx, MOutLocs, Preds);
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

  auto [MInLocs, MOutLocs] = allocValueTables(5, 3);

  initValueArray(MOutLocs, 5, 3);

  unsigned EntryBlk = 0, Loop1Blk = 1;

  ValueIDNum LiveInRsp(EntryBlk, 0, RspLoc);
  ValueIDNum LiveInRax(EntryBlk, 0, RaxLoc);
  ValueIDNum LiveInRbx(EntryBlk, 0, RbxLoc);
  ValueIDNum RspPHIInBlk1(Loop1Blk, 0, RspLoc);
  ValueIDNum RaxPHIInBlk1(Loop1Blk, 0, RaxLoc);
  ValueIDNum RbxPHIInBlk1(Loop1Blk, 0, RbxLoc);
  DbgOpID LiveInRspID = addValueDbgOp(LiveInRsp);
  DbgOpID LiveInRaxID = addValueDbgOp(LiveInRax);
  DbgOpID LiveInRbxID = addValueDbgOp(LiveInRbx);
  DbgOpID RspPHIInBlk1ID = addValueDbgOp(RspPHIInBlk1);
  addValueDbgOp(RaxPHIInBlk1);
  DbgOpID RbxPHIInBlk1ID = addValueDbgOp(RbxPHIInBlk1);

  DebugVariable Var(FuncVariable, std::nullopt, nullptr);
  DbgValueProperties EmptyProps(EmptyExpr, false, false);
  SmallVector<DbgValue, 32> VLiveOuts;
  VLiveOuts.resize(5, DbgValue(EmptyProps, DbgValue::Undef));
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
  MOutLocs[0][0] = LiveInRsp;
  MOutLocs[1][0] = LiveInRax;
  MOutLocs[2][0] = LiveInRbx;

  bool Result;
  SmallVector<DbgOpID> OutValues;

  // See that we can merge as normal on a backedge.
  VLiveOuts[0] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[1] = DbgValue(LiveInRaxID, EmptyProps);
  VLiveOuts[2] = DbgValue(LiveInRbxID, EmptyProps);
  OutValues.clear();
  Result = pickVPHILoc(OutValues, *MBB1, VLiveOutIdx, MOutLocs, Preds);
  // Should have picked a PHI in $rsp in block 1.
  EXPECT_TRUE(Result);
  if (Result) {
    EXPECT_EQ(OutValues[0], RspPHIInBlk1ID);
  }

  // Check too that permuting the live-out locations prevents merging
  MOutLocs[0][0] = LiveInRax;
  MOutLocs[1][0] = LiveInRbx;
  MOutLocs[2][0] = LiveInRsp;
  OutValues.clear();
  Result = pickVPHILoc(OutValues, *MBB1, VLiveOutIdx, MOutLocs, Preds);
  EXPECT_FALSE(Result);

  MOutLocs[0][0] = LiveInRsp;
  MOutLocs[1][0] = LiveInRax;
  MOutLocs[2][0] = LiveInRbx;

  // Feeding a PHI back on one backedge shouldn't merge (block 1 self backedge
  // wants LiveInRax).
  MOutLocs[1][0] = RspPHIInBlk1;
  OutValues.clear();
  Result = pickVPHILoc(OutValues, *MBB1, VLiveOutIdx, MOutLocs, Preds);
  EXPECT_FALSE(Result);

  // If the variables value on that edge is a VPHI feeding into itself, that's
  // fine.
  VLiveOuts[0] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[1] = DbgValue(1, EmptyProps, DbgValue::VPHI);
  VLiveOuts[2] = DbgValue(LiveInRbxID, EmptyProps);
  OutValues.clear();
  Result = pickVPHILoc(OutValues, *MBB1, VLiveOutIdx, MOutLocs, Preds);
  EXPECT_TRUE(Result);
  if (Result) {
    EXPECT_EQ(OutValues[0], RspPHIInBlk1ID);
  }

  // Likewise: the other backedge being a VPHI from block 1 should be accepted.
  MOutLocs[2][0] = RspPHIInBlk1;
  VLiveOuts[0] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[1] = DbgValue(1, EmptyProps, DbgValue::VPHI);
  VLiveOuts[2] = DbgValue(1, EmptyProps, DbgValue::VPHI);
  OutValues.clear();
  Result = pickVPHILoc(OutValues, *MBB1, VLiveOutIdx, MOutLocs, Preds);
  EXPECT_TRUE(Result);
  if (Result) {
    EXPECT_EQ(OutValues[0], RspPHIInBlk1ID);
  }

  // Here's where it becomes tricky: we should not merge if there are two
  // _distinct_ backedge PHIs. We can't have a PHI that happens in both rsp
  // and rax for example. We can only pick one location as the live-in.
  MOutLocs[2][0] = RaxPHIInBlk1;
  OutValues.clear();
  Result = pickVPHILoc(OutValues, *MBB1, VLiveOutIdx, MOutLocs, Preds);
  EXPECT_FALSE(Result);

  // The above test sources correct machine-PHI-value from two places. Now
  // try with one machine-PHI-value, but placed in two different locations
  // on the backedge. Again, we can't merge a location here, there's no
  // location that works on all paths.
  MOutLocs[0][0] = LiveInRsp;
  MOutLocs[1][0] = RspPHIInBlk1;
  MOutLocs[2][0] = LiveInRsp;
  MOutLocs[2][1] = RspPHIInBlk1;
  OutValues.clear();
  Result = pickVPHILoc(OutValues, *MBB1, VLiveOutIdx, MOutLocs, Preds);
  EXPECT_FALSE(Result);

  // Scatter various PHI values across the available locations. Only rbx (loc 2)
  // has the right value in both backedges -- that's the loc that should be
  // picked.
  MOutLocs[0][2] = LiveInRsp;
  MOutLocs[1][0] = RspPHIInBlk1;
  MOutLocs[1][1] = RaxPHIInBlk1;
  MOutLocs[1][2] = RbxPHIInBlk1;
  MOutLocs[2][0] = LiveInRsp;
  MOutLocs[2][1] = RspPHIInBlk1;
  MOutLocs[2][2] = RbxPHIInBlk1;
  OutValues.clear();
  Result = pickVPHILoc(OutValues, *MBB1, VLiveOutIdx, MOutLocs, Preds);
  EXPECT_TRUE(Result);
  if (Result) {
    EXPECT_EQ(OutValues[0], RbxPHIInBlk1ID);
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
  MTracker->lookupOrTrackRegister(RAX);

  DbgOpID LiveInRspID = DbgOpID(false, 0);
  DbgOpID LiveInRaxID = DbgOpID(false, 1);

  DebugVariable Var(FuncVariable, std::nullopt, nullptr);
  DbgValueProperties EmptyProps(EmptyExpr, false, false);
  SmallVector<DbgValue, 32> VLiveOuts;
  VLiveOuts.resize(4, DbgValue(EmptyProps, DbgValue::Undef));
  InstrRefBasedLDV::LiveIdxT VLiveOutIdx;
  VLiveOutIdx[MBB0] = &VLiveOuts[0];
  VLiveOutIdx[MBB1] = &VLiveOuts[1];
  VLiveOutIdx[MBB2] = &VLiveOuts[2];
  VLiveOutIdx[MBB3] = &VLiveOuts[3];

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

  // vlocJoin is here to propagate incoming values, and eliminate PHIs. Start
  // off by propagating a value into the merging block, number 3.
  DbgValue JoinedLoc = DbgValue(3, EmptyProps, DbgValue::NoVal);
  VLiveOuts[1] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[2] = DbgValue(LiveInRspID, EmptyProps);
  bool Result = vlocJoin(*MBB3, VLiveOutIdx, AllBlocks, JoinedLoc);
  EXPECT_TRUE(Result); // Output locs should have changed.
  EXPECT_EQ(JoinedLoc.Kind, DbgValue::Def);
  EXPECT_EQ(JoinedLoc.getDbgOpID(0), LiveInRspID);

  // And if we did it a second time, leaving the live-ins as it was, then
  // we should report no change.
  Result = vlocJoin(*MBB3, VLiveOutIdx, AllBlocks, JoinedLoc);
  EXPECT_FALSE(Result);

  // If the live-in variable values are different, but there's no PHI placed
  // in this block, then just pick a location. It should be the first (in RPO)
  // predecessor to avoid being a backedge.
  VLiveOuts[1] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[2] = DbgValue(LiveInRaxID, EmptyProps);
  JoinedLoc = DbgValue(3, EmptyProps, DbgValue::NoVal);
  Result = vlocJoin(*MBB3, VLiveOutIdx, AllBlocks, JoinedLoc);
  EXPECT_TRUE(Result);
  EXPECT_EQ(JoinedLoc.Kind, DbgValue::Def);
  // RPO is blocks 0 2 1 3, so LiveInRax is picked as the first predecessor
  // of this join.
  EXPECT_EQ(JoinedLoc.getDbgOpID(0), LiveInRaxID);

  // No tests for whether vlocJoin will pass-through a variable with differing
  // expressions / properties. Those can only come about due to assignments; and
  // for any assignment at all, a PHI should have been placed at the dominance
  // frontier. We rely on the IDF calculator being accurate (which is OK,
  // because so does the rest of LLVM).

  // Try placing a PHI. With differing input values (LiveInRsp, LiveInRax),
  // this PHI should not be eliminated.
  JoinedLoc = DbgValue(3, EmptyProps, DbgValue::VPHI);
  Result = vlocJoin(*MBB3, VLiveOutIdx, AllBlocks, JoinedLoc);
  // Expect no change.
  EXPECT_FALSE(Result);
  EXPECT_EQ(JoinedLoc.Kind, DbgValue::VPHI);
  // This should not have been assigned a fixed value.
  EXPECT_EQ(JoinedLoc.getDbgOpID(0), DbgOpID::UndefID);
  EXPECT_EQ(JoinedLoc.BlockNo, 3);

  // Try a simple PHI elimination. Put a PHI in block 3, but LiveInRsp on both
  // incoming edges. Re-load in and out-locs with unrelated values; they're
  // irrelevant.
  VLiveOuts[1] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[2] = DbgValue(LiveInRspID, EmptyProps);
  JoinedLoc = DbgValue(3, EmptyProps, DbgValue::VPHI);
  Result = vlocJoin(*MBB3, VLiveOutIdx, AllBlocks, JoinedLoc);
  EXPECT_TRUE(Result);
  EXPECT_EQ(JoinedLoc.Kind, DbgValue::Def);
  EXPECT_EQ(JoinedLoc.getDbgOpID(0), LiveInRspID);

  // Try the same PHI elimination but with one incoming value being a VPHI
  // referring to the same value.
  VLiveOuts[1] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[2] = DbgValue(2, EmptyProps, DbgValue::VPHI);
  VLiveOuts[2].setDbgOpIDs(LiveInRspID);
  JoinedLoc = DbgValue(3, EmptyProps, DbgValue::VPHI);
  Result = vlocJoin(*MBB3, VLiveOutIdx, AllBlocks, JoinedLoc);
  EXPECT_TRUE(Result);
  EXPECT_EQ(JoinedLoc.Kind, DbgValue::VPHI);
  EXPECT_EQ(JoinedLoc.getDbgOpID(0), LiveInRspID);

  // If the "current" live-in is a VPHI, but not a VPHI generated in the current
  // block, then it's the remains of an earlier value propagation. We should
  // value propagate through this merge. Even if the current incoming values
  // disagree, because we've previously determined any VPHI here is redundant.
  VLiveOuts[1] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[2] = DbgValue(LiveInRaxID, EmptyProps);
  JoinedLoc = DbgValue(2, EmptyProps, DbgValue::VPHI);
  Result = vlocJoin(*MBB3, VLiveOutIdx, AllBlocks, JoinedLoc);
  EXPECT_TRUE(Result);
  EXPECT_EQ(JoinedLoc.Kind, DbgValue::Def);
  EXPECT_EQ(JoinedLoc.getDbgOpID(0), LiveInRaxID); // from block 2

  // The above test, but test that we will install one value-propagated VPHI
  // over another.
  VLiveOuts[1] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[2] = DbgValue(0, EmptyProps, DbgValue::VPHI);
  JoinedLoc = DbgValue(2, EmptyProps, DbgValue::VPHI);
  Result = vlocJoin(*MBB3, VLiveOutIdx, AllBlocks, JoinedLoc);
  EXPECT_TRUE(Result);
  EXPECT_EQ(JoinedLoc.Kind, DbgValue::VPHI);
  EXPECT_EQ(JoinedLoc.BlockNo, 0);

  // We shouldn't eliminate PHIs when properties disagree.
  DbgValueProperties PropsWithIndirect(EmptyExpr, true, false);
  VLiveOuts[1] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[2] = DbgValue(LiveInRspID, PropsWithIndirect);
  JoinedLoc = DbgValue(3, EmptyProps, DbgValue::VPHI);
  Result = vlocJoin(*MBB3, VLiveOutIdx, AllBlocks, JoinedLoc);
  EXPECT_FALSE(Result);
  EXPECT_EQ(JoinedLoc.Kind, DbgValue::VPHI);
  EXPECT_EQ(JoinedLoc.BlockNo, 3);

  // Even if properties disagree, we should still value-propagate if there's no
  // PHI to be eliminated. The disagreeing values should work themselves out,
  // seeing how we've determined no PHI is necessary.
  VLiveOuts[1] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[2] = DbgValue(LiveInRspID, PropsWithIndirect);
  JoinedLoc = DbgValue(2, EmptyProps, DbgValue::VPHI);
  Result = vlocJoin(*MBB3, VLiveOutIdx, AllBlocks, JoinedLoc);
  EXPECT_TRUE(Result);
  EXPECT_EQ(JoinedLoc.Kind, DbgValue::Def);
  EXPECT_EQ(JoinedLoc.getDbgOpID(0), LiveInRspID);
  // Also check properties come from block 2, the first RPO predecessor to block
  // three.
  EXPECT_EQ(JoinedLoc.Properties, PropsWithIndirect);

  // Again, disagreeing properties, this time the expr, should cause a PHI to
  // not be eliminated.
  DIExpression *NewExpr =
      DIExpression::prepend(EmptyExpr, DIExpression::ApplyOffset, 4);
  DbgValueProperties PropsWithExpr(NewExpr, false, false);
  VLiveOuts[1] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[2] = DbgValue(LiveInRspID, PropsWithExpr);
  JoinedLoc = DbgValue(3, EmptyProps, DbgValue::VPHI);
  Result = vlocJoin(*MBB3, VLiveOutIdx, AllBlocks, JoinedLoc);
  EXPECT_FALSE(Result);

  // Try placing a PHI with variadic debug values. With differing input values
  // (LiveInRsp, LiveInRax), this PHI should not be eliminated.
  DIExpression *TwoOpExpr =
      DIExpression::get(Ctx, {dwarf::DW_OP_LLVM_arg, 0, dwarf::DW_OP_LLVM_arg,
                              1, dwarf::DW_OP_plus});
  DbgValueProperties TwoOpProps(TwoOpExpr, false, true);
  DbgOpID Locs0[] = {LiveInRspID, LiveInRaxID};
  DbgOpID Locs1[] = {LiveInRaxID, LiveInRspID};
  JoinedLoc = DbgValue(3, TwoOpProps, DbgValue::VPHI);
  VLiveOuts[1] = DbgValue(Locs0, TwoOpProps);
  VLiveOuts[2] = DbgValue(Locs1, TwoOpProps);
  Result = vlocJoin(*MBB3, VLiveOutIdx, AllBlocks, JoinedLoc);
  // Expect no change.
  EXPECT_FALSE(Result);
  EXPECT_EQ(JoinedLoc.Kind, DbgValue::VPHI);
  // This should not have been assigned a fixed value.
  EXPECT_EQ(JoinedLoc.getDbgOpID(0), DbgOpID::UndefID);
  EXPECT_EQ(JoinedLoc.BlockNo, 3);
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
  MTracker->lookupOrTrackRegister(RAX);

  DbgOpID LiveInRspID = DbgOpID(false, 0);
  DbgOpID LiveInRaxID = DbgOpID(false, 1);

  DebugVariable Var(FuncVariable, std::nullopt, nullptr);
  DbgValueProperties EmptyProps(EmptyExpr, false, false);
  SmallVector<DbgValue, 32> VLiveOuts;
  VLiveOuts.resize(3, DbgValue(EmptyProps, DbgValue::Undef));
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

  // Test some back-edge-specific behaviours of vloc join. Mostly: the fact that
  // VPHIs that arrive on backedges can be eliminated, despite having different
  // values to the predecessor.

  // First: when there's no VPHI placed already, propagate the live-in value of
  // the first RPO predecessor.
  VLiveOuts[0] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[1] = DbgValue(LiveInRaxID, EmptyProps);
  DbgValue JoinedLoc = DbgValue(LiveInRaxID, EmptyProps);
  bool Result = vlocJoin(*MBB1, VLiveOutIdx, AllBlocks, JoinedLoc);
  EXPECT_TRUE(Result);
  EXPECT_EQ(JoinedLoc.Kind, DbgValue::Def);
  EXPECT_EQ(JoinedLoc.getDbgOpID(0), LiveInRspID);

  // If there is a VPHI: don't elimiante it if there are disagreeing values.
  VLiveOuts[0] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[1] = DbgValue(LiveInRaxID, EmptyProps);
  JoinedLoc = DbgValue(1, EmptyProps, DbgValue::VPHI);
  Result = vlocJoin(*MBB1, VLiveOutIdx, AllBlocks, JoinedLoc);
  EXPECT_FALSE(Result);
  EXPECT_EQ(JoinedLoc.Kind, DbgValue::VPHI);
  EXPECT_EQ(JoinedLoc.BlockNo, 1);

  // If we feed this VPHI back into itself though, we can eliminate it.
  VLiveOuts[0] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[1] = DbgValue(1, EmptyProps, DbgValue::VPHI);
  JoinedLoc = DbgValue(1, EmptyProps, DbgValue::VPHI);
  Result = vlocJoin(*MBB1, VLiveOutIdx, AllBlocks, JoinedLoc);
  EXPECT_TRUE(Result);
  EXPECT_EQ(JoinedLoc.Kind, DbgValue::Def);
  EXPECT_EQ(JoinedLoc.getDbgOpID(0), LiveInRspID);

  // Don't eliminate backedge VPHIs if the predecessors have different
  // properties.
  DIExpression *NewExpr =
      DIExpression::prepend(EmptyExpr, DIExpression::ApplyOffset, 4);
  DbgValueProperties PropsWithExpr(NewExpr, false, false);
  VLiveOuts[0] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[1] = DbgValue(1, PropsWithExpr, DbgValue::VPHI);
  JoinedLoc = DbgValue(1, EmptyProps, DbgValue::VPHI);
  Result = vlocJoin(*MBB1, VLiveOutIdx, AllBlocks, JoinedLoc);
  EXPECT_FALSE(Result);
  EXPECT_EQ(JoinedLoc.Kind, DbgValue::VPHI);
  EXPECT_EQ(JoinedLoc.BlockNo, 1);

  // Backedges with VPHIs, but from the wrong block, shouldn't be eliminated.
  VLiveOuts[0] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[1] = DbgValue(0, EmptyProps, DbgValue::VPHI);
  JoinedLoc = DbgValue(1, EmptyProps, DbgValue::VPHI);
  Result = vlocJoin(*MBB1, VLiveOutIdx, AllBlocks, JoinedLoc);
  EXPECT_FALSE(Result);
  EXPECT_EQ(JoinedLoc.Kind, DbgValue::VPHI);
  EXPECT_EQ(JoinedLoc.BlockNo, 1);
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
  MTracker->lookupOrTrackRegister(RAX);
  Register RBX = getRegByName("RBX");
  MTracker->lookupOrTrackRegister(RBX);

  DbgOpID LiveInRspID = DbgOpID(false, 0);
  DbgOpID LiveInRaxID = DbgOpID(false, 1);
  DbgOpID LiveInRbxID = DbgOpID(false, 2);

  DebugVariable Var(FuncVariable, std::nullopt, nullptr);
  DbgValueProperties EmptyProps(EmptyExpr, false, false);
  SmallVector<DbgValue, 32> VLiveOuts;
  VLiveOuts.resize(5, DbgValue(EmptyProps, DbgValue::Undef));
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

  // Test a normal VPHI isn't eliminated.
  VLiveOuts[0] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[1] = DbgValue(LiveInRaxID, EmptyProps);
  VLiveOuts[2] = DbgValue(LiveInRbxID, EmptyProps);
  DbgValue JoinedLoc = DbgValue(1, EmptyProps, DbgValue::VPHI);
  bool Result = vlocJoin(*MBB1, VLiveOutIdx, AllBlocks, JoinedLoc);
  EXPECT_FALSE(Result);
  EXPECT_EQ(JoinedLoc.Kind, DbgValue::VPHI);
  EXPECT_EQ(JoinedLoc.BlockNo, 1);

  // Common VPHIs on backedges should merge.
  VLiveOuts[0] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[1] = DbgValue(1, EmptyProps, DbgValue::VPHI);
  VLiveOuts[2] = DbgValue(1, EmptyProps, DbgValue::VPHI);
  JoinedLoc = DbgValue(1, EmptyProps, DbgValue::VPHI);
  Result = vlocJoin(*MBB1, VLiveOutIdx, AllBlocks, JoinedLoc);
  EXPECT_TRUE(Result);
  EXPECT_EQ(JoinedLoc.Kind, DbgValue::Def);
  EXPECT_EQ(JoinedLoc.getDbgOpID(0), LiveInRspID);

  // They shouldn't merge if one of their properties is different.
  DbgValueProperties PropsWithIndirect(EmptyExpr, true, false);
  VLiveOuts[0] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[1] = DbgValue(1, EmptyProps, DbgValue::VPHI);
  VLiveOuts[2] = DbgValue(1, PropsWithIndirect, DbgValue::VPHI);
  JoinedLoc = DbgValue(1, EmptyProps, DbgValue::VPHI);
  Result = vlocJoin(*MBB1, VLiveOutIdx, AllBlocks, JoinedLoc);
  EXPECT_FALSE(Result);
  EXPECT_EQ(JoinedLoc.Kind, DbgValue::VPHI);
  EXPECT_EQ(JoinedLoc.BlockNo, 1);

  // VPHIs from different blocks should not merge.
  VLiveOuts[0] = DbgValue(LiveInRspID, EmptyProps);
  VLiveOuts[1] = DbgValue(1, EmptyProps, DbgValue::VPHI);
  VLiveOuts[2] = DbgValue(2, EmptyProps, DbgValue::VPHI);
  JoinedLoc = DbgValue(1, EmptyProps, DbgValue::VPHI);
  Result = vlocJoin(*MBB1, VLiveOutIdx, AllBlocks, JoinedLoc);
  EXPECT_FALSE(Result);
  EXPECT_EQ(JoinedLoc.Kind, DbgValue::VPHI);
  EXPECT_EQ(JoinedLoc.BlockNo, 1);
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

  auto [MInLocs, MOutLocs] = allocValueTables(1, 2);

  ValueIDNum LiveInRsp = ValueIDNum(0, 0, RspLoc);
  DbgOpID LiveInRspID = addValueDbgOp(LiveInRsp);
  MInLocs[0][0] = MOutLocs[0][0] = LiveInRsp;

  DebugVariable Var(FuncVariable, std::nullopt, nullptr);
  DebugVariableID VarID = LDV->getDVMap().insertDVID(Var, OutermostLoc);
  DbgValueProperties EmptyProps(EmptyExpr, false, false);

  SmallSet<DebugVariableID, 4> AllVars;
  AllVars.insert(VarID);

  // Mild hack: rather than constructing machine instructions in each block
  // and creating lexical scopes across them, instead just tell
  // buildVLocValueMap that there's an assignment in every block. That makes
  // every block in scope.
  SmallPtrSet<MachineBasicBlock *, 4> AssignBlocks;
  AssignBlocks.insert(MBB0);

  SmallVector<VLocTracker, 1> VLocs;
  VLocs.resize(1, VLocTracker(LDV->getDVMap(), Overlaps, EmptyExpr));

  InstrRefBasedLDV::LiveInsT Output;

  // Test that, with no assignments at all, no mappings are created for the
  // variable in this function.
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output.size(), 0ul);

  // If we put an assignment in the transfer function, that should... well,
  // do nothing, because we don't store the live-outs.
  VLocs[0].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
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
  DbgOpID LiveInRspID = addValueDbgOp(LiveInRsp);
  DbgOpID LiveInRaxID = addValueDbgOp(LiveInRax);
  DbgOpID RspPHIInBlk3ID = addValueDbgOp(RspPHIInBlk3);

  auto [MInLocs, MOutLocs] = allocValueTables(4, 2);

  initValueArray(MInLocs, 4, 2);
  initValueArray(MOutLocs, 4, 2);

  DebugVariable Var(FuncVariable, std::nullopt, nullptr);
  DebugVariableID VarID = LDV->getDVMap().insertDVID(Var, OutermostLoc);
  DbgValueProperties EmptyProps(EmptyExpr, false, false);

  SmallSet<DebugVariableID, 4> AllVars;
  AllVars.insert(VarID);

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
  VLocs.resize(4, VLocTracker(LDV->getDVMap(), Overlaps, EmptyExpr));

  InstrRefBasedLDV::LiveInsT Output;

  // Start off with LiveInRsp in every location.
  for (unsigned int I = 0; I < 4; ++I) {
    MInLocs[I][0] = MInLocs[I][1] = LiveInRsp;
    MOutLocs[I][0] = MOutLocs[I][1] = LiveInRsp;
  }

  auto ClearOutputs = [&]() {
    for (auto &Elem : Output)
      Elem.clear();
  };
  Output.resize(4);

  // No assignments -> no values.
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 0ul);
  EXPECT_EQ(Output[2].size(), 0ul);
  EXPECT_EQ(Output[3].size(), 0ul);

  // An assignment in the end block should also not affect other blocks; or
  // produce any live-ins.
  VLocs[3].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 0ul);
  EXPECT_EQ(Output[2].size(), 0ul);
  EXPECT_EQ(Output[3].size(), 0ul);
  ClearOutputs();

  // Assignments in either of the side-of-diamond blocks should also not be
  // propagated anywhere.
  VLocs[3].Vars.clear();
  VLocs[2].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 0ul);
  EXPECT_EQ(Output[2].size(), 0ul);
  EXPECT_EQ(Output[3].size(), 0ul);
  VLocs[2].Vars.clear();
  ClearOutputs();

  VLocs[1].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 0ul);
  EXPECT_EQ(Output[2].size(), 0ul);
  EXPECT_EQ(Output[3].size(), 0ul);
  VLocs[1].Vars.clear();
  ClearOutputs();

  // However: putting an assignment in the first block should propagate variable
  // values through to all other blocks, as it dominates.
  VLocs[0].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  ASSERT_EQ(Output[3].size(), 1ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.getDbgOpID(0), LiveInRspID);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.getDbgOpID(0), LiveInRspID);
  EXPECT_EQ(Output[3][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[3][0].second.getDbgOpID(0), LiveInRspID);
  ClearOutputs();
  VLocs[0].Vars.clear();

  // Additionally, even if that value isn't available in the register file, it
  // should still be propagated, as buildVLocValueMap shouldn't care about
  // what's in the registers (except for PHIs).
  // values through to all other blocks, as it dominates.
  VLocs[0].Vars.insert({VarID, DbgValue(LiveInRaxID, EmptyProps)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  ASSERT_EQ(Output[3].size(), 1ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.getDbgOpID(0), LiveInRaxID);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.getDbgOpID(0), LiveInRaxID);
  EXPECT_EQ(Output[3][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[3][0].second.getDbgOpID(0), LiveInRaxID);
  ClearOutputs();
  VLocs[0].Vars.clear();

  // We should get a live-in to the merging block, if there are two assigns of
  // the same value in either side of the diamond.
  VLocs[1].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  VLocs[2].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 0ul);
  EXPECT_EQ(Output[2].size(), 0ul);
  ASSERT_EQ(Output[3].size(), 1ul);
  EXPECT_EQ(Output[3][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[3][0].second.getDbgOpID(0), LiveInRspID);
  ClearOutputs();
  VLocs[1].Vars.clear();
  VLocs[2].Vars.clear();

  // If we assign a value in the entry block, then 'undef' on a branch, we
  // shouldn't have a live-in in the merge block.
  VLocs[0].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  VLocs[1].Vars.insert({VarID, DbgValue(EmptyProps, DbgValue::Undef)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  EXPECT_EQ(Output[3].size(), 0ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.getDbgOpID(0), LiveInRspID);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.getDbgOpID(0), LiveInRspID);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[1].Vars.clear();

  // Having different values joining into the merge block should mean we have
  // no live-in in that block. Block ones LiveInRax value doesn't appear as a
  // live-in anywhere, it's block internal.
  VLocs[0].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  VLocs[1].Vars.insert({VarID, DbgValue(LiveInRaxID, EmptyProps)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  EXPECT_EQ(Output[3].size(), 0ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.getDbgOpID(0), LiveInRspID);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.getDbgOpID(0), LiveInRspID);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[1].Vars.clear();

  // But on the other hand, if there's a location in the register file where
  // those two values can be joined, do so.
  MOutLocs[1][0] = LiveInRax;
  VLocs[0].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  VLocs[1].Vars.insert({VarID, DbgValue(LiveInRaxID, EmptyProps)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  ASSERT_EQ(Output[3].size(), 1ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.getDbgOpID(0), LiveInRspID);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.getDbgOpID(0), LiveInRspID);
  EXPECT_EQ(Output[3][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[3][0].second.getDbgOpID(0), RspPHIInBlk3ID);
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
  DbgOpID LiveInRspID = addValueDbgOp(LiveInRsp);
  DbgOpID LiveInRaxID = addValueDbgOp(LiveInRax);
  DbgOpID RspPHIInBlk1ID = addValueDbgOp(RspPHIInBlk1);
  DbgOpID RspDefInBlk1ID = addValueDbgOp(RspDefInBlk1);
  DbgOpID RaxPHIInBlk1ID = addValueDbgOp(RaxPHIInBlk1);

  auto [MInLocs, MOutLocs] = allocValueTables(3, 2);

  initValueArray(MInLocs, 3, 2);
  initValueArray(MOutLocs, 3, 2);

  DebugVariable Var(FuncVariable, std::nullopt, nullptr);
  DebugVariableID VarID = LDV->getDVMap().insertDVID(Var, OutermostLoc);
  DbgValueProperties EmptyProps(EmptyExpr, false, false);
  DIExpression *TwoOpExpr =
      DIExpression::get(Ctx, {dwarf::DW_OP_LLVM_arg, 0, dwarf::DW_OP_LLVM_arg,
                              1, dwarf::DW_OP_plus});
  DbgValueProperties VariadicProps(TwoOpExpr, false, true);

  SmallSet<DebugVariableID, 4> AllVars;
  AllVars.insert(VarID);

  SmallPtrSet<MachineBasicBlock *, 4> AssignBlocks;
  AssignBlocks.insert(MBB0);
  AssignBlocks.insert(MBB1);
  AssignBlocks.insert(MBB2);

  SmallVector<VLocTracker, 3> VLocs;
  VLocs.resize(3, VLocTracker(LDV->getDVMap(), Overlaps, EmptyExpr));

  InstrRefBasedLDV::LiveInsT Output;

  // Start off with LiveInRsp in every location.
  for (unsigned int I = 0; I < 3; ++I) {
    MInLocs[I][0] = MInLocs[I][1] = LiveInRsp;
    MOutLocs[I][0] = MOutLocs[I][1] = LiveInRsp;
  }

  auto ClearOutputs = [&]() {
    for (auto &Elem : Output)
      Elem.clear();
  };
  Output.resize(3);

  // Easy starter: a dominating assign should propagate to all blocks.
  VLocs[0].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.getDbgOpID(0), LiveInRspID);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.getDbgOpID(0), LiveInRspID);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[1].Vars.clear();

  // A variadic assignment should behave the same.
  DbgOpID Locs0[] = {LiveInRspID, LiveInRaxID};
  VLocs[0].Vars.insert({VarID, DbgValue(Locs0, VariadicProps)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output, MOutLocs,
                    MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.getDbgOpID(0), LiveInRspID);
  EXPECT_EQ(Output[1][0].second.getDbgOpID(1), LiveInRaxID);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.getDbgOpID(0), LiveInRspID);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[1].Vars.clear();

  // Put an undef assignment in the loop. Should get no live-in value.
  VLocs[0].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  VLocs[1].Vars.insert({VarID, DbgValue(EmptyProps, DbgValue::Undef)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 0ul);
  EXPECT_EQ(Output[2].size(), 0ul);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[1].Vars.clear();

  // Assignment of the same value should naturally join.
  VLocs[0].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  VLocs[1].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.getDbgOpID(0), LiveInRspID);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.getDbgOpID(0), LiveInRspID);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[1].Vars.clear();

  // Assignment of different values shouldn't join with no machine PHI vals.
  // Will be live-in to exit block as it's dominated.
  VLocs[0].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  VLocs[1].Vars.insert({VarID, DbgValue(LiveInRaxID, EmptyProps)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 0ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.getDbgOpID(0), LiveInRaxID);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[1].Vars.clear();

  // Install a completely unrelated PHI value, that we should not join on. Try
  // with unrelated assign in loop block again.
  MInLocs[1][0] = RspPHIInBlk1;
  MOutLocs[1][0] = RspDefInBlk1;
  VLocs[0].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  VLocs[1].Vars.insert({VarID, DbgValue(LiveInRaxID, EmptyProps)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 0ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.getDbgOpID(0), LiveInRaxID);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[1].Vars.clear();

  // Now, if we assign RspDefInBlk1 in the loop block, we should be able to
  // find the appropriate PHI.
  MInLocs[1][0] = RspPHIInBlk1;
  MOutLocs[1][0] = RspDefInBlk1;
  VLocs[0].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  VLocs[1].Vars.insert({VarID, DbgValue(RspDefInBlk1ID, EmptyProps)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.getDbgOpID(0), RspPHIInBlk1ID);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.getDbgOpID(0), RspDefInBlk1ID);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[1].Vars.clear();

  // If the PHI happens in a different location, the live-in should happen
  // there.
  MInLocs[1][0] = LiveInRsp;
  MOutLocs[1][0] = LiveInRsp;
  MInLocs[1][1] = RaxPHIInBlk1;
  MOutLocs[1][1] = RspDefInBlk1;
  VLocs[0].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  VLocs[1].Vars.insert({VarID, DbgValue(RspDefInBlk1ID, EmptyProps)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.getDbgOpID(0), RaxPHIInBlk1ID);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.getDbgOpID(0), RspDefInBlk1ID);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[1].Vars.clear();

  // The PHI happening in both places should be handled too. Exactly where
  // isn't important, but if the location picked changes, this test will let
  // you know.
  MInLocs[1][0] = RaxPHIInBlk1;
  MOutLocs[1][0] = RspDefInBlk1;
  MInLocs[1][1] = RaxPHIInBlk1;
  MOutLocs[1][1] = RspDefInBlk1;
  VLocs[0].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  VLocs[1].Vars.insert({VarID, DbgValue(RspDefInBlk1ID, EmptyProps)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  // Today, the first register is picked.
  EXPECT_EQ(Output[1][0].second.getDbgOpID(0), RspPHIInBlk1ID);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.getDbgOpID(0), RspDefInBlk1ID);
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
  MInLocs[1][0] = RspPHIInBlk1;
  MOutLocs[1][0] = RspPHIInBlk1;
  MInLocs[1][1] = LiveInRax;
  MOutLocs[1][1] = LiveInRax;
  VLocs[0].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  VLocs[1].Vars.insert({VarID, DbgValue(RspPHIInBlk1ID, EmptyProps)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.getDbgOpID(0), RspPHIInBlk1ID);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.getDbgOpID(0), RspPHIInBlk1ID);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[1].Vars.clear();

  // Test that we can eliminate PHIs. A PHI will be placed at the loop head
  // because there's a def in it.
  MInLocs[1][0] = LiveInRsp;
  MOutLocs[1][0] = LiveInRsp;
  VLocs[0].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  VLocs[1].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.getDbgOpID(0), LiveInRspID);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.getDbgOpID(0), LiveInRspID);
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
  DbgOpID LiveInRspID = addValueDbgOp(LiveInRsp);
  DbgOpID LiveInRaxID = addValueDbgOp(LiveInRax);
  DbgOpID RspPHIInBlk1ID = addValueDbgOp(RspPHIInBlk1);
  DbgOpID RspPHIInBlk2ID = addValueDbgOp(RspPHIInBlk2);
  DbgOpID RspDefInBlk2ID = addValueDbgOp(RspDefInBlk2);

  auto [MInLocs, MOutLocs] = allocValueTables(5, 2);

  initValueArray(MInLocs, 5, 2);
  initValueArray(MOutLocs, 5, 2);

  DebugVariable Var(FuncVariable, std::nullopt, nullptr);
  DebugVariableID VarID = LDV->getDVMap().insertDVID(Var, OutermostLoc);
  DbgValueProperties EmptyProps(EmptyExpr, false, false);

  SmallSet<DebugVariableID, 4> AllVars;
  AllVars.insert(VarID);

  SmallPtrSet<MachineBasicBlock *, 5> AssignBlocks;
  AssignBlocks.insert(MBB0);
  AssignBlocks.insert(MBB1);
  AssignBlocks.insert(MBB2);
  AssignBlocks.insert(MBB3);
  AssignBlocks.insert(MBB4);

  SmallVector<VLocTracker, 5> VLocs;
  VLocs.resize(5, VLocTracker(LDV->getDVMap(), Overlaps, EmptyExpr));

  InstrRefBasedLDV::LiveInsT Output;

  // Start off with LiveInRsp in every location.
  for (unsigned int I = 0; I < 5; ++I) {
    MInLocs[I][0] = MInLocs[I][1] = LiveInRsp;
    MOutLocs[I][0] = MOutLocs[I][1] = LiveInRsp;
  }

  auto ClearOutputs = [&]() {
    for (auto &Elem : Output)
      Elem.clear();
  };
  Output.resize(5);

  // A dominating assign should propagate to all blocks.
  VLocs[0].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  ASSERT_EQ(Output[3].size(), 1ul);
  ASSERT_EQ(Output[4].size(), 1ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.getDbgOpID(0), LiveInRspID);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.getDbgOpID(0), LiveInRspID);
  EXPECT_EQ(Output[3][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[3][0].second.getDbgOpID(0), LiveInRspID);
  EXPECT_EQ(Output[4][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[4][0].second.getDbgOpID(0), LiveInRspID);
  ClearOutputs();
  VLocs[0].Vars.clear();

  // Test that an assign in the inner loop causes unresolved PHIs at the heads
  // of both loops, and no output location. Dominated blocks do get values.
  VLocs[0].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  VLocs[2].Vars.insert({VarID, DbgValue(LiveInRaxID, EmptyProps)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 0ul);
  EXPECT_EQ(Output[2].size(), 0ul);
  ASSERT_EQ(Output[3].size(), 1ul);
  ASSERT_EQ(Output[4].size(), 1ul);
  EXPECT_EQ(Output[3][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[3][0].second.getDbgOpID(0), LiveInRaxID);
  EXPECT_EQ(Output[4][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[4][0].second.getDbgOpID(0), LiveInRaxID);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[2].Vars.clear();

  // Same test, but with no assignment in block 0. We should still get values
  // in dominated blocks.
  VLocs[2].Vars.insert({VarID, DbgValue(LiveInRaxID, EmptyProps)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 0ul);
  EXPECT_EQ(Output[2].size(), 0ul);
  ASSERT_EQ(Output[3].size(), 1ul);
  ASSERT_EQ(Output[4].size(), 1ul);
  EXPECT_EQ(Output[3][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[3][0].second.getDbgOpID(0), LiveInRaxID);
  EXPECT_EQ(Output[4][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[4][0].second.getDbgOpID(0), LiveInRaxID);
  ClearOutputs();
  VLocs[2].Vars.clear();

  // Similarly, assignments in the outer loop gives location to dominated
  // blocks, but no PHI locations are found at the outer loop head.
  VLocs[0].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  VLocs[3].Vars.insert({VarID, DbgValue(LiveInRaxID, EmptyProps)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 0ul);
  EXPECT_EQ(Output[2].size(), 0ul);
  EXPECT_EQ(Output[3].size(), 0ul);
  ASSERT_EQ(Output[4].size(), 1ul);
  EXPECT_EQ(Output[4][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[4][0].second.getDbgOpID(0), LiveInRaxID);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[3].Vars.clear();

  VLocs[0].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  VLocs[1].Vars.insert({VarID, DbgValue(LiveInRaxID, EmptyProps)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 0ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  ASSERT_EQ(Output[3].size(), 1ul);
  ASSERT_EQ(Output[4].size(), 1ul);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.getDbgOpID(0), LiveInRaxID);
  EXPECT_EQ(Output[3][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[3][0].second.getDbgOpID(0), LiveInRaxID);
  EXPECT_EQ(Output[4][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[4][0].second.getDbgOpID(0), LiveInRaxID);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[1].Vars.clear();

  // With an assignment of the same value in the inner loop, we should work out
  // that all PHIs can be eliminated and the same value is live-through the
  // whole function.
  VLocs[0].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  VLocs[2].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 1ul);
  EXPECT_EQ(Output[2].size(), 1ul);
  ASSERT_EQ(Output[3].size(), 1ul);
  ASSERT_EQ(Output[4].size(), 1ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.getDbgOpID(0), LiveInRspID);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.getDbgOpID(0), LiveInRspID);
  EXPECT_EQ(Output[3][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[3][0].second.getDbgOpID(0), LiveInRspID);
  EXPECT_EQ(Output[4][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[4][0].second.getDbgOpID(0), LiveInRspID);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[2].Vars.clear();

  // If we have an assignment in the inner loop, and a PHI for it at the inner
  // loop head, we could find a live-in location for the inner loop. But because
  // the outer loop has no PHI, we can't find a variable value for outer loop
  // head, so can't have a live-in value for the inner loop head.
  MInLocs[2][0] = RspPHIInBlk2;
  MOutLocs[2][0] = LiveInRax;
  // NB: all other machine locations are LiveInRsp, disallowing a PHI in block
  // one. Even though RspPHIInBlk2 isn't available later in the function, we
  // should still produce a live-in value. The fact it's unavailable is a
  // different concern.
  VLocs[0].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  VLocs[2].Vars.insert({VarID, DbgValue(LiveInRaxID, EmptyProps)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  EXPECT_EQ(Output[1].size(), 0ul);
  EXPECT_EQ(Output[2].size(), 0ul);
  ASSERT_EQ(Output[3].size(), 1ul);
  ASSERT_EQ(Output[4].size(), 1ul);
  EXPECT_EQ(Output[3][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[3][0].second.getDbgOpID(0), LiveInRaxID);
  EXPECT_EQ(Output[4][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[4][0].second.getDbgOpID(0), LiveInRaxID);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[2].Vars.clear();

  // Have an assignment in inner loop that can have a PHI resolved; and add a
  // machine value PHI to the outer loop head, so that we can find a location
  // all the way through the function.
  MInLocs[1][0] = RspPHIInBlk1;
  MOutLocs[1][0] = RspPHIInBlk1;
  MInLocs[2][0] = RspPHIInBlk2;
  MOutLocs[2][0] = RspDefInBlk2;
  MInLocs[3][0] = RspDefInBlk2;
  MOutLocs[3][0] = RspDefInBlk2;
  VLocs[0].Vars.insert({VarID, DbgValue(LiveInRspID, EmptyProps)});
  VLocs[2].Vars.insert({VarID, DbgValue(RspDefInBlk2ID, EmptyProps)});
  buildVLocValueMap(OutermostLoc, AllVars, AssignBlocks, Output,
                    MOutLocs, MInLocs, VLocs);
  EXPECT_EQ(Output[0].size(), 0ul);
  ASSERT_EQ(Output[1].size(), 1ul);
  ASSERT_EQ(Output[2].size(), 1ul);
  ASSERT_EQ(Output[3].size(), 1ul);
  ASSERT_EQ(Output[4].size(), 1ul);
  EXPECT_EQ(Output[1][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[1][0].second.getDbgOpID(0), RspPHIInBlk1ID);
  EXPECT_EQ(Output[2][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[2][0].second.getDbgOpID(0), RspPHIInBlk2ID);
  EXPECT_EQ(Output[3][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[3][0].second.getDbgOpID(0), RspDefInBlk2ID);
  EXPECT_EQ(Output[4][0].second.Kind, DbgValue::Def);
  EXPECT_EQ(Output[4][0].second.getDbgOpID(0), RspDefInBlk2ID);
  ClearOutputs();
  VLocs[0].Vars.clear();
  VLocs[2].Vars.clear();
}
