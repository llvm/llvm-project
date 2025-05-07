//===----------- llvm/unittest/CodeGen/LexicalScopesTest.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/LexicalScopes.h"
#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetOptions.h"

#include "gtest/gtest.h"

using namespace llvm;

namespace {
// Include helper functions to ease the manipulation of MachineFunctions
#include "MFCommon.inc"

class LexicalScopesTest : public testing::Test {
public:
  // Boilerplate,
  LLVMContext Ctx;
  Module Mod;
  std::unique_ptr<MachineFunction> MF;
  DICompileUnit *OurCU;
  DIFile *OurFile;
  DISubprogram *OurFunc;
  DILexicalBlock *OurBlock, *AnotherBlock;
  DISubprogram *ToInlineFunc;
  DILexicalBlock *ToInlineBlock;
  // DebugLocs that we'll used to create test environments.
  DebugLoc OutermostLoc, InBlockLoc, NotNestedBlockLoc, InlinedLoc;

  // Test environment blocks -- these form a diamond control flow pattern,
  // MBB1 being the entry block, blocks two and three being the branches, and
  // block four joining the branches and being an exit block.
  MachineBasicBlock *MBB1, *MBB2, *MBB3, *MBB4;

  // Some meaningless instructions -- the first is fully meaningless,
  // while the second is supposed to impersonate DBG_VALUEs through its
  // opcode.
  MCInstrDesc BeanInst{};
  MCInstrDesc DbgValueInst{};

  LexicalScopesTest() : Ctx(), Mod("beehives", Ctx) {
    memset(&BeanInst, 0, sizeof(BeanInst));
    BeanInst.Opcode = 1;
    BeanInst.Size = 1;

    memset(&DbgValueInst, 0, sizeof(DbgValueInst));
    DbgValueInst.Opcode = TargetOpcode::DBG_VALUE;
    DbgValueInst.Size = 1;
    DbgValueInst.Flags = 1U << MCID::Meta;

    // Boilerplate that creates a MachineFunction and associated blocks.
    MF = createMachineFunction(Ctx, Mod);
    llvm::Function &F = const_cast<llvm::Function &>(MF->getFunction());
    auto BB1 = BasicBlock::Create(Ctx, "a", &F);
    auto BB2 = BasicBlock::Create(Ctx, "b", &F);
    auto BB3 = BasicBlock::Create(Ctx, "c", &F);
    auto BB4 = BasicBlock::Create(Ctx, "d", &F);
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

    // Create metadata: CU, subprogram, some blocks and an inline function
    // scope.
    DIBuilder DIB(Mod);
    OurFile = DIB.createFile("xyzzy.c", "/cave");
    OurCU =
        DIB.createCompileUnit(dwarf::DW_LANG_C99, OurFile, "nou", false, "", 0);
    auto OurSubT = DIB.createSubroutineType(DIB.getOrCreateTypeArray({}));
    OurFunc =
        DIB.createFunction(OurCU, "bees", "", OurFile, 1, OurSubT, 1,
                           DINode::FlagZero, DISubprogram::SPFlagDefinition);
    F.setSubprogram(OurFunc);
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
};

// Fill blocks with dummy instructions, test some base lexical scope
// functionaliy.
TEST_F(LexicalScopesTest, FlatLayout) {
  BuildMI(*MBB1, MBB1->end(), OutermostLoc, BeanInst);
  BuildMI(*MBB2, MBB2->end(), OutermostLoc, BeanInst);
  BuildMI(*MBB3, MBB3->end(), OutermostLoc, BeanInst);
  BuildMI(*MBB4, MBB4->end(), OutermostLoc, BeanInst);

  LexicalScopes LS;
  EXPECT_TRUE(LS.empty());
  LS.reset();
  EXPECT_EQ(LS.getCurrentFunctionScope(), nullptr);

  LS.initialize(*MF);
  EXPECT_FALSE(LS.empty());
  LexicalScope *FuncScope = LS.getCurrentFunctionScope();
  EXPECT_EQ(FuncScope->getParent(), nullptr);
  EXPECT_EQ(FuncScope->getDesc(), OurFunc);
  EXPECT_EQ(FuncScope->getInlinedAt(), nullptr);
  EXPECT_EQ(FuncScope->getScopeNode(), OurFunc);
  EXPECT_FALSE(FuncScope->isAbstractScope());
  EXPECT_EQ(FuncScope->getChildren().size(), 0u);

  // There should be one range, covering the whole function. Test that it
  // points at the correct instructions.
  auto &Ranges = FuncScope->getRanges();
  ASSERT_EQ(Ranges.size(), 1u);
  EXPECT_EQ(Ranges.front().first, &*MF->begin()->begin());
  auto BBIt = MF->end();
  BBIt = std::prev(BBIt);
  EXPECT_EQ(Ranges.front().second, &*BBIt->begin());

  EXPECT_TRUE(FuncScope->dominates(FuncScope));
  SmallPtrSet<const MachineBasicBlock *, 4> MBBVec;
  LS.getMachineBasicBlocks(OutermostLoc.get(), MBBVec);

  EXPECT_EQ(MBBVec.size(), 4u);
  // All the blocks should be in that set; the outermost loc should dominate
  // them; and no other scope should.
  for (auto &MBB : *MF) {
    EXPECT_EQ(MBBVec.count(&MBB), 1u);
    EXPECT_TRUE(LS.dominates(OutermostLoc.get(), &MBB));
    EXPECT_FALSE(LS.dominates(InBlockLoc.get(), &MBB));
    EXPECT_FALSE(LS.dominates(InlinedLoc.get(), &MBB));
  }
}

// Examine relationship between two nested scopes inside the function, the
// outer function and the lexical block within it.
TEST_F(LexicalScopesTest, BlockScopes) {
  BuildMI(*MBB1, MBB1->end(), InBlockLoc, BeanInst);
  BuildMI(*MBB2, MBB2->end(), InBlockLoc, BeanInst);
  BuildMI(*MBB3, MBB3->end(), InBlockLoc, BeanInst);
  BuildMI(*MBB4, MBB4->end(), InBlockLoc, BeanInst);

  LexicalScopes LS;
  LS.initialize(*MF);
  LexicalScope *FuncScope = LS.getCurrentFunctionScope();
  EXPECT_EQ(FuncScope->getDesc(), OurFunc);
  auto &Children = FuncScope->getChildren();
  ASSERT_EQ(Children.size(), 1u);
  auto *BlockScope = Children[0];
  EXPECT_EQ(LS.findLexicalScope(InBlockLoc.get()), BlockScope);
  EXPECT_EQ(BlockScope->getDesc(), InBlockLoc->getScope());
  EXPECT_FALSE(BlockScope->isAbstractScope());

  EXPECT_TRUE(FuncScope->dominates(BlockScope));
  EXPECT_FALSE(BlockScope->dominates(FuncScope));
  EXPECT_EQ(FuncScope->getParent(), nullptr);
  EXPECT_EQ(BlockScope->getParent(), FuncScope);

  SmallPtrSet<const MachineBasicBlock *, 4> MBBVec;
  LS.getMachineBasicBlocks(OutermostLoc.get(), MBBVec);

  EXPECT_EQ(MBBVec.size(), 4u);
  for (auto &MBB : *MF) {
    EXPECT_EQ(MBBVec.count(&MBB), 1u);
    EXPECT_TRUE(LS.dominates(OutermostLoc.get(), &MBB));
    EXPECT_TRUE(LS.dominates(InBlockLoc.get(), &MBB));
    EXPECT_FALSE(LS.dominates(InlinedLoc.get(), &MBB));
  }
}

// Test inlined scopes functionality and relationship with the outer scopes.
TEST_F(LexicalScopesTest, InlinedScopes) {
  BuildMI(*MBB1, MBB1->end(), InlinedLoc, BeanInst);
  BuildMI(*MBB2, MBB2->end(), InlinedLoc, BeanInst);
  BuildMI(*MBB3, MBB3->end(), InlinedLoc, BeanInst);
  BuildMI(*MBB4, MBB4->end(), InlinedLoc, BeanInst);

  LexicalScopes LS;
  LS.initialize(*MF);
  LexicalScope *FuncScope = LS.getCurrentFunctionScope();
  auto &Children = FuncScope->getChildren();
  ASSERT_EQ(Children.size(), 1u);
  auto *BlockScope = Children[0];
  auto &BlockChildren = BlockScope->getChildren();
  ASSERT_EQ(BlockChildren.size(), 1u);
  auto *InlinedScope = BlockChildren[0];

  EXPECT_FALSE(InlinedScope->isAbstractScope());
  EXPECT_EQ(InlinedScope->getInlinedAt(), InlinedLoc.getInlinedAt());
  EXPECT_EQ(InlinedScope->getDesc(), InlinedLoc.getScope());
  EXPECT_EQ(InlinedScope->getChildren().size(), 0u);

  EXPECT_EQ(FuncScope->getParent(), nullptr);
  EXPECT_EQ(BlockScope->getParent(), FuncScope);
  EXPECT_EQ(InlinedScope->getParent(), BlockScope);

  const auto &AbstractScopes = LS.getAbstractScopesList();
  ASSERT_EQ(AbstractScopes.size(), 1u);
  const auto &AbstractScope = *AbstractScopes[0];
  EXPECT_TRUE(AbstractScope.isAbstractScope());
  EXPECT_EQ(AbstractScope.getDesc(), InlinedLoc.getScope());
  EXPECT_EQ(AbstractScope.getInlinedAt(), nullptr);
  EXPECT_EQ(AbstractScope.getParent(), nullptr);
}

// Test behaviour in a function that has empty DebugLocs.
TEST_F(LexicalScopesTest, FuncWithEmptyGap) {
  BuildMI(*MBB1, MBB1->end(), OutermostLoc, BeanInst);
  BuildMI(*MBB2, MBB2->end(), DebugLoc(), BeanInst);
  BuildMI(*MBB3, MBB3->end(), DebugLoc(), BeanInst);
  BuildMI(*MBB4, MBB4->end(), OutermostLoc, BeanInst);

  LexicalScopes LS;
  LS.initialize(*MF);
  LexicalScope *FuncScope = LS.getCurrentFunctionScope();

  // A gap in a range that contains no other location, is not actually a
  // gap as far as lexical scopes are concerned.
  auto &Ranges = FuncScope->getRanges();
  ASSERT_EQ(Ranges.size(), 1u);
  EXPECT_EQ(Ranges[0].first, &*MF->begin()->begin());
  auto BBIt = MF->end();
  BBIt = std::prev(BBIt);
  EXPECT_EQ(Ranges[0].second, &*BBIt->begin());
}

// Now a function with intervening not-in-scope instructions.
TEST_F(LexicalScopesTest, FuncWithRealGap) {
  MachineInstr *FirstI = BuildMI(*MBB1, MBB1->end(), InBlockLoc, BeanInst);
  BuildMI(*MBB2, MBB2->end(), OutermostLoc, BeanInst);
  BuildMI(*MBB3, MBB3->end(), OutermostLoc, BeanInst);
  MachineInstr *LastI = BuildMI(*MBB4, MBB4->end(), InBlockLoc, BeanInst);

  LexicalScopes LS;
  LS.initialize(*MF);
  LexicalScope *BlockScope = LS.findLexicalScope(InBlockLoc.get());
  ASSERT_NE(BlockScope, nullptr);

  // Within the block scope, there's a gap between the first and last
  // block / instruction, where it's only the outermost scope.
  auto &Ranges = BlockScope->getRanges();
  ASSERT_EQ(Ranges.size(), 2u);
  EXPECT_EQ(Ranges[0].first, FirstI);
  EXPECT_EQ(Ranges[0].second, FirstI);
  EXPECT_EQ(Ranges[1].first, LastI);
  EXPECT_EQ(Ranges[1].second, LastI);

  // The outer function scope should cover the whole function, including
  // blocks the lexicalblock covers.
  LexicalScope *FuncScope = LS.getCurrentFunctionScope();
  auto &FuncRanges = FuncScope->getRanges();
  ASSERT_EQ(FuncRanges.size(), 1u);
  EXPECT_NE(FuncRanges[0].first, FuncRanges[0].second);
  EXPECT_EQ(FuncRanges[0].first, FirstI);
  EXPECT_EQ(FuncRanges[0].second, LastI);
}

// Examine the relationship between two scopes that don't nest (are siblings).
TEST_F(LexicalScopesTest, NotNested) {
  MachineInstr *FirstI = BuildMI(*MBB1, MBB1->end(), InBlockLoc, BeanInst);
  MachineInstr *SecondI =
      BuildMI(*MBB2, MBB2->end(), NotNestedBlockLoc, BeanInst);
  MachineInstr *ThirdI =
      BuildMI(*MBB3, MBB3->end(), NotNestedBlockLoc, BeanInst);
  MachineInstr *FourthI = BuildMI(*MBB4, MBB4->end(), InBlockLoc, BeanInst);

  LexicalScopes LS;
  LS.initialize(*MF);
  LexicalScope *FuncScope = LS.getCurrentFunctionScope();
  LexicalScope *BlockScope = LS.findLexicalScope(InBlockLoc.get());
  LexicalScope *OtherBlockScope = LS.findLexicalScope(NotNestedBlockLoc.get());
  ASSERT_NE(FuncScope, nullptr);
  ASSERT_NE(BlockScope, nullptr);
  ASSERT_NE(OtherBlockScope, nullptr);

  // The function should cover everything; the two blocks are distinct and
  // should not.
  auto &FuncRanges = FuncScope->getRanges();
  ASSERT_EQ(FuncRanges.size(), 1u);
  EXPECT_EQ(FuncRanges[0].first, FirstI);
  EXPECT_EQ(FuncRanges[0].second, FourthI);

  // Two ranges, start and end instructions.
  auto &BlockRanges = BlockScope->getRanges();
  ASSERT_EQ(BlockRanges.size(), 2u);
  EXPECT_EQ(BlockRanges[0].first, FirstI);
  EXPECT_EQ(BlockRanges[0].second, FirstI);
  EXPECT_EQ(BlockRanges[1].first, FourthI);
  EXPECT_EQ(BlockRanges[1].second, FourthI);

  // One inner range, covering the two inner blocks.
  auto &OtherBlockRanges = OtherBlockScope->getRanges();
  ASSERT_EQ(OtherBlockRanges.size(), 1u);
  EXPECT_EQ(OtherBlockRanges[0].first, SecondI);
  EXPECT_EQ(OtherBlockRanges[0].second, ThirdI);
}

// Test the scope-specific and block-specific dominates methods.
TEST_F(LexicalScopesTest, TestDominates) {
  BuildMI(*MBB1, MBB1->end(), InBlockLoc, BeanInst);
  BuildMI(*MBB2, MBB2->end(), NotNestedBlockLoc, BeanInst);
  BuildMI(*MBB3, MBB3->end(), NotNestedBlockLoc, BeanInst);
  BuildMI(*MBB4, MBB4->end(), InBlockLoc, BeanInst);

  LexicalScopes LS;
  LS.initialize(*MF);
  LexicalScope *FuncScope = LS.getCurrentFunctionScope();
  LexicalScope *BlockScope = LS.findLexicalScope(InBlockLoc.get());
  LexicalScope *OtherBlockScope = LS.findLexicalScope(NotNestedBlockLoc.get());
  ASSERT_NE(FuncScope, nullptr);
  ASSERT_NE(BlockScope, nullptr);
  ASSERT_NE(OtherBlockScope, nullptr);

  EXPECT_TRUE(FuncScope->dominates(BlockScope));
  EXPECT_TRUE(FuncScope->dominates(OtherBlockScope));
  EXPECT_FALSE(BlockScope->dominates(FuncScope));
  EXPECT_FALSE(BlockScope->dominates(OtherBlockScope));
  EXPECT_FALSE(OtherBlockScope->dominates(FuncScope));
  EXPECT_FALSE(OtherBlockScope->dominates(BlockScope));

  // Outermost scope dominates everything, as all insts are within it.
  EXPECT_TRUE(LS.dominates(OutermostLoc.get(), MBB1));
  EXPECT_TRUE(LS.dominates(OutermostLoc.get(), MBB2));
  EXPECT_TRUE(LS.dominates(OutermostLoc.get(), MBB3));
  EXPECT_TRUE(LS.dominates(OutermostLoc.get(), MBB4));

  // One inner block dominates the outer pair of blocks,
  EXPECT_TRUE(LS.dominates(InBlockLoc.get(), MBB1));
  EXPECT_FALSE(LS.dominates(InBlockLoc.get(), MBB2));
  EXPECT_FALSE(LS.dominates(InBlockLoc.get(), MBB3));
  EXPECT_TRUE(LS.dominates(InBlockLoc.get(), MBB4));

  // While the other dominates the inner two blocks.
  EXPECT_FALSE(LS.dominates(NotNestedBlockLoc.get(), MBB1));
  EXPECT_TRUE(LS.dominates(NotNestedBlockLoc.get(), MBB2));
  EXPECT_TRUE(LS.dominates(NotNestedBlockLoc.get(), MBB3));
  EXPECT_FALSE(LS.dominates(NotNestedBlockLoc.get(), MBB4));
}

// Test getMachineBasicBlocks returns all dominated blocks.
TEST_F(LexicalScopesTest, TestGetBlocks) {
  BuildMI(*MBB1, MBB1->end(), InBlockLoc, BeanInst);
  BuildMI(*MBB2, MBB2->end(), NotNestedBlockLoc, BeanInst);
  BuildMI(*MBB3, MBB3->end(), NotNestedBlockLoc, BeanInst);
  BuildMI(*MBB4, MBB4->end(), InBlockLoc, BeanInst);

  LexicalScopes LS;
  LS.initialize(*MF);
  LexicalScope *FuncScope = LS.getCurrentFunctionScope();
  LexicalScope *BlockScope = LS.findLexicalScope(InBlockLoc.get());
  LexicalScope *OtherBlockScope = LS.findLexicalScope(NotNestedBlockLoc.get());
  ASSERT_NE(FuncScope, nullptr);
  ASSERT_NE(BlockScope, nullptr);
  ASSERT_NE(OtherBlockScope, nullptr);

  SmallPtrSet<const MachineBasicBlock *, 4> OutermostBlocks, InBlockBlocks,
      NotNestedBlockBlocks;
  LS.getMachineBasicBlocks(OutermostLoc.get(), OutermostBlocks);
  LS.getMachineBasicBlocks(InBlockLoc.get(), InBlockBlocks);
  LS.getMachineBasicBlocks(NotNestedBlockLoc.get(), NotNestedBlockBlocks);

  EXPECT_EQ(OutermostBlocks.count(MBB1), 1u);
  EXPECT_EQ(OutermostBlocks.count(MBB2), 1u);
  EXPECT_EQ(OutermostBlocks.count(MBB3), 1u);
  EXPECT_EQ(OutermostBlocks.count(MBB4), 1u);

  EXPECT_EQ(InBlockBlocks.count(MBB1), 1u);
  EXPECT_EQ(InBlockBlocks.count(MBB2), 0u);
  EXPECT_EQ(InBlockBlocks.count(MBB3), 0u);
  EXPECT_EQ(InBlockBlocks.count(MBB4), 1u);

  EXPECT_EQ(NotNestedBlockBlocks.count(MBB1), 0u);
  EXPECT_EQ(NotNestedBlockBlocks.count(MBB2), 1u);
  EXPECT_EQ(NotNestedBlockBlocks.count(MBB3), 1u);
  EXPECT_EQ(NotNestedBlockBlocks.count(MBB4), 0u);
}

TEST_F(LexicalScopesTest, TestMetaInst) {
  // Instruction Layout looks like this, where 'F' means funcscope, and
  // 'B' blockscope:
  // bb1:
  //   F: bean
  //   B: bean
  // bb2:
  //   F: bean
  //   B: DBG_VALUE
  // bb3:
  //   F: bean
  //   B: DBG_VALUE
  // bb4:
  //   F: bean
  //   B: bean
  // The block / 'B' should only dominate bb1 and bb4. DBG_VALUE is a meta
  // instruction, and shouldn't contribute to scopes.
  BuildMI(*MBB1, MBB1->end(), OutermostLoc, BeanInst);
  BuildMI(*MBB1, MBB1->end(), InBlockLoc, BeanInst);
  BuildMI(*MBB2, MBB2->end(), OutermostLoc, BeanInst);
  BuildMI(*MBB2, MBB2->end(), InBlockLoc, DbgValueInst);
  BuildMI(*MBB3, MBB3->end(), OutermostLoc, BeanInst);
  BuildMI(*MBB3, MBB3->end(), InBlockLoc, DbgValueInst);
  BuildMI(*MBB4, MBB4->end(), OutermostLoc, BeanInst);
  BuildMI(*MBB4, MBB4->end(), InBlockLoc, BeanInst);

  LexicalScopes LS;
  LS.initialize(*MF);
  LexicalScope *FuncScope = LS.getCurrentFunctionScope();
  LexicalScope *BlockScope = LS.findLexicalScope(InBlockLoc.get());
  ASSERT_NE(FuncScope, nullptr);
  ASSERT_NE(BlockScope, nullptr);

  EXPECT_TRUE(LS.dominates(OutermostLoc.get(), MBB1));
  EXPECT_TRUE(LS.dominates(OutermostLoc.get(), MBB2));
  EXPECT_TRUE(LS.dominates(OutermostLoc.get(), MBB3));
  EXPECT_TRUE(LS.dominates(OutermostLoc.get(), MBB4));
  EXPECT_TRUE(LS.dominates(InBlockLoc.get(), MBB1));
  EXPECT_FALSE(LS.dominates(InBlockLoc.get(), MBB2));
  EXPECT_FALSE(LS.dominates(InBlockLoc.get(), MBB3));
  EXPECT_TRUE(LS.dominates(InBlockLoc.get(), MBB4));
}

} // anonymous namespace
