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
  LLVMContext Ctx;
  Module Mod;
  std::unique_ptr<MachineFunction> MF;
  DICompileUnit *OurCU;
  DIFile *OurFile;
  DISubprogram *OurFunc;
  DILexicalBlock *OurBlock, *AnotherBlock;
  DISubprogram *ToInlineFunc;
  DILexicalBlock *ToInlineBlock;

  DebugLoc OutermostLoc, InBlockLoc, NotNestedBlockLoc, InlinedLoc;

  MachineBasicBlock *MBB1, *MBB2, *MBB3, *MBB4;

  InstrRefLDVTest() : Ctx(), Mod("beehives", Ctx) {
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
    auto OurSubT = DIB.createSubroutineType(DIB.getOrCreateTypeArray(None));
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
