//===- MachineModuleInfoTest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "gtest/gtest.h"

#include <memory>

using namespace llvm;

namespace {

class MachineModuleInfoTest : public testing::Test {
protected:
  LLVMContext Ctx;
  std::unique_ptr<Module> M;
  std::unique_ptr<TargetMachine> TM;

  void SetUp() override {
    InitializeAllTargets();
    InitializeAllTargetMCs();

    M = std::make_unique<Module>("test", Ctx);

    Triple TargetTriple("x86_64--");
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget("", TargetTriple, Error);
    if (!T)
      GTEST_SKIP();

    TargetOptions Options;
    TM.reset(T->createTargetMachine(TargetTriple, "", "", Options, std::nullopt,
                                    std::nullopt, CodeGenOptLevel::None));
    M->setDataLayout(TM->createDataLayout());
  }

  Function *createEmptyFunction(StringRef Name) {
    auto *Ty = FunctionType::get(Type::getVoidTy(Ctx), false);
    return Function::Create(Ty, GlobalValue::ExternalLinkage, Name, M.get());
  }
};

// Deleting an ungrouped function erases its MachineFunction immediately.
TEST_F(MachineModuleInfoTest, DeleteUngroupedMachineFunction) {
  MachineModuleInfo MMI(TM.get());

  Function *F = createEmptyFunction("f");
  MMI.getOrCreateMachineFunction(*F);
  ASSERT_NE(MMI.getMachineFunction(*F), nullptr);

  MMI.deleteMachineFunctionFor(*F);
  EXPECT_EQ(MMI.getMachineFunction(*F), nullptr);
}

// When two functions are grouped for deletion, their MachineFunctions must not
// be erased until every function in the group has been finalized.
TEST_F(MachineModuleInfoTest, DeleteGroupedMachineFunctions) {
  MachineModuleInfo MMI(TM.get());

  Function *F1 = createEmptyFunction("f1");
  Function *F2 = createEmptyFunction("f2");
  MMI.getOrCreateMachineFunction(*F1);
  MMI.getOrCreateMachineFunction(*F2);
  ASSERT_NE(MMI.getMachineFunction(*F1), nullptr);
  ASSERT_NE(MMI.getMachineFunction(*F2), nullptr);

  MMI.groupMachineFunctionsForDeletion(*F1, *F2);

  // Finalizing only the first grouped function must defer erasure of both
  // MachineFunctions.
  MMI.deleteMachineFunctionFor(*F1);
  EXPECT_NE(MMI.getMachineFunction(*F1), nullptr);
  EXPECT_NE(MMI.getMachineFunction(*F2), nullptr);

  // Once the last grouped function is finalized, all grouped MachineFunctions
  // are erased together.
  MMI.deleteMachineFunctionFor(*F2);
  EXPECT_EQ(MMI.getMachineFunction(*F1), nullptr);
  EXPECT_EQ(MMI.getMachineFunction(*F2), nullptr);
}

// A group of three functions is only erased once all three are finalized,
// regardless of the order in which they are deleted.
TEST_F(MachineModuleInfoTest, DeleteGroupedMachineFunctionsThreeMembers) {
  MachineModuleInfo MMI(TM.get());

  Function *F1 = createEmptyFunction("f1");
  Function *F2 = createEmptyFunction("f2");
  Function *F3 = createEmptyFunction("f3");
  Function *F4 = createEmptyFunction("f4");
  MMI.getOrCreateMachineFunction(*F1);
  MMI.getOrCreateMachineFunction(*F2);
  MMI.getOrCreateMachineFunction(*F3);
  MMI.getOrCreateMachineFunction(*F4);

  // Build a single group {F1, F2, F3} out of two pairwise groupings.
  MMI.groupMachineFunctionsForDeletion(*F1, *F2);
  MMI.groupMachineFunctionsForDeletion(*F2, *F3);

  MMI.deleteMachineFunctionFor(*F3);
  EXPECT_NE(MMI.getMachineFunction(*F1), nullptr);
  EXPECT_NE(MMI.getMachineFunction(*F2), nullptr);
  EXPECT_NE(MMI.getMachineFunction(*F3), nullptr);

  MMI.deleteMachineFunctionFor(*F4);
  EXPECT_EQ(MMI.getMachineFunction(*F4), nullptr);

  MMI.deleteMachineFunctionFor(*F1);
  EXPECT_NE(MMI.getMachineFunction(*F1), nullptr);
  EXPECT_NE(MMI.getMachineFunction(*F2), nullptr);
  EXPECT_NE(MMI.getMachineFunction(*F3), nullptr);

  MMI.deleteMachineFunctionFor(*F2);
  EXPECT_EQ(MMI.getMachineFunction(*F1), nullptr);
  EXPECT_EQ(MMI.getMachineFunction(*F2), nullptr);
  EXPECT_EQ(MMI.getMachineFunction(*F3), nullptr);
}

} // namespace
