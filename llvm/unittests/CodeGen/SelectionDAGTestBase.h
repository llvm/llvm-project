//===---- llvm/unittest/CodeGen/SelectionDAGTestBase.h --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

using namespace llvm;

class SelectionDAGTestBase : public testing::Test {
protected:
  static void SetUpTestCase() {
    InitializeAllTargets();
    InitializeAllTargetMCs();
  }

  void SetUp() override {
    StringRef Assembly = "@g = global i32 0\n"
                         "@g_alias = alias i32, i32* @g\n"
                         "define i32 @f() {\n"
                         "  %1 = load i32, i32* @g\n"
                         "  ret i32 %1\n"
                         "}";

    Triple TargetTriple("aarch64--");
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget("", TargetTriple, Error);
    // FIXME: These tests do not depend on AArch64 specifically, but we have to
    // initialize a target. A skeleton Target for unittests would allow us to
    // always run these tests.
    if (!T)
      GTEST_SKIP();

    TargetOptions Options;
    TM = std::unique_ptr<TargetMachine>(
        T->createTargetMachine(TargetTriple, "", "+sve", Options, std::nullopt,
                               std::nullopt, CodeGenOptLevel::Aggressive));
    if (!TM)
      GTEST_SKIP();

    SMDiagnostic SMError;
    M = parseAssemblyString(Assembly, SMError, Context);
    ASSERT_TRUE(M && "Could not parse module!");
    M->setDataLayout(TM->createDataLayout());

    F = M->getFunction("f");
    ASSERT_TRUE(F && "Could not get function f!");
    G = M->getGlobalVariable("g");
    ASSERT_TRUE(G && "Could not get global g!");
    AliasedG = M->getNamedAlias("g_alias");
    ASSERT_TRUE(AliasedG && "Could not get alias g_alias!");

    MachineModuleInfo MMI(TM.get());

    MF = std::make_unique<MachineFunction>(*F, *TM, *TM->getSubtargetImpl(*F),
                                           MMI.getContext(), 0);

    DAG = std::make_unique<SelectionDAG>(*TM, CodeGenOptLevel::None);
    if (!DAG)
      reportFatalUsageError("Failed to create SelectionDAG?");
    OptimizationRemarkEmitter ORE(F);
    DAG->init(*MF, ORE, nullptr, nullptr, nullptr, nullptr, nullptr, MMI,
              nullptr);
  }

  TargetLoweringBase::LegalizeTypeAction getTypeAction(EVT VT) {
    return DAG->getTargetLoweringInfo().getTypeAction(Context, VT);
  }

  EVT getTypeToTransformTo(EVT VT) {
    return DAG->getTargetLoweringInfo().getTypeToTransformTo(Context, VT);
  }

  LLVMContext Context;
  std::unique_ptr<TargetMachine> TM;
  std::unique_ptr<Module> M;
  Function *F;
  GlobalVariable *G;
  GlobalAlias *AliasedG;
  std::unique_ptr<MachineFunction> MF;
  std::unique_ptr<SelectionDAG> DAG;
};
