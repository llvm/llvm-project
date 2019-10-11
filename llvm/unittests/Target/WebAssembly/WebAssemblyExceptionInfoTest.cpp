//=== WebAssemblyExceptionInfoTest.cpp - WebAssebmlyExceptionInfo unit tests =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "WebAssemblyExceptionInfo.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineDominanceFrontier.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

std::unique_ptr<LLVMTargetMachine> createTargetMachine() {
  auto TT(Triple::normalize("wasm32-unknown-unknown"));
  std::string CPU("");
  std::string FS("");

  LLVMInitializeWebAssemblyTargetInfo();
  LLVMInitializeWebAssemblyTarget();
  LLVMInitializeWebAssemblyTargetMC();

  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);
  assert(TheTarget);

  return std::unique_ptr<LLVMTargetMachine>(static_cast<LLVMTargetMachine*>(
      TheTarget->createTargetMachine(TT, CPU, FS, TargetOptions(), None, None,
                                     CodeGenOpt::Default)));
}

std::unique_ptr<Module> parseMIR(LLVMContext &Context,
                                 std::unique_ptr<MIRParser> &MIR,
                                 const TargetMachine &TM, StringRef MIRCode,
                                 const char *FuncName, MachineModuleInfo &MMI) {
  SMDiagnostic Diagnostic;
  std::unique_ptr<MemoryBuffer> MBuffer = MemoryBuffer::getMemBuffer(MIRCode);
  MIR = createMIRParser(std::move(MBuffer), Context);
  if (!MIR)
    return nullptr;

  std::unique_ptr<Module> M = MIR->parseIRModule();
  if (!M)
    return nullptr;

  M->setDataLayout(TM.createDataLayout());

  if (MIR->parseMachineFunctions(*M, MMI))
    return nullptr;

  return M;
}

} // namespace

TEST(WebAssemblyExceptionInfoTest, TEST0) {
  std::unique_ptr<LLVMTargetMachine> TM = createTargetMachine();
  ASSERT_TRUE(TM);

  StringRef MIRString = R"MIR(
--- |
  target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
  target triple = "wasm32-unknown-unknown"

  declare i32 @__gxx_wasm_personality_v0(...)

  define void @test0() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
    unreachable
  }

...
---
name: test0
liveins:
  - { reg: '$arguments' }
  - { reg: '$value_stack' }
body: |
  bb.0:
    successors: %bb.1, %bb.2
    liveins: $arguments, $value_stack
    BR %bb.1, implicit-def dead $arguments

  bb.1:
  ; predecessors: %bb.0
    successors: %bb.7
    liveins: $value_stack
    BR %bb.7, implicit-def $arguments

  bb.2 (landing-pad):
  ; predecessors: %bb.0
    successors: %bb.3, %bb.9
    liveins: $value_stack
    %0:exnref = CATCH implicit-def $arguments
    CLEANUPRET implicit-def dead $arguments

  bb.3 (landing-pad):
  ; predecessors: %bb.2
    successors: %bb.4, %bb.6
    liveins: $value_stack
    %1:exnref = CATCH implicit-def $arguments
    BR_IF %bb.4, %58:i32, implicit-def $arguments, implicit-def $value_stack, implicit $value_stack
    BR %bb.6, implicit-def $arguments

  bb.4:
  ; predecessors: %bb.3
    successors: %bb.5, %bb.8
    liveins: $value_stack
    BR %bb.5, implicit-def dead $arguments

  bb.5:
  ; predecessors: %bb.4
    successors: %bb.7
    liveins: $value_stack
    CATCHRET %bb.7, %bb.0, implicit-def dead $arguments

  bb.6:
  ; predecessors: %bb.3
    successors: %bb.10, %bb.9
    liveins: $value_stack
    BR %bb.10, implicit-def dead $arguments

  bb.7:
  ; predecessors: %bb.5, %bb.1
    liveins: $value_stack
    RETURN implicit-def $arguments

  bb.8 (landing-pad):
  ; predecessors: %bb.4
    successors: %bb.9
    liveins: $value_stack
    %2:exnref = CATCH implicit-def $arguments
    CLEANUPRET implicit-def dead $arguments

  bb.9 (landing-pad):
  ; predecessors: %bb.2, %bb.6, %bb.8
    liveins: $value_stack
    %3:exnref = CATCH implicit-def $arguments
    CLEANUPRET implicit-def dead $arguments

  bb.10:
  ; predecessors: %bb.6
    liveins: $value_stack
    UNREACHABLE implicit-def $arguments
)MIR";

  LLVMContext Context;
  std::unique_ptr<MIRParser> MIR;
  MachineModuleInfo MMI(TM.get());
  std::unique_ptr<Module> M =
      parseMIR(Context, MIR, *TM, MIRString, "test0", MMI);
  ASSERT_TRUE(M);

  Function *F = M->getFunction("test0");
  auto *MF = MMI.getMachineFunction(*F);
  ASSERT_TRUE(MF);

  WebAssemblyExceptionInfo WEI;
  MachineDominatorTree MDT;
  MachineDominanceFrontier MDF;
  MDT.runOnMachineFunction(*MF);
  MDF.getBase().analyze(MDT.getBase());
  WEI.recalculate(MDT, MDF);

  // Exception info structure:
  // |- bb2 (ehpad), bb3, bb4, bb5, bb6, bb8, bb9, bb10
  //   |- bb3 (ehpad), bb4, bb5, bb6, bb8, bb10
  //     |- bb8 (ehpad)
  //   |- bb9 (ehpad)

  auto *MBB2 = MF->getBlockNumbered(2);
  auto *WE0 = WEI.getExceptionFor(MBB2);
  ASSERT_TRUE(WE0);
  EXPECT_EQ(WE0->getEHPad(), MBB2);
  EXPECT_EQ(WE0->getParentException(), nullptr);
  EXPECT_EQ(WE0->getExceptionDepth(), (unsigned)1);

  auto *MBB3 = MF->getBlockNumbered(3);
  auto *WE0_0 = WEI.getExceptionFor(MBB3);
  ASSERT_TRUE(WE0_0);
  EXPECT_EQ(WE0_0->getEHPad(), MBB3);
  EXPECT_EQ(WE0_0->getParentException(), WE0);
  EXPECT_EQ(WE0_0->getExceptionDepth(), (unsigned)2);

  auto *MBB4 = MF->getBlockNumbered(4);
  WE0_0 = WEI.getExceptionFor(MBB4);
  ASSERT_TRUE(WE0_0);
  EXPECT_EQ(WE0_0->getEHPad(), MBB3);

  auto *MBB5 = MF->getBlockNumbered(5);
  WE0_0 = WEI.getExceptionFor(MBB5);
  ASSERT_TRUE(WE0_0);
  EXPECT_EQ(WE0_0->getEHPad(), MBB3);

  auto *MBB6 = MF->getBlockNumbered(6);
  WE0_0 = WEI.getExceptionFor(MBB6);
  ASSERT_TRUE(WE0_0);
  EXPECT_EQ(WE0_0->getEHPad(), MBB3);

  auto *MBB10 = MF->getBlockNumbered(10);
  WE0_0 = WEI.getExceptionFor(MBB10);
  ASSERT_TRUE(WE0_0);
  EXPECT_EQ(WE0_0->getEHPad(), MBB3);

  auto *MBB8 = MF->getBlockNumbered(8);
  auto *WE0_0_0 = WEI.getExceptionFor(MBB8);
  ASSERT_TRUE(WE0_0_0);
  EXPECT_EQ(WE0_0_0->getEHPad(), MBB8);
  EXPECT_EQ(WE0_0_0->getParentException(), WE0_0);
  EXPECT_EQ(WE0_0_0->getExceptionDepth(), (unsigned)3);

  auto *MBB9 = MF->getBlockNumbered(9);
  auto *WE0_1 = WEI.getExceptionFor(MBB9);
  ASSERT_TRUE(WE0_1);
  EXPECT_EQ(WE0_1->getEHPad(), MBB9);
  EXPECT_EQ(WE0_1->getParentException(), WE0);
  EXPECT_EQ(WE0_1->getExceptionDepth(), (unsigned)2);
}

TEST(WebAssemblyExceptionInfoTest, TEST1) {
  std::unique_ptr<LLVMTargetMachine> TM = createTargetMachine();
  ASSERT_TRUE(TM);

  StringRef MIRString = R"MIR(
--- |
  target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
  target triple = "wasm32-unknown-unknown"

  declare i32 @__gxx_wasm_personality_v0(...)

  define void @test1() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
    unreachable
  }

...
---
name: test1
liveins:
  - { reg: '$arguments' }
  - { reg: '$value_stack' }
body: |
  bb.0:
    successors: %bb.9, %bb.1
    liveins: $arguments, $value_stack
    BR %bb.9, implicit-def dead $arguments

  bb.1 (landing-pad):
  ; predecessors: %bb.0
    successors: %bb.2, %bb.8
    liveins: $value_stack
    %0:exnref = CATCH implicit-def $arguments
    BR_IF %bb.2, %32:i32, implicit-def $arguments, implicit-def $value_stack, implicit $value_stack
    BR %bb.8, implicit-def $arguments

  bb.2:
  ; predecessors: %bb.1
    successors: %bb.7, %bb.3, %bb.11
    liveins: $value_stack
    BR %bb.7, implicit-def dead $arguments

  bb.3 (landing-pad):
  ; predecessors: %bb.2
    successors: %bb.4, %bb.6
    liveins: $value_stack
    %1:exnref = CATCH implicit-def $arguments
    BR_IF %bb.4, %43:i32, implicit-def $arguments, implicit-def $value_stack, implicit $value_stack
    BR %bb.6, implicit-def $arguments

  bb.4:
  ; predecessors: %bb.3
    successors: %bb.5, %bb.10
    liveins: $value_stack
    BR %bb.5, implicit-def dead $arguments

  bb.5:
  ; predecessors: %bb.4
    successors: %bb.7(0x80000000); %bb.7(200.00%)
    liveins: $value_stack
    CATCHRET %bb.7, %bb.1, implicit-def dead $arguments

  bb.6:
  ; predecessors: %bb.3
    successors: %bb.12, %bb.11
    liveins: $value_stack
    BR %bb.12, implicit-def dead $arguments

  bb.7:
  ; predecessors: %bb.2, %bb.5
    successors: %bb.9(0x80000000); %bb.9(200.00%)
    liveins: $value_stack
    CATCHRET %bb.9, %bb.0, implicit-def dead $arguments

  bb.8:
  ; predecessors: %bb.1
    liveins: $value_stack
    UNREACHABLE implicit-def $arguments

  bb.9:
  ; predecessors: %bb.0, %bb.7
    liveins: $value_stack
    RETURN implicit-def $arguments

  bb.10 (landing-pad):
  ; predecessors: %bb.4
    successors: %bb.11
    liveins: $value_stack
    %2:exnref = CATCH implicit-def $arguments
    CLEANUPRET implicit-def dead $arguments

  bb.11 (landing-pad):
  ; predecessors: %bb.2, %bb.6, %bb.10
    liveins: $value_stack
    %3:exnref = CATCH implicit-def $arguments
    CLEANUPRET implicit-def dead $arguments

  bb.12:
  ; predecessors: %bb.6
    liveins: $value_stack
    UNREACHABLE implicit-def $arguments
)MIR";

  LLVMContext Context;
  std::unique_ptr<MIRParser> MIR;
  MachineModuleInfo MMI(TM.get());
  std::unique_ptr<Module> M =
      parseMIR(Context, MIR, *TM, MIRString, "test1", MMI);
  ASSERT_TRUE(M);

  Function *F = M->getFunction("test1");
  auto *MF = MMI.getMachineFunction(*F);
  ASSERT_TRUE(MF);

  WebAssemblyExceptionInfo WEI;
  MachineDominatorTree MDT;
  MachineDominanceFrontier MDF;
  MDT.runOnMachineFunction(*MF);
  MDF.getBase().analyze(MDT.getBase());
  WEI.recalculate(MDT, MDF);

  // Exception info structure:
  // |- bb1 (ehpad), bb2, bb3, bb4, bb5, bb6, bb7, bb8, bb10, bb11, bb12
  //   |- bb3 (ehpad), bb4, bb5, bb6, bb10, bb12
  //     |- bb10 (ehpad)
  //   |- bb11 (ehpad)

  auto *MBB1 = MF->getBlockNumbered(1);
  auto *WE0 = WEI.getExceptionFor(MBB1);
  ASSERT_TRUE(WE0);
  EXPECT_EQ(WE0->getEHPad(), MBB1);
  EXPECT_EQ(WE0->getParentException(), nullptr);
  EXPECT_EQ(WE0->getExceptionDepth(), (unsigned)1);

  auto *MBB2 = MF->getBlockNumbered(2);
  WE0 = WEI.getExceptionFor(MBB2);
  ASSERT_TRUE(WE0);
  EXPECT_EQ(WE0->getEHPad(), MBB1);

  auto *MBB7 = MF->getBlockNumbered(7);
  WE0 = WEI.getExceptionFor(MBB7);
  ASSERT_TRUE(WE0);
  EXPECT_EQ(WE0->getEHPad(), MBB1);

  auto *MBB8 = MF->getBlockNumbered(8);
  WE0 = WEI.getExceptionFor(MBB8);
  ASSERT_TRUE(WE0);
  EXPECT_EQ(WE0->getEHPad(), MBB1);

  auto *MBB3 = MF->getBlockNumbered(3);
  auto *WE0_0 = WEI.getExceptionFor(MBB3);
  ASSERT_TRUE(WE0_0);
  EXPECT_EQ(WE0_0->getEHPad(), MBB3);
  EXPECT_EQ(WE0_0->getParentException(), WE0);
  EXPECT_EQ(WE0_0->getExceptionDepth(), (unsigned)2);

  auto *MBB4 = MF->getBlockNumbered(4);
  WE0_0 = WEI.getExceptionFor(MBB4);
  ASSERT_TRUE(WE0_0);
  EXPECT_EQ(WE0_0->getEHPad(), MBB3);

  auto *MBB5 = MF->getBlockNumbered(5);
  WE0_0 = WEI.getExceptionFor(MBB5);
  ASSERT_TRUE(WE0_0);
  EXPECT_EQ(WE0_0->getEHPad(), MBB3);

  auto *MBB6 = MF->getBlockNumbered(6);
  WE0_0 = WEI.getExceptionFor(MBB6);
  ASSERT_TRUE(WE0_0);
  EXPECT_EQ(WE0_0->getEHPad(), MBB3);

  auto *MBB12 = MF->getBlockNumbered(12);
  WE0_0 = WEI.getExceptionFor(MBB12);
  ASSERT_TRUE(WE0_0);
  EXPECT_EQ(WE0_0->getEHPad(), MBB3);

  auto *MBB10 = MF->getBlockNumbered(10);
  auto *WE0_0_0 = WEI.getExceptionFor(MBB10);
  ASSERT_TRUE(WE0_0_0);
  EXPECT_EQ(WE0_0_0->getEHPad(), MBB10);
  EXPECT_EQ(WE0_0_0->getParentException(), WE0_0);
  EXPECT_EQ(WE0_0_0->getExceptionDepth(), (unsigned)3);

  auto *MBB11 = MF->getBlockNumbered(11);
  auto *WE0_1 = WEI.getExceptionFor(MBB11);
  ASSERT_TRUE(WE0_1);
  EXPECT_EQ(WE0_1->getEHPad(), MBB11);
  EXPECT_EQ(WE0_1->getParentException(), WE0);
  EXPECT_EQ(WE0_1->getExceptionDepth(), (unsigned)2);
}
