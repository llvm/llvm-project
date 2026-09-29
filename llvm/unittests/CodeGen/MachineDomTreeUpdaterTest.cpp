//===- MachineDomTreeUpdaterTest.cpp - MachineDomTreeUpdater unit tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineDomTreeUpdater.h"
#include "CodeGenTestBase.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/TargetSelect.h"
#include "gtest/gtest.h"

using namespace llvm;

class MachineDomTreeUpdaterTest : public CodeGenTestBase {
public:
  static void SetUpTestCase() {
    InitializeAllTargets();
    InitializeAllTargetMCs();
  }

  void SetUp() override { setUpImpl("x86_64-unknown-linux-gnu", "", ""); }
};

TEST_F(MachineDomTreeUpdaterTest, EagerUpdateBasicOperations) {
  StringRef MIRString = R"(
--- |
  define i64 @f0(i64 %i, ptr %p) {
  bb0:
    store i64 %i, ptr %p, align 4
    switch i64 %i, label %bb1 [
      i64 1, label %bb2
      i64 2, label %bb3
    ]
  bb1:                                              ; preds = %bb0
    ret i64 1
  bb2:                                              ; preds = %bb0
    ret i64 2
  bb3:                                              ; preds = %bb0
    ret i64 3
  }
...
---
name:            f0
body:             |
  bb.0.bb0:
    successors: %bb.2, %bb.4
    liveins: $rdi, $rsi

    %1:gr64 = COPY $rsi
    %0:gr64 = COPY $rdi
    MOV64mr %1, 1, $noreg, 0, $noreg, %0 :: (store (s64) into %ir.p)
    %2:gr64 = SUB64ri32 %0, 1, implicit-def $eflags
    JCC_1 %bb.2, 4, implicit $eflags
    JMP_1 %bb.4

  bb.4.bb0:
    successors: %bb.3, %bb.1

    %3:gr64 = SUB64ri32 %0, 2, implicit-def $eflags
    JCC_1 %bb.3, 4, implicit $eflags
    JMP_1 %bb.1

  bb.1.bb1:
    %6:gr64 = MOV32ri64 1
    $rax = COPY %6
    RET 0, $rax

  bb.2.bb2:
    %5:gr64 = MOV32ri64 2
    $rax = COPY %5
    RET 0, $rax

  bb.3.bb3:
    %4:gr64 = MOV32ri64 3
    $rax = COPY %4
    RET 0, $rax

...
)";

  ASSERT_TRUE(parseMIR(MIRString));

  MachineFunction &MF = getMF("f0");

  MachineDominatorTree DT(MF);
  MachinePostDominatorTree PDT(MF);
  MachineDomTreeUpdater DTU(DT, PDT,
                            MachineDomTreeUpdater::UpdateStrategy::Eager);

  ASSERT_TRUE(DTU.hasDomTree());
  ASSERT_TRUE(DTU.hasPostDomTree());
  ASSERT_TRUE(DTU.isEager());
  ASSERT_FALSE(DTU.isLazy());
  ASSERT_TRUE(DTU.getDomTree().verify());
  ASSERT_TRUE(DTU.getPostDomTree().verify());
  ASSERT_FALSE(DTU.hasPendingUpdates());

  auto B = MF.begin();
  [[maybe_unused]] auto BB0 = B;
  auto BB1 = ++B;
  auto BB2 = ++B;
  [[maybe_unused]] auto BB3 = ++B;
  auto BB4 = ++B;
  EXPECT_EQ(BB1->succ_size(), 2u);
  ASSERT_TRUE(DT.dominates(&*BB1, &*BB2));
  ASSERT_TRUE(DT.dominates(&*BB1, &*BB4));
  BB1->removeSuccessor(&*BB4);
  DTU.deleteBB(&*BB4);
  EXPECT_EQ(BB1->succ_size(), 1u);
  ASSERT_TRUE(DT.dominates(&*BB1, &*BB2));
}

TEST_F(MachineDomTreeUpdaterTest, LazyUpdateBasicOperations) {
  StringRef MIRString = R"(
--- |
  define i64 @f0(i64 %i, ptr %p) {
  bb0:
    store i64 %i, ptr %p, align 4
    switch i64 %i, label %bb1 [
      i64 1, label %bb2
      i64 2, label %bb3
    ]
  bb1:                                              ; preds = %bb0
    ret i64 1
  bb2:                                              ; preds = %bb0
    ret i64 2
  bb3:                                              ; preds = %bb0
    ret i64 3
  }
...
---
name:            f0
body:             |
  bb.0.bb0:
    successors: %bb.2, %bb.4
    liveins: $rdi, $rsi

    %1:gr64 = COPY $rsi
    %0:gr64 = COPY $rdi
    MOV64mr %1, 1, $noreg, 0, $noreg, %0 :: (store (s64) into %ir.p)
    %2:gr64 = SUB64ri32 %0, 1, implicit-def $eflags
    JCC_1 %bb.2, 4, implicit $eflags
    JMP_1 %bb.4

  bb.4.bb0:
    successors: %bb.3, %bb.1

    %3:gr64 = SUB64ri32 %0, 2, implicit-def $eflags
    JCC_1 %bb.3, 4, implicit $eflags
    JMP_1 %bb.1

  bb.1.bb1:
    %6:gr64 = MOV32ri64 1
    $rax = COPY %6
    RET 0, $rax

  bb.2.bb2:
    %5:gr64 = MOV32ri64 2
    $rax = COPY %5
    RET 0, $rax

  bb.3.bb3:
    %4:gr64 = MOV32ri64 3
    $rax = COPY %4
    RET 0, $rax

...
)";

  ASSERT_TRUE(parseMIR(MIRString));

  MachineFunction &MF = getMF("f0");

  MachineDominatorTree DT(MF);
  MachinePostDominatorTree PDT(MF);
  MachineDomTreeUpdater DTU(DT, PDT,
                            MachineDomTreeUpdater::UpdateStrategy::Lazy);

  ASSERT_TRUE(DTU.hasDomTree());
  ASSERT_TRUE(DTU.hasPostDomTree());
  ASSERT_FALSE(DTU.isEager());
  ASSERT_TRUE(DTU.isLazy());
  ASSERT_TRUE(DTU.getDomTree().verify());
  ASSERT_TRUE(DTU.getPostDomTree().verify());
  ASSERT_FALSE(DTU.hasPendingUpdates());

  auto B = MF.begin();
  [[maybe_unused]] auto BB0 = B;
  auto BB1 = ++B;
  auto BB2 = ++B;
  [[maybe_unused]] auto BB3 = ++B;
  auto BB4 = ++B;
  EXPECT_EQ(BB1->succ_size(), 2u);
  ASSERT_TRUE(DT.dominates(&*BB1, &*BB2));
  ASSERT_TRUE(DT.dominates(&*BB1, &*BB4));
  BB1->removeSuccessor(&*BB4);
  DTU.deleteBB(&*BB4);
  ASSERT_TRUE(DTU.hasPendingDeletedBB());
  EXPECT_EQ(BB1->succ_size(), 1u);
  ASSERT_TRUE(DT.dominates(&*BB1, &*BB2));
  ASSERT_NE(DT.getNode(&*BB4), nullptr);
  DTU.flush();
}
