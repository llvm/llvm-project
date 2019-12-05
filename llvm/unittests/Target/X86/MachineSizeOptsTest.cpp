//===- MachineSizeOptsTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineSizeOpts.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

std::unique_ptr<LLVMTargetMachine> createTargetMachine() {
  auto TT(Triple::normalize("x86_64--"));
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);
  return std::unique_ptr<LLVMTargetMachine>(static_cast<LLVMTargetMachine*>(
      TheTarget->createTargetMachine(TT, "", "", TargetOptions(), None, None,
                                     CodeGenOpt::Default)));
}

class MachineSizeOptsTest : public testing::Test {
 protected:
  static const char* MIRString;
  LLVMContext Context;
  std::unique_ptr<LLVMTargetMachine> TM;
  std::unique_ptr<MachineModuleInfo> MMI;
  std::unique_ptr<MIRParser> Parser;
  std::unique_ptr<Module> M;
  struct BFIData {
    std::unique_ptr<MachineDominatorTree> MDT;
    std::unique_ptr<MachineLoopInfo> MLI;
    std::unique_ptr<MachineBranchProbabilityInfo> MBPI;
    std::unique_ptr<MachineBlockFrequencyInfo> MBFI;
    BFIData(MachineFunction &MF) {
      MDT.reset(new MachineDominatorTree(MF));
      MLI.reset(new MachineLoopInfo(*MDT));
      MBPI.reset(new MachineBranchProbabilityInfo());
      MBFI.reset(new MachineBlockFrequencyInfo(MF, *MBPI, *MLI));
    }
    MachineBlockFrequencyInfo *get() { return MBFI.get(); }
  };

  static void SetUpTestCase() {
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86Target();
    LLVMInitializeX86TargetMC();
  }

  void SetUp() override {
    TM = createTargetMachine();
    std::unique_ptr<MemoryBuffer> MBuffer =
        MemoryBuffer::getMemBuffer(MIRString);
    Parser = createMIRParser(std::move(MBuffer), Context);
    if (!Parser)
      report_fatal_error("null MIRParser");
    M = Parser->parseIRModule();
    if (!M)
      report_fatal_error("parseIRModule failed");
    M->setTargetTriple(TM->getTargetTriple().getTriple());
    M->setDataLayout(TM->createDataLayout());
    MMI = std::make_unique<MachineModuleInfo>(TM.get());
    if (Parser->parseMachineFunctions(*M, *MMI.get()))
      report_fatal_error("parseMachineFunctions failed");
  }

  MachineFunction *getMachineFunction(Module *M, StringRef Name) {
    auto F = M->getFunction(Name);
    if (!F)
      report_fatal_error("null Function");
    auto &MF = MMI->getOrCreateMachineFunction(*F);
    return &MF;
  }
};

TEST_F(MachineSizeOptsTest, Test) {
  MachineFunction *F = getMachineFunction(M.get(), "f");
  ASSERT_TRUE(F != nullptr);
  MachineFunction *G = getMachineFunction(M.get(), "g");
  ASSERT_TRUE(G != nullptr);
  MachineFunction *H = getMachineFunction(M.get(), "h");
  ASSERT_TRUE(H != nullptr);
  ProfileSummaryInfo PSI = ProfileSummaryInfo(*M.get());
  ASSERT_TRUE(PSI.hasProfileSummary());
  BFIData BFID_F(*F);
  BFIData BFID_G(*G);
  BFIData BFID_H(*H);
  MachineBlockFrequencyInfo *MBFI_F = BFID_F.get();
  MachineBlockFrequencyInfo *MBFI_G = BFID_G.get();
  MachineBlockFrequencyInfo *MBFI_H = BFID_H.get();
  MachineBasicBlock &BB0 = F->front();
  auto iter = BB0.succ_begin();
  MachineBasicBlock *BB1 = *iter;
  iter++;
  MachineBasicBlock *BB2 = *iter;
  iter++;
  ASSERT_TRUE(iter == BB0.succ_end());
  MachineBasicBlock *BB3 = *BB1->succ_begin();
  ASSERT_TRUE(BB3 == *BB2->succ_begin());
  EXPECT_FALSE(shouldOptimizeForSize(F, &PSI, MBFI_F, PGSOQueryType::Test));
  EXPECT_TRUE(shouldOptimizeForSize(G, &PSI, MBFI_G, PGSOQueryType::Test));
  EXPECT_FALSE(shouldOptimizeForSize(H, &PSI, MBFI_H, PGSOQueryType::Test));
  EXPECT_FALSE(shouldOptimizeForSize(&BB0, &PSI, MBFI_F, PGSOQueryType::Test));
  EXPECT_FALSE(shouldOptimizeForSize(BB1, &PSI, MBFI_F, PGSOQueryType::Test));
  EXPECT_TRUE(shouldOptimizeForSize(BB2, &PSI, MBFI_F, PGSOQueryType::Test));
  EXPECT_FALSE(shouldOptimizeForSize(BB3, &PSI, MBFI_F, PGSOQueryType::Test));
}

const char* MachineSizeOptsTest::MIRString = R"MIR(
--- |
  define i32 @g(i32 %x) !prof !14 {
    ret i32 0
  }

  define i32 @h(i32 %x) !prof !15 {
    ret i32 0
  }

  define i32 @f(i32 %x) !prof !16 {
  bb0:
    %y1 = icmp eq i32 %x, 0
    br i1 %y1, label %bb1, label %bb2, !prof !17

  bb1:                                              ; preds = %bb0
    %z1 = call i32 @g(i32 %x)
    br label %bb3

  bb2:                                              ; preds = %bb0
    %z2 = call i32 @h(i32 %x)
    br label %bb3

  bb3:                                              ; preds = %bb2, %bb1
    %y2 = phi i32 [ 0, %bb1 ], [ 1, %bb2 ]
    ret i32 %y2
  }

  !llvm.module.flags = !{!0}

  !0 = !{i32 1, !"ProfileSummary", !1}
  !1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
  !2 = !{!"ProfileFormat", !"InstrProf"}
  !3 = !{!"TotalCount", i64 10000}
  !4 = !{!"MaxCount", i64 10}
  !5 = !{!"MaxInternalCount", i64 1}
  !6 = !{!"MaxFunctionCount", i64 1000}
  !7 = !{!"NumCounts", i64 3}
  !8 = !{!"NumFunctions", i64 3}
  !9 = !{!"DetailedSummary", !10}
  !10 = !{!11, !12, !13}
  !11 = !{i32 10000, i64 1000, i32 1}
  !12 = !{i32 999000, i64 300, i32 3}
  !13 = !{i32 999999, i64 5, i32 10}
  !14 = !{!"function_entry_count", i64 1}
  !15 = !{!"function_entry_count", i64 100}
  !16 = !{!"function_entry_count", i64 400}
  !17 = !{!"branch_weights", i32 100, i32 1}

...
---
name:            g
body:             |
  bb.0:
    %1:gr32 = MOV32r0 implicit-def dead $eflags
    $eax = COPY %1
    RET 0, $eax

...
---
name:            h
body:             |
  bb.0:
    %1:gr32 = MOV32r0 implicit-def dead $eflags
    $eax = COPY %1
    RET 0, $eax

...
---
name:            f
tracksRegLiveness: true
body:             |
  bb.0:
    successors: %bb.1(0x7ebb907a), %bb.2(0x01446f86)
    liveins: $edi

    %1:gr32 = COPY $edi
    TEST32rr %1, %1, implicit-def $eflags
    JCC_1 %bb.2, 5, implicit $eflags
    JMP_1 %bb.1

  bb.1:
    successors: %bb.3(0x80000000)

    ADJCALLSTACKDOWN64 0, 0, 0, implicit-def dead $rsp, implicit-def dead $eflags, implicit-def dead $ssp, implicit $rsp, implicit $ssp
    $edi = COPY %1
    CALL64pcrel32 @g, csr_64, implicit $rsp, implicit $ssp, implicit $edi, implicit-def $rsp, implicit-def $ssp, implicit-def $eax
    ADJCALLSTACKUP64 0, 0, implicit-def dead $rsp, implicit-def dead $eflags, implicit-def dead $ssp, implicit $rsp, implicit $ssp
    %5:gr32 = COPY $eax
    %4:gr32 = MOV32r0 implicit-def dead $eflags
    JMP_1 %bb.3

  bb.2:
    successors: %bb.3(0x80000000)

    ADJCALLSTACKDOWN64 0, 0, 0, implicit-def dead $rsp, implicit-def dead $eflags, implicit-def dead $ssp, implicit $rsp, implicit $ssp
    $edi = COPY %1
    CALL64pcrel32 @h, csr_64, implicit $rsp, implicit $ssp, implicit $edi, implicit-def $rsp, implicit-def $ssp, implicit-def $eax
    ADJCALLSTACKUP64 0, 0, implicit-def dead $rsp, implicit-def dead $eflags, implicit-def dead $ssp, implicit $rsp, implicit $ssp
    %3:gr32 = COPY $eax
    %2:gr32 = MOV32ri 1

  bb.3:
    %0:gr32 = PHI %2, %bb.2, %4, %bb.1
    $eax = COPY %0
    RET 0, $eax

...
)MIR";

} // anonymous namespace
