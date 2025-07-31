//===- MachineInstrBundleIteratorTest.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MIRPrinter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineModuleSlotTracker.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/FileCheck/FileCheck.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "gtest/gtest.h"

using namespace llvm;

class MachineMetadataTest : public testing::Test {
public:
  MachineMetadataTest() {}

protected:
  LLVMContext Context;
  std::unique_ptr<Module> M;
  std::unique_ptr<MIRParser> MIR;

  static void SetUpTestCase() {
    InitializeAllTargetInfos();
    InitializeAllTargets();
    InitializeAllTargetMCs();
  }

  void SetUp() override { M = std::make_unique<Module>("Dummy", Context); }

  void addHooks(ModuleSlotTracker &MST, const MachineOperand &MO) {
    // Setup hooks to assign slot numbers for the specified machine metadata.
    MST.setProcessHook([&MO](AbstractSlotTrackerStorage *AST, const Module *M,
                             bool ShouldInitializeAllMetadata) {
      if (ShouldInitializeAllMetadata) {
        if (MO.isMetadata())
          AST->createMetadataSlot(MO.getMetadata());
      }
    });
    MST.setProcessHook([&MO](AbstractSlotTrackerStorage *AST, const Function *F,
                             bool ShouldInitializeAllMetadata) {
      if (!ShouldInitializeAllMetadata) {
        if (MO.isMetadata())
          AST->createMetadataSlot(MO.getMetadata());
      }
    });
  }

  std::unique_ptr<TargetMachine>
  createTargetMachine(std::string TargetStr, StringRef CPU, StringRef FS) {
    Triple TT(TargetStr);
    std::string Error;
    const Target *T = TargetRegistry::lookupTarget(TT, Error);
    if (!T)
      return nullptr;
    TargetOptions Options;
    return std::unique_ptr<TargetMachine>(T->createTargetMachine(
        TT, CPU, FS, Options, std::nullopt, std::nullopt));
  }

  std::unique_ptr<Module> parseMIR(const TargetMachine &TM, StringRef MIRCode,
                                   MachineModuleInfo &MMI) {
    SMDiagnostic Diagnostic;
    std::unique_ptr<MemoryBuffer> MBuffer = MemoryBuffer::getMemBuffer(MIRCode);
    MIR = createMIRParser(std::move(MBuffer), Context);
    if (!MIR)
      return nullptr;

    std::unique_ptr<Module> Mod = MIR->parseIRModule();
    if (!Mod)
      return nullptr;

    Mod->setDataLayout(TM.createDataLayout());

    if (MIR->parseMachineFunctions(*Mod, MMI)) {
      M.reset();
      return nullptr;
    }

    return Mod;
  }
};

// Helper to dump the printer output into a string.
static std::string print(std::function<void(raw_ostream &OS)> PrintFn) {
  std::string Str;
  raw_string_ostream OS(Str);
  PrintFn(OS);
  return Str;
}

TEST_F(MachineMetadataTest, TrivialHook) {
  // Verify that post-process hook is invoked to assign slot numbers for
  // machine metadata.
  ASSERT_TRUE(M);

  // Create a MachineOperand with a metadata and print it.
  Metadata *MDS = MDString::get(Context, "foo");
  MDNode *Node = MDNode::get(Context, MDS);
  MachineOperand MO = MachineOperand::CreateMetadata(Node);

  // Checking some preconditions on the newly created
  // MachineOperand.
  ASSERT_TRUE(MO.isMetadata());
  ASSERT_EQ(MO.getMetadata(), Node);

  ModuleSlotTracker MST(M.get());
  addHooks(MST, MO);

  // Print a MachineOperand containing a metadata node.
  EXPECT_EQ("!0", print([&](raw_ostream &OS) {
              MO.print(OS, MST, LLT{}, /*OpIdx*/ ~0U, /*PrintDef=*/false,
                       /*IsStandalone=*/false,
                       /*ShouldPrintRegisterTies=*/false, /*TiedOperandIdx=*/0,
                       /*TRI=*/nullptr);
            }));
  // Print the definition of that metadata node.
  EXPECT_EQ("!0 = !{!\"foo\"}",
            print([&](raw_ostream &OS) { Node->print(OS, MST); }));
}

TEST_F(MachineMetadataTest, BasicHook) {
  // Verify that post-process hook is invoked to assign slot numbers for
  // machine metadata. When both LLVM IR and machine IR contain metadata,
  // ensure that machine metadata is always assigned after LLVM IR.
  ASSERT_TRUE(M);

  // Create a MachineOperand with a metadata and print it.
  Metadata *MachineMDS = MDString::get(Context, "foo");
  MDNode *MachineNode = MDNode::get(Context, MachineMDS);
  MachineOperand MO = MachineOperand::CreateMetadata(MachineNode);

  // Checking some preconditions on the newly created
  // MachineOperand.
  ASSERT_TRUE(MO.isMetadata());
  ASSERT_EQ(MO.getMetadata(), MachineNode);

  // Create metadata in LLVM IR.
  NamedMDNode *MD = M->getOrInsertNamedMetadata("namedmd");
  Metadata *MDS = MDString::get(Context, "bar");
  MDNode *Node = MDNode::get(Context, MDS);
  MD->addOperand(Node);

  ModuleSlotTracker MST(M.get());
  addHooks(MST, MO);

  // Print a MachineOperand containing a metadata node.
  EXPECT_EQ("!1", print([&](raw_ostream &OS) {
              MO.print(OS, MST, LLT{}, /*OpIdx*/ ~0U, /*PrintDef=*/false,
                       /*IsStandalone=*/false,
                       /*ShouldPrintRegisterTies=*/false, /*TiedOperandIdx=*/0,
                       /*TRI=*/nullptr);
            }));
  // Print the definition of these unnamed metadata nodes.
  EXPECT_EQ("!0 = !{!\"bar\"}",
            print([&](raw_ostream &OS) { Node->print(OS, MST); }));
  EXPECT_EQ("!1 = !{!\"foo\"}",
            print([&](raw_ostream &OS) { MachineNode->print(OS, MST); }));
}

static bool checkOutput(std::string CheckString, std::string Output) {
  auto CheckBuffer = MemoryBuffer::getMemBuffer(CheckString, "");
  auto OutputBuffer = MemoryBuffer::getMemBuffer(Output, "Output", false);

  SmallString<4096> CheckFileBuffer;
  FileCheckRequest Req;
  FileCheck FC(Req);
  StringRef CheckFileText = FC.CanonicalizeFile(*CheckBuffer, CheckFileBuffer);

  SourceMgr SM;
  SM.AddNewSourceBuffer(MemoryBuffer::getMemBuffer(CheckFileText, "CheckFile"),
                        SMLoc());
  if (FC.readCheckFile(SM, CheckFileText))
    return false;

  auto OutBuffer = OutputBuffer->getBuffer();
  SM.AddNewSourceBuffer(std::move(OutputBuffer), SMLoc());
  return FC.checkInput(SM, OutBuffer);
}

TEST_F(MachineMetadataTest, MMSlotTrackerAArch64) {
  auto TM = createTargetMachine(Triple::normalize("aarch64--"), "", "");
  if (!TM)
    GTEST_SKIP();

  StringRef MIRString = R"MIR(
--- |
  define i32 @test0(i32* %p) {
    %r = load i32, i32* %p, align 4
    ret i32 %r
  }
...
---
name:            test0
liveins:
  - { reg: '$x0', virtual-reg: '%0' }
body:             |
  bb.0 (%ir-block.0):
    liveins: $x0

  %0:gpr64common = COPY $x0
  %1:gpr32 = LDRWui %0, 0 :: (load (s32) from %ir.p)
...
)MIR";

  MachineModuleInfo MMI(TM.get());
  M = parseMIR(*TM, MIRString, MMI);
  ASSERT_TRUE(M);

  auto *MF = MMI.getMachineFunction(*M->getFunction("test0"));
  auto *MBB = MF->getBlockNumbered(0);

  auto &MI = MBB->back();
  ASSERT_TRUE(MI.hasOneMemOperand());

  // Create and attached scoped AA metadata on that instruction with one MMO.
  MDBuilder MDB(Context);
  MDNode *Domain = MDB.createAnonymousAliasScopeDomain("domain");
  MDNode *Scope0 = MDB.createAnonymousAliasScope(Domain, "scope0");
  MDNode *Scope1 = MDB.createAnonymousAliasScope(Domain, "scope1");
  MDNode *Set0 = MDNode::get(Context, {Scope0});
  MDNode *Set1 = MDNode::get(Context, {Scope1});

  AAMDNodes AAInfo;
  AAInfo.TBAA = AAInfo.TBAAStruct = nullptr;
  AAInfo.Scope = Set0;
  AAInfo.NoAlias = Set1;

  auto *OldMMO = MI.memoperands().front();
  auto *NewMMO = MF->getMachineMemOperand(OldMMO, AAInfo);
  MI.setMemRefs(*MF, NewMMO);

  MachineModuleSlotTracker MST(MMI, MF);
  // Print that MI with new machine metadata, which slot numbers should be
  // assigned.
  EXPECT_EQ("%1:gpr32 = LDRWui %0, 0 :: (load (s32) from %ir.p, "
            "!alias.scope !0, !noalias !3)",
            print([&](raw_ostream &OS) {
              MI.print(OS, MST, /*IsStandalone=*/false, /*SkipOpers=*/false,
                       /*SkipDebugLoc=*/false, /*AddNewLine=*/false);
            }));

  std::vector<const MDNode *> Generated{Domain, Scope0, Scope1, Set0, Set1};
  // Examine machine metadata collected. They should match ones
  // afore-generated.
  std::vector<const MDNode *> Collected;
  MachineModuleSlotTracker::MachineMDNodeListType MDList;
  MST.collectMachineMDNodes(MDList);
  for (auto &MD : MDList)
    Collected.push_back(MD.second);

  llvm::sort(Generated);
  llvm::sort(Collected);
  EXPECT_EQ(Collected, Generated);

  // FileCheck the output from MIR printer.
  std::string Output = print([&](raw_ostream &OS) { printMIR(OS, MMI, *MF); });
  std::string CheckString = R"(
CHECK: machineMetadataNodes:
CHECK-DAG: ![[MMDOMAIN:[0-9]+]] = distinct !{!{{[0-9]+}}, !"domain"}
CHECK-DAG: ![[MMSCOPE0:[0-9]+]] = distinct !{!{{[0-9]+}}, ![[MMDOMAIN]], !"scope0"}
CHECK-DAG: ![[MMSCOPE1:[0-9]+]] = distinct !{!{{[0-9]+}}, ![[MMDOMAIN]], !"scope1"}
CHECK-DAG: ![[MMSET0:[0-9]+]] = !{![[MMSCOPE0]]}
CHECK-DAG: ![[MMSET1:[0-9]+]] = !{![[MMSCOPE1]]}
CHECK: body:
CHECK: %1:gpr32 = LDRWui %0, 0 :: (load (s32) from %ir.p, !alias.scope ![[MMSET0]], !noalias ![[MMSET1]])
)";
  EXPECT_TRUE(checkOutput(CheckString, Output));
}

TEST_F(MachineMetadataTest, isMetaInstruction) {
  auto TM = createTargetMachine(Triple::normalize("x86_64--"), "", "");
  if (!TM)
    GTEST_SKIP();

  StringRef MIRString = R"MIR(
--- |
  define void @test0(i32 %b) {
    ret void
  }
  !0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
  !1 = !DIFile(filename: "a.c", directory: "/tmp")
  !2 = !{i32 7, !"Dwarf Version", i32 4}
  !3 = !{i32 2, !"Debug Info Version", i32 3}
  !4 = !{i32 1, !"wchar_size", i32 4}
  !5 = !{i32 7, !"uwtable", i32 1}
  !6 = !{i32 7, !"frame-pointer", i32 2}
  !7 = !{!""}
  !8 = distinct !DISubprogram(name: "test0", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
  !9 = !DISubroutineType(types: !10)
  !10 = !{null, !11}
  !11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
  !12 = !{}
  !13 = !DILocalVariable(name: "b", arg: 1, scope: !8, file: !1, line: 1, type: !11)
  !14 = !DILocation(line: 1, column: 16, scope: !8)
...
---
name:            test0
machineFunctionInfo
body:             |
  bb.0:
  $rdi = IMPLICIT_DEF
  KILL $rsi
  CFI_INSTRUCTION undefined $rax
  EH_LABEL 0
  GC_LABEL 0
  DBG_VALUE $rax, $noreg, !13, !DIExpression(), debug-location !14
  DBG_LABEL 0
  LIFETIME_START 0
  LIFETIME_END 0
  PSEUDO_PROBE 6699318081062747564, 1, 0, 0
  $xmm0 = ARITH_FENCE $xmm0
  MEMBARRIER
...
)MIR";

  MachineModuleInfo MMI(TM.get());
  M = parseMIR(*TM, MIRString, MMI);
  ASSERT_TRUE(M);

  auto *MF = MMI.getMachineFunction(*M->getFunction("test0"));
  auto *MBB = MF->getBlockNumbered(0);

  for (auto It = MBB->begin(); It != MBB->end(); ++It) {
    MachineInstr &MI = *It;
    ASSERT_TRUE(MI.isMetaInstruction());
  }
}

TEST_F(MachineMetadataTest, MMSlotTrackerX64) {
  auto TM = createTargetMachine(Triple::normalize("x86_64--"), "", "");
  if (!TM)
    GTEST_SKIP();

  StringRef MIRString = R"MIR(
--- |
  define i32 @test0(i32* %p) {
    %r = load i32, i32* %p, align 4
    ret i32 %r
  }
...
---
name:            test0
liveins:
  - { reg: '$rdi', virtual-reg: '%0' }
body:             |
  bb.0 (%ir-block.0):
    liveins: $rdi

  %0:gr64 = COPY $rdi
  %1:gr32 = MOV32rm %0, 1, $noreg, 0, $noreg :: (load (s32) from %ir.p)
...
)MIR";

  MachineModuleInfo MMI(TM.get());
  M = parseMIR(*TM, MIRString, MMI);
  ASSERT_TRUE(M);

  auto *MF = MMI.getMachineFunction(*M->getFunction("test0"));
  auto *MBB = MF->getBlockNumbered(0);

  auto &MI = MBB->back();
  ASSERT_FALSE(MI.memoperands_empty());
  ASSERT_TRUE(MI.hasOneMemOperand());

  // Create and attached scoped AA metadata on that instruction with one MMO.
  MDBuilder MDB(Context);
  MDNode *Domain = MDB.createAnonymousAliasScopeDomain("domain");
  MDNode *Scope0 = MDB.createAnonymousAliasScope(Domain, "scope0");
  MDNode *Scope1 = MDB.createAnonymousAliasScope(Domain, "scope1");
  MDNode *Set0 = MDNode::get(Context, {Scope0});
  MDNode *Set1 = MDNode::get(Context, {Scope1});

  AAMDNodes AAInfo;
  AAInfo.TBAA = AAInfo.TBAAStruct = nullptr;
  AAInfo.Scope = Set0;
  AAInfo.NoAlias = Set1;

  auto *OldMMO = MI.memoperands().front();
  auto *NewMMO = MF->getMachineMemOperand(OldMMO, AAInfo);
  MI.setMemRefs(*MF, NewMMO);

  MachineModuleSlotTracker MST(MMI, MF);
  // Print that MI with new machine metadata, which slot numbers should be
  // assigned.
  EXPECT_EQ("%1:gr32 = MOV32rm %0, 1, $noreg, 0, $noreg :: (load (s32) from %ir.p, "
            "!alias.scope !0, !noalias !3)",
            print([&](raw_ostream &OS) {
              MI.print(OS, MST, /*IsStandalone=*/false, /*SkipOpers=*/false,
                       /*SkipDebugLoc=*/false, /*AddNewLine=*/false);
            }));

  std::vector<const MDNode *> Generated{Domain, Scope0, Scope1, Set0, Set1};
  // Examine machine metadata collected. They should match ones
  // afore-generated.
  std::vector<const MDNode *> Collected;
  MachineModuleSlotTracker::MachineMDNodeListType MDList;
  MST.collectMachineMDNodes(MDList);
  for (auto &MD : MDList)
    Collected.push_back(MD.second);

  llvm::sort(Generated);
  llvm::sort(Collected);
  EXPECT_EQ(Collected, Generated);

  // FileCheck the output from MIR printer.
  std::string Output = print([&](raw_ostream &OS) { printMIR(OS, MMI, *MF); });
  std::string CheckString = R"(
CHECK: machineMetadataNodes:
CHECK-DAG: ![[MMDOMAIN:[0-9]+]] = distinct !{!{{[0-9]+}}, !"domain"}
CHECK-DAG: ![[MMSCOPE0:[0-9]+]] = distinct !{!{{[0-9]+}}, ![[MMDOMAIN]], !"scope0"}
CHECK-DAG: ![[MMSCOPE1:[0-9]+]] = distinct !{!{{[0-9]+}}, ![[MMDOMAIN]], !"scope1"}
CHECK-DAG: ![[MMSET0:[0-9]+]] = !{![[MMSCOPE0]]}
CHECK-DAG: ![[MMSET1:[0-9]+]] = !{![[MMSCOPE1]]}
CHECK: body:
CHECK: %1:gr32 = MOV32rm %0, 1, $noreg, 0, $noreg :: (load (s32) from %ir.p, !alias.scope ![[MMSET0]], !noalias ![[MMSET1]])
)";
  EXPECT_TRUE(checkOutput(CheckString, Output));
}

TEST_F(MachineMetadataTest, MMSlotTrackerAMDGPU) {
  auto TM = createTargetMachine(Triple::normalize("amdgcn-amd-amdhsa"),
                                "gfx1010", "");
  if (!TM)
    GTEST_SKIP();

  StringRef MIRString = R"MIR(
--- |
  define i32 @test0(i32* %p) {
    %r = load i32, i32* %p, align 4
    ret i32 %r
  }
...
---
name:            test0
liveins:
  - { reg: '$vgpr0', virtual-reg: '%0' }
  - { reg: '$vgpr1', virtual-reg: '%1' }
  - { reg: '$sgpr30_sgpr31', virtual-reg: '%2' }
body:             |
  bb.0 (%ir-block.0):
    liveins: $vgpr0, $vgpr1, $sgpr30_sgpr31

    %2:sreg_64 = COPY $sgpr30_sgpr31
    %1:vgpr_32 = COPY $vgpr1
    %0:vgpr_32 = COPY $vgpr0
    %8:vreg_64 = REG_SEQUENCE %0, %subreg.sub0, %1, %subreg.sub1
    %6:vreg_64 = COPY %8
    %5:vgpr_32 = FLAT_LOAD_DWORD killed %6, 0, 0, implicit $exec, implicit $flat_scr :: (load (s32) from %ir.p)
...
)MIR";

  MachineModuleInfo MMI(TM.get());
  M = parseMIR(*TM, MIRString, MMI);
  ASSERT_TRUE(M);

  auto *MF = MMI.getMachineFunction(*M->getFunction("test0"));
  auto *MBB = MF->getBlockNumbered(0);

  auto &MI = MBB->back();
  ASSERT_FALSE(MI.memoperands_empty());
  ASSERT_TRUE(MI.hasOneMemOperand());

  // Create and attached scoped AA metadata on that instruction with one MMO.
  MDBuilder MDB(Context);
  MDNode *Domain = MDB.createAnonymousAliasScopeDomain("domain");
  MDNode *Scope0 = MDB.createAnonymousAliasScope(Domain, "scope0");
  MDNode *Scope1 = MDB.createAnonymousAliasScope(Domain, "scope1");
  MDNode *Set0 = MDNode::get(Context, {Scope0});
  MDNode *Set1 = MDNode::get(Context, {Scope1});

  AAMDNodes AAInfo;
  AAInfo.TBAA = AAInfo.TBAAStruct = nullptr;
  AAInfo.Scope = Set0;
  AAInfo.NoAlias = Set1;

  auto *OldMMO = MI.memoperands().front();
  auto *NewMMO = MF->getMachineMemOperand(OldMMO, AAInfo);
  MI.setMemRefs(*MF, NewMMO);

  MachineModuleSlotTracker MST(MMI, MF);
  // Print that MI with new machine metadata, which slot numbers should be
  // assigned.
  EXPECT_EQ(
      "%5:vgpr_32 = FLAT_LOAD_DWORD killed %4, 0, 0, implicit $exec, implicit "
      "$flat_scr :: (load (s32) from %ir.p, !alias.scope !0, !noalias !3)",
      print([&](raw_ostream &OS) {
        MI.print(OS, MST, /*IsStandalone=*/false, /*SkipOpers=*/false,
                 /*SkipDebugLoc=*/false, /*AddNewLine=*/false);
      }));

  std::vector<const MDNode *> Generated{Domain, Scope0, Scope1, Set0, Set1};
  // Examine machine metadata collected. They should match ones
  // afore-generated.
  std::vector<const MDNode *> Collected;
  MachineModuleSlotTracker::MachineMDNodeListType MDList;
  MST.collectMachineMDNodes(MDList);
  for (auto &MD : MDList)
    Collected.push_back(MD.second);

  llvm::sort(Generated);
  llvm::sort(Collected);
  EXPECT_EQ(Collected, Generated);

  // FileCheck the output from MIR printer.
  std::string Output = print([&](raw_ostream &OS) { printMIR(OS, MMI, *MF); });
  std::string CheckString = R"(
CHECK: machineMetadataNodes:
CHECK-DAG: ![[MMDOMAIN:[0-9]+]] = distinct !{!{{[0-9]+}}, !"domain"}
CHECK-DAG: ![[MMSCOPE0:[0-9]+]] = distinct !{!{{[0-9]+}}, ![[MMDOMAIN]], !"scope0"}
CHECK-DAG: ![[MMSCOPE1:[0-9]+]] = distinct !{!{{[0-9]+}}, ![[MMDOMAIN]], !"scope1"}
CHECK-DAG: ![[MMSET0:[0-9]+]] = !{![[MMSCOPE0]]}
CHECK-DAG: ![[MMSET1:[0-9]+]] = !{![[MMSCOPE1]]}
CHECK: body:
CHECK: %5:vgpr_32 = FLAT_LOAD_DWORD killed %4, 0, 0, implicit $exec, implicit $flat_scr :: (load (s32) from %ir.p, !alias.scope ![[MMSET0]], !noalias ![[MMSET1]])
)";
  EXPECT_TRUE(checkOutput(CheckString, Output));
}

TEST_F(MachineMetadataTest, TiedOpsRewritten) {
  auto TM = createTargetMachine(Triple::normalize("powerpc64--"), "", "");
  if (!TM)
    GTEST_SKIP();
  StringRef MIRString = R"MIR(
---
name:            foo
alignment:       16
tracksRegLiveness: true
frameInfo:
  maxAlignment:    16
machineFunctionInfo: {}
body:             |
  bb.0:
    liveins: $r3
    %0:gprc = COPY $r3
    %0 = RLWIMI killed %0, $r3, 1, 0, 30
    $r3 = COPY %0
    BLR8 implicit $r3, implicit $lr8, implicit $rm

...
)MIR";
  MachineModuleInfo MMI(TM.get());
  M = parseMIR(*TM, MIRString, MMI);
  ASSERT_TRUE(M);
  auto *MF = MMI.getMachineFunction(*M->getFunction("foo"));
  MachineFunctionProperties &Properties = MF->getProperties();
  ASSERT_TRUE(Properties.hasTiedOpsRewritten());
}

TEST_F(MachineMetadataTest, NoTiedOpsRewritten) {
  auto TM = createTargetMachine(Triple::normalize("powerpc64--"), "", "");
  if (!TM)
    GTEST_SKIP();
  StringRef MIRString = R"MIR(
---
name:            foo
alignment:       16
tracksRegLiveness: true
frameInfo:
  maxAlignment:    16
machineFunctionInfo: {}
body:             |
  bb.0:
    liveins: $r3
    %0:gprc = COPY $r3
    %1:gprc = RLWIMI killed %0, $r3, 1, 0, 30
    $r3 = COPY %1
    BLR8 implicit $r3, implicit $lr8, implicit $rm

...
)MIR";
  MachineModuleInfo MMI(TM.get());
  M = parseMIR(*TM, MIRString, MMI);
  ASSERT_TRUE(M);
  auto *MF = MMI.getMachineFunction(*M->getFunction("foo"));
  MachineFunctionProperties &Properties = MF->getProperties();
  ASSERT_FALSE(Properties.hasTiedOpsRewritten());
}
