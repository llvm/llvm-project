//===- llvm/unittest/IR/DebugInfo.cpp - DebugInfo tests -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/Local.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("DebugInfoTest", errs());
  return Mod;
}

namespace {

TEST(DINodeTest, getFlag) {
  // Some valid flags.
  EXPECT_EQ(DINode::FlagPublic, DINode::getFlag("DIFlagPublic"));
  EXPECT_EQ(DINode::FlagProtected, DINode::getFlag("DIFlagProtected"));
  EXPECT_EQ(DINode::FlagPrivate, DINode::getFlag("DIFlagPrivate"));
  EXPECT_EQ(DINode::FlagVector, DINode::getFlag("DIFlagVector"));
  EXPECT_EQ(DINode::FlagRValueReference,
            DINode::getFlag("DIFlagRValueReference"));

  // FlagAccessibility shouldn't work.
  EXPECT_EQ(0u, DINode::getFlag("DIFlagAccessibility"));

  // Some other invalid strings.
  EXPECT_EQ(0u, DINode::getFlag("FlagVector"));
  EXPECT_EQ(0u, DINode::getFlag("Vector"));
  EXPECT_EQ(0u, DINode::getFlag("other things"));
  EXPECT_EQ(0u, DINode::getFlag("DIFlagOther"));
}

TEST(DINodeTest, getFlagString) {
  // Some valid flags.
  EXPECT_EQ(StringRef("DIFlagPublic"),
            DINode::getFlagString(DINode::FlagPublic));
  EXPECT_EQ(StringRef("DIFlagProtected"),
            DINode::getFlagString(DINode::FlagProtected));
  EXPECT_EQ(StringRef("DIFlagPrivate"),
            DINode::getFlagString(DINode::FlagPrivate));
  EXPECT_EQ(StringRef("DIFlagVector"),
            DINode::getFlagString(DINode::FlagVector));
  EXPECT_EQ(StringRef("DIFlagRValueReference"),
            DINode::getFlagString(DINode::FlagRValueReference));

  // FlagAccessibility actually equals FlagPublic.
  EXPECT_EQ(StringRef("DIFlagPublic"),
            DINode::getFlagString(DINode::FlagAccessibility));

  // Some other invalid flags.
  EXPECT_EQ(StringRef(),
            DINode::getFlagString(DINode::FlagPublic | DINode::FlagVector));
  EXPECT_EQ(StringRef(), DINode::getFlagString(DINode::FlagFwdDecl |
                                               DINode::FlagArtificial));
  EXPECT_EQ(StringRef(),
            DINode::getFlagString(static_cast<DINode::DIFlags>(0xffff)));
}

TEST(DINodeTest, splitFlags) {
// Some valid flags.
#define CHECK_SPLIT(FLAGS, VECTOR, REMAINDER)                                  \
  {                                                                            \
    SmallVector<DINode::DIFlags, 8> V;                                         \
    EXPECT_EQ(REMAINDER, DINode::splitFlags(FLAGS, V));                        \
    EXPECT_TRUE(makeArrayRef(V).equals(VECTOR));                               \
  }
  CHECK_SPLIT(DINode::FlagPublic, {DINode::FlagPublic}, DINode::FlagZero);
  CHECK_SPLIT(DINode::FlagProtected, {DINode::FlagProtected}, DINode::FlagZero);
  CHECK_SPLIT(DINode::FlagPrivate, {DINode::FlagPrivate}, DINode::FlagZero);
  CHECK_SPLIT(DINode::FlagVector, {DINode::FlagVector}, DINode::FlagZero);
  CHECK_SPLIT(DINode::FlagRValueReference, {DINode::FlagRValueReference},
              DINode::FlagZero);
  DINode::DIFlags Flags[] = {DINode::FlagFwdDecl, DINode::FlagVector};
  CHECK_SPLIT(DINode::FlagFwdDecl | DINode::FlagVector, Flags,
              DINode::FlagZero);
  CHECK_SPLIT(DINode::FlagZero, {}, DINode::FlagZero);
#undef CHECK_SPLIT
}

TEST(StripTest, LoopMetadata) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"(
    define void @f() !dbg !5 {
      ret void, !dbg !10, !llvm.loop !11
    }

    !llvm.dbg.cu = !{!0}
    !llvm.debugify = !{!3, !3}
    !llvm.module.flags = !{!4}

    !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
    !1 = !DIFile(filename: "loop.ll", directory: "/")
    !2 = !{}
    !3 = !{i32 1}
    !4 = !{i32 2, !"Debug Info Version", i32 3}
    !5 = distinct !DISubprogram(name: "f", linkageName: "f", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !7)
    !6 = !DISubroutineType(types: !2)
    !7 = !{!8}
    !8 = !DILocalVariable(name: "1", scope: !5, file: !1, line: 1, type: !9)
    !9 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
    !10 = !DILocation(line: 1, column: 1, scope: !5)
    !11 = distinct !{!11, !10, !10}
)");

  // Look up the debug info emission kind for the CU via the loop metadata
  // attached to the terminator. If, when stripping non-line table debug info,
  // we update the terminator's metadata correctly, we should be able to
  // observe the change in emission kind for the CU.
  auto getEmissionKind = [&]() {
    Instruction &I = *M->getFunction("f")->getEntryBlock().getFirstNonPHI();
    MDNode *LoopMD = I.getMetadata(LLVMContext::MD_loop);
    return cast<DILocation>(LoopMD->getOperand(1))
        ->getScope()
        ->getSubprogram()
        ->getUnit()
        ->getEmissionKind();
  };

  EXPECT_EQ(getEmissionKind(), DICompileUnit::FullDebug);

  bool Changed = stripNonLineTableDebugInfo(*M);
  EXPECT_TRUE(Changed);

  EXPECT_EQ(getEmissionKind(), DICompileUnit::LineTablesOnly);

  bool BrokenDebugInfo = false;
  bool HardError = verifyModule(*M, &errs(), &BrokenDebugInfo);
  EXPECT_FALSE(HardError);
  EXPECT_FALSE(BrokenDebugInfo);
}

TEST(MetadataTest, DeleteInstUsedByDbgValue) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"(
    define i16 @f(i16 %a) !dbg !6 {
      %b = add i16 %a, 1, !dbg !11
      call void @llvm.dbg.value(metadata i16 %b, metadata !9, metadata !DIExpression()), !dbg !11
      ret i16 0, !dbg !11
    }
    declare void @llvm.dbg.value(metadata, metadata, metadata) #0
    attributes #0 = { nounwind readnone speculatable willreturn }

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!5}

    !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
    !1 = !DIFile(filename: "t.ll", directory: "/")
    !2 = !{}
    !5 = !{i32 2, !"Debug Info Version", i32 3}
    !6 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
    !7 = !DISubroutineType(types: !2)
    !8 = !{!9}
    !9 = !DILocalVariable(name: "1", scope: !6, file: !1, line: 1, type: !10)
    !10 = !DIBasicType(name: "ty16", size: 16, encoding: DW_ATE_unsigned)
    !11 = !DILocation(line: 1, column: 1, scope: !6)
)");

  // Find %b = add ...
  Instruction &I = *M->getFunction("f")->getEntryBlock().getFirstNonPHI();

  // Find the dbg.value using %b.
  SmallVector<DbgValueInst *, 1> DVIs;
  findDbgValues(DVIs, &I);

  // Delete %b. The dbg.value should now point to undef.
  I.eraseFromParent();
  EXPECT_TRUE(isa<UndefValue>(DVIs[0]->getValue()));
}

} // end namespace
