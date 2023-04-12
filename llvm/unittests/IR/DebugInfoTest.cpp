//===- llvm/unittest/IR/DebugInfo.cpp - DebugInfo tests -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugInfo.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
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
    EXPECT_TRUE(ArrayRef(V).equals(VECTOR));                                   \
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
  EXPECT_EQ(DVIs[0]->getNumVariableLocationOps(), 1u);
  EXPECT_TRUE(isa<UndefValue>(DVIs[0]->getValue(0)));
}

TEST(DIBuiler, CreateFile) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M(new Module("MyModule", Ctx));
  DIBuilder DIB(*M);

  DIFile *F = DIB.createFile("main.c", "/");
  EXPECT_EQ(std::nullopt, F->getSource());

  std::optional<DIFile::ChecksumInfo<StringRef>> Checksum;
  std::optional<StringRef> Source;
  F = DIB.createFile("main.c", "/", Checksum, Source);
  EXPECT_EQ(Source, F->getSource());

  Source = "";
  F = DIB.createFile("main.c", "/", Checksum, Source);
  EXPECT_EQ(Source, F->getSource());
}

TEST(DIBuilder, CreateFortranArrayTypeWithAttributes) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M(new Module("MyModule", Ctx));
  DIBuilder DIB(*M);

  DISubrange *Subrange = DIB.getOrCreateSubrange(1,1);
  SmallVector<Metadata*, 4> Subranges;
  Subranges.push_back(Subrange);
  DINodeArray Subscripts = DIB.getOrCreateArray(Subranges);

  auto getDIExpression = [&DIB](int offset) {
    SmallVector<uint64_t, 4> ops;
    ops.push_back(llvm::dwarf::DW_OP_push_object_address);
    DIExpression::appendOffset(ops, offset);
    ops.push_back(llvm::dwarf::DW_OP_deref);

    return DIB.createExpression(ops);
  };

  DIFile *F = DIB.createFile("main.c", "/");
  DICompileUnit *CU = DIB.createCompileUnit(
      dwarf::DW_LANG_C, DIB.createFile("main.c", "/"), "llvm-c", true, "", 0);

  DIVariable *DataLocation =
      DIB.createTempGlobalVariableFwdDecl(CU, "dl", "_dl", F, 1, nullptr, true);
  DIExpression *Associated = getDIExpression(1);
  DIExpression *Allocated = getDIExpression(2);
  DIExpression *Rank = DIB.createConstantValueExpression(3);

  DICompositeType *ArrayType = DIB.createArrayType(0, 0, nullptr, Subscripts,
                                                   DataLocation, Associated,
                                                   Allocated, Rank);

  EXPECT_TRUE(isa_and_nonnull<DICompositeType>(ArrayType));
  EXPECT_EQ(ArrayType->getRawDataLocation(), DataLocation);
  EXPECT_EQ(ArrayType->getRawAssociated(), Associated);
  EXPECT_EQ(ArrayType->getRawAllocated(), Allocated);
  EXPECT_EQ(ArrayType->getRawRank(), Rank);

  // Avoid memory leak.
  DIVariable::deleteTemporary(DataLocation);
}

TEST(DIBuilder, CreateSetType) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M(new Module("MyModule", Ctx));
  DIBuilder DIB(*M);
  DIScope *Scope = DISubprogram::getDistinct(
      Ctx, nullptr, "", "", nullptr, 0, nullptr, 0, nullptr, 0, 0,
      DINode::FlagZero, DISubprogram::SPFlagZero, nullptr);
  DIType *Type = DIB.createBasicType("Int", 64, dwarf::DW_ATE_signed);
  DIFile *F = DIB.createFile("main.c", "/");

  DIDerivedType *SetType = DIB.createSetType(Scope, "set1", F, 1, 64, 64, Type);
  EXPECT_TRUE(isa_and_nonnull<DIDerivedType>(SetType));
}

TEST(DIBuilder, CreateStringType) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M(new Module("MyModule", Ctx));
  DIBuilder DIB(*M);
  DIScope *Scope = DISubprogram::getDistinct(
      Ctx, nullptr, "", "", nullptr, 0, nullptr, 0, nullptr, 0, 0,
      DINode::FlagZero, DISubprogram::SPFlagZero, nullptr);
  DIFile *F = DIB.createFile("main.c", "/");
  StringRef StrName = "string";
  DIVariable *StringLen =
      DIB.createAutoVariable(Scope, StrName, F, 0, nullptr, false,
                             DINode::FlagZero, dwarf::DW_MSPACE_LLVM_none, 0);
  auto getDIExpression = [&DIB](int offset) {
    SmallVector<uint64_t, 4> ops;
    ops.push_back(llvm::dwarf::DW_OP_push_object_address);
    DIExpression::appendOffset(ops, offset);
    ops.push_back(llvm::dwarf::DW_OP_deref);

    return DIB.createExpression(ops);
  };
  DIExpression *StringLocationExp = getDIExpression(1);
  DIStringType *StringType =
      DIB.createStringType(StrName, StringLen, StringLocationExp);

  EXPECT_TRUE(isa_and_nonnull<DIStringType>(StringType));
  EXPECT_EQ(StringType->getName(), StrName);
  EXPECT_EQ(StringType->getStringLength(), StringLen);
  EXPECT_EQ(StringType->getStringLocationExp(), StringLocationExp);

  StringRef StrNameExp = "stringexp";
  DIExpression *StringLengthExp = getDIExpression(2);
  DIStringType *StringTypeExp =
      DIB.createStringType(StrNameExp, StringLengthExp, StringLocationExp);

  EXPECT_TRUE(isa_and_nonnull<DIStringType>(StringTypeExp));
  EXPECT_EQ(StringTypeExp->getName(), StrNameExp);
  EXPECT_EQ(StringTypeExp->getStringLocationExp(), StringLocationExp);
  EXPECT_EQ(StringTypeExp->getStringLengthExp(), StringLengthExp);
}

TEST(DIBuilder, DIEnumerator) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M(new Module("MyModule", Ctx));
  DIBuilder DIB(*M);
  APSInt I1(APInt(32, 1));
  APSInt I2(APInt(33, 1));

  auto *E = DIEnumerator::get(Ctx, I1, I1.isSigned(), "name");
  EXPECT_TRUE(E);

  auto *E1 = DIEnumerator::getIfExists(Ctx, I1, I1.isSigned(), "name");
  EXPECT_TRUE(E1);

  auto *E2 = DIEnumerator::getIfExists(Ctx, I2, I1.isSigned(), "name");
  EXPECT_FALSE(E2);
}

TEST(DbgAssignIntrinsicTest, replaceVariableLocationOp) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"(
    define dso_local void @fun(i32 %v1, ptr %p1, ptr %p2) !dbg !7 {
    entry:
      call void @llvm.dbg.assign(metadata i32 %v1, metadata !14, metadata !DIExpression(), metadata !17, metadata ptr %p1, metadata !DIExpression()), !dbg !16
      ret void
    }

    declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata)

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!3}

    !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
    !1 = !DIFile(filename: "test.cpp", directory: "/")
    !3 = !{i32 2, !"Debug Info Version", i32 3}
    !7 = distinct !DISubprogram(name: "fun", linkageName: "fun", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
    !8 = !DISubroutineType(types: !9)
    !9 = !{null}
    !10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
    !11 = !{}
    !14 = !DILocalVariable(name: "Local", scope: !7, file: !1, line: 3, type: !10)
    !16 = !DILocation(line: 0, scope: !7)
    !17 = distinct !DIAssignID()
    )");
  // Check the test IR isn't malformed.
  ASSERT_TRUE(M);

  Function &Fun = *M->getFunction("fun");
  Value *V1 = Fun.getArg(0);
  Value *P1 = Fun.getArg(1);
  Value *P2 = Fun.getArg(2);
  DbgAssignIntrinsic *DAI = cast<DbgAssignIntrinsic>(Fun.begin()->begin());
  ASSERT_TRUE(V1 == DAI->getVariableLocationOp(0));
  ASSERT_TRUE(P1 == DAI->getAddress());

#define TEST_REPLACE(Old, New, ExpectedValue, ExpectedAddr)                    \
  DAI->replaceVariableLocationOp(Old, New);                                    \
  EXPECT_EQ(DAI->getVariableLocationOp(0), ExpectedValue);                     \
  EXPECT_EQ(DAI->getAddress(), ExpectedAddr);

  // Replace address only.
  TEST_REPLACE(/*Old*/ P1, /*New*/ P2, /*Value*/ V1, /*Address*/ P2);
  // Replace value only.
  TEST_REPLACE(/*Old*/ V1, /*New*/ P2, /*Value*/ P2, /*Address*/ P2);
  // Replace both.
  TEST_REPLACE(/*Old*/ P2, /*New*/ P1, /*Value*/ P1, /*Address*/ P1);

  // Replace address only, value uses a DIArgList.
  // Value = {DIArgList(V1)}, Addr = P1.
  DAI->setRawLocation(DIArgList::get(C, ValueAsMetadata::get(V1)));
  DAI->setExpression(DIExpression::get(
      C, {dwarf::DW_OP_LLVM_arg, 0, dwarf::DW_OP_stack_value}));
  TEST_REPLACE(/*Old*/ P1, /*New*/ P2, /*Value*/ V1, /*Address*/ P2);
#undef TEST_REPLACE
}

TEST(IsHeterogeneousDebugTest, V3Module) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"(
    !llvm.module.flags = !{!0}
    !0 = !{i32 2, !"Debug Info Version", i32 3}
)");
  EXPECT_FALSE(isHeterogeneousDebug(*M));
}

TEST(IsHeterogeneousDebugTest, V4Module) {
  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"(
    !llvm.module.flags = !{!0}
    !0 = !{i32 2, !"Debug Info Version", i32 4}
)");
  EXPECT_TRUE(isHeterogeneousDebug(*M));
}

TEST(AssignmentTrackingTest, InstrMethods) {
  // Test the assignment tracking Instruction methods.
  // This includes:
  //     Instruction::mergeDIAssignID

  LLVMContext C;
  std::unique_ptr<Module> M = parseIR(C, R"(
    define dso_local void @fun() #0 !dbg !8 {
    entry:
      %Local = alloca [2 x i32], align 4, !DIAssignID !12
      call void @llvm.dbg.assign(metadata i1 undef, metadata !13, metadata !DIExpression(), metadata !12, metadata [2 x i32]* %Local, metadata !DIExpression()), !dbg !18
      %arrayidx = getelementptr inbounds [2 x i32], [2 x i32]* %Local, i64 0, i64 0, !dbg !19
      store i32 5, i32* %arrayidx, align 4, !dbg !20, !DIAssignID !21
      call void @llvm.dbg.assign(metadata i32 5, metadata !13, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32), metadata !21, metadata i32* %arrayidx, metadata !DIExpression()), !dbg !18
      %arrayidx1 = getelementptr inbounds [2 x i32], [2 x i32]* %Local, i64 0, i64 1, !dbg !22
      store i32 6, i32* %arrayidx1, align 4, !dbg !23, !DIAssignID !24
      call void @llvm.dbg.assign(metadata i32 6, metadata !13, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32), metadata !24, metadata i32* %arrayidx1, metadata !DIExpression()), !dbg !18
      ret void, !dbg !25
    }

    declare void @llvm.dbg.assign(metadata, metadata, metadata, metadata, metadata, metadata) #1

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!2, !3, !4, !5, !6}
    !llvm.ident = !{!7}

    !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 14.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
    !1 = !DIFile(filename: "test.cpp", directory: "/")
    !2 = !{i32 7, !"Dwarf Version", i32 5}
    !3 = !{i32 2, !"Debug Info Version", i32 3}
    !4 = !{i32 1, !"wchar_size", i32 4}
    !5 = !{i32 7, !"uwtable", i32 1}
    !6 = !{i32 7, !"frame-pointer", i32 2}
    !7 = !{!"clang version 14.0.0"}
    !8 = distinct !DISubprogram(name: "fun", linkageName: "fun", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !11)
    !9 = !DISubroutineType(types: !10)
    !10 = !{null}
    !11 = !{}
    !12 = distinct !DIAssignID()
    !13 = !DILocalVariable(name: "Local", scope: !8, file: !1, line: 2, type: !14)
    !14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !15, size: 64, elements: !16)
    !15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
    !16 = !{!17}
    !17 = !DISubrange(count: 2)
    !18 = !DILocation(line: 0, scope: !8)
    !19 = !DILocation(line: 3, column: 3, scope: !8)
    !20 = !DILocation(line: 3, column: 12, scope: !8)
    !21 = distinct !DIAssignID()
    !22 = !DILocation(line: 4, column: 3, scope: !8)
    !23 = !DILocation(line: 4, column: 12, scope: !8)
    !24 = distinct !DIAssignID()
    !25 = !DILocation(line: 5, column: 1, scope: !8)
  )");

  // Check the test IR isn't malformed.
  ASSERT_TRUE(M);
  Function &Fun = *M->getFunction("fun");
  SmallVector<Instruction *> Stores;
  for (auto &BB : Fun) {
    for (auto &I : BB) {
      if (isa<StoreInst>(&I))
        Stores.push_back(&I);
    }
  }

  // The test requires (at least) 2 stores.
  ASSERT_TRUE(Stores.size() == 2);
  // Use SetVectors to check that the attachments and markers are unique
  // (another test requirement).
  SetVector<Metadata *> OrigIDs;
  SetVector<DbgAssignIntrinsic *> Markers;
  for (const Instruction *SI : Stores) {
    Metadata *ID = SI->getMetadata(LLVMContext::MD_DIAssignID);
    ASSERT_TRUE(OrigIDs.insert(ID));
    ASSERT_TRUE(ID != nullptr);
    auto Range = at::getAssignmentMarkers(SI);
    ASSERT_TRUE(std::distance(Range.begin(), Range.end()) == 1);
    ASSERT_TRUE(Markers.insert(*Range.begin()));
  }

  // Test 1 - mergeDIAssignID.
  //
  // Input            store0->mergeDIAssignID(store1)
  // -----            -------------------------
  // store0 !x        store0 !x
  // dbg.assign0 !x   dbg.assign !x
  // store1 !y        store1 !x
  // dbg.assign1 !y   dbg.assign1 !x
  {
    Stores[0]->mergeDIAssignID(Stores[1]);
    // Check that the stores share the same ID.
    Metadata *NewID0 = Stores[0]->getMetadata(LLVMContext::MD_DIAssignID);
    Metadata *NewID1 = Stores[1]->getMetadata(LLVMContext::MD_DIAssignID);
    EXPECT_NE(NewID0, nullptr);
    EXPECT_EQ(NewID0, NewID1);
    EXPECT_EQ(Markers[0]->getAssignID(), NewID0);
    EXPECT_EQ(Markers[1]->getAssignID(), NewID0);
  }

  // Test 2 - mergeDIAssignID.
  //
  // Input            store0->mergeDIAssignID(store1)
  // -----            -------------------------
  // store0 !x        store0 !x
  // dbg.assign0 !x   dbg.assign !x
  // store1           store1
  {
    Stores[1]->setMetadata(LLVMContext::MD_DIAssignID, nullptr);
    Stores[0]->mergeDIAssignID(Stores[1]);
    // Check that store1 doesn't get a new ID.
    Metadata *NewID0 = Stores[0]->getMetadata(LLVMContext::MD_DIAssignID);
    Metadata *NewID1 = Stores[1]->getMetadata(LLVMContext::MD_DIAssignID);
    EXPECT_NE(NewID0, nullptr);
    EXPECT_EQ(NewID1, nullptr);
    EXPECT_EQ(Markers[0]->getAssignID(), NewID0);
  }

  // Test 3 - mergeDIAssignID.
  //
  // Input            store1->mergeDIAssignID(store0)
  // -----            -------------------------
  // store0 !x        store0 !x
  // dbg.assign0 !x   dbg.assign !x
  // store1           store1 !x
  {
    Stores[1]->setMetadata(LLVMContext::MD_DIAssignID, nullptr);
    Stores[1]->mergeDIAssignID(Stores[0]);
    // Check that the stores share the same ID (note store1 starts with none).
    Metadata *NewID0 = Stores[0]->getMetadata(LLVMContext::MD_DIAssignID);
    Metadata *NewID1 = Stores[1]->getMetadata(LLVMContext::MD_DIAssignID);
    EXPECT_NE(NewID0, nullptr);
    EXPECT_EQ(NewID0, NewID1);
    EXPECT_EQ(Markers[0]->getAssignID(), NewID0);
  }

  // Test 4 - mergeDIAssignID.
  //
  // Input            store1->mergeDIAssignID(store0)
  // -----            -------------------------
  // store0 !x        store0 !x
  // dbg.assign0 !x   dbg.assign !x
  // store1 !x        store1 !x
  {
    Stores[0]->mergeDIAssignID(Stores[1]);
    // Check that the stores share the same ID.
    Metadata *NewID0 = Stores[0]->getMetadata(LLVMContext::MD_DIAssignID);
    Metadata *NewID1 = Stores[1]->getMetadata(LLVMContext::MD_DIAssignID);
    EXPECT_NE(NewID0, nullptr);
    EXPECT_EQ(NewID0, NewID1);
    EXPECT_EQ(Markers[0]->getAssignID(), NewID0);
  }

  // Test 5 - dropUnknownNonDebugMetadata.
  //
  // Input            store0->dropUnknownNonDebugMetadata()
  // -----            -------------------------
  // store0 !x        store0 !x
  {
    Stores[0]->dropUnknownNonDebugMetadata();
    Metadata *NewID0 = Stores[0]->getMetadata(LLVMContext::MD_DIAssignID);
    EXPECT_NE(NewID0, nullptr);
  }
}

} // end namespace
