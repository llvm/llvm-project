//===- MemProfUseTest.cpp - MemProf use tests -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/ProfileData/InstrProfWriter.h"
#include "llvm/ProfileData/MemProf.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Transforms/Instrumentation/MemProfiler.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace llvm {
namespace memprof {
namespace {
using testing::Contains;
using testing::ElementsAre;
using testing::Pair;
using testing::SizeIs;
using testing::UnorderedElementsAre;

TEST(MemProf, ExtractDirectCallsFromIR) {
  // The following IR is generated from:
  //
  // void f1();
  // void f2();
  // void f3();
  //
  // void foo() {
  //   f1();
  //   f2(); f3();
  // }
  StringRef IR = R"IR(
define dso_local void @_Z3foov() !dbg !10 {
entry:
  call void @_Z2f1v(), !dbg !13
  call void @_Z2f2v(), !dbg !14
  call void @_Z2f3v(), !dbg !15
  ret void, !dbg !16
}

declare !dbg !17 void @_Z2f1v()

declare !dbg !18 void @_Z2f2v()

declare !dbg !19 void @_Z2f3v()

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "foobar.cc", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 1, !"MemProfProfileFilename", !"memprof.profraw"}
!6 = !{i32 8, !"PIC Level", i32 2}
!7 = !{i32 7, !"PIE Level", i32 2}
!8 = !{i32 7, !"uwtable", i32 2}
!9 = !{!"clang"}
!10 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 5, type: !11, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{}
!13 = !DILocation(line: 6, column: 3, scope: !10)
!14 = !DILocation(line: 7, column: 3, scope: !10)
!15 = !DILocation(line: 7, column: 9, scope: !10)
!16 = !DILocation(line: 8, column: 1, scope: !10)
!17 = !DISubprogram(name: "f1", linkageName: "_Z2f1v", scope: !1, file: !1, line: 1, type: !11, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!18 = !DISubprogram(name: "f2", linkageName: "_Z2f2v", scope: !1, file: !1, line: 2, type: !11, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!19 = !DISubprogram(name: "f3", linkageName: "_Z2f3v", scope: !1, file: !1, line: 3, type: !11, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
)IR";

  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(IR, Err, Ctx);
  ASSERT_TRUE(M);

  auto *F = M->getFunction("_Z3foov");
  ASSERT_NE(F, nullptr);

  TargetLibraryInfoWrapperPass WrapperPass;
  auto &TLI = WrapperPass.getTLI(*F);
  auto Calls = extractCallsFromIR(*M, TLI);

  // Expect exactly one caller.
  ASSERT_THAT(Calls, SizeIs(1));

  auto It = Calls.begin();
  ASSERT_NE(It, Calls.end());

  const auto &[CallerGUID, CallSites] = *It;
  EXPECT_EQ(CallerGUID, IndexedMemProfRecord::getGUID("_Z3foov"));

  // Verify that call sites show up in the ascending order of their source
  // locations.
  EXPECT_THAT(
      CallSites,
      ElementsAre(
          Pair(LineLocation(1, 3), IndexedMemProfRecord::getGUID("_Z2f1v")),
          Pair(LineLocation(2, 3), IndexedMemProfRecord::getGUID("_Z2f2v")),
          Pair(LineLocation(2, 9), IndexedMemProfRecord::getGUID("_Z2f3v"))));
}

TEST(MemProf, ExtractDirectCallsFromIRInline) {
  // The following IR is generated from:
  //
  // void f1();
  // static inline void f2() {
  //   // For an interesting line number.
  //   f1();
  // }
  // static inline void f3() {
  //   /****/ f2();  // For an interesting column number.
  // }
  //
  // void g1();
  // void g2();
  // static inline void g3() {
  //   /**/ g1();  // For an interesting column number.
  //   g2();
  // }
  //
  // void foo() {
  //   f3();
  //   /***/ g3();  // For an interesting column number.
  // }
  StringRef IR = R"IR(
define dso_local void @_Z3foov() local_unnamed_addr !dbg !10 {
entry:
  tail call void @_Z2f1v(), !dbg !13
  tail call void @_Z2g1v(), !dbg !18
  tail call void @_Z2g2v(), !dbg !21
  ret void, !dbg !22
}

declare !dbg !23 void @_Z2f1v() local_unnamed_addr

declare !dbg !24 void @_Z2g1v() local_unnamed_addr

declare !dbg !25 void @_Z2g2v() local_unnamed_addr

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "foobar.cc", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 1, !"MemProfProfileFilename", !"memprof.profraw"}
!6 = !{i32 8, !"PIC Level", i32 2}
!7 = !{i32 7, !"PIE Level", i32 2}
!8 = !{i32 7, !"uwtable", i32 2}
!9 = !{!"clang"}
!10 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 17, type: !11, scopeLine: 17, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{}
!13 = !DILocation(line: 4, column: 3, scope: !14, inlinedAt: !15)
!14 = distinct !DISubprogram(name: "f2", linkageName: "_ZL2f2v", scope: !1, file: !1, line: 2, type: !11, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!15 = distinct !DILocation(line: 7, column: 10, scope: !16, inlinedAt: !17)
!16 = distinct !DISubprogram(name: "f3", linkageName: "_ZL2f3v", scope: !1, file: !1, line: 6, type: !11, scopeLine: 6, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!17 = distinct !DILocation(line: 18, column: 3, scope: !10)
!18 = !DILocation(line: 13, column: 8, scope: !19, inlinedAt: !20)
!19 = distinct !DISubprogram(name: "g3", linkageName: "_ZL2g3v", scope: !1, file: !1, line: 12, type: !11, scopeLine: 12, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!20 = distinct !DILocation(line: 19, column: 9, scope: !10)
!21 = !DILocation(line: 14, column: 3, scope: !19, inlinedAt: !20)
!22 = !DILocation(line: 20, column: 1, scope: !10)
!23 = !DISubprogram(name: "f1", linkageName: "_Z2f1v", scope: !1, file: !1, line: 1, type: !11, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!24 = !DISubprogram(name: "g1", linkageName: "_Z2g1v", scope: !1, file: !1, line: 10, type: !11, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!25 = !DISubprogram(name: "g2", linkageName: "_Z2g2v", scope: !1, file: !1, line: 11, type: !11, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
)IR";

  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(IR, Err, Ctx);
  ASSERT_TRUE(M);

  auto *F = M->getFunction("_Z3foov");
  ASSERT_NE(F, nullptr);

  TargetLibraryInfoWrapperPass WrapperPass;
  auto &TLI = WrapperPass.getTLI(*F);
  auto Calls = extractCallsFromIR(*M, TLI);

  // Expect exactly 4 callers.
  ASSERT_THAT(Calls, SizeIs(4));

  // Verify each key-value pair.

  auto FooIt = Calls.find(IndexedMemProfRecord::getGUID("_Z3foov"));
  ASSERT_NE(FooIt, Calls.end());
  const auto &[FooCallerGUID, FooCallSites] = *FooIt;
  EXPECT_EQ(FooCallerGUID, IndexedMemProfRecord::getGUID("_Z3foov"));
  EXPECT_THAT(
      FooCallSites,
      ElementsAre(
          Pair(LineLocation(1, 3), IndexedMemProfRecord::getGUID("_ZL2f3v")),
          Pair(LineLocation(2, 9), IndexedMemProfRecord::getGUID("_ZL2g3v"))));

  auto F2It = Calls.find(IndexedMemProfRecord::getGUID("_ZL2f2v"));
  ASSERT_NE(F2It, Calls.end());
  const auto &[F2CallerGUID, F2CallSites] = *F2It;
  EXPECT_EQ(F2CallerGUID, IndexedMemProfRecord::getGUID("_ZL2f2v"));
  EXPECT_THAT(F2CallSites,
              ElementsAre(Pair(LineLocation(2, 3),
                               IndexedMemProfRecord::getGUID("_Z2f1v"))));

  auto F3It = Calls.find(IndexedMemProfRecord::getGUID("_ZL2f3v"));
  ASSERT_NE(F3It, Calls.end());
  const auto &[F3CallerGUID, F3CallSites] = *F3It;
  EXPECT_EQ(F3CallerGUID, IndexedMemProfRecord::getGUID("_ZL2f3v"));
  EXPECT_THAT(F3CallSites,
              ElementsAre(Pair(LineLocation(1, 10),
                               IndexedMemProfRecord::getGUID("_ZL2f2v"))));

  auto G3It = Calls.find(IndexedMemProfRecord::getGUID("_ZL2g3v"));
  ASSERT_NE(G3It, Calls.end());
  const auto &[G3CallerGUID, G3CallSites] = *G3It;
  EXPECT_EQ(G3CallerGUID, IndexedMemProfRecord::getGUID("_ZL2g3v"));
  EXPECT_THAT(
      G3CallSites,
      ElementsAre(
          Pair(LineLocation(1, 8), IndexedMemProfRecord::getGUID("_Z2g1v")),
          Pair(LineLocation(2, 3), IndexedMemProfRecord::getGUID("_Z2g2v"))));
}

TEST(MemProf, ExtractDirectCallsFromIRCallingNew) {
  // The following IR is generated from:
  //
  // int *foo() {
  //   return ::new (int);
  // }
  StringRef IR = R"IR(
define dso_local noundef ptr @_Z3foov() #0 !dbg !10 {
entry:
  %call = call noalias noundef nonnull ptr @_Znwm(i64 noundef 4) #2, !dbg !13
  ret ptr %call, !dbg !14
}

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znwm(i64 noundef) #1

attributes #0 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nobuiltin allocsize(0) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { builtin allocsize(0) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "foobar.cc", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 1, !"MemProfProfileFilename", !"memprof.profraw"}
!6 = !{i32 8, !"PIC Level", i32 2}
!7 = !{i32 7, !"PIE Level", i32 2}
!8 = !{i32 7, !"uwtable", i32 2}
!9 = !{!"clang"}
!10 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{}
!13 = !DILocation(line: 2, column: 10, scope: !10)
!14 = !DILocation(line: 2, column: 3, scope: !10)
)IR";

  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(IR, Err, Ctx);
  ASSERT_TRUE(M);

  auto *F = M->getFunction("_Z3foov");
  ASSERT_NE(F, nullptr);

  TargetLibraryInfoWrapperPass WrapperPass;
  auto &TLI = WrapperPass.getTLI(*F);
  auto Calls = extractCallsFromIR(*M, TLI);

  // Expect exactly one caller.
  ASSERT_THAT(Calls, SizeIs(1));

  // Verify each key-value pair.

  auto FooIt = Calls.find(IndexedMemProfRecord::getGUID("_Z3foov"));
  ASSERT_NE(FooIt, Calls.end());
  const auto &[FooCallerGUID, FooCallSites] = *FooIt;
  EXPECT_EQ(FooCallerGUID, IndexedMemProfRecord::getGUID("_Z3foov"));
  EXPECT_THAT(FooCallSites, ElementsAre(Pair(LineLocation(1, 10), 0)));
}

// Populate those fields returned by getHotColdSchema.
MemInfoBlock makePartialMIB() {
  MemInfoBlock MIB;
  MIB.AllocCount = 1;
  MIB.TotalSize = 5;
  MIB.TotalLifetime = 10;
  MIB.TotalLifetimeAccessDensity = 23;
  return MIB;
}

IndexedMemProfRecord
makeRecordV2(std::initializer_list<CallStackId> AllocFrames,
             std::initializer_list<CallStackId> CallSiteFrames,
             const MemInfoBlock &Block, const MemProfSchema &Schema) {
  IndexedMemProfRecord MR;
  for (const auto &CSId : AllocFrames)
    MR.AllocSites.emplace_back(CSId, Block, Schema);
  for (const auto &CSId : CallSiteFrames)
    MR.CallSiteIds.push_back(CSId);
  return MR;
}

static const auto Err = [](Error E) {
  FAIL() << E;
  consumeError(std::move(E));
};

// Make sure that we can undrift direct calls.
TEST(MemProf, ComputeUndriftingMap) {
  // Suppose that the source code has changed from:
  //
  //   void bar();
  //   void baz();
  //   void zzz();
  //
  //   void foo() {
  //     /**/ bar();  // LineLocation(1, 8)
  //     zzz();       // LineLocation(2, 3)
  //     baz();       // LineLocation(3, 3)
  //   }
  //
  // to:
  //
  //   void bar();
  //   void baz();
  //
  //   void foo() {
  //     bar();        // LineLocation(1, 3)
  //     /**/ baz();   // LineLocation(2, 8)
  //   }
  //
  // Notice that the calls to bar and baz have drifted while zzz has been
  // removed.
  StringRef IR = R"IR(
define dso_local void @_Z3foov() #0 !dbg !10 {
entry:
  call void @_Z3barv(), !dbg !13
  call void @_Z3bazv(), !dbg !14
  ret void, !dbg !15
}

declare !dbg !16 void @_Z3barv() #1

declare !dbg !17 void @_Z3bazv() #1

attributes #0 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "foobar.cc", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 1, !"MemProfProfileFilename", !"memprof.profraw"}
!6 = !{i32 8, !"PIC Level", i32 2}
!7 = !{i32 7, !"PIE Level", i32 2}
!8 = !{i32 7, !"uwtable", i32 2}
!9 = !{!"clang"}
!10 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 4, type: !11, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{}
!13 = !DILocation(line: 5, column: 3, scope: !10)
!14 = !DILocation(line: 6, column: 8, scope: !10)
!15 = !DILocation(line: 7, column: 1, scope: !10)
!16 = !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !1, file: !1, line: 1, type: !11, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!17 = !DISubprogram(name: "baz", linkageName: "_Z3bazv", scope: !1, file: !1, line: 2, type: !11, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
)IR";

  LLVMContext Ctx;
  SMDiagnostic SMErr;
  std::unique_ptr<Module> M = parseAssemblyString(IR, SMErr, Ctx);
  ASSERT_TRUE(M);

  auto *F = M->getFunction("_Z3foov");
  ASSERT_NE(F, nullptr);

  TargetLibraryInfoWrapperPass WrapperPass;
  auto &TLI = WrapperPass.getTLI(*F);
  auto Calls = extractCallsFromIR(*M, TLI);

  uint64_t GUIDFoo = IndexedMemProfRecord::getGUID("_Z3foov");
  uint64_t GUIDBar = IndexedMemProfRecord::getGUID("_Z3barv");
  uint64_t GUIDBaz = IndexedMemProfRecord::getGUID("_Z3bazv");
  uint64_t GUIDZzz = IndexedMemProfRecord::getGUID("_Z3zzzv");

  // Verify that extractCallsFromIR extracts caller-callee pairs as expected.
  EXPECT_THAT(Calls,
              UnorderedElementsAre(Pair(
                  GUIDFoo, ElementsAre(Pair(LineLocation(1, 3), GUIDBar),
                                       Pair(LineLocation(2, 8), GUIDBaz)))));

  llvm::InstrProfWriter Writer;
  std::unique_ptr<IndexedInstrProfReader> Reader;

  const MemInfoBlock MIB = makePartialMIB();

  Writer.setMemProfVersionRequested(Version3);
  Writer.setMemProfFullSchema(false);

  ASSERT_THAT_ERROR(Writer.mergeProfileKind(InstrProfKind::MemProf),
                    Succeeded());

  const IndexedMemProfRecord IndexedMR = makeRecordV2(
      /*AllocFrames=*/{0x111, 0x222, 0x333},
      /*CallSiteFrames=*/{}, MIB, getHotColdSchema());

  IndexedMemProfData MemProfData;
  // The call sites within foo.
  MemProfData.Frames.try_emplace(0, GUIDFoo, 1, 8, false);
  MemProfData.Frames.try_emplace(1, GUIDFoo, 2, 3, false);
  MemProfData.Frames.try_emplace(2, GUIDFoo, 3, 3, false);
  // Line/column numbers below don't matter.
  MemProfData.Frames.try_emplace(3, GUIDBar, 9, 9, false);
  MemProfData.Frames.try_emplace(4, GUIDZzz, 9, 9, false);
  MemProfData.Frames.try_emplace(5, GUIDBaz, 9, 9, false);
  MemProfData.CallStacks.try_emplace(
      0x111, std::initializer_list<FrameId>{3, 0}); // bar called by foo
  MemProfData.CallStacks.try_emplace(
      0x222, std::initializer_list<FrameId>{4, 1}); // zzz called by foo
  MemProfData.CallStacks.try_emplace(
      0x333, std::initializer_list<FrameId>{5, 2}); // baz called by foo
  MemProfData.Records.try_emplace(0x9999, IndexedMR);
  Writer.addMemProfData(MemProfData, Err);

  auto Profile = Writer.writeBuffer();

  auto ReaderOrErr =
      IndexedInstrProfReader::create(std::move(Profile), nullptr);
  EXPECT_THAT_ERROR(ReaderOrErr.takeError(), Succeeded());
  Reader = std::move(ReaderOrErr.get());

  // Verify that getMemProfCallerCalleePairs extracts caller-callee pairs as
  // expected.
  auto Pairs = Reader->getMemProfCallerCalleePairs();
  ASSERT_THAT(Pairs, SizeIs(4));
  ASSERT_THAT(
      Pairs,
      Contains(Pair(GUIDFoo, ElementsAre(Pair(LineLocation(1, 8), GUIDBar),
                                         Pair(LineLocation(2, 3), GUIDZzz),
                                         Pair(LineLocation(3, 3), GUIDBaz)))));

  // Verify that computeUndriftMap identifies undrifting opportunities:
  //
  //   Profile                 IR
  //   (Line: 1, Column: 8) -> (Line: 1, Column: 3)
  //   (Line: 3, Column: 3) -> (Line: 2, Column: 8)
  auto UndriftMap = computeUndriftMap(*M, Reader.get(), TLI);
  ASSERT_THAT(UndriftMap,
              UnorderedElementsAre(Pair(
                  GUIDFoo, UnorderedElementsAre(
                               Pair(LineLocation(1, 8), LineLocation(1, 3)),
                               Pair(LineLocation(3, 3), LineLocation(2, 8))))));
}
} // namespace
} // namespace memprof
} // namespace llvm
