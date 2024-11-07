//===- MemProfUseTest.cpp - MemProf use tests -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/ProfileData/MemProf.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Instrumentation/MemProfiler.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {
using namespace llvm;
using namespace llvm::memprof;
using testing::FieldsAre;
using testing::Pair;
using testing::SizeIs;

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

  auto Calls = extractCallsFromIR(*M);

  // Expect exactly one caller.
  ASSERT_THAT(Calls, SizeIs(1));

  auto It = Calls.begin();
  ASSERT_NE(It, Calls.end());

  const auto &[CallerGUID, CallSites] = *It;
  EXPECT_EQ(CallerGUID, IndexedMemProfRecord::getGUID("_Z3foov"));
  ASSERT_THAT(CallSites, SizeIs(3));

  // Verify that call sites show up in the ascending order of their source
  // locations.
  EXPECT_THAT(CallSites[0],
              Pair(FieldsAre(1U, 3U), IndexedMemProfRecord::getGUID("_Z2f1v")));
  EXPECT_THAT(CallSites[1],
              Pair(FieldsAre(2U, 3U), IndexedMemProfRecord::getGUID("_Z2f2v")));
  EXPECT_THAT(CallSites[2],
              Pair(FieldsAre(2U, 9U), IndexedMemProfRecord::getGUID("_Z2f3v")));
}
} // namespace
