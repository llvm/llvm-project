; REQUIRES: x86_64-linux
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/profile-mismatch-indirect-call.prof -report-profile-staleness -persist-profile-staleness -S 2>%t -o %t.ll
; RUN: FileCheck %s --input-file %t
; RUN: FileCheck %s --input-file %t.ll -check-prefix=CHECK-MD

; The profile records one sampled target, "foo", and one unsampled target at
; both a direct call to "foo" and an indirect call. The direct call matches the
; sampled target, while the indirect IR call is compatible with any profiled
; target. Only the "baz" (IR) vs "qux" (profile) callsite at line offset 4 is
; a real mismatch.

; CHECK: (1/4) of callsites' profile are invalid and (100/400) of samples are discarded due to callsite location mismatch.

; CHECK-MD: ![[#]] = !{!"NumMismatchedCallsites", i64 1, !"NumRecoveredCallsites", i64 0, !"TotalProfiledCallsites", i64 4, !"MismatchedCallsiteSamples", i64 100, !"RecoveredCallsiteSamples", i64 0}

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@fp = dso_local local_unnamed_addr global ptr null, align 8

define dso_local void @test() local_unnamed_addr #0 !dbg !9 {
entry:
  tail call void @foo(), !dbg !12
  %0 = load ptr, ptr @fp, align 8, !dbg !13
  tail call void %0(), !dbg !13
  tail call void @bar(), !dbg !14
  tail call void @baz(), !dbg !15
  ret void, !dbg !16
}

declare void @foo() local_unnamed_addr

declare void @bar() local_unnamed_addr

declare void @baz() local_unnamed_addr

attributes #0 = { nounwind uwtable "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "test")
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 7, !"uwtable", i32 2}
!6 = !{!"clang"}
!9 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 5, type: !10, scopeLine: 5, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !DILocation(line: 6, column: 3, scope: !9)
!13 = !DILocation(line: 7, column: 3, scope: !9)
!14 = !DILocation(line: 8, column: 3, scope: !9)
!15 = !DILocation(line: 9, column: 3, scope: !9)
!16 = !DILocation(line: 10, column: 1, scope: !9)
