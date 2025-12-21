;; This test verifies that we assign a deterministic location for merged
;; instructions when -pick-merged-source-locations is enabled. We use the
;; simplifycfg pass to test this behaviour since it was a common source of
;; merged instructions, however we intend this to apply to all users of the
;; getMergedLocation API.

;; Run simplifycfg and check that only 1 call to bar remains and it's debug
;; location has a valid line number (lexicographically smallest).
; RUN: opt %s -passes=simplifycfg -hoist-common-insts -pick-merged-source-locations -S | FileCheck %s --check-prefix=ENABLED
; ENABLED: call i32 @bar{{.*!dbg !}}[[TAG:[0-9]+]]
; ENABLED-NOT: call i32 @bar
; ENABLED: ![[TAG]] = !DILocation(line: 9, column: 16, scope: !9)

;; Run simplifycfg without the pass to ensure that we don't spuriously start
;; passing the test if simplifycfg behaviour changes.
; RUN: opt %s -passes=simplifycfg -hoist-common-insts -pick-merged-source-locations=false -S | FileCheck %s --check-prefix=DISABLED
; DISABLED: call i32 @bar{{.*!dbg !}}[[TAG:[0-9]+]]
; DISABLED-NOT: call i32 @bar
; DISABLED: ![[TAG]] = !DILocation(line: 0, scope: !9)

; ModuleID = '../llvm/test/DebugInfo/Inputs/debug-info-merge-call.c'
source_filename = "../llvm/test/DebugInfo/Inputs/debug-info-merge-call.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define dso_local i32 @test(i32 %n) !dbg !9 {
entry:
  %call = call i32 @foo(i32 %n), !dbg !12
  %cmp1 = icmp sgt i32 %n, 100, !dbg !13
  br i1 %cmp1, label %if.then, label %if.else, !dbg !13

if.then:                                          ; preds = %entry
  %call2 = call i32 @bar(i32 %n), !dbg !14
  %add = add nsw i32 %call2, %call, !dbg !15
  br label %if.end, !dbg !16

if.else:                                          ; preds = %entry
  %call4 = call i32 @bar(i32 %n), !dbg !17
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %r.0 = phi i32 [ %add, %if.then ], [ %call4, %if.else ], !dbg !18
  ret i32 %r.0, !dbg !19
}

declare !dbg !20 i32 @foo(i32)

declare !dbg !21 i32 @bar(i32)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 21.0.0git (git@github.com:snehasish/llvm-project.git 6ce41db6b0275d060d6e60f88b96a1657024345c)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "../llvm/test/DebugInfo/Inputs/debug-info-merge-call.c", directory: "/usr/local/google/home/snehasishk/working/llvm-project/build-assert", checksumkind: CSK_MD5, checksum: "ac1be6c40dad11691922d600f9d55c55")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{!"clang version 21.0.0git (git@github.com:snehasish/llvm-project.git 6ce41db6b0275d060d6e60f88b96a1657024345c)"}
!9 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 5, type: !10, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{}
!12 = !DILocation(line: 7, column: 13, scope: !9)
!13 = !DILocation(line: 8, column: 8, scope: !9)
!14 = !DILocation(line: 9, column: 16, scope: !9)
!15 = !DILocation(line: 9, column: 14, scope: !9)
!16 = !DILocation(line: 10, column: 3, scope: !9)
!17 = !DILocation(line: 11, column: 10, scope: !9)
!18 = !DILocation(line: 0, scope: !9)
!19 = !DILocation(line: 13, column: 3, scope: !9)
!20 = !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !10, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!21 = !DISubprogram(name: "bar", scope: !1, file: !1, line: 1, type: !10, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)

