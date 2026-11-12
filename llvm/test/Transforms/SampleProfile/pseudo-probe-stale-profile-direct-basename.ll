; Test direct basename matching for orphan functions.
;
; IR has _Z3fool (foo(long)) — orphan, called only from caller() which
; has no profile. Profile has _Z3fooi (foo(int)) — unused top-level entry.
; Direct basename matching should match _Z3fool -> _Z3fooi.

; REQUIRES: x86_64-linux
; REQUIRES: asserts
; RUN: llvm-profdata merge --sample --extbinary %S/Inputs/pseudo-probe-stale-profile-direct-basename.prof -o %t.prof
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%t.prof --salvage-stale-profile --salvage-unused-profile -S --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl 2>&1 | FileCheck %s

; CHECK: Function _Z3fool is not in profile or profile symbol list.
; CHECK: Direct basename match: _Z3fool (IR) -> _Z3fooi (Profile) [basename: foo]
; CHECK: Direct basename matching found 1 matches
; CHECK: Run stale profile matching for _Z3fool

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @_Z3fool(i64 %y) #0 !dbg !11 {
entry:
  call void @llvm.pseudoprobe(i64 5326982120444056491, i64 1, i32 0, i64 -1), !dbg !14
  ret void, !dbg !15
}

define dso_local void @caller() #0 !dbg !16 {
entry:
  call void @llvm.pseudoprobe(i64 -7421642274262752513, i64 1, i32 0, i64 -1), !dbg !18
  call void @_Z3fool(i64 0), !dbg !19
  ret void, !dbg !20
}

declare void @llvm.pseudoprobe(i64, i64, i32, i64)

attributes #0 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.pseudo_probe_desc = !{!9, !10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/tmp")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 7, !"uwtable", i32 2}
!9 = !{i64 5326982120444056491, i64 4294967295, !"_Z3fool"}
!10 = !{i64 -7421642274262752513, i64 4294967295, !"caller"}
!11 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fool", scope: !1, file: !1, line: 3, type: !12, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!12 = !DISubroutineType(types: !13)
!13 = !{}
!14 = !DILocation(line: 4, column: 1, scope: !11)
!15 = !DILocation(line: 5, column: 1, scope: !11)
!16 = distinct !DISubprogram(name: "caller", linkageName: "caller", scope: !1, file: !1, line: 7, type: !12, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!18 = !DILocation(line: 8, column: 1, scope: !16)
!19 = !DILocation(line: 9, column: 1, scope: !16)
!20 = !DILocation(line: 10, column: 1, scope: !16)
