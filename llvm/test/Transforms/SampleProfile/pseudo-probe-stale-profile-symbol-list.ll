; Test that module functions are still found in the profile symbol list by both
; the loader and the stale profile matcher.
;
; _Z3fool has no profile but the profile has an unused _Z3fooi entry. It is in
; the symbol list, so the matcher must treat it as cold rather than renamed and
; not give it the 52 samples of its basename match.
;
; _Z3barl.llvm.123 is in the symbol list under its raw name, so the loader
; must treat it as cold (entry count 0) rather than unknown (-1).

; REQUIRES: x86_64-linux
; RUN: llvm-profdata merge --sample --extbinary --prof-sym-list=%S/Inputs/pseudo-probe-stale-profile-symbol-list.text %S/Inputs/pseudo-probe-stale-profile-direct-basename.prof -o %t.prof
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%t.prof --salvage-stale-profile --salvage-unused-profile -S | FileCheck %s

; CHECK: define dso_local void @_Z3fool(i64 %y) #[[#]] !dbg ![[#]] !prof ![[#COLD:]]
; CHECK: define dso_local void @_Z3barl.llvm.123(i64 %y) #[[#]] !dbg ![[#]] !prof ![[#COLD]]
; CHECK: define dso_local void @caller() #[[#]] !dbg ![[#]] !prof ![[#UNKNOWN:]]
; CHECK-DAG: ![[#COLD]] = !{!"function_entry_count", i64 0}
; CHECK-DAG: ![[#UNKNOWN]] = !{!"function_entry_count", i64 -1}

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @_Z3fool(i64 %y) #0 !dbg !11 {
entry:
  call void @llvm.pseudoprobe(i64 5326982120444056491, i64 1, i32 0, i64 -1), !dbg !14
  ret void, !dbg !15
}

define dso_local void @_Z3barl.llvm.123(i64 %y) #0 !dbg !21 {
entry:
  call void @llvm.pseudoprobe(i64 6900393724926961028, i64 1, i32 0, i64 -1), !dbg !22
  ret void, !dbg !23
}

define dso_local void @caller() #0 !dbg !16 {
entry:
  call void @llvm.pseudoprobe(i64 -7421642274262752513, i64 1, i32 0, i64 -1), !dbg !18
  call void @_Z3fool(i64 0), !dbg !19
  call void @_Z3barl.llvm.123(i64 0), !dbg !24
  ret void, !dbg !20
}

declare void @llvm.pseudoprobe(i64, i64, i32, i64)

attributes #0 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.pseudo_probe_desc = !{!9, !10, !25}

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
!20 = !DILocation(line: 11, column: 1, scope: !16)
!21 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barl.llvm.123", scope: !1, file: !1, line: 13, type: !12, scopeLine: 13, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!22 = !DILocation(line: 14, column: 1, scope: !21)
!23 = !DILocation(line: 15, column: 1, scope: !21)
!24 = !DILocation(line: 10, column: 1, scope: !16)
!25 = !{i64 6900393724926961028, i64 4294967295, !"_Z3barl.llvm.123"}
