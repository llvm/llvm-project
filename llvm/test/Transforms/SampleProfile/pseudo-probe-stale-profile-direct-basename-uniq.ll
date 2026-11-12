; Test direct basename matching against a profile whose names carry
; ".__uniq.N" suffixes (-funique-internal-linkage-names). The suffix hashes the
; source path, so moving a file renames every internal function in it; the
; profile below was collected before such a move. Because the profile has
; uniq names, getCanonicalFnName keeps ".__uniq.N", and _ZL3fool.__uniq.222.llvm.7
; is looked up as _ZL3fool.__uniq.222.

; REQUIRES: x86_64-linux
; RUN: llvm-profdata merge --sample --extbinary %S/Inputs/pseudo-probe-stale-profile-direct-basename-uniq.prof -o %t.prof
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%t.prof --salvage-stale-profile --salvage-unused-profile -S | FileCheck %s

; CHECK: define internal void @_ZL3bazl.__uniq.222(i64 %y) {{.*}} !prof ![[#BAZ:]]
; CHECK: define void @_ZL3fool.__uniq.222.llvm.7(i64 %y) {{.*}} !prof ![[#FOO:]]
; CHECK-DAG: ![[#BAZ]] = !{!"function_entry_count", i64 54}
; CHECK-DAG: ![[#FOO]] = !{!"function_entry_count", i64 52}

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define internal void @_ZL3bazl.__uniq.222(i64 %y) #0 !dbg !7 {
entry:
  call void @llvm.pseudoprobe(i64 -8017183693835869934, i64 1, i32 0, i64 -1), !dbg !10
  ret void, !dbg !10
}

define void @_ZL3fool.__uniq.222.llvm.7(i64 %y) #0 !dbg !11 {
entry:
  call void @llvm.pseudoprobe(i64 7381965757683757353, i64 1, i32 0, i64 -1), !dbg !12
  ret void, !dbg !12
}

define dso_local void @caller() #0 !dbg !13 {
entry:
  call void @llvm.pseudoprobe(i64 -1768971689307247648, i64 1, i32 0, i64 -1), !dbg !14
  call void @_ZL3bazl.__uniq.222(i64 0), !dbg !15
  call void @_ZL3fool.__uniq.222.llvm.7(i64 0), !dbg !17
  ret void, !dbg !19
}

declare void @llvm.pseudoprobe(i64 immarg, i64 immarg, i32 immarg, i64 immarg) #1

attributes #0 = { "use-sample-profile" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.pseudo_probe_desc = !{!4, !5, !6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/tmp")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i64 -8017183693835869934, i64 4294967295, !"_ZL3bazl.__uniq.222"}
!5 = !{i64 7381965757683757353, i64 4294967295, !"_ZL3fool.__uniq.222"}
!6 = !{i64 -1768971689307247648, i64 562954248388607, !"caller"}
!7 = distinct !DISubprogram(name: "baz", linkageName: "_ZL3bazl.__uniq.222", scope: !1, file: !1, line: 23, type: !8, scopeLine: 23, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DISubroutineType(types: !9)
!9 = !{}
!10 = !DILocation(line: 24, column: 1, scope: !7)
!11 = distinct !DISubprogram(name: "foo", linkageName: "_ZL3fool.__uniq.222", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!12 = !DILocation(line: 4, column: 1, scope: !11)
!13 = distinct !DISubprogram(name: "caller", linkageName: "caller", scope: !1, file: !1, line: 7, type: !8, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!14 = !DILocation(line: 9, column: 1, scope: !13)
!15 = !DILocation(line: 9, column: 1, scope: !16)
!16 = !DILexicalBlockFile(scope: !13, file: !1, discriminator: 455082007)
!17 = !DILocation(line: 10, column: 1, scope: !18)
!18 = !DILexicalBlockFile(scope: !13, file: !1, discriminator: 455082015)
!19 = !DILocation(line: 11, column: 1, scope: !13)
