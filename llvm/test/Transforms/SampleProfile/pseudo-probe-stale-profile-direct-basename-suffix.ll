; Test that direct basename matching handles IR names carrying clone suffixes
; (e.g. ".llvm.N" added by ThinLTO promotion), which the Itanium demangler
; does not treat as function encodings. The pseudo probe GUIDs are those of
; the canonical names, as in a ThinLTO backend. The coroutine clone
; _Z3fool.resume must not make the basename "foo" ambiguous.

; REQUIRES: x86_64-linux
; RUN: llvm-profdata merge --sample --extbinary %S/Inputs/pseudo-probe-stale-profile-direct-basename-suffix.prof -o %t.prof
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%t.prof --salvage-stale-profile --salvage-unused-profile -S | FileCheck %s

; CHECK: define dso_local void @_Z3fool.llvm.7(i64 %y) {{.*}} !prof ![[#FOO:]]
; CHECK: define dso_local void @_Z3barl.part.0(i64 %y) {{.*}} !prof ![[#BAR:]]
; CHECK: define dso_local void @_Z3bazl.__uniq.123(i64 %y) {{.*}} !prof ![[#BAZ:]]
; CHECK: define dso_local void @_Z3quxl.cfi(i64 %y) {{.*}} !prof ![[#QUX:]]
; CHECK-DAG: ![[#FOO]] = !{!"function_entry_count", i64 52}
; CHECK-DAG: ![[#BAR]] = !{!"function_entry_count", i64 53}
; CHECK-DAG: ![[#BAZ]] = !{!"function_entry_count", i64 54}
; CHECK-DAG: ![[#QUX]] = !{!"function_entry_count", i64 55}

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @_Z3fool.llvm.7(i64 %y) #0 !dbg !9 {
entry:
  call void @llvm.pseudoprobe(i64 5326982120444056491, i64 1, i32 0, i64 -1), !dbg !12
  ret void, !dbg !12
}

define dso_local void @_Z3barl.part.0(i64 %y) #0 !dbg !13 {
entry:
  call void @llvm.pseudoprobe(i64 -9164787269840974918, i64 1, i32 0, i64 -1), !dbg !14
  ret void, !dbg !14
}

define dso_local void @_Z3bazl.__uniq.123(i64 %y) #0 !dbg !15 {
entry:
  call void @llvm.pseudoprobe(i64 -2504982094396869733, i64 1, i32 0, i64 -1), !dbg !16
  ret void, !dbg !16
}

define dso_local void @_Z3quxl.cfi(i64 %y) #0 !dbg !17 {
entry:
  call void @llvm.pseudoprobe(i64 2523411590769414898, i64 1, i32 0, i64 -1), !dbg !18
  ret void, !dbg !18
}

define dso_local void @_Z3fool.resume(i64 %y) #0 {
entry:
  ret void
}

define dso_local void @caller() #0 !dbg !19 {
entry:
  call void @llvm.pseudoprobe(i64 -1768971689307247648, i64 1, i32 0, i64 -1), !dbg !20
  call void @_Z3fool.llvm.7(i64 0), !dbg !21
  call void @_Z3barl.part.0(i64 0), !dbg !23
  call void @_Z3bazl.__uniq.123(i64 0), !dbg !25
  call void @_Z3quxl.cfi(i64 0), !dbg !27
  ret void, !dbg !29
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.pseudoprobe(i64 immarg, i64 immarg, i32 immarg, i64 immarg) #1

attributes #0 = { "use-sample-profile" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.pseudo_probe_desc = !{!4, !5, !6, !7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/tmp")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i64 5326982120444056491, i64 4294967295, !"_Z3fool"}
!5 = !{i64 -9164787269840974918, i64 4294967295, !"_Z3barl"}
!6 = !{i64 -2504982094396869733, i64 4294967295, !"_Z3bazl"}
!7 = !{i64 2523411590769414898, i64 4294967295, !"_Z3quxl"}
!8 = !{i64 -1768971689307247648, i64 1125904201809919, !"caller"}
!9 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fool", scope: !1, file: !1, line: 3, type: !10, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{}
!12 = !DILocation(line: 4, column: 1, scope: !9)
!13 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barl", scope: !1, file: !1, line: 13, type: !10, scopeLine: 13, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!14 = !DILocation(line: 14, column: 1, scope: !13)
!15 = distinct !DISubprogram(name: "baz", linkageName: "_Z3bazl", scope: !1, file: !1, line: 23, type: !10, scopeLine: 23, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!16 = !DILocation(line: 24, column: 1, scope: !15)
!17 = distinct !DISubprogram(name: "qux", linkageName: "_Z3quxl", scope: !1, file: !1, line: 33, type: !10, scopeLine: 33, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!18 = !DILocation(line: 34, column: 1, scope: !17)
!19 = distinct !DISubprogram(name: "caller", linkageName: "caller", scope: !1, file: !1, line: 7, type: !10, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!20 = !DILocation(line: 9, column: 1, scope: !19)
!21 = !DILocation(line: 9, column: 1, scope: !22)
!22 = !DILexicalBlockFile(scope: !19, file: !1, discriminator: 455082007)
!23 = !DILocation(line: 9, column: 1, scope: !24)
!24 = !DILexicalBlockFile(scope: !19, file: !1, discriminator: 455082015)
!25 = !DILocation(line: 9, column: 1, scope: !26)
!26 = !DILexicalBlockFile(scope: !19, file: !1, discriminator: 455082023)
!27 = !DILocation(line: 9, column: 1, scope: !28)
!28 = !DILexicalBlockFile(scope: !19, file: !1, discriminator: 455082031)
!29 = !DILocation(line: 10, column: 1, scope: !19)
