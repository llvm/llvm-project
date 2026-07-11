; The profile has two same-basename bar profiles. LCS matching first matches
; _Z3barv to _Z3bari, then also observes that _Z3barv matches _Z3barl while
; matching the caller anchors. Rewriting _Z3barv's saved profile mapping to
; _Z3barl causes both _Z3barv and _Z3barPv to use the same location map and
; trips "Run stale profile matching only once per function".

; REQUIRES: asserts
; RUN: llvm-profdata merge --sample --extbinary %S/Inputs/pseudo-probe-stale-profile-lcs-anchor-overwrite.prof -o %t.prof
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%t.prof --salvage-stale-profile --salvage-unused-profile -S -o /dev/null

define ptr @_Z6calleePv() #0 !dbg !5 {
entry:
  call void @llvm.pseudoprobe(i64 7108221232740920931, i64 1, i32 0, i64 -1), !dbg !9
  ret ptr null
}

define void @_Z3barv() #0 !dbg !6 {
entry:
  call void @llvm.pseudoprobe(i64 -1069303473483922844, i64 1, i32 0, i64 -1), !dbg !10
  ret void
}

define ptr @_Z3barPv() #0 !dbg !7 {
entry:
  call void @llvm.pseudoprobe(i64 5678655469166311522, i64 1, i32 0, i64 -1), !dbg !11
  %call = call ptr @_Z6calleePv(), !dbg !12
  ret ptr %call
}

define ptr @_Z3foov() #0 !dbg !8 {
entry:
  call void @llvm.pseudoprobe(i64 9191153033785521275, i64 1, i32 0, i64 -1), !dbg !13
  call void null(), !dbg !14
  call void @_Z3barv(), !dbg !15
  call void null(), !dbg !16
  %call = call ptr @_Z3barPv(), !dbg !17
  ret ptr %call
}

declare void @llvm.pseudoprobe(i64, i64, i32, i64) #1

attributes #0 = { "use-sample-profile" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}
!llvm.pseudo_probe_desc = !{!18, !19, !20, !21}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cc", directory: "/tmp")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DISubroutineType(types: !4)
!4 = !{}
!5 = distinct !DISubprogram(name: "callee", linkageName: "_Z6calleePv", scope: !1, file: !1, line: 1, type: !3, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!6 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !1, file: !1, line: 5, type: !3, scopeLine: 5, spFlags: DISPFlagDefinition, unit: !0)
!7 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barPv", scope: !1, file: !1, line: 7, type: !3, scopeLine: 7, spFlags: DISPFlagDefinition, unit: !0)
!8 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 11, type: !3, scopeLine: 11, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DILocation(line: 2, column: 5, scope: !5)
!10 = !DILocation(line: 5, column: 13, scope: !6)
!11 = !DILocation(line: 8, column: 12, scope: !7)
!12 = !DILocation(line: 8, column: 12, scope: !22)
!13 = !DILocation(line: 12, column: 5, scope: !8)
!14 = !DILocation(line: 12, column: 5, scope: !23)
!15 = !DILocation(line: 13, column: 5, scope: !24)
!16 = !DILocation(line: 14, column: 5, scope: !25)
!17 = !DILocation(line: 15, column: 12, scope: !26)
!18 = !{i64 7108221232740920931, i64 4294967295, !"_Z6calleePv"}
!19 = !{i64 -1069303473483922844, i64 4294967295, !"_Z3barv"}
!20 = !{i64 5678655469166311522, i64 281479271677951, !"_Z3barPv"}
!21 = !{i64 9191153033785521275, i64 1125904201809919, !"_Z3foov"}
!22 = !DILexicalBlockFile(scope: !7, file: !1, discriminator: 455082007)
!23 = !DILexicalBlockFile(scope: !8, file: !1, discriminator: 387973143)
!24 = !DILexicalBlockFile(scope: !8, file: !1, discriminator: 455082015)
!25 = !DILexicalBlockFile(scope: !8, file: !1, discriminator: 387973159)
!26 = !DILexicalBlockFile(scope: !8, file: !1, discriminator: 455082031)
