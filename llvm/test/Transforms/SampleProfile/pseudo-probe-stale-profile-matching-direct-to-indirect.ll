; REQUIRES: asserts && x86-registered-target
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/pseudo-probe-stale-profile-matching-direct-to-indirect.prof --salvage-stale-profile -S --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl 2>&1 | FileCheck %s

;; Test that stale profile matching correctly matches a direct call in the IR
;; against an UnknownIndirectCallee profile anchor when the direct callee is
;; one of the actual call targets that were collapsed.
;;
;; The profile has multiple call targets at location 2 (A:43 B:40), which
;; findProfileAnchors collapses into UnknownIndirectCallee. After code changes
;; that shifted probe IDs, the IR has a direct call to @A at probe 3. With the
;; AnchorTargetMap fix, the LCS matching recognizes that A is one of the saved
;; targets and aligns the callsites correctly.
;;
;; IR (probes):       [1,  2,  3(A),  4,  5(B),  6(C)]
;; Profile (locs):    [1,  2(A+B),    3,  4(B),  5(C)]
;;                              \
;;                               \ (collapsed to UnknownIndirectCallee,
;;                                  but AnchorTargetMap saves [A, B])
;;
;; LCS anchor matching:
;;   IR anchors:      [3(A),       5(B),  6(C)]
;;                     |           |       |
;;   Profile anchors: [2(indirect) 4(B),  5(C)]
;;
;; Full matching (anchors + non-anchors):
;;   [1,  2,  3(A),  4,  5(B),  6(C)]
;;    |   |     |    |     |      |
;;   [1,  2,  *2*,   3,   4(B), 5(C)]
;;          (rematch)
;;   Output: [3->2, 4->3, 5->4, 6->5]

; CHECK: Run stale profile matching for test_direct_call_matched_to_indirect
; CHECK-NEXT: Location is matched from 1 to 1
; CHECK-NEXT: Location is matched from 2 to 2
; CHECK-NEXT: Callsite with callee:A is matched from 3 to 2
; CHECK-NEXT: Location is rematched backwards from 2 to 1
; CHECK-NEXT: Location is matched from 4 to 3
; CHECK-NEXT: Callsite with callee:B is matched from 5 to 4
; CHECK-NEXT: Callsite with callee:C is matched from 6 to 5

define dso_local i32 @test_direct_call_matched_to_indirect(i32 %x) #0 !dbg !10 {
entry:
  call void @llvm.pseudoprobe(i64 -3371303159763101805, i64 1, i32 0, i64 -1), !dbg !14
  call void @llvm.pseudoprobe(i64 -3371303159763101805, i64 2, i32 0, i64 -1), !dbg !14
  %call = call i32 @A(i32 %x), !dbg !15
  %add = add i32 %x, %call
  call void @llvm.pseudoprobe(i64 -3371303159763101805, i64 4, i32 0, i64 -1), !dbg !14
  %call1 = call i32 @B(i32 %add), !dbg !17
  %add2 = add i32 %add, %call1
  %call3 = call i32 @C(i32 %add2), !dbg !19
  %add4 = add i32 %add2, %call3
  ret i32 %add4
}

declare !dbg !21 i32 @A(i32)
declare !dbg !22 i32 @B(i32)
declare !dbg !23 i32 @C(i32)

attributes #0 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.pseudo_probe_desc = !{!24}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, debugInfoForProfiling: true)
!1 = !DIFile(filename: "test.c", directory: "/home/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!10 = distinct !DISubprogram(name: "test_direct_call_matched_to_indirect", scope: !1, file: !1, line: 10, type: !11, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DILocation(line: 0, scope: !10)
!15 = !DILocation(line: 13, column: 8, scope: !16)
!16 = !DILexicalBlockFile(scope: !10, file: !1, discriminator: 186646559)
!17 = !DILocation(line: 15, column: 8, scope: !18)
!18 = !DILexicalBlockFile(scope: !10, file: !1, discriminator: 186646575)
!19 = !DILocation(line: 16, column: 8, scope: !20)
!20 = !DILexicalBlockFile(scope: !10, file: !1, discriminator: 186646583)
!21 = !DISubprogram(name: "A", scope: !1, file: !1, line: 2, type: !11, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!22 = !DISubprogram(name: "B", scope: !1, file: !1, line: 3, type: !11, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!23 = !DISubprogram(name: "C", scope: !1, file: !1, line: 4, type: !11, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!24 = !{i64 -3371303159763101805, i64 281547712924884, !"test_direct_call_matched_to_indirect"}
