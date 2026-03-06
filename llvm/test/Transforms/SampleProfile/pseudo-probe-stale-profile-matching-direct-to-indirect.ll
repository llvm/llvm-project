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

@c = external global i32, align 4

define dso_local i32 @test_direct_call_matched_to_indirect(i32 noundef %x) #0 !dbg !10 {
entry:
    #dbg_value(i32 %x, !15, !DIExpression(), !16)
  call void @llvm.pseudoprobe(i64 -3371303159763101805, i64 1, i32 0, i64 -1), !dbg !17
  call void @llvm.pseudoprobe(i64 -3371303159763101805, i64 2, i32 0, i64 -1), !dbg !18
  %call = call i32 @A(i32 noundef %x), !dbg !19
  %add = add nsw i32 %x, %call, !dbg !21
    #dbg_value(i32 %add, !15, !DIExpression(), !16)
  call void @llvm.pseudoprobe(i64 -3371303159763101805, i64 4, i32 0, i64 -1), !dbg !22
  %call1 = call i32 @B(i32 noundef %add), !dbg !23
  %add2 = add nsw i32 %add, %call1, !dbg !25
    #dbg_value(i32 %add2, !15, !DIExpression(), !16)
  %call3 = call i32 @C(i32 noundef %add2), !dbg !26
  %add4 = add nsw i32 %add2, %call3, !dbg !28
    #dbg_value(i32 %add4, !15, !DIExpression(), !16)
  ret i32 %add4, !dbg !29
}

declare !dbg !30 i32 @A(i32 noundef) #1
declare !dbg !31 i32 @B(i32 noundef) #1
declare !dbg !32 i32 @C(i32 noundef) #1

declare void @llvm.pseudoprobe(i64, i64, i32, i64) #2

attributes #0 = { nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "use-sample-profile" }
attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}
!llvm.pseudo_probe_desc = !{!33}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 19.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/home/", checksumkind: CSK_MD5, checksum: "be98aa946f37f0ad8d307c9121efe101")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang version 19.0.0"}
!10 = distinct !DISubprogram(name: "test_direct_call_matched_to_indirect", scope: !1, file: !1, line: 10, type: !11, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{!15}
!15 = !DILocalVariable(name: "x", arg: 1, scope: !10, file: !1, line: 10, type: !13)
!16 = !DILocation(line: 0, scope: !10)
!17 = !DILocation(line: 11, column: 10, scope: !10)
!18 = !DILocation(line: 12, column: 10, scope: !10)
!19 = !DILocation(line: 13, column: 8, scope: !20)
!20 = !DILexicalBlockFile(scope: !10, file: !1, discriminator: 186646559)
!21 = !DILocation(line: 13, column: 5, scope: !10)
!22 = !DILocation(line: 14, column: 10, scope: !10)
!23 = !DILocation(line: 15, column: 8, scope: !24)
!24 = !DILexicalBlockFile(scope: !10, file: !1, discriminator: 186646575)
!25 = !DILocation(line: 15, column: 5, scope: !10)
!26 = !DILocation(line: 16, column: 8, scope: !27)
!27 = !DILexicalBlockFile(scope: !10, file: !1, discriminator: 186646583)
!28 = !DILocation(line: 16, column: 5, scope: !10)
!29 = !DILocation(line: 17, column: 3, scope: !10)
!30 = !DISubprogram(name: "A", scope: !1, file: !1, line: 2, type: !11, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!31 = !DISubprogram(name: "B", scope: !1, file: !1, line: 3, type: !11, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!32 = !DISubprogram(name: "C", scope: !1, file: !1, line: 4, type: !11, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!33 = !{i64 -3371303159763101805, i64 281547712924884, !"test_direct_call_matched_to_indirect"}
