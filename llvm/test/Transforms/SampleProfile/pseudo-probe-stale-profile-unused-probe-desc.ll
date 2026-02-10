; REQUIRES: asserts && x86-registered-target
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/pseudo-probe-stale-profile-unused-probe-desc.prof --salvage-stale-profile --salvage-unused-profile -S --debug-only=sample-profile,sample-profile-matcher,sample-profile-impl --min-call-count-for-cg-matching=0 --min-func-count-for-cg-matching=0 2>&1 | FileCheck %s

;; Test scenario (post-link):
;;
;; int main() {
;;   // optimized away in pre-link
;;   foo_hot();
;;   if (false)
;;     foo_cold();
;;   return 0;
;; }
;;
;; - Profile: main calls foo_hot, but never calls foo_cold.
;; - Post-link IR: foo_hot was optimized away in pre-link and its probe descriptor persists
;;   in !llvm.pseudo_probe_desc.
;; - Expected result: foo_cold is not matched to foo_hot's profile.

; CHECK: Function foo_cold is not in profile or profile symbol list.
; CHECK: Run stale profile matching for main
; CHECK-NOT: Function:foo_cold matches profile:foo_hot

define dso_local void @foo_cold() #0 !dbg !11 {
entry:
  call void @llvm.pseudoprobe(i64 5929568435854063828, i64 1, i32 0, i64 -1), !dbg !13
  ret void, !dbg !14
}

define dso_local i32 @main() #0 !dbg !15 {
entry:
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 1, i32 0, i64 -1), !dbg !18
  br i1 false, label %if.then, label %if.end, !dbg !23

if.then:
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 2, i32 0, i64 -1), !dbg !24
  call void @foo_cold(), !dbg !19
  br label %if.end

if.end:
  call void @llvm.pseudoprobe(i64 -2624081020897602054, i64 4, i32 0, i64 -1), !dbg !25
  ret i32 0, !dbg !21
}

declare void @llvm.pseudoprobe(i64, i64, i32, i64)

attributes #0 = { nounwind uwtable "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.pseudo_probe_desc = !{!6, !7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 7, !"uwtable", i32 2}
!6 = !{i64 5929568435854063828, i64 844429225099263, !"foo_cold"}
!7 = !{i64 -2624081020897602054, i64 1126003093360596, !"main"}
!8 = !{i64 -2908166055299645123, i64 844429225099263, !"foo_hot"}
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = distinct !DISubprogram(name: "foo_cold", scope: !1, file: !1, line: 2, type: !9, scopeLine: 2, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!12 = !DISubroutineType(types: !22)
!13 = !DILocation(line: 3, column: 1, scope: !11)
!14 = !DILocation(line: 4, column: 1, scope: !11)
!15 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 10, type: !12, scopeLine: 10, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!18 = !DILocation(line: 11, column: 1, scope: !15)
!19 = !DILocation(line: 13, column: 5, scope: !20)
!20 = !DILexicalBlockFile(scope: !15, file: !1, discriminator: 186646559)
!21 = !DILocation(line: 16, column: 1, scope: !15)
!22 = !{!2}
!23 = !DILocation(line: 12, column: 3, scope: !15)
!24 = !DILocation(line: 13, column: 3, scope: !15)
!25 = !DILocation(line: 15, column: 1, scope: !15)
!26 = distinct !DISubprogram(name: "foo_hot", scope: !1, file: !1, line: 5, type: !9, scopeLine: 5, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
