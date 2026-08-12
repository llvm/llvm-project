; RUN: llc -O0 < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s
;;
;; Two stacked layer-bearing frames in ONE inlined-at chain. The front end
;; inlined funcB into funcA (snapshot on the funcA frame), and funcA was then
;; inlined into kernelA -- which ALSO carried its own snapshot layer. So the
;; store's chain is:
;;
;;   funcB : 52   (head)   -- no irlayers
;;     -> funcA : 38       -- irlayers: tile ir @ 45   (INNER, wins)
;;       -> kernelA : 15   -- irlayers: tile ir @ 12   (OUTER, shadowed)
;;
;; The producer disassembles the whole module into ONE frozen snapshot text, so both the
;; funcA op (line 45) and the kernelA call op (line 12) index into the SAME
;; intermediate DIFile -- they differ only by (line, col), not by file. The
;; shadowing is therefore line-level: the NVPTX backend walks head -> outward
;; to the FIRST (innermost) layer-bearing frame and stops, so the funcA line (45)
;; is emitted and the kernelA line (12) never is. (Distinct intermediate DIFiles
;; only arise across separately-snapshotted modules.)

;; Primary .loc = the head (funcB, line 52); the winning intermediate layer is
;; funcA's (line 45), read off a middle (non-head, non-outermost) frame.
; CHECK: .loc [[SRC:[0-9]+]] 52 3
; CHECK-NEXT: .loc_intermediate [[INT:[0-9]+]] 45 7
;; Exactly one .loc_intermediate: the shadowed kernelA line (12) is not emitted.
;; (A wrong outermost-frame walk would print `12 1` here; emitting both frames
;; would add a second .loc_intermediate -- either way this fails.)
; CHECK-NOT: .loc_intermediate

; CHECK: .file [[SRC]] "/k{{/|\\\\}}kernel.py"
;; A single intermediate .file (the shared module snapshot), named by its carried
;; checksum digest (not MD5(filename)).
; CHECK: .file [[INT]] ".{{/|\\\\}}aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

;; One code_block for the one shared snapshot file.
; CHECK: .nv_intermediate_source_section {
; CHECK-NEXT: .code_block {
; CHECK-NEXT: .ir_name: "tile ir"
; CHECK-NEXT: .sourceFileName: [[INT]]
; CHECK-NEXT: .source_begin
; CHECK-NEXT: whole-module-tile-ir-snapshot
; CHECK-NEXT: .source_end
; CHECK-NEXT: }
; CHECK-NEXT: }

define dso_local ptx_kernel void @test_kernel(ptr noundef %v) #0 !dbg !6 {
entry:
  store ptr null, ptr %v, align 8, !dbg !9
  ret void, !dbg !8
}

attributes #0 = { noinline optnone "target-cpu"="sm_75" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "tile", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "kernel.py", directory: "/k")
!2 = !{i32 7, !"Dwarf Version", i32 2}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !5)
!5 = !{null}

;; Three source scopes: kernelA (the function), funcA, funcB.
!6 = distinct !DISubprogram(name: "kernelA", scope: !1, file: !1, line: 10, type: !4, spFlags: DISPFlagDefinition, unit: !0)
!7 = distinct !DISubprogram(name: "funcA", scope: !1, file: !1, line: 30, type: !4, spFlags: DISPFlagDefinition, unit: !0)
!12 = distinct !DISubprogram(name: "funcB", scope: !1, file: !1, line: 50, type: !4, spFlags: DISPFlagDefinition, unit: !0)

;; ONE whole-module tile-IR snapshot DIFile. Both frames' layers reference it,
;; at different (line, col): funcA's op at 45:7, kernelA's call op at 12:1.
!14 = !DIFile(filename: "module.tileir", directory: ".", checksumkind: CSK_MD5, checksum: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", source: "whole-module-tile-ir-snapshot")

;; funcA's layer (the INNER layer that wins).
!15 = !DILayerLoc(line: 45, column: 7, file: !14, kind: "tile ir")
!16 = !DILayerLocList(!15)

;; kernelA's layer (the OUTER layer that is SHADOWED) -- SAME file, different line.
!25 = !DILayerLoc(line: 12, column: 1, file: !14, kind: "tile ir")
!26 = !DILayerLocList(!25)

;; The inlined-at chain: funcB head (no layer) -> funcA frame (layer 45) ->
;; kernelA frame (layer 12, shadowed / outermost).
!9 = !DILocation(line: 52, column: 3, scope: !12, inlinedAt: !10)
!10 = !DILocation(line: 38, column: 7, scope: !7, inlinedAt: !11, irlayers: !16)
!11 = !DILocation(line: 15, column: 1, scope: !6, irlayers: !26)
!8 = !DILocation(line: 10, column: 1, scope: !6)
