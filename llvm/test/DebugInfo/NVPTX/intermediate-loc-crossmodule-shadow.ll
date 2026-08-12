; RUN: llc -O0 < %s -mtriple=nvptx64-nvidia-cuda \
; RUN:   | FileCheck %s --implicit-check-not=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
; RUN:                  --implicit-check-not=kernelA-snapshot-text
;;
;; Cross-module variant of the shadowing test: a
;; separately-snapshotted kernelB is inlined POST-snapshot into a
;; separately-snapshotted kernelA, so the two layer-bearing frames reference
;; DIFFERENT intermediate DIFiles:
;;
;;   kernelB : 60  (head)  -- irlayers: B.tileir @ 200  (INNER, wins)
;;     -> kernelA : 20     -- irlayers: A.tileir @ 50   (OUTER, shadowed)
;;
;; the NVPTX backend stops at the first layer-bearing frame (the head here),
;; so B.tileir is recorded/emitted and A.tileir is never touched -- proving the
;; shadowed frame's whole DIFile (its .file entry AND its source) is absent, not
;; just its line. Only the inlined op is present, so A.tileir appears solely on
;; the shadowed frame; a native kernelA op would legitimately emit A.tileir, but
;; that is out of scope for isolating the shadow.

;; Primary .loc = head (kernelB, line 60); intermediate = kernelB's B.tileir layer.
; CHECK: .loc [[SRC:[0-9]+]] 60 5
; CHECK-NEXT: .loc_intermediate [[BINT:[0-9]+]] 200 3
;; Only one .loc_intermediate: the shadowed kernelA A.tileir layer is not emitted.
; CHECK-NOT: .loc_intermediate

; CHECK: .file [[SRC]] "/k{{/|\\\\}}kernel.py"
;; Only B.tileir's checksum-named .file appears; A.tileir's (aaaa...) is shadowed
;; out entirely (see the --implicit-check-not patterns on the RUN line).
; CHECK: .file [[BINT]] ".{{/|\\\\}}bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

;; One code_block, for B.tileir only.
; CHECK: .nv_intermediate_source_section {
; CHECK-NEXT: .code_block {
; CHECK-NEXT: .ir_name: "tile ir"
; CHECK-NEXT: .sourceFileName: [[BINT]]
; CHECK-NEXT: .source_begin
; CHECK-NEXT: kernelB-snapshot-text
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

;; kernelA (the function) and kernelB (inlined into it post-snapshot).
!6 = distinct !DISubprogram(name: "kernelA", scope: !1, file: !1, line: 10, type: !4, spFlags: DISPFlagDefinition, unit: !0)
!7 = distinct !DISubprogram(name: "kernelB", scope: !1, file: !1, line: 40, type: !4, spFlags: DISPFlagDefinition, unit: !0)

;; kernelB's OWN module snapshot (B.tileir) -- the INNER layer that wins.
!14 = !DIFile(filename: "B.tileir", directory: ".", checksumkind: CSK_MD5, checksum: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", source: "kernelB-snapshot-text")
!15 = !DILayerLoc(line: 200, column: 3, file: !14, kind: "tile ir")
!16 = !DILayerLocList(!15)

;; kernelA's OWN module snapshot (A.tileir) -- the OUTER layer, shadowed.
!24 = !DIFile(filename: "A.tileir", directory: ".", checksumkind: CSK_MD5, checksum: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", source: "kernelA-snapshot-text")
!25 = !DILayerLoc(line: 50, column: 1, file: !24, kind: "tile ir")
!26 = !DILayerLocList(!25)

;; Chain: kernelB op (head, B.tileir layer) inlinedAt kernelA call site
;; (A.tileir layer, shadowed / outermost).
!9 = !DILocation(line: 60, column: 5, scope: !7, inlinedAt: !10, irlayers: !16)
!10 = !DILocation(line: 20, column: 1, scope: !6, irlayers: !26)
!8 = !DILocation(line: 10, column: 1, scope: !6)
