; RUN: llc -O0 < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s
;;
;; When more than one frame of an inlined-at chain carries layers, the innermost
;; one is emitted: it is the most specific intermediate position for the
;; instruction. Layers on the frames enclosing it are not emitted.
;;
;;   funcB : 52   (head)   -- no layers
;;     -> funcA : 38       -- layer at line 45 (innermost, emitted)
;;       -> kernelA : 15   -- layer at line 12 (enclosing, not emitted)
;;
;; Both layers point into the same intermediate file, so only the line tells
;; them apart.

;; The primary .loc is the head's. The layer is funcA's, a frame that is neither
;; the head nor the outermost.
; CHECK: .loc [[SRC:[0-9]+]] 52 3
; CHECK-NEXT: .loc_intermediate [[INT:[0-9]+]] 45 7
;; kernelA's layer (line 12) is not emitted.
; CHECK-NOT: .loc_intermediate

; CHECK: .file [[SRC]] "/k{{/|\\\\}}kernel.py"
;; One intermediate .file, shared by both layers.
; CHECK: .file [[INT]] ".{{/|\\\\}}aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

;; One code_block for that file.
; CHECK: .nv_intermediate_source_section {
; CHECK-NEXT: .code_block {
; CHECK-NEXT: .ir_name: "intermediate ir"
; CHECK-NEXT: .sourceFileName: [[INT]]
; CHECK-NEXT: .source_begin
; CHECK-NEXT: whole-module-ir-text
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

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "kernel.py", directory: "/k")
!2 = !{i32 7, !"Dwarf Version", i32 2}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !5)
!5 = !{null}

;; Three source scopes: kernelA (the function), funcA, funcB.
!6 = distinct !DISubprogram(name: "kernelA", scope: !1, file: !1, line: 10, type: !4, spFlags: DISPFlagDefinition, unit: !0)
!7 = distinct !DISubprogram(name: "funcA", scope: !1, file: !1, line: 30, type: !4, spFlags: DISPFlagDefinition, unit: !0)
!12 = distinct !DISubprogram(name: "funcB", scope: !1, file: !1, line: 50, type: !4, spFlags: DISPFlagDefinition, unit: !0)

;; One intermediate file; both layers point into it.
!14 = !DIFile(filename: "module.ir", directory: ".", checksumkind: CSK_MD5, checksum: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", source: "whole-module-ir-text")

;; funcA's layer: innermost, emitted.
!15 = !DILayerLoc(line: 45, column: 7, file: !14, kind: "intermediate ir")
!16 = !DILayerLocList(!15)

;; kernelA's layer: enclosing, not emitted.
!25 = !DILayerLoc(line: 12, column: 1, file: !14, kind: "intermediate ir")
!26 = !DILayerLocList(!25)

;; The inlined-at chain: funcB (no layer), funcA (line 45), kernelA (line 12).
!9 = !DILocation(line: 52, column: 3, scope: !12, inlinedAt: !10)
!10 = !DILocation(line: 38, column: 7, scope: !7, inlinedAt: !11, irlayers: !16)
!11 = !DILocation(line: 15, column: 1, scope: !6, irlayers: !26)
!8 = !DILocation(line: 10, column: 1, scope: !6)
