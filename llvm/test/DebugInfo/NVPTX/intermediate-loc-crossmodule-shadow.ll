; RUN: llc -O0 < %s -mtriple=nvptx64-nvidia-cuda \
; RUN:   | FileCheck %s --implicit-check-not=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
; RUN:                  --implicit-check-not=kernelA-ir-text
;;
;; Variant of intermediate-loc-inline-shadow.ll where the two frames' layers
;; point into different intermediate files. The enclosing frame's layer is not
;; emitted, so its file leaves no trace: no .file entry and no source text.
;;
;;   kernelB : 60  (head)  -- layer in B.ir, line 200 (innermost, emitted)
;;     -> kernelA : 20     -- layer in A.ir, line 50  (enclosing, dropped)

;; The head's own layer is emitted.
; CHECK: .loc [[SRC:[0-9]+]] 60 5
; CHECK-NEXT: .loc_intermediate [[BINT:[0-9]+]] 200 3
;; kernelA's layer is not emitted.
; CHECK-NOT: .loc_intermediate

; CHECK: .file [[SRC]] "/k{{/|\\\\}}kernel.py"
;; Only B.ir gets a .file. The --implicit-check-not patterns reject
;; A.ir's name and source anywhere in the output.
; CHECK: .file [[BINT]] ".{{/|\\\\}}bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

;; One code_block, for B.ir only.
; CHECK: .nv_intermediate_source_section {
; CHECK-NEXT: .code_block {
; CHECK-NEXT: .ir_name: "intermediate ir"
; CHECK-NEXT: .sourceFileName: [[BINT]]
; CHECK-NEXT: .source_begin
; CHECK-NEXT: kernelB-ir-text
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

;; kernelA (the function) and kernelB (inlined into it).
!6 = distinct !DISubprogram(name: "kernelA", scope: !1, file: !1, line: 10, type: !4, spFlags: DISPFlagDefinition, unit: !0)
!7 = distinct !DISubprogram(name: "kernelB", scope: !1, file: !1, line: 40, type: !4, spFlags: DISPFlagDefinition, unit: !0)

;; kernelB's layer, in B.ir: innermost, emitted.
!14 = !DIFile(filename: "B.ir", directory: ".", checksumkind: CSK_MD5, checksum: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", source: "kernelB-ir-text")
!15 = !DILayerLoc(line: 200, column: 3, file: !14, kind: "intermediate ir")
!16 = !DILayerLocList(!15)

;; kernelA's layer, in A.ir: enclosing, not emitted.
!24 = !DIFile(filename: "A.ir", directory: ".", checksumkind: CSK_MD5, checksum: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", source: "kernelA-ir-text")
!25 = !DILayerLoc(line: 50, column: 1, file: !24, kind: "intermediate ir")
!26 = !DILayerLocList(!25)

;; The inlined-at chain: kernelB (head), then kernelA's call site.
!9 = !DILocation(line: 60, column: 5, scope: !7, inlinedAt: !10, irlayers: !16)
!10 = !DILocation(line: 20, column: 1, scope: !6, irlayers: !26)
!8 = !DILocation(line: 10, column: 1, scope: !6)
