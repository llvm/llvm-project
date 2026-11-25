; RUN: llc -O0 < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s
;;
;; A DILocation may carry more than one layer, one per intermediate IR level the
;; program was lowered through. Check that each entry emits its own
;; .loc_intermediate, in list order and right after the primary .loc, and that
;; each contributes its own .code_block.

; CHECK: .loc [[SRC:[0-9]+]] 2 5
; CHECK-NEXT: .loc_intermediate [[HIGH:[0-9]+]] 100 1
; CHECK-NEXT: .loc_intermediate [[LOW:[0-9]+]] 7 3

;; Each layer file gets its own secondary .file, named by its checksum digest.
; CHECK-DAG: .file [[HIGH]] ".{{/|\\\\}}aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
; CHECK-DAG: .file [[LOW]] ".{{/|\\\\}}bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

;; Both levels appear in the source section, in first-reference order.
; CHECK: .nv_intermediate_source_section {
; CHECK-NEXT: .code_block {
; CHECK-NEXT: .ir_name: "high ir"
; CHECK-NEXT: .sourceFileName: [[HIGH]]
; CHECK-NEXT: .source_begin
; CHECK-NEXT: high ir text
; CHECK-NEXT: .source_end
; CHECK-NEXT: }
; CHECK-NEXT: .code_block {
; CHECK-NEXT: .ir_name: "low ir"
; CHECK-NEXT: .sourceFileName: [[LOW]]
; CHECK-NEXT: .source_begin
; CHECK-NEXT: low ir text
; CHECK-NEXT: .source_end
; CHECK-NEXT: }
; CHECK-NEXT: }

define dso_local ptx_kernel void @test_kernel(ptr noundef %v) #0 !dbg !8 {
entry:
  %v.addr = alloca ptr, align 8
  store ptr %v, ptr %v.addr, align 8, !dbg !20
  ret void, !dbg !21
}

attributes #0 = { noinline optnone "target-cpu"="sm_75" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cu", directory: "/test")
!2 = !{i32 7, !"Dwarf Version", i32 2}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DISubprogram(name: "test_kernel", scope: !1, file: !1, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DISubroutineType(types: !10)
!10 = !{null}

;; Two intermediate levels: a high-level IR lowered to a low-level IR.
!14 = !DIFile(filename: "high.ir", directory: ".", checksumkind: CSK_MD5, checksum: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", source: "high ir text")
!15 = !DIFile(filename: "low.ir", directory: ".", checksumkind: CSK_MD5, checksum: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", source: "low ir text")
!16 = !DILayerLoc(line: 100, column: 1, file: !14, kind: "high ir")
!17 = !DILayerLoc(line: 7, column: 3, file: !15, kind: "low ir")
!18 = !DILayerLocList(!16, !17)

!20 = !DILocation(line: 2, column: 5, scope: !8, irlayers: !18)
!21 = !DILocation(line: 3, column: 1, scope: !8)
