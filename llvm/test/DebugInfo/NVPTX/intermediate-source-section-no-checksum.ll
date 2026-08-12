; RUN: llc -O0 < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s
;;
;; An intermediate file that carries source but NO checksum still needs a unique
;; hash for its secondary .file name -- ptxas stores the .code_block's .source in
;; the cubin keyed by that name -- so emission hashes the DIFile's directory and
;; filename instead of using the path verbatim. Contrast
;; intermediate-checksum-secondary-file.ll, where a checksum is present and that
;; content-addressed digest is used instead.

;; MD5(".kernel.tileir") == directory "." concatenated with filename.
; CHECK-DAG: .loc_intermediate [[F:[0-9]+]] 42 5
; CHECK-DAG: .file [[F]] ".{{/|\\\\}}f3c6d19eaf8d63898bcec70cb38e2482"

;; ...and the source still reaches the section.
; CHECK: .nv_intermediate_source_section {
; CHECK-NEXT: .code_block {
; CHECK-NEXT: .ir_name: "tile ir"
; CHECK-NEXT: .sourceFileName: [[F]]
; CHECK-NEXT: .source_begin
; CHECK-NEXT: tile ir source
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

;; Source present, checksum absent.
!14 = !DIFile(filename: "kernel.tileir", directory: ".", source: "tile ir source")
!30 = !DILayerLocList(!31)
!31 = !DILayerLoc(line: 42, column: 5, file: !14, kind: "tile ir")

!20 = !DILocation(line: 2, column: 5, scope: !8, irlayers: !30)
!21 = !DILocation(line: 3, column: 1, scope: !8)
