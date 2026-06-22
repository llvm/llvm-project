;; Test DW_AT_LLVM_language_dialect on DW_TAG_compile_unit for NVPTX.
;;
;; Three compile units, each with one kernel:
;;   CU0 (default.cu): default_kernel - no dialect field
;;   CU1 (simt.cu): simt_kernel - explicit dialect: DW_LLVM_LANG_DIALECT_simt
;;   CU2 (tile.cu): tile_kernel - explicit dialect: DW_LLVM_LANG_DIALECT_tile
;;
;; DW_AT_LLVM_language_dialect is only emitted when
;; DW_LLVM_LANG_DIALECT_simt or DW_LLVM_LANG_DIALECT_tile is explicitly
;; specified.

;; --- IR round-trip ---
; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s --check-prefix=IR

; IR: define void @default_kernel()
; IR: define void @simt_kernel()
; IR: define void @tile_kernel()
; IR: !llvm.dbg.cu = !{!{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}}
; IR: !{{[0-9]+}} = distinct !DICompileUnit(
; IR-NOT: dialect:
; IR-SAME: )
; IR: !{{[0-9]+}} = distinct !DICompileUnit({{.*}}dialect: DW_LLVM_LANG_DIALECT_simt{{.*}})
; IR: !{{[0-9]+}} = distinct !DICompileUnit({{.*}}dialect: DW_LLVM_LANG_DIALECT_tile{{.*}})

; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda | FileCheck %s --check-prefix=DWARF

;; The default CU has no dialect attribute; only explicit CUs emit it.
;; NVPTX emits DW_AT_LLVM_language_dialect as enum values.
; DWARF: .section .debug_info
; DWARF: DW_TAG_compile_unit
; DWARF-NOT: DW_AT_LLVM_language_dialect
; DWARF: DW_TAG_compile_unit
;; DW_LLVM_LANG_DIALECT_simt = 1
; DWARF:      .b8 1 // DW_AT_LLVM_language_dialect
; DWARF: DW_TAG_compile_unit
;; DW_LLVM_LANG_DIALECT_tile = 2
; DWARF:      .b8 2 // DW_AT_LLVM_language_dialect

define void @default_kernel() !dbg !5 {
  ret void, !dbg !10
}

define void @simt_kernel() !dbg !15 {
  ret void, !dbg !16
}

define void @tile_kernel() !dbg !8 {
  ret void, !dbg !11
}

!llvm.dbg.cu = !{!0, !1, !2}
!llvm.module.flags = !{!12, !13}

;; CU for the default case (no dialect).
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!3 = !DIFile(filename: "default.cu", directory: "/tmp")
!4 = !{}
!9 = !DISubroutineType(types: !4)
!5 = distinct !DISubprogram(name: "default_kernel", scope: !3, file: !3, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !4)
!10 = !DILocation(line: 1, column: 1, scope: !5)

;; CU for explicit SIMT.
!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, dialect: DW_LLVM_LANG_DIALECT_simt)
!18 = !DIFile(filename: "simt.cu", directory: "/tmp")
!15 = distinct !DISubprogram(name: "simt_kernel", scope: !18, file: !18, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !4)
!16 = !DILocation(line: 1, column: 1, scope: !15)

;; CU for tile.
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, dialect: DW_LLVM_LANG_DIALECT_tile)
!6 = !DIFile(filename: "tile.cu", directory: "/tmp")
!8 = distinct !DISubprogram(name: "tile_kernel", scope: !6, file: !6, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!11 = !DILocation(line: 1, column: 1, scope: !8)

!12 = !{i32 2, !"Dwarf Version", i32 2}
!13 = !{i32 2, !"Debug Info Version", i32 3}
