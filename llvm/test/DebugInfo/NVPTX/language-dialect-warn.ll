;; Verify NVPTX DWARF emission for the supported language dialects.
;;
;; Note: this file used to also exercise the "unknown dialect" warning path
;; via `dialect: 42`, but the assembly parser now rejects dialect values
;; outside `dwarf::DW_LLVM_LANG_DIALECT_max`. The NVPTX warning code remains
;; as a defensive safety net for programmatic IR construction and future
;; dialect-enum expansion, and any positive coverage of `simt` / `tile`
;; emission lives here (with broader IR round-trip coverage in
;; llvm/test/DebugInfo/language-dialect.ll).

; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda 2>&1 | FileCheck %s \
; RUN:     --implicit-check-not "warning: unknown NVPTX language dialect"

;; DW_LLVM_LANG_DIALECT_simt = 1
; CHECK: DW_TAG_compile_unit
; CHECK:      .b8 1 // DW_AT_LLVM_language_dialect
;; DW_LLVM_LANG_DIALECT_tile = 2
; CHECK: DW_TAG_compile_unit
; CHECK:      .b8 2 // DW_AT_LLVM_language_dialect

define void @kernel_simt() !dbg !15 {
  ret void, !dbg !16
}

define void @kernel_tile() !dbg !17 {
  ret void, !dbg !18
}

!llvm.dbg.cu = !{!1, !2}
!llvm.module.flags = !{!12, !13}

!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !19, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, dialect: DW_LLVM_LANG_DIALECT_simt)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, dialect: DW_LLVM_LANG_DIALECT_tile)
!19 = !DIFile(filename: "test-simt.cu", directory: "/tmp")
!20 = !DIFile(filename: "test-tile.cu", directory: "/tmp")
!4 = !{}
!9 = !DISubroutineType(types: !4)
!15 = distinct !DISubprogram(name: "kernel_simt", scope: !19, file: !19, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !4)
!16 = !DILocation(line: 1, column: 1, scope: !15)
!17 = distinct !DISubprogram(name: "kernel_tile", scope: !20, file: !20, line: 1, type: !9, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!18 = !DILocation(line: 1, column: 1, scope: !17)

!12 = !{i32 2, !"Dwarf Version", i32 2}
!13 = !{i32 2, !"Debug Info Version", i32 3}
