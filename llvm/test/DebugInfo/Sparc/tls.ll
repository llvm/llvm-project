; RUN: llc -mtriple=sparcv9 -O0 < %s | FileCheck %s --check-prefixes=ASM,ASM64
; RUN: llc -mtriple=sparc -O0 < %s | FileCheck %s --check-prefixes=ASM,ASM32
; RUN: llc -mtriple=sparcv9 -O0 -filetype=obj < %s | llvm-readelf -r - | FileCheck %s --check-prefix=REL64
; RUN: llc -mtriple=sparc -O0 -filetype=obj < %s | llvm-readelf -r - | FileCheck %s --check-prefix=REL32
; RUN: llc -mtriple=sparcv9 -O0 -filetype=obj < %s | llvm-dwarfdump --debug-info - 2>&1 | \
; RUN:   FileCheck %s --check-prefixes=DWARF,DWARF64 --implicit-check-not=warning:
; RUN: llc -mtriple=sparc -O0 -filetype=obj < %s | llvm-dwarfdump --debug-info - 2>&1 | \
; RUN:   FileCheck %s --check-prefixes=DWARF,DWARF32 --implicit-check-not=warning:

;; The DWARF location of a TLS variable is a DTP-relative offset, not an
;; absolute address, as GCC emits it.

; ASM-LABEL:  .section .debug_info
; ASM:        DW_AT_location
; ASM64-NEXT: .byte 14
; ASM64-NEXT: .xword %r_tls_dtpoff64(tv)
; ASM32-NEXT: .byte 12
; ASM32-NEXT: .word %r_tls_dtpoff32(tv)
; ASM-NEXT:   .byte 224

; ASM:        DW_AT_location
; ASM64-NEXT: .byte 14
; ASM64-NEXT: .xword %r_tls_dtpoff64(lv)
; ASM32-NEXT: .byte 12
; ASM32-NEXT: .word %r_tls_dtpoff32(lv)
; ASM-NEXT:   .byte 224

; REL64:      Relocation section '.rela.debug_info'
; REL64:      R_SPARC_TLS_DTPOFF64 {{0+}} tv + 0
; REL64:      R_SPARC_TLS_DTPOFF64 {{0+}}4 lv + 0

; REL32:      Relocation section '.rela.debug_info'
; REL32:      R_SPARC_TLS_DTPOFF32 {{0+}} tv + 0
; REL32:      R_SPARC_TLS_DTPOFF32 {{0+}}4 lv + 0

;; llvm-dwarfdump must not warn about the R_SPARC_TLS_DTPOFF* relocations.
;; It does not apply relocations inside an expression, so the operand is not
;; checked here.
; DWARF:      DW_AT_name ("tv")
; DWARF64:    DW_AT_location (DW_OP_const8u 0x{{[0-9a-f]+}}, DW_OP_GNU_push_tls_address)
; DWARF32:    DW_AT_location (DW_OP_const4u 0x{{[0-9a-f]+}}, DW_OP_GNU_push_tls_address)
; DWARF:      DW_AT_name ("lv")
; DWARF64:    DW_AT_location (DW_OP_const8u 0x{{[0-9a-f]+}}, DW_OP_GNU_push_tls_address)
; DWARF32:    DW_AT_location (DW_OP_const4u 0x{{[0-9a-f]+}}, DW_OP_GNU_push_tls_address)

@tv = thread_local global i32 42, align 4, !dbg !0
@lv = internal thread_local global i32 1, align 4, !dbg !9

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!7, !8}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "tv", scope: !4, file: !2, line: 1, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "t.c", directory: "/tmp")
!3 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!4 = distinct !DICompileUnit(language: DW_LANG_C11, file: !2, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !6)
!6 = !{!0, !9}
!7 = !{i32 7, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression())
!10 = !DIGlobalVariable(name: "lv", scope: !4, file: !2, line: 2, type: !3, isLocal: true, isDefinition: true)
