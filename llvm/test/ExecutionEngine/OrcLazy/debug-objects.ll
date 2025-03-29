; REQUIRES: native && x86_64-linux

; In-memory debug-objects contain DWARF
;
; RUN: lli --jit-linker=rtdyld  --orc-lazy-debug=jit-debug-objects %s | llvm-dwarfdump --diff - | FileCheck %s
; RUN: lli --jit-linker=jitlink --orc-lazy-debug=jit-debug-objects %s | llvm-dwarfdump --diff - | FileCheck %s
;
; CHECK: -:	file format elf64-x86-64
; TODO: Synthesized Mach-O objects error out with:
;       truncated or malformed object (offset field of section 8 in
;       LC_SEGMENT_64 command 0 extends past the end of the file)
;
; CHECK: .debug_info contents:
; CHECK: format = DWARF32
; CHECK: DW_TAG_compile_unit
; CHECK:               DW_AT_producer	("clang version 18.0.0git")
; CHECK:               DW_AT_language	(DW_LANG_C11)
; CHECK:               DW_AT_name	("source-file.c")
; CHECK:               DW_AT_comp_dir	("/workspace")
; CHECK:   DW_TAG_subprogram
; CHECK:                 DW_AT_frame_base	(DW_OP_reg7 RSP)
; CHECK:                 DW_AT_name	("main")
; CHECK:                 DW_AT_decl_file	("/workspace/source-file.c")
; CHECK:                 DW_AT_decl_line	(1)
; CHECK:                 DW_AT_type	("int")
; CHECK:                 DW_AT_external	(true)
; CHECK:   DW_TAG_base_type
; CHECK:                 DW_AT_name	("int")
; CHECK:                 DW_AT_encoding	(DW_ATE_signed)
; CHECK:                 DW_AT_byte_size	(0x04)
; CHECK:   NULL

; Text section of the in-memory debug-objects have non-null load-address
;
; RUN: lli --jit-linker=rtdyld --orc-lazy-debug=jit-debug-objects %s | \
; RUN:                              llvm-objdump --section-headers - | \
; RUN:                              FileCheck --check-prefix=CHECK_LOAD_ADDR %s
; RUN: lli --jit-linker=jitlink --orc-lazy-debug=jit-debug-objects %s | \
; RUN:                               llvm-objdump --section-headers - | \
; RUN:                               FileCheck --check-prefix=CHECK_LOAD_ADDR %s
;
; CHECK_LOAD_ADDR:      .text
; CHECK_LOAD_ADDR-NOT:  0000000000000000
; CHECK_LOAD_ADDR-SAME: TEXT

define i32 @main() !dbg !3 {
entry:
  ret i32 0, !dbg !8
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !DIFile(filename: "source-file.c", directory: "/workspace")
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 18.0.0git", emissionKind: FullDebug)
!3 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, unit: !2, retainedNodes: !7)
!4 = !DISubroutineType(types: !5)
!5 = !{!6}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{}
!8 = !DILocation(line: 1, column: 14, scope: !3)
