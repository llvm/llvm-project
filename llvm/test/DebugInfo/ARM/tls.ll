; RUN: llc -O0 -filetype=asm -mtriple=armv7-linux-gnuehabi < %s \
; RUN:     | FileCheck %s --check-prefix=CHECK
; RUN: llc -O0 -filetype=asm -mtriple=armv7-linux-gnuehabi -emulated-tls < %s \
; RUN:     | FileCheck %s --check-prefix=EMU

; Generated with clang with source
; __thread int x;

@x = thread_local global i32 0, align 4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

; 6 byte of data
; CHECK: .byte 6 @ DW_AT_location
; DW_OP_const4u
; CHECK: .byte 12
; The debug relocation of the address of the tls variable
; CHECK: .long x(tlsldo)

; TODO: Add expected output for -emulated-tls tests.
; EMU-NOT: .long x(tlsldo)

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5 ", isOptimized: false, emissionKind: 0, file: !1, enums: !2, retainedTypes: !2, subprograms: !2, globals: !3, imports: !2)
!1 = !DIFile(filename: "tls.c", directory: "/tmp")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariable(name: "x", line: 1, isLocal: false, isDefinition: true, scope: null, file: !5, type: !6, variable: i32* @x)
!5 = !DIFile(filename: "tls.c", directory: "/tmp")
!6 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 1, !"Debug Info Version", i32 3}
!9 = !{!"clang version 3.5 "}
