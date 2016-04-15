; RUN: llc -generate-arange-section < %s | FileCheck %s

; CHECK: .section .debug_aranges,"",@progbits

; First CU
; CHECK-NEXT: .long   44                      # Length of ARange Set
; CHECK-NEXT: .short  2                       # DWARF Arange version number
; CHECK-NEXT: .long   .Lcu_begin0             # Offset Into Debug Info Section
; CHECK-NEXT: .byte   8                       # Address Size (in bytes)
; CHECK-NEXT: .byte   0                       # Segment Size (in bytes)
; CHECK-NEXT: .zero   4,255
; CHECK-NEXT: .quad   kittens
; CHECK-NEXT: .quad   rainbows-kittens
; CHECK-NEXT: .quad   0                       # ARange terminator
; CHECK-NEXT: .quad   0

; Second CU
; CHECK-NEXT: .long   44                      # Length of ARange Set
; CHECK-NEXT: .short  2                       # DWARF Arange version number
; CHECK-NEXT: .long   .Lcu_begin1             # Offset Into Debug Info Section
; CHECK-NEXT: .byte   8                       # Address Size (in bytes)
; CHECK-NEXT: .byte   0                       # Segment Size (in bytes)
; CHECK-NEXT: .zero   4,255
; CHECK-NEXT: .quad   rainbows
; CHECK-NEXT: .quad   .Lsec_end0-rainbows
; CHECK-NEXT: .quad   0                       # ARange terminator
; CHECK-NEXT: .quad   0


; Generated from: clang -c -g -emit-llvm
;                 llvm-link test1.bc test2.bc -o test.bc
; test1.c: int kittens = 4;
; test2.c: int rainbows = 5;




; ModuleID = 'test.bc'
target triple = "x86_64-unknown-linux-gnu"

@kittens = global i32 4, align 4
@rainbows = global i32 5, align 4

!llvm.dbg.cu = !{!0, !7}
!llvm.module.flags = !{!12, !13}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.4 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !3, imports: !2)
!1 = !DIFile(filename: "test1.c", directory: "/home/kayamon")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariable(name: "kittens", line: 1, isLocal: false, isDefinition: true, scope: null, file: !5, type: !6, variable: i32* @kittens)
!5 = !DIFile(filename: "test1.c", directory: "/home/kayamon")
!6 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.4 ", isOptimized: false, emissionKind: FullDebug, file: !8, enums: !2, retainedTypes: !2, globals: !9, imports: !2)
!8 = !DIFile(filename: "test2.c", directory: "/home/kayamon")
!9 = !{!10}
!10 = !DIGlobalVariable(name: "rainbows", line: 1, isLocal: false, isDefinition: true, scope: null, file: !11, type: !6, variable: i32* @rainbows)
!11 = !DIFile(filename: "test2.c", directory: "/home/kayamon")
!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 1, !"Debug Info Version", i32 3}
