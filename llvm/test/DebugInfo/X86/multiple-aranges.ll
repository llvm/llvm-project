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
source_filename = "test/DebugInfo/X86/multiple-aranges.ll"
target triple = "x86_64-unknown-linux-gnu"

@kittens = global i32 4, align 4, !dbg !0
@rainbows = global i32 5, align 4, !dbg !4

!llvm.dbg.cu = !{!7, !10}
!llvm.module.flags = !{!12, !13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "kittens", scope: null, file: !2, line: 1, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "test1.c", directory: "/home/kayamon")
!3 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression())
!5 = !DIGlobalVariable(name: "rainbows", scope: null, file: !6, line: 1, type: !3, isLocal: false, isDefinition: true)
!6 = !DIFile(filename: "test2.c", directory: "/home/kayamon")
!7 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang version 3.4 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !8, retainedTypes: !8, globals: !9, imports: !8)
!8 = !{}
!9 = !{!0}
!10 = distinct !DICompileUnit(language: DW_LANG_C99, file: !6, producer: "clang version 3.4 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !8, retainedTypes: !8, globals: !11, imports: !8)
!11 = !{!4}
!12 = !{i32 2, !"Dwarf Version", i32 4}
!13 = !{i32 1, !"Debug Info Version", i32 3}

