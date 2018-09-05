; RUN: llc -o - %s | FileCheck --check-prefix=CHECK-DWARF2 %s
; RUN: llc -dwarf-version=4 -o - %s | FileCheck --check-prefix=CHECK-DWARF4 %s

; Exercise DW_OP_stack_value on global constants.

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-DWARF2: .byte   6                       # DW_AT_location
; CHECK-DWARF2-NEXT: .byte   52
; CHECK-DWARF2-NEXT: .byte   147
; CHECK-DWARF2-NEXT: .byte   2
; CHECK-DWARF2-NEXT: .byte   48
; CHECK-DWARF2-NEXT: .byte   147
; CHECK-DWARF2-NEXT: .byte   2

; CHECK-DWARF4: .byte   8                       # DW_AT_location
; CHECK-DWARF4-NEXT:.byte   52
; CHECK-DWARF4-NEXT:.byte   159
; CHECK-DWARF4-NEXT:.byte   147
; CHECK-DWARF4-NEXT:.byte   2
; CHECK-DWARF4-NEXT:.byte   48
; CHECK-DWARF4-NEXT:.byte   159
; CHECK-DWARF4-NEXT:.byte   147
; CHECK-DWARF4-NEXT:.byte   2

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang", file: !4, globals: !1, emissionKind: FullDebug)
!1 = !{!2, !10}
!2 = !DIGlobalVariableExpression(var: !8, expr: !3)
!3 = !DIExpression(DW_OP_constu, 4, DW_OP_stack_value, DW_OP_LLVM_fragment, 0, 16)
!4 = !DIFile(filename: "<stdin>", directory: "/")
!5 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !{i32 2, !"Dwarf Version", i32 2}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DIGlobalVariable(name: "a", scope: null, isLocal: false, isDefinition: true, type: !5)
!9 = !DIExpression(DW_OP_constu, 0, DW_OP_stack_value, DW_OP_LLVM_fragment, 16, 16)
!10 = !DIGlobalVariableExpression(var: !8, expr: !9)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7}
