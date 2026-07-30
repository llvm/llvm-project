; RUN: llc %s -mtriple=x86_64 -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s

; CHECK-DAG-LABEL:  DW_AT_name ("negconstant")
; CHECK:  DW_AT_const_value (-1)
; CHECK-DAG-LABEL:  DW_AT_name ("negconstant2")
; CHECK:  DW_AT_const_value (-2)
; CHECK-DAG-LABEL:  DW_AT_name ("posconstant")
; CHECK:  DW_AT_const_value (1)
; CHECK-DAG-LABEL:  DW_AT_name ("posconstant1")
; CHECK:  DW_AT_const_value (2)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14, !15, !16}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "globalconst.c", directory: "/")
!2 = !{}
!3 = !{!4, !7, !9, !11}
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression(DW_OP_consts, 18446744073709551615, DW_OP_stack_value))
!5 = distinct !DIGlobalVariable(name: "negconstant", scope: !0, file: !1, line: 2, type: !6, isLocal: false, isDefinition: true)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression(DW_OP_consts, 18446744073709551614, DW_OP_stack_value))
!8 = distinct !DIGlobalVariable(name: "negconstant2", scope: !0, file: !1, line: 3, type: !6, isLocal: false, isDefinition: true)
!9 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression(DW_OP_consts, 1, DW_OP_stack_value))
!10 = distinct !DIGlobalVariable(name: "posconstant", scope: !0, file: !1, line: 2, type: !6, isLocal: false, isDefinition: true)
!11 = !DIGlobalVariableExpression(var: !12, expr: !DIExpression(DW_OP_consts, 2, DW_OP_stack_value))
!12 = distinct !DIGlobalVariable(name: "posconstant2", scope: !0, file: !1, line: 3, type: !6, isLocal: false, isDefinition: true)
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"wchar_size", i32 4}
!16 = !{i32 7, !"PIC Level", i32 2}
