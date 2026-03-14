; RUN: llc %s -mtriple=x86_64 -filetype=obj -o - 2>&1 | FileCheck %s --check-prefix=ERROR
; ERROR: invalid expression

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14, !15, !16}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "invalidconst.c", directory: "/")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression(DW_OP_consts, 18446744073709551615, DW_OP_stack_value, DW_OP_consts, 18446744073709551615, DW_OP_stack_value))
!5 = distinct !DIGlobalVariable(name: "constant", scope: !0, file: !1, line: 2, type: !6, isLocal: false, isDefinition: true)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"wchar_size", i32 4}
!16 = !{i32 7, !"PIC Level", i32 2}
