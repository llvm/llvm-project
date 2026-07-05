; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

@foo = global i32 0

; CHECK: !named = !{!0, !1, !2, !3, !4, !5, !6, !DIExpression(DW_OP_constu, 42, DW_OP_stack_value)}
!named = !{!0, !1, !2, !3, !4, !5, !6, !7}

!0 = !DIFile(filename: "scope.h", directory: "/path/to/dir")
!1 = distinct !{}
!2 = !DIFile(filename: "path/to/file", directory: "/path/to/dir")
!3 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!4 = distinct !{}

; CHECK: !5 = !DIGlobalVariable(name: "foo", linkageName: "foo", scope: !0, file: !2, line: 7, type: !3, isLocal: true, isDefinition: false, align: 32)
!5 = !DIGlobalVariable(name: "foo", linkageName: "foo", scope: !0,
                       file: !2, line: 7, type: !3, isLocal: true,
                       isDefinition: false, align: 32)

; CHECK: !6 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression(DW_OP_constu, 42, DW_OP_stack_value))
!6 = !DIGlobalVariableExpression(var: !5, expr: !7)
!7 = !DIExpression(DW_OP_constu, 42, DW_OP_stack_value)
