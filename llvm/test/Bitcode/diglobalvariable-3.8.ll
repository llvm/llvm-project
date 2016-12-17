; RUN: llvm-dis -o - %s.bc | FileCheck %s

; CHECK: !0 = distinct !DIGlobalVariable(name: "a", scope: null, isLocal: false, isDefinition: true, expr: !1)
; CHECK: !1 = !DIExpression(DW_OP_constu, 42, DW_OP_stack_value)

!named = !{!0}

!0 = distinct !DIGlobalVariable(name: "a", variable: i32 42)
