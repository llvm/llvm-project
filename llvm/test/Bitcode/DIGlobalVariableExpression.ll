; RUN: llvm-dis -o - %s.bc | FileCheck %s
; RUN: llvm-dis -o - %s.bc | llvm-as - | llvm-bcanalyzer -dump - | FileCheck %s --check-prefix=BC

; BC: GLOBAL_VAR_EXPR
; BC: GLOBAL_DECL_ATTACHMENT
; CHECK: @g = common global i32 0, align 4, !dbg ![[G:[0-9]+]]
; CHECK: @h = common global i32 0, align 4, !dbg ![[H:[0-9]+]]
; CHECK: ![[G]] = {{.*}}!DIGlobalVariableExpression(var: ![[GVAR:[0-9]+]], expr: ![[GEXPR:[0-9]+]])
; CHECK: ![[GVAR]] = distinct !DIGlobalVariable(name: "g",
; CHECK: !DIGlobalVariableExpression(var: ![[CVAR:[0-9]+]], expr: ![[CEXPR:[0-9]+]])
; CHECK: ![[CVAR]] = distinct !DIGlobalVariable(name: "c",
; CHECK: ![[CEXPR]] = !DIExpression(DW_OP_constu, 23, DW_OP_stack_value)
; CHECK: ![[H]] = {{.*}}!DIGlobalVariableExpression(var: ![[HVAR:[0-9]+]])
; CHECK: ![[HVAR]] = distinct !DIGlobalVariable(name: "h",
; CHECK: ![[GEXPR]] = !DIExpression(DW_OP_plus, 1)
@g = common global i32 0, align 4, !dbg !0
@h = common global i32 0, align 4, !dbg !11

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DIGlobalVariable(name: "g", scope: !1, file: !2, line: 1, type: !5, isLocal: false, isDefinition: true, expr: !DIExpression(DW_OP_plus, 1))
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang version 4.0.0 (trunk 286129) (llvm/trunk 286128)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !3, globals: !4)
!2 = !DIFile(filename: "a.c", directory: "/")
!3 = !{}
!4 = !{!0, !10, !11}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"PIC Level", i32 2}
!9 = !{!"clang version 4.0.0 (trunk 286129) (llvm/trunk 286128)"}
!10 = distinct !DIGlobalVariable(name: "c", scope: !1, file: !2, line: 1, type: !5, isLocal: false, isDefinition: true, expr: !DIExpression(DW_OP_constu, 23, DW_OP_stack_value))
!11 = distinct !DIGlobalVariable(name: "h", scope: !1, file: !2, line: 2, type: !5, isLocal: false, isDefinition: true)
