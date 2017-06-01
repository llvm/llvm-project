; RUN: llvm-dis -o - %s.bc | FileCheck %s

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!20, !21}

!0 = distinct !DIGlobalVariable(name: "g", scope: !1, file: !2, line: 1, type: !5, isLocal: false, isDefinition: true)
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang (llvm/trunk 288154)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !3, globals: !4)
!2 = !DIFile(filename: "a.c", directory: "/")
!3 = !{}
!4 = !{!10, !11, !12, !13}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
; DW_OP_deref should be moved to the back of the expression.
;
; CHECK: !DIExpression(DW_OP_plus, 0, DW_OP_deref, DW_OP_LLVM_fragment, 8, 32)
!6 = !DIExpression(DW_OP_deref, DW_OP_plus, 0, DW_OP_LLVM_fragment, 8, 32)
; CHECK: !DIExpression(DW_OP_plus, 0, DW_OP_deref)
!7 = !DIExpression(DW_OP_deref, DW_OP_plus, 0)
; CHECK: !DIExpression(DW_OP_plus, 1, DW_OP_deref)
!8 = !DIExpression(DW_OP_plus, 1, DW_OP_deref)
; CHECK: !DIExpression(DW_OP_deref)
!9 = !DIExpression(DW_OP_deref)
!10 = !DIGlobalVariableExpression(var: !0, expr: !6)
!11 = !DIGlobalVariableExpression(var: !0, expr: !7)
!12 = !DIGlobalVariableExpression(var: !0, expr: !8)
!13 = !DIGlobalVariableExpression(var: !0, expr: !9)
!20 = !{i32 2, !"Dwarf Version", i32 4}
!21 = !{i32 2, !"Debug Info Version", i32 3}
