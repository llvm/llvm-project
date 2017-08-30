; RUN: not opt -S <%s 2>&1| FileCheck %s

; CHECK: fragment is larger than or outside of variable
; CHECK: !DIGlobalVariableExpression(var: ![[VAR:[0-9]+]],
; CHECK-SAME:                        expr: !DIExpression(DW_OP_LLVM_fragment, 0, 64))
; CHECK: ![[VAR]] = !DIGlobalVariable(name: "g"

@g = common global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!6, !7}

!0 = !DIGlobalVariableExpression(var: !3, expr: !DIExpression(DW_OP_LLVM_fragment, 0, 64))
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, emissionKind: FullDebug)
!2 = !DIFile(filename: "a.c", directory: "/")
!3 = !DIGlobalVariable(name: "g", scope: !1, file: !2, line: 1, type: !5, isLocal: false, isDefinition: true)
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
