; RUN: llvm-dis -o - %s.bc | FileCheck %s

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}

; CHECK: !0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.1", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.1", isOptimized: true, runtimeVersion: 0, emissionKind: 1, enums: !2, globals: !3)
!1 = !DIFile(filename: "g.c", directory: "/")
!2 = !{}
; CHECK: !3 = !{!4}
!3 = !{!4}
; CHECK: !4 = {{.*}}!DIGlobalVariableExpression(var: !5, expr: !8)
; CHECK: !5 = !DIGlobalVariable(name: "c", scope: !0, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true)
; CHECK: !8 = !DIExpression(DW_OP_constu, 42, DW_OP_stack_value)
!4 = !DIGlobalVariable(name: "c", scope: !0, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, variable: i32 42)
!5 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !6)
!6 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = !{i32 2, !"Dwarf Version", i32 2}
!8 = !{i32 2, !"Debug Info Version", i32 3}

