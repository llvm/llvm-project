; RUN: llvm-dis -o - %s.bc | FileCheck %s

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!8, !9}

!0 = distinct !DIGlobalVariable(name: "g", scope: !1, file: !2, line: 1, type: !5, isLocal: false, isDefinition: true)
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang (llvm/trunk 304286)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !3, globals: !4)
!2 = !DIFile(filename: "a.c", directory: "/")
!3 = !{}
!4 = !{!7}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
; CHECK: !DIExpression(DW_OP_constu, 42, DW_OP_minus)
!6 = !DIExpression(DW_OP_minus, 42)
!7 = !DIGlobalVariableExpression(var: !0, expr: !6)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
