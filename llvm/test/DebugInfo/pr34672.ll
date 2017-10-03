; RUN: opt -verify %s 2>&1 | FileCheck %s
; CHECK: invalid type ref
; CHECK-NOT: invalid type ref

@global = common global i32 0, align 4, !dbg !2

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "adrian", emissionKind: FullDebug)
!1 = !DIFile(filename: "broken.c", directory: "/")
!2 = !DIGlobalVariableExpression(var: !3, expr: !DIExpression())
!3 = !DIGlobalVariable(name: "g", scope: !0, file: !1, line: 1, type: !1, isLocal: false, isDefinition: true)
!5 = !{i32 2, !"Dwarf Version", i32 4}
!6 = !{i32 1, !"Debug Info Version", i32 3}

