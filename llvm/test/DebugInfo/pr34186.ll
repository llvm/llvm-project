; Make sure we reject GVs without a type and we verify each exactly once.
; RUN: not llc %s 2>&1 | FileCheck %s
; CHECK: missing global variable type
; CHECK-NOT: missing global variable type

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!63, !64}
!1 = distinct !DIGlobalVariable(name: "pat", scope: !2, file: !3, line: 27, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "", emissionKind: FullDebug, globals: !5)
!3 = !DIFile(filename: "patatino.c", directory: "/")
!5 = !{!6}
!6 = !DIGlobalVariableExpression(var: !1)
!63 = !{i32 2, !"Dwarf Version", i32 4}
!64 = !{i32 2, !"Debug Info Version", i32 3}
