; Make sure we reject GVs without a type.
; Currently the verifier when traversing the graph induced by the debug info
; metadata can reach the GV both from a DICompileUnit and a DIGlobalVariable
; expression, so we emit a diagnostic twice. This is, not ideal, but the
; alternative is that of keeping a map of visited GVs, which has non trivial
; memory usage consequences on large testcases, or when LTO is the mode of
; operation.
; RUN: llvm-as -disable-output %s -o - 2>&1 | FileCheck %s
; CHECK: missing global variable type
; CHECK: missing global variable type
; CHECK-NOT: missing global variable type
; CHECK: warning: ignoring invalid debug info

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!63, !64}
!1 = distinct !DIGlobalVariable(name: "pat", scope: !2, file: !3, line: 27, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "", emissionKind: FullDebug, globals: !5)
!3 = !DIFile(filename: "patatino.c", directory: "/")
!5 = !{!6}
!6 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!63 = !{i32 2, !"Dwarf Version", i32 4}
!64 = !{i32 2, !"Debug Info Version", i32 3}
