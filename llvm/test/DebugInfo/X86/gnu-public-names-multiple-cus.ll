; RUN: llc < %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

!llvm.dbg.cu = !{!4, !11}
!llvm.module.flags = !{!7}

; CHECK: .section .debug_pubnames
; CHECK: .asciz "a"

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "a", scope: null, file: !2, line: 2, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "g.c", directory: "/tmp")
!3 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!4 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6)
!5 = !{}
!6 = !{!0}
!7 = !{i32 1, !"Debug Info Version", i32 3}

; CHECK: .section .debug_gnu_pubnames
; CHECK: .asciz "b"

!8 = !DIGlobalVariableExpression(var: !9, expr: !DIExpression())
!9 = !DIGlobalVariable(name: "b", scope: null, file: !2, line: 2, type: !3, isLocal: false, isDefinition: true)
!10 = !{!8}
!11 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !10, gnuPubnames: true)
