; RUN: llc -o - %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: .byte    0                       # DW_AT_location

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang", file: !4, globals: !1, emissionKind: FullDebug)
!1 = !{!2}
!2 = distinct !DIGlobalVariable(name: "a", scope: null, isLocal: false, isDefinition: true, expr: !3, type: !5)
!3 = !DIExpression(DW_OP_plus, 4)
!4 = !DIFile(filename: "<stdin>", directory: "/")
!5 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)

!6 = !{i32 2, !"Dwarf Version", i32 2}
!7 = !{i32 2, !"Debug Info Version", i32 3}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7}
