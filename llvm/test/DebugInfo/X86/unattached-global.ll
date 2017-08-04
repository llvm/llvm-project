; RUN: %llc_dwarf -filetype=obj -o - %s | llvm-dwarfdump - | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: DW_TAG_variable
; CHECK: DW_AT_name {{.*}}"a"
; CHECK-NOT: DW_AT_location
; CHECK: DW_TAG

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang", file: !5, globals: !1, emissionKind: FullDebug)
!1 = !{!2}
!2 = !DIGlobalVariableExpression(var: !3, expr: !4)
!3 = distinct !DIGlobalVariable(name: "a", scope: null, isLocal: false, isDefinition: true, type: !6)
!4 = !DIExpression(DW_OP_plus_uconst, 4)
!5 = !DIFile(filename: "<stdin>", directory: "/")
!6 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)

!7 = !{i32 2, !"Dwarf Version", i32 2}
!8 = !{i32 2, !"Debug Info Version", i32 3}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
