; RUN: llc -o - %s | FileCheck --check-prefix=CHECK-DWARF2 %s
; RUN: llc -dwarf-version=4 -o - %s | FileCheck --check-prefix=CHECK-DWARF4 %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-DWARF2:      .byte	13                      # DW_AT_location
; CHECK-DWARF2-NEXT: .byte	3
; CHECK-DWARF2-NEXT: .quad	g
; CHECK-DWARF2-NEXT: .byte	16
; CHECK-DWARF2-NEXT: .byte	4
; CHECK-DWARF2-NEXT: .byte	16
; CHECK-DWARF2-NEXT: .byte	4

; CHECK-DWARF4:      .byte	14                      # DW_AT_location
; CHECK-DWARF4-NEXT: .byte	3
; CHECK-DWARF4-NEXT: .quad	g
; CHECK-DWARF4-NEXT: .byte	16
; CHECK-DWARF4-NEXT: .byte	4
; CHECK-DWARF4-NEXT: .byte	16
; CHECK-DWARF4-NEXT: .byte	4
; CHECK-DWARF4-NEXT: .byte	159

@g = global i32 0, !dbg !2

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang", file: !4, globals: !1, emissionKind: FullDebug)
!1 = !{!2}
!2 = distinct !DIGlobalVariable(name: "a", scope: null, isLocal: false, isDefinition: true, expr: !3, type: !5)
!3 = !DIExpression(DW_OP_constu, 4, DW_OP_constu, 4, DW_OP_stack_value)
!4 = !DIFile(filename: "<stdin>", directory: "/")
!5 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)

!6 = !{i32 2, !"Dwarf Version", i32 2}
!7 = !{i32 2, !"Debug Info Version", i32 3}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7}
