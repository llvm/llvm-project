; RUN: opt -passes=dce -S %s | FileCheck %s --check-prefixes=CHECK,P64
; RUN: opt -passes=dce -data-layout=p:32:32 -S %s | FileCheck %s --check-prefixes=CHECK,P32

; Salvaging a dead ashr must sign-extend the shifted operand to the
; address-sized DWARF stack before DW_OP_shra.

define void @ashr_const_i32(i32 %x) !dbg !5 {
; CHECK-LABEL: @ashr_const_i32(
; P64: #dbg_value(i32 %x, ![[#]], !DIExpression(DW_OP_LLVM_convert, 32, DW_ATE_signed, DW_OP_LLVM_convert, 64, DW_ATE_signed, DW_OP_constu, 1, DW_OP_shra, DW_OP_stack_value)
; P32: #dbg_value(i32 %x, ![[#]], !DIExpression(DW_OP_constu, 1, DW_OP_shra, DW_OP_stack_value)
  %s = ashr i32 %x, 1
    #dbg_value(i32 %s, !9, !DIExpression(), !10)
  ret void
}

define void @ashr_var_i32(i32 %x, i32 %y) !dbg !11 {
; CHECK-LABEL: @ashr_var_i32(
; P64: #dbg_value(!DIArgList(i32 %x, i32 %y), ![[#]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_convert, 32, DW_ATE_signed, DW_OP_LLVM_convert, 64, DW_ATE_signed, DW_OP_LLVM_arg, 1, DW_OP_shra, DW_OP_stack_value)
; P32: #dbg_value(!DIArgList(i32 %x, i32 %y), ![[#]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_shra, DW_OP_stack_value)
  %s = ashr i32 %x, %y
    #dbg_value(i32 %s, !12, !DIExpression(), !13)
  ret void
}

; The shifted value already has a location operand, so the extension must apply
; to that operand only.
define void @ashr_const_i32_list(i32 %x, i32 %z) !dbg !14 {
; CHECK-LABEL: @ashr_const_i32_list(
; P64: #dbg_value(!DIArgList(i32 %z, i32 %x), ![[#]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_LLVM_convert, 32, DW_ATE_signed, DW_OP_LLVM_convert, 64, DW_ATE_signed, DW_OP_constu, 1, DW_OP_shra, DW_OP_plus, DW_OP_stack_value)
; P32: #dbg_value(!DIArgList(i32 %z, i32 %x), ![[#]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_constu, 1, DW_OP_shra, DW_OP_plus, DW_OP_stack_value)
  %s = ashr i32 %x, 1
    #dbg_value(!DIArgList(i32 %z, i32 %s), !15, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value), !16)
  ret void
}

define void @ashr_const_i64(i64 %x) !dbg !17 {
; CHECK-LABEL: @ashr_const_i64(
; P64: #dbg_value(i64 %x, ![[#]], !DIExpression(DW_OP_constu, 3, DW_OP_shra, DW_OP_stack_value)
; P32: #dbg_value(i64 poison, ![[#]], !DIExpression()
  %s = ashr i64 %x, 3
    #dbg_value(i64 %s, !18, !DIExpression(), !19)
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "x", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "a.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !{})
!5 = distinct !DISubprogram(name: "ashr_const_i32", scope: !1, file: !1, line: 1, type: !4, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!9 = !DILocalVariable(name: "s", scope: !5, file: !1, line: 2, type: !6)
!10 = !DILocation(line: 2, column: 3, scope: !5)
!11 = distinct !DISubprogram(name: "ashr_var_i32", scope: !1, file: !1, line: 1, type: !4, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!12 = !DILocalVariable(name: "s", scope: !11, file: !1, line: 2, type: !6)
!13 = !DILocation(line: 2, column: 3, scope: !11)
!14 = distinct !DISubprogram(name: "ashr_const_i32_list", scope: !1, file: !1, line: 1, type: !4, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!15 = !DILocalVariable(name: "s", scope: !14, file: !1, line: 2, type: !6)
!16 = !DILocation(line: 2, column: 3, scope: !14)
!17 = distinct !DISubprogram(name: "ashr_const_i64", scope: !1, file: !1, line: 1, type: !4, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!18 = !DILocalVariable(name: "s", scope: !17, file: !1, line: 2, type: !7)
!19 = !DILocation(line: 2, column: 3, scope: !17)
