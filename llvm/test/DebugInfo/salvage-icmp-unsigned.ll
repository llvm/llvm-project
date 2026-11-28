; RUN: opt %s -passes=dce -S | FileCheck %s --check-prefixes=CHECK,P64
; RUN: opt %s -passes=dce -S -data-layout="p:32:32" | FileCheck %s --check-prefixes=CHECK,P32

; DWARF relational operators compare the address-sized generic type as signed.
; Salvaged unsigned comparisons of that width flip the sign bit of both
; operands, and unsigned comparisons wider than an address are dropped.

; CHECK-LABEL: define void @ugt_i64_const(
; P64: #dbg_value(i64 %x, ![[#]], !DIExpression(DW_OP_constu, 9223372036854775808, DW_OP_xor, DW_OP_constu, 9223372036854775815, DW_OP_gt, DW_OP_stack_value),
; P32: #dbg_value(i1 poison, ![[#]], !DIExpression(),
define void @ugt_i64_const(i64 %x) !dbg !5 {
  %c = icmp ugt i64 %x, 7, !dbg !10
    #dbg_value(i1 %c, !9, !DIExpression(), !10)
  ret void, !dbg !10
}

; CHECK-LABEL: define void @ult_i64_var(
; P64: #dbg_value(!DIArgList(i64 %x, i64 %y), ![[#]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_constu, 9223372036854775808, DW_OP_xor, DW_OP_swap, DW_OP_constu, 9223372036854775808, DW_OP_xor, DW_OP_swap, DW_OP_lt, DW_OP_stack_value),
; P32: #dbg_value(i1 poison, ![[#]], !DIExpression(),
define void @ult_i64_var(i64 %x, i64 %y) !dbg !11 {
  %c = icmp ult i64 %x, %y, !dbg !12
    #dbg_value(i1 %c, !13, !DIExpression(), !12)
  ret void, !dbg !12
}

; CHECK-LABEL: define void @uge_i32_const(
; P64: #dbg_value(i32 %x, ![[#]], !DIExpression(DW_OP_constu, 4294967295, DW_OP_ge, DW_OP_stack_value),
; P32: #dbg_value(i32 %x, ![[#]], !DIExpression(DW_OP_constu, 2147483648, DW_OP_xor, DW_OP_constu, 2147483647, DW_OP_ge, DW_OP_stack_value),
define void @uge_i32_const(i32 %x) !dbg !14 {
  %c = icmp uge i32 %x, -1, !dbg !15
    #dbg_value(i1 %c, !16, !DIExpression(), !15)
  ret void, !dbg !15
}

; CHECK-LABEL: define void @ule_ptr_var(
; P64: #dbg_value(!DIArgList(ptr %p, ptr %q), ![[#]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_constu, 9223372036854775808, DW_OP_xor, DW_OP_swap, DW_OP_constu, 9223372036854775808, DW_OP_xor, DW_OP_swap, DW_OP_le, DW_OP_stack_value),
; P32: #dbg_value(!DIArgList(ptr %p, ptr %q), ![[#]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_constu, 2147483648, DW_OP_xor, DW_OP_swap, DW_OP_constu, 2147483648, DW_OP_xor, DW_OP_swap, DW_OP_le, DW_OP_stack_value),
define void @ule_ptr_var(ptr %p, ptr %q) !dbg !17 {
  %c = icmp ule ptr %p, %q, !dbg !18
    #dbg_value(i1 %c, !19, !DIExpression(), !18)
  ret void, !dbg !18
}

; Signed comparisons are unaffected.
; CHECK-LABEL: define void @sgt_i64_const(
; CHECK: #dbg_value(i64 %x, ![[#]], !DIExpression(DW_OP_consts, 18446744073709551615, DW_OP_gt, DW_OP_stack_value),
define void @sgt_i64_const(i64 %x) !dbg !20 {
  %c = icmp sgt i64 %x, -1, !dbg !21
    #dbg_value(i1 %c, !22, !DIExpression(), !21)
  ret void, !dbg !21
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !{})
!5 = distinct !DISubprogram(name: "ugt_i64_const", scope: !1, file: !1, line: 1, type: !4, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!9 = !DILocalVariable(name: "c", scope: !5, file: !1, line: 1, type: !6)
!10 = !DILocation(line: 1, scope: !5)
!11 = distinct !DISubprogram(name: "ult_i64_var", scope: !1, file: !1, line: 2, type: !4, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!12 = !DILocation(line: 2, scope: !11)
!13 = !DILocalVariable(name: "c", scope: !11, file: !1, line: 2, type: !6)
!14 = distinct !DISubprogram(name: "uge_i32_const", scope: !1, file: !1, line: 3, type: !4, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!15 = !DILocation(line: 3, scope: !14)
!16 = !DILocalVariable(name: "c", scope: !14, file: !1, line: 3, type: !6)
!17 = distinct !DISubprogram(name: "ule_ptr_var", scope: !1, file: !1, line: 4, type: !4, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!18 = !DILocation(line: 4, scope: !17)
!19 = !DILocalVariable(name: "c", scope: !17, file: !1, line: 4, type: !6)
!20 = distinct !DISubprogram(name: "sgt_i64_const", scope: !1, file: !1, line: 5, type: !4, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!21 = !DILocation(line: 5, scope: !20)
!22 = !DILocalVariable(name: "c", scope: !20, file: !1, line: 5, type: !6)
