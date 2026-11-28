; RUN: opt %s -passes=dce -S | FileCheck %s

; Salvage a dead integer select as F + (C & 1) * (T - F).

target datalayout = "p:64:64"

; CHECK-LABEL: define void @const_true
; CHECK: #dbg_value(!DIArgList(i1 %c, i32 %f), ![[#]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_constu, 1, DW_OP_and, DW_OP_constu, 1, DW_OP_LLVM_arg, 1, DW_OP_minus, DW_OP_mul, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value),
define void @const_true(i1 %c, i32 %f) !dbg !5 {
  %sel = select i1 %c, i32 1, i32 %f, !dbg !10
    #dbg_value(i32 %sel, !8, !DIExpression(), !10)
  ret void, !dbg !10
}

; CHECK-LABEL: define void @const_false
; CHECK: #dbg_value(!DIArgList(i1 %c, i32 %t), ![[#]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_constu, 1, DW_OP_and, DW_OP_LLVM_arg, 1, DW_OP_constu, 4294967295, DW_OP_minus, DW_OP_mul, DW_OP_plus_uconst, 4294967295, DW_OP_stack_value),
define void @const_false(i1 %c, i32 %t) !dbg !11 {
  %sel = select i1 %c, i32 %t, i32 -1, !dbg !12
    #dbg_value(i32 %sel, !13, !DIExpression(), !12)
  ret void, !dbg !12
}

; With constant arms only the condition is referenced, so no DIArgList is needed.
; CHECK-LABEL: define void @const_arms
; CHECK: #dbg_value(i1 %c, ![[#]], !DIExpression(DW_OP_constu, 1, DW_OP_and, DW_OP_constu, 3, DW_OP_constu, 5, DW_OP_minus, DW_OP_mul, DW_OP_plus_uconst, 5, DW_OP_stack_value),
define void @const_arms(i1 %c) !dbg !23 {
  %sel = select i1 %c, i32 3, i32 5, !dbg !24
    #dbg_value(i32 %sel, !25, !DIExpression(), !24)
  ret void, !dbg !24
}

; The select is used twice; each use gets its own copy of the operands.
; CHECK-LABEL: define void @arglist
; CHECK: #dbg_value(!DIArgList(i1 %c, i64 %x, i1 %c, i64 %t, i64 %f, i64 %t, i64 %f), ![[#]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_constu, 1, DW_OP_and, DW_OP_LLVM_arg, 3, DW_OP_LLVM_arg, 4, DW_OP_minus, DW_OP_mul, DW_OP_LLVM_arg, 4, DW_OP_plus, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_LLVM_arg, 2, DW_OP_constu, 1, DW_OP_and, DW_OP_LLVM_arg, 5, DW_OP_LLVM_arg, 6, DW_OP_minus, DW_OP_mul, DW_OP_LLVM_arg, 6, DW_OP_plus, DW_OP_minus, DW_OP_stack_value),
define void @arglist(i1 %c, i64 %t, i64 %f, i64 %x) !dbg !14 {
  %sel = select i1 %c, i64 %t, i64 %f, !dbg !15
    #dbg_value(!DIArgList(i64 %sel, i64 %x, i64 %sel), !16, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_LLVM_arg, 2, DW_OP_minus, DW_OP_stack_value), !15)
  ret void, !dbg !15
}

; Wider than an address: the modular arithmetic would be wrong.
; CHECK-LABEL: define void @too_wide
; CHECK: #dbg_value(i128 poison,
define void @too_wide(i1 %c, i128 %t, i128 %f) !dbg !17 {
  %sel = select i1 %c, i128 %t, i128 %f, !dbg !18
    #dbg_value(i128 %sel, !19, !DIExpression(), !18)
  ret void, !dbg !18
}

; CHECK-LABEL: define void @vector
; CHECK: #dbg_value(<2 x i32> poison,
define void @vector(<2 x i1> %c, <2 x i32> %t, <2 x i32> %f) !dbg !20 {
  %sel = select <2 x i1> %c, <2 x i32> %t, <2 x i32> %f, !dbg !21
    #dbg_value(<2 x i32> %sel, !22, !DIExpression(), !21)
  ret void, !dbg !21
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !{})
!5 = distinct !DISubprogram(name: "const_true", scope: !1, file: !1, line: 1, type: !4, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!7 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!8 = !DILocalVariable(name: "a", scope: !5, file: !1, line: 1, type: !6)
!10 = !DILocation(line: 1, scope: !5)
!11 = distinct !DISubprogram(name: "const_false", scope: !1, file: !1, line: 2, type: !4, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!12 = !DILocation(line: 2, scope: !11)
!13 = !DILocalVariable(name: "b", scope: !11, file: !1, line: 2, type: !6)
!14 = distinct !DISubprogram(name: "arglist", scope: !1, file: !1, line: 3, type: !4, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!15 = !DILocation(line: 3, scope: !14)
!16 = !DILocalVariable(name: "c", scope: !14, file: !1, line: 3, type: !7)
!17 = distinct !DISubprogram(name: "too_wide", scope: !1, file: !1, line: 4, type: !4, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!18 = !DILocation(line: 4, scope: !17)
!19 = !DILocalVariable(name: "d", scope: !17, file: !1, line: 4, type: !7)
!20 = distinct !DISubprogram(name: "vector", scope: !1, file: !1, line: 5, type: !4, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!21 = !DILocation(line: 5, scope: !20)
!22 = !DILocalVariable(name: "e", scope: !20, file: !1, line: 5, type: !7)
!23 = distinct !DISubprogram(name: "const_arms", scope: !1, file: !1, line: 6, type: !4, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!24 = !DILocation(line: 6, scope: !23)
!25 = !DILocalVariable(name: "f", scope: !23, file: !1, line: 6, type: !6)
