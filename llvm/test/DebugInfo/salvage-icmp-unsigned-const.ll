; RUN: opt %s -passes=dce -S | FileCheck %s

; Salvaging an icmp with a constant operand must not sign-extend the constant
; for equality and unsigned predicates, since the DWARF comparison operates on
; the zero-extended value.

; CHECK-LABEL: define void @eq_i32_minus_one(
; CHECK: #dbg_value(i32 %a, ![[#]], !DIExpression(DW_OP_constu, 4294967295, DW_OP_eq, DW_OP_stack_value),
define void @eq_i32_minus_one(i32 %a) !dbg !5 {
  %c = icmp eq i32 %a, -1, !dbg !10
    #dbg_value(i1 %c, !9, !DIExpression(), !10)
  ret void, !dbg !10
}

; CHECK-LABEL: define void @ne_i32_minus_one(
; CHECK: #dbg_value(i32 %a, ![[#]], !DIExpression(DW_OP_constu, 4294967295, DW_OP_ne, DW_OP_stack_value),
define void @ne_i32_minus_one(i32 %a) !dbg !11 {
  %c = icmp ne i32 %a, -1, !dbg !12
    #dbg_value(i1 %c, !13, !DIExpression(), !12)
  ret void, !dbg !12
}

; CHECK-LABEL: define void @ult_i8_negative(
; CHECK: #dbg_value(i8 %a, ![[#]], !DIExpression(DW_OP_constu, 200, DW_OP_lt, DW_OP_stack_value),
define void @ult_i8_negative(i8 %a) !dbg !14 {
  %c = icmp ult i8 %a, 200, !dbg !15
    #dbg_value(i1 %c, !16, !DIExpression(), !15)
  ret void, !dbg !15
}

; CHECK-LABEL: define void @eq_i1_true(
; CHECK: #dbg_value(i1 %a, ![[#]], !DIExpression(DW_OP_constu, 1, DW_OP_eq, DW_OP_stack_value),
define void @eq_i1_true(i1 %a) !dbg !17 {
  %c = icmp eq i1 %a, true, !dbg !18
    #dbg_value(i1 %c, !19, !DIExpression(), !18)
  ret void, !dbg !18
}

; CHECK-LABEL: define void @eq_i64_minus_one(
; CHECK: #dbg_value(i64 %a, ![[#]], !DIExpression(DW_OP_constu, 18446744073709551615, DW_OP_eq, DW_OP_stack_value),
define void @eq_i64_minus_one(i64 %a) !dbg !20 {
  %c = icmp eq i64 %a, -1, !dbg !21
    #dbg_value(i1 %c, !22, !DIExpression(), !21)
  ret void, !dbg !21
}

; Signed predicates keep the sign-extended constant.
; CHECK-LABEL: define void @sgt_i32_minus_one(
; CHECK: #dbg_value(i32 %a, ![[#]], !DIExpression(DW_OP_consts, 18446744073709551615, DW_OP_gt, DW_OP_stack_value),
define void @sgt_i32_minus_one(i32 %a) !dbg !23 {
  %c = icmp sgt i32 %a, -1, !dbg !24
    #dbg_value(i1 %c, !25, !DIExpression(), !24)
  ret void, !dbg !24
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DISubroutineType(types: !4)
!4 = !{null}
!5 = distinct !DISubprogram(name: "eq_i32_minus_one", scope: !1, file: !1, line: 1, type: !3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!8 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!9 = !DILocalVariable(name: "c", scope: !5, file: !1, line: 1, type: !8)
!10 = !DILocation(line: 1, scope: !5)
!11 = distinct !DISubprogram(name: "ne_i32_minus_one", scope: !1, file: !1, line: 2, type: !3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!12 = !DILocation(line: 2, scope: !11)
!13 = !DILocalVariable(name: "c", scope: !11, file: !1, line: 2, type: !8)
!14 = distinct !DISubprogram(name: "ult_i8_negative", scope: !1, file: !1, line: 3, type: !3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!15 = !DILocation(line: 3, scope: !14)
!16 = !DILocalVariable(name: "c", scope: !14, file: !1, line: 3, type: !8)
!17 = distinct !DISubprogram(name: "eq_i1_true", scope: !1, file: !1, line: 4, type: !3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!18 = !DILocation(line: 4, scope: !17)
!19 = !DILocalVariable(name: "c", scope: !17, file: !1, line: 4, type: !8)
!20 = distinct !DISubprogram(name: "eq_i64_minus_one", scope: !1, file: !1, line: 5, type: !3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!21 = !DILocation(line: 5, scope: !20)
!22 = !DILocalVariable(name: "c", scope: !20, file: !1, line: 5, type: !8)
!23 = distinct !DISubprogram(name: "sgt_i32_minus_one", scope: !1, file: !1, line: 6, type: !3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!24 = !DILocation(line: 6, scope: !23)
!25 = !DILocalVariable(name: "c", scope: !23, file: !1, line: 6, type: !8)
