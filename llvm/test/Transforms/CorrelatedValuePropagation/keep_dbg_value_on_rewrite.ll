; RUN: opt < %s -passes=correlated-propagation -S | FileCheck %s

; When replacing an instruction (ie sdiv->udiv), check that the debug
; info keeps the original value of the variable, rather than the value
; of the new calculation
; See issues #218621, #218622, #218623 and 218366

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @srem_to_urem(i32 noundef %x) #0 !dbg !9 {
; CHECK:      [[X_NONNEG:%.*]] = sub i32 0, [[X:%.*]]
; CHECK-NEXT: [[REM1:%.*]] = urem i32 [[X_NONNEG]], 42
; CHECK-NEXT: [[REM1_NEG:%.*]] = sub i32 0, [[REM1]]
; CHECK-NEXT: #dbg_value(i32 [[X]], [[META17:![0-9]+]], !DIExpression(DW_OP_constu, 42, DW_OP_mod, DW_OP_stack_value), [[META15:![0-9]+]])
;
entry:
  %rem = srem i32 %x, 42
    #dbg_value(i32 %rem, !17, !DIExpression(), !15)
  %cmp = icmp slt i32 %x, 0
  %rem. = select i1 %cmp, i32 %rem, i32 24
  ret i32 %rem.
}

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @sdiv_to_udiv(i32 noundef %x) #0 !dbg !21 {
; CHECK:      [[X_NONNEG:%.*]] = sub i32 0, [[X:%.*]]
; CHECK-NEXT: [[DIV1:%.*]] = udiv i32 [[X_NONNEG]], 42
; CHECK-NEXT: [[DIV1_NEG:%.*]] = sub i32 0, [[DIV1]]
; CHECK-NEXT: #dbg_value(i32 [[X]], [[META25:![0-9]+]], !DIExpression(DW_OP_constu, 42, DW_OP_div, DW_OP_stack_value), [[META23:![0-9]+]])
;
entry:
  %div = sdiv i32 %x, 42
    #dbg_value(i32 %div, !25, !DIExpression(), !23)
  %cmp = icmp slt i32 %x, 0
  %div. = select i1 %cmp, i32 %div, i32 24
  ret i32 %div.
}

; Function Attrs: noinline nounwind uwtable
define dso_local i64 @sext_to_zext(i32 noundef %x) #0 !dbg !29 {
; CHECK:      [[CONV:%.*]] = zext nneg i32 [[X:%.*]] to i64
; CHECK-NEXT: #dbg_value(i32 [[X]], [[META36:![0-9]+]], !DIExpression(DW_OP_LLVM_convert, 32, DW_ATE_signed, DW_OP_LLVM_convert, 64, DW_ATE_signed, DW_OP_stack_value), [[META34:![0-9]+]])
;
entry:
  %conv = sext i32 %x to i64
    #dbg_value(i64 %conv, !36, !DIExpression(), !34)
  %cmp = icmp sge i32 %x, 0
  %conv. = select i1 %cmp, i64 %conv, i64 24
  ret i64 %conv.
}

; Function Attrs: noinline nounwind uwtable
define dso_local i32 @ashr_to_lshr(i32 noundef %x, i32 noundef %y) #0 !dbg !40 {
; CHECK:      [[SHR:%.*]] = lshr i32 [[X:%.*]], [[Y:%.*]]
; CHECK-NEXT: #dbg_value(!DIArgList(i32 [[X]], i32 [[Y]]), [[META47:![0-9]+]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_shra, DW_OP_stack_value), [[META44:![0-9]+]])
;
entry:
  %shr = ashr i32 %x, %y
    #dbg_value(i32 %shr, !47, !DIExpression(), !44)
  %cmp = icmp sge i32 %x, 0
  %shr. = select i1 %cmp, i32 %shr, i32 24
  ret i32 %shr.
}

attributes #0 = { noinline nounwind uwtable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 24.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "case.c", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!9 = distinct !DISubprogram(name: "srem_to_urem", scope: !1, file: !1, line: 3, type: !10, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !{}
!15 = !DILocation(line: 0, scope: !9)
!17 = !DILocalVariable(name: "remainder", scope: !9, file: !1, line: 4, type: !12)
!21 = distinct !DISubprogram(name: "sdiv_to_udiv", scope: !1, file: !1, line: 8, type: !10, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!23 = !DILocation(line: 0, scope: !21)
!25 = !DILocalVariable(name: "quotient", scope: !21, file: !1, line: 9, type: !12)
!29 = distinct !DISubprogram(name: "sext_to_zext", scope: !1, file: !1, line: 13, type: !30, scopeLine: 13, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!30 = !DISubroutineType(types: !31)
!31 = !{!32, !12}
!32 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!34 = !DILocation(line: 0, scope: !29)
!36 = !DILocalVariable(name: "extended", scope: !29, file: !1, line: 14, type: !32)
!40 = distinct !DISubprogram(name: "ashr_to_lshr", scope: !1, file: !1, line: 18, type: !41, scopeLine: 18, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !13)
!41 = !DISubroutineType(types: !42)
!42 = !{!12, !12, !12}
!44 = !DILocation(line: 0, scope: !40)
!47 = !DILocalVariable(name: "shifted", scope: !40, file: !1, line: 19, type: !12)
