; RUN: opt -passes=correlated-propagation -S < %s | FileCheck %s
;
; CorrelatedValuePropagation can use facts that hold at an instruction to
; change the instruction's representation. Those facts do not make the new
; representation a valid source-level value. Salvage debug users of the
; original instruction with a DIExpression (DW_OP_shra / DW_OP_div /
; DW_OP_mod) instead of transferring them onto the rewritten value. If a
; salvage is not valid, kill the location rather than describing a wrong value.

declare void @llvm.dbg.value(metadata, metadata, metadata)

; The branch proves that %x is non-negative at the shift. This permits the
; ashr-to-lshr rewrite. Salvage the source variable as an arithmetic shift of
; the original operands rather than describing the logical shift.
define i16 @ashr_debug_value(i16 %x) !dbg !10 {
; CHECK-LABEL: define {{.*}} @ashr_debug_value(
; CHECK:       [[SHIFT:%.*]] = lshr i16 %x, 1
; CHECK-NEXT:    #dbg_value(i16 %x, [[SHIFTED:![0-9]+]], !DIExpression(DW_OP_constu, 1, DW_OP_shra, DW_OP_stack_value),
; CHECK-NEXT:    ret i16 [[SHIFT]]
entry:
  %nonnegative = icmp sge i16 %x, 0, !dbg !16
  br i1 %nonnegative, label %shift, label %exit, !dbg !16

shift:
  %shr = ashr i16 %x, 1, !dbg !17
  call void @llvm.dbg.value(metadata i16 %shr, metadata !13,
                             metadata !DIExpression()), !dbg !17
  ret i16 %shr, !dbg !18

exit:
  ret i16 24, !dbg !19
}

; A dbg_value that uses %x and is only positionally attached to the ashr must
; be preserved. dropDbgRecords() would delete it; findDbgUsers does not.
define i16 @ashr_preserves_unrelated_dbg_value(i16 %x) !dbg !40 {
; CHECK-LABEL: define {{.*}} @ashr_preserves_unrelated_dbg_value(
; CHECK:       #dbg_value(i16 %x, [[ARGVAR:![0-9]+]], !DIExpression(),
; CHECK:       [[SHIFT:%.*]] = lshr i16 %x, 1
; CHECK:       #dbg_value(i16 %x, [[SHIFTED:![0-9]+]], !DIExpression(DW_OP_constu, 1, DW_OP_shra, DW_OP_stack_value),
; CHECK:       ret i16 [[SHIFT]]
entry:
  %nonnegative = icmp sge i16 %x, 0, !dbg !44
  br i1 %nonnegative, label %shift, label %exit, !dbg !44

shift:
  call void @llvm.dbg.value(metadata i16 %x, metadata !41,
                             metadata !DIExpression()), !dbg !45
  %shr = ashr i16 %x, 1, !dbg !45
  call void @llvm.dbg.value(metadata i16 %shr, metadata !42,
                             metadata !DIExpression()), !dbg !45
  ret i16 %shr, !dbg !46

exit:
  ret i16 24, !dbg !47
}

; Variable shift amounts are salvaged with a DIArgList and DW_OP_shra.
define i16 @ashr_variable_debug_value(i16 %x, i16 %shamt) !dbg !50 {
; CHECK-LABEL: define {{.*}} @ashr_variable_debug_value(
; CHECK:       [[SHIFT:%.*]] = lshr i16 %x, %shamt
; CHECK-NEXT:    #dbg_value(!DIArgList(i16 %x, i16 %shamt), [[SHIFTED:![0-9]+]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_shra, DW_OP_stack_value),
; CHECK-NEXT:    ret i16 [[SHIFT]]
entry:
  %nonnegative = icmp sge i16 %x, 0, !dbg !54
  br i1 %nonnegative, label %shift, label %exit, !dbg !54

shift:
  %shr = ashr i16 %x, %shamt, !dbg !55
  call void @llvm.dbg.value(metadata i16 %shr, metadata !52,
                             metadata !DIExpression()), !dbg !55
  ret i16 %shr, !dbg !56

exit:
  ret i16 24, !dbg !57
}

; Compose salvage ops with an existing DIExpression on the ashr user.
define i16 @ashr_debug_value_existing_expr(i16 %x) !dbg !60 {
; CHECK-LABEL: define {{.*}} @ashr_debug_value_existing_expr(
; CHECK:       [[SHIFT:%.*]] = lshr i16 %x, 1
; CHECK-NEXT:    #dbg_value(i16 %x, [[SHIFTED:![0-9]+]], !DIExpression(DW_OP_constu, 1, DW_OP_shra, DW_OP_plus_uconst, 1, DW_OP_stack_value),
; CHECK-NEXT:    ret i16 [[SHIFT]]
entry:
  %nonnegative = icmp sge i16 %x, 0, !dbg !64
  br i1 %nonnegative, label %shift, label %exit, !dbg !64

shift:
  %shr = ashr i16 %x, 1, !dbg !65
  call void @llvm.dbg.value(metadata i16 %shr, metadata !62,
                             metadata !DIExpression(DW_OP_plus_uconst, 1)), !dbg !65
  ret i16 %shr, !dbg !66

exit:
  ret i16 24, !dbg !67
}

; The range facts on this block permit the division to be narrowed from i16
; to i8. Salvage the original i16 signed division rather than describing the
; sign-extended i8 quotient.
define i16 @sdiv_debug_value(i16 %x) !dbg !20 {
; CHECK-LABEL: define {{.*}} @sdiv_debug_value(
; CHECK:       [[LHS:%.*]] = trunc i16 %x to i8
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i8 [[LHS]], 42
; CHECK-NEXT:    [[SEXT:%.*]] = sext i8 [[DIV]] to i16
; CHECK-NEXT:    #dbg_value(i16 %x, [[QUOTIENT:![0-9]+]], !DIExpression(DW_OP_constu, 42, DW_OP_div, DW_OP_stack_value),
; CHECK-NEXT:    ret i16 [[SEXT]]
entry:
  %lower = icmp sgt i16 %x, -43, !dbg !24
  %upper = icmp slt i16 %x, 43, !dbg !24
  %in.range = and i1 %lower, %upper, !dbg !24
  br i1 %in.range, label %narrow, label %exit, !dbg !24

narrow:
  %div = sdiv i16 %x, 42, !dbg !25
  call void @llvm.dbg.value(metadata i16 %div, metadata !22,
                             metadata !DIExpression()), !dbg !25
  ret i16 %div, !dbg !26

exit:
  ret i16 24, !dbg !27
}

; Remainders have the same narrowing implementation as divisions. Test the
; separate opcode because signed remainders have different source semantics
; for negative operands.
define i16 @srem_debug_value(i16 %x) !dbg !30 {
; CHECK-LABEL: define {{.*}} @srem_debug_value(
; CHECK:       [[LHS:%.*]] = trunc i16 %x to i8
; CHECK-NEXT:    [[REM:%.*]] = srem i8 [[LHS]], 42
; CHECK-NEXT:    [[SEXT:%.*]] = sext i8 [[REM]] to i16
; CHECK-NEXT:    #dbg_value(i16 %x, [[REMAINDER:![0-9]+]], !DIExpression(DW_OP_constu, 42, DW_OP_mod, DW_OP_stack_value),
; CHECK-NEXT:    ret i16 [[SEXT]]
entry:
  %lower = icmp sgt i16 %x, -43, !dbg !34
  %upper = icmp slt i16 %x, 43, !dbg !34
  %in.range = and i1 %lower, %upper, !dbg !34
  br i1 %in.range, label %narrow, label %exit, !dbg !34

narrow:
  %rem = srem i16 %x, 42, !dbg !35
  call void @llvm.dbg.value(metadata i16 %rem, metadata !32,
                             metadata !DIExpression()), !dbg !35
  ret i16 %rem, !dbg !36

exit:
  ret i16 24, !dbg !37
}

; Constants wider than 64 bits cannot be encoded in a DIExpression, so salvage
; is not valid. Kill the location (poison) instead of describing the narrowed
; sign-extended value.
define i128 @sdiv_kill_wide_constant(i128 %x) !dbg !70 {
; CHECK-LABEL: define {{.*}} @sdiv_kill_wide_constant(
; CHECK:       [[LHS:%.*]] = trunc i128 %x to i8
; CHECK-NEXT:    [[DIV:%.*]] = sdiv i8 [[LHS]], 42
; CHECK-NEXT:    [[SEXT:%.*]] = sext i8 [[DIV]] to i128
; CHECK-NEXT:    #dbg_value(i128 poison, [[QUOTIENT:![0-9]+]], !DIExpression(),
; CHECK-NEXT:    ret i128 [[SEXT]]
entry:
  %lower = icmp sgt i128 %x, -43, !dbg !74
  %upper = icmp slt i128 %x, 43, !dbg !74
  %in.range = and i1 %lower, %upper, !dbg !74
  br i1 %in.range, label %narrow, label %exit, !dbg !74

narrow:
  %div = sdiv i128 %x, 42, !dbg !75
  call void @llvm.dbg.value(metadata i128 %div, metadata !72,
                             metadata !DIExpression()), !dbg !75
  ret i128 %div, !dbg !76

exit:
  ret i128 24, !dbg !77
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, splitDebugInlining: false)
!1 = !DIFile(filename: "debug-records.c", directory: "/")
!2 = !{}
!3 = !DISubroutineType(types: !2)
!4 = !DIBasicType(name: "i16", size: 16, encoding: DW_ATE_signed)
!5 = !DIBasicType(name: "i128", size: 128, encoding: DW_ATE_signed)
!6 = !{i32 2, !"Debug Info Version", i32 3}

!10 = distinct !DISubprogram(name: "ashr_debug_value", scope: !1, file: !1, line: 1, type: !3, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!13 = !DILocalVariable(name: "shifted", scope: !10, file: !1, line: 2, type: !4)
!16 = !DILocation(line: 3, column: 7, scope: !10)
!17 = !DILocation(line: 4, column: 17, scope: !10)
!18 = !DILocation(line: 5, column: 3, scope: !10)
!19 = !DILocation(line: 6, column: 3, scope: !10)

!20 = distinct !DISubprogram(name: "sdiv_debug_value", scope: !1, file: !1, line: 10, type: !3, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!22 = !DILocalVariable(name: "quotient", scope: !20, file: !1, line: 11, type: !4)
!24 = !DILocation(line: 12, column: 7, scope: !20)
!25 = !DILocation(line: 13, column: 18, scope: !20)
!26 = !DILocation(line: 14, column: 3, scope: !20)
!27 = !DILocation(line: 15, column: 3, scope: !20)

!30 = distinct !DISubprogram(name: "srem_debug_value", scope: !1, file: !1, line: 20, type: !3, scopeLine: 20, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!32 = !DILocalVariable(name: "remainder", scope: !30, file: !1, line: 21, type: !4)
!34 = !DILocation(line: 22, column: 7, scope: !30)
!35 = !DILocation(line: 23, column: 18, scope: !30)
!36 = !DILocation(line: 24, column: 3, scope: !30)
!37 = !DILocation(line: 25, column: 3, scope: !30)

!40 = distinct !DISubprogram(name: "ashr_preserves_unrelated_dbg_value", scope: !1, file: !1, line: 30, type: !3, scopeLine: 30, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!41 = !DILocalVariable(name: "arg", scope: !40, file: !1, line: 31, type: !4)
!42 = !DILocalVariable(name: "shifted", scope: !40, file: !1, line: 32, type: !4)
!44 = !DILocation(line: 33, column: 7, scope: !40)
!45 = !DILocation(line: 34, column: 17, scope: !40)
!46 = !DILocation(line: 35, column: 3, scope: !40)
!47 = !DILocation(line: 36, column: 3, scope: !40)

!50 = distinct !DISubprogram(name: "ashr_variable_debug_value", scope: !1, file: !1, line: 40, type: !3, scopeLine: 40, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!52 = !DILocalVariable(name: "shifted", scope: !50, file: !1, line: 41, type: !4)
!54 = !DILocation(line: 42, column: 7, scope: !50)
!55 = !DILocation(line: 43, column: 17, scope: !50)
!56 = !DILocation(line: 44, column: 3, scope: !50)
!57 = !DILocation(line: 45, column: 3, scope: !50)

!60 = distinct !DISubprogram(name: "ashr_debug_value_existing_expr", scope: !1, file: !1, line: 50, type: !3, scopeLine: 50, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!62 = !DILocalVariable(name: "shifted", scope: !60, file: !1, line: 51, type: !4)
!64 = !DILocation(line: 52, column: 7, scope: !60)
!65 = !DILocation(line: 53, column: 17, scope: !60)
!66 = !DILocation(line: 54, column: 3, scope: !60)
!67 = !DILocation(line: 55, column: 3, scope: !60)

!70 = distinct !DISubprogram(name: "sdiv_kill_wide_constant", scope: !1, file: !1, line: 60, type: !3, scopeLine: 60, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!72 = !DILocalVariable(name: "quotient", scope: !70, file: !1, line: 61, type: !5)
!74 = !DILocation(line: 62, column: 7, scope: !70)
!75 = !DILocation(line: 63, column: 18, scope: !70)
!76 = !DILocation(line: 64, column: 3, scope: !70)
!77 = !DILocation(line: 65, column: 3, scope: !70)
