; RUN: opt -passes=correlated-propagation -S < %s | FileCheck %s
;
; CorrelatedValuePropagation can use facts that hold at an instruction to
; change the instruction's representation. Those facts do not make the new
; representation a valid source-level value for an attached debug record.
; Keep the optimized instruction, but make the affected source variable
; unavailable instead of transferring a misleading record.

declare void @llvm.dbg.value(metadata, metadata, metadata)

; The branch proves that %x is non-negative at the shift. This permits the
; ashr-to-lshr rewrite. A logical shift is nevertheless not the source value
; of `shifted` for all source-level observations.
define i16 @ashr_debug_value(i16 %x) !dbg !10 {
; CHECK-LABEL: define i16 @ashr_debug_value(
; CHECK: [[SHIFT:%.*]] = lshr i16 %x, 1
; CHECK-NOT: #dbg_value(i16 [[SHIFT]], ![[SHIFTED:[0-9]+]]
; CHECK: ret i16 [[SHIFT]]
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

; The range facts on this block permit the division to be narrowed from i16
; to i8. The replacement computes an i8 quotient and sign-extends it, so its
; debug record must not describe the original i16 division.
define i16 @sdiv_debug_value(i16 %x) !dbg !20 {
; CHECK-LABEL: define i16 @sdiv_debug_value(
; CHECK: [[LHS:%.*]] = trunc i16 %x to i8
; CHECK: [[DIV:%.*]] = sdiv i8 [[LHS]], 42
; CHECK: [[SEXT:%.*]] = sext i8 [[DIV]] to i16
; CHECK-NOT: #dbg_value(i16 [[SEXT]], ![[QUOTIENT:[0-9]+]]
; CHECK: ret i16 [[SEXT]]
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
; CHECK-LABEL: define i16 @srem_debug_value(
; CHECK: [[LHS:%.*]] = trunc i16 %x to i8
; CHECK: [[REM:%.*]] = srem i8 [[LHS]], 42
; CHECK: [[SEXT:%.*]] = sext i8 [[REM]] to i16
; CHECK-NOT: #dbg_value(i16 [[SEXT]], ![[REMAINDER:[0-9]+]]
; CHECK: ret i16 [[SEXT]]
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

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, splitDebugInlining: false)
!1 = !DIFile(filename: "debug-records.c", directory: "/")
!2 = !{}
!3 = !DISubroutineType(types: !2)
!4 = !DIBasicType(name: "i16", size: 16, encoding: DW_ATE_signed)
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
