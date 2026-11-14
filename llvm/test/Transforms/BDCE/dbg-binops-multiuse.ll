; RUN: opt -S -passes=bdce < %s | FileCheck %s

; Check that BDCE salvages debug uses when simplifying multi-use binary
; operators. If the operation cannot be represented in a DIExpression, the
; debug value must be killed instead of being replaced with the first operand.

define void @test(i64 %a, i128 %wide) !dbg !5 {
; CHECK-LABEL: define void @test(
; CHECK-SAME: i64 [[A:%.*]], i128 [[WIDE:%.*]])
; CHECK-NEXT:  entry:
; CHECK-NEXT:    #dbg_value(i64 [[A]], [[OR_VAR:![0-9]+]], !DIExpression(DW_OP_constu, 3, DW_OP_or, DW_OP_stack_value)
; CHECK-NEXT:    [[OR_USE:%.*]] = and i64 [[A]], 8
; CHECK-NEXT:    #dbg_value(i64 [[A]], [[XOR_VAR:![0-9]+]], !DIExpression(DW_OP_constu, 3, DW_OP_xor, DW_OP_stack_value)
; CHECK-NEXT:    [[XOR_USE:%.*]] = and i64 [[A]], 8
; CHECK-NEXT:    #dbg_value(i64 [[A]], [[AND_VAR:![0-9]+]], !DIExpression(DW_OP_constu, 24, DW_OP_and, DW_OP_stack_value)
; CHECK-NEXT:    [[AND_USE:%.*]] = and i64 [[A]], 8
; CHECK-NEXT:    #dbg_value(i128 poison, [[WIDE_VAR:![0-9]+]], !DIExpression()
; CHECK-NEXT:    [[WIDE_USE:%.*]] = and i128 [[WIDE]], 8
; CHECK-NEXT:    call void @use(i64 [[OR_USE]])
; CHECK-NEXT:    call void @use(i64 [[XOR_USE]])
; CHECK-NEXT:    call void @use(i64 [[AND_USE]])
; CHECK-NEXT:    call void @use_wide(i128 [[WIDE_USE]])
; CHECK-NEXT:    ret void
entry:
  %or = or i64 %a, 3
    #dbg_value(i64 %or, !9, !DIExpression(), !14)
  %or.use = and i64 %or, 8

  %xor = xor i64 %a, 3
    #dbg_value(i64 %xor, !10, !DIExpression(), !14)
  %xor.use = and i64 %xor, 8

  %and = and i64 %a, 24
    #dbg_value(i64 %and, !11, !DIExpression(), !14)
  %and.use = and i64 %and, 8

  %wide.or = or i128 %wide, 3
    #dbg_value(i128 %wide.or, !12, !DIExpression(), !14)
  %wide.use = and i128 %wide.or, 8

  call void @use(i64 %or.use)
  call void @use(i64 %xor.use)
  call void @use(i64 %and.use)
  call void @use_wide(i128 %wide.use)
  ret void
}

declare void @use(i64)
declare void @use_wide(i128)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 2, !"Dwarf Version", i32 5}
!5 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !13, !15}
!8 = !{!9, !10, !11, !12}
!9 = !DILocalVariable(name: "or", scope: !5, file: !1, line: 2, type: !13)
!10 = !DILocalVariable(name: "xor", scope: !5, file: !1, line: 3, type: !13)
!11 = !DILocalVariable(name: "and", scope: !5, file: !1, line: 4, type: !13)
!12 = !DILocalVariable(name: "wide", scope: !5, file: !1, line: 5, type: !15)
!13 = !DIBasicType(name: "uint64_t", size: 64, encoding: DW_ATE_unsigned)
!14 = !DILocation(line: 2, column: 1, scope: !5)
!15 = !DIBasicType(name: "uint128_t", size: 128, encoding: DW_ATE_unsigned)
