; RUN: opt -S -passes=tailcallelim < %s | FileCheck %s

; Check that TailCallElim's cleanupAndFinalize drops the debug location of `AccRecInstrNew`,
; because it is cloned from `AccRecInstr` (`%accumulate*` in the following IR) with the 
; debug location and inserted into the block of a different branch. 
; @test6_multiple_returns tests that when RetSelects.empty() is false (no select instruction is inserted)
; @test7_multiple_accumulators tests it when RetSelects.empty() is true

define i32 @test6_multiple_returns(i32 %x, i32 %y) local_unnamed_addr !dbg !5 {
; CHECK-LABEL: define i32 @test6_multiple_returns(
; CHECK:  case0:
; CHECK:    [[ACCUMULATOR_RET_TR2:%.*]] = add i32 %accumulator.tr, %helper{{$}}
; CHECK:  case99:
; CHECK:     [[ACCUMULATOR_RET_TR:%.*]] = add i32 %accumulator.tr, 18{{$}}
; CHECK:  default:
;
entry:
  switch i32 %x, label %default [
    i32 0, label %case0
    i32 99, label %case99
  ], !dbg !8

case0:                                            ; preds = %entry
  %helper = call i32 @test6_helper(), !dbg !9
  ret i32 %helper, !dbg !10

case99:                                           ; preds = %entry
  %sub1 = add i32 %x, -1, !dbg !11
  %recurse1 = call i32 @test6_multiple_returns(i32 %sub1, i32 %y), !dbg !12
  ret i32 18, !dbg !13

default:                                          ; preds = %entry
  %sub2 = add i32 %x, -1, !dbg !14
  %recurse2 = call i32 @test6_multiple_returns(i32 %sub2, i32 %y), !dbg !15
  %accumulate = add i32 %recurse2, %y, !dbg !16
  ret i32 %accumulate, !dbg !17
}

declare i32 @test6_helper()

define i32 @test7_multiple_accumulators(i32 %a) local_unnamed_addr !dbg !18 {
; CHECK-LABEL: define i32 @test7_multiple_accumulators(
; CHECK:  if.end3:
; CHECK:    [[ACCUMULATOR_RET_TR:%.*]] = add nsw i32 %accumulator.tr, [[ACCUMULATE2:.*]]{{$}}
; CHECK:  return:
; CHECK:    [[ACCUMULATOR_RET_TR1:%.*]] = add nsw i32 %accumulator.tr, 0{{$}}
;
entry:
  %tobool = icmp eq i32 %a, 0, !dbg !19
  br i1 %tobool, label %return, label %if.end, !dbg !20

if.end:                                           ; preds = %entry
  %and = and i32 %a, 1, !dbg !21
  %tobool1 = icmp eq i32 %and, 0, !dbg !22
  %sub = add nsw i32 %a, -1, !dbg !23
  br i1 %tobool1, label %if.end3, label %if.then2, !dbg !24

if.then2:                                         ; preds = %if.end
  %recurse1 = tail call i32 @test7_multiple_accumulators(i32 %sub), !dbg !25
  %accumulate1 = add nsw i32 %recurse1, 1, !dbg !26
  br label %return, !dbg !27

if.end3:                                          ; preds = %if.end
  %recurse2 = tail call i32 @test7_multiple_accumulators(i32 %sub), !dbg !28
  %accumulate2 = mul nsw i32 %recurse2, 2, !dbg !29
  br label %return, !dbg !30

return:                                           ; preds = %if.end3, %if.then2, %entry
  %retval.0 = phi i32 [ %accumulate1, %if.then2 ], [ %accumulate2, %if.end3 ], [ 0, %entry ], !dbg !31
  ret i32 %retval.0, !dbg !32
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "all.ll", directory: "/")
!2 = !{i32 24}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "test6_multiple_returns", linkageName: "test6_multiple_returns", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 1, column: 1, scope: !5)
!9 = !DILocation(line: 2, column: 1, scope: !5)
!10 = !DILocation(line: 3, column: 1, scope: !5)
!11 = !DILocation(line: 4, column: 1, scope: !5)
!12 = !DILocation(line: 5, column: 1, scope: !5)
!13 = !DILocation(line: 6, column: 1, scope: !5)
!14 = !DILocation(line: 7, column: 1, scope: !5)
!15 = !DILocation(line: 8, column: 1, scope: !5)
!16 = !DILocation(line: 9, column: 1, scope: !5)
!17 = !DILocation(line: 10, column: 1, scope: !5)
!18 = distinct !DISubprogram(name: "test7_multiple_accumulators", linkageName: "test7_multiple_accumulators", scope: null, file: !1, line: 11, type: !6, scopeLine: 11, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!19 = !DILocation(line: 11, column: 1, scope: !18)
!20 = !DILocation(line: 12, column: 1, scope: !18)
!21 = !DILocation(line: 13, column: 1, scope: !18)
!22 = !DILocation(line: 14, column: 1, scope: !18)
!23 = !DILocation(line: 15, column: 1, scope: !18)
!24 = !DILocation(line: 16, column: 1, scope: !18)
!25 = !DILocation(line: 17, column: 1, scope: !18)
!26 = !DILocation(line: 18, column: 1, scope: !18)
!27 = !DILocation(line: 19, column: 1, scope: !18)
!28 = !DILocation(line: 20, column: 1, scope: !18)
!29 = !DILocation(line: 21, column: 1, scope: !18)
!30 = !DILocation(line: 22, column: 1, scope: !18)
!31 = !DILocation(line: 23, column: 1, scope: !18)
!32 = !DILocation(line: 24, column: 1, scope: !18)
