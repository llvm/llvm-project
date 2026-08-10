; RUN: opt -passes='loop(simple-loop-unswitch)' -S < %s | FileCheck %s
; RUN: opt -passes='loop-mssa(simple-loop-unswitch)' -S < %s | FileCheck %s

; Check that SimpleLoopUnswitch's unswitchTrivialBranch() and unswitchTrivialSwitch()
; propagates debug locations to the new terminators replacing the old ones.

define i32 @test1(ptr %var, i1 %cond1, i1 %cond2) !dbg !5 {
; CHECK-LABEL: define i32 @test1(
; CHECK:       loop_begin:
; CHECK-NEXT:    br label %[[CONTINUE:.*]], !dbg [[DBG8:![0-9]+]]
;
entry:
  br label %loop_begin, !dbg !8

loop_begin:                                       ; preds = %do_something, %entry
  br i1 %cond1, label %continue, label %loop_exit, !dbg !9

continue:                                         ; preds = %loop_begin
  %var_val = load i32, ptr %var, align 4, !dbg !10
  br i1 %cond2, label %do_something, label %loop_exit, !dbg !11

do_something:                                     ; preds = %continue
  call void @some_func(), !dbg !12
  br label %loop_begin, !dbg !13

loop_exit:                                        ; preds = %continue, %loop_begin
  ret i32 0, !dbg !14
}

define i32 @test7(i32 %cond1, i32 %x, i32 %y) !dbg !15 {
; CHECK-LABEL: define i32 @test7(
; CHECK-SAME: i32 [[COND1:%.*]], i32 [[X:%.*]], i32 [[Y:%.*]])
; CHECK-NEXT:  entry:
; CHECK-NEXT:    switch i32 [[COND1]], label %[[ENTRY_SPLIT:.*]] [
; CHECK-NEXT:      i32 0, label %[[LOOP_EXIT:.*]]
; CHECK-NEXT:      i32 1, label %[[LOOP_EXIT]]
; CHECK-NEXT:    ], !dbg [[DBG16:![0-9]+]]
; CHECK:       loop_begin:
; CHECK-NEXT:    br label %[[LATCH:.*]], !dbg [[DBG16]]
;
entry:
  br label %loop_begin, !dbg !16

loop_begin:                                       ; preds = %latch, %entry
  switch i32 %cond1, label %latch [
  i32 0, label %loop_exit
  i32 1, label %loop_exit
  ], !dbg !17

latch:                                            ; preds = %loop_begin
  call void @some_func(), !dbg !18
  br label %loop_begin, !dbg !19

loop_exit:                                        ; preds = %loop_begin, %loop_begin
  %result1 = phi i32 [ %x, %loop_begin ], [ %x, %loop_begin ], !dbg !20
  %result2 = phi i32 [ %y, %loop_begin ], [ %y, %loop_begin ], !dbg !21
  %result = add i32 %result1, %result2, !dbg !22
  ret i32 %result, !dbg !23
}

; Function Attrs: noreturn
declare void @some_func()

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

; CHECK: [[DBG8]] = !DILocation(line: 2,

; CHECK: [[DBG16]] = !DILocation(line: 9,

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test2.ll", directory: "/")
!2 = !{i32 15}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "test1", linkageName: "test1", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 1, column: 1, scope: !5)
!9 = !DILocation(line: 2, column: 1, scope: !5)
!10 = !DILocation(line: 3, column: 1, scope: !5)
!11 = !DILocation(line: 4, column: 1, scope: !5)
!12 = !DILocation(line: 5, column: 1, scope: !5)
!13 = !DILocation(line: 6, column: 1, scope: !5)
!14 = !DILocation(line: 7, column: 1, scope: !5)
!15 = distinct !DISubprogram(name: "test7", linkageName: "test7", scope: null, file: !1, line: 8, type: !6, scopeLine: 8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!16 = !DILocation(line: 8, column: 1, scope: !15)
!17 = !DILocation(line: 9, column: 1, scope: !15)
!18 = !DILocation(line: 10, column: 1, scope: !15)
!19 = !DILocation(line: 11, column: 1, scope: !15)
!20 = !DILocation(line: 12, column: 1, scope: !15)
!21 = !DILocation(line: 13, column: 1, scope: !15)
!22 = !DILocation(line: 14, column: 1, scope: !15)
!23 = !DILocation(line: 15, column: 1, scope: !15)
