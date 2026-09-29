; RUN: opt -passes='simple-loop-unswitch<nontrivial>' -S < %s | FileCheck %s

define i32 @basic(i32 %N, i1 %cond, i32 %select_input) !dbg !5 {
; CHECK-LABEL: define i32 @basic(

; Check that SimpleLoopUnswitch's unswitchNontrivialInvariants() drops the
; debug location of the hoisted terminator and doesn't give any debug location
; to the new freeze, since it's inserted in a hoist block.
; Also check that in unswitchNontrivialInvariants(), the new br instruction
; inherits the debug location of the old terminator in the same block.

; CHECK:       entry:
; CHECK-NEXT:    [[COND_FR:%.*]] = freeze i1 [[COND:%.*]]{{$}}
; CHECK-NEXT:    br i1 [[COND_FR]], label %[[ENTRY_SPLIT_US:.*]], label %[[ENTRY_SPLIT:.*]]{{$}}
; CHECK:       for.body.us:
; CHECK-NEXT:    br label %0, !dbg [[DBG13:![0-9]+]]

; Check that in turnSelectIntoBranch(), the new phi inherits the debug
; location of the old select instruction replaced.

; CHECK:       1:
; CHECK-NEXT:    [[UNSWITCHED_SELECT_US:%.*]] = phi i32 [ [[SELECT_INPUT:%.*]], %0 ], !dbg [[DBG13]]

; Check that in BuildClonedLoopBlocks(), the new phi inherits the debug
; location of the instruction at the insertion point and the new br
; instruction inherits the debug location of the old terminator.

; CHECK:       for.body:
; CHECK-NEXT:    br label %2, !dbg [[DBG13]]
; CHECK:       for.cond.cleanup:
; CHECK:         [[DOTUS_PHI:%.*]] = phi i32 [ [[RES_LCSSA:%.*]], %[[FOR_COND_CLEANUP_SPLIT:.*]] ], [ [[RES_LCSSA_US:%.*]], %[[FOR_COND_CLEANUP_SPLIT_US:.*]] ], !dbg [[DBG17:![0-9]+]]
entry:
  br label %for.cond, !dbg !8

for.cond:                                         ; preds = %for.body, %entry
  %res = phi i32 [ 0, %entry ], [ %add, %for.body ], !dbg !9
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ], !dbg !10
  %cmp = icmp slt i32 %i, %N, !dbg !11
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !dbg !12

for.body:                                         ; preds = %for.cond
  %cond1 = select i1 %cond, i32 %select_input, i32 42, !dbg !13
  %add = add nuw nsw i32 %cond1, %res, !dbg !14
  %inc = add nuw nsw i32 %i, 1, !dbg !15
  br label %for.cond, !dbg !16

for.cond.cleanup:                                 ; preds = %for.cond
  ret i32 %res, !dbg !17
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

; CHECK: [[DBG13]] = !DILocation(line: 6,
; CHECK: [[DBG17]] = !DILocation(line: 10,

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "main.ll", directory: "/")
!2 = !{i32 10}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "basic", linkageName: "basic", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
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
