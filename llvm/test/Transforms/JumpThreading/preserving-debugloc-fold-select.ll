; RUN: opt < %s -S -passes=jump-threading | FileCheck %s

; Test the debug location update of the newly created PHINode
; which replaces the select instruction in .exit block.

define i32 @unfold3(i32 %u, i32 %v, i32 %w, i32 %x, i32 %y, i32 %z, i32 %j) !dbg !5 {
; CHECK:       .exit.thread4:
; CHECK-NEXT:    [[TMP0:%.*]] = phi i32 {{.*}}, !dbg [[DBG29:![0-9]+]]
; CHECK-NEXT:    ret i32 [[TMP0]], !dbg [[DBG30:![0-9]+]]
;
; CHECK: [[DBG29]] = !DILocation(line: 13,
;
entry:
  %add3 = add nsw i32 %j, 2, !dbg !19
  %cmp.i = icmp slt i32 %u, %v, !dbg !20
  br i1 %cmp.i, label %.exit, label %cond.false.i, !dbg !21

cond.false.i:                                     ; preds = %entry
  %cmp4.i = icmp sgt i32 %u, %v, !dbg !22
  br i1 %cmp4.i, label %.exit, label %cond.false.6.i, !dbg !23

cond.false.6.i:                                   ; preds = %cond.false.i
  %cmp8.i = icmp slt i32 %w, %x, !dbg !24
  br i1 %cmp8.i, label %.exit, label %cond.false.10.i, !dbg !25

cond.false.10.i:                                  ; preds = %cond.false.6.i
  %cmp13.i = icmp sgt i32 %w, %x, !dbg !26
  br i1 %cmp13.i, label %.exit, label %cond.false.15.i, !dbg !27

cond.false.15.i:                                  ; preds = %cond.false.10.i
  %phitmp = icmp sge i32 %y, %z, !dbg !28
  br label %.exit, !dbg !29

.exit:                                            ; preds = %cond.false.15.i, %cond.false.10.i, %cond.false.6.i, %cond.false.i, %entry
  %cond23.i = phi i1 [ false, %entry ], [ true, %cond.false.i ], [ false, %cond.false.6.i ], [ %phitmp, %cond.false.15.i ], [ true, %cond.false.10.i ], !dbg !30
  %j.add3 = select i1 %cond23.i, i32 %j, i32 %add3, !dbg !31
  ret i32 %j.add3, !dbg !32
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "preserving-debugloc-trytofoldselect.ll", directory: "/")
!2 = !{i32 14}
!3 = !{i32 8}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "unfold3", linkageName: "unfold3", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !{!9, !11, !13, !14, !15, !16, !17, !18}
!9 = !DILocalVariable(name: "1", scope: !5, file: !1, line: 1, type: !10)
!10 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!11 = !DILocalVariable(name: "2", scope: !5, file: !1, line: 2, type: !12)
!12 = !DIBasicType(name: "ty8", size: 8, encoding: DW_ATE_unsigned)
!13 = !DILocalVariable(name: "3", scope: !5, file: !1, line: 4, type: !12)
!14 = !DILocalVariable(name: "4", scope: !5, file: !1, line: 6, type: !12)
!15 = !DILocalVariable(name: "5", scope: !5, file: !1, line: 8, type: !12)
!16 = !DILocalVariable(name: "6", scope: !5, file: !1, line: 10, type: !12)
!17 = !DILocalVariable(name: "7", scope: !5, file: !1, line: 12, type: !12)
!18 = !DILocalVariable(name: "8", scope: !5, file: !1, line: 13, type: !10)
!19 = !DILocation(line: 1, column: 1, scope: !5)
!20 = !DILocation(line: 2, column: 1, scope: !5)
!21 = !DILocation(line: 3, column: 1, scope: !5)
!22 = !DILocation(line: 4, column: 1, scope: !5)
!23 = !DILocation(line: 5, column: 1, scope: !5)
!24 = !DILocation(line: 6, column: 1, scope: !5)
!25 = !DILocation(line: 7, column: 1, scope: !5)
!26 = !DILocation(line: 8, column: 1, scope: !5)
!27 = !DILocation(line: 9, column: 1, scope: !5)
!28 = !DILocation(line: 10, column: 1, scope: !5)
!29 = !DILocation(line: 11, column: 1, scope: !5)
!30 = !DILocation(line: 12, column: 1, scope: !5)
!31 = !DILocation(line: 13, column: 1, scope: !5)
!32 = !DILocation(line: 14, column: 1, scope: !5)
