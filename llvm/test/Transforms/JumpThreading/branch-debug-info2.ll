; RUN: opt -passes=jump-threading -S %s -o - | FileCheck %s
define void @prepare_next_shadow() !dbg !7 {
entry:
  br label %for.cond, !dbg !9

for.cond:                                         ; preds = %for.inc, %entry
  br label %shadow_to_ptr.exit, !dbg !10

shadow_to_ptr.exit:                               ; preds = %for.cond
  %shr.i = lshr i64 0, 12, !dbg !11
  %cmp.i = icmp ult i64 %shr.i, undef, !dbg !12
  br i1 %cmp.i, label %cleanup.i, label %if.end.i60, !dbg !13

if.end.i60:                                       ; preds = %shadow_to_ptr.exit
  %sub.i = sub i64 %shr.i, undef, !dbg !14
  %cmp3.i = icmp ugt i64 %sub.i, 32763, !dbg !15
  %conv7.i = trunc i64 %sub.i to i32, !dbg !16
  %spec.select.i = select i1 %cmp3.i, i32 -1, i32 %conv7.i, !dbg !17
; Jump threading is going to fold the select in to the branch. Ensure debug
; info is not lost, and is merged from the select and the branch.
; CHECK-NOT: br i1 %cmp3.i, label %for.inc, label %if.end22
; CHECK: br i1 %cmp3.i, label %for.inc, label %if.end22, !dbg [[DBG:![0-9]+]]
; CHECK: [[DBG]] = !DILocation(line: 9, column: 1, scope: !{{.*}})

  br label %ptr_to_shadow.exit, !dbg !17

cleanup.i:                                        ; preds = %shadow_to_ptr.exit
  br label %ptr_to_shadow.exit, !dbg !19

ptr_to_shadow.exit:                               ; preds = %cleanup.i, %if.end.i60
  %call1861 = phi i32 [ %spec.select.i, %if.end.i60 ], [ -1, %cleanup.i ], !dbg !20
  %cmp19 = icmp slt i32 %call1861, 0, !dbg !21
  br i1 %cmp19, label %for.inc, label %if.end22, !dbg !22

if.end22:                                         ; preds = %ptr_to_shadow.exit
  unreachable, !dbg !23

for.inc:                                          ; preds = %ptr_to_shadow.exit
  br label %for.cond, !dbg !24
}

!llvm.ident = !{!0}
!llvm.dbg.cu = !{!1}
!llvm.debugify = !{!4, !5}
!llvm.module.flags = !{!6}

!0 = !{!""}
!1 = distinct !DICompileUnit(language: DW_LANG_C, file: !2, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !3)
!2 = !DIFile(filename: "jump_threading.ll", directory: "/")
!3 = !{}
!4 = !{i32 16}
!5 = !{i32 0}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DISubprogram(name: "prepare_next_shadow", linkageName: "prepare_next_shadow", scope: null, file: !2, line: 1, type: !8, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1, retainedNodes: !3)
!8 = !DISubroutineType(types: !3)
!9 = !DILocation(line: 1, column: 1, scope: !7)
!10 = !DILocation(line: 2, column: 1, scope: !7)
!11 = !DILocation(line: 3, column: 1, scope: !7)
!12 = !DILocation(line: 4, column: 1, scope: !7)
!13 = !DILocation(line: 5, column: 1, scope: !7)
!14 = !DILocation(line: 6, column: 1, scope: !7)
!15 = !DILocation(line: 7, column: 1, scope: !7)
!16 = !DILocation(line: 8, column: 1, scope: !7)
!17 = !DILocation(line: 9, column: 1, scope: !7)
!19 = !DILocation(line: 11, column: 1, scope: !7)
!20 = !DILocation(line: 12, column: 1, scope: !7)
!21 = !DILocation(line: 13, column: 1, scope: !7)
!22 = !DILocation(line: 14, column: 1, scope: !7)
!23 = !DILocation(line: 15, column: 1, scope: !7)
!24 = !DILocation(line: 16, column: 1, scope: !7)
