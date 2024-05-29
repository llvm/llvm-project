; RUN: opt < %s -loop-reduce -S 2>&1 | FileCheck %s
; RUN: opt --try-experimental-debuginfo-iterators < %s -loop-reduce -S 2>&1 | FileCheck %s
;; This test case checks that whether the new icmp instruction preserves
;; the debug location of the original instruction for %exitcond
; CHECK: icmp uge i32 %indvar.next, %n, !dbg ![[DBGLOC:[0-9]+]]
; CHECK: ![[DBGLOC]] = !DILocation(line: 6, column: 1, scope

; ModuleID = 'simplified-dbg.bc'
source_filename = "abc.ll"
target datalayout = "n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define void @foobar(i32 %n) !dbg !6 {
bb.nph:
  %cond = icmp eq i32 %n, 0, !dbg !16
  call void @llvm.dbg.value(metadata i1 %cond, metadata !9, metadata !DIExpression()), !dbg !16
  %umax = select i1 %cond, i32 1, i32 %n, !dbg !17
  call void @llvm.dbg.value(metadata i32 %umax, metadata !11, metadata !DIExpression()), !dbg !17
  br label %bb, !dbg !18

bb:                                               ; preds = %bb, %bb.nph
  %i.03 = phi i32 [ 0, %bb.nph ], [ %indvar.next, %bb ], !dbg !19
  call void @llvm.dbg.value(metadata i32 %i.03, metadata !13, metadata !DIExpression()), !dbg !19
  %indvar.next = add nuw nsw i32 %i.03, 1, !dbg !20
  call void @llvm.dbg.value(metadata i32 %indvar.next, metadata !14, metadata !DIExpression()), !dbg !20
  %exitcond = icmp eq i32 %indvar.next, %umax, !dbg !21
  call void @llvm.dbg.value(metadata i1 %exitcond, metadata !15, metadata !DIExpression()), !dbg !21
  br i1 %exitcond, label %return, label %bb, !dbg !22

return:                                           ; preds = %bb
  ret void, !dbg !23
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!3, !4}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "simplified.ll", directory: "/")
!2 = !{}
!3 = !{i32 8}
!4 = !{i32 5}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "foobar", linkageName: "foobar", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!7 = !DISubroutineType(types: !2)
!8 = !{!9, !11, !13, !14, !15}
!9 = !DILocalVariable(name: "1", scope: !6, file: !1, line: 1, type: !10)
!10 = !DIBasicType(name: "ty8", size: 8, encoding: DW_ATE_unsigned)
!11 = !DILocalVariable(name: "2", scope: !6, file: !1, line: 2, type: !12)
!12 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!13 = !DILocalVariable(name: "3", scope: !6, file: !1, line: 4, type: !12)
!14 = !DILocalVariable(name: "4", scope: !6, file: !1, line: 5, type: !12)
!15 = !DILocalVariable(name: "5", scope: !6, file: !1, line: 6, type: !10)
!16 = !DILocation(line: 1, column: 1, scope: !6)
!17 = !DILocation(line: 2, column: 1, scope: !6)
!18 = !DILocation(line: 3, column: 1, scope: !6)
!19 = !DILocation(line: 4, column: 1, scope: !6)
!20 = !DILocation(line: 5, column: 1, scope: !6)
!21 = !DILocation(line: 6, column: 1, scope: !6)
!22 = !DILocation(line: 7, column: 1, scope: !6)
!23 = !DILocation(line: 8, column: 1, scope: !6)
