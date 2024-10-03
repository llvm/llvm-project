; RUN: opt -S -passes=simplifycfg < %s | FileCheck %s

; Check that SimplifyCFGPass's performBlockTailMerging() propagates
; the debug location of the old block terminator to the new branch
; instruction.

define i32 @foo(i64 %x, i64 %y) !dbg !5 {
; CHECK-LABEL: define i32 @foo(
; CHECK:       a:
; CHECK:         br label %[[COMMON_RET:.*]], !dbg [[DBG14:![0-9]+]]
; CHECK:       b:
; CHECK:         br label %[[COMMON_RET]], !dbg [[DBG17:![0-9]+]]
;
entry:
  %eq = icmp eq i64 %x, %y, !dbg !8
  br i1 %eq, label %b, label %switch, !dbg !9

switch:                                           ; preds = %entry
  %lt = icmp slt i64 %x, %y, !dbg !10
  %qux = select i1 %lt, i32 0, i32 2, !dbg !11
  switch i32 %qux, label %bees [
  i32 0, label %a
  i32 1, label %b
  i32 2, label %b
  ], !dbg !12

a:                                                ; preds = %switch
  tail call void @bees.a(), !dbg !13
  ret i32 1, !dbg !14

b:                                                ; preds = %switch, %switch, %entry
  %retval = phi i32 [ 0, %switch ], [ 0, %switch ], [ 2, %entry ], !dbg !15
  tail call void @bees.b(), !dbg !16
  ret i32 %retval, !dbg !17

bees:                                             ; preds = %switch
  tail call void @llvm.trap(), !dbg !18
  unreachable, !dbg !19
}


declare void @llvm.trap()
declare void @bees.a()
declare void @bees.b()

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

; CHECK: [[DBG14]] = !DILocation(line: 7,
; CHECK: [[DBG17]] = !DILocation(line: 10,

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "main.ll", directory: "/")
!2 = !{i32 12}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
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
!18 = !DILocation(line: 11, column: 1, scope: !5)
!19 = !DILocation(line: 12, column: 1, scope: !5)
