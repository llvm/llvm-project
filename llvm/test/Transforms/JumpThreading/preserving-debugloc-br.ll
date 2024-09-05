; RUN: opt -S -passes=jump-threading < %s | FileCheck %s

; @process_block_branch checks that JumpThreading's processBlock() propagates 
; the debug location to the new branch instruction.

; @process_threadable_edges_branch checks that JumpThreading's processThreadableEdges()
; propagates the debug location to the new branch instruction.

define i32 @process_block_branch(i32 %action) #0 !dbg !5 {
; CHECK-LABEL: define i32 @process_block_branch(
; CHECK:       for.cond:
; CHECK-NEXT:    br label %for.cond, !dbg [[DBG10:![0-9]+]]
;
entry:
  switch i32 %action, label %lor.rhs [
  i32 1, label %if.then
  i32 0, label %lor.end
  ], !dbg !8

if.then:                                          ; preds = %for.cond, %lor.end, %entry
  ret i32 undef, !dbg !9

lor.rhs:                                          ; preds = %entry
  br label %lor.end, !dbg !10

lor.end:                                          ; preds = %lor.rhs, %entry
  %cmp103 = xor i1 undef, undef, !dbg !11
  br i1 %cmp103, label %for.cond, label %if.then, !dbg !12

for.cond:                                         ; preds = %for.body, %lor.end
  br i1 undef, label %if.then, label %for.body, !dbg !13

for.body:                                         ; preds = %for.cond
  br label %for.cond, !dbg !14
}

define void @process_threadable_edges_branch(i32 %value) #0 !dbg !15 {
; CHECK-LABEL: define void @process_threadable_edges_branch(
; CHECK:    L0:
; CHECK:      br label %L2, !dbg [[DBG17:![0-9]+]]
;
entry:
  %cmp = icmp eq i32 %value, 32, !dbg !16
  %add = add i32 %value, 64, !dbg !17
  br i1 %cmp, label %L0, label %L2, !dbg !18

L0:                                               ; preds = %entry
  %0 = call i32 @f2(), !dbg !19
  %1 = call i32 @f2(), !dbg !20
  switch i32 %add, label %L3 [
  i32 32, label %L1
  i32 96, label %L2
  ], !dbg !21

L1:                                               ; preds = %L0
  call void @f3(), !dbg !22
  ret void, !dbg !23

L2:                                               ; preds = %L0, %entry
  call void @f4(i32 %add), !dbg !24
  ret void, !dbg !25

L3:                                               ; preds = %L0
  call void @f3(), !dbg !26
  ret void, !dbg !27
}

declare i32 @f1()

declare i32 @f2()

declare void @f3()

declare void @f4(i32)

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

;.
; CHECK: [[DBG10]] = !DILocation(line: 6,
; CHECK: [[DBG17]] = !DILocation(line: 13,
;.

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "temp.ll", directory: "/")
!2 = !{i32 30}
!3 = !{i32 0}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "process_block_branch", linkageName: "process_block_branch", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 1, column: 1, scope: !5)
!9 = !DILocation(line: 2, column: 1, scope: !5)
!10 = !DILocation(line: 3, column: 1, scope: !5)
!11 = !DILocation(line: 4, column: 1, scope: !5)
!12 = !DILocation(line: 5, column: 1, scope: !5)
!13 = !DILocation(line: 6, column: 1, scope: !5)
!14 = !DILocation(line: 7, column: 1, scope: !5)
!15 = distinct !DISubprogram(name: "process_threadable_edges_branch", linkageName: "process_threadable_edges_branch", scope: null, file: !1, line: 8, type: !6, scopeLine: 8, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!16 = !DILocation(line: 8, column: 1, scope: !15)
!17 = !DILocation(line: 9, column: 1, scope: !15)
!18 = !DILocation(line: 10, column: 1, scope: !15)
!19 = !DILocation(line: 11, column: 1, scope: !15)
!20 = !DILocation(line: 12, column: 1, scope: !15)
!21 = !DILocation(line: 13, column: 1, scope: !15)
!22 = !DILocation(line: 14, column: 1, scope: !15)
!23 = !DILocation(line: 15, column: 1, scope: !15)
!24 = !DILocation(line: 16, column: 1, scope: !15)
!25 = !DILocation(line: 17, column: 1, scope: !15)
!26 = !DILocation(line: 18, column: 1, scope: !15)
!27 = !DILocation(line: 19, column: 1, scope: !15)
