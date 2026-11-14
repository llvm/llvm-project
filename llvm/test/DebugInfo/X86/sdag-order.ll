; RUN: llc %s --stop-after=finalize-isel -o - | FileCheck %s

;; Check the DBG_VALUE which is salvaged from the dbg.value using an otherwised
;; unused value is emitted at the correct position in the function.
;; Prior (-) to patching (+), these DBG_VALUEs would sink to the bottom of the
;; function:
;; │   bb.1.if.then:
;; │-    $rax = COPY %1
;; │     DBG_VALUE 0, $noreg, !9, !DIExpression(DW_OP_plus_uconst, 4, DW_OP_stack_value)
;; │+    $rax = COPY %1
;; │     RET 0, $rax

; CHECK: bb.1.if.then:
; CHECK-NEXT: DBG_VALUE 0, $noreg, ![[#]], !DIExpression(DW_OP_plus_uconst, 4, DW_OP_stack_value)

; CHECK-LABEL: name: poison_order
; CHECK: bb.2.unsalvageable:
; CHECK-NEXT: DBG_VALUE $noreg, $noreg, ![[#]], !DIExpression()
; CHECK-NEXT: ADJCALLSTACKDOWN64
; CHECK: CALL64pcrel32 target-flags(x86-plt) @h

;; Check that a dangling value terminated by a later assignment is poisoned at
;; its original position, rather than next to the new assignment.
; CHECK-LABEL: name: poison_before_reassignment
; CHECK: bb.2.reassign:
; CHECK-NEXT: DBG_VALUE $noreg, $noreg, ![[VAR:[0-9]+]], !DIExpression()
; CHECK-NEXT: ADJCALLSTACKDOWN64
; CHECK: CALL64pcrel32 target-flags(x86-plt) @h
; CHECK: DBG_VALUE 7, $noreg, ![[VAR]], !DIExpression()
; CHECK-NOT: DBG_VALUE $noreg, $noreg, ![[VAR]]
; CHECK: RET 0

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @badger(ptr sret(i64) %sret) !dbg !5 {
entry:
  %f.i = getelementptr i8, ptr null, i64 4
  br label %if.then

if.then:                                          ; preds = %entry
  tail call void @llvm.dbg.value(metadata ptr %f.i, metadata !9, metadata !DIExpression()), !dbg !11
  ret void
}

define void @poison_order(i1 %cond) !dbg !12 {
entry:
  %call = tail call i32 @g(i32 0), !dbg !16
  br i1 %cond, label %return, label %unsalvageable

return:                                           ; preds = %entry
  ret void

unsalvageable:                                    ; preds = %entry
  tail call void @llvm.dbg.value(metadata i32 %call, metadata !14, metadata !DIExpression()), !dbg !17
  tail call void @h(), !dbg !18
  store volatile i32 1, ptr @q, align 4, !dbg !19
  ret void
}

define void @poison_before_reassignment(i1 %cond) !dbg !20 {
entry:
  %call = tail call i32 @g(i32 0), !dbg !23
  br i1 %cond, label %return, label %reassign

return:                                           ; preds = %entry
  ret void

reassign:                                         ; preds = %entry
  tail call void @llvm.dbg.value(metadata i32 %call, metadata !22, metadata !DIExpression()), !dbg !24
  tail call void @h(), !dbg !25
  tail call void @llvm.dbg.value(metadata i32 7, metadata !22, metadata !DIExpression()), !dbg !26
  store volatile i32 2, ptr @q, align 4, !dbg !27
  ret void
}

@q = global i32 0

declare i32 @g(i32)
declare void @h()

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.ll", directory: "/")
!2 = !{i32 3}
!3 = !{i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "_ZNK1d1gEv", linkageName: "_ZNK1d1gEv", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !{!9}
!9 = !DILocalVariable(name: "1", scope: !5, file: !1, line: 1, type: !10)
!10 = !DIBasicType(name: "ty64", size: 64, encoding: DW_ATE_unsigned)
!11 = !DILocation(line: 5, column: 1, scope: !5)
!12 = distinct !DISubprogram(name: "poison_order", scope: !1, file: !1, line: 10, type: !6, scopeLine: 10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !13)
!13 = !{!14}
!14 = !DILocalVariable(name: "x", scope: !12, file: !1, line: 11, type: !15)
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !DILocation(line: 11, column: 11, scope: !12)
!17 = !DILocation(line: 15, column: 3, scope: !12)
!18 = !DILocation(line: 16, column: 3, scope: !12)
!19 = !DILocation(line: 17, column: 3, scope: !12)
!20 = distinct !DISubprogram(name: "poison_before_reassignment", scope: !1, file: !1, line: 20, type: !6, scopeLine: 20, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !21)
!21 = !{!22}
!22 = !DILocalVariable(name: "x", scope: !20, file: !1, line: 21, type: !15)
!23 = !DILocation(line: 21, column: 11, scope: !20)
!24 = !DILocation(line: 25, column: 3, scope: !20)
!25 = !DILocation(line: 26, column: 3, scope: !20)
!26 = !DILocation(line: 27, column: 3, scope: !20)
!27 = !DILocation(line: 28, column: 3, scope: !20)
