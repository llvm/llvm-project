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
