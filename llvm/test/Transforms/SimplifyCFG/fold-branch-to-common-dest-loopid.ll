; RUN: opt -passes=simplifycfg -S %s | FileCheck %s
; RUN: opt -strip-debug %s | opt -passes=simplifycfg -S | FileCheck %s

; Both %loop and %latch branch back to %loop. Folding %latch's branch into
; %loop's must not replace %loop's loop metadata (which carries an unroll
; count) with %latch's, which holds only debug locations and so is absent
; without debug info.

define void @keep_pbi_loopid(ptr %p, i32 %n, i32 %m) !dbg !7 {
; CHECK-LABEL: @keep_pbi_loopid(
; CHECK:         br i1 {{.*}}, label %loop, label %exit, !llvm.loop [[LOOP:![0-9]+]]
; CHECK:       [[LOOP]] = distinct !{[[LOOP]], [[COUNT:![0-9]+]]}
; CHECK:       [[COUNT]] = !{!"llvm.loop.unroll.count", i32 4}
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ], [ %inc, %latch ]
  store i32 %i, ptr %p, align 4
  %inc = add nsw i32 %i, 1
  %c1 = icmp slt i32 %inc, %n
  br i1 %c1, label %loop, label %latch, !llvm.loop !0

latch:
  %c2 = icmp slt i32 %inc, %m
  br i1 %c2, label %loop, label %exit, !llvm.loop !2

exit:
  ret void
}

!llvm.dbg.cu = !{!4}
!llvm.module.flags = !{!6}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.count", i32 4}
!2 = distinct !{!2, !3}
!3 = !DILocation(line: 1, scope: !7)
!4 = distinct !DICompileUnit(language: DW_LANG_C99, file: !5, emissionKind: FullDebug)
!5 = !DIFile(filename: "t.c", directory: "/")
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DISubprogram(name: "keep_pbi_loopid", scope: !5, file: !5, line: 1, type: !8, spFlags: DISPFlagDefinition, unit: !4)
!8 = !DISubroutineType(types: !{})
