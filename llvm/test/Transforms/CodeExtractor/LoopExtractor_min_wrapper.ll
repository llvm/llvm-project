; RUN: opt < %s -break-crit-edges -loop-simplify -loop-extract -S | FileCheck %s

; This function is just a minimal wrapper around a loop and should not be extracted.
define void @test() {
; CHECK-LABEL: @test(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %loop
; CHECK:       loop:
; CHECK-NEXT:    %index = phi i32 [ 0, %entry ], [ %next, %loop.loop_crit_edge ]
; CHECK-NEXT:    call void @foo()
; CHECK-NEXT:    %next = add nsw i32 %index, -1
; CHECK-NEXT:    %repeat = icmp sgt i32 %index, 1
; CHECK-NEXT:    br i1 %repeat, label %loop.loop_crit_edge, label %exit
; CHECK:       loop.loop_crit_edge:
; CHECK-NEXT:    br label %loop
; CHECK:       exit:
; CHECK-NEXT:    ret void

entry:
  br label %loop

loop:                                             ; preds = %loop, %entry
  %index = phi i32 [ 0, %entry ], [ %next, %loop ]
  call void @foo()
  %next = add nsw i32 %index, -1
  %repeat = icmp sgt i32 %index, 1
  br i1 %repeat, label %loop, label %exit

exit:                                             ; preds = %loop
  ret void
}

declare void @foo()

; CHECK-NOT: define
