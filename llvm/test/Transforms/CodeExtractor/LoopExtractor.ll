; RUN: opt < %s -break-crit-edges -loop-simplify -loop-extract -S | FileCheck %s

; This function has 2 simple loops and they should be extracted into 2 new functions.
define void @test3() {
; CHECK-LABEL: @test3(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %codeRepl1
; CHECK:       codeRepl1:
; CHECK-NEXT:    call void @test3.loop.0()
; CHECK-NEXT:    br label %loop.0.loop.1_crit_edge
; CHECK:       loop.0.loop.1_crit_edge:
; CHECK-NEXT:    br label %codeRepl
; CHECK:       codeRepl:
; CHECK-NEXT:    call void @test3.loop.1()
; CHECK-NEXT:    br label %exit
; CHECK:       exit:
; CHECK-NEXT:    ret void

entry:
  br label %loop.0

loop.0:                                           ; preds = %loop.0, %entry
  %index.0 = phi i32 [ 10, %entry ], [ %next.0, %loop.0 ]
  tail call void @foo()
  %next.0 = add nsw i32 %index.0, -1
  %repeat.0 = icmp sgt i32 %index.0, 1
  br i1 %repeat.0, label %loop.0, label %loop.1

loop.1:                                           ; preds = %loop.0, %loop.1
  %index.1 = phi i32 [ %next.1, %loop.1 ], [ 10, %loop.0 ]
  tail call void @foo()
  %next.1 = add nsw i32 %index.1, -1
  %repeat.1 = icmp sgt i32 %index.1, 1
  br i1 %repeat.1, label %loop.1, label %exit

exit:                                             ; preds = %loop.1
  ret void
}

declare void @foo()

; CHECK-LABEL: define internal void @test3.loop.1()
; CHECK-NEXT:  newFuncRoot:
; CHECK-NEXT:    br label %loop.1
; CHECK:       exit.exitStub:
; CHECK-NEXT:    ret void
; CHECK:       loop.1:
; CHECK-NEXT:    %index.1 = phi i32 [ %next.1, %loop.1.loop.1_crit_edge ], [ 10, %newFuncRoot ]
; CHECK-NEXT:    tail call void @foo()
; CHECK-NEXT:    %next.1 = add nsw i32 %index.1, -1
; CHECK-NEXT:    %repeat.1 = icmp sgt i32 %index.1, 1
; CHECK-NEXT:    br i1 %repeat.1, label %loop.1.loop.1_crit_edge, label %exit.exitStub
; CHECK:       loop.1.loop.1_crit_edge:
; CHECK-NEXT:    br label %loop.1

; CHECK-LABEL: define internal void @test3.loop.0()
; CHECK-NEXT:  newFuncRoot:
; CHECK-NEXT:    br label %loop.0
; CHECK:       loop.0.loop.1_crit_edge.exitStub:
; CHECK-NEXT:    ret void
; CHECK:       loop.0:
; CHECK-NEXT:    %index.0 = phi i32 [ 10, %newFuncRoot ], [ %next.0, %loop.0.loop.0_crit_edge ]
; CHECK-NEXT:    tail call void @foo()
; CHECK-NEXT:    %next.0 = add nsw i32 %index.0, -1
; CHECK-NEXT:    %repeat.0 = icmp sgt i32 %index.0, 1
; CHECK-NEXT:    br i1 %repeat.0, label %loop.0.loop.0_crit_edge, label %loop.0.loop.1_crit_edge.exitStub
; CHECK:       loop.0.loop.0_crit_edge:
; CHECK-NEXT:    br label %loop.0
