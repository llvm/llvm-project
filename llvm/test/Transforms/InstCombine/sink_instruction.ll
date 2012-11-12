; RUN: opt -instcombine %s -S | FileCheck %s

;; This tests that the instructions in the entry blocks are sunk into each
;; arm of the 'if'.

define i32 @test1(i1 %C, i32 %A, i32 %B) {
; CHECK: @test1
entry:
        %tmp.2 = sdiv i32 %A, %B                ; <i32> [#uses=1]
        %tmp.9 = add i32 %B, %A         ; <i32> [#uses=1]
        br i1 %C, label %then, label %endif

then:           ; preds = %entry
        ret i32 %tmp.9

endif:          ; preds = %entry
; CHECK: sdiv i32
; CHECK-NEXT: ret i32
        ret i32 %tmp.2
}


;; PHI use, sink divide before call.
define i32 @test2(i32 %x) nounwind ssp {
; CHECK: @test2
; CHECK-NOT: sdiv i32
entry:
  br label %bb

bb:                                               ; preds = %bb2, %entry
  %x_addr.17 = phi i32 [ %x, %entry ], [ %x_addr.0, %bb2 ] ; <i32> [#uses=4]
  %i.06 = phi i32 [ 0, %entry ], [ %4, %bb2 ]     ; <i32> [#uses=1]
  %0 = add nsw i32 %x_addr.17, 1                  ; <i32> [#uses=1]
  %1 = sdiv i32 %0, %x_addr.17                    ; <i32> [#uses=1]
  %2 = icmp eq i32 %x_addr.17, 0                  ; <i1> [#uses=1]
  br i1 %2, label %bb1, label %bb2

bb1:                                              ; preds = %bb
; CHECK: bb1:
; CHECK-NEXT: add nsw i32 %x_addr.17, 1
; CHECK-NEXT: sdiv i32
; CHECK-NEXT: tail call i32 @bar()
  %3 = tail call i32 @bar() nounwind       ; <i32> [#uses=0]
  br label %bb2

bb2:                                              ; preds = %bb, %bb1
  %x_addr.0 = phi i32 [ %1, %bb1 ], [ %x_addr.17, %bb ] ; <i32> [#uses=2]
  %4 = add nsw i32 %i.06, 1                       ; <i32> [#uses=2]
  %exitcond = icmp eq i32 %4, 1000000             ; <i1> [#uses=1]
  br i1 %exitcond, label %bb4, label %bb

bb4:                                              ; preds = %bb2
  ret i32 %x_addr.0
}

declare i32 @bar()
