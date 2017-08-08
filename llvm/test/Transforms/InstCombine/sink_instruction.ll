; RUN: opt -instcombine -S < %s | FileCheck %s

;; This tests that the instructions in the entry blocks are sunk into each
;; arm of the 'if'.

define i32 @test1(i1 %C, i32 %A, i32 %B) {
; CHECK-LABEL: @test1(
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
; CHECK-LABEL: @test2(
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

define i32 @test3(i32* nocapture readonly %P, i32 %i) {
entry:
  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds i32, i32* %P, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  switch i32 %i, label %sw.epilog [
    i32 5, label %sw.bb
    i32 2, label %sw.bb
  ]

sw.bb:                                            ; preds = %entry, %entry
; CHECK-LABEL: sw.bb:
; CHECK: %idxprom = sext i32 %i to i64
; CHECK: %arrayidx = getelementptr inbounds i32, i32* %P, i64 %idxprom
; CHECK: %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %i
  br label %sw.epilog

sw.epilog:                                        ; preds = %entry, %sw.bb
  %sum.0 = phi i32 [ %add, %sw.bb ], [ 0, %entry ]
  ret i32 %sum.0
}
