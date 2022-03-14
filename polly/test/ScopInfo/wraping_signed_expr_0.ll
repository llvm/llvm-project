; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
;    void f(int *A, char N, char p) {
;      for (char i = 0; i < N; i++) {
;        A[i + 3] = 0;
;      }
;    }
;
; The wrap function has no inbounds GEP but the nowrap function has. Therefore,
; we will add the assumption that i+1 won't overflow only to the former.
;
; CHECK:      Function: wrap
; CHECK:      Invalid Context:
; CHECK:      [N] -> {  : N >= 126 }
;
;
; FIXME: This is a negative test as nowrap should not need an assumed context.
;        However %tmp5 in @nowrap is translated to the SCEV <3,+,1><nw><%bb2>
;        which lacks the <nsw> flags we would need to avoid runtime checks.
;
; CHECK:      Function: nowrap
; CHECK:      Invalid Context:
; CHECK-NOT:  [N] -> {  :  }
;
target datalayout = "e-m:e-i8:64-f80:128-n8:16:32:64-S128"

define void @wrap(i32* %A, i8 %N, i8 %p) {
bb:
  br label %bb2

bb2:                                              ; preds = %bb7, %bb
  %indvars.iv = phi i8 [ %indvars.iv.next, %bb7 ], [ 0, %bb ]
  %tmp3 = icmp slt i8 %indvars.iv, %N
  br i1 %tmp3, label %bb4, label %bb8

bb4:                                              ; preds = %bb2
  %tmp5 = add i8 %indvars.iv, 3
  %tmp6 = getelementptr i32, i32* %A, i8 %tmp5
  store i32 0, i32* %tmp6, align 4
  br label %bb7

bb7:                                              ; preds = %bb4
  %indvars.iv.next = add nsw nuw i8 %indvars.iv, 1
  br label %bb2

bb8:                                              ; preds = %bb2
  ret void
}

define void @nowrap(i32* %A, i8 %N, i8 %p) {
bb:
  br label %bb2

bb2:                                              ; preds = %bb7, %bb
  %indvars.iv = phi i8 [ %indvars.iv.next, %bb7 ], [ 0, %bb ]
  %tmp3 = icmp slt i8 %indvars.iv, %N
  br i1 %tmp3, label %bb4, label %bb8

bb4:                                              ; preds = %bb2
  %tmp5 = add nsw nuw i8 %indvars.iv, 3
  %tmp6 = getelementptr inbounds i32, i32* %A, i8 %tmp5
  store i32 0, i32* %tmp6, align 4
  br label %bb7

bb7:                                              ; preds = %bb4
  %indvars.iv.next = add nsw nuw i8 %indvars.iv, 1
  br label %bb2

bb8:                                              ; preds = %bb2
  ret void
}
