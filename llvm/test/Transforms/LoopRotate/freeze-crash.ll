; RUN: opt -loop-rotate -disable-output %s
; RUN: opt -passes=rotate -disable-output %s

; Make sure we don't crash on this test.
define void @foo(i32* %arg) {
bb:
  %tmp = load i32, i32* %arg, align 4
  br label %bb1

bb1:                                              ; preds = %bb7, %bb
  %tmp2 = phi i32 [ %tmp, %bb ], [ 1, %bb7 ]
  %tmp3 = sub i32 0, %tmp2
  %tmp4 = icmp ult i32 0, %tmp3
  %tmp5 = freeze i1 %tmp4
  br i1 %tmp5, label %bb7, label %bb6

bb6:                                              ; preds = %bb1
  ret void

bb7:                                              ; preds = %bb1
  %tmp8 = getelementptr inbounds i8, i8* undef, i64 8
  br label %bb1
}

