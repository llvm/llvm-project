; RUN: llc < %s -mtriple armv7--linux-gnueabi

define arm_aapcscc i8* @__strtok_r_1c(i8* %arg, i8 signext %arg1, i8** nocapture %arg2) nounwind {
bb:
  br label %bb3

bb3:                                              ; preds = %bb3, %bb
  %tmp = phi i8* [ %tmp5, %bb3 ], [ %arg, %bb ]
  %tmp4 = load i8* %tmp, align 1
  %tmp5 = getelementptr inbounds i8* %tmp, i32 1
  br i1 undef, label %bb3, label %bb7

bb7:                                              ; preds = %bb13, %bb3
  %tmp8 = phi i8 [ %tmp14, %bb13 ], [ %tmp4, %bb3 ]
  %tmp9 = phi i8* [ %tmp12, %bb13 ], [ %tmp, %bb3 ]
  %tmp10 = icmp ne i8 %tmp8, %arg1
  %tmp12 = getelementptr inbounds i8* %tmp9, i32 1
  br i1 %tmp10, label %bb13, label %bb15

bb13:                                             ; preds = %bb7
  %tmp14 = load i8* %tmp12, align 1
  br label %bb7

bb15:                                             ; preds = %bb7
  store i8* %tmp9, i8** %arg2, align 4
  ret i8* %tmp
}
