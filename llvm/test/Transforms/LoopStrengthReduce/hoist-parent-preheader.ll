; RUN: opt < %s -loop-reduce -verify
target triple = "x86_64-apple-darwin10"

define void @myquicksort(ptr %a) nounwind ssp {
entry:
  br i1 undef, label %loop1, label %return

loop1:                                            ; preds = %bb13.loopexit, %entry
  %indvar419 = phi i64 [ %indvar.next420, %loop2.exit ], [ 0, %entry ]
  %tmp474 = shl i64 %indvar419, 2
  %tmp484 = add i64 %tmp474, 4
  br label %loop2

loop2:                                            ; preds = %loop1, %loop2.backedge
  %indvar414 = phi i64 [ %indvar.next415, %loop2.backedge ], [ 0, %loop1 ]
  %tmp473 = mul i64 %indvar414, -4
  %tmp485 = add i64 %tmp484, %tmp473
  %storemerge4 = getelementptr i8, ptr %a, i64 %tmp485
  %0 = icmp ugt ptr %storemerge4, %a
  br i1 false, label %loop2.exit, label %loop2.backedge

loop2.backedge:                                   ; preds = %loop2
  %indvar.next415 = add i64 %indvar414, 1
  br label %loop2

loop2.exit:                                       ; preds = %loop2
  %indvar.next420 = add i64 %indvar419, 1
  br i1 undef, label %loop1, label %return

return:                                           ; preds = %loop2.exit, %entry
  ret void
}
