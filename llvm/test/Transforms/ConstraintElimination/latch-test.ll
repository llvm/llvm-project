target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx15.0.0"

define fastcc i1 @test2() {
entry:
  br label %loop

loop:                                ; preds = %loop, %entry
  %iv = phi i64 [ %iv.next, %loop ], [ -1, %entry ]
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, 0
  br i1 %ec, label %for.cond, label %loop

for.cond:                                         ; preds = %loop
  ret i1 false
}


define fastcc i1 @test1(i32 %bf.load.i.i.i.i724) {
entry:
  br label %for.body4.i.i.i.i

for.body4.i.i.i.i:                                ; preds = %for.body4.i.i.i.i, %entry
  %__n.addr.116.i.i.i.i = phi i64 [ %inc.i.i.i.i, %for.body4.i.i.i.i ], [ -1, %entry ]
  %inc.i.i.i.i = add i64 %__n.addr.116.i.i.i.i, 1
  %exitcond.not.i.i.i.i = icmp eq i64 %inc.i.i.i.i, 0
  br i1 %exitcond.not.i.i.i.i, label %for.cond, label %for.body4.i.i.i.i

for.cond:                                         ; preds = %for.body4.i.i.i.i
  %tobool.not.i.i.i.i.i.i727 = icmp eq i32 %bf.load.i.i.i.i724, 0
  %cond.i.i.i.i.i.i = select i1 %tobool.not.i.i.i.i.i.i727, ptr null, ptr null
  ret i1 false
}


define i8 @test3(i32 %0) {
entry:
  br label %loop.header

loop.header:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %c.1 = icmp eq i32 %iv, 0
  br i1 true, label %then, label %loop.latch

loop.latch:                                     ; preds = %loop.header
  %iv.next = add i32 %iv, 1
  %c.2 = icmp eq i32 %iv.next, 0
  br i1 %c.2, label %exit.3, label %loop.header

then:
  %c.3 = icmp eq i32 %0, 31
  br i1 %c.3, label %exit.2, label %exit


exit:
  ret i8 1

exit.2:
  ret i8 0


exit.3:
  ret i8 2
}


define void @test3(i32 %0) {
entry:
  br label %for.body496.i1357

for.cond.cleanup495.i1365:                        ; preds = %if.else549.i
  ret void

for.body496.i1357:                                ; preds = %if.else549.i, %entry
  %K.01766.i = phi i32 [ -1, %entry ], [ %inc.i1363, %if.else549.i ]
  %cmp499.i1359 = icmp eq i32 %K.01766.i, 0
  br i1 %cmp499.i1359, label %if.then502.i, label %if.else549.i

if.then502.i:                                     ; preds = %for.body496.i1357
  %cmp4.not.i1467.i = icmp eq i32 %0, 31
  br i1 %cmp4.not.i1467.i, label %if.end6.i1470.i, label %_ZN4llvm18SaturatingMultiplyIjEENSt3__19enable_ifIXsr3stdE13is_unsigned_vIT_EES3_E4typeES3_S3_Pb.exit1485.i

if.end6.i1470.i:                                  ; preds = %if.then502.i
  ret void

_ZN4llvm18SaturatingMultiplyIjEENSt3__19enable_ifIXsr3stdE13is_unsigned_vIT_EES3_E4typeES3_S3_Pb.exit1485.i: ; preds = %if.then502.i
  ret void

if.else549.i:                                     ; preds = %for.body496.i1357
  %inc.i1363 = add i32 %K.01766.i, 1
  %exitcond.not.i1364 = icmp eq i32 %inc.i1363, 0
  br i1 %exitcond.not.i1364, label %for.cond.cleanup495.i1365, label %for.body496.i1357
}

