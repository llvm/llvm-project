; RUN: llc -mtriple=x86_64-unknown-linux-gnu --start-before loop-reduce --stop-after loop-reduce %s -o %t

; No check in a crash test.

define i32 @pr201758() {
entry:
  br label %for.cond3
for.cond3:
  tail call void @llvm.assume(i1 false)
  br label %while.cond
while.cond:
  %k.2 = phi i32 [ 0, %while.cond ], [ 1, %for.cond3 ]
  br i1 false, label %while.cond.1, label %while.cond
while.cond.1:
  %k.2.1 = phi i32 [ %dec.1, %while.cond.1 ], [ %k.2, %while.cond ]
  %dec.1 = add i32 %k.2.1, 1
  br i1 false, label %while.cond.2, label %while.cond.1
while.cond.2:
  %k.2.2 = phi i32 [ %dec.2, %while.cond.2 ], [ %k.2.1, %while.cond.1 ]
  %dec.2 = add i32 %k.2.2, 1
  br i1 false, label %while.cond.3, label %while.cond.2
while.cond.3:
  %k.2.3 = phi i32 [ %dec.3, %while.cond.3 ], [ %k.2.2, %while.cond.2 ]
  %dec.3 = add i32 %k.2.3, 1
  br i1 false, label %while.cond.4, label %while.cond.3
while.cond.4:
  %k.2.4 = phi i32 [ %dec.4, %while.cond.4 ], [ %k.2.3, %while.cond.3 ]
  %dec.4 = add i32 %k.2.4, 1
  br i1 false, label %while.cond.5, label %while.cond.4
while.cond.5:
  %k.2.5 = phi i32 [ %dec.5, %while.cond.5 ], [ %k.2.4, %while.cond.4 ]
  %dec.5 = add i32 %k.2.5, 1
  br i1 false, label %while.cond.115, label %while.cond.5
while.cond.115:
  %k.2.112 = phi i32 [ %dec.114, %while.cond.115 ], [ %k.2.5, %while.cond.5 ]
  %dec.114 = add i32 %k.2.112, 1
  br i1 false, label %while.cond.1.1, label %while.cond.115
while.cond.1.1:
  %k.2.1.1 = phi i32 [ %dec.1.1, %while.cond.1.1 ], [ %k.2.112, %while.cond.115 ]
  %dec.1.1 = add i32 %k.2.1.1, 1
  br i1 false, label %while.cond.2.1, label %while.cond.1.1
while.cond.2.1:
  %k.2.2.1 = phi i32 [ %dec.2.1, %while.cond.2.1 ], [ %k.2.1.1, %while.cond.1.1 ]
  %dec.2.1 = add i32 %k.2.2.1, 1
  br i1 false, label %while.cond.3.1, label %while.cond.2.1
while.cond.3.1:
  %k.2.3.1 = phi i32 [ %dec.3.1, %while.cond.3.1 ], [ %k.2.2.1, %while.cond.2.1 ]
  %dec.3.1 = add i32 %k.2.3.1, 1
  br i1 false, label %while.cond.4.1, label %while.cond.3.1
while.cond.4.1:
  %k.2.4.1 = phi i32 [ %dec.4.1, %while.cond.4.1 ], [ %k.2.3.1, %while.cond.3.1 ]
  %dec.4.1 = add i32 %k.2.4.1, 1
  br i1 false, label %while.cond.5.1, label %while.cond.4.1
while.cond.5.1:
  %k.2.5.1 = phi i32 [ %dec.5.1, %while.cond.5.1 ], [ %k.2.4.1, %while.cond.4.1 ]
  %dec.5.1 = add i32 %k.2.5.1, 1
  br i1 false, label %while.cond.221, label %while.cond.5.1
while.cond.221:
  %k.2.218 = phi i32 [ %dec.220, %while.cond.221 ], [ %k.2.5.1, %while.cond.5.1 ]
  %dec.220 = add i32 %k.2.218, 1
  br i1 false, label %while.cond.1.2, label %while.cond.221
while.cond.1.2:
  %k.2.1.2 = phi i32 [ %dec.1.2, %while.cond.1.2 ], [ %k.2.218, %while.cond.221 ]
  %dec.1.2 = add i32 %k.2.1.2, 1
  br i1 false, label %while.cond.2.2, label %while.cond.1.2
while.cond.2.2:
  %k.2.2.2 = phi i32 [ %dec.2.2, %while.cond.2.2 ], [ %k.2.1.2, %while.cond.1.2 ]
  %dec.2.2 = add i32 %k.2.2.2, 1
  br i1 false, label %while.cond.3.2, label %while.cond.2.2
while.cond.3.2:
  %k.2.3.2 = phi i32 [ %dec.3.2, %while.cond.3.2 ], [ %k.2.2.2, %while.cond.2.2 ]
  %dec.3.2 = add i32 %k.2.3.2, 1
  br i1 false, label %while.cond.4.2, label %while.cond.3.2
while.cond.4.2:
  %k.2.4.2 = phi i32 [ %dec.4.2, %while.cond.4.2 ], [ %k.2.3.2, %while.cond.3.2 ]
  %dec.4.2 = add i32 %k.2.4.2, 1
  br i1 false, label %while.cond.5.2, label %while.cond.4.2
while.cond.5.2:
  %k.2.5.2 = phi i32 [ %dec.5.2, %while.cond.5.2 ], [ %k.2.4.2, %while.cond.4.2 ]
  %dec.5.2 = add i32 %k.2.5.2, 1
  br i1 false, label %while.cond.327, label %while.cond.5.2
while.cond.327:
  %k.2.324 = phi i32 [ %dec.326, %while.cond.327 ], [ %k.2.5.2, %while.cond.5.2 ]
  %dec.326 = add i32 %k.2.324, 1
  br i1 false, label %while.cond.1.3, label %while.cond.327
while.cond.1.3:
  %k.2.1.3 = phi i32 [ %dec.1.3, %while.cond.1.3 ], [ %k.2.324, %while.cond.327 ]
  %dec.1.3 = add i32 %k.2.1.3, 1
  br i1 false, label %while.cond.2.3, label %while.cond.1.3
while.cond.2.3:
  %k.2.2.3 = phi i32 [ %dec.2.3, %while.cond.2.3 ], [ %k.2.1.3, %while.cond.1.3 ]
  %dec.2.3 = add i32 %k.2.2.3, 1
  br i1 false, label %while.cond.3.3, label %while.cond.2.3
while.cond.3.3:
  %k.2.3.3 = phi i32 [ %dec.3.3, %while.cond.3.3 ], [ %k.2.2.3, %while.cond.2.3 ]
  %dec.3.3 = add i32 %k.2.3.3, 1
  br i1 false, label %while.cond.4.3, label %while.cond.3.3
while.cond.4.3:
  %k.2.4.3 = phi i32 [ %dec.4.3, %while.cond.4.3 ], [ %k.2.3.3, %while.cond.3.3 ]
  %dec.4.3 = add i32 %k.2.4.3, 1
  br i1 false, label %while.cond.5.3, label %while.cond.4.3
while.cond.5.3:
  %k.2.5.3 = phi i32 [ %dec.5.3, %while.cond.5.3 ], [ %k.2.4.3, %while.cond.4.3 ]
  %dec.5.3 = add i32 %k.2.5.3, 1
  br i1 false, label %while.cond.433, label %while.cond.5.3
while.cond.433:
  %k.2.430 = phi i32 [ %dec.432, %while.cond.433 ], [ %k.2.5.3, %while.cond.5.3 ]
  %dec.432 = add i32 %k.2.430, 1
  br i1 false, label %while.cond.1.4, label %while.cond.433
while.cond.1.4:
  %k.2.1.4 = phi i32 [ %dec.1.4, %while.cond.1.4 ], [ %k.2.430, %while.cond.433 ]
  %dec.1.4 = add i32 %k.2.1.4, 1
  br i1 false, label %while.cond.2.4, label %while.cond.1.4
while.cond.2.4:
  %k.2.2.4 = phi i32 [ %dec.2.4, %while.cond.2.4 ], [ %k.2.1.4, %while.cond.1.4 ]
  %dec.2.4 = add i32 %k.2.2.4, 1
  br i1 false, label %while.cond.3.4, label %while.cond.2.4
while.cond.3.4:
  %k.2.3.4 = phi i32 [ %dec.3.4, %while.cond.3.4 ], [ %k.2.2.4, %while.cond.2.4 ]
  %dec.3.4 = add i32 %k.2.3.4, 1
  br i1 false, label %while.cond.4.4, label %while.cond.3.4
while.cond.4.4:
  %k.2.4.4 = phi i32 [ %dec.4.4, %while.cond.4.4 ], [ %k.2.3.4, %while.cond.3.4 ]
  %dec.4.4 = add i32 %k.2.4.4, 1
  br i1 false, label %while.cond.5.4, label %while.cond.4.4
while.cond.5.4:
  %k.2.5.4 = phi i32 [ %dec.5.4, %while.cond.5.4 ], [ %k.2.4.4, %while.cond.4.4 ]
  %dec.5.4 = add i32 %k.2.5.4, 1
  br i1 false, label %while.cond.539, label %while.cond.5.4
while.cond.539:
  %k.2.536 = phi i32 [ %dec.538, %while.cond.539 ], [ %k.2.5.4, %while.cond.5.4 ]
  %dec.538 = add i32 %k.2.536, 1
  br i1 false, label %while.cond.1.5, label %while.cond.539
while.cond.1.5:
  %k.2.1.5 = phi i32 [ %dec.1.5, %while.cond.1.5 ], [ %k.2.536, %while.cond.539 ]
  %dec.1.5 = add i32 %k.2.1.5, 1
  br i1 false, label %while.cond.2.5, label %while.cond.1.5
while.cond.2.5:
  %k.2.2.5 = phi i32 [ %dec.2.5, %while.cond.2.5 ], [ %k.2.1.5, %while.cond.1.5 ]
  %dec.2.5 = add i32 %k.2.2.5, 1
  br i1 false, label %while.cond.3.5, label %while.cond.2.5
while.cond.3.5:
  %k.2.3.5 = phi i32 [ %dec.3.5, %while.cond.3.5 ], [ %k.2.2.5, %while.cond.2.5 ]
  %dec.3.5 = add i32 %k.2.3.5, -1
  br i1 false, label %while.cond.4.5, label %while.cond.3.5
while.cond.4.5:
  %k.2.4.5 = phi i32 [ %dec.4.5, %while.cond.4.5 ], [ %k.2.3.5, %while.cond.3.5 ]
  %dec.4.5 = add i32 %k.2.4.5, -1
  br i1 false, label %while.cond.5.5, label %while.cond.4.5
while.cond.5.5:
  store i32 %k.2.4.5, ptr null, align 4
  br label %for.cond3
}
declare void @llvm.assume(i1 noundef)
