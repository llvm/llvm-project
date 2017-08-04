; RUN: opt < %s -loop-extract -S | FileCheck %s

@label = common local_unnamed_addr global i8* null

; CHECK: define
; no outlined function
; CHECK-NOT: define
define i32 @sterix(i32 %n) {
entry:
  %tobool = icmp ne i32 %n, 0
  ; this blockaddress references a basic block that goes in the extracted loop
  %cond = select i1 %tobool, i8* blockaddress(@sterix, %for.cond), i8* blockaddress(@sterix, %exit)
  store i8* %cond, i8** @label
  %cmp5 = icmp sgt i32 %n, 0
  br i1 %cmp5, label %for.body, label %exit

for.cond:
  %mul = shl nsw i32 %s.06, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %exit.loopexit, label %for.body

for.body:
  %i.07 = phi i32 [ %inc, %for.cond ], [ 0, %entry ]
  %s.06 = phi i32 [ %mul, %for.cond ], [ 1, %entry ]
  %inc = add nuw nsw i32 %i.07, 1
  br label %for.cond

exit.loopexit:
  %phitmp = icmp ne i32 %s.06, 2
  %phitmp8 = zext i1 %phitmp to i32
  br label %exit

exit:
  %s.1 = phi i32 [ 1, %entry ], [ %phitmp8, %exit.loopexit ]
  ret i32 %s.1
}
