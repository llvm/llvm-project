; RUN: opt < %s -functionattrs -S | FileCheck %s

; CHECK: define i32 @test1(i32 %p, i32 %q)
define i32 @test1(i32 %p, i32 %q) {
entry:
  %cmp = icmp sgt i32 %p, %q
  br i1 %cmp, label %cond.end, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %entry
  %tobool = icmp ne i32 %p, 0
  %tobool1 = icmp ne i32 %q, 0
  %or.cond = and i1 %tobool, %tobool1
  %p.q = select i1 %or.cond, i32 %p, i32 %q
  ret i32 %p.q

cond.end:                                         ; preds = %entry
  ret i32 %p
}

; CHECK: define i32 @test2(i32 %p1, i32 returned %p2)
define i32 @test2(i32 %p1, i32 returned %p2) {
  %_tmp4 = icmp eq i32 %p1, %p2
  br i1 %_tmp4, label %bb2, label %bb1

bb2:                                              ; preds = %0
  ret i32 %p1

bb1:                                              ; preds = %bb1, %0
  br label %bb1
}
