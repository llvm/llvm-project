; Check that we don't insert unnecessary CC spills
;
; RUN: llc < %s -mtriple=s390x-linux-gnu

declare signext i32 @f()

define signext i32 @test(ptr %ptr) {
; CHECK-NOT: ipm

entry:
  %0 = load i32, ptr %ptr, align 4
  %tobool = icmp eq i32 %0, 0
  %call = tail call signext i32 @f()
  %1 = icmp slt i32 %call, 40
  %2 = or i1 %tobool, %1
  %retv = select i1 %2, i32 %call, i32 40
  ret i32 %retv
}

