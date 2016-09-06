; RUN: llc -mcpu=corei7 -mtriple=x86_64-linux -force-precise-rotation-cost < %s | FileCheck %s -check-prefix=CHECK

define void @bar()  {
; Test that all edges in the loop chain are fall through with profile data.
;
; CHECK-LABEL: bar:
; CHECK: latch
; CHECK: header
; CHECK: if.then
; CHECK: end

entry:
  br label %header

header:
  call void @e()
  %call = call zeroext i1 @a()
  br i1 %call, label %if.then, label %latch, !prof !1

if.then:
  call void @f()
  %call3 = call zeroext i1 @a()
  br i1 %call3, label %latch, label %end, !prof !2

latch:
  call void @h()
  %call2 = call zeroext i1 @a()
  br i1 %call2, label %header, label %end, !prof !3

end:
  ret void
}

declare zeroext i1 @a()
declare void @e()
declare void @f()
declare void @g()
declare void @h()

!1 = !{!"branch_weights", i32 16, i32 16}
!2 = !{!"branch_weights", i32 97, i32 3}
!3 = !{!"branch_weights", i32 97, i32 3}
