; RUN: llc -mcpu=corei7 -mtriple=x86_64-linux < %s | FileCheck %s

define void @foo() !prof !1 {
; Test if a cold block in a loop will be placed at the end of the function
; chain.
;
; CHECK-LABEL: foo:
; CHECK: callq b
; CHECK: callq c
; CHECK: callq e
; CHECK: callq f
; CHECK: callq d

entry:
  br label %header

header:
  call void @b()
  %call = call zeroext i1 @a()
  br i1 %call, label %if.then, label %if.else, !prof !4

if.then:
  call void @c()
  br label %if.end

if.else:
  call void @d()
  br label %if.end

if.end:
  call void @e()
  %call2 = call zeroext i1 @a()
  br i1 %call2, label %header, label %end, !prof !5

end:
  call void @f()
  ret void
}

define void @nested_loop_0(i1 %flag) !prof !1 {
; Test if a block that is cold in the inner loop but not cold in the outer loop
; will merged to the outer loop chain.
;
; CHECK-LABEL: nested_loop_0:
; CHECK: callq c
; CHECK: callq d
; CHECK: callq b
; CHECK: callq e
; CHECK: callq f

entry:
  br label %header

header:
  call void @b()
  %call4 = call zeroext i1 @a()
  br i1 %call4, label %header2, label %end

header2:
  call void @c()
  %call = call zeroext i1 @a()
  br i1 %call, label %if.then, label %if.else, !prof !2

if.then:
  call void @d()
  %call3 = call zeroext i1 @a()
  br i1 %call3, label %header2, label %header, !prof !3

if.else:
  call void @e()
  br i1 %flag, label %header2, label %header, !prof !3

end:
  call void @f()
  ret void
}

define void @nested_loop_1() !prof !1 {
; Test if a cold block in an inner loop will be placed at the end of the
; function chain.
;
; CHECK-LABEL: nested_loop_1:
; CHECK: callq b
; CHECK: callq c
; CHECK: callq e
; CHECK: callq d

entry:
  br label %header

header:
  call void @b()
  br label %header2

header2:
  call void @c()
  %call = call zeroext i1 @a()
  br i1 %call, label %end, label %if.else, !prof !4

if.else:
  call void @d()
  %call2 = call zeroext i1 @a()
  br i1 %call2, label %header2, label %header, !prof !5

end:
  call void @e()
  ret void
}

declare zeroext i1 @a()
declare void @b()
declare void @c()
declare void @d()
declare void @e()
declare void @f()

!1 = !{!"function_entry_count", i64 1}
!2 = !{!"branch_weights", i32 100, i32 1}
!3 = !{!"branch_weights", i32 1, i32 10}
!4 = !{!"branch_weights", i32 1000, i32 1}
!5 = !{!"branch_weights", i32 100, i32 1}
