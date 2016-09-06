; RUN: opt < %s -instcombine -S | FileCheck %s

; A == B implies A >u B is false.
; CHECK-LABEL: @test1
; CHECK-NOT: select
; CHECK: call void @foo(i32 10)
define void @test1(i32 %a, i32 %b) {
  %cmp1 = icmp eq i32 %a, %b
  br i1 %cmp1, label %taken, label %end

taken:
  %cmp2 = icmp ugt i32 %a, %b
  %c = select i1 %cmp2, i32 0, i32 10
  call void @foo(i32 %c)
  br label %end

end:
  ret void
}

; If A == B is false then A != B is true.
; CHECK-LABEL: @test2
; CHECK-NOT: select
; CHECK: call void @foo(i32 20)
define void @test2(i32 %a, i32 %b) {
  %cmp1 = icmp eq i32 %a, %b
  br i1 %cmp1, label %end, label %taken

taken:
  %cmp2 = icmp ne i32 %a, %b
  %c = select i1 %cmp2, i32 20, i32 0
  call void @foo(i32 %c)
  br label %end

end:
  ret void
}

; A >u 10 implies A >u 10 is true.
; CHECK-LABEL: @test3
; CHECK-NOT: select
; CHECK: call void @foo(i32 30)
define void @test3(i32 %a) {
  %cmp1 = icmp ugt i32 %a, 10
  br i1 %cmp1, label %taken, label %end

taken:
  %cmp2 = icmp ugt i32 %a, 10
  %c = select i1 %cmp2, i32 30, i32 0
  call void @foo(i32 %c)
  br label %end

end:
  ret void
}

; CHECK-LABEL: @PR23333
; CHECK-NOT: select
; CHECK: ret i8 1
define i8 @PR23333(i8 addrspace(1)* %ptr) {
   %cmp = icmp eq i8 addrspace(1)* %ptr, null
   br i1 %cmp, label %taken, label %end

taken:
   %cmp2 = icmp ne i8 addrspace(1)* %ptr, null
   %res = select i1 %cmp2, i8 2, i8 1
   ret i8 %res

end:
   ret i8 0
}

; We know the condition of the select is true based on a dominating condition.
; Therefore, we can replace %cond with %len. However, now the inner icmp is
; always false and can be elided.
; CHECK-LABEL: @test4
; CHECK-NOT: select
define void @test4(i32 %len) {
entry:
  %0 = call i32 @bar(i32 %len);
  %cmp = icmp ult i32 %len, 4
  br i1 %cmp, label %bb, label %b1
bb:
  %cond = select i1 %cmp, i32 %len, i32 8
; CHECK-NOT:  %cmp11 = icmp eq i32 %{{.*}}, 8
  %cmp11 = icmp eq i32 %cond, 8
; CHECK: br i1 false, label %b0, label %b1
  br i1 %cmp11, label %b0, label %b1

b0:
  call void @foo(i32 %len)
  br label %b1

b1:
; CHECK: phi i32 [ %len, %bb ], [ undef, %b0 ], [ %0, %entry ]
  %1 = phi i32 [ %cond, %bb ], [ undef, %b0 ], [ %0, %entry ]
  br label %ret

ret:
  call void @foo(i32 %1)
  ret void
}

; A >u 10 implies A >u 9 is true.
; CHECK-LABEL: @test5
; CHECK-NOT: select
; CHECK: call void @foo(i32 30)
define void @test5(i32 %a) {
  %cmp1 = icmp ugt i32 %a, 10
  br i1 %cmp1, label %taken, label %end

taken:
  %cmp2 = icmp ugt i32 %a, 9
  %c = select i1 %cmp2, i32 30, i32 0
  call void @foo(i32 %c)
  br label %end

end:
  ret void
}

declare void @foo(i32)
declare i32 @bar(i32)
