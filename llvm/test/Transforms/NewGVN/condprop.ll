; RUN: opt < %s -basicaa -newgvn -S | FileCheck %s


declare void @foo(i1)
declare void @bar(i32)

; CHECK-LABEL: @test3(
define void @test3(i32 %x, i32 %y) {
  %xz = icmp eq i32 %x, 0
  %yz = icmp eq i32 %y, 0
  %z = and i1 %xz, %yz
  br i1 %z, label %both_zero, label %nope
both_zero:
  call void @foo(i1 %xz)
; CHECK: call void @foo(i1 true)
  call void @foo(i1 %yz)
; CHECK: call void @foo(i1 true)
  call void @bar(i32 %x)
; CHECK: call void @bar(i32 0)
  call void @bar(i32 %y)
; CHECK: call void @bar(i32 0)
  ret void
nope:
  call void @foo(i1 %z)
; CHECK: call void @foo(i1 false)
  ret void
}

; CHECK-LABEL: @test5(
define i1 @test5(i32 %x, i32 %y) {
  %cmp = icmp eq i32 %x, %y
  br i1 %cmp, label %same, label %different

same:
  %cmp2 = icmp ne i32 %x, %y
; CHECK: ret i1 false
  ret i1 %cmp2

different:
  %cmp3 = icmp eq i32 %x, %y
; CHECK: ret i1 false
  ret i1 %cmp3
}


; CHECK-LABEL: @test7(
define i1 @test7(i32 %x, i32 %y) {
  %cmp = icmp sgt i32 %x, %y
  br i1 %cmp, label %same, label %different

same:
  %cmp2 = icmp sle i32 %x, %y
; CHECK: ret i1 false
  ret i1 %cmp2

different:
  %cmp3 = icmp sgt i32 %x, %y
; CHECK: ret i1 false
  ret i1 %cmp3
}

; CHECK-LABEL: @test7_fp(
define i1 @test7_fp(float %x, float %y) {
  %cmp = fcmp ogt float %x, %y
  br i1 %cmp, label %same, label %different

same:
  %cmp2 = fcmp ule float %x, %y
; CHECK: ret i1 false
  ret i1 %cmp2

different:
  %cmp3 = fcmp ogt float %x, %y
; CHECK: ret i1 false
  ret i1 %cmp3
}

; PR1768
; CHECK-LABEL: @test9(
define i32 @test9(i32 %i, i32 %j) {
  %cmp = icmp eq i32 %i, %j
  br i1 %cmp, label %cond_true, label %ret

cond_true:
  %diff = sub i32 %i, %j
  ret i32 %diff
; CHECK: ret i32 0

ret:
  ret i32 5
; CHECK: ret i32 5
}

; PR1768
; CHECK-LABEL: @test10(
define i32 @test10(i32 %j, i32 %i) {
  %cmp = icmp eq i32 %i, %j
  br i1 %cmp, label %cond_true, label %ret

cond_true:
  %diff = sub i32 %i, %j
  ret i32 %diff
; CHECK: ret i32 0

ret:
  ret i32 5
; CHECK: ret i32 5
}

declare i32 @yogibar()

; CHECK-LABEL: @test11(
define i32 @test11(i32 %x) {
  %v0 = call i32 @yogibar()
  %v1 = call i32 @yogibar()
  %cmp = icmp eq i32 %v0, %v1
  br i1 %cmp, label %cond_true, label %next

cond_true:
  ret i32 %v1
; CHECK: ret i32 %v0

next:
  %cmp2 = icmp eq i32 %x, %v0
  br i1 %cmp2, label %cond_true2, label %next2

cond_true2:
  ret i32 %v0
; CHECK: ret i32 %x

next2:
  ret i32 0
}

; CHECK-LABEL: @test12(
define i32 @test12(i32 %x) {
  %cmp = icmp eq i32 %x, 0
  br i1 %cmp, label %cond_true, label %cond_false

cond_true:
  br label %ret

cond_false:
  br label %ret

ret:
  %res = phi i32 [ %x, %cond_true ], [ %x, %cond_false ]
; CHECK: %res = phi i32 [ 0, %cond_true ], [ %x, %cond_false ]
  ret i32 %res
}
