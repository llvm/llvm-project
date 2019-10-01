; RUN: opt < %s -simple-loop-unswitch -enable-nontrivial-unswitch -S 2>&1 | FileCheck %s
; RUN: opt < %s -passes='unswitch<nontrivial>' -S 2>&1 | FileCheck %s
;
; Checking that (dead) blocks from inner loop are deleted after unswitch.
;
declare void @foo()

; CHECK-LABEL: @Test
define void @Test(i32) {
entry:
  br label %outer
outer:
  %oi = phi i32 [ 0, %entry ], [ %oinc, %outer_continue]
  br label %inner
inner:
  %ii = phi i32 [ 0, %outer ], [ %iinc, %continue]
  call void @foo() 
  switch i32 %0, label %get_out2 [
    i32 0, label %continue
    i32 1, label %case1
    i32 2, label %get_out
  ]
;
; since we unswitch on the above switch, %case1 and %continue blocks
; become dead in the original loop
;
; CHECK-NOT: case1:
case1:
  br label %continue
; CHECK-NOT: {{^}}continue:
continue:
  %iinc = add i32 %ii, 1
  %icmp = icmp eq i32 %ii, 100
  br i1 %icmp, label %inner, label %outer_continue

outer_continue:
  %oinc = add i32 %oi, 1
  %ocmp = icmp eq i32 %oi, 100
  br i1 %ocmp, label %outer, label %get_out

get_out:
  ret void
get_out2:
  unreachable
}

;
; This comes from PR38778
; CHECK-LABEL: @Test2
define void @Test2(i32) {
header:
  br label %loop
loop:
  switch i32 %0, label %continue [
    i32 -2147483648, label %check
    i32 98, label %guarded1
    i32 99, label %guarded2
  ]
; CHECK-NOT: {{^}}guarded1:
guarded1:
  br i1 undef, label %continue, label %leave
guarded2:
  br label %continue
check:
  %val = add i32 0, 1
  br i1 undef, label %continue, label %leave
continue:
  br label %loop
leave:
  %local = phi i32 [ 0, %guarded1 ], [ %val, %check ]
  ret void
}

;
; Yet another test from PR38778
;
; CHECK-LABEL: @Test3
define void @Test3(i32) {
header:
  br label %outer
outer:
  %bad_input.i = icmp eq i32 %0, -2147483648
  br label %inner
inner:
  br i1 %bad_input.i, label %overflow, label %switchme
overflow:
  br label %continue
switchme:
  switch i32 %0, label %continue [
    i32 88, label %go_out
    i32 99, label %case2
  ]
; CHECK-NOT: {{^}}case2:
case2:
  br label %continue
continue:
  %local_11_92 = phi i32 [ 0, %switchme ], [ 18, %case2 ], [ 0, %overflow ]
  br i1 undef, label %outer, label %inner
go_out:
  unreachable
}
