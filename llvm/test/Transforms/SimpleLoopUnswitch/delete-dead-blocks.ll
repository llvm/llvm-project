; RUN: opt < %s -simple-loop-unswitch -enable-nontrivial-unswitch -S 2>&1 | FileCheck %s
; RUN: opt < %s -passes=unswitch -enable-nontrivial-unswitch -S 2>&1 | FileCheck %s
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
