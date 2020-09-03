; RUN: opt < %s -gvn -enable-pre -S | FileCheck %s
; RUN: opt < %s -passes="gvn<pre>" -enable-pre=false -S | FileCheck %s

declare void @may_exit() nounwind

declare void @may_exit_1(i32) nounwind

define i32 @main(i32 %p, i32 %q) {

; CHECK-LABEL: @main(

block1:
    %cmp = icmp eq i32 %p, %q 
	br i1 %cmp, label %block2, label %block3

block2:
 %a = add i32 %p, 1
 br label %block4

block3:
  br label %block4
; CHECK: %.pre = add i32 %p, 1
; CHECK-NEXT: br label %block4

block4:
  %b = add i32 %p, 1
  ret i32 %b
; CHECK: %b.pre-phi = phi i32 [ %.pre, %block3 ], [ %a, %block2 ]
; CHECK-NEXT: ret i32 %b.pre-phi
}

; Don't PRE across implicit control flow.
define i32 @test2(i32 %p, i32 %q) {

; CHECK-LABEL: @test2
; CHECK: block1:

block1:
  %cmp = icmp eq i32 %p, %q
  br i1 %cmp, label %block2, label %block3

block2:
 %a = sdiv i32 %p, %q
 br label %block4

block3:
  br label %block4

; CHECK: block4:
; CHECK-NEXT: call void @may_exit(
; CHECK-NEXT: %b = sdiv
; CHECK-NEXT: ret i32 %b

block4:
  call void @may_exit() nounwind
  %b = sdiv i32 %p, %q
  ret i32 %b
}

; Don't PRE across implicit control flow.
define i32 @test3(i32 %p, i32 %q, i1 %r) {

; CHECK-LABEL: @test3
; CHECK: block1:

block1:
  br i1 %r, label %block2, label %block3

block2:
 %a = sdiv i32 %p, %q
 br label %block4

block3:
  br label %block4

block4:

; CHECK: block4:
; CHECK-NEXT: phi i32
; CHECK-NEXT: call void @may_exit_1(
; CHECK-NEXT: %b = sdiv
; CHECK-NEXT: ret i32 %b

  %phi = phi i32 [ 0, %block3 ], [ %a, %block2 ]
  call void @may_exit_1(i32 %phi) nounwind
  %b = sdiv i32 %p, %q
  ret i32 %b

}

; It's OK to PRE an instruction that is guaranteed to be safe to execute
; speculatively.
; TODO: Does it make any sense in this case?
define i32 @test4(i32 %p, i32 %q) {

; CHECK-LABEL: @test4
; CHECK: block1:

block1:
  %cmp = icmp eq i32 %p, %q
  br i1 %cmp, label %block2, label %block3

block2:
 %a = sdiv i32 %p, 6
 br label %block4

block3:
  br label %block4

; CHECK: block4:
; CHECK-NEXT: %b.pre-phi = phi i32
; CHECK-NEXT: call void @may_exit(
; CHECK-NEXT: ret i32 %b

block4:
  call void @may_exit() nounwind
  %b = sdiv i32 %p, 6
  ret i32 %b
}

; It is OK to PRE across implicit control flow if we don't insert new
; instructions.
define i32 @test5(i1 %cond, i32 %p, i32 %q) {

; CHECK-LABEL: @test5
; CHECK: block1:

block1:
  br i1 %cond, label %block2, label %block3

block2:
 %a = sdiv i32 %p, %q
 br label %block4

block3:
  %b = sdiv i32 %p, %q
  br label %block4

; CHECK: block4:
; CHECK-NEXT: %c.pre-phi = phi i32 [ %b, %block3 ], [ %a, %block2 ]
; CHECK-NEXT: call void @may_exit()
; CHECK-NEXT: ret i32 %c.pre-phi

block4:
  call void @may_exit() nounwind
  %c = sdiv i32 %p, %q
  ret i32 %c
}
