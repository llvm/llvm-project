; RUN: opt -passes=inline -mtriple=aarch64--linux-gnu -S -o - < %s -inline-threshold=0 | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

declare void @pad()
@glbl = external global i32

define i1 @outer1() {
; CHECK-LABEL: @outer1(
; CHECK-NOT: call i1 @inner1
  %C = call i1 @inner1()
  ret i1 %C
}

define i1 @inner1() {
entry:
  br label %if_true

if_true:
  %phi = phi i1 [0, %entry], [%phi, %if_true] ; Simplified to 0
  br i1 %phi, label %if_true, label %exit

exit:
  store i32 0, ptr @glbl
  store i32 1, ptr @glbl
  store i32 2, ptr @glbl
  store i32 3, ptr @glbl
  store i32 4, ptr @glbl
  ret i1 %phi
}


define i1 @outer2(i1 %val) {
; CHECK-LABEL: @outer2(
; CHECK: call i1 @inner2
  %C = call i1 @inner2(i1 %val)
  ret i1 %C
}

define i1 @inner2(i1 %val) {
entry:
  br label %if_true

if_true:
  %phi = phi i1 [%val, %entry], [%phi, %if_true] ; Cannot be simplified to a constant
  br i1 %phi, label %if_true, label %exit

exit:
  call void @pad()
  ret i1 %phi
}


define i1 @outer3(i1 %cond) {
; CHECK-LABEL: @outer3(
; CHECK-NOT: call i1 @inner3
  %C = call i1 @inner3(i1 %cond)
  ret i1 %C
}

define i1 @inner3(i1 %cond) {
entry:
  br i1 %cond, label %if_true, label %exit

if_true:
  br label %exit

exit:
  %phi = phi i32 [0, %entry], [0, %if_true] ; Simplified to 0
  %cmp = icmp eq i32 %phi, 0
  store i32 0, ptr @glbl
  store i32 1, ptr @glbl
  store i32 2, ptr @glbl
  store i32 3, ptr @glbl
  store i32 4, ptr @glbl
  ret i1 %cmp
}


define i1 @outer4(i1 %cond) {
; CHECK-LABEL: @outer4(
; CHECK-NOT: call i1 @inner4
  %C = call i1 @inner4(i1 %cond, i32 0)
  ret i1 %C
}

define i1 @inner4(i1 %cond, i32 %val) {
entry:
  br i1 %cond, label %if_true, label %exit

if_true:
  br label %exit

exit:
  %phi = phi i32 [0, %entry], [%val, %if_true] ; Simplified to 0
  %cmp = icmp eq i32 %phi, 0
  call void @pad()
  ret i1 %cmp
}


define i1 @outer5_1(i1 %cond) {
; CHECK-LABEL: @outer5_1(
; CHECK-NOT: call i1 @inner5
  %C = call i1 @inner5(i1 %cond, i32 0, i32 0)
  ret i1 %C
}


define i1 @outer5_2(i1 %cond) {
; CHECK-LABEL: @outer5_2(
; CHECK: call i1 @inner5
  %C = call i1 @inner5(i1 %cond, i32 0, i32 1)
  ret i1 %C
}

define i1 @inner5(i1 %cond, i32 %val1, i32 %val2) {
entry:
  br i1 %cond, label %if_true, label %exit

if_true:
  br label %exit

exit:
  %phi = phi i32 [%val1, %entry], [%val2, %if_true] ; Can be simplified to a constant if %val1 and %val2 are the same constants
  %cmp = icmp eq i32 %phi, 0
  call void @pad()
  store i32 0, ptr @glbl
  ret i1 %cmp
}


define i1 @outer6(i1 %cond, i32 %val) {
; CHECK-LABEL: @outer6(
; CHECK-NOT: call i1 @inner6
  %C = call i1 @inner6(i1 true, i32 %val, i32 0)
  ret i1 %C
}

define i1 @inner6(i1 %cond, i32 %val1, i32 %val2) {
entry:
  br i1 %cond, label %if_true, label %exit

if_true:
  br label %exit

exit:
  %phi = phi i32 [%val1, %entry], [%val2, %if_true] ; Simplified to 0
  %cmp = icmp eq i32 %phi, 0
  call void @pad()
  store i32 0, ptr @glbl
  store i32 1, ptr @glbl
  ret i1 %cmp
}


define i1 @outer7(i1 %cond, i32 %val) {
; CHECK-LABEL: @outer7(
; CHECK-NOT: call i1 @inner7
  %C = call i1 @inner7(i1 false, i32 0, i32 %val)
  ret i1 %C
}

define i1 @inner7(i1 %cond, i32 %val1, i32 %val2) {
entry:
  br i1 %cond, label %if_true, label %exit

if_true:
  br label %exit

exit:
  %phi = phi i32 [%val1, %entry], [%val2, %if_true] ; Simplified to 0
  %cmp = icmp eq i32 %phi, 0
  call void @pad()
  store i32 0, ptr @glbl
  store i32 1, ptr @glbl
  ret i1 %cmp
}


define i1 @outer8_1() {
; CHECK-LABEL: @outer8_1(
; CHECK-NOT: call i1 @inner8
  %C = call i1 @inner8(i32 0)
  ret i1 %C
}



define i1 @outer8_2() {
; CHECK-LABEL: @outer8_2(
; CHECK-NOT: call i1 @inner8
  %C = call i1 @inner8(i32 3)
  ret i1 %C
}

define i1 @inner8(i32 %cond) {
entry:
  switch i32 %cond, label %default [ i32 0, label %zero
                                     i32 1, label %one
                                     i32 2, label %two ]

zero:
  br label %exit

one:
  br label %exit

two:
  br label %exit

default:
  br label %exit

exit:
  %phi = phi i32 [0, %zero], [1, %one], [2, %two], [-1, %default] ; Can be simplified to a constant if the switch condition is known
  %cmp = icmp eq i32 %phi, 0
  call void @pad()
  ret i1 %cmp
}


define i1 @outer9(i1 %cond) {
; CHECK-LABEL: @outer9(
; CHECK-NOT: call i1 @inner9
  %C = call i1 @inner9(i32 0, i1 %cond)
  ret i1 %C
}

define i1 @inner9(i32 %cond1, i1 %cond2) {
entry:
  switch i32 %cond1, label %exit [ i32 0, label %zero
                                   i32 1, label %one
                                   i32 2, label %two ]

zero:
  br label %exit

one:
  br label %exit

two:
  br i1 %cond2, label %two_true, label %two_false

two_true:
  br label %exit

two_false:
  br label %exit

exit:
  %phi = phi i32 [0, %zero], [1, %one], [2, %two_true], [2, %two_false], [-1, %entry] ; Simplified to 0
  %cmp = icmp eq i32 %phi, 0
  call void @pad()
  store i32 0, ptr @glbl
  ret i1 %cmp
}


define i32 @outer10(i1 %cond) {
; CHECK-LABEL: @outer10(
; CHECK-NOT: call i32 @inner10
  %A = alloca i32
  %C = call i32 @inner10(i1 %cond, ptr %A)
  ret i32 %C
}

define i32 @inner10(i1 %cond, ptr %A) {
entry:
  br label %if_true

if_true:
  %phi = phi ptr [%A, %entry], [%phi, %if_true] ; Simplified to %A
  %load = load i32, ptr %phi
  br i1 %cond, label %if_true, label %exit

exit:
  call void @pad()
  ret i32 %load
}


define i32 @outer11(i1 %cond, ptr %ptr) {
; CHECK-LABEL: @outer11(
; CHECK: call i32 @inner11
  %C = call i32 @inner11(i1 %cond, ptr %ptr)
  ret i32 %C
}

define i32 @inner11(i1 %cond, ptr %ptr) {
entry:
  br label %if_true

if_true:
  %phi = phi ptr [%ptr, %entry], [%phi, %if_true] ; Cannot be simplified
  %load = load i32, ptr %phi
  br i1 %cond, label %if_true, label %exit

exit:
  call void @pad()
  ret i32 %load
}


define i32 @outer12(i1 %cond) {
; CHECK-LABEL: @outer12(
; CHECK-NOT: call i32 @inner12
  %A = alloca i32
  %C = call i32 @inner12(i1 %cond, ptr %A)
  ret i32 %C
}

define i32 @inner12(i1 %cond, ptr %ptr) {
entry:
  br i1 %cond, label %if_true, label %exit

if_true:
  br label %exit

exit:
  %phi = phi ptr [%ptr, %entry], [%ptr, %if_true] ; Simplified to %A
  %load = load i32, ptr %phi
  call void @pad()
  ret i32 %load
}


define i32 @outer13(i1 %cond) {
; CHECK-LABEL: @outer13(
; CHECK-NOT: call i32 @inner13
  %A = alloca i32
  %C = call i32 @inner13(i1 %cond, ptr %A)
  ret i32 %C
}

define i32 @inner13(i1 %cond, ptr %ptr) {
entry:
  %gep1 = getelementptr inbounds i32, ptr %ptr, i32 2
  %gep2 = getelementptr inbounds i32, ptr %ptr, i32 1
  br i1 %cond, label %if_true, label %exit

if_true:
  %gep3 = getelementptr inbounds i32, ptr %gep2, i32 1
  br label %exit

exit:
  %phi = phi ptr [%gep1, %entry], [%gep3, %if_true] ; Simplifeid to %gep1
  %load = load i32, ptr %phi
  call void @pad()
  ret i32 %load
}


define i32 @outer14(i1 %cond) {
; CHECK-LABEL: @outer14(
; CHECK: call i32 @inner14
  %A1 = alloca i32
  %A2 = alloca i32
  %C = call i32 @inner14(i1 %cond, ptr %A1, ptr %A2)
  ret i32 %C
}

define i32 @inner14(i1 %cond, ptr %ptr1, ptr %ptr2) {
entry:
  br i1 %cond, label %if_true, label %exit

if_true:
  br label %exit

exit:
  %phi = phi ptr [%ptr1, %entry], [%ptr2, %if_true] ; Cannot be simplified
  %load = load i32, ptr %phi
  call void @pad()
  store i32 0, ptr @glbl
  ret i32 %load
}


define i32 @outer15(i1 %cond, ptr %ptr) {
; CHECK-LABEL: @outer15(
; CHECK-NOT: call i32 @inner15
  %A = alloca i32
  %C = call i32 @inner15(i1 true, ptr %ptr, ptr %A)
  ret i32 %C
}

define i32 @inner15(i1 %cond, ptr %ptr1, ptr %ptr2) {
entry:
  br i1 %cond, label %if_true, label %exit

if_true:
  br label %exit

exit:
  %phi = phi ptr [%ptr1, %entry], [%ptr2, %if_true] ; Simplified to %A
  %load = load i32, ptr %phi
  call void @pad()
  store i32 0, ptr @glbl
  store i32 1, ptr @glbl
  ret i32 %load
}


define i32 @outer16(i1 %cond, ptr %ptr) {
; CHECK-LABEL: @outer16(
; CHECK-NOT: call i32 @inner16
  %A = alloca i32
  %C = call i32 @inner16(i1 false, ptr %A, ptr %ptr)
  ret i32 %C
}

define i32 @inner16(i1 %cond, ptr %ptr1, ptr %ptr2) {
entry:
  br i1 %cond, label %if_true, label %exit

if_true:
  br label %exit

exit:
  %phi = phi ptr [%ptr1, %entry], [%ptr2, %if_true] ; Simplified to %A
  %load = load i32, ptr %phi
  call void @pad()
  store i32 0, ptr @glbl
  store i32 1, ptr @glbl
  ret i32 %load
}


define i1 @outer17(i1 %cond) {
; CHECK-LABEL: @outer17(
; CHECK: call i1 @inner17
  %A = alloca i32
  %C = call i1 @inner17(i1 %cond, ptr %A)
  ret i1 %C
}

define i1 @inner17(i1 %cond, ptr %ptr) {
entry:
  br i1 %cond, label %if_true, label %exit

if_true:
  br label %exit

exit:
  %phi = phi ptr [null, %entry], [%ptr, %if_true] ; Cannot be mapped to a constant
  %cmp = icmp eq ptr %phi, null
  call void @pad()
  ret i1 %cmp
}


define i1 @outer18(i1 %cond) {
; CHECK-LABEL: @outer18(
; CHECK-NOT: call i1 @inner18
  %C = call i1 @inner18(i1 %cond, i1 true)
  ret i1 %C
}

define i1 @inner18(i1 %cond1, i1 %cond2) {
entry:
  br i1 %cond1, label %block1, label %block2

block1:
  br i1 %cond2, label %block3, label %block4

block2:
  br i1 %cond2, label %block5, label %block4

block3:
  %phi = phi i32 [0, %block1], [1, %block4], [0, %block5] ; Simplified to 0
  %cmp = icmp eq i32 %phi, 0
  call void @pad()
  ret i1 %cmp

block4:                                                   ; Unreachable block
  br label %block3

block5:
  br label %block3
}


define i1 @outer19(i1 %cond) {
; CHECK-LABEL: @outer19(
; CHECK: call i1 @inner19
  %A = alloca i32
  %C = call i1 @inner19(i1 %cond, ptr %A)
  ret i1 %C
}

define i1 @inner19(i1 %cond, ptr %ptr) {
entry:
  br i1 %cond, label %if_true, label %exit

if_true:
  br label %exit

exit:
  %phi = phi ptr [%ptr, %entry], [null, %if_true] ; Cannot be mapped to a constant
  %cmp = icmp eq ptr %phi, null
  call void @pad()
  ret i1 %cmp
}
