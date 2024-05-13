; RUN: opt -passes=inline -mtriple=aarch64--linux-gnu -S -o - < %s -inline-threshold=0 | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

declare void @pad()
@glbl = external global i32

define i32 @outer_add1(i32 %a) {
; CHECK-LABEL: @outer_add1(
; CHECK-NOT: call i32 @add
  %C = call i32 @add(i32 %a, i32 0)
  ret i32 %C
}

define i32 @outer_add2(i32 %a) {
; CHECK-LABEL: @outer_add2(
; CHECK-NOT: call i32 @add
  %C = call i32 @add(i32 0, i32 %a)
  ret i32 %C
}

define i32 @add(i32 %a, i32 %b) {
  %add = add i32 %a, %b
  call void @pad()
  store i32 0, ptr @glbl
  ret i32 %add
}



define i32 @outer_sub1(i32 %a) {
; CHECK-LABEL: @outer_sub1(
; CHECK-NOT: call i32 @sub1
  %C = call i32 @sub1(i32 %a, i32 0)
  ret i32 %C
}

define i32 @sub1(i32 %a, i32 %b) {
  %sub = sub i32 %a, %b
  call void @pad()
  store i32 0, ptr @glbl
  ret i32 %sub
}


define i32 @outer_sub2(i32 %a) {
; CHECK-LABEL: @outer_sub2(
; CHECK-NOT: call i32 @sub2
  %C = call i32 @sub2(i32 %a)
  ret i32 %C
}

define i32 @sub2(i32 %a) {
  %sub = sub i32 %a, %a
  call void @pad()
  ret i32 %sub
}



define i32 @outer_mul1(i32 %a) {
; CHECK-LABEL: @outer_mul1(
; CHECK-NOT: call i32 @mul
  %C = call i32 @mul(i32 %a, i32 0)
  ret i32 %C
}

define i32 @outer_mul2(i32 %a) {
; CHECK-LABEL: @outer_mul2(
; CHECK-NOT: call i32 @mul
  %C = call i32 @mul(i32 %a, i32 1)
  ret i32 %C
}

define i32 @mul(i32 %a, i32 %b) {
  %mul = mul i32 %a, %b
  call void @pad()
  store i32 0, ptr @glbl
  ret i32 %mul
}



define i32 @outer_div1(i32 %a) {
; CHECK-LABEL: @outer_div1(
; CHECK-NOT: call i32 @div1
  %C = call i32 @div1(i32 0, i32 %a)
  ret i32 %C
}

define i32 @outer_div2(i32 %a) {
; CHECK-LABEL: @outer_div2(
; CHECK-NOT: call i32 @div1
  %C = call i32 @div1(i32 %a, i32 1)
  ret i32 %C
}

define i32 @div1(i32 %a, i32 %b) {
  %div = sdiv i32 %a, %b
  call void @pad()
  store i32 0, ptr @glbl
  ret i32 %div
}


define i32 @outer_div3(i32 %a) {
; CHECK-LABEL: @outer_div3(
; CHECK-NOT: call i32 @div
  %C = call i32 @div2(i32 %a)
  ret i32 %C
}

define i32 @div2(i32 %a) {
  %div = sdiv i32 %a, %a
  call void @pad()
  ret i32 %div
}



define i32 @outer_rem1(i32 %a) {
; CHECK-LABEL: @outer_rem1(
; CHECK-NOT: call i32 @rem
  %C = call i32 @rem1(i32 0, i32 %a)
  ret i32 %C
}

define i32 @outer_rem2(i32 %a) {
; CHECK-LABEL: @outer_rem2(
; CHECK-NOT: call i32 @rem
  %C = call i32 @rem1(i32 %a, i32 1)
  ret i32 %C
}

define i32 @rem1(i32 %a, i32 %b) {
  %rem = urem i32 %a, %b
  call void @pad()
  store i32 0, ptr @glbl
  ret i32 %rem
}


define i32 @outer_rem3(i32 %a) {
; CHECK-LABEL: @outer_rem3(
; CHECK-NOT: call i32 @rem
  %C = call i32 @rem2(i32 %a)
  ret i32 %C
}

define i32 @rem2(i32 %a) {
  %rem = urem i32 %a, %a
  call void @pad()
  ret i32 %rem
}



define i32 @outer_shl1(i32 %a) {
; CHECK-LABEL: @outer_shl1(
; CHECK-NOT: call i32 @shl
  %C = call i32 @shl(i32 %a, i32 0)
  ret i32 %C
}

define i32 @shl(i32 %a, i32 %b) {
  %shl = shl i32 %a, %b
  call void @pad()
  store i32 0, ptr @glbl
  ret i32 %shl
}



define i32 @outer_shr1(i32 %a) {
; CHECK-LABEL: @outer_shr1(
; CHECK-NOT: call i32 @shr
  %C = call i32 @shr(i32 %a, i32 0)
  ret i32 %C
}

define i32 @shr(i32 %a, i32 %b) {
  %shr = ashr i32 %a, %b
  call void @pad()
  store i32 0, ptr @glbl
  ret i32 %shr
}



define i1 @outer_and1(i1 %a) {
; check-label: @outer_and1(
; check-not: call i1 @and1
  %c = call i1 @and1(i1 %a, i1 false)
  ret i1 %c
}

define i1 @outer_and2(i1 %a) {
; check-label: @outer_and2(
; check-not: call i1 @and1
  %c = call i1 @and1(i1 %a, i1 true)
  ret i1 %c
}

define i1 @and1(i1 %a, i1 %b) {
  %and = and i1 %a, %b
  call void @pad()
  store i32 0, ptr @glbl
  ret i1 %and
}


define i1 @outer_and3(i1 %a) {
; check-label: @outer_and3(
; check-not: call i1 @and2
  %c = call i1 @and2(i1 %a)
  ret i1 %c
}

define i1 @and2(i1 %a) {
  %and = and i1 %a, %a
  call void @pad()
  ret i1 %and
}



define i1 @outer_or1(i1 %a) {
; check-label: @outer_or1(
; check-not: call i1 @or1
  %c = call i1 @or1(i1 %a, i1 false)
  ret i1 %c
}

define i1 @outer_or2(i1 %a) {
; check-label: @outer_or2(
; check-not: call i1 @or1
  %c = call i1 @or1(i1 %a, i1 true)
  ret i1 %c
}

define i1 @or1(i1 %a, i1 %b) {
  %or = or i1 %a, %b
  call void @pad()
  store i32 0, ptr @glbl
  ret i1 %or
}


define i1 @outer_or3(i1 %a) {
; check-label: @outer_or3(
; check-not: call i1 @or2
  %c = call i1 @or2(i1 %a)
  ret i1 %c
}

define i1 @or2(i1 %a) {
  %or = or i1 %a, %a
  call void @pad()
  ret i1 %or
}



define i1 @outer_xor1(i1 %a) {
; check-label: @outer_xor1(
; check-not: call i1 @xor
  %c = call i1 @xor1(i1 %a, i1 false)
  ret i1 %c
}

define i1 @xor1(i1 %a, i1 %b) {
  %xor = xor i1 %a, %b
  call void @pad()
  store i32 0, ptr @glbl
  ret i1 %xor
}


define i1 @outer_xor3(i1 %a) {
; check-label: @outer_xor3(
; check-not: call i1 @xor
  %c = call i1 @xor2(i1 %a)
  ret i1 %c
}

define i1 @xor2(i1 %a) {
  %xor = xor i1 %a, %a
  call void @pad()
  ret i1 %xor
}
