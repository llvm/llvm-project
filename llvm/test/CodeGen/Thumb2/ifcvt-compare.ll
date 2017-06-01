; RUN: llc -mtriple=thumbv7-unknown-linux %s -o - | FileCheck %s

declare void @x()

define void @f0(i32 %x) optsize {
  ; CHECK-LABEL: f0:
  ; CHECK: cbnz
  %p = icmp eq i32 %x, 0
  br i1 %p, label %t, label %f

t:
  call void @x()
  br label %f

f:
  ret void
}

define void @f1(i32 %x) optsize {
  ; CHECK-LABEL: f1:
  ; CHECK: cmp r0, #1
  ; CHECK: it ne
  ; CHECK-NEXT: bxne lr
  %p = icmp eq i32 %x, 1
  br i1 %p, label %t, label %f

t:
  call void @x()
  br label %f

f:
  ret void
}

define void @f2(i32 %x) {
  ; CHECK-LABEL: f2:
  ; CHECK: cmp r0, #0
  ; CHECK: it ne
  ; CHECK-NEXT: bxne lr
  %p = icmp eq i32 %x, 0
  br i1 %p, label %t, label %f

t:
  call void @x()
  br label %f

f:
  ret void
}
