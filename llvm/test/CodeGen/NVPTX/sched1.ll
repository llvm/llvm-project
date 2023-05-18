; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; Ensure source scheduling is working

define void @foo(ptr %a) {
; CHECK: .func foo
; CHECK: ld.u32
; CHECK-NEXT: ld.u32
; CHECK-NEXT: ld.u32
; CHECK-NEXT: ld.u32
; CHECK-NEXT: add.s32
; CHECK-NEXT: add.s32
; CHECK-NEXT: add.s32
  %val0 = load i32, ptr %a
  %ptr1 = getelementptr i32, ptr %a, i32 1
  %val1 = load i32, ptr %ptr1
  %ptr2 = getelementptr i32, ptr %a, i32 2
  %val2 = load i32, ptr %ptr2
  %ptr3 = getelementptr i32, ptr %a, i32 3
  %val3 = load i32, ptr %ptr3

  %t0 = add i32 %val0, %val1
  %t1 = add i32 %t0, %val2
  %t2 = add i32 %t1, %val3

  store i32 %t2, ptr %a

  ret void
}

